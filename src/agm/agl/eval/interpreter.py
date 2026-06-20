"""AgL tree-walking interpreter (Component 6) — v2 rewrite.

Evaluates a ``CheckedProgram`` (v2 AST) using a ``Scope`` chain.

Control flow for AgL exceptions uses Python exceptions (``AglRaise``).
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, assert_never, cast

from agm.agl._text import normalize_newlines
from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.scope import Scope
from agm.agl.eval.values import (
    UNIT_VALUE,
    AgentValue,
    BoolValue,
    Closure,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.runtime.convert import (
    CastConversionError,
    StrictJsonParseError,
    convert_value,
    parse_json_strict,
)
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    AssignTarget,
    BinaryOp,
    BinOp,
    Block,
    BoolLit,
    Call,
    Case,
    Cast,
    CatchClause,
    ConfigPragma,
    Constructor,
    DecimalLit,
    DictLit,
    Do,
    ElseSentinel,
    EnumDef,
    Expr,
    FieldAccess,
    FuncDef,
    If,
    ImportDecl,
    IndexAccess,
    IndexTarget,
    InterpSegment,
    IntLit,
    IsTest,
    Lambda,
    LetDecl,
    ListLit,
    NameTarget,
    NullLit,
    ParamDecl,
    Pattern,
    ProgramDecl,
    Raise,
    RecordDef,
    StringLit,
    Template,
    TextSegment,
    Try,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.typecheck.types import (
    CastKind,
    DecimalType,
    DictType,
    EnumType,
    FunctionType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    UnitType,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import ParseResult
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.runtime.request import ValidationError
    from agm.agl.runtime.trace import TraceStore
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.typecheck.env import CheckedProgram, FunctionSignature, TypeEnvironment
    from agm.agl.typecheck.graph import CheckedModule, CheckedModuleGraph
    from agm.core.process import ProcessCaptureResult


_Ordered = TypeVar("_Ordered", int, decimal.Decimal, str)

# Pinned decimal context for all AgL arithmetic (F7).  AgL semantics must not
# depend on the host's ambient ``decimal`` context — a host that lowered
# ``getcontext().prec`` would otherwise change results such as ``1 / 3``.
_AGL_DECIMAL_CONTEXT = decimal.Context(prec=28, rounding=decimal.ROUND_HALF_EVEN)


@dataclass(frozen=True)
class _IndexedAssignmentTarget:
    containers: tuple[tuple[Value, Value], ...]
    container: Value
    index: Value


def _make_exc_value(
    type_name: str, message: str, *, trace_id: str = "", **extra: Value
) -> ExceptionValue:
    """Create an ``ExceptionValue`` with ``message`` and optional extra fields."""
    fields: dict[str, Value] = {
        "message": TextValue(message),
        "trace_id": TextValue(trace_id),
    }
    fields.update(extra)
    return ExceptionValue(type_name=type_name, fields=fields)


def _make_match_error(subject: Value, *, trace_id: str = "") -> ExceptionValue:
    """Create a ``MatchError`` ``ExceptionValue`` for a non-matching *subject*."""
    scrutinee_type = _describe_value(subject)
    scrutinee_json = value_to_json_obj(subject)
    return _make_exc_value(
        "MatchError",
        f"Non-exhaustive case: no pattern matched value of type {scrutinee_type!r}",
        trace_id=trace_id,
        scrutinee_type=TextValue(scrutinee_type),
        scrutinee=JsonValue(scrutinee_json),
    )


class Interpreter:
    """Tree-walking interpreter for a checked AgL v2 program.

    ``checked``       — the type-checked program with side tables.
    ``registry``      — the host agent registry.
    ``contracts``     — materialized ``OutputContract`` per call node_id.
    ``type_env``      — the ``TypeEnvironment`` from the checked program.
    ``loop_limit``    — default bound for ``do`` loops without an explicit limit.
    ``strict_json``   — default strict-JSON flag for codec operations.
    ``source``        — normalized program source (for error context slicing).
    ``max_call_depth`` — recursion depth limit (raises RecursionError when exceeded).
    """

    def __init__(
        self,
        checked: "CheckedProgram",
        registry: "AgentRegistry",
        contracts: dict[int, "OutputContract"],
        type_env: "TypeEnvironment",
        *,
        loop_limit: int,
        strict_json: bool,
        source: str = "",
        shell_exec_timeout: float | None = None,
        trace: "TraceStore | None" = None,
        max_call_depth: int = 256,
        param_values: "Mapping[str, Value] | None" = None,
    ) -> None:
        from agm.agl.runtime.trace import noop_trace

        self._checked = checked
        self._registry = registry
        self._contracts = contracts
        self._type_env = type_env
        self._source = normalize_newlines(source)
        self._loop_limit = loop_limit
        self._strict_json = strict_json
        self._shell_exec_timeout = shell_exec_timeout
        self._trace: "TraceStore" = trace if trace is not None else noop_trace()
        self._max_call_depth = max_call_depth
        self._call_depth = 0
        self._param_values: "Mapping[str, Value]" = (
            param_values if param_values is not None else {}
        )
        self._module_frames: dict[ModuleId, Scope] = {}
        self._module_sigs: dict[ModuleId, dict[str, FunctionSignature]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, root_scope: Scope) -> None:
        """Execute the program body in *root_scope*.

        All arithmetic runs under a pinned 28-digit ``decimal`` context so AgL
        semantics never depend on the host's ambient context (F7).

        May raise ``AglRaise`` for uncaught AgL exceptions.
        """
        self._module_frames = {ENTRY_ID: root_scope}
        self._module_sigs = {ENTRY_ID: self._checked.function_signatures}
        with decimal.localcontext(_AGL_DECIMAL_CONTEXT):
            # Pre-pass: install FuncDef closures and AgentValue bindings in the
            # root scope so mutual recursion and forward references work.
            for item in self._checked.resolved.program.body.items:
                if isinstance(item, FuncDef):
                    sig = self._checked.function_signatures.get(item.name)
                    ret_type: Type = sig.result if sig is not None else UnitType()
                    params: tuple[tuple[str, Expr | None], ...] = tuple(
                        (p.name, p.default) for p in item.params
                    )
                    closure = Closure(
                        env=root_scope,
                        params=params,
                        body=item.body,
                        return_type=ret_type,
                    )
                    root_scope.define(
                        item.name, closure, mutable=False, decl_span=item.span
                    )
                elif isinstance(item, AgentDecl):
                    root_scope.define(
                        item.name,
                        AgentValue(name=item.name),
                        mutable=False,
                        decl_span=item.span,
                    )
                elif isinstance(item, ParamDecl):
                    self._eval_param_decl(item, root_scope)

            # Main pass: evaluate items in sequence.
            for item in self._checked.resolved.program.body.items:
                self._eval_item(item, root_scope)

    # ------------------------------------------------------------------
    # Item evaluation
    # ------------------------------------------------------------------

    def _eval_item(self, item: object, scope: Scope) -> Value:
        """Evaluate one block item; return its value (UNIT_VALUE for binders/decls)."""
        # --- Binders ---
        if isinstance(item, LetDecl):
            value = self._eval_expr(item.value, scope)
            target_type = self._binding_type_for(item.node_id)
            scope.define(
                item.name, _coerce(value, target_type), mutable=False, decl_span=item.span
            )
            return UNIT_VALUE
        if isinstance(item, VarDecl):
            value = self._eval_expr(item.value, scope)
            target_type = self._binding_type_for(item.node_id)
            scope.define(
                item.name, _coerce(value, target_type), mutable=True, decl_span=item.span
            )
            return UNIT_VALUE
        if isinstance(item, AssignStmt):
            ref = self._checked.resolved.resolution[item.node_id]
            target_type = self._binding_type_for(ref.decl_node_id)
            slot_type = _assign_target_slot_type(item.target, target_type)
            if isinstance(item.target, NameTarget):
                value = self._eval_expr(item.value, scope)
                coerced = _coerce(value, slot_type)
                updated = coerced
            else:
                root = scope.lookup(ref.name)
                if root is None:  # pragma: no cover
                    raise RuntimeError(f"Undefined variable at runtime: {ref.name!r}")
                prepared = self._eval_index_assignment_target(
                    item.target, root.value, scope, span=item.span
                )
                value = self._eval_expr(item.value, scope)
                coerced = _coerce(value, slot_type)
                updated = self._assign_prepared_index_target(
                    prepared,
                    coerced,
                    span=item.span,
                )
            scope.assign_value(ref.name, updated)
            self._trace.mutation(name=ref.name, value=updated, span=item.span)
            return UNIT_VALUE
        # --- Declarations (already handled in pre-pass or have no runtime action) ---
        if isinstance(item, (FuncDef, AgentDecl, ParamDecl)):
            # Closures and AgentValues installed in pre-pass.
            return UNIT_VALUE
        if isinstance(item, (RecordDef, EnumDef, TypeAlias, ProgramDecl, ConfigPragma, ImportDecl)):
            return UNIT_VALUE
        # --- Must be an Expr ---
        return self._eval_expr(cast("Expr", item), scope)

    def _eval_block(self, block: Block, scope: Scope) -> Value:
        """Evaluate a Block in a new child scope; return last item's value."""
        block_scope = Scope(parent=scope)
        return self._eval_block_items(block, block_scope)

    def _eval_block_items(self, block: Block, scope: Scope) -> Value:
        """Evaluate Block items directly in *scope*; return last item's value."""
        result: Value = UNIT_VALUE
        for item in block.items:
            result = self._eval_item(item, scope)
        return result

    def _eval_param_decl(self, item: ParamDecl, scope: Scope) -> None:
        """Bind a root ``param`` from an external value or its default expression."""
        if item.name in self._param_values:
            scope.define(
                item.name,
                self._param_values[item.name],
                mutable=False,
                decl_span=item.span,
            )
            return
        if item.default is None:
            raise AssertionError(
                f"Required param {item.name!r} has no value at execution time; "
                "pre-execution validation should have caught this."
            )
        value = self._eval_expr(item.default, scope)
        target_type = self._binding_type_for(item.node_id)
        scope.define(
            item.name,
            _coerce(value, target_type),
            mutable=False,
            decl_span=item.span,
        )

    # ------------------------------------------------------------------
    # Expression evaluation
    # ------------------------------------------------------------------

    def _eval_expr(self, expr: Expr, scope: Scope) -> Value:
        # --- Literals ---
        if isinstance(expr, UnitLit):
            return UNIT_VALUE
        if isinstance(expr, IntLit):
            return IntValue(expr.value)
        if isinstance(expr, DecimalLit):
            return DecimalValue(expr.value)
        if isinstance(expr, BoolLit):
            return BoolValue(expr.value)
        if isinstance(expr, NullLit):
            return JsonValue(None)
        if isinstance(expr, StringLit):
            return TextValue(expr.value)
        # --- Name / member access ---
        if isinstance(expr, VarRef):
            return self._eval_var_ref(expr, scope)
        if isinstance(expr, FieldAccess):
            return self._eval_field_access(expr, scope)
        if isinstance(expr, IndexAccess):
            return self._eval_index_access(expr, scope)
        # --- Compound expressions ---
        if isinstance(expr, Template):
            return TextValue(self._eval_template(expr, scope))
        if isinstance(expr, Call):
            return self._eval_call(expr, scope)
        if isinstance(expr, Lambda):
            return self._eval_lambda(expr, scope)
        if isinstance(expr, Block):
            return self._eval_block(expr, scope)
        if isinstance(expr, If):
            return self._eval_if(expr, scope)
        if isinstance(expr, Case):
            return self._eval_case(expr, scope)
        if isinstance(expr, Do):
            return self._eval_do(expr, scope)
        if isinstance(expr, Try):
            return self._eval_try(expr, scope)
        if isinstance(expr, Raise):
            return self._eval_raise(expr, scope)
        if isinstance(expr, Constructor):
            return self._eval_constructor(expr, scope)
        if isinstance(expr, BinaryOp):
            return self._eval_binary_op(expr, scope)
        if isinstance(expr, UnaryNot):
            return self._eval_unary_not(expr, scope)
        if isinstance(expr, UnaryNeg):
            return self._eval_unary_neg(expr, scope)
        if isinstance(expr, IsTest):
            return self._eval_is_test(expr, scope)
        if isinstance(expr, Cast):
            return self._eval_cast(expr, scope)
        if isinstance(expr, ListLit):
            return self._eval_list_lit(expr, scope)
        if isinstance(expr, DictLit):
            return self._eval_dict_lit(expr, scope)
        assert_never(expr)  # pragma: no cover

    def _eval_var_ref(self, expr: VarRef, scope: Scope) -> Value:
        from agm.agl.scope.symbols import BinderKind

        ref = self._checked.resolved.resolution.get(expr.node_id)
        if ref is not None and ref.kind in (BinderKind.function_binding, BinderKind.agent_binding):
            frame = self._module_frames.get(ref.module_id)
            if frame is not None:  # pragma: no branch
                binding = frame.bindings.get(ref.name)
                if binding is not None:
                    return binding.value
        # Fall back to lexical lookup for local/param/let/var/pattern/catch bindings
        binding = scope.lookup(expr.name)
        if binding is None:  # pragma: no cover
            raise RuntimeError(f"Undefined variable at runtime: {expr.name!r}")
        return binding.value

    def _eval_field_access(self, expr: FieldAccess, scope: Scope) -> Value:
        obj = self._eval_expr(expr.obj, scope)
        if isinstance(obj, (RecordValue, ExceptionValue)):
            return obj.fields[expr.field]
        raise RuntimeError(  # pragma: no cover
            f"Field access on non-record/exception: {type(obj).__name__}"
        )

    def _eval_index_access(self, expr: IndexAccess, scope: Scope) -> Value:
        obj = self._eval_expr(expr.obj, scope)
        index = self._eval_expr(expr.index, scope)
        return self._index_value(obj, index, span=expr.span)

    def _index_value(self, obj: Value, index: Value, *, span: SourceSpan) -> Value:
        if isinstance(obj, ListValue):
            if not isinstance(index, IntValue):
                raise RuntimeError(  # pragma: no cover
                    f"List index must be IntValue, got {type(index).__name__}"
                )
            return obj.elements[self._normalize_list_index(index.value, len(obj.elements), span)]
        if isinstance(obj, DictValue):
            if not isinstance(index, TextValue):
                raise RuntimeError(  # pragma: no cover
                    f"Dict index must be TextValue, got {type(index).__name__}"
                )
            if index.value not in obj.entries:
                raise self._key_error(index.value, span)
            return obj.entries[index.value]
        raise RuntimeError(  # pragma: no cover
            f"Index access on non-list/dict: {type(obj).__name__}"
        )

    def _normalize_list_index(self, index: int, length: int, span: SourceSpan) -> int:
        normalized = index if index >= 0 else length + index
        if normalized < 0 or normalized >= length:
            raise self._index_error(index, length, span)
        return normalized

    def _index_error(self, index: int, length: int, span: SourceSpan) -> AglRaise:
        return AglRaise(
            _make_exc_value(
                "IndexError",
                f"List index {index} out of range for length {length}",
                trace_id=self._trace.new_event_id(),
                index=IntValue(index),
                length=IntValue(length),
            ),
            span=span,
        )

    def _key_error(self, key: str, span: SourceSpan) -> AglRaise:
        return AglRaise(
            _make_exc_value(
                "KeyError",
                f"Dict key {key!r} is missing",
                trace_id=self._trace.new_event_id(),
                key=TextValue(key),
            ),
            span=span,
        )

    def _index_target_expr_path(self, target: IndexTarget) -> tuple[Expr, ...]:
        path: list[Expr] = []
        self._append_index_target_expr_path(target.obj, path)
        path.append(target.index)
        return tuple(path)

    def _append_index_target_expr_path(self, expr: Expr, path: list[Expr]) -> None:
        if isinstance(expr, VarRef):
            return
        if isinstance(expr, IndexAccess):
            self._append_index_target_expr_path(expr.obj, path)
            path.append(expr.index)
            return
        raise RuntimeError(  # pragma: no cover
            "Indexed assignment requires a variable list or dict root."
        )

    def _eval_index_assignment_target(
        self,
        target: IndexTarget,
        root: Value,
        scope: Scope,
        *,
        span: SourceSpan,
    ) -> _IndexedAssignmentTarget:
        path = self._index_target_expr_path(target)
        containers: list[tuple[Value, Value]] = []
        current = root
        for index_expr in path[:-1]:
            index = self._eval_expr(index_expr, scope)
            containers.append((current, index))
            current = self._index_value(current, index, span=span)
        if not path:  # pragma: no cover
            raise RuntimeError("Indexed assignment requires at least one index.")
        index = self._eval_expr(path[-1], scope)
        self._index_value(current, index, span=span)
        return _IndexedAssignmentTarget(tuple(containers), current, index)

    def _assign_prepared_index_target(
        self,
        target: _IndexedAssignmentTarget,
        value: Value,
        *,
        span: SourceSpan,
    ) -> Value:
        updated = self._replace_index_value(target.container, target.index, value, span=span)
        for container, index in reversed(target.containers):
            updated = self._replace_index_value(container, index, updated, span=span)
        return updated

    def _replace_index_value(
        self,
        obj: Value,
        index: Value,
        value: Value,
        *,
        span: SourceSpan,
    ) -> Value:
        if isinstance(obj, ListValue):
            if not isinstance(index, IntValue):
                raise RuntimeError(  # pragma: no cover
                    f"List index must be IntValue, got {type(index).__name__}"
                )
            normalized = self._normalize_list_index(index.value, len(obj.elements), span)
            elements = list(obj.elements)
            elements[normalized] = value
            return ListValue(tuple(elements))
        if isinstance(obj, DictValue):
            if not isinstance(index, TextValue):
                raise RuntimeError(  # pragma: no cover
                    f"Dict index must be TextValue, got {type(index).__name__}"
                )
            entries = dict(obj.entries)
            entries[index.value] = value
            return DictValue(entries)
        raise RuntimeError(  # pragma: no cover
            f"Indexed assignment on non-list/dict: {type(obj).__name__}"
        )

    def _eval_template(self, expr: Template, scope: Scope) -> str:
        """Evaluate *expr* to a string: text segments verbatim, interpolation rendered."""
        from agm.agl.runtime.render import render_value

        parts: list[str] = []
        for seg in expr.segments:
            if isinstance(seg, TextSegment):
                parts.append(seg.text)
            elif isinstance(seg, InterpSegment):
                value = self._eval_expr(seg.expr, scope)
                parts.append(render_value(value))
            else:
                assert_never(seg)  # pragma: no cover
        return "".join(parts)

    # ------------------------------------------------------------------
    # Call dispatch
    # ------------------------------------------------------------------

    def _eval_call(self, expr: Call, scope: Scope) -> Value:
        """Evaluate a Call node: builtin, user function, or closure."""
        from agm.agl.scope.symbols import BuiltinKind

        builtin_kind = self._checked.resolved.builtin_calls.get(expr.node_id)
        if builtin_kind is BuiltinKind.PRINT:
            return self._eval_print_call(expr, scope)
        if builtin_kind is BuiltinKind.ASK:
            return self._eval_ask_call(expr, scope)
        if builtin_kind is BuiltinKind.ASK_REQUEST:
            return self._eval_ask_request_call(expr, scope)
        if builtin_kind is BuiltinKind.EXEC:
            return self._eval_exec_call(expr, scope)
        if builtin_kind is BuiltinKind.PARSE_JSON:
            return self._eval_parse_json_call(expr, scope)

        # User-defined function or closure call.
        callee_val = self._eval_expr(expr.callee, scope)
        if not isinstance(callee_val, Closure):
            raise RuntimeError(  # pragma: no cover
                f"Call on non-closure: {type(callee_val).__name__}"
            )
        return self._apply_closure(callee_val, expr, scope)

    def _apply_closure(self, closure: Closure, call: Call, call_scope: Scope) -> Value:
        """Bind arguments and evaluate the closure body with depth limit enforcement."""
        if self._call_depth >= self._max_call_depth:
            raise AglRaise(
                _make_exc_value(
                    "RecursionError",
                    f"Maximum call depth ({self._max_call_depth}) exceeded",
                    trace_id=self._trace.new_event_id(),
                    limit=IntValue(self._max_call_depth),
                ),
                span=call.span,
            )

        # Determine if this is a declared-name call so we can use named/default args.
        from agm.agl.scope.symbols import BinderKind

        callee_sig: FunctionSignature | None = None
        if isinstance(call.callee, VarRef):
            ref = self._checked.resolved.resolution.get(call.callee.node_id)
            if ref is not None and ref.kind == BinderKind.function_binding:
                callee_sig = self._get_function_signature(ref.module_id, ref.name)

        # Build the function scope (closes over the closure's captured env).
        fn_scope = Scope(parent=closure.env)

        sig = callee_sig
        if sig is not None:
            self._bind_declared_args(fn_scope, closure, sig, call, call_scope)
        else:
            self._bind_positional_args(fn_scope, closure, call, call_scope)

        self._call_depth += 1
        try:
            result = self._eval_expr(closure.body, fn_scope)
        finally:
            self._call_depth -= 1
        return _coerce(result, closure.return_type)

    def _bind_declared_args(
        self,
        fn_scope: Scope,
        closure: Closure,
        sig: "FunctionSignature",
        call: Call,
        call_scope: Scope,
    ) -> None:
        """Bind arguments using the FunctionSignature (supports named args + defaults)."""
        # Build a lookup of named args provided at the call site.
        named: dict[str, Expr] = {na.name: na.value for na in call.named_args}
        pos_exprs = list(call.args)
        # Build a lookup of default expressions from the closure's param list.
        closure_defaults: dict[str, Expr | None] = dict(closure.params)
        pos_idx = 0
        for param_name, param_type, has_default in sig.params:
            span = call.span
            if param_name in named:
                val = self._eval_expr(named[param_name], call_scope)
            elif pos_idx < len(pos_exprs):
                val = self._eval_expr(pos_exprs[pos_idx], call_scope)
                pos_idx += 1
            elif has_default:
                default_expr = closure_defaults.get(param_name)
                if default_expr is None:
                    raise RuntimeError(  # pragma: no cover
                        f"Missing default expression for param {param_name!r}"
                    )
                # Defaults are evaluated in the closure's environment.
                val = self._eval_expr(default_expr, closure.env)
            else:
                raise RuntimeError(  # pragma: no cover
                    f"Missing required argument {param_name!r}"
                )
            fn_scope.define(
                param_name,
                _coerce(val, param_type),
                mutable=False,
                decl_span=span,
            )

    def _bind_positional_args(
        self,
        fn_scope: Scope,
        closure: Closure,
        call: Call,
        call_scope: Scope,
    ) -> None:
        """Bind positional arguments only (lambda calls and closures without a sig)."""
        for idx, (param_name, default_expr) in enumerate(closure.params):
            if idx < len(call.args):
                val = self._eval_expr(call.args[idx], call_scope)
            elif default_expr is not None:
                val = self._eval_expr(default_expr, closure.env)
            else:
                raise RuntimeError(  # pragma: no cover
                    f"Missing argument for parameter {param_name!r}"
                )
            fn_scope.define(param_name, val, mutable=False, decl_span=call.span)

    def _eval_lambda(self, expr: Lambda, scope: Scope) -> Closure:
        """Create a Closure from a Lambda expression, capturing the current scope."""
        # The checker already inferred the lambda's full FunctionType (including
        # the result type for annotation-free lambdas); reuse it instead of
        # re-resolving the return-type annotation on every closure creation.
        fn_type = self._checked.node_types.get(expr.node_id)
        assert isinstance(fn_type, FunctionType), "lambda missing inferred FunctionType"
        ret_type: Type = fn_type.result
        params: tuple[tuple[str, Expr | None], ...] = tuple(
            (p.name, p.default) for p in expr.params
        )
        return Closure(env=scope, params=params, body=expr.body, return_type=ret_type)

    # ------------------------------------------------------------------
    # Control-flow expressions
    # ------------------------------------------------------------------

    def _eval_if(self, expr: If, scope: Scope) -> Value:
        """Evaluate an ``if`` expression; yield UNIT_VALUE when no branch matches."""
        has_else = any(isinstance(branch.cond, ElseSentinel) for branch in expr.branches)
        for branch in expr.branches:
            take = isinstance(branch.cond, ElseSentinel) or self._require_bool(
                self._eval_expr(branch.cond, scope)
            )
            if take:
                branch_scope = Scope(parent=scope)
                result = self._eval_expr(branch.body, branch_scope)
                return result if has_else else UNIT_VALUE
        # No matching branch and no else → unit (design §4.3).
        return UNIT_VALUE

    def _eval_case(self, expr: Case, scope: Scope) -> Value:
        """Evaluate a ``case`` expression; raise MatchError on no match."""
        subject = self._eval_expr(expr.subject, scope)
        for branch in expr.branches:
            matched, bindings = _match_pattern(branch.pattern, subject)
            if matched:
                branch_scope = Scope(parent=scope)
                for name, val in bindings.items():
                    branch_scope.define(name, val, mutable=False, decl_span=branch.span)
                return self._eval_expr(branch.body, branch_scope)
        raise AglRaise(
            _make_match_error(subject, trace_id=self._trace.new_event_id()),
            span=expr.span,
        )

    def _eval_do(self, expr: Do, scope: Scope) -> Value:
        """Evaluate a ``do[N] body until cond`` expression; yields unit on success."""
        limit = expr.limit if expr.limit is not None else self._loop_limit
        last_cond = False
        for _iteration in range(limit):
            iter_scope = Scope(parent=scope)
            if isinstance(expr.body, Block):
                self._eval_block_items(expr.body, iter_scope)
            else:
                self._eval_expr(expr.body, iter_scope)
            cond = self._eval_expr(expr.condition, iter_scope)
            last_cond = self._require_bool(cond)
            if last_cond:
                return UNIT_VALUE
        raise AglRaise(
            _make_exc_value(
                "MaxIterationsExceeded",
                f"Loop exhausted after {limit} iterations",
                trace_id=self._trace.new_event_id(),
                limit=IntValue(limit),
                condition=TextValue(self._source_slice(expr.condition.span)),
                last_condition_value=BoolValue(last_cond),
                metadata=JsonValue(None),
            ),
            span=expr.span,
        )

    def _eval_try(self, expr: Try, scope: Scope) -> Value:
        """Evaluate a ``try`` expression; catch matching handlers."""
        try:
            return self._eval_expr(expr.body, scope)
        except AglRaise as exc:
            for handler in expr.handlers:
                if _matches_catch(handler, exc.exc):
                    catch_scope = Scope(parent=scope)
                    if handler.binding is not None:
                        catch_scope.define(
                            handler.binding,
                            exc.exc,
                            mutable=False,
                            decl_span=handler.span,
                        )
                    return self._eval_expr(handler.body, catch_scope)
            # No handler matched: re-propagate.
            raise

    def _eval_raise(self, expr: Raise, scope: Scope) -> Value:
        exc_val = self._eval_expr(expr.exc, scope)
        if isinstance(exc_val, ExceptionValue):
            raise AglRaise(exc_val, span=expr.span)
        # Unreachable: the checker requires an exception-typed operand.
        raise AssertionError(  # pragma: no cover
            f"raise: expected ExceptionValue, got {type(exc_val).__name__}"
        )

    # ------------------------------------------------------------------
    # Builtin call handlers
    # ------------------------------------------------------------------

    def _extract_parse_policy(self, named_map: dict[str, Expr], scope: Scope) -> Value | None:
        """Extract the ``on_parse_error:`` parse policy if present.

        The checker guarantees that ``on_parse_error`` is always a static
        ``ParsePolicy`` constructor (``Abort()`` or ``Retry(n: <int literal>)``).
        We extract it directly from the constructor AST rather than evaluating it
        through ``_eval_constructor``, because the checker special-cases these
        constructors and does not record them in ``node_types``.

        Returns an ``EnumValue`` for ``Retry(n: N)`` or ``None`` for absent /
        ``Abort()`` (both map to a single-attempt abort policy).
        """
        if "on_parse_error" not in named_map:
            return None
        policy_expr = named_map["on_parse_error"]
        if isinstance(policy_expr, Constructor) and policy_expr.name == "Retry":
            # The checker guarantees exactly one NamedArg(name="n", value=IntLit).
            n_val = next(
                (arg.value.value for arg in policy_expr.args
                 if arg.name == "n" and isinstance(arg.value, IntLit)),
                0,  # pragma: no cover  – checker enforces IntLit; default unreachable
            )
            return EnumValue(
                type_name="ParsePolicy",
                variant="Retry",
                fields={"n": IntValue(n_val)},
            )
        # Absent or Abort() → single-attempt abort.
        return None

    def _eval_to_text(self, expr: Expr, scope: Scope) -> str:
        """Evaluate *expr* to plain text — a Template renders directly, any other
        value is rendered (a bare ``text`` value yields its raw string)."""
        if isinstance(expr, Template):
            return self._eval_template(expr, scope)
        val = self._eval_expr(expr, scope)
        if isinstance(val, TextValue):
            return val.value
        from agm.agl.runtime.render import render_value

        return render_value(val)

    def _eval_print_call(self, expr: Call, scope: Scope) -> Value:
        """Evaluate ``print(expr)`` — output and return unit."""
        from agm.agl.runtime.render import render_value

        arg = self._eval_expr(expr.args[0], scope)
        rendered = render_value(arg)
        print(rendered)
        self._trace.print_stmt(rendered=rendered, span=expr.span)
        return UNIT_VALUE

    def _eval_ask_call(self, expr: Call, scope: Scope) -> Value:
        """Evaluate an ``ask(prompt, agent: agentval, ...)`` builtin call."""
        from agm.agl.runtime.request import AgentRequest

        # Resolve agent: check named arg ``agent:`` first.
        agent_name = "ask"
        named_map: dict[str, Expr] = {na.name: na.value for na in expr.named_args}
        if "agent" in named_map:
            agent_val = self._eval_expr(named_map["agent"], scope)
            if isinstance(agent_val, AgentValue):
                agent_name = agent_val.name

        # First positional arg is the prompt (Template or any expr).
        prompt_text = self._eval_to_text(expr.args[0], scope)

        call_span = expr.span
        contract = self._contracts.get(expr.node_id)
        result_type = self._checked.node_types.get(expr.node_id)
        if isinstance(result_type, UnitType):
            self._trace.agent_call_attempt(
                agent=agent_name,
                attempt=0,
                prompt=prompt_text,
                span=call_span,
            )
            request = AgentRequest(
                agent=agent_name,
                prompt=prompt_text,
                output_contract=None,
            )
            try:
                self._registry.dispatch(agent_name, request)
            except AglRaise as exc:
                if exc.span is None:
                    exc.span = call_span
                raise
            return UNIT_VALUE
        if contract is None:
            from agm.agl.runtime.codec import TextCodec
            from agm.agl.runtime.contract import OutputContract

            contract = OutputContract(
                target_type=TextType(),
                codec=TextCodec(),
                strict_json=None,
                format_instructions="",
                json_schema=None,
            )

        def acquire(
            attempt: int,
            last_raw: str | None,
            last_errors: tuple[ValidationError, ...],
        ) -> tuple[str, str]:
            attempt_trace_id = self._trace.agent_call_attempt(
                agent=agent_name,
                attempt=attempt,
                prompt=prompt_text,
                span=call_span,
            )
            request = AgentRequest(
                agent=agent_name,
                prompt=prompt_text,
                attempt=attempt,
                previous_invalid_output=last_raw,
                validation_errors=list(last_errors),
                output_contract=contract,
            )
            try:
                response = self._registry.dispatch(agent_name, request)
            except AglRaise as exc:
                if exc.span is None:
                    exc.span = call_span
                raise
            return response.content, attempt_trace_id

        def on_parsed(raw: str, result: "ParseResult") -> None:
            if result.errors:
                error_summary = "; ".join(e.message for e in result.errors)
            else:
                error_summary = result.error_msg
            self._trace.parse_result(
                ok=result.ok and result.value is not None,
                raw=raw,
                normalized_raw=result.normalized_raw or "",
                error_summary=error_summary,
                span=call_span,
            )

        def make_failure_message(last_raw: str | None, max_attempts: int) -> str:
            return (
                f"Agent {agent_name!r} failed to produce a valid "
                f"{contract.target_type!r} after {max_attempts} attempt(s). "
                f"Last output: {last_raw!r}"
            )

        parse_policy = self._extract_parse_policy(named_map, scope)
        return self._run_parse_attempts(
            acquire=acquire,
            contract=contract,
            parse_policy=parse_policy,
            agent_label=agent_name,
            make_failure_message=make_failure_message,
            on_parsed=on_parsed,
            raise_span=call_span,
        )

    def _eval_ask_request_call(self, expr: Call, scope: Scope) -> Value:
        """Evaluate ``ask-request(prompt, ...)`` — build the ``AgentRequest`` the
        matching ``ask`` call would dispatch, without invoking the agent.

        Side-effect-free: no registry dispatch, no trace events, no retries.
        The agent name, rendered prompt, and materialized output contract are
        assembled into an ``AgentRequest`` record value (first attempt: attempt
        = 0, no retry context).
        """
        # Resolve agent name exactly as ``ask`` does (default "ask").
        agent_name = "ask"
        named_map: dict[str, Expr] = {na.name: na.value for na in expr.named_args}
        if "agent" in named_map:
            agent_val = self._eval_expr(named_map["agent"], scope)
            if isinstance(agent_val, AgentValue):
                agent_name = agent_val.name

        prompt_text = self._eval_to_text(expr.args[0], scope)

        contract = self._contracts.get(expr.node_id)
        if expr.type_arg is not None:
            from agm.agl.syntax.types import UnitT

            if isinstance(expr.type_arg, UnitT):
                return RecordValue(
                    type_name="AgentRequest",
                    fields={
                        "agent": TextValue(agent_name),
                        "prompt": TextValue(prompt_text),
                        "attempt": IntValue(0),
                        "output_contract": EnumValue(
                            type_name="OutputContractOption",
                            variant="None",
                            fields={},
                        ),
                    },
                )
        if contract is None:
            from agm.agl.runtime.codec import TextCodec
            from agm.agl.runtime.contract import OutputContract

            contract = OutputContract(
                target_type=TextType(),
                codec=TextCodec(),
                strict_json=None,
                format_instructions="",
                json_schema=None,
            )

        output_contract_value = RecordValue(
            type_name="OutputContract",
            fields={
                "target_type": TextValue(repr(contract.target_type)),
                "codec_name": TextValue(contract.codec.name),
                "strict_json": JsonValue(contract.strict_json),
                "format_instructions": TextValue(contract.format_instructions),
                "json_schema": JsonValue(contract.json_schema),
                "structured_exec": BoolValue(contract.structured_exec),
            },
        )
        return RecordValue(
            type_name="AgentRequest",
            fields={
                "agent": TextValue(agent_name),
                "prompt": TextValue(prompt_text),
                "attempt": IntValue(0),
                "output_contract": EnumValue(
                    type_name="OutputContractOption",
                    variant="Some",
                    fields={"value": output_contract_value},
                ),
            },
        )

    def _eval_exec_call(self, expr: Call, scope: Scope) -> Value:
        """Evaluate an ``exec(command, ...)`` builtin call."""
        # First arg is the command (Template or any expr).
        command = self._eval_to_text(expr.args[0], scope)

        exec_span = expr.span
        contract = self._contracts.get(expr.node_id)
        named_exec_map: dict[str, Expr] = {na.name: na.value for na in expr.named_args}
        exec_parse_policy = self._extract_parse_policy(named_exec_map, scope)

        def execute_command() -> tuple[str, str]:
            result, trace_id = self._run_shell_capture(command, exec_span)
            if result.returncode is not None and result.returncode != 0:
                exit_code = result.returncode
                raise AglRaise(
                    _make_exc_value(
                        "ExecError",
                        f"Shell command exited with code {exit_code}: {command!r}",
                        trace_id=self._trace.new_event_id(),
                        command=TextValue(command),
                        exit_code=IntValue(exit_code),
                        stdout=TextValue(result.stdout.rstrip("\n")),
                        stderr=TextValue(result.stderr.rstrip("\n")),
                        timed_out=BoolValue(False),
                    ),
                    span=exec_span,
                )
            return result.stdout.rstrip("\n"), trace_id

        # Structured exec: return a raw ExecResult record without parsing. A
        # non-zero exit is data here (reported in the record), not an error.
        if contract is not None and contract.structured_exec:
            result, _ = self._run_shell_capture(command, exec_span)
            exit_code = result.returncode if result.returncode is not None else 0
            return RecordValue(
                type_name="ExecResult",
                fields={
                    "stdout": TextValue(result.stdout.rstrip("\n")),
                    "exit_code": IntValue(exit_code),
                    "stderr": TextValue(result.stderr.rstrip("\n")),
                    "timed_out": BoolValue(False),
                },
            )

        stdout, exec_trace_id = execute_command()

        if contract is None:
            return TextValue(stdout)

        if isinstance(contract.target_type, TextType):
            return TextValue(stdout)

        def acquire_exec(
            attempt: int,
            _last_raw: str | None,
            _last_errors: tuple[ValidationError, ...],
        ) -> tuple[str, str]:
            if attempt == 0:
                return stdout, exec_trace_id
            return execute_command()

        def make_exec_failure_message(last_raw: str | None, max_attempts: int) -> str:
            return (
                f"exec output failed to parse as {contract.target_type!r} "
                f"after {max_attempts} attempt(s). Last output: {last_raw!r}"
            )

        return self._run_parse_attempts(
            acquire=acquire_exec,
            contract=contract,
            parse_policy=exec_parse_policy,
            agent_label="exec",
            make_failure_message=make_exec_failure_message,
            raise_span=exec_span,
        )

    def _run_shell_capture(
        self, command: str, exec_span: "SourceSpan | None"
    ) -> "tuple[ProcessCaptureResult, str]":
        """Run *command* via the shell, emit the exec trace, and return ``(result, trace_id)``.

        Raises ``ExecError`` on a spawn failure or idle timeout — these are
        always errors. A non-zero exit code is *not* raised here: the caller
        decides whether a failing exit should raise (text/parse exec) or be
        returned as data (structured exec).
        """
        from agm.core.process import run_capture_result

        result = run_capture_result(
            ["sh", "-c", command],
            idle_timeout=self._shell_exec_timeout,
            isolate_process_group=True,
        )
        if result.spawn_error is not None:
            self._trace.exec_command(
                command=command,
                exit_code=-1,
                duration=result.elapsed,
                stdout="",
                stderr=result.spawn_error,
                timed_out=False,
                span=exec_span,
            )
            raise AglRaise(
                _make_exc_value(
                    "ExecError",
                    f"Failed to spawn shell: {result.spawn_error}",
                    trace_id=self._trace.new_event_id(),
                    command=TextValue(command),
                    exit_code=IntValue(-1),
                    stdout=TextValue(""),
                    stderr=TextValue(""),
                    timed_out=BoolValue(False),
                ),
                span=exec_span,
            )
        # returncode is None only on a timeout (handled below), where -1 stands in.
        exit_code = result.returncode if result.returncode is not None else -1
        trace_id = self._trace.exec_command(
            command=command,
            exit_code=exit_code,
            duration=result.elapsed,
            stdout=result.stdout.rstrip("\n"),
            stderr=result.stderr.rstrip("\n"),
            timed_out=result.timed_out,
            span=exec_span,
        )
        if result.timed_out:
            raise AglRaise(
                _make_exc_value(
                    "ExecError",
                    f"Shell command timed out (idle timeout exceeded): {command!r}",
                    trace_id=self._trace.new_event_id(),
                    command=TextValue(command),
                    exit_code=IntValue(exit_code),
                    stdout=TextValue(result.stdout.rstrip("\n")),
                    stderr=TextValue(result.stderr.rstrip("\n")),
                    timed_out=BoolValue(True),
                ),
                span=exec_span,
            )
        return result, trace_id

    def _run_parse_attempts(
        self,
        *,
        acquire: "Callable[[int, str | None, tuple[ValidationError, ...]], tuple[str, str]]",
        contract: "OutputContract",
        parse_policy: object,
        agent_label: str,
        make_failure_message: "Callable[[str | None, int], str]",
        on_parsed: "Callable[[str, ParseResult], None] | None" = None,
        raise_span: "SourceSpan | None" = None,
    ) -> Value:
        """Run the shared parse/retry loop and return the parsed value.

        *parse_policy* is an ``EnumValue`` with variant ``"Retry"`` for
        retry-with-N-extra-attempts, or anything else for single-attempt (abort).
        """
        # Determine max attempts from parse policy.
        if isinstance(parse_policy, EnumValue) and parse_policy.variant == "Retry":
            n_field = parse_policy.fields.get("n")
            extra = n_field.value if isinstance(n_field, IntValue) else 0
            max_attempts = 1 + extra
        else:
            max_attempts = 1

        if contract.strict_json is not None:
            effective_strict = contract.strict_json
        else:
            effective_strict = self._strict_json

        schema = contract.json_schema if isinstance(contract.json_schema, dict) else None

        last_raw: str | None = None
        last_normalized: str | None = None
        last_errors: tuple[ValidationError, ...] = ()
        last_trace_id = ""
        for attempt in range(max_attempts):
            raw, last_trace_id = acquire(attempt, last_raw, last_errors)
            result = contract.codec.parse(
                raw,
                contract.target_type,
                strict_json=effective_strict,
                schema=schema,
            )
            if on_parsed is not None:
                on_parsed(raw, result)
            if result.ok and result.value is not None:
                return result.value
            last_raw = raw
            last_normalized = result.normalized_raw
            if result.errors:
                last_errors = result.errors
            elif result.error_msg:
                from agm.agl.runtime.request import ValidationError as _VE

                last_errors = (
                    _VE(
                        category="invalid_json",
                        message=result.error_msg,
                        path="$",
                        field=None,
                    ),
                )
            else:
                last_errors = ()

        errors_json: list[object] = [e.to_json_obj() for e in last_errors]
        normalized_text = last_normalized if last_normalized is not None else (last_raw or "")
        raise AglRaise(
            _make_exc_value(
                "AgentParseError",
                make_failure_message(last_raw, max_attempts),
                trace_id=last_trace_id,
                raw=TextValue(last_raw or ""),
                normalized_raw=TextValue(normalized_text),
                agent=TextValue(agent_label),
                attempts=IntValue(max_attempts),
                target_type=TextValue(str(contract.target_type)),
                expected_schema=JsonValue(contract.json_schema),
                validation_errors=JsonValue(errors_json),
                metadata=JsonValue(None),
            ),
            span=raise_span,
        )

    # ------------------------------------------------------------------
    # Constructor and operator evaluation
    # ------------------------------------------------------------------

    def _eval_constructor(self, expr: Constructor, scope: Scope) -> Value:
        """Evaluate a record, enum-variant, or exception constructor."""
        arg_values: dict[str, Value] = {}
        for arg in expr.args:
            arg_values[arg.name] = self._eval_expr(arg.value, scope)

        typ = self._checked.node_types.get(expr.node_id)

        if isinstance(typ, RecordType):
            coerced = {
                fname: _coerce(fval, typ.fields[fname])
                for fname, fval in arg_values.items()
            }
            return RecordValue(type_name=typ.name, fields=coerced)

        from agm.agl.typecheck.types import ExceptionType as ExcType

        if isinstance(typ, ExcType):
            exc_trace_id = self._trace.new_event_id()
            fields: dict[str, Value] = {"trace_id": TextValue(exc_trace_id)}
            for fname, fval in arg_values.items():
                field_type = typ.fields.get(fname)
                fields[fname] = _coerce(fval, field_type) if field_type is not None else fval
            return ExceptionValue(type_name=typ.name, fields=fields)

        # Enum-variant constructor.
        assert isinstance(typ, EnumType), (
            "constructor type must be record, enum, or exception"
        )
        variant_name = expr.name
        variant_fields = typ.variants.get(variant_name, {})
        coerced2 = {
            fname: _coerce(fval, variant_fields[fname])
            for fname, fval in arg_values.items()
        }
        return EnumValue(type_name=typ.name, variant=variant_name, fields=coerced2)

    def _eval_binary_op(self, expr: BinaryOp, scope: Scope) -> Value:
        op = expr.op
        left = self._eval_expr(expr.left, scope)

        # Short-circuit for and/or.
        if op is BinOp.AND or op is BinOp.OR:
            return self._eval_bool_binop(op, left, expr.right, scope)

        right = self._eval_expr(expr.right, scope)

        if op is BinOp.ADD:
            return _add(left, right)
        if op is BinOp.SUB or op is BinOp.MUL:
            return _arith(left, right, op)
        if op is BinOp.DIV:
            return _div(left, right, trace=self._trace)

        if (
            op is BinOp.EQ
            or op is BinOp.NEQ
            or op is BinOp.LT
            or op is BinOp.LE
            or op is BinOp.GT
            or op is BinOp.GE
        ):
            return _compare(left, right, op)

        if op is BinOp.IN:
            return _in_op(left, right)

        assert_never(op)  # pragma: no cover

    def _eval_bool_binop(
        self, op: BinOp, left: Value, right_expr: Expr, scope: Scope
    ) -> BoolValue:
        """Short-circuit evaluation of ``and``/``or``."""
        left_bool = self._require_bool(left)
        if op is BinOp.AND:
            if not left_bool:
                return BoolValue(False)
        else:  # BinOp.OR
            if left_bool:
                return BoolValue(True)
        right = self._eval_expr(right_expr, scope)
        return BoolValue(self._require_bool(right))

    @staticmethod
    def _require_bool(value: Value) -> bool:
        if isinstance(value, BoolValue):
            return value.value
        raise AssertionError(  # pragma: no cover
            f"and/or operand is not a bool: {type(value).__name__}"
        )

    def _eval_unary_not(self, expr: UnaryNot, scope: Scope) -> BoolValue:
        operand = self._eval_expr(expr.operand, scope)
        if isinstance(operand, BoolValue):
            return BoolValue(not operand.value)
        raise AssertionError(  # pragma: no cover
            f"not: expected bool, got {type(operand).__name__}"
        )

    def _eval_unary_neg(self, expr: UnaryNeg, scope: Scope) -> IntValue | DecimalValue:
        operand = self._eval_expr(expr.operand, scope)
        if isinstance(operand, IntValue):
            return IntValue(-operand.value)
        if isinstance(operand, DecimalValue):
            return DecimalValue(-operand.value)
        raise RuntimeError(  # pragma: no cover
            f"unary -: expected number, got {type(operand).__name__}"
        )

    def _eval_is_test(self, expr: IsTest, scope: Scope) -> BoolValue:
        value = self._eval_expr(expr.expr, scope)
        if isinstance(value, EnumValue):
            variant_matches = value.variant == expr.variant
            result = variant_matches != expr.negated
            return BoolValue(result)
        raise RuntimeError(  # pragma: no cover
            f"is test on non-enum value: {type(value).__name__}"
        )

    def _eval_cast(self, expr: Cast, scope: Scope) -> Value:
        """Evaluate a ``cast`` or ``as?`` expression (M5).

        For ``as`` (``test_only=False``):
            Evaluates the source expression once, calls ``convert_value``, and on
            ``CastConversionError`` raises an ``AglRaise`` wrapping a ``CastError``
            ``ExceptionValue`` with the fields from D2.

        For ``as?`` (``test_only=True``):
            Evaluates the source expression once, then runs ``convert_value`` under a
            trial.  Returns ``BoolValue(True)`` on success and ``BoolValue(False)``
            only when ``CastConversionError`` is raised — all other exceptions
            (including ``AglRaise`` from the source and unexpected errors) propagate
            unchanged.
        """
        spec = self._checked.cast_specs[expr.node_id]
        target = spec.target_type
        source_type = self._checked.node_types[expr.expr.node_id]

        # Evaluate the source expression exactly once (for both as and as?).
        value = self._eval_expr(expr.expr, scope)

        if not expr.test_only:
            # as: convert; on failure raise CastError
            try:
                return convert_value(value, source_type, target)
            except CastConversionError as e:
                raise AglRaise(
                    _make_exc_value(
                        "CastError",
                        e.message,
                        trace_id=self._trace.new_event_id(),
                        source_type=TextValue(e.source_type),
                        target_type=TextValue(e.target_type),
                        raw=TextValue(e.raw),
                    ),
                    span=expr.span,
                ) from e
        else:
            # as?: total casts always succeed — short-circuit without converting.
            if spec.kind in (CastKind.TOTAL_NOOP, CastKind.TOTAL_RENDER, CastKind.TOTAL_JSON):
                return BoolValue(True)
            # Fallible cast: trial conversion — only catch CastConversionError.
            try:
                convert_value(value, source_type, target)
                return BoolValue(True)
            except CastConversionError:
                return BoolValue(False)

    def _eval_parse_json_call(self, expr: Call, scope: Scope) -> Value:
        """Evaluate a ``parse_json(text)`` call (M5).

        Parses the single positional ``text`` argument with the strict JSON parser.
        On success returns ``JsonValue(parsed_obj)``.  On ``StrictJsonParseError``
        raises an ``AglRaise`` wrapping a ``JsonParseError`` ``ExceptionValue``
        with fields ``message``, ``trace_id``, and ``raw``.
        """
        arg_val = self._eval_expr(expr.args[0], scope)
        assert isinstance(arg_val, TextValue), (
            f"parse_json: expected TextValue, got {type(arg_val).__name__}"
        )
        try:
            obj = parse_json_strict(arg_val.value)
        except StrictJsonParseError as e:
            raise AglRaise(
                _make_exc_value(
                    "JsonParseError",
                    e.message,
                    trace_id=self._trace.new_event_id(),
                    raw=TextValue(arg_val.value),
                ),
                span=expr.span,
            ) from e
        return JsonValue(obj)

    def _eval_list_lit(self, expr: ListLit, scope: Scope) -> ListValue:
        elements = tuple(self._eval_expr(e, scope) for e in expr.elements)
        return ListValue(elements=elements)

    def _eval_dict_lit(self, expr: DictLit, scope: Scope) -> DictValue:
        entries: dict[str, Value] = {}
        for entry in expr.entries:
            entries[entry.key.value] = self._eval_expr(entry.value, scope)
        return DictValue(entries=entries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _binding_type_for(self, decl_node_id: int) -> Type:
        target_type = self._type_env.get_binding_type(decl_node_id)
        assert target_type is not None, "binding type must be recorded by the checker"
        return target_type

    def _source_slice(self, span: "SourceSpan") -> str:
        """Return the normalized-source text covered by *span*."""
        if not self._source:
            return ""
        return self._source[span.start_offset : span.end_offset]

    def _get_function_signature(self, module_id: ModuleId, name: str) -> FunctionSignature | None:
        """Look up a function signature in the given module's signatures table."""
        sigs = self._module_sigs.get(module_id)
        if sigs is None:  # pragma: no cover
            return None
        return sigs.get(name)

    def execute_with_frames(
        self,
        entry_frame: Scope,
        module_frames: dict[ModuleId, Scope],
        module_sigs: dict[ModuleId, dict[str, FunctionSignature]],
    ) -> None:
        """Execute the entry module's block using pre-built per-module frames.

        Called by :func:`execute_graph` after all module frames have been
        assembled and closures installed.  Handles entry-module ``param``
        declarations and then runs the entry body.
        """
        with decimal.localcontext(_AGL_DECIMAL_CONTEXT):
            self._module_frames = module_frames
            self._module_sigs = module_sigs
            # Pre-pass: install entry-module params (agents and functions are
            # already installed in entry_frame by execute_graph).
            for item in self._checked.resolved.program.body.items:
                if isinstance(item, ParamDecl):
                    self._eval_param_decl(item, entry_frame)
            # Main pass.
            for item in self._checked.resolved.program.body.items:
                self._eval_item(item, entry_frame)


# ---------------------------------------------------------------------------
# Helpers: coercion, comparison, arithmetic, pattern matching
# ---------------------------------------------------------------------------


def _coerce(value: Value, target: Type) -> Value:
    """Coerce *value* toward its statically-checked *target* type (design §5.8).

    Materializes implicit coercions:
    - ``int → decimal`` widening;
    - ``json`` boundary wrapping;
    - element-wise recursion through list/dict/record/enum.
    """
    if isinstance(target, JsonType):
        if isinstance(value, JsonValue):
            return value
        return JsonValue(value_to_json_obj(value))
    if isinstance(target, DecimalType) and isinstance(value, IntValue):
        return DecimalValue(decimal.Decimal(value.value))
    if isinstance(target, ListType) and isinstance(value, ListValue):
        return ListValue(elements=tuple(_coerce(e, target.elem) for e in value.elements))
    if isinstance(target, DictType) and isinstance(value, DictValue):
        return DictValue(
            entries={k: _coerce(v, target.value) for k, v in value.entries.items()}
        )
    if isinstance(target, RecordType) and isinstance(value, RecordValue):
        return RecordValue(
            type_name=value.type_name,
            fields={
                k: _coerce(v, target.fields[k]) if k in target.fields else v
                for k, v in value.fields.items()
            },
        )
    if isinstance(target, EnumType) and isinstance(value, EnumValue):
        variant_fields = target.variants.get(value.variant, {})
        return EnumValue(
            type_name=value.type_name,
            variant=value.variant,
            fields={
                k: _coerce(v, variant_fields[k]) if k in variant_fields else v
                for k, v in value.fields.items()
            },
        )
    return value


def _add(left: Value, right: Value) -> Value:
    """Addition: numeric or text concatenation."""
    if isinstance(left, IntValue) and isinstance(right, IntValue):
        return IntValue(left.value + right.value)
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(
        right, (IntValue, DecimalValue)
    ):
        return DecimalValue(_to_decimal(left) + _to_decimal(right))
    if isinstance(left, TextValue) and isinstance(right, TextValue):
        return TextValue(left.value + right.value)
    raise RuntimeError(f"Cannot add {type(left).__name__} and {type(right).__name__}")


def _arith(left: Value, right: Value, op: BinOp) -> Value:
    """Subtraction and multiplication."""
    if isinstance(left, IntValue) and isinstance(right, IntValue):
        if op == BinOp.SUB:
            return IntValue(left.value - right.value)
        return IntValue(left.value * right.value)
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(
        right, (IntValue, DecimalValue)
    ):
        ld, rd = _to_decimal(left), _to_decimal(right)
        if op == BinOp.SUB:
            return DecimalValue(ld - rd)
        return DecimalValue(ld * rd)
    raise RuntimeError(f"Cannot perform {op.value} on {type(left).__name__}")


def _div(left: Value, right: Value, *, trace: "TraceStore") -> Value:
    """Division: always yields decimal.

    *trace* is used to mint an event id on division by zero.
    """
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(
        right, (IntValue, DecimalValue)
    ):
        rd = _to_decimal(right)
        if rd == decimal.Decimal(0):
            raise AglRaise(
                _make_exc_value(
                    "ArithmeticError",
                    "Division by zero",
                    trace_id=trace.new_event_id(),
                    operation=TextValue("/"),
                )
            )
        return DecimalValue(_to_decimal(left) / rd)
    raise RuntimeError(f"Cannot divide {type(left).__name__}")


def _to_decimal(value: Value) -> decimal.Decimal:
    if isinstance(value, IntValue):
        return decimal.Decimal(value.value)
    if isinstance(value, DecimalValue):
        return value.value
    raise RuntimeError(f"Not a numeric value: {type(value).__name__}")


def _value_eq(left: Value, right: Value) -> bool:
    """Value equality with int→decimal widening (design §4.3)."""
    if isinstance(left, IntValue) and isinstance(right, DecimalValue):
        return decimal.Decimal(left.value) == right.value
    if isinstance(left, DecimalValue) and isinstance(right, IntValue):
        return left.value == decimal.Decimal(right.value)
    return left == right


def _compare(left: Value, right: Value, op: BinOp) -> BoolValue:
    """Equality and ordering comparison."""
    if op == BinOp.EQ:
        return BoolValue(_value_eq(left, right))
    if op == BinOp.NEQ:
        return BoolValue(not _value_eq(left, right))

    # Widen for ordering comparison.
    if isinstance(left, IntValue) and isinstance(right, DecimalValue):
        left = DecimalValue(decimal.Decimal(left.value))
    elif isinstance(left, DecimalValue) and isinstance(right, IntValue):
        right = DecimalValue(decimal.Decimal(right.value))

    if isinstance(left, IntValue) and isinstance(right, IntValue):
        return _order_result(left.value, right.value, op)
    if isinstance(left, DecimalValue) and isinstance(right, DecimalValue):
        return _order_result(left.value, right.value, op)
    if isinstance(left, TextValue) and isinstance(right, TextValue):
        return _order_result(left.value, right.value, op)
    raise RuntimeError(
        f"Cannot compare {type(left).__name__} and {type(right).__name__}"
    )


def _order_result(left: _Ordered, right: _Ordered, op: BinOp) -> BoolValue:
    """Apply an ordering operator to two same-kind comparable keys."""
    if op == BinOp.LT:
        return BoolValue(left < right)
    if op == BinOp.LE:
        return BoolValue(left <= right)
    if op == BinOp.GT:
        return BoolValue(left > right)
    return BoolValue(left >= right)


def _in_op(left: Value, right: Value) -> BoolValue:
    """``x in container`` operator."""
    if isinstance(right, ListValue):
        return BoolValue(any(_value_eq(left, elem) for elem in right.elements))
    if isinstance(right, DictValue):
        if isinstance(left, TextValue):
            return BoolValue(left.value in right.entries)
        return BoolValue(False)
    if isinstance(right, TextValue) and isinstance(left, TextValue):
        return BoolValue(left.value in right.value)
    raise RuntimeError(
        f"Cannot use 'in' with {type(left).__name__} and {type(right).__name__}"
    )


def _matches_catch(handler: CatchClause, exc: ExceptionValue) -> bool:
    """Check if *handler* catches an exception of type *exc.type_name*."""
    if handler.exc_type is None:
        return True
    if handler.exc_type == "_" or handler.exc_type == "Exception":
        return True
    return handler.exc_type == exc.type_name


def _assign_target_slot_type(target: AssignTarget, root_type: Type) -> Type:
    """Return the type assigned by *target*, starting from *root_type*."""
    if isinstance(target, NameTarget):
        return root_type
    container_type = _index_target_container_type(target.obj, root_type)
    if isinstance(container_type, ListType):
        return container_type.elem
    if isinstance(container_type, DictType):
        return container_type.value
    raise RuntimeError(  # pragma: no cover
        f"Indexed assignment on non-list/dict type: {container_type!r}"
    )


def _index_target_container_type(obj: Expr, root_type: Type) -> Type:
    if isinstance(obj, VarRef):
        return root_type
    if isinstance(obj, IndexAccess):
        container_type = _index_target_container_type(obj.obj, root_type)
        if isinstance(container_type, ListType):
            return container_type.elem
        if isinstance(container_type, DictType):
            return container_type.value
    raise RuntimeError(  # pragma: no cover
        "Indexed assignment requires a variable list or dict root."
    )


def _match_pattern(pattern: Pattern, value: Value) -> tuple[bool, dict[str, Value]]:
    """Try to match *pattern* against *value*.

    Returns ``(matched, bindings)`` where ``bindings`` maps new variable
    names to their bound values.
    """
    from agm.agl.syntax.nodes import (
        BoolLit,
        ConstructorPattern,
        DecimalLit,
        IntLit,
        LiteralPattern,
        NullLit,
        StringLit,
    )

    if isinstance(pattern, WildcardPattern):
        return True, {}

    if isinstance(pattern, VarPattern):
        return True, {pattern.name: value}

    if isinstance(pattern, LiteralPattern):
        lit = pattern.literal
        if isinstance(lit, IntLit):
            pat_val: Value = IntValue(lit.value)
        elif isinstance(lit, DecimalLit):
            pat_val = DecimalValue(lit.value)
        elif isinstance(lit, BoolLit):
            pat_val = BoolValue(lit.value)
        elif isinstance(lit, StringLit):
            pat_val = TextValue(lit.value)
        elif isinstance(lit, NullLit):
            pat_val = JsonValue(None)
        else:
            assert_never(lit)  # pragma: no cover
        return _value_eq(value, pat_val), {}

    if isinstance(pattern, ConstructorPattern):
        assert isinstance(value, EnumValue), "constructor pattern on non-enum value"
        if value.variant != pattern.name:
            return False, {}
        bindings: dict[str, Value] = {}
        for field_pat in pattern.fields:
            field_val = value.fields[field_pat.name]
            matched, sub_bindings = _match_pattern(field_pat.pattern, field_val)
            if not matched:
                return False, {}
            bindings.update(sub_bindings)
        return True, bindings

    assert_never(pattern)  # pragma: no cover


def _describe_value(value: Value) -> str:
    """Return the AgL type-name of *value* (design §8.1 ``scrutinee_type``)."""
    if isinstance(value, EnumValue):
        return value.type_name
    if isinstance(value, RecordValue):
        return value.type_name
    if isinstance(value, ExceptionValue):
        return value.type_name
    if isinstance(value, TextValue):
        return "text"
    if isinstance(value, IntValue):
        return "int"
    if isinstance(value, DecimalValue):
        return "decimal"
    if isinstance(value, BoolValue):
        return "bool"
    if isinstance(value, JsonValue):
        return "json"
    if isinstance(value, ListValue):
        return "list"
    if isinstance(value, DictValue):
        return "dict"
    if isinstance(value, UnitValue):
        return "unit"
    if isinstance(value, AgentValue):
        return "agent"
    # Closure is the only remaining Value member.
    assert isinstance(value, Closure), f"unexpected value kind: {type(value).__name__}"
    return "function"


# ---------------------------------------------------------------------------
# Module-graph execution
# ---------------------------------------------------------------------------


def _merge_graph_into_checked_program(
    entry_cm: CheckedModule,
    checked_graph: CheckedModuleGraph,
) -> CheckedProgram:
    """Build a merged :class:`CheckedProgram` from the entry module plus all library modules.

    The interpreter uses a single ``CheckedProgram`` for dispatch tables
    (``resolved.builtin_calls``, ``node_types``, ``contract_specs``,
    ``cast_specs``).  Because library module bodies (closures) are also
    evaluated by the same interpreter, their per-module side tables must be
    merged into a single combined program.  Node ids are disjoint across
    modules (M2 seeds each module with a distinct id range), so the merge is
    unambiguous.
    """
    from agm.agl.scope.symbols import BindingRef, BuiltinKind
    from agm.agl.scope.symbols import ResolvedProgram as _RP
    from agm.agl.typecheck.env import CheckedProgram as _CP
    from agm.agl.typecheck.env import OutputContractSpec
    from agm.agl.typecheck.types import CastSpec

    # Merge resolution and builtin_calls from all modules into a combined view
    # rooted at the entry program.
    merged_resolution: dict[int, BindingRef] = dict(entry_cm.resolved.resolution)
    merged_builtin_calls: dict[int, BuiltinKind] = dict(entry_cm.resolved.builtin_calls)
    for mid, cm in checked_graph.modules.items():
        if mid == checked_graph.entry_id:
            continue
        merged_resolution.update(cm.resolved.resolution)
        merged_builtin_calls.update(cm.resolved.builtin_calls)

    # Build a merged ResolvedProgram using the entry's program and root_scope
    # but with combined resolution + builtin_calls tables.
    merged_resolved = _RP(
        program=entry_cm.resolved.program,
        resolution=merged_resolution,
        builtin_calls=merged_builtin_calls,
        root_scope=entry_cm.resolved.root_scope,
        declared_agents=entry_cm.resolved.declared_agents,
        declared_functions=entry_cm.resolved.declared_functions,
        config_pragmas=entry_cm.resolved.config_pragmas,
        program_name=entry_cm.resolved.program_name,
        warnings=entry_cm.resolved.warnings,
    )

    # Merge node_types, contract_specs, cast_specs from all modules.
    merged_node_types: dict[int, Type] = dict(entry_cm.node_types)
    merged_contract_specs: dict[int, OutputContractSpec] = dict(entry_cm.contract_specs)
    merged_cast_specs: dict[int, CastSpec] = dict(entry_cm.cast_specs)
    for mid, cm in checked_graph.modules.items():
        if mid == checked_graph.entry_id:
            continue
        merged_node_types.update(cm.node_types)
        merged_contract_specs.update(cm.contract_specs)
        merged_cast_specs.update(cm.cast_specs)

    return _CP(
        resolved=merged_resolved,
        node_types=merged_node_types,
        contract_specs=merged_contract_specs,
        call_sites=entry_cm.call_sites,
        warnings=entry_cm.warnings,
        type_env=entry_cm.type_env,
        function_signatures=entry_cm.function_signatures,
        cast_specs=merged_cast_specs,
    )


def execute_graph(
    checked_graph: CheckedModuleGraph,
    registry: AgentRegistry,
    contracts: dict[int, OutputContract],
    *,
    loop_limit: int,
    strict_json: bool,
    source: str = "",
    shell_exec_timeout: float | None = None,
    trace: TraceStore | None = None,
    max_call_depth: int = 256,
    param_values: Mapping[str, Value] | None = None,
) -> dict[str, Value]:
    """Execute an AgL module graph starting from the entry module.

    Builds one :class:`~agm.agl.eval.scope.Scope` frame per module, installs
    closures for all ``def`` declarations across all modules, then executes
    the entry module's body.

    Parameters
    ----------
    checked_graph:
        Output of :func:`~agm.agl.typecheck.graph.check_graph`.
    registry:
        Agent registry for ``ask`` dispatch.
    contracts:
        Materialized :class:`~agm.agl.runtime.contract.OutputContract` per
        agent/exec call node id.
    loop_limit:
        Default iteration bound for ``do`` loops.
    strict_json:
        Default strict-JSON flag for codec operations.
    source:
        Entry module source text (for error context slicing).
    shell_exec_timeout:
        Idle timeout for ``exec`` calls, or ``None`` for no limit.
    trace:
        Trace store, or ``None`` for the no-op store.
    max_call_depth:
        Recursion depth limit.
    param_values:
        Optional mapping of param name → value for entry module params.

    Returns
    -------
    dict[str, Value]
        Snapshot of the entry module's root scope after execution.
    """
    # Build one Scope frame per module (all at once so forward refs are safe).
    module_frames: dict[ModuleId, Scope] = {
        mid: Scope(parent=None) for mid in checked_graph.modules
    }

    # Install closures for every module's def declarations into ITS frame.
    # All frames are created before any closure is installed so mutual
    # recursion across modules works (closure.env captures the frame ref).
    for mid, cm in checked_graph.modules.items():
        frame = module_frames[mid]
        for item in cm.resolved.program.body.items:
            if isinstance(item, FuncDef):
                sig = cm.function_signatures.get(item.name)
                ret_type: Type = sig.result if sig is not None else UnitType()
                params: tuple[tuple[str, Expr | None], ...] = tuple(
                    (p.name, p.default) for p in item.params
                )
                closure = Closure(
                    env=frame, params=params, body=item.body, return_type=ret_type
                )
                frame.define(item.name, closure, mutable=False, decl_span=item.span)

    # Install entry-module agent declarations into the entry frame.
    entry_cm = checked_graph.modules[checked_graph.entry_id]
    entry_frame = module_frames[checked_graph.entry_id]
    for item in entry_cm.resolved.program.body.items:
        if isinstance(item, AgentDecl):
            entry_frame.define(
                item.name, AgentValue(name=item.name), mutable=False, decl_span=item.span
            )

    # Build module_sigs for cross-module function signature lookup.
    module_sigs: dict[ModuleId, dict[str, FunctionSignature]] = {
        mid: cm.function_signatures for mid, cm in checked_graph.modules.items()
    }

    # Create a merged CheckedProgram that combines all modules' dispatch tables.
    # Node ids are disjoint across modules (M2 seeds each with a distinct range)
    # so the merge is unambiguous.  Library closures are evaluated by the same
    # interpreter and need their builtin_calls/node_types/etc. tables present.
    entry_checked = _merge_graph_into_checked_program(entry_cm, checked_graph)
    interp = Interpreter(
        checked=entry_checked,
        registry=registry,
        contracts=contracts,
        type_env=entry_cm.type_env,
        loop_limit=loop_limit,
        strict_json=strict_json,
        source=source,
        shell_exec_timeout=shell_exec_timeout,
        trace=trace,
        max_call_depth=max_call_depth,
        param_values=param_values,
    )
    interp.execute_with_frames(entry_frame, module_frames, module_sigs)
    return entry_frame.snapshot()
