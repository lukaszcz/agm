"""AgL tree-walking interpreter (Component 6).

``Interpreter`` evaluates a ``CheckedProgram`` using a ``WorkflowRuntime``
and a root ``Scope``.  It implements all M1 statements and expressions.

Control flow for AgL exceptions uses Python exceptions (``AglRaise``).
"""

from __future__ import annotations

import decimal
from typing import TYPE_CHECKING, TypeVar, assert_never

from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.scope import Scope
from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.syntax.nodes import (
    AgentCall,
    BinaryOp,
    BinOp,
    BoolLit,
    CaseExpr,
    CaseStmt,
    CatchClause,
    Constructor,
    DecimalLit,
    DictLit,
    DoUntil,
    ElseSentinel,
    EnumDef,
    Expr,
    ExprStmt,
    FieldAccess,
    IfStmt,
    InputDecl,
    InterpSegment,
    IntLit,
    IsTest,
    LetDecl,
    ListLit,
    NullLit,
    PassStmt,
    Pattern,
    PrintStmt,
    Raise,
    RecordDef,
    SetStmt,
    Stmt,
    StringLit,
    Template,
    TextSegment,
    TryCatch,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.typecheck.types import (
    DecimalType,
    DictType,
    EnumType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.runtime.render import RendererFn
    from agm.agl.typecheck.env import CheckedProgram, TypeEnvironment


_Ordered = TypeVar("_Ordered", int, decimal.Decimal, str)


def _make_exc_value(type_name: str, message: str, **extra: Value) -> ExceptionValue:
    """Create an ``ExceptionValue`` with ``message`` and optional extra fields."""
    fields: dict[str, Value] = {
        "message": TextValue(message),
        "trace_id": TextValue(""),  # M1: no trace store yet
    }
    fields.update(extra)
    return ExceptionValue(type_name=type_name, fields=fields)


class Interpreter:
    """Tree-walking interpreter for a checked AgL program.

    ``checked``   — the type-checked program with side tables.
    ``registry``  — the host agent registry.
    ``contracts`` — materialized ``OutputContract`` per agent-call node_id.
    ``type_env`` — the ``TypeEnvironment`` from the checked program, used to
        resolve record/enum constructors at runtime.
    ``renderers`` — the interpolation renderer table (built-ins merged with any
        host-registered renderers); threaded into every ``${expr as name}``
        rendering so registered renderers are actually invoked (F1, M3b).
    ``loop_limit`` — default bound for ``do`` loops without an explicit limit.
    ``strict_json`` — default strict-JSON flag for codec operations.
    """

    def __init__(
        self,
        checked: "CheckedProgram",
        registry: "AgentRegistry",
        contracts: dict[int, "OutputContract"],
        type_env: "TypeEnvironment",
        renderers: "Mapping[str, RendererFn] | None" = None,
        *,
        loop_limit: int,
        strict_json: bool,
    ) -> None:
        from agm.agl.runtime.render import builtin_renderers

        self._checked = checked
        self._registry = registry
        self._contracts = contracts
        self._type_env = type_env
        # ``WorkflowRuntime.run`` always passes the merged built-in + registered
        # renderer table (F1).  ``None`` (e.g. direct construction in unit tests
        # that exercise only built-in rendering) falls back to the built-ins.
        self._renderers: "Mapping[str, RendererFn]" = (
            renderers if renderers is not None else builtin_renderers()
        )
        self._loop_limit = loop_limit
        self._strict_json = strict_json
        # Root scope — populated from inputs before execute() is called.
        self._root_scope: Scope = Scope(parent=None)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, root_scope: Scope) -> None:
        """Execute the program body in *root_scope*.

        May raise ``AglRaise`` for uncaught AgL exceptions.
        """
        self._root_scope = root_scope
        for stmt in self._checked.resolved.program.body:
            self._exec_stmt(stmt, root_scope)

    # ------------------------------------------------------------------
    # Statement execution
    # ------------------------------------------------------------------

    def _exec_stmt(self, stmt: Stmt, scope: Scope) -> None:
        if isinstance(stmt, LetDecl):
            self._exec_let(stmt, scope)
        elif isinstance(stmt, VarDecl):
            self._exec_var(stmt, scope)
        elif isinstance(stmt, SetStmt):
            self._exec_set(stmt, scope)
        elif isinstance(stmt, PassStmt):
            pass
        elif isinstance(stmt, PrintStmt):
            self._exec_print(stmt, scope)
        elif isinstance(stmt, ExprStmt):
            self._eval_expr(stmt.expr, scope)  # result discarded
        elif isinstance(stmt, DoUntil):
            self._exec_do_until(stmt, scope)
        elif isinstance(stmt, IfStmt):
            self._exec_if(stmt, scope)
        elif isinstance(stmt, CaseStmt):
            self._exec_case_stmt(stmt, scope)
        elif isinstance(stmt, TryCatch):
            self._exec_try_catch(stmt, scope)
        elif isinstance(stmt, Raise):
            self._exec_raise(stmt, scope)
        elif isinstance(stmt, InputDecl):
            pass  # already pre-bound; nothing to do at execution time
        elif isinstance(stmt, (RecordDef, EnumDef, TypeAlias)):
            pass  # type declarations: no runtime action
        else:
            assert_never(stmt)  # pragma: no cover

    def _exec_let(self, stmt: LetDecl, scope: Scope) -> None:
        value = self._eval_expr(stmt.value, scope)
        # The checker records a declared type for every let binding; coerce the
        # value toward it (int → decimal, element-wise in containers, json
        # boundary wrapping — design §5.8).
        target_type = self._binding_type_for(stmt.node_id)
        scope.define(stmt.name, _coerce(value, target_type), mutable=False, decl_span=stmt.span)

    def _exec_var(self, stmt: VarDecl, scope: Scope) -> None:
        value = self._eval_expr(stmt.value, scope)
        target_type = self._binding_type_for(stmt.node_id)
        scope.define(stmt.name, _coerce(value, target_type), mutable=True, decl_span=stmt.span)

    def _exec_set(self, stmt: SetStmt, scope: Scope) -> None:
        value = self._eval_expr(stmt.value, scope)
        # Coerce toward the mutable binding's declared type (design §5.8).
        ref = self._checked.resolved.resolution[stmt.node_id]
        target_type = self._binding_type_for(ref.decl_node_id)
        scope.set_value(stmt.target, _coerce(value, target_type))

    def _binding_type_for(self, decl_node_id: int) -> Type:
        """Return the declared type the checker recorded for a binding node.

        The type checker assigns a declared type to every ``let``/``var``/input
        binding before evaluation, so this is always present.
        """
        target_type = self._type_env.get_binding_type(decl_node_id)
        assert target_type is not None, "binding type must be recorded by the checker"
        return target_type

    def _exec_print(self, stmt: PrintStmt, scope: Scope) -> None:
        from agm.agl.runtime.render import render_for_console

        value = self._eval_expr(stmt.value, scope)
        text = render_for_console(value)
        print(text)

    def _exec_do_until(self, stmt: DoUntil, scope: Scope) -> None:
        limit = stmt.limit if stmt.limit is not None else self._loop_limit
        for iteration in range(limit):
            # Each iteration opens a fresh nested scope.
            iter_scope = Scope(parent=scope)
            for s in stmt.body:
                self._exec_stmt(s, iter_scope)
            cond = self._eval_expr(stmt.condition, iter_scope)
            if isinstance(cond, BoolValue) and cond.value:
                return
        # Exhausted without condition becoming true.
        raise AglRaise(
            _make_exc_value(
                "MaxIterationsExceeded",
                f"Loop exhausted after {limit} iterations",
                limit=IntValue(limit),
            )
        )

    def _exec_if(self, stmt: IfStmt, scope: Scope) -> None:
        for branch in stmt.branches:
            if isinstance(branch.cond, ElseSentinel):
                branch_scope = Scope(parent=scope)
                for s in branch.body:
                    self._exec_stmt(s, branch_scope)
                return
            cond = self._eval_expr(branch.cond, scope)
            if isinstance(cond, BoolValue) and cond.value:
                branch_scope = Scope(parent=scope)
                for s in branch.body:
                    self._exec_stmt(s, branch_scope)
                return
        # No branch matched and no else: do nothing.

    def _exec_case_stmt(self, stmt: CaseStmt, scope: Scope) -> None:
        subject = self._eval_expr(stmt.subject, scope)
        for branch in stmt.branches:
            matched, bindings = _match_pattern(branch.pattern, subject)
            if matched:
                branch_scope = Scope(parent=scope)
                for name, val in bindings.items():
                    branch_scope.define(
                        name, val, mutable=False, decl_span=branch.span
                    )
                for s in branch.body:
                    self._exec_stmt(s, branch_scope)
                return
        # No match: raise MatchError.
        raise AglRaise(
            _make_exc_value(
                "MatchError",
                f"Non-exhaustive case: no pattern matched value {_describe_value(subject)!r}",
            )
        )

    def _exec_try_catch(self, stmt: TryCatch, scope: Scope) -> None:
        try:
            try_scope = Scope(parent=scope)
            for s in stmt.body:
                self._exec_stmt(s, try_scope)
        except AglRaise as exc:
            for handler in stmt.handlers:
                if _matches_catch(handler, exc.exc):
                    catch_scope = Scope(parent=scope)
                    if handler.binding is not None:
                        catch_scope.define(
                            handler.binding,
                            exc.exc,
                            mutable=False,
                            decl_span=handler.span,
                        )
                    for s in handler.body:
                        self._exec_stmt(s, catch_scope)
                    return
            # No handler matched: re-propagate.
            raise

    def _exec_raise(self, stmt: Raise, scope: Scope) -> None:
        exc_val = self._eval_expr(stmt.exc, scope)
        if isinstance(exc_val, ExceptionValue):
            raise AglRaise(exc_val)
        raise RuntimeError(
            f"raise: expected an ExceptionValue, got {type(exc_val).__name__}"
        )

    # ------------------------------------------------------------------
    # Expression evaluation
    # ------------------------------------------------------------------

    def _eval_expr(self, expr: Expr, scope: Scope) -> Value:
        # Scalar literals
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
        # Name / member access
        if isinstance(expr, VarRef):
            return self._eval_var_ref(expr, scope)
        if isinstance(expr, FieldAccess):
            return self._eval_field_access(expr, scope)
        # Compound expressions (templates, calls, operators, …).
        if isinstance(expr, Template):
            return self._eval_template(expr, scope)
        if isinstance(expr, AgentCall):
            return self._eval_agent_call(expr, scope)
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
        if isinstance(expr, CaseExpr):
            return self._eval_case_expr(expr, scope)
        if isinstance(expr, ListLit):
            return self._eval_list_lit(expr, scope)
        if isinstance(expr, DictLit):
            return self._eval_dict_lit(expr, scope)
        assert_never(expr)  # pragma: no cover

    def _eval_unary_not(self, expr: UnaryNot, scope: Scope) -> BoolValue:
        operand = self._eval_expr(expr.operand, scope)
        if isinstance(operand, BoolValue):
            return BoolValue(not operand.value)
        raise RuntimeError(f"not: expected bool, got {type(operand).__name__}")

    def _eval_unary_neg(self, expr: UnaryNeg, scope: Scope) -> IntValue | DecimalValue:
        operand = self._eval_expr(expr.operand, scope)
        if isinstance(operand, IntValue):
            return IntValue(-operand.value)
        if isinstance(operand, DecimalValue):
            return DecimalValue(-operand.value)
        # Unreachable: the type checker requires a numeric operand for unary '-'.
        raise RuntimeError(  # pragma: no cover
            f"unary -: expected number, got {type(operand).__name__}"
        )

    def _eval_list_lit(self, expr: ListLit, scope: Scope) -> ListValue:
        elements = tuple(self._eval_expr(e, scope) for e in expr.elements)
        return ListValue(elements=elements)

    def _eval_dict_lit(self, expr: DictLit, scope: Scope) -> DictValue:
        entries: dict[str, Value] = {}
        for entry in expr.entries:
            entries[entry.key.value] = self._eval_expr(entry.value, scope)
        return DictValue(entries=entries)

    def _eval_var_ref(self, expr: VarRef, scope: Scope) -> Value:
        binding = scope.lookup(expr.name)
        if binding is None:  # pragma: no cover
            # Unreachable: name resolution rejects undefined variables statically.
            raise RuntimeError(f"Undefined variable at runtime: {expr.name!r}")
        return binding.value

    def _eval_field_access(self, expr: FieldAccess, scope: Scope) -> Value:
        obj = self._eval_expr(expr.obj, scope)
        # The type checker only admits field access on record and exception
        # values, and it guarantees the named field exists, so missing-field /
        # wrong-kind cases are statically unreachable.
        if isinstance(obj, (RecordValue, ExceptionValue)):
            return obj.fields[expr.field]
        raise RuntimeError(  # pragma: no cover
            f"Field access on non-record/exception: {type(obj).__name__}"
        )

    def _eval_template(self, expr: Template, scope: Scope) -> TextValue:
        """Evaluate a template by rendering each segment and concatenating."""
        from agm.agl.runtime.render import render_for_prompt

        parts: list[str] = []
        for seg in expr.segments:
            if isinstance(seg, TextSegment):
                parts.append(seg.text)
            elif isinstance(seg, InterpSegment):
                value = self._eval_expr(seg.expr, scope)
                # Determine the variable name for the boundary tag.
                var_name: str | None = None
                if isinstance(seg.expr, VarRef):
                    var_name = seg.expr.name
                parts.append(
                    render_for_prompt(
                        value,
                        renderer_name=seg.render,
                        var_name=var_name,
                        renderers=self._renderers,
                    )
                )
            else:
                assert_never(seg)  # pragma: no cover
        return TextValue("".join(parts))

    def _eval_agent_call(self, expr: AgentCall, scope: Scope) -> Value:
        """Dispatch an agent call and return the typed result."""
        from agm.agl.scope.symbols import CallKind

        call_kind = self._checked.resolved.call_kinds.get(expr.node_id)
        if call_kind == CallKind.shell_exec:
            # Unreachable through the public ``run`` pipeline: the checker rejects
            # ``exec`` calls statically until M4 (``supports_shell_exec``).  Reaching
            # here means the interpreter was driven with a hand-built
            # CheckedProgram that bypassed that gate — an internal invariant
            # violation, NOT a user-facing AgL exception.  Fail loudly.
            raise AssertionError(
                "shell_exec call reached the interpreter; the checker must reject "
                "'exec' until M4 (supports_shell_exec). Real exec support lands in M4."
            )

        # Determine the agent name for dispatch.
        if call_kind is None or call_kind == CallKind.default_agent:
            agent_name = "prompt"
        else:
            agent_name = expr.agent

        # Get the output contract for this call site.
        contract = self._contracts.get(expr.node_id)
        if contract is None:
            # Fallback for text-target calls without a contract (shouldn't happen
            # after typecheck, but defensive).
            from agm.agl.runtime.codec import TextCodec
            from agm.agl.runtime.contract import OutputContract

            contract = OutputContract(
                target_type=TextType(),
                codec=TextCodec(),
                strict_json=None,
                format_instructions="",
                json_schema=None,
            )

        # Resolve the parse policy (from call options or runtime default).
        parse_policy = expr.options.parse_policy

        # Perform the call with retry support.
        return self._dispatch_agent_call(
            agent_name=agent_name,
            expr=expr,
            scope=scope,
            contract=contract,
            parse_policy=parse_policy,
        )

    def _dispatch_agent_call(
        self,
        agent_name: str,
        expr: AgentCall,
        scope: Scope,
        contract: "OutputContract",
        parse_policy: object,  # ParsePolicy | None
    ) -> Value:
        """Execute an agent call with the given contract and parse policy."""
        from agm.agl.runtime.request import AgentRequest, ValidationError
        from agm.agl.syntax.nodes import RetryPolicy

        # Determine max retries.
        if isinstance(parse_policy, RetryPolicy):
            max_attempts = 1 + parse_policy.extra
        else:
            max_attempts = 1  # AbortPolicy or None → one attempt only

        # Render the template once (re-used on retries per design).
        rendered_template = self._eval_template(expr.template, scope)
        prompt_text = rendered_template.value

        # Determine effective strict_json.
        if contract.strict_json is not None:
            effective_strict = contract.strict_json
        else:
            effective_strict = self._strict_json

        last_raw: str | None = None
        last_normalized: str | None = None
        last_errors: tuple[ValidationError, ...] = ()
        for attempt in range(max_attempts):
            request = AgentRequest(
                agent=agent_name,
                prompt=prompt_text,
                attempt=attempt,
                previous_invalid_output=last_raw,
                validation_errors=list(last_errors),
                output_contract=contract,
            )
            response = self._registry.dispatch(agent_name, request)
            raw = response.content

            # Parse via the codec.  Reuse the schema already materialized on the
            # contract so the codec never re-derives it per attempt (BONUS, M3b).
            schema = contract.json_schema if isinstance(contract.json_schema, dict) else None
            result = contract.codec.parse(
                raw,
                contract.target_type,
                strict_json=effective_strict,
                schema=schema,
            )
            if result.ok and result.value is not None:
                return result.value

            last_raw = raw
            last_normalized = result.normalized_raw
            last_errors = result.errors

        # All attempts exhausted → raise AgentParseError.  Validation errors are
        # threaded as JSON-shaped values (a list of per-error objects) so the
        # exception's ``validation_errors`` field (text-renderable) matches the
        # exception field schema (design §7.5 / §7.9).  ``normalized_raw`` carries
        # the recovered/extracted JSON text when the failure was a schema or
        # conversion error (F5); it falls back to the raw output when no value
        # could be recovered at all.
        errors_json: list[object] = [e.to_json_obj() for e in last_errors]
        normalized_text = last_normalized if last_normalized is not None else (last_raw or "")
        raise AglRaise(
            _make_exc_value(
                "AgentParseError",
                f"Agent {agent_name!r} failed to produce a valid {contract.target_type!r} "
                f"after {max_attempts} attempt(s). Last output: {last_raw!r}",
                raw=TextValue(last_raw or ""),
                normalized_raw=TextValue(normalized_text),
                agent=TextValue(agent_name),
                attempts=IntValue(max_attempts),
                target_type=TextValue(str(contract.target_type)),
                expected_schema=JsonValue(contract.json_schema),
                validation_errors=JsonValue(errors_json),
                metadata=JsonValue(None),
            )
        )

    def _eval_constructor(self, expr: Constructor, scope: Scope) -> Value:
        """Evaluate a record or enum-variant constructor."""
        # Evaluate arguments.
        arg_values: dict[str, Value] = {}
        for arg in expr.args:
            arg_values[arg.name] = self._eval_expr(arg.value, scope)

        # The checker resolved this constructor's type (records the resolved
        # nominal type in ``node_types``), already resolving any alias qualifier
        # transparently (design §5.4) — the interpreter never re-resolves it.
        typ = self._checked.node_types.get(expr.node_id)

        # The type checker validates that the constructor names a known type and
        # that every supplied field is declared, so each ``arg`` field is present
        # in the corresponding record/variant field table.
        if isinstance(typ, RecordType):
            coerced = {
                fname: _coerce(fval, typ.fields[fname]) for fname, fval in arg_values.items()
            }
            return RecordValue(type_name=typ.name, fields=coerced)
        # Otherwise an enum-variant constructor.  The variant is the constructor's
        # own name (the qualifier, if any, only selects the enum).
        assert isinstance(typ, EnumType), "constructor type must be record or enum"
        variant_name = expr.name
        variant_fields = typ.variants.get(variant_name, {})
        coerced2 = {
            fname: _coerce(fval, variant_fields[fname]) for fname, fval in arg_values.items()
        }
        return EnumValue(type_name=typ.name, variant=variant_name, fields=coerced2)

    def _eval_binary_op(self, expr: BinaryOp, scope: Scope) -> Value:
        op = expr.op
        left = self._eval_expr(expr.left, scope)

        # Short-circuit for and/or (right operand evaluated lazily). The checker
        # guarantees both operands are bool, so the result is always a BoolValue
        # (design §4.3).
        if op is BinOp.AND or op is BinOp.OR:
            return self._eval_bool_binop(op, left, expr.right, scope)

        right = self._eval_expr(expr.right, scope)

        # Arithmetic.
        if op is BinOp.ADD:
            return _add(left, right)
        if op is BinOp.SUB or op is BinOp.MUL:
            return _arith(left, right, op)
        if op is BinOp.DIV:
            return _div(left, right)

        # Comparison.
        if (
            op is BinOp.EQ
            or op is BinOp.NEQ
            or op is BinOp.LT
            or op is BinOp.LE
            or op is BinOp.GT
            or op is BinOp.GE
        ):
            return _compare(left, right, op)

        # Membership.
        if op is BinOp.IN:
            return _in_op(left, right)

        assert_never(op)  # pragma: no cover

    def _eval_bool_binop(
        self, op: BinOp, left: Value, right_expr: Expr, scope: Scope
    ) -> BoolValue:
        """Short-circuit evaluation of ``and``/``or`` returning a BoolValue.

        The checker guarantees both operands are bool. ``and`` short-circuits
        when the left operand is ``False``; ``or`` short-circuits when it is
        ``True`` — in those cases the right operand is NOT evaluated.
        """
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
        # Unreachable: the checker requires bool operands for 'and'/'or'.
        raise AssertionError(  # pragma: no cover
            f"and/or operand is not a bool: {type(value).__name__}"
        )

    def _eval_is_test(self, expr: IsTest, scope: Scope) -> BoolValue:
        value = self._eval_expr(expr.expr, scope)
        if isinstance(value, EnumValue):
            variant_matches = value.variant == expr.variant
            result = variant_matches != expr.negated
            return BoolValue(result)
        # Unreachable: 'is' tests require an enum-typed operand statically.
        raise RuntimeError(  # pragma: no cover
            f"is test on non-enum value: {type(value).__name__}"
        )

    def _eval_case_expr(self, expr: CaseExpr, scope: Scope) -> Value:
        subject = self._eval_expr(expr.subject, scope)
        for branch in expr.branches:
            matched, bindings = _match_pattern(branch.pattern, subject)
            if matched:
                branch_scope = Scope(parent=scope)
                for name, val in bindings.items():
                    branch_scope.define(
                        name, val, mutable=False, decl_span=branch.span
                    )
                return self._eval_expr(branch.body, branch_scope)
        raise AglRaise(
            _make_exc_value(
                "MatchError",
                "Non-exhaustive case expression: no pattern matched",
            )
        )


# ---------------------------------------------------------------------------
# Helpers: coercion, comparison, arithmetic, pattern matching
# ---------------------------------------------------------------------------


def _coerce(value: Value, target: Type) -> Value:
    """Coerce *value* toward its statically-checked *target* type (design §5.8).

    The checker has already proven the value is assignable to *target*; this
    materializes the implicit coercions in the runtime representation:

    - ``int → decimal`` widening (the single scalar coercion);
    - ``json`` boundary: a JSON-shaped value bound to a ``json`` slot is stored
      in the one canonical ``json`` representation — a :class:`JsonValue`
      wrapping the JSON-shaped object (so ``print``/rendering/equality are
      consistent regardless of the literal it came from);
    - element-wise recursion through ``list``/``dict``/record/enum so e.g.
      ``int`` widens to ``decimal`` and json-wrapping applies inside containers.
    """
    if isinstance(target, JsonType):
        # Already json → leave as-is; otherwise wrap the JSON-shaped object.
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
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(right, (IntValue, DecimalValue)):
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
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(right, (IntValue, DecimalValue)):
        ld, rd = _to_decimal(left), _to_decimal(right)
        if op == BinOp.SUB:
            return DecimalValue(ld - rd)
        return DecimalValue(ld * rd)
    raise RuntimeError(f"Cannot perform {op.value} on {type(left).__name__}")


def _div(left: Value, right: Value) -> Value:
    """Division: always yields decimal."""
    if isinstance(left, (IntValue, DecimalValue)) and isinstance(right, (IntValue, DecimalValue)):
        rd = _to_decimal(right)
        if rd == decimal.Decimal(0):
            raise AglRaise(
                _make_exc_value(
                    "ArithmeticError",
                    "Division by zero",
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
    """Value equality with int→decimal widening (design §4.3).

    The single source of truth for ``=`` comparison and ``in`` membership, so
    ``IntValue(1)`` equals ``DecimalValue(1)`` consistently across both passes.
    """
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

    # Ordering: numeric or text operands of the same kind.  ``_compare`` only
    # reaches here with ``op`` in {LT, LE, GT, GE}; EQ/NEQ are handled earlier
    # and no other op flows into ``_compare``.
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
    """Apply an ordering operator to two same-kind comparable keys.

    Reached only with ``op`` in {LT, LE, GT, GE}; the four ordering ops are
    exhaustive, so the final branch is ``GE`` (no unreachable fallback).
    """
    if op == BinOp.LT:
        return BoolValue(left < right)
    if op == BinOp.LE:
        return BoolValue(left <= right)
    if op == BinOp.GT:
        return BoolValue(left > right)
    # Only GE remains.
    return BoolValue(left >= right)


def _in_op(left: Value, right: Value) -> BoolValue:
    """``x in container`` operator.

    List membership uses the same value-equality semantics as ``=`` (incl.
    int→decimal widening), so ``1 in [1.0]`` is true (design §4.3).
    """
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
        # Bare ``catch`` without a type: catches everything.
        return True
    # Match by type name: catches the exact type or any subtype.
    # In v1 the hierarchy is flat: every concrete type is a subtype of
    # "Exception" (the abstract base).
    if handler.exc_type == "_" or handler.exc_type == "Exception":
        return True
    return handler.exc_type == exc.type_name


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
        return value == pat_val, {}

    if isinstance(pattern, ConstructorPattern):
        # The type checker only admits constructor patterns against enum
        # subjects and guarantees each named field exists on the variant, so the
        # subject is always an EnumValue and every field lookup succeeds.
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
    """Return a brief human-readable description of *value* for error messages."""
    if isinstance(value, EnumValue):
        return f"{value.type_name}.{value.variant}"
    if isinstance(value, RecordValue):
        return value.type_name
    if isinstance(value, ExceptionValue):
        return value.type_name
    return type(value).__name__
