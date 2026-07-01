"""Type-checking pass (Component 5) for AgL v2.

``check(resolved, capabilities)`` performs a bidirectional type pass over the
``ResolvedProgram``, using the ``HostCapabilities`` to validate codec and
renderer names, and returns a ``CheckedProgram``.

Rules implemented
-----------------
1.  Type declaration validation: duplicate names, unknown referenced types,
    recursive records/enums, alias cycles, and built-in-name shadowing.
2.  Function declarations: parameter/return types resolved; ordering enforced
    (required before defaulted); FunctionSignature registered.
3.  Binding type inference:
    - ``let/var name: T = e`` — check ``e`` against ``T``.
    - Other untyped initializers infer from the literal/expression.
    - ``param name[: T] [= default]`` — defaults to ``text`` when unannotated
      and defaultless; otherwise defaults are checked/inferred.
4.  ``name := e`` — expected type is the binding's declared type.
5.  ``print(expr)`` — accepts any value and yields ``unit``.
6.  ``render(expr, pretty:, quote_strings:)`` — accepts any value and yields ``text``.
7.  ``ask(prompt, ...)`` — named-agent or default-agent call with codec.
8.  ``exec(cmd, ...)`` — shell call; requires ``supports_shell_exec``.
9.  Declared-name calls — checked against the full ``FunctionSignature``.
10. Value calls — checked against the ``FunctionType``; named args disallowed.
11. Lambdas — inferred or annotated return type.
12. Block typing — last item is the block's value; LetDecl/VarDecl at end is error.
13. ``if`` with no ``else`` yields ``unit``; with ``else`` branches must unify.
14. ``case`` — exhaustiveness warning on enum scrutinees.
15. ``loop`` — yields ``unit``; until/while conditions must be bool; bound must be int.
16. ``try/catch`` — body and handler types must unify.
17. ``raise`` — yields ``BottomType`` (bottom, assignable to any target).
18. Assignability (design §5.8): ``int`` widens to ``decimal``; ``json``
    accepts any JSON-shaped value.  Bottom type is assignable to any target.
19. Duplicate constructor argument names, duplicate dict keys, and all the
    constructor checks carried over from v1.

The checker raises ``AglTypeError`` on the first error (first-error abort).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeGuard

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    BinderKind,
    BindingRef,
    BuiltinKind,
    ResolvedProgram,
)
from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    BottomType,
    CastKind,
    CastSpec,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
    cast_classification,
    comparable_types,
    contains_type_var,
    is_assignable,
    substitute,
)
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    BinaryOp,
    BinOp,
    Block,
    BoolLit,
    Break,
    Call,
    Case,
    Cast,
    CatchClause,
    ConfigDecl,
    ConstructorPattern,
    Continue,
    DecimalLit,
    DictLit,
    ElseSentinel,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    Expr,
    FieldAccess,
    FuncDef,
    If,
    ImportDecl,
    IndexAccess,
    IndexTarget,
    InfixDecl,
    InterpSegment,
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    LiteralPattern,
    Loop,
    NameTarget,
    NullLit,
    Param,
    ParamDecl,
    ParamKind,
    Pattern,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    StringLit,
    Template,
    Try,
    TypeAlias,
    TypeApply,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr
from agm.agl.typecheck.arguments import bind_call_args, bind_pattern_args
from agm.agl.typecheck.builder import _BUILTIN_TYPE_NAMES as _BUILTIN_TYPE_NAMES
from agm.agl.typecheck.builder import _TypeBuilder
from agm.agl.typecheck.builtins import BuiltinCallChecker
from agm.agl.typecheck.constructors import ConstructorChecker
from agm.agl.typecheck.env import (
    AglTypeError,
    ArgumentBindings,
    CallSiteRecord,
    CheckedProgram,
    FunctionSignature,
    OutputContractSpec,
    ParamSpec,
    TypeEnvironment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Built-in function names that user-defined defs may not shadow. Derived from
# the single source of truth in ``scope.symbols`` so the two never drift.
_BUILTIN_FUNC_NAMES: frozenset[str] = frozenset(BUILTIN_CALL_NAMES)


_IndexLike = IndexAccess | IndexTarget


def _std_param(name: str, typ: Type, has_default: bool = False) -> ParamSpec:
    """Create a ``ParamSpec`` with ``STANDARD`` kind (used for all built-in params)."""
    return ParamSpec(name=name, type=typ, kind=ParamKind.STANDARD, has_default=has_default)


def _builtin_function_signature(name: str) -> FunctionSignature | None:
    t = TypeVarType("T")
    match name:
        case "print":
            return FunctionSignature(
                params=(_std_param("value", t),), result=UnitType(), type_params=("T",)
            )
        case "render":
            return FunctionSignature(
                params=(_std_param("value", t),), result=TextType(), type_params=("T",)
            )
        case "parse_json":
            return FunctionSignature(
                params=(_std_param("value", TextType()),), result=JsonType()
            )
        case "ask":
            return FunctionSignature(
                params=(
                    _std_param("prompt", TextType()),
                    _std_param("agent", AgentType(), has_default=True),
                    _std_param("format", TextType(), has_default=True),
                    _std_param("strict_json", BoolType(), has_default=True),
                    _std_param(
                        "on_parse_error",
                        EnumType(name="ParsePolicy", variants={}),
                        has_default=True,
                    ),
                ),
                result=t,
                type_params=("T",),
            )
        case "ask-request":
            return FunctionSignature(
                params=(_std_param("prompt", TextType()),),
                result=RecordType(name="AgentRequest", fields={}),
            )
        case "exec":
            return FunctionSignature(
                params=(_std_param("command", TextType()),),
                result=RecordType(name="ExecResult", fields={}),
            )
        case _:
            return None


def _builtin_function_signature_alternates(name: str) -> tuple[FunctionSignature, ...]:
    expected = _builtin_function_signature(name)
    if expected is None:
        return ()
    if name == "ask":
        return (
            expected,
            FunctionSignature(
                params=(_std_param("prompt", TextType()),), result=TextType()
            ),
        )
    return (expected,)


def _signature_matches(actual: FunctionSignature, expected: FunctionSignature) -> bool:
    if actual.type_params != expected.type_params:
        return False
    if len(actual.params) != len(expected.params):
        return False
    for ap, ep in zip(actual.params, expected.params):
        if ap.name != ep.name or ap.has_default != ep.has_default:
            return False
        if isinstance(ep.type, RecordType):
            if not isinstance(ap.type, RecordType) or ap.type.name != ep.type.name:
                return False
        elif isinstance(ep.type, EnumType):
            if not isinstance(ap.type, EnumType) or ap.type.name != ep.type.name:
                return False
        elif ap.type != ep.type:
            return False
    if isinstance(expected.result, RecordType):
        return isinstance(actual.result, RecordType) and actual.result.name == expected.result.name
    return actual.result == expected.result


def _is_index_like(node: object) -> TypeGuard[_IndexLike]:
    return isinstance(node, (IndexAccess, IndexTarget))


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------


class _Checker:
    """Stateful type-checking visitor for AgL v2.

    Walks the program's items in order, maintaining a binding-type lookup
    table (``node_id → Type``) populated for declarations and inline inference.

    Uses expected-type propagation: ``_check_expr(node, expected)`` propagates
    an outer type context into the expression.  When ``expected`` is ``None``
    the expression is inferred bottom-up.
    """

    def __init__(
        self,
        env: TypeEnvironment,
        resolved: ResolvedProgram,
        capabilities: HostCapabilities,
    ) -> None:
        self._env = env
        self._resolved = resolved
        self._caps = capabilities
        self._node_types: dict[int, Type] = {}
        self._contract_specs: dict[int, OutputContractSpec] = {}
        self._call_sites: list[CallSiteRecord] = []
        self._warnings: list[Diagnostic] = []
        # Type variables currently in scope (non-empty inside a generic def body).
        self._current_type_vars: frozenset[str] = frozenset()
        self._cast_specs: dict[int, CastSpec] = {}
        # Argument bindings computed during the check, reused by the lowerer so it
        # never re-binds.  Keyed by Call/Pattern node_id (see ``ArgumentBindings``).
        self._function_call_bindings: dict[int, tuple[Expr | None, ...]] = {}
        self._constructor_call_bindings: dict[int, dict[str, Expr]] = {}
        self._constructor_pattern_bindings: dict[int, tuple[tuple[str, Pattern], ...]] = {}
        self._builtins = BuiltinCallChecker(self)
        self._constructors = ConstructorChecker(self)

    # ------------------------------------------------------------------
    # D6b: required-after-defaulted check (shared by def and lambda)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required_after_defaulted(params: Sequence[Param]) -> None:
        """D6b: no required positional-fillable param may follow a defaulted one.

        Named-only params are order-free — their defaults may appear in any order.
        Raises ``AglTypeError`` if any POSITIONAL_ONLY or STANDARD param without a
        default appears after one with a default.
        """
        seen_pos_default = False
        for p in params:
            is_pos_fillable = p.kind in (ParamKind.POSITIONAL_ONLY, ParamKind.STANDARD)
            if is_pos_fillable:
                has_default = p.default is not None
                if has_default:
                    seen_pos_default = True
                elif seen_pos_default:
                    raise AglTypeError(
                        f"Parameter '{p.name}' has no default but follows a defaulted "
                        "positional parameter. Required positional parameters must come "
                        "before parameters with defaults.",
                        span=p.span,
                    )

    # ------------------------------------------------------------------
    # Pre-registration of function signatures
    # ------------------------------------------------------------------

    def _preregister_funcdef(self, node: FuncDef) -> None:
        """Resolve and register the signature of a top-level ``def``."""
        self._validate_funcdef_header(node)
        if node.return_type is None:
            return

        sig, func_type = self._build_funcdef_signature(node, result_type=node.return_type)
        self._register_funcdef_signature(node, sig, func_type)

    def _validate_funcdef_header(self, node: FuncDef) -> None:
        """Validate declaration-level properties that do not need a return type."""
        if node.name in _BUILTIN_TYPE_NAMES:
            raise AglTypeError(
                f"'{node.name}' is a built-in type name and cannot be used as a function name.",
                span=node.span,
            )
        if node.name in _BUILTIN_FUNC_NAMES and not node.is_builtin:
            raise AglTypeError(
                f"'{node.name}' is a built-in function name and cannot be redefined.",
                span=node.span,
            )
        if node.is_builtin and node.name not in _BUILTIN_FUNC_NAMES:
            raise AglTypeError(
                f"Unknown builtin function '{node.name}'.",
                span=node.span,
            )
        if node.is_builtin and node.return_type is None:
            raise AglTypeError(
                f"Builtin function '{node.name}' must declare a return type.",
                span=node.span,
            )
        # D6b: no required positional-fillable param may follow a defaulted one.
        self._check_required_after_defaulted(node.params)

    def _build_funcdef_signature(
        self, node: FuncDef, *, result_type: TypeExpr | Type
    ) -> tuple[FunctionSignature, FunctionType]:
        """Resolve a ``def`` signature with either an annotated or inferred result."""
        type_vars: frozenset[str] = frozenset(node.type_params)
        params: list[ParamSpec] = []
        for p in node.params:
            pt = self._env.resolve_type_expr(p.type_expr, span=p.span, type_vars=type_vars)
            has_default = p.default is not None
            params.append(ParamSpec(name=p.name, type=pt, kind=p.kind, has_default=has_default))

        resolved_result = (
            self._env.resolve_type_expr(result_type, span=node.span, type_vars=type_vars)
            if isinstance(result_type, TypeExpr)
            else result_type
        )
        sig = FunctionSignature(
            params=tuple(params), result=resolved_result, type_params=node.type_params
        )
        func_type = FunctionType(
            params=tuple(p.type for p in params),
            result=resolved_result,
        )
        return sig, func_type

    def _register_funcdef_signature(
        self, node: FuncDef, sig: FunctionSignature, func_type: FunctionType
    ) -> None:
        """Register a resolved ``def`` signature in every function side table."""
        if node.is_builtin:
            expected_sigs = _builtin_function_signature_alternates(node.name)
            assert expected_sigs
            if not any(_signature_matches(sig, expected_sig) for expected_sig in expected_sigs):
                raise AglTypeError(
                    f"Builtin function '{node.name}' has an invalid signature.",
                    span=node.span,
                )
        self._env.register_function_signature(node.name, sig)
        self._env.register_function_signature_by_node_id(node.node_id, sig)
        self._env.set_binding_type(node.node_id, func_type)

    # ------------------------------------------------------------------
    # Program-level check
    # ------------------------------------------------------------------

    def check_program(self, program: Program) -> None:
        """Type-check the entire program."""
        # Pre-pass: register all FuncDef signatures so calls can reference them
        # in declaration order (mutual forward references are not supported but
        # order-independence within the block is).
        for item in program.body.items:
            if isinstance(item, FuncDef):
                self._preregister_funcdef(item)

        self._check_block(program.body, expected=None)

    # ------------------------------------------------------------------
    # Block and item dispatch
    # ------------------------------------------------------------------

    def _check_block(self, block: Block, *, expected: Type | None) -> Type:
        """Type-check a block and return the type of its last item."""
        if not block.items:
            return UnitType()

        last = block.items[-1]
        if isinstance(last, (LetDecl, VarDecl)):
            raise AglTypeError(
                "a 'let'/'var' declaration must be followed by an expression in a block.",
                span=last.span,
            )

        result_type: Type = UnitType()
        for item in block.items:
            item_type = self._check_item(item, expected=expected if item is last else None)
            if item is last:
                result_type = item_type
        return result_type

    def _check_item(self, item: Item, *, expected: Type | None) -> Type:
        """Dispatch a single block item, returning its type contribution."""
        # --- Declarations ---
        if isinstance(item, FuncDef):
            # Signature already registered in pre-pass; check body now.
            self._check_funcdef_body(item)
            return UnitType()
        if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            return UnitType()
        if isinstance(item, ConfigDecl):
            self._check_config(item)
            return UnitType()
        if isinstance(item, AgentDecl):
            self._env.set_binding_type(item.node_id, AgentType())
            return UnitType()
        if isinstance(item, ParamDecl):
            self._check_param(item)
            return UnitType()
        if isinstance(item, ProgramDecl):
            return UnitType()
        if isinstance(item, (ImportDecl, ExportDecl, InfixDecl)):
            return UnitType()  # The graph module-system pass processes imports/exports.
        # --- Binders ---
        if isinstance(item, (LetDecl, VarDecl)):
            self._check_binding(item)
            return UnitType()
        if isinstance(item, AssignStmt):
            self._check_assign_stmt(item)
            return UnitType()
        # --- Expr ---
        return self._check_expr(item, expected=expected)

    # ------------------------------------------------------------------
    # Declaration checkers
    # ------------------------------------------------------------------

    def _check_funcdef_body(self, node: FuncDef) -> None:
        """Check the body of a ``def`` against its registered signature."""
        sig = self._env.get_function_signature(node.name)
        if sig is None:
            self._infer_funcdef_signature(node)
            return
        if node.is_builtin:
            return
        assert node.body is not None, f"FuncDef '{node.name}' has no body"
        # Save and update current type vars for this def's scope. A non-generic
        # def resets the set to empty (defs never nest, but this stays correct
        # regardless): the body's annotations see exactly this def's type vars.
        old_type_vars = self._current_type_vars
        self._current_type_vars = frozenset(sig.type_params)
        try:
            # Bind params in the env.
            for p, spec in zip(node.params, sig.params):
                self._env.set_binding_type(p.node_id, spec.type)
            # Check defaults against declared parameter types.
            for p, spec in zip(node.params, sig.params):
                if p.default is not None:
                    def_type = self._check_expr(p.default, expected=spec.type)
                    self._assert_assignable(def_type, spec.type, p.span)
            # Check body against declared return type.
            body_type = self._check_expr(node.body, expected=sig.result)
            if not isinstance(body_type, BottomType):
                self._assert_assignable(body_type, sig.result, node.span)
        finally:
            self._current_type_vars = old_type_vars

    def _infer_funcdef_signature(self, node: FuncDef) -> None:
        """Infer and register an unannotated ``def`` signature from its body."""
        self._validate_funcdef_header(node)
        assert node.return_type is None
        assert node.body is not None, f"FuncDef '{node.name}' has no body"
        sig, _func_type = self._build_funcdef_signature(node, result_type=UnitType())
        old_type_vars = self._current_type_vars
        self._current_type_vars = frozenset(sig.type_params)
        try:
            for p, spec in zip(node.params, sig.params):
                self._env.set_binding_type(p.node_id, spec.type)
            for p, spec in zip(node.params, sig.params):
                if p.default is not None:
                    def_type = self._check_expr(p.default, expected=spec.type)
                    self._assert_assignable(def_type, spec.type, p.span)
            body_type = self._check_expr(node.body, expected=None)
            if isinstance(body_type, BottomType):
                raise AglTypeError(
                    f"Cannot infer return type of function '{node.name}': body always raises. "
                    "Add a return type annotation.",
                    span=node.span,
                )
            inferred_sig = FunctionSignature(
                params=sig.params, result=body_type, type_params=sig.type_params
            )
            inferred_type = FunctionType(
                params=tuple(p.type for p in sig.params), result=body_type
            )
            self._register_funcdef_signature(node, inferred_sig, inferred_type)
        finally:
            self._current_type_vars = old_type_vars

    def _check_param(self, stmt: ParamDecl) -> None:
        ann_type = (
            self._env.resolve_type_expr(stmt.annotation, span=stmt.span)
            if stmt.annotation is not None
            else None
        )
        if stmt.default is not None:
            val_type = self._check_expr(stmt.default, expected=ann_type)
            if ann_type is not None:
                self._assert_assignable(val_type, ann_type, stmt.span)
                declared_type = ann_type
            else:
                if isinstance(val_type, BottomType):
                    raise AglTypeError(
                        "Cannot infer type of param: default always raises. "
                        "Add a type annotation.",
                        span=stmt.span,
                    )
                declared_type = val_type
        else:
            declared_type = ann_type if ann_type is not None else TextType()
        self._env.set_binding_type(stmt.node_id, declared_type)

    def _check_config(self, stmt: ConfigDecl) -> None:
        """Check a ``config`` declaration against the engine-key registry.

        For ``Option[T]`` engine keys (``log-file``, ``timeout``) the value
        expression may be either an ``Option[T]`` or the inner type ``T``: a bare
        ``T`` value is projected into ``some(value)`` by the lowerer.  For all
        other keys the value, if present, must be assignable to the engine-key
        type.  The engine-key type is recorded as the binding type either way.
        """
        from agm.agl.semantics.engine_keys import get_engine_key_type

        declared_type = get_engine_key_type(stmt.name)
        if declared_type is None:
            return  # pragma: no cover — unknown keys rejected earlier by scope
        if stmt.value is not None:
            val_type = self._check_expr(stmt.value, expected=declared_type)
            if isinstance(declared_type, EnumType) and declared_type.type_args:
                inner = declared_type.type_args[0]
                if not (
                    is_assignable(val_type, declared_type)
                    or is_assignable(val_type, inner)
                ):
                    raise AglTypeError(
                        f"config '{stmt.name}' expects '{declared_type!r}' or "
                        f"'{inner!r}', got '{val_type!r}'.",
                        span=stmt.span,
                    )
            else:
                self._assert_assignable(val_type, declared_type, stmt.span)
        self._env.set_binding_type(stmt.node_id, declared_type)

    def _check_binding(self, stmt: LetDecl | VarDecl) -> None:
        ann_type = self._resolve_annotation(stmt.type_ann, stmt.span)
        val_type = self._check_expr(stmt.value, expected=ann_type)
        if ann_type is not None:
            self._assert_assignable(val_type, ann_type, stmt.span)
            declared_type = ann_type
        else:
            if isinstance(val_type, BottomType):
                raise AglTypeError(
                    "Cannot infer type of binding: value always raises. Add a type annotation.",
                    span=stmt.span,
                )
            declared_type = val_type
        self._env.set_binding_type(stmt.node_id, declared_type)

    def _check_assign_stmt(self, stmt: AssignStmt) -> None:
        if isinstance(stmt.target, NameTarget):
            ref = self._resolved.resolution[stmt.node_id]
            target_type = self._require_binding_type(ref)
            val_type = self._check_expr(stmt.value, expected=target_type)
            self._assert_assignable(val_type, target_type, stmt.span)
            return

        if isinstance(stmt.target, IndexTarget):
            self._check_indexed_assign_stmt(stmt, stmt.target)
            return

        raise AglTypeError(
            "assignment target must be a mutable variable or indexed mutable variable.",
            span=stmt.span,
        )

    def _check_indexed_assign_stmt(self, stmt: AssignStmt, target: IndexTarget) -> None:
        ref = self._resolved.resolution[stmt.node_id]
        if not ref.mutable:
            raise AglTypeError(
                f"Cannot assign through index of '{ref.name}': "
                "indexed assignment requires a mutable 'var' binding.",
                span=target.span,
            )
        root_type = self._require_binding_type(ref)
        elem_type = self._check_index_target_type(target, root_type)
        value_type = self._check_expr(stmt.value, expected=elem_type)
        self._assert_assignable(value_type, elem_type, stmt.span)

    def _check_index_target_type(self, target: IndexTarget, root_type: Type) -> Type:
        container_type = self._check_index_target_container_type(target.obj, root_type)
        return self._check_index_operand(container_type, target.index, span=target.span)

    def _check_index_target_container_type(self, obj: Expr, root_type: Type) -> Type:
        if isinstance(obj, VarRef):
            return root_type
        if isinstance(obj, IndexAccess):
            container_type = self._check_index_target_container_type(obj.obj, root_type)
            indexed_type = self._check_index_operand(container_type, obj.index, span=obj.span)
            self._node_types[obj.node_id] = indexed_type
            return indexed_type
        raise AglTypeError(
            "indexed assignment requires a variable list or dict root.",
            span=obj.span,
        )

    # ------------------------------------------------------------------
    # Expression type inference
    # ------------------------------------------------------------------

    def _check_expr(self, expr: Expr, *, expected: Type | None) -> Type:
        """Infer/check the type of *expr*, recording it in ``_node_types``."""
        typ = self._infer_expr(expr, expected=expected)
        self._node_types[expr.node_id] = typ
        return typ

    def _infer_expr(self, expr: Expr, *, expected: Type | None) -> Type:
        """Bottom-up inference with optional top-down ``expected`` context."""
        if isinstance(expr, UnitLit):
            return UnitType()
        if isinstance(expr, IntLit):
            return IntType()
        if isinstance(expr, DecimalLit):
            return DecimalType()
        if isinstance(expr, BoolLit):
            return BoolType()
        if isinstance(expr, NullLit):
            return JsonType()
        if isinstance(expr, StringLit):
            return TextType()
        if isinstance(expr, Template):
            return self._check_template(expr)
        if isinstance(expr, VarRef):
            return self._check_varref(expr, expected=expected)
        if isinstance(expr, TypeApply):
            return self._check_type_apply(expr)
        if isinstance(expr, Call):
            return self._check_call(expr, expected=expected)
        if isinstance(expr, Lambda):
            return self._check_lambda(expr, expected=expected)
        if isinstance(expr, Block):
            return self._check_block(expr, expected=expected)
        if isinstance(expr, If):
            return self._check_if(expr, expected=expected)
        if isinstance(expr, Case):
            return self._check_case(expr, expected=expected)
        if isinstance(expr, Loop):
            return self._check_loop(expr)
        if isinstance(expr, Try):
            return self._check_try(expr, expected=expected)
        if isinstance(expr, Raise):
            return self._infer_raise(expr)
        if isinstance(expr, Break | Continue):
            return BottomType()
        if isinstance(expr, BinaryOp):
            return self._check_binary_op(expr)
        if isinstance(expr, UnaryNot):
            operand_type = self._check_expr(expr.operand, expected=None)
            if not isinstance(operand_type, BoolType):
                raise AglTypeError(
                    f"'not' requires a bool operand; got '{operand_type!r}'.",
                    span=expr.operand.span,
                )
            return BoolType()
        if isinstance(expr, UnaryNeg):
            return self._check_unary_neg(expr)
        if isinstance(expr, IsTest):
            return self._check_is_test(expr)
        if isinstance(expr, FieldAccess):
            return self._check_field_access(expr, expected=expected)
        if _is_index_like(expr):
            return self._check_index_access(expr)
        if isinstance(expr, ListLit):
            return self._check_list_lit(expr, expected=expected)
        if isinstance(expr, Cast):
            return self._check_cast(expr)
        # DictLit is the last literal Expr union member (Break/Continue handled above).
        assert isinstance(expr, DictLit), f"Unexpected expression kind: {type(expr).__name__}"
        return self._check_dict_lit(expr, expected=expected)

    # --- VarRef ---

    def _check_varref(self, node: VarRef, *, expected: Type | None = None) -> Type:
        # Bare constructor reference → zero-arg construction or generic constructor as value.
        if node.node_id in self._resolved.constructor_refs:
            ctor_ref = self._resolved.constructor_refs[node.node_id]
            if ctor_ref.type_params:
                return self._constructors.check_generic_constructor_as_value(
                    ctor_ref=ctor_ref, span=node.span, expected=expected
                )
            owner = self._constructors.resolve_constructor_owner(ctor_ref, node.span)
            return self._constructors.check_constructor_as_value(
                owner=owner, variant=ctor_ref.variant, span=node.span
            )
        ref = self._resolved.resolution[node.node_id]
        # A constructor_binding resolves to a type declaration, not a value.
        # Catch bare type name references (e.g. ``mylib::Color``) and raise a
        # user-facing error instead of an internal assertion failure.
        if ref.kind is BinderKind.constructor_binding:
            raise AglTypeError(
                f"'{node.name}' is a type name, not a value; "
                "use it with a constructor call (e.g. 'EnumName.Variant' or 'RecordName(...)').",
                span=node.span,
            )
        typ = self._require_binding_type(ref)
        # D5: generic def used as a value — must be instantiated from context.
        # Use the node-id-keyed lookup (populated by the graph function-signature
        # pre-pass and by _preregister_funcdef) to get the correct signature even
        # when two modules define functions with the same name but different signatures.
        # Both _preregister_funcdef (single-module) and the graph pre-pass seed the
        # node-id table, so the name-keyed fallback is not needed.
        if ref.kind is BinderKind.function_binding:
            sig = self._env.get_function_signature_by_node_id(ref.decl_node_id)
            if sig is not None and sig.type_params:
                if not isinstance(expected, FunctionType):
                    raise AglTypeError(
                        f"Cannot infer type arguments for generic function '{ref.name}' "
                        f"used as a value; annotate the binding "
                        f"(e.g. 'let f: (int) -> int = {ref.name}') or call it directly.",
                        span=node.span,
                    )
                assert isinstance(typ, FunctionType)
                subst: dict[str, Type] = {}
                self._match(typ, expected, subst, span=node.span, challenge=False)
                for p in sig.type_params:
                    if p not in subst:
                        raise AglTypeError(
                            f"Cannot infer type argument '{p}' for generic function "
                            f"'{ref.name}' from the expected type; "
                            f"annotate the binding more precisely.",
                            span=node.span,
                        )
                return substitute(typ, subst)
        return typ

    def _require_binding_type(self, ref: BindingRef) -> Type:
        typ = self._env.resolve_binding(ref)
        if typ is None and ref.kind is BinderKind.function_binding:
            raise AglTypeError(
                f"Cannot infer return type of function '{ref.name}' before it is checked. "
                "Add a return type annotation.",
                span=None,
            )
        assert typ is not None, f"Binding {ref!r} has no recorded type; checker invariant violated."
        return typ

    # --- Explicit value-position type application ---

    def _resolve_explicit_type_args(
        self,
        *,
        owner_name: str,
        type_params: tuple[str, ...],
        type_args: tuple[TypeExpr, ...],
        span: SourceSpan,
    ) -> dict[str, Type]:
        if len(type_args) != len(type_params):
            raise AglTypeError(
                f"'{owner_name}' requires {len(type_params)} type argument(s), "
                f"but {len(type_args)} were supplied.",
                span=span,
            )
        return {
            p: self._env.resolve_type_expr(ta, span=span, type_vars=self._current_type_vars)
            for p, ta in zip(type_params, type_args)
        }

    def _check_type_apply(self, node: TypeApply) -> Type:
        # A constructor used as a value with explicit type arguments
        # (e.g. ``some::[int]`` or ``Option.none::[int]``): delegate to the
        # constructor checker, which instantiates the constructor with the
        # supplied type arguments and returns a function value (payload) or
        # the constructed nominal value (nullary variant).
        if isinstance(node.expr, VarRef) and node.expr.node_id in self._resolved.constructor_refs:
            ctor_ref = self._resolved.constructor_refs[node.expr.node_id]
            typ = self._constructors.check_constructor_type_apply(
                ctor_ref=ctor_ref, type_args=node.type_args, span=node.span
            )
            self._node_types[node.expr.node_id] = typ
            return typ
        if (
            isinstance(node.expr, FieldAccess)
            and node.expr.node_id in self._resolved.qualified_constructor_refs
        ):
            owner_name, variant, owner_module_id = (
                self._resolved.qualified_constructor_refs[node.expr.node_id]
            )
            typ = self._constructors.check_qualified_constructor_type_apply(
                owner_name=owner_name,
                variant=variant,
                owner_module_id=owner_module_id,
                type_args=node.type_args,
                span=node.span,
            )
            self._node_types[node.expr.node_id] = typ
            return typ

        if not isinstance(node.expr, VarRef):
            raise AglTypeError(
                "Explicit type arguments can only be applied to a generic function value.",
                span=node.span,
            )

        ref = self._resolved.resolution.get(node.expr.node_id)
        if ref is None or ref.kind is not BinderKind.function_binding:
            raise AglTypeError(
                "Explicit type arguments can only be applied to a generic function value.",
                span=node.span,
            )

        sig = self._env.get_function_signature_by_node_id(ref.decl_node_id)
        if sig is None:
            raise AglTypeError(
                f"Cannot infer return type of function '{ref.name}' before it is checked. "
                "Add a return type annotation.",
                span=node.span,
            )
        if not sig.type_params:
            raise AglTypeError(
                f"'{ref.name}' is not a generic function and does not accept type arguments.",
                span=node.span,
            )

        typ = self._require_binding_type(ref)
        assert isinstance(typ, FunctionType)
        subst = self._resolve_explicit_type_args(
            owner_name=ref.name,
            type_params=sig.type_params,
            type_args=node.type_args,
            span=node.span,
        )
        concrete = substitute(typ, subst)
        self._node_types[node.expr.node_id] = concrete
        return concrete

    # --- Cast ---

    def _check_cast(self, node: Cast) -> Type:
        source_type = self._check_expr(node.expr, expected=None)
        target_type = self._env.resolve_type_expr(node.target_type, span=node.span)
        kind = cast_classification(source_type, target_type)
        if kind == CastKind.STATIC_ERROR:
            raise AglTypeError(
                f"cannot cast '{source_type!r}' to '{target_type!r}'.",
                span=node.span,
            )
        self._cast_specs[node.node_id] = CastSpec(target_type=target_type, kind=kind)
        return BoolType() if node.test_only else target_type

    # --- Call dispatch ---

    def _check_call(self, node: Call, *, expected: Type | None) -> Type:
        """Dispatch a Call node to the appropriate checker."""
        # Built-in?
        if node.node_id in self._resolved.builtin_calls:
            kind = self._resolved.builtin_calls[node.node_id]
            if kind == BuiltinKind.PRINT:
                return self._builtins.check_print(node)
            if kind == BuiltinKind.RENDER:
                return self._builtins.check_render(node)
            if kind == BuiltinKind.ASK:
                return self._builtins.check_ask(node, expected=expected)
            if kind == BuiltinKind.ASK_REQUEST:
                return self._builtins.check_ask_request(node)
            if kind == BuiltinKind.PARSE_JSON:
                return self._builtins.check_parse_json(node)
            # EXEC
            return self._builtins.check_exec(node, expected=expected)

        # Constructor call?
        if (
            isinstance(node.callee, VarRef)
            and node.callee.node_id in self._resolved.constructor_refs
        ):
            return self._constructors.check_constructor_callee_call(node, expected=expected)
        if (
            isinstance(node.callee, FieldAccess)
            and node.callee.node_id in self._resolved.qualified_constructor_refs
        ):
            return self._constructors.check_qualified_constructor_callee_call(
                node, expected=expected
            )

        # Declared function or cross-module constructor by name?
        # Take the declared-name (named/default) path ONLY when the callee is a
        # bare VarRef that resolves to a top-level function_binding (a ``def``).
        # A let/var-bound function value, a param, or a field access must all
        # take the value-call path — they are not declared names and do not
        # support named/defaulted arguments.
        # Exception: a cross-module constructor_binding (module_qualifier present)
        # is handled as a constructor call.
        if isinstance(node.callee, VarRef):
            callee_ref = self._resolved.resolution.get(node.callee.node_id)
            if callee_ref is not None and callee_ref.kind is BinderKind.function_binding:
                return self._check_declared_name_call(
                    node,
                    node.callee.name,
                    expected=expected,
                    callee_node_id=callee_ref.decl_node_id,
                )
            if (
                callee_ref is not None
                and callee_ref.kind is BinderKind.constructor_binding
                and not callee_ref.module_id.is_entry
            ):
                return self._constructors.check_cross_module_constructor_call(
                    node, callee_ref, expected=expected
                )

        # Value call (lambda or higher-order).
        return self._check_value_call(node, expected=expected)

    # --- type-variable matching (one-sided unification) ---

    def _match(
        self,
        template: Type,
        concrete: Type,
        subst: dict[str, Type],
        *,
        span: SourceSpan,
        challenge: bool = True,
    ) -> None:
        """One-sided unification: bind type vars in *template* to *concrete* types.

        With ``challenge=True`` (the default, used for argument inference) an
        already-bound type var that disagrees with *concrete* raises
        ``AglTypeError``.  With ``challenge=False`` only currently-unbound
        variables are bound and existing bindings are left untouched — used to
        fill remaining type vars from an expected result type (which must not
        override what the arguments already inferred).

        Silently stops on structural shape mismatches (the assignability check
        will report the error).
        """
        if isinstance(template, TypeVarType):
            p = template.name
            if p in subst:
                if challenge and subst[p] != concrete:
                    raise AglTypeError(
                        f"Inconsistent type argument: '{p}' was inferred as "
                        f"'{subst[p]!r}' from one argument but '{concrete!r}' from another.",
                        span=span,
                    )
            else:
                subst[p] = concrete
            return
        if isinstance(template, ListType) and isinstance(concrete, ListType):
            self._match(template.elem, concrete.elem, subst, span=span, challenge=challenge)
        elif isinstance(template, DictType) and isinstance(concrete, DictType):
            self._match(template.value, concrete.value, subst, span=span, challenge=challenge)
        elif isinstance(template, FunctionType) and isinstance(concrete, FunctionType):
            if len(template.params) == len(concrete.params):
                for tp, cp in zip(template.params, concrete.params):
                    self._match(tp, cp, subst, span=span, challenge=challenge)
                self._match(template.result, concrete.result, subst, span=span, challenge=challenge)
        elif (
            isinstance(template, (RecordType, EnumType))
            and isinstance(concrete, (RecordType, EnumType))
            and type(template) is type(concrete)
            and template.name == concrete.name
            and len(template.type_args) == len(concrete.type_args)
        ):
            for ta, ca in zip(template.type_args, concrete.type_args):
                self._match(ta, ca, subst, span=span, challenge=challenge)
        # Shape mismatch, primitive mismatch, or nominal mismatch: stop (best-effort).

    def _infer_arg(
        self,
        template: Type,
        arg_expr: Expr,
        subst: dict[str, Type],
        hint: Mapping[str, Type],
        *,
        span: SourceSpan,
    ) -> None:
        """Check one argument against a (possibly type-var) parameter/field *template*
        and unify the result into *subst*.

        When the template, after applying the already-solved bindings, is fully
        concrete it is passed to ``_check_expr`` as the expected type so that
        annotation-requiring literals (notably ``[]``) can be resolved.  *hint*
        supplies advisory bindings derived from the expected result type — used
        only to make the expected type concrete, never to seed ``subst`` (which
        stays authoritative, driven by the checked argument type).
        """
        partially = substitute(template, {**hint, **subst})
        expected = None if contains_type_var(partially) else partially
        arg_type = self._check_expr(arg_expr, expected=expected)
        self._match(template, arg_type, subst, span=span)

    def _require_all_solved(
        self,
        type_params: tuple[str, ...],
        subst: Mapping[str, Type],
        *,
        span: SourceSpan,
        message_for: Callable[[str], str],
    ) -> None:
        """Raise ``AglTypeError`` for the first type parameter left unsolved after inference."""
        for p in type_params:
            if p not in subst:
                raise AglTypeError(message_for(p), span=span)

    def _result_hint(
        self, result_template: Type, expected: Type | None, *, span: SourceSpan
    ) -> dict[str, Type]:
        """Advisory type-arg bindings derived from an expected result type.

        Matched non-challenging against *result_template* so a surrounding
        annotation (e.g. ``let b: Box[int] = …``) can give an argument literal a
        concrete expected type before the arguments themselves are checked.
        Empty when *expected* is absent or its shape does not match.
        """
        hint: dict[str, Type] = {}
        if expected is not None:
            self._match(result_template, expected, hint, span=span, challenge=False)
        return hint

    # --- shared call-argument checker ---

    @staticmethod
    def _bind_call_args(
        params: tuple[ParamSpec, ...],
        node: Call,
        func_name: str,
    ) -> tuple[Expr | None, ...]:
        """Bind a call's positional/named args against *params* (declaration order).

        Shared by the concrete check (``_check_call_args``) and the generic
        inference path, so a bare-name shorthand in named-only territory is always
        matched to its parameter by NAME rather than by raw positional index.
        Raises ``AglTypeError`` on any binding violation.
        """
        return bind_call_args(
            params,
            node.args,
            node.named_args,
            call_span=node.span,
            context_desc=f"call to '{func_name}'",
        )

    def _check_call_args(
        self,
        params: tuple[ParamSpec, ...],
        node: Call,
        func_name: str,
    ) -> None:
        """Check positional and named arguments against a concrete parameter list.

        Delegates to ``bind_arguments`` for arity / duplicate / unknown / missing /
        positional-kind checks, then type-checks each bound expression against its
        parameter type.  Raises ``AglTypeError`` on any violation.

        The caller is responsible for re-checking positional arg expressions during
        type-variable inference before calling this helper with the substituted params.
        """
        binding = self._bind_call_args(params, node, func_name)
        # Record the binding for the lowerer (binding is type-independent, so the
        # generic path's pre- and post-substitution calls produce the same tuple).
        self._function_call_bindings[node.node_id] = binding

        # Type-check each bound expression.  A ``None`` binding means "use default";
        # the default's type was checked at definition time, so skip it here.
        for spec, bound_expr in zip(params, binding):
            if bound_expr is None:
                continue
            at = self._check_expr(bound_expr, expected=spec.type)
            self._assert_assignable(at, spec.type, bound_expr.span)

    # --- generic declared-name call ---

    def _check_generic_declared_call(
        self,
        node: Call,
        func_name: str,
        sig: FunctionSignature,
        *,
        expected: Type | None,
    ) -> Type:
        """Check a call to a generic (parametric) declared function."""
        # --- Explicit type argument path ---
        if node.type_args:
            if len(node.type_args) != len(sig.type_params):
                raise AglTypeError(
                    f"'{func_name}' requires {len(sig.type_params)} type argument(s), "
                    f"but {len(node.type_args)} were supplied.",
                    span=node.span,
                )
            subst: dict[str, Type] = {}
            for p, ta in zip(sig.type_params, node.type_args):
                resolved_arg = self._env.resolve_type_expr(
                    ta, span=node.span, type_vars=self._current_type_vars
                )
                subst[p] = resolved_arg
        else:
            # --- Inference path ---
            subst = {}
            # A hint from the expected result type lets annotation-requiring
            # literals in argument position (e.g. an empty list) resolve.
            hint = self._result_hint(sig.result, expected, span=node.span)
            # Use bind_arguments to get the per-param binding in declaration order.
            # This ensures that bare-name shorthands in named-only territory are
            # matched to their param by NAME rather than by raw positional index.
            # Any arity/unknown/missing error raised here propagates immediately —
            # it is the same error _check_call_args (called below) would raise.
            inf_binding = self._bind_call_args(sig.params, node, func_name)
            # Infer type variables from each bound expression against its param's
            # pre-substitution type (in declaration order).
            for spec, bound_expr in zip(sig.params, inf_binding):
                if bound_expr is not None:
                    self._infer_arg(spec.type, bound_expr, subst, hint, span=bound_expr.span)
            # Try to fill remaining unsolved vars from expected result type.
            if expected is not None:
                self._match(sig.result, expected, subst, span=node.span, challenge=False)
            # Verify all type params were inferred.
            self._require_all_solved(
                sig.type_params,
                subst,
                span=node.span,
                message_for=lambda p: (
                    f"Cannot infer type argument '{p}' for call to '{func_name}'; "
                    f"supply it explicitly via '{func_name}::[…]'."
                ),
            )

        # Substitute to get the concrete parameter types while preserving kind/default.
        sub_params = tuple(
            ParamSpec(
                name=p.name,
                type=substitute(p.type, subst),
                kind=p.kind,
                has_default=p.has_default,
            )
            for p in sig.params
        )
        sub_result = substitute(sig.result, subst)

        # Validate arguments against the substituted parameter list.
        self._check_call_args(sub_params, node, func_name)

        return sub_result

    # --- declared-name call ---

    def _check_declared_name_call(
        self,
        node: Call,
        func_name: str,
        *,
        expected: Type | None,
        callee_node_id: int,
    ) -> Type:
        # Use the node-id-keyed lookup populated by the function-signature pre-pass
        # (graph mode) and by _preregister_funcdef (single-program mode).  Keying
        # by the callee's globally-unique decl_node_id avoids the same-name collision
        # where two modules define functions with identical names but different
        # signatures, which would cause the name-keyed table to return the wrong
        # signature for a qualified cross-module call.
        sig = self._env.get_function_signature_by_node_id(callee_node_id)
        if sig is None:
            raise AglTypeError(
                f"Cannot infer return type of function '{func_name}' before it is checked. "
                "Add a return type annotation.",
                span=node.span,
            )

        # Dispatch to the generic path when the function has type parameters.
        if sig.type_params:
            return self._check_generic_declared_call(
                node, func_name, sig, expected=expected
            )

        # Non-generic path: reject unexpected explicit type args.
        if node.type_args:
            raise AglTypeError(
                f"'{func_name}' is not a generic function and does not accept "
                f"type arguments.",
                span=node.span,
            )

        self._check_call_args(sig.params, node, func_name)
        return sig.result

    # --- value call (lambda / higher-order) ---

    def _check_value_call(self, node: Call, *, expected: Type | None) -> Type:
        if node.named_args:
            raise AglTypeError(
                "Named arguments are only allowed at declared-function call sites.",
                span=node.span,
            )
        callee_type = self._check_expr(node.callee, expected=None)
        if not isinstance(callee_type, FunctionType):
            raise AglTypeError(
                f"callee is not a function; got '{callee_type!r}'.",
                span=node.callee.span,
            )
        if len(node.args) != len(callee_type.params):
            raise AglTypeError(
                f"Arity mismatch: function expects {len(callee_type.params)} argument(s), "
                f"got {len(node.args)}.",
                span=node.span,
            )
        for arg, ptype in zip(node.args, callee_type.params):
            at = self._check_expr(arg, expected=ptype)
            self._assert_assignable(at, ptype, arg.span)
        return callee_type.result

    # --- Lambda ---

    def _check_lambda(self, node: Lambda, *, expected: Type | None) -> Type:
        # D6b: no required positional-fillable param may follow a defaulted one.
        self._check_required_after_defaulted(node.params)
        # Lambda annotations may reference the rigid type variables of an
        # enclosing generic ``def`` body (the body is checked with them in scope).
        type_vars = self._current_type_vars
        param_types: list[Type] = []
        for p in node.params:
            pt = self._env.resolve_type_expr(p.type_expr, span=p.span, type_vars=type_vars)
            param_types.append(pt)
            self._env.set_binding_type(p.node_id, pt)

        if node.return_type is not None:
            result_type = self._env.resolve_type_expr(
                node.return_type, span=node.span, type_vars=type_vars
            )
            body_type = self._check_expr(node.body, expected=result_type)
            if not isinstance(body_type, BottomType):
                self._assert_assignable(body_type, result_type, node.span)
        else:
            body_type = self._check_expr(node.body, expected=None)
            if isinstance(body_type, BottomType):
                raise AglTypeError(
                    "Cannot infer return type of lambda: body always raises.",
                    span=node.span,
                )
            result_type = body_type

        return FunctionType(params=tuple(param_types), result=result_type)

    # --- if ---

    def _check_if(self, node: If, *, expected: Type | None) -> Type:
        has_else = any(isinstance(b.cond, ElseSentinel) for b in node.branches)
        branch_types: list[Type] = []
        body_expected = expected if has_else else UnitType()
        for branch in node.branches:
            if not isinstance(branch.cond, ElseSentinel):
                cond_type = self._check_expr(branch.cond, expected=None)
                self._require_bool_condition(cond_type, branch.cond.span, "if")
            bt = self._check_expr(branch.body, expected=body_expected)
            if not has_else:
                self._assert_assignable(bt, UnitType(), branch.body.span)
            branch_types.append(bt)

        if not has_else:
            return UnitType()

        return self._unify_branch_types(branch_types, node.span, "If expression")

    # --- case ---

    def _check_case(self, node: Case, *, expected: Type | None) -> Type:
        subj_type = self._check_expr(node.subject, expected=None)
        branch_types: list[Type] = []
        for branch in node.branches:
            self._bind_pattern_types(branch.pattern, subj_type, branch)
            bt = self._check_expr(branch.body, expected=expected)
            branch_types.append(bt)
        self._warn_non_exhaustive(subj_type, [b.pattern for b in node.branches], node.span)

        return self._unify_branch_types(branch_types, node.span, "Case expression")

    # --- loop ---

    def _check_loop(self, node: Loop) -> Type:
        if node.bound is not None:
            bound_type = self._check_expr(node.bound, expected=None)
            if not isinstance(bound_type, IntType):
                raise AglTypeError(
                    f"do-loop bound must be int; got '{bound_type!r}'.",
                    span=node.bound.span,
                )
        if node.for_range_to is not None:
            # Integer-range for: for VAR in a to/downto b [by k]
            assert node.for_iter is not None
            start_type = self._check_expr(node.for_iter, expected=None)
            if not isinstance(start_type, IntType):
                raise AglTypeError(
                    f"'for' range start must be int; got '{start_type!r}'.",
                    span=node.for_iter.span,
                )
            to_type = self._check_expr(node.for_range_to, expected=None)
            if not isinstance(to_type, IntType):
                raise AglTypeError(
                    f"'for' range bound must be int; got '{to_type!r}'.",
                    span=node.for_range_to.span,
                )
            if node.for_range_by is not None:
                by_type = self._check_expr(node.for_range_by, expected=None)
                if not isinstance(by_type, IntType):
                    raise AglTypeError(
                        f"'for' range step must be int; got '{by_type!r}'.",
                        span=node.for_range_by.span,
                    )
                # Static guard: a literal step <= 0 is always wrong.
                # IntLit(0) → zero; UnaryNeg(IntLit(k)) with k >= 1 → always negative.
                by_expr = node.for_range_by
                if isinstance(by_expr, IntLit) and by_expr.value <= 0:
                    raise AglTypeError(
                        "loop step must be positive; got a literal step of "
                        f"{by_expr.value}.",
                        span=by_expr.span,
                    )
                if isinstance(by_expr, UnaryNeg) and isinstance(by_expr.operand, IntLit):
                    raise AglTypeError(
                        "loop step must be positive; got a literal negative step.",
                        span=by_expr.span,
                    )
            self._env.set_binding_type(node.node_id, IntType())
        elif node.for_iter is not None:
            # Collection for: for VAR in COLLECTION
            iter_type = self._check_expr(node.for_iter, expected=None)
            if isinstance(iter_type, ListType):
                elem_type: Type = iter_type.elem
            elif isinstance(iter_type, DictType):
                elem_type = TextType()
            elif isinstance(iter_type, TextType):
                elem_type = TextType()
            else:
                raise AglTypeError(
                    f"'for' collection must be list[T], dict[text,V], or text; "
                    f"got '{iter_type!r}'.",
                    span=node.for_iter.span,
                )
            self._env.set_binding_type(node.node_id, elem_type)
        if node.while_cond is not None:
            while_type = self._check_expr(node.while_cond, expected=None)
            self._require_bool_condition(while_type, node.while_cond.span, "while")
        self._check_expr(node.body, expected=None)
        if node.until_cond is not None:
            cond_type = self._check_expr(node.until_cond, expected=None)
            self._require_bool_condition(cond_type, node.until_cond.span, "until")
        return UnitType()

    # --- try ---

    def _check_try(self, node: Try, *, expected: Type | None) -> Type:
        body_type = self._check_expr(node.body, expected=expected)
        handler_types: list[Type] = [body_type]
        for clause in node.handlers:
            ht = self._check_catch_clause(clause, expected=expected)
            handler_types.append(ht)
        return self._unify_branch_types(handler_types, node.span, "Try expression")

    def _check_catch_clause(self, clause: CatchClause, *, expected: Type | None) -> Type:
        if clause.exc_type is None or clause.exc_type == "_":
            from agm.agl.semantics.types import EXCEPTION_BASE

            exc_type: ExceptionType = EXCEPTION_BASE
        else:
            # resolve_named_type is used instead of get_type so that open-imported
            # exception types (cross-module graph mode) are found as well.
            resolved = self._env.resolve_named_type(clause.exc_type)
            if resolved is None or not isinstance(resolved, ExceptionType):
                raise AglTypeError(
                    f"'{clause.exc_type}' is not a known exception type.",
                    span=clause.span,
                )
            exc_type = resolved
        if clause.binding is not None:
            self._env.set_binding_type(clause.node_id, exc_type)
        return self._check_expr(clause.body, expected=expected)

    # --- raise ---

    def _infer_raise(self, node: Raise) -> BottomType:
        exc_type = self._check_expr(node.exc, expected=None)
        if not isinstance(exc_type, ExceptionType):
            raise AglTypeError(
                f"'raise' requires an exception value; got '{exc_type!r}'.",
                span=node.exc.span,
            )
        return BottomType()

    # --- template ---

    def _check_template(self, node: Template) -> TextType:
        for seg in node.segments:
            if isinstance(seg, InterpSegment):
                is_nonempty_container_literal = (
                    isinstance(seg.expr, DictLit) and bool(seg.expr.entries)
                ) or (isinstance(seg.expr, ListLit) and bool(seg.expr.elements))
                if is_nonempty_container_literal:
                    assert isinstance(seg.expr, (ListLit, DictLit))
                    seg_type = self._check_template_literal(seg.expr)
                    self._node_types[seg.expr.node_id] = seg_type
                else:
                    self._check_expr(seg.expr, expected=None)
        return TextType()

    def _check_template_literal(self, expr: ListLit | DictLit) -> Type:
        """Check a non-empty container literal in a ``${ … }`` context."""
        if isinstance(expr, ListLit):
            for elem in expr.elements:
                self._check_template_literal_child(elem)
            return ListType(elem=JsonType())
        # DictLit — caller guarantees non-empty.
        seen_keys: dict[str, SourceSpan] = {}
        for entry in expr.entries:
            key = entry.key.value
            if key in seen_keys:
                raise AglTypeError(
                    f"Duplicate key '{key}' in dict literal.",
                    span=entry.span,
                )
            seen_keys[key] = entry.span
        for entry in expr.entries:
            self._check_template_literal_child(entry.value)
        return DictType(value=JsonType())

    def _check_template_literal_child(self, expr: Expr) -> Type:
        """Check a single child of a template container literal."""
        if isinstance(expr, (StringLit, IntLit, DecimalLit, BoolLit, NullLit)):
            return self._check_expr(expr, expected=JsonType())
        if isinstance(expr, ListLit):
            if not expr.elements:
                return self._check_expr(expr, expected=JsonType())
            result = self._check_template_literal(expr)
            self._node_types[expr.node_id] = result
            return result
        if isinstance(expr, DictLit):
            if not expr.entries:
                return self._check_expr(expr, expected=JsonType())
            result = self._check_template_literal(expr)
            self._node_types[expr.node_id] = result
            return result
        child_type = self._check_expr(expr, expected=None)
        self._assert_assignable(child_type, JsonType(), expr.span)
        return child_type

    # --- binary ops ---

    def _check_binary_op(self, node: BinaryOp) -> Type:
        left_type = self._check_expr(node.left, expected=None)
        right_type = self._check_expr(node.right, expected=None)
        op = node.op

        if op in (BinOp.AND, BinOp.OR):
            op_name = "and" if op is BinOp.AND else "or"
            if not isinstance(left_type, BoolType):
                raise AglTypeError(
                    f"'{op_name}' requires bool operands; left operand has type "
                    f"'{left_type!r}'.",
                    span=node.left.span,
                )
            if not isinstance(right_type, BoolType):
                raise AglTypeError(
                    f"'{op_name}' requires bool operands; right operand has type "
                    f"'{right_type!r}'.",
                    span=node.right.span,
                )
            return BoolType()

        if op in (BinOp.EQ, BinOp.NEQ):
            # D2: reject operations on bare type variables.
            if isinstance(left_type, TypeVarType):
                raise AglTypeError(
                    f"operation '=' is not permitted on a value of abstract type "
                    f"variable '{left_type.name}'.",
                    span=node.span,
                )
            if isinstance(right_type, TypeVarType):
                raise AglTypeError(
                    f"operation '=' is not permitted on a value of abstract type "
                    f"variable '{right_type.name}'.",
                    span=node.span,
                )
            if not comparable_types(left_type, right_type):
                raise AglTypeError(
                    f"Equality operands must have the same type; "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return BoolType()

        if op in (BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE):
            # D2: reject operations on bare type variables.
            if isinstance(left_type, TypeVarType):
                raise AglTypeError(
                    f"ordering operator is not permitted on a value of abstract type "
                    f"variable '{left_type.name}'.",
                    span=node.span,
                )
            if isinstance(right_type, TypeVarType):
                raise AglTypeError(
                    f"ordering operator is not permitted on a value of abstract type "
                    f"variable '{right_type.name}'.",
                    span=node.span,
                )
            numeric_pair = isinstance(left_type, (IntType, DecimalType)) and isinstance(
                right_type, (IntType, DecimalType)
            )
            text_pair = isinstance(left_type, TextType) and isinstance(right_type, TextType)
            if not (numeric_pair or text_pair):
                raise AglTypeError(
                    f"Ordering operators require both operands to be numeric "
                    f"(int or decimal) or both to be text; "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return BoolType()

        if op == BinOp.IN:
            return self._check_in_op(left_type, right_type, node.span)

        if op == BinOp.ADD:
            return self._check_add(left_type, right_type, node.span)

        if op == BinOp.SUB:
            return self._check_numeric_binop(left_type, right_type, node.span, "-")

        if op == BinOp.MUL:
            return self._check_numeric_binop(left_type, right_type, node.span, "*")

        if op == BinOp.DIV:
            # D2: reject operations on bare type variables.
            if isinstance(left_type, TypeVarType):
                raise AglTypeError(
                    f"operation '/' is not permitted on a value of abstract type variable "
                    f"'{left_type.name}'.",
                    span=node.span,
                )
            if isinstance(right_type, TypeVarType):
                raise AglTypeError(
                    f"operation '/' is not permitted on a value of abstract type variable "
                    f"'{right_type.name}'.",
                    span=node.span,
                )
            if not (
                isinstance(left_type, (IntType, DecimalType))
                and isinstance(right_type, (IntType, DecimalType))
            ):
                raise AglTypeError(
                    f"'/' requires numeric operands; "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return DecimalType()

        raise AglTypeError(  # pragma: no cover
            f"Unknown binary operator: {op!r}",
            span=node.span,
        )

    def _check_add(self, left_type: Type, right_type: Type, span: SourceSpan) -> Type:
        # D2: reject operations on bare type variables.
        if isinstance(left_type, TypeVarType):
            raise AglTypeError(
                f"operation '+' is not permitted on a value of abstract type variable "
                f"'{left_type.name}'.",
                span=span,
            )
        if isinstance(right_type, TypeVarType):
            raise AglTypeError(
                f"operation '+' is not permitted on a value of abstract type variable "
                f"'{right_type.name}'.",
                span=span,
            )
        if isinstance(left_type, TextType) and isinstance(right_type, TextType):
            return TextType()
        if isinstance(left_type, (IntType, DecimalType)) and isinstance(
            right_type, (IntType, DecimalType)
        ):
            if isinstance(left_type, IntType) and isinstance(right_type, IntType):
                return IntType()
            return DecimalType()
        raise AglTypeError(
            f"'+' requires both operands to be text or both to be numeric; "
            f"got '{left_type!r}' and '{right_type!r}'.",
            span=span,
        )

    def _check_numeric_binop(
        self, left_type: Type, right_type: Type, span: SourceSpan, op_str: str
    ) -> Type:
        # D2: reject operations on bare type variables.
        if isinstance(left_type, TypeVarType):
            raise AglTypeError(
                f"operation '{op_str}' is not permitted on a value of abstract type variable "
                f"'{left_type.name}'.",
                span=span,
            )
        if isinstance(right_type, TypeVarType):
            raise AglTypeError(
                f"operation '{op_str}' is not permitted on a value of abstract type variable "
                f"'{right_type.name}'.",
                span=span,
            )
        if not (
            isinstance(left_type, (IntType, DecimalType))
            and isinstance(right_type, (IntType, DecimalType))
        ):
            raise AglTypeError(
                f"'{op_str}' requires numeric operands; "
                f"got '{left_type!r}' and '{right_type!r}'.",
                span=span,
            )
        if isinstance(left_type, IntType) and isinstance(right_type, IntType):
            return IntType()
        return DecimalType()

    def _check_in_op(self, left_type: Type, right_type: Type, span: SourceSpan) -> Type:
        # D2: reject operations on bare type variables.
        if isinstance(left_type, TypeVarType):
            raise AglTypeError(
                f"operation 'in' is not permitted on a value of abstract type variable "
                f"'{left_type.name}'.",
                span=span,
            )
        if isinstance(right_type, TypeVarType):
            raise AglTypeError(
                f"operation 'in' is not permitted on a value of abstract type variable "
                f"'{right_type.name}'.",
                span=span,
            )
        if isinstance(left_type, TextType) and isinstance(right_type, TextType):
            return BoolType()
        if isinstance(right_type, ListType):
            if not is_assignable(left_type, right_type.elem):
                raise AglTypeError(
                    f"'in' element type mismatch: '{left_type!r}' in 'list[{right_type.elem!r}]'.",
                    span=span,
                )
            return BoolType()
        if isinstance(right_type, DictType) and isinstance(left_type, TextType):
            return BoolType()
        raise AglTypeError(
            f"'in' requires (text in text), (T in list[T]), or (text in dict); "
            f"got '{left_type!r}' in '{right_type!r}'.",
            span=span,
        )

    # --- unary ---

    def _check_unary_neg(self, node: UnaryNeg) -> Type:
        t = self._check_expr(node.operand, expected=None)
        # D2: reject operations on bare type variables.
        if isinstance(t, TypeVarType):
            raise AglTypeError(
                f"unary '-' is not permitted on a value of abstract type variable '{t.name}'.",
                span=node.span,
            )
        if not isinstance(t, (IntType, DecimalType)):
            raise AglTypeError(
                f"Unary '-' requires a numeric operand; got '{t!r}'.",
                span=node.span,
            )
        return t

    # --- is test ---

    def _check_is_test(self, node: IsTest) -> BoolType:
        expr_type = self._check_expr(node.expr, expected=None)
        # D2: reject operations on bare type variables.
        if isinstance(expr_type, TypeVarType):
            raise AglTypeError(
                f"an abstract type variable '{expr_type.name}' cannot be tested with 'is'.",
                span=node.span,
            )
        if not isinstance(expr_type, EnumType):
            raise AglTypeError(
                f"'is' / 'is not' requires an enum-typed left-hand side; "
                f"got '{expr_type!r}'.",
                span=node.span,
            )
        if node.qualifier is not None:
            self._check_variant_qualifier(node.qualifier, expr_type, node.span)
        if node.variant not in expr_type.variants:
            raise AglTypeError(
                f"Variant '{node.variant}' does not belong to enum '{expr_type.name}'.",
                span=node.span,
            )
        return BoolType()

    def _check_variant_qualifier(
        self, qualifier: str, enum_type: EnumType, span: SourceSpan
    ) -> None:
        # The qualifier names an enum type. A generic enum's bare name is not
        # resolvable as a concrete type (bare generic names are rejected), so
        # resolve it through the generic-template namespace; its declared name
        # is the qualifier itself. ``enum_type`` is the scrutinee's already
        # instantiated type, whose ``.name`` is the bare enum name.
        gdef = self._env.get_generic_type(qualifier)
        type_mismatch: bool
        if gdef is not None:
            resolved_name = qualifier if gdef.kind == "enum" else None
            type_mismatch = resolved_name != enum_type.name
        else:
            resolved = self._env.resolve_named_type(qualifier)
            resolved_name = resolved.name if isinstance(resolved, EnumType) else None
            # Compare by full identity (name + module_id) so foo::Color ≠ bar::Color.
            type_mismatch = resolved != enum_type
        if resolved_name is None:
            raise AglTypeError(
                f"'{qualifier}' is not a known enum type.",
                span=span,
            )
        if type_mismatch:
            raise AglTypeError(
                f"Qualifier '{qualifier}' resolves to enum '{resolved_name}', "
                f"but the value has enum type '{enum_type.name}'.",
                span=span,
            )

    def _check_module_qualified_variant(
        self,
        module_qualifier: object,  # Qualifier
        enum_name: str,
        enum_type: EnumType,
        span: SourceSpan,
    ) -> None:
        """Validate a module-qualified enum-type qualifier, e.g. ``mylib::Color``."""
        from agm.agl.syntax.types import NameT, Qualifier

        assert isinstance(module_qualifier, Qualifier)
        fake_name_t = NameT(
            name=enum_name,
            span=span,
            node_id=-1,
            module_qualifier=module_qualifier,
        )
        try:
            resolved = self._env.resolve_type_expr(fake_name_t, span=span)
        except AglTypeError as exc:
            raise AglTypeError(
                f"'{'.'.join(module_qualifier.segments)}::{enum_name}' "
                "is not a known enum type.",
                span=span,
            ) from exc
        if not isinstance(resolved, EnumType):
            raise AglTypeError(
                f"'{'.'.join(module_qualifier.segments)}::{enum_name}' "
                "is not a known enum type.",
                span=span,
            )
        if resolved != enum_type:
            raise AglTypeError(
                f"Qualifier '{'.'.join(module_qualifier.segments)}::{enum_name}' "
                f"resolves to enum '{resolved.name}', "
                f"but the value has enum type '{enum_type.name}'.",
                span=span,
            )

    # --- field access ---

    def _check_field_access(self, node: FieldAccess, expected: Type | None = None) -> Type:
        # Bare qualified constructor reference → value-position construction.
        if node.node_id in self._resolved.qualified_constructor_refs:
            owner_name, variant, owner_module_id = (
                self._resolved.qualified_constructor_refs[node.node_id]
            )
            return self._constructors.check_qualified_constructor_as_value(
                owner_name=owner_name, variant=variant, owner_module_id=owner_module_id,
                span=node.span, expected=expected,
            )
        obj_type = self._check_expr(node.obj, expected=None)
        # D2: reject operations on bare type variables.
        if isinstance(obj_type, TypeVarType):
            raise AglTypeError(
                f"a value of type variable '{obj_type.name}' has no fields.",
                span=node.span,
            )
        if isinstance(obj_type, ExceptionType):
            if node.field not in obj_type.fields:
                raise AglTypeError(
                    f"Exception type '{obj_type.name}' has no field '{node.field}'.",
                    span=node.span,
                )
            return obj_type.fields[node.field]
        if isinstance(obj_type, RecordType):
            if node.field not in obj_type.fields:
                raise AglTypeError(
                    f"Record '{obj_type.name}' has no field '{node.field}'.",
                    span=node.span,
                )
            return obj_type.fields[node.field]
        raise AglTypeError(
            f"Field access requires a record or exception value; got '{obj_type!r}'.",
            span=node.span,
        )

    # --- index access ---

    def _check_index_access(self, node: _IndexLike) -> Type:
        obj_type = self._check_expr(node.obj, expected=None)
        return self._check_index_operand(obj_type, node.index, span=node.span)

    def _check_index_operand(
        self,
        obj_type: Type,
        index: Expr,
        *,
        span: SourceSpan,
    ) -> Type:
        # D2: reject operations on bare type variables.
        if isinstance(obj_type, TypeVarType):
            raise AglTypeError(
                f"a value of type variable '{obj_type.name}' is not indexable.",
                span=span,
            )
        if isinstance(obj_type, ListType):
            index_type = self._check_expr(index, expected=IntType())
            self._assert_assignable(index_type, IntType(), index.span)
            return obj_type.elem

        if isinstance(obj_type, DictType):
            index_type = self._check_expr(index, expected=TextType())
            self._assert_assignable(index_type, TextType(), index.span)
            return obj_type.value

        raise AglTypeError(
            f"indexing requires a list or dict; got '{obj_type!r}'.",
            span=span,
        )


    # --- list / dict literals ---

    def _expected_elem_type(self, expected: Type | None) -> Type | None:
        if isinstance(expected, ListType):
            return expected.elem
        if isinstance(expected, JsonType):
            return JsonType()
        return None

    def _expected_value_type(self, expected: Type | None) -> Type | None:
        if isinstance(expected, DictType):
            return expected.value
        if isinstance(expected, JsonType):
            return JsonType()
        return None

    def _check_list_lit(self, node: ListLit, *, expected: Type | None) -> Type:
        elem_expected = self._expected_elem_type(expected)
        if not node.elements:
            if elem_expected is None:
                raise AglTypeError(
                    "Empty list literal requires a type annotation "
                    "(e.g. 'let xs: list[text] = []').",
                    span=node.span,
                )
            return ListType(elem=elem_expected)
        if elem_expected is not None:
            for elem in node.elements:
                et = self._check_expr(elem, expected=elem_expected)
                self._assert_assignable(et, elem_expected, elem.span)
            return ListType(elem=elem_expected)
        unified = self._unify_elements(node.elements, kind="List", span=node.span)
        return ListType(elem=unified)

    def _check_dict_lit(self, node: DictLit, *, expected: Type | None) -> Type:
        seen_keys: dict[str, SourceSpan] = {}
        for entry in node.entries:
            key = entry.key.value
            if key in seen_keys:
                raise AglTypeError(
                    f"Duplicate key '{key}' in dict literal.",
                    span=entry.span,
                )
            seen_keys[key] = entry.span
        val_expected = self._expected_value_type(expected)
        if not node.entries:
            if val_expected is None:
                raise AglTypeError(
                    "Empty dict literal requires a type annotation.",
                    span=node.span,
                )
            return DictType(value=val_expected)
        if val_expected is not None:
            for entry in node.entries:
                et = self._check_expr(entry.value, expected=val_expected)
                self._assert_assignable(et, val_expected, entry.span)
            return DictType(value=val_expected)
        unified = self._unify_elements(
            [entry.value for entry in node.entries], kind="Dict", span=node.span
        )
        return DictType(value=unified)

    def _unify_elements(
        self, elements: Sequence[Expr], *, kind: str, span: SourceSpan
    ) -> Type:
        """Unify literal element types with int → decimal widening."""
        types = [self._check_expr(e, expected=None) for e in elements]
        unified = types[0]
        for t in types[1:]:
            if is_assignable(unified, t):
                unified = t
            elif not is_assignable(t, unified):
                raise AglTypeError(
                    f"{kind} literal elements have inconsistent types: "
                    f"'{unified!r}' and '{t!r}'.",
                    span=span,
                )
        return unified

    # ------------------------------------------------------------------
    # Pattern binding helpers
    # ------------------------------------------------------------------

    def _bind_pattern_types(self, pattern: object, subj_type: Type, owner: object) -> None:
        """Record binding types for variables introduced by *pattern*."""
        if isinstance(pattern, WildcardPattern):
            pass
        elif isinstance(pattern, LiteralPattern):
            lit_type = self._check_expr(pattern.literal, expected=None)
            if not comparable_types(lit_type, subj_type):
                raise AglTypeError(
                    f"Literal pattern of type '{lit_type!r}' is incompatible with "
                    f"scrutinee of type '{subj_type!r}'.",
                    span=pattern.span,
                )
        elif isinstance(pattern, VarPattern):
            if pattern.node_id in self._resolved.bare_variant_patterns:
                # A bare name denoting an in-scope constructor: a nullary
                # variant pattern, not a binder.  Validate it statically.
                self._check_bare_variant_pattern(pattern, subj_type)
            else:
                self._env.set_binding_type(pattern.node_id, subj_type)
        else:
            assert isinstance(pattern, ConstructorPattern), (
                f"Unexpected pattern kind: {type(pattern).__name__}"
            )
            if not isinstance(subj_type, EnumType):
                raise AglTypeError(
                    f"Cannot match constructor pattern '{pattern.name}' against "
                    f"non-enum type '{subj_type!r}'.",
                    span=pattern.span,
                )
            if pattern.qualifier is not None:
                if pattern.module_qualifier is not None:
                    # Module-qualified variant qualifier, e.g. ``mylib::Color.Red``.
                    self._check_module_qualified_variant(
                        pattern.module_qualifier, pattern.qualifier, subj_type, pattern.span
                    )
                else:
                    self._check_variant_qualifier(pattern.qualifier, subj_type, pattern.span)
            variant_name = pattern.name
            if variant_name not in subj_type.variants:
                raise AglTypeError(
                    f"Variant '{variant_name}' does not belong to enum "
                    f"'{subj_type.name}'.",
                    span=pattern.span,
                )
            vfields = subj_type.variants[variant_name]

            # Retrieve the registered field kinds for this variant constructor.
            field_kinds = self._env.get_constructor_field_kinds(
                subj_type.name, variant_name, module_id=subj_type.module_id
            )
            assert field_kinds is not None, (
                f"field kinds not registered for {subj_type.name}.{variant_name}"
            )

            # Route through the shared binder (handles zones, duplicates, unknowns).
            binding = bind_pattern_args(
                field_kinds,
                pattern.positional,
                pattern.named,
                call_span=pattern.span,
                context_desc=f"pattern for variant '{variant_name}'",
            )

            # Record the side-table entry and recursively check bound sub-patterns.
            bound_pairs: list[tuple[str, Pattern]] = []
            for (fname, _), bound_pat in zip(field_kinds, binding):
                if bound_pat is not None:
                    bound_pairs.append((fname, bound_pat))
                    field_type = vfields[fname]
                    self._bind_pattern_types(bound_pat, field_type, pattern)
            self._constructor_pattern_bindings[pattern.node_id] = tuple(bound_pairs)

    def _check_bare_variant_pattern(self, pattern: VarPattern, subj_type: Type) -> None:
        """Validate a bare-name nullary variant pattern (``| Red =>``).

        The name has already been classified as an in-scope constructor by the
        resolver; here it must additionally be a *nullary* variant of the
        scrutinee's enum.  A field-bearing variant requires the explicit call
        form so the discarded payload is acknowledged.
        """
        if not isinstance(subj_type, EnumType):
            raise AglTypeError(
                f"Cannot match constructor pattern '{pattern.name}' against "
                f"non-enum type '{subj_type!r}'.",
                span=pattern.span,
            )
        if pattern.name not in subj_type.variants:
            raise AglTypeError(
                f"Variant '{pattern.name}' does not belong to enum '{subj_type.name}'.",
                span=pattern.span,
            )
        if subj_type.variants[pattern.name]:
            raise AglTypeError(
                f"'{pattern.name}' is a variant of enum '{subj_type.name}' that has "
                f"fields, so a bare name cannot match it. Write '{pattern.name}(...)' "
                f"(or '{pattern.name}(_)') to match and ignore the payload, or "
                f"destructure the fields explicitly.",
                span=pattern.span,
            )

    def _warn_non_exhaustive(
        self, subj_type: Type, patterns: list[Pattern], span: SourceSpan
    ) -> None:
        """Emit a warning when an enum ``case`` leaves some variants uncovered."""
        if not isinstance(subj_type, EnumType):
            return
        covered: set[str] = set()
        for pattern in patterns:
            if isinstance(pattern, WildcardPattern):
                return
            if isinstance(pattern, VarPattern):
                if pattern.node_id not in self._resolved.bare_variant_patterns:
                    return  # a genuine binder is a catch-all
                covered.add(pattern.name)
                continue
            assert isinstance(pattern, ConstructorPattern), (
                f"unexpected pattern kind on enum scrutinee: {type(pattern).__name__}"
            )
            covered.add(pattern.name)
        missing = [name for name in subj_type.variants if name not in covered]
        if not missing:
            return
        self._warnings.append(
            Diagnostic(
                message=(
                    f"Non-exhaustive case on enum '{subj_type.name}': missing "
                    f"variant(s) {', '.join(missing)}. An unmatched value raises "
                    f"MatchError at runtime."
                ),
                line=span.start_line,
                column=span.start_col,
                end_line=span.end_line,
                end_column=span.end_col,
                severity="warning",
            )
        )

    # ------------------------------------------------------------------
    # Branch unification
    # ------------------------------------------------------------------

    def _unify_branch_types(
        self,
        branch_types: list[Type],
        span: SourceSpan,
        construct: str,
    ) -> Type:
        """Unify branch types with int→decimal widening and BottomType filtering."""
        non_bottom = [t for t in branch_types if not isinstance(t, BottomType)]
        if not non_bottom:
            return BottomType()
        result_type = non_bottom[0]
        for bt in non_bottom[1:]:
            if bt == result_type:
                continue
            if isinstance(result_type, IntType) and isinstance(bt, DecimalType):
                result_type = DecimalType()
            elif isinstance(result_type, DecimalType) and isinstance(bt, IntType):
                pass
            else:
                raise AglTypeError(
                    f"{construct} branches have incompatible types: "
                    f"'{result_type!r}' and '{bt!r}'.",
                    span=span,
                )
        return result_type

    # ------------------------------------------------------------------
    # Bool condition helper
    # ------------------------------------------------------------------

    def _require_bool_condition(self, cond_type: Type, span: SourceSpan, kw: str) -> None:
        if not isinstance(cond_type, BoolType):
            raise AglTypeError(
                f"'{kw}' condition must be bool; got '{cond_type!r}'.",
                span=span,
            )

    # ------------------------------------------------------------------
    # Assignability helpers
    # ------------------------------------------------------------------

    def _assert_assignable(self, value_type: Type, target_type: Type, span: SourceSpan) -> None:
        if is_assignable(value_type, target_type):
            return
        raise AglTypeError(
            f"Type mismatch: expected '{target_type!r}', got '{value_type!r}'.",
            span=span,
        )

    def _resolve_annotation(self, ann: TypeExpr | None, span: SourceSpan) -> Type | None:
        if ann is None:
            return None
        # Annotations inside a generic ``def`` body (e.g. ``let x: T = …``) may
        # reference the def's rigid type variables.
        return self._env.resolve_type_expr(ann, span=span, type_vars=self._current_type_vars)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def result(self, resolved: ResolvedProgram) -> CheckedProgram:
        return CheckedProgram(
            resolved=resolved,
            node_types=self._node_types,
            contract_specs=self._contract_specs,
            call_sites=tuple(self._call_sites),
            warnings=tuple(self._warnings),
            type_env=self._env,
            function_signatures=self._env.all_function_signatures(),
            cast_specs=self._cast_specs,
            argument_bindings=ArgumentBindings(
                function_calls=self._function_call_bindings,
                constructor_calls=self._constructor_call_bindings,
                constructor_patterns=self._constructor_pattern_bindings,
            ),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check(
    resolved: ResolvedProgram,
    capabilities: HostCapabilities,
    *,
    seed_env: TypeEnvironment | None = None,
) -> CheckedProgram:
    """Run the full type-checking pass.

    Parameters
    ----------
    resolved:
        Output of the scope resolution pass.
    capabilities:
        Immutable host capability catalog (agents, codecs, renderers).
    seed_env:
        When given, the working ``TypeEnvironment`` starts pre-populated with
        the seed's user-declared types and prior binding types.

    Returns
    -------
    CheckedProgram
        The annotated program with type side tables and contract specs.

    Raises
    ------
    AglTypeError
        On the first static type violation (first-error abort).
    """
    env = TypeEnvironment()
    if seed_env is not None:
        env.seed_from(seed_env)
    program = resolved.program

    builder = _TypeBuilder(env)
    builder.collect(program)

    checker = _Checker(env=env, resolved=resolved, capabilities=capabilities)
    checker.check_program(program)

    return checker.result(resolved)
