"""Type-checking pass for AgL.

``check_module(resolved, capabilities)`` performs a bidirectional type pass over the
``ModuleResolution``, using the ``HostCapabilities`` to validate codec and
renderer names, and returns a ``CheckedModule``.

Rules implemented
-----------------
1.  Type declaration validation: duplicate names, unknown referenced types,
    uninhabitable recursive records/enums, alias cycles, and built-in-name shadowing.
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
14. ``case`` — validates subject, pattern, and branch types.
15. ``loop`` — yields ``unit``; until/while conditions must be bool; bound must be int.
16. ``try/catch`` — body and handler types must unify.
17. ``raise``/``return`` — yield ``BottomType`` (bottom, assignable to any target).
18. Assignability: ``int`` widens to ``decimal``; ``json``
    accepts any JSON-shaped value.  Bottom type is assignable to any target.
19. Duplicate constructor argument names, duplicate dict keys, and constructor
    well-formedness checks.

The checker raises ``AglTypeError`` on the first error (first-error abort).
"""

from __future__ import annotations

import keyword
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Literal, TypeGuard, assert_never, cast

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    BinderKind,
    BindingRef,
    BuiltinKind,
    ConstructorRef,
    ModuleResolution,
)
from agm.agl.semantics.type_table import TypeTable, comparable_types
from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    BottomType,
    CastKind,
    CastSpec,
    DecimalType,
    DictType,
    EnumOwnerForm,
    EnumOwnerFormKind,
    EnumType,
    ExceptionType,
    FunctionType,
    InferenceVarType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
    cast_classification,
    contains_inference_var,
    is_assignable,
    substitute,
)
from agm.agl.syntax.nodes import (
    AgentDecl,
    AsPattern,
    AssignStmt,
    BinaryOp,
    BinOp,
    Block,
    BoolLit,
    Break,
    BuiltinVarDecl,
    Call,
    Case,
    Cast,
    CatchClause,
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
    Placeholder,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    Return,
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
from agm.agl.syntax.types import Qualifier, TypeExpr
from agm.agl.typecheck.arguments import bind_call_args, bind_pattern_args
from agm.agl.typecheck.builder import _BUILTIN_TYPE_NAMES as _BUILTIN_TYPE_NAMES
from agm.agl.typecheck.builder import _TypeBuilder
from agm.agl.typecheck.builtins import (
    BuiltinCallChecker,
    PendingBuiltinObligation,
)
from agm.agl.typecheck.constructors import ConstructorChecker
from agm.agl.typecheck.env import (
    AglTypeError,
    ArgumentBindings,
    CallSiteRecord,
    CheckedModule,
    FunctionSignature,
    OutputContractSpec,
    ParamSpec,
    PartialCallSpec,
    TypeEnvironment,
    assert_checked_module_closed,
)
from agm.agl.typecheck.inference import (
    ConstraintRole,
    InferenceEngine,
    InferenceError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Built-in function names that user-defined defs may not shadow. Derived from
# the single source of truth in ``scope.symbols`` so the two never drift.
_BUILTIN_FUNC_NAMES: frozenset[str] = frozenset(BUILTIN_CALL_NAMES)


@dataclass(frozen=True, slots=True)
class _ExternTarget:
    """Extern identity carried by first-class function provenance."""

    name: str
    result_type: Type
    decl_node_id: int
    module_id: ModuleId


_ExternTargets = tuple[_ExternTarget, ...]


@dataclass(frozen=True, slots=True)
class PendingExternCallObligation:
    """Syntax-derived extern inventory metadata awaiting region finalization."""

    node_id: int
    callee: str
    target_type: Type
    span: SourceSpan


_PendingFinalization = PendingBuiltinObligation | PendingExternCallObligation


@dataclass(slots=True)
class _InferenceRegion:
    """Provisional checker state owned by one independently checked expression."""

    engine: InferenceEngine
    node_types: dict[int, Type]
    function_call_param_types: dict[int, tuple[Type, ...]]
    finalization_obligations: list[_PendingFinalization]
    added_side_table_keys: dict[str, set[int]] = field(default_factory=dict)
    call_sites_start: int = 0
    warnings_start: int = 0
    return_target_lengths: tuple[int, ...] = ()
    reconciled_resolution_originals: dict[int, BindingRef] = field(default_factory=dict)
    reconciled_constructor_ref_originals: dict[int, ConstructorRef | None] = field(
        default_factory=dict
    )


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
            return FunctionSignature(params=(_std_param("value", TextType()),), result=JsonType())
        case "ask":
            return FunctionSignature(
                params=(
                    _std_param("prompt", TextType()),
                    _std_param("agent", AgentType(), has_default=True),
                    _std_param("format", TextType(), has_default=True),
                    _std_param("strict_json", BoolType(), has_default=True),
                    _std_param(
                        "on_parse_error",
                        EnumType(name="ParsePolicy"),
                        has_default=True,
                    ),
                ),
                result=t,
                type_params=("T",),
            )
        case "ask-request":
            return FunctionSignature(
                params=(_std_param("prompt", TextType()),),
                result=RecordType(name="AgentRequest"),
            )
        case "exec":
            return FunctionSignature(
                params=(_std_param("command", TextType()),),
                result=RecordType(name="ExecResult"),
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
            FunctionSignature(params=(_std_param("prompt", TextType()),), result=TextType()),
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


def _validate_extern_name(name: str, span: SourceSpan) -> None:
    """Reject an extern name that is not a valid, non-reserved Python identifier.

    The companion Python module must define a function with exactly this name,
    so the name must be usable as a Python ``def`` name: a valid identifier
    and not a hard Python keyword.  Python *soft* keywords
    (``match``, ``type``, …) remain acceptable since they are valid ``def``
    names in Python itself.
    """
    if not name.isidentifier() or keyword.iskeyword(name):
        raise AglTypeError(
            f"extern function name '{name}' must be a valid Python identifier "
            "and not a Python keyword, because the companion module must "
            "define a Python function with exactly this name.",
            span=span,
        )


def _contains_banned_extern_type(
    t: Type, type_table: TypeTable, _seen: frozenset[Type] = frozenset()
) -> bool:
    """Return ``True`` if *t* contains a function or agent type anywhere.

    The FFI is a pure data boundary: function and agent values can never cross
    it, so they are static errors anywhere in an extern's parameter or return
    types, including nested inside ``list``/``dict``/record/enum
    instantiations.  Type variables are permitted at any depth — dynamic
    sealing keeps values at those positions opaque.  Record/enum field and
    variant shapes are resolved through *type_table*; *_seen* tracks the
    nominal instantiations already on the current path so a recursive type
    (e.g. a self-referential record) is examined once rather than forever.
    """
    match t:
        case FunctionType() | AgentType():
            return True
        case ListType():
            return _contains_banned_extern_type(t.elem, type_table, _seen)
        case DictType():
            return _contains_banned_extern_type(t.value, type_table, _seen)
        case RecordType():
            if t in _seen:
                return False
            seen = _seen | {t}
            return any(
                _contains_banned_extern_type(ta, type_table, seen) for ta in t.type_args
            ) or any(
                _contains_banned_extern_type(ft, type_table, seen)
                for ft in type_table.record_fields(t).values()
            )
        case EnumType():
            if t in _seen:
                return False
            seen = _seen | {t}
            return any(
                _contains_banned_extern_type(ta, type_table, seen) for ta in t.type_args
            ) or any(
                _contains_banned_extern_type(ft, type_table, seen)
                for vfields in type_table.enum_variants(t).values()
                for ft in vfields.values()
            )
        case ExceptionType():
            if t in _seen:
                return False
            seen = _seen | {t}
            return any(
                _contains_banned_extern_type(ft, type_table, seen)
                for ft in type_table.exception_fields(t).values()
            )
        case (
            TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | UnitType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return False
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------


class _Checker:
    """Stateful type-checking visitor for AgL.

    Walks the program's items in order, maintaining a binding-type lookup
    table (``node_id → Type``) populated for declarations and inline inference.

    Uses expected-type propagation: ``_check_expr(node, expected)`` propagates
    an outer type context into the expression.  When ``expected`` is ``None``
    the expression is inferred bottom-up.
    """

    def __init__(
        self,
        env: TypeEnvironment,
        resolved: ModuleResolution,
        capabilities: HostCapabilities,
    ) -> None:
        self._env = env
        self._resolved = replace(
            resolved,
            resolution=dict(resolved.resolution),
            constructor_refs=dict(resolved.constructor_refs),
            qualified_constructor_refs=dict(resolved.qualified_constructor_refs),
        )
        self._caps = capabilities
        self._node_types: dict[int, Type] = {}
        self._inference_region: _InferenceRegion | None = None
        self._contract_specs: dict[int, OutputContractSpec] = {}
        self._call_sites: list[CallSiteRecord] = []
        self._warnings: list[Diagnostic] = []
        # Type variables currently in scope (non-empty inside a generic def body).
        self._current_type_vars: frozenset[str] = frozenset()
        self._cast_specs: dict[int, CastSpec] = {}
        # Argument bindings computed during the check, reused by the lowerer so it
        # never re-binds.  Keyed by Call/Pattern node_id (see ``ArgumentBindings``).
        self._function_call_bindings: dict[int, tuple[Expr | None, ...]] = {}
        self._function_call_param_types: dict[int, tuple[Type, ...]] = {}
        self._constructor_call_bindings: dict[int, dict[str, Expr]] = {}
        self._constructor_pattern_bindings: dict[int, tuple[tuple[str, Pattern], ...]] = {}
        # ``None`` means a final binder; a ConstructorRef means a final bare
        # constructor pattern. Scope only supplied provisional nested binders.
        self._pattern_classifications: dict[int, ConstructorRef | None] = {}
        self._partial_calls: dict[int, PartialCallSpec] = {}
        # Extern provenance for first-class function values.  A value can name
        # one or more externs (e.g. through a branch); when such a function
        # value is actually called, dry-run inventory records that call site.
        self._extern_expr_targets: dict[int, _ExternTargets] = {}
        self._extern_binding_targets: dict[int, _ExternTargets] = {}
        self._builtins = BuiltinCallChecker(self)
        self._constructors = ConstructorChecker(self)
        # Function return contexts. The top entry is either an annotated expected
        # result type or None for inference; the return collector records operand
        # types in inference mode.
        self._return_expected_stack: list[Type | None] = []
        self._return_collected_stack: list[list[Type]] = []
        self._return_extern_targets_stack: list[list[_ExternTarget]] = []

    # ------------------------------------------------------------------
    # required-after-defaulted check (shared by def and lambda)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required_after_defaulted(params: Sequence[Param]) -> None:
        """no required positional-fillable param may follow a defaulted one.

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
        if node.is_extern:
            self._validate_extern_signature(node, sig)
            self._env.register_extern_node_id(node.node_id)
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
        if node.is_extern:
            _validate_extern_name(node.name, node.span)
            if node.return_type is None:
                raise AglTypeError(
                    f"Extern function '{node.name}' must declare a return type.",
                    span=node.span,
                )
        # no required positional-fillable param may follow a defaulted one.
        self._check_required_after_defaulted(node.params)

    def _validate_extern_signature(self, node: FuncDef, sig: FunctionSignature) -> None:
        """Reject types that cannot cross the Python boundary in an extern's signature.

        Two kinds are rejected: a function or agent type anywhere (opaque values
        that can never marshal across the FFI), and a type with no finite schema
        (its recursive instantiations never close, so its boundary schema — like
        its JSON schema — cannot be built). A finite recursive type is allowed:
        it crosses as a ``BoundaryRef`` structure.
        """
        for p, spec in zip(node.params, sig.params):
            self._reject_uncrossable_extern_type(
                spec.type,
                span=p.span,
                use="an extern parameter type",
                banned_message=(
                    f"extern function '{node.name}' parameter '{p.name}' has a "
                    "function or agent type, which cannot cross the Python boundary."
                ),
            )
        self._reject_uncrossable_extern_type(
            sig.result,
            span=node.span,
            use="an extern return type",
            banned_message=(
                f"extern function '{node.name}' has a return type containing a "
                "function or agent type, which cannot cross the Python boundary."
            ),
        )

    def _reject_uncrossable_extern_type(
        self, typ: Type, *, span: SourceSpan, use: str, banned_message: str
    ) -> None:
        """Reject one extern parameter/result type that cannot cross the boundary.

        Finite-schema is checked BEFORE the banned-type walk: a type whose
        instantiations never close (growing polymorphic recursion) has an
        infinite structure, and ``_contains_banned_extern_type`` walks that
        structure — its cycle guard only catches repeated instantiations, not
        ever-growing ones. ``no_finite_schema_message`` works at the
        declaration level and always terminates, so it rejects such a type
        first, leaving the banned-type walk only finite closures to traverse.
        """
        type_table = self._env.type_table
        message = type_table.no_finite_schema_message(typ, use=use)
        if message is not None:
            raise AglTypeError(message, span=span)
        if _contains_banned_extern_type(typ, type_table):
            raise AglTypeError(banned_message, span=span)

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

    def check_module(self, program: Program) -> None:
        """Type-check the entire program."""
        # Pre-pass: register all FuncDef signatures before any body is checked,
        # so a call may reference any top-level function regardless of
        # declaration order, including mutually recursive pairs.
        for item in program.body.items:
            if isinstance(item, FuncDef):
                self._preregister_funcdef(item)

        self._check_block(program.body, expected=None)

    # ------------------------------------------------------------------
    # Block and item dispatch
    # ------------------------------------------------------------------

    def _check_block(self, block: Block, *, expected: Type | None) -> Type:
        """Type-check every block item and return the last item's type."""
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
        if isinstance(last, Expr):
            self._set_extern_expr_targets(
                block.node_id, self._extern_targets_for_expr(last, result_type)
            )
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
        if isinstance(item, BuiltinVarDecl):
            self._check_builtin_var(item)
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
            return UnitType()  # The program module-system pass processes imports/exports.
        # --- Binders ---
        if isinstance(item, (LetDecl, VarDecl)):
            return self._check_binding(item)
        if isinstance(item, AssignStmt):
            return self._check_assign_stmt(item)
        # --- Expr ---
        return self._check_expr(item, expected=expected)

    # ------------------------------------------------------------------
    # Declaration checkers
    # ------------------------------------------------------------------

    @contextmanager
    def _return_context(self, expected: Type | None) -> Iterator[list[Type]]:
        """Push a return-target frame for the enclosing function body.

        ``expected`` is the annotated result type, or ``None`` in inference mode.
        The yielded list collects the operand types of ``return`` statements when
        inferring the result type.
        """
        collected: list[Type] = []
        self._return_expected_stack.append(expected)
        self._return_collected_stack.append(collected)
        self._return_extern_targets_stack.append([])
        try:
            yield collected
        finally:
            self._return_extern_targets_stack.pop()
            self._return_collected_stack.pop()
            self._return_expected_stack.pop()

    def _check_funcdef_body(self, node: FuncDef) -> None:
        """Check the body of a ``def`` against its registered signature.

        An ``extern def`` has no body but DOES declare real AgL default
        expressions (evaluated on the AgL side before crossing the Python
        boundary), so it still binds its params and checks its defaults —
        only the body check itself is skipped.  A ``builtin def`` skips both:
        its defaults are pinned by the hardcoded builtin signature table, not
        evaluated as ordinary AgL expressions.
        """
        # Signatures are declaration schemes, not name-keyed templates. In
        # program context an imported declaration may use the same spelling as this
        # local unannotated definition; only this declaration's node id can
        # determine whether its signature was pre-registered.
        sig = self._env.get_function_signature_by_node_id(node.node_id)
        if sig is None:
            self._infer_funcdef_signature(node)
            return
        if node.is_builtin:
            return
        assert node.is_extern or node.body is not None, f"FuncDef '{node.name}' has no body"
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
                    def_type = self._check_boundary_expr(p.default, expected=spec.type)
                    self._assert_assignable(def_type, spec.type, p.span)
            if node.is_extern:
                return
            # Check body against declared return type.
            assert node.body is not None
            with self._return_context(sig.result):
                body_type = self._check_expr(node.body, expected=sig.result)
                return_targets = self._current_return_extern_targets()
            if not isinstance(body_type, BottomType):
                self._assert_assignable(body_type, sig.result, node.span)
            targets = (
                self._merge_extern_targets(
                    self._extern_targets_for_expr(node.body, body_type), return_targets
                )
                if isinstance(sig.result, FunctionType)
                else ()
            )
            self._set_extern_binding_targets(node.node_id, targets)
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
                    def_type = self._check_boundary_expr(p.default, expected=spec.type)
                    self._assert_assignable(def_type, spec.type, p.span)
            with self._return_context(None) as collected:
                body_type = self._check_expr(node.body, expected=None)
                return_targets = self._current_return_extern_targets()
            if isinstance(body_type, BottomType) and not collected:
                raise AglTypeError(
                    f"Cannot infer return type of function '{node.name}': body always raises. "
                    "Add a return type annotation.",
                    span=node.span,
                )
            try:
                result_type = self._unify_branch_types(
                    [*collected, body_type], node.span, "Function return"
                )
            except AglTypeError as exc:
                raise AglTypeError(
                    f"Cannot infer return type of function '{node.name}': return values have "
                    "incompatible types. Add a return type annotation.",
                    span=node.span,
                    related=exc.related,
                ) from exc
            inferred_sig = FunctionSignature(
                params=sig.params, result=result_type, type_params=sig.type_params
            )
            inferred_type = FunctionType(
                params=tuple(p.type for p in sig.params), result=result_type
            )
            self._register_funcdef_signature(node, inferred_sig, inferred_type)
            targets = (
                self._merge_extern_targets(
                    self._extern_targets_for_expr(node.body, body_type), return_targets
                )
                if isinstance(result_type, FunctionType)
                else ()
            )
            self._set_extern_binding_targets(node.node_id, targets)
        finally:
            self._current_type_vars = old_type_vars

    def _check_param(self, stmt: ParamDecl) -> None:
        ann_type = (
            self._env.resolve_type_expr(stmt.annotation, span=stmt.span)
            if stmt.annotation is not None
            else None
        )
        if stmt.default is not None:
            val_type = self._check_boundary_expr(stmt.default, expected=ann_type)
            if ann_type is not None:
                self._assert_assignable(val_type, ann_type, stmt.span)
                declared_type = ann_type
            else:
                if isinstance(val_type, BottomType):
                    raise AglTypeError(
                        "Cannot infer type of param: default always raises. Add a type annotation.",
                        span=stmt.span,
                    )
                declared_type = val_type
        else:
            declared_type = ann_type if ann_type is not None else TextType()
        # A non-text param round-trips through the JSON boundary (schema +
        # decode) at lowering time (see ``type_schema.build_param_decoder``).
        # Reject both kinds of non-decodable type here rather than crashing at
        # lowering: infinite instantiation closures have no finite schema, and
        # opaque/non-data values (unit, agent, functions, exceptions, …) have no
        # JSON wire representation at all. Text params are taken verbatim.
        if not isinstance(declared_type, TextType):
            message = self._env.type_table.no_finite_schema_message(
                declared_type, use="a parameter type"
            )
            if message is not None:
                raise AglTypeError(message, span=stmt.span)
            if not self._type_is_wire_serializable(declared_type):
                raise AglTypeError(
                    f"Param type '{declared_type!r}' cannot be decoded from JSON; "
                    "use text or a JSON-serializable data type.",
                    span=stmt.span,
                )
        self._env.set_binding_type(stmt.node_id, declared_type)

    def _type_is_wire_serializable(self, typ: Type) -> bool:
        """Return whether *typ* can be decoded from a JSON/schema boundary."""
        return self._wire_type_is_serializable(typ, seen=frozenset())

    def _wire_type_is_serializable(self, typ: Type, *, seen: frozenset[Type]) -> bool:
        schema_type = self._env.type_table.canonical_schema_type(typ)
        if isinstance(schema_type, (TextType, JsonType, BoolType, IntType, DecimalType)):
            return True
        if isinstance(schema_type, ListType):
            return self._wire_type_is_serializable(schema_type.elem, seen=seen)
        if isinstance(schema_type, DictType):
            return self._wire_type_is_serializable(schema_type.value, seen=seen)
        if isinstance(schema_type, RecordType):
            if schema_type in seen:
                return True
            next_seen = seen | {schema_type}
            return all(
                self._wire_type_is_serializable(field_type, seen=next_seen)
                for field_type in self._env.type_table.record_fields(schema_type).values()
            )
        if isinstance(schema_type, EnumType):
            if schema_type in seen:
                return True
            next_seen = seen | {schema_type}
            return all(
                self._wire_type_is_serializable(field_type, seen=next_seen)
                for variant_fields in self._env.type_table.enum_variants(schema_type).values()
                for field_type in variant_fields.values()
            )
        if isinstance(
            schema_type,
            (
                ExceptionType,
                UnitType,
                AgentType,
                FunctionType,
                BottomType,
                TypeVarType,
                InferenceVarType,
            ),
        ):
            return False
        assert_never(schema_type)  # pragma: no cover

    def _check_builtin_var(self, node: BuiltinVarDecl) -> None:
        """Check a ``builtin var`` declaration against the engine-key registry.

        The name whitelist that gates every builtin declaration applies here: a
        ``builtin var`` must name a known engine key, and its declared type must
        match the key's canonical type.  The canonical engine-key type is
        recorded as the binding type.
        """
        from agm.agl.semantics.engine_keys import get_engine_key_type

        key_type = get_engine_key_type(node.name)
        if key_type is None:
            raise AglTypeError(
                f"Unknown builtin var '{node.name}'.",
                span=node.span,
            )
        declared = self._env.resolve_type_expr(node.type_ann, span=node.span, type_vars=frozenset())
        if declared != key_type:
            raise AglTypeError(
                f"builtin var '{node.name}' must have type '{key_type!r}', got '{declared!r}'.",
                span=node.span,
            )
        self._env.set_binding_type(node.node_id, key_type)

    @staticmethod
    def _binder_result(value_type: Type) -> Type:
        """A binder/assignment item propagates bottom when its value always exits."""
        return BottomType() if isinstance(value_type, BottomType) else UnitType()

    def _check_binding(self, stmt: LetDecl | VarDecl) -> Type:
        ann_type = self._resolve_annotation(stmt.type_ann, stmt.span)
        val_type = self._check_boundary_expr(stmt.value, expected=ann_type)
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
        targets = (
            self._extern_expr_targets.get(stmt.value.node_id, ())
            if isinstance(declared_type, FunctionType)
            else ()
        )
        self._set_extern_binding_targets(stmt.node_id, targets)
        return self._binder_result(val_type)

    def _check_assign_stmt(self, stmt: AssignStmt) -> Type:
        if isinstance(stmt.target, NameTarget):
            ref = self._resolved.resolution[stmt.node_id]
            if not ref.mutable:
                raise AglTypeError(
                    f"Cannot assign to '{ref.name}': assignment requires a mutable binding.",
                    span=stmt.target.span,
                )
            target_type = self._require_binding_type(ref)
            val_type = self._check_boundary_expr(stmt.value, expected=target_type)
            self._assert_assignable(val_type, target_type, stmt.span)
            targets = (
                self._extern_expr_targets.get(stmt.value.node_id, ())
                if isinstance(target_type, FunctionType)
                else ()
            )
            self._set_extern_binding_targets(ref.decl_node_id, targets)
            return self._binder_result(val_type)

        if isinstance(stmt.target, IndexTarget):
            return self._check_indexed_assign_stmt(stmt, stmt.target)

        raise AglTypeError(
            "assignment target must be a mutable variable or indexed mutable variable.",
            span=stmt.span,
        )

    def _check_indexed_assign_stmt(self, stmt: AssignStmt, target: IndexTarget) -> Type:
        ref = self._resolved.resolution[stmt.node_id]
        if not ref.mutable:
            raise AglTypeError(
                f"Cannot assign through index of '{ref.name}': "
                "indexed assignment requires a mutable 'var' binding.",
                span=target.span,
            )
        root_type = self._require_binding_type(ref)
        elem_type = self._check_index_target_type(target, root_type)
        value_type = self._check_boundary_expr(stmt.value, expected=elem_type)
        self._assert_assignable(value_type, elem_type, stmt.span)
        return self._binder_result(value_type)

    def _check_index_target_type(self, target: IndexTarget, root_type: Type) -> Type:
        container_type = self._check_index_target_container_type(target.obj, root_type)
        return self._check_index_operand(container_type, target.index, span=target.span)

    def _check_index_target_container_type(self, obj: Expr, root_type: Type) -> Type:
        if isinstance(obj, VarRef):
            return root_type
        if isinstance(obj, IndexAccess):
            container_type = self._check_index_target_container_type(obj.obj, root_type)
            indexed_type = self._check_index_operand(container_type, obj.index, span=obj.span)
            self._record_node_type(obj.node_id, indexed_type)
            return indexed_type
        raise AglTypeError(
            "indexed assignment requires a variable list or dict root.",
            span=obj.span,
        )

    # ------------------------------------------------------------------
    # Expression type inference
    # ------------------------------------------------------------------

    def _check_boundary_expr(self, expr: Expr, *, expected: Type | None) -> Type:
        """Check an initializer in its own inference region before installation."""
        outer_region = self._inference_region
        self._inference_region = None
        try:
            return self._check_expr(expr, expected=expected)
        finally:
            self._inference_region = outer_region

    def _check_expr(self, expr: Expr, *, expected: Type | None) -> Type:
        """Infer/check an expression, finalizing its owning inference region."""
        if self._inference_region is not None:
            typ = self._infer_expr(expr, expected=expected)
            self._record_node_type(expr.node_id, typ)
            return self._inference_region.engine.zonk(typ)

        region = _InferenceRegion(
            InferenceEngine(),
            {},
            {},
            [],
            {},
            len(self._call_sites),
            len(self._warnings),
            tuple(len(targets) for targets in self._return_extern_targets_stack),
        )
        # Some side tables are written while an expression is being checked,
        # while call-site metadata remains typed obligations until region close.
        # The region records only its additions, so rollback never copies state
        # accumulated by earlier independently checked expressions.
        completed = False
        self._inference_region = region
        try:
            typ = self._infer_expr(expr, expected=expected)
            self._record_node_type(expr.node_id, typ)
            if expected is not None:
                region.engine.complete_from_context(
                    typ,
                    expected,
                    region.engine.origin(
                        expr.span, role=ConstraintRole.EXPECTED_RESULT, subject="expression"
                    ),
                )
            region.engine.check_requirements()
            for obligation in region.finalization_obligations:
                if isinstance(obligation, PendingBuiltinObligation):
                    self._builtins.finalize(
                        replace(
                            obligation,
                            target_type=region.engine.zonk(obligation.target_type),
                            result_type=region.engine.zonk(obligation.result_type),
                        )
                    )
                else:
                    self._finalize_extern_call_obligation(
                        replace(obligation, target_type=region.engine.zonk(obligation.target_type))
                    )
            self._finalize_extern_provenance(region)
            if region.engine.has_variables():
                final_type = region.engine.zonk(typ)
                final_node_types = {
                    node_id: region.engine.zonk(node_type)
                    for node_id, node_type in region.node_types.items()
                }
                final_param_types = {
                    node_id: tuple(region.engine.zonk(param_type) for param_type in param_types)
                    for node_id, param_types in region.function_call_param_types.items()
                }
                # Extern-target result types are already validated for leaked
                # inference variables by ``_finalize_extern_provenance`` above (via
                # ``_zonk_extern_targets``), so they are deliberately not re-walked
                # here — this pass covers only node/param types and call sites.
                region.engine.assert_no_inference_vars(
                    (
                        *final_node_types.values(),
                        *(t for ts in final_param_types.values() for t in ts),
                        *(
                            call_site.target_type
                            for call_site in self._call_sites[region.call_sites_start :]
                        ),
                    )
                )
            else:
                # No inference variable was ever allocated, so the recorded types
                # are already final — skip zonking and the leak-check entirely.
                final_type = typ
                final_node_types = region.node_types
                final_param_types = region.function_call_param_types
            self._node_types.update(final_node_types)
            self._function_call_param_types.update(final_param_types)
            completed = True
            return final_type
        except InferenceError as exc:
            raise AglTypeError(str(exc), span=exc.span, related=exc.related) from exc
        finally:
            if not completed:
                self._rollback_region_side_tables(region)
            self._inference_region = None

    def _register_builtin_obligation(self, obligation: PendingBuiltinObligation) -> None:
        """Queue one built-in contract operation in source registration order."""
        assert self._inference_region is not None
        self._inference_region.finalization_obligations.append(obligation)

    def _register_extern_call_obligation(self, node: Call, callee: str, target_type: Type) -> None:
        """Queue typed extern inventory metadata in source registration order."""
        assert self._inference_region is not None
        self._inference_region.finalization_obligations.append(
            PendingExternCallObligation(
                node_id=node.node_id,
                callee=callee,
                target_type=target_type,
                span=node.span,
            )
        )

    def _record_node_type(self, node_id: int, typ: Type) -> None:
        """Store a node type provisionally while its inference region is open."""
        if self._inference_region is None:
            self._node_types[node_id] = typ
        else:
            self._inference_region.node_types[node_id] = typ

    def _record_function_call_param_types(
        self, node_id: int, param_types: tuple[Type, ...]
    ) -> None:
        """Store direct-call parameter types with the region that owns their variables."""
        assert self._inference_region is not None
        self._inference_region.function_call_param_types[node_id] = param_types

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
        if isinstance(expr, Placeholder):
            raise AssertionError("compiler bug: placeholder reached expression type checking")
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
        if isinstance(expr, Return):
            return self._check_return(expr)
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
        if node.node_id in self._resolved.qualified_constructor_refs:
            owner_name, variant, owner_module_id = self._resolved.qualified_constructor_refs[
                node.node_id
            ]
            if node.type_qualifier is not None and node.type_qualifier.type_args is not None:
                return self._constructors.check_qualified_constructor_type_apply(
                    owner_name=owner_name,
                    variant=variant,
                    owner_module_id=owner_module_id,
                    type_args=node.type_qualifier.type_args,
                    span=node.span,
                )
            return self._constructors.check_qualified_constructor_as_value(
                owner_name=owner_name,
                variant=variant,
                owner_module_id=owner_module_id,
                span=node.span,
                expected=expected,
            )
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
            if not ref.module_id.is_entry:
                return self._constructors.check_cross_module_constructor_as_value(
                    ref, span=node.span, expected=expected
                )
            raise AglTypeError(
                f"'{node.name}' is a type name, not a value; "
                "use it with a constructor call (e.g. 'EnumName::Variant' or 'RecordName(...)').",
                span=node.span,
            )
        typ = self._require_binding_type(ref)
        # Every generic function occurrence receives fresh flexible variables.
        # They remain local to the enclosing expression region, so a higher-order
        # call can connect this occurrence to evidence from its other arguments.
        if ref.kind is BinderKind.function_binding:
            sig = self._env.get_function_signature_by_node_id(ref.decl_node_id)
            if sig is not None and sig.type_params:
                assert isinstance(typ, FunctionType)
                assert self._inference_region is not None
                engine = self._inference_region.engine
                instantiation = engine.instantiate(sig.type_params, (typ,))
                concrete = instantiation.templates[0]
                assert isinstance(concrete, FunctionType)
                for type_param in sig.type_params:
                    engine.require_solved(
                        instantiation.variables[type_param],
                        engine.origin(
                            node.span,
                            role=ConstraintRole.EXPECTED_RESULT,
                            subject=ref.name,
                            type_param=type_param,
                        ),
                    )
                if expected is not None:
                    engine.complete_from_context(
                        concrete,
                        expected,
                        engine.origin(
                            node.span, role=ConstraintRole.EXPECTED_RESULT, subject=ref.name
                        ),
                    )
                self._set_extern_expr_targets(
                    node.node_id, self._extern_targets_for_ref(ref, concrete)
                )
                return concrete
        self._set_extern_expr_targets(node.node_id, self._extern_targets_for_ref(ref, typ))
        return typ

    def _active_inference_engine(self) -> InferenceEngine:
        """Return the solver for the expression region currently being checked."""
        assert self._inference_region is not None
        return self._inference_region.engine

    def _constrain_argument(
        self,
        slot_type: Type,
        arg_expr: Expr,
        *,
        role: ConstraintRole,
        subject: str,
        error_subject: str,
    ) -> Type:
        """Check ``arg_expr`` against ``slot_type`` and register the right constraint.

        A flexible slot or argument type is unified into the active inference region;
        a fully concrete position falls back to ordinary assignability.
        """
        argument_type = self._check_expr(arg_expr, expected=slot_type)
        if contains_inference_var(slot_type) or contains_inference_var(argument_type):
            engine = self._active_inference_engine()
            try:
                engine.unify(
                    slot_type,
                    argument_type,
                    engine.origin(arg_expr.span, role=role, subject=subject),
                )
            except InferenceError as exc:
                raise AglTypeError(
                    f"Inconsistent type argument for {error_subject}: {exc}",
                    span=exc.span,
                    related=exc.related,
                ) from exc
        else:
            self._assert_assignable(argument_type, slot_type, arg_expr.span)
        return argument_type

    def _instantiate_generic_constructor_value(
        self,
        *,
        type_params: tuple[str, ...],
        field_templates: tuple[Type, ...],
        result_template: Type,
        span: SourceSpan,
        expected: Type | None,
        subject: str,
    ) -> Type:
        """Freshen a generic constructor value in the active expression region."""
        engine = self._active_inference_engine()
        instantiation = engine.instantiate(type_params, (*field_templates, result_template))
        for type_param in type_params:
            engine.require_solved(
                instantiation.variables[type_param],
                engine.origin(
                    span,
                    role=ConstraintRole.EXPECTED_RESULT,
                    subject=subject,
                    type_param=type_param,
                ),
            )
        result = instantiation.templates[-1]
        concrete: Type = (
            FunctionType(params=instantiation.templates[:-1], result=result)
            if field_templates
            else result
        )
        if expected is not None:
            engine.complete_from_context(
                concrete,
                expected,
                engine.origin(span, role=ConstraintRole.EXPECTED_RESULT, subject=subject),
            )
        return concrete

    def _zonk_constructor_owner(
        self, owner: RecordType | EnumType | ExceptionType
    ) -> RecordType | EnumType | ExceptionType:
        """Resolve a nominal owner before a constructor-side TypeTable lookup."""
        if self._inference_region is not None:
            zonked = self._inference_region.engine.zonk(owner)
            assert isinstance(zonked, (RecordType, EnumType, ExceptionType))
            owner = zonked
        if contains_inference_var(owner):
            raise AssertionError(
                "TypeTable received a constructor owner with flexible type arguments"
            )
        return owner

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
        # (e.g. ``some::[int]``; qualified refs use ``Option[int]::none``): delegate to the
        # constructor checker, which instantiates the constructor with the
        # supplied type arguments and returns a function value (payload) or
        # the constructed nominal value (nullary variant).
        if isinstance(node.expr, VarRef) and node.expr.node_id in self._resolved.constructor_refs:
            ctor_ref = self._resolved.constructor_refs[node.expr.node_id]
            typ = self._constructors.check_constructor_type_apply(
                ctor_ref=ctor_ref, type_args=node.type_args, span=node.span
            )
            self._record_node_type(node.expr.node_id, typ)
            return typ
        if (
            isinstance(node.expr, VarRef)
            and node.expr.node_id in self._resolved.qualified_constructor_refs
        ):
            raise self._qualified_constructor_typed_call_error(node.span)
        if isinstance(node.expr, VarRef):
            constructor_ref = self._resolved.resolution.get(node.expr.node_id)
            if (
                constructor_ref is not None
                and constructor_ref.kind is BinderKind.constructor_binding
                and not constructor_ref.module_id.is_entry
            ):
                typ = self._constructors.check_cross_module_constructor_type_apply(
                    constructor_ref, type_args=node.type_args, span=node.span
                )
                self._record_node_type(node.expr.node_id, typ)
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
        self._record_node_type(node.expr.node_id, concrete)
        self._set_extern_expr_targets(node.node_id, self._extern_targets_for_ref(ref, concrete))
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
        # Only a FALLIBLE cast derives a JSON schema at lowering time
        # (TOTAL_RENDER/TOTAL_JSON/TOTAL_NOOP never do) — and it may need one
        # not just for a bare record/enum target but for a composite target
        # containing one (e.g. list[Perfect[int]], dict[text, Perfect[int]]).
        # A type whose reachable instantiation closure is infinite has no
        # finite schema to derive, so reject it here rather than crashing at
        # lowering. no_finite_schema_message returns None for scalar/finite
        # targets, so this is a safe no-op for every other FALLIBLE cast.
        if kind == CastKind.FALLIBLE:
            message = self._env.type_table.no_finite_schema_message(
                target_type, use="a cast target"
            )
            if message is not None:
                raise AglTypeError(message, span=node.span)
            if not self._type_is_wire_serializable(target_type):
                raise AglTypeError(
                    f"Cast target '{target_type!r}' is not JSON-serializable; "
                    "use a JSON-serializable data type.",
                    span=node.span,
                )
        self._record_cast_spec(node.node_id, CastSpec(target_type=target_type, kind=kind))
        return BoolType() if node.test_only else target_type

    # --- Call dispatch ---

    def _check_call(self, node: Call, *, expected: Type | None) -> Type:
        """Dispatch a Call node to the appropriate checker."""
        holes = self._partial_hole_indices(node)
        return self._check_call_dispatch(node, expected=expected, hole_indices=holes)

    @staticmethod
    def _partial_hole_indices(node: Call) -> dict[int, int]:
        holes = [
            expr
            for expr in (*node.args, *(arg.value for arg in node.named_args))
            if isinstance(expr, Placeholder)
        ]
        if all(hole.index is None for hole in holes):
            return {hole.node_id: index for index, hole in enumerate(holes)}
        return {hole.node_id: hole.index - 1 for hole in holes if hole.index is not None}

    @staticmethod
    def _call_result_type(
        params: tuple[ParamSpec, ...],
        result: Type,
        binding: tuple[Expr | None, ...],
        hole_indices: Mapping[int, int],
    ) -> Type:
        if not hole_indices:
            return result
        hole_types: list[Type | None] = [None] * len(hole_indices)
        for spec, bound_expr in zip(params, binding):
            if isinstance(bound_expr, Placeholder):
                hole_types[hole_indices[bound_expr.node_id]] = spec.type
        assert all(typ is not None for typ in hole_types), (
            "compiler bug: partial call hole was not bound to a parameter"
        )
        return FunctionType(
            params=tuple(typ for typ in hole_types if typ is not None),
            result=result,
        )

    def _record_side_table_addition(
        self, table_name: str, table: Mapping[int, object], node_id: int
    ) -> None:
        """Remember an insertion so a failed region can remove just its delta."""
        region = self._inference_region
        if region is not None and node_id not in table:
            region.added_side_table_keys.setdefault(table_name, set()).add(node_id)

    def _rollback_region_side_tables(self, region: _InferenceRegion) -> None:
        """Discard provisional side-table additions and append-only suffixes."""
        for table_name, table in (
            ("function_call_bindings", self._function_call_bindings),
            ("constructor_call_bindings", self._constructor_call_bindings),
            ("constructor_pattern_bindings", self._constructor_pattern_bindings),
            ("pattern_classifications", self._pattern_classifications),
            ("partial_calls", self._partial_calls),
            ("contract_specs", self._contract_specs),
            ("cast_specs", self._cast_specs),
            ("extern_expr_targets", self._extern_expr_targets),
            ("extern_binding_targets", self._extern_binding_targets),
        ):
            for node_id in region.added_side_table_keys.get(table_name, set()):
                table.pop(node_id, None)
        self._resolved.resolution.update(region.reconciled_resolution_originals)
        for node_id, constructor_ref in region.reconciled_constructor_ref_originals.items():
            if constructor_ref is None:
                self._resolved.constructor_refs.pop(node_id, None)
            else:
                self._resolved.constructor_refs[node_id] = constructor_ref
        del self._call_sites[region.call_sites_start :]
        del self._warnings[region.warnings_start :]
        for targets, start in zip(
            self._return_extern_targets_stack, region.return_target_lengths, strict=True
        ):
            del targets[start:]

    def _record_contract_spec(self, node_id: int, spec: OutputContractSpec) -> None:
        """Store a region-owned output contract specification."""
        self._record_side_table_addition("contract_specs", self._contract_specs, node_id)
        self._contract_specs[node_id] = spec

    def _record_cast_spec(self, node_id: int, spec: CastSpec) -> None:
        """Store a region-owned cast specification."""
        self._record_side_table_addition("cast_specs", self._cast_specs, node_id)
        self._cast_specs[node_id] = spec

    def _record_function_call_binding(self, node_id: int, binding: tuple[Expr | None, ...]) -> None:
        """Store a region-owned declared-call argument binding."""
        self._record_side_table_addition(
            "function_call_bindings", self._function_call_bindings, node_id
        )
        self._function_call_bindings[node_id] = binding

    def _record_constructor_pattern_binding(
        self, node_id: int, binding: tuple[tuple[str, Pattern], ...]
    ) -> None:
        """Store a region-owned constructor-pattern argument binding."""
        self._record_side_table_addition(
            "constructor_pattern_bindings", self._constructor_pattern_bindings, node_id
        )
        self._constructor_pattern_bindings[node_id] = binding

    def _record_pattern_classification(
        self, node_id: int, constructor: ConstructorRef | None
    ) -> None:
        """Publish one final bare/as-pattern classification for this check region."""
        self._record_side_table_addition(
            "pattern_classifications", self._pattern_classifications, node_id
        )
        self._pattern_classifications[node_id] = constructor

    def _record_constructor_call_binding(self, node_id: int, binding: dict[str, Expr]) -> None:
        """Store a region-owned constructor-call argument binding."""
        self._record_side_table_addition(
            "constructor_call_bindings", self._constructor_call_bindings, node_id
        )
        self._constructor_call_bindings[node_id] = binding

    def _append_warning(self, warning: Diagnostic) -> None:
        """Append a warning produced while finalizing the active region."""
        self._warnings.append(warning)

    def _append_call_site(self, call_site: CallSiteRecord) -> None:
        """Append a call site produced while finalizing the active region."""
        self._call_sites.append(call_site)

    def _record_partial_call(
        self,
        node: Call,
        binding: tuple[Expr | None, ...],
        hole_indices: Mapping[int, int],
        *,
        callee_kind: Literal["declared", "constructor", "value"] = "declared",
    ) -> None:
        self._record_side_table_addition("partial_calls", self._partial_calls, node.node_id)
        self._partial_calls[node.node_id] = PartialCallSpec(
            argument_holes=tuple(
                hole_indices[expr.node_id] if isinstance(expr, Placeholder) else None
                for expr in binding
            ),
            callee_kind=callee_kind,
        )

    def _set_extern_expr_targets(self, node_id: int, targets: _ExternTargets) -> None:
        """Record or clear extern provenance for a function-valued expression."""
        if targets:
            self._record_side_table_addition(
                "extern_expr_targets", self._extern_expr_targets, node_id
            )
            self._extern_expr_targets[node_id] = targets
        else:
            self._extern_expr_targets.pop(node_id, None)

    def _set_extern_binding_targets(self, node_id: int, targets: _ExternTargets) -> None:
        """Record or clear extern provenance for a function-valued binding."""
        if targets:
            self._record_side_table_addition(
                "extern_binding_targets", self._extern_binding_targets, node_id
            )
            self._extern_binding_targets[node_id] = targets
        else:
            self._extern_binding_targets.pop(node_id, None)

    @staticmethod
    def _merge_extern_targets(
        *target_groups: Sequence[_ExternTarget],
    ) -> _ExternTargets:
        """Merge extern provenance groups in source order, removing duplicates."""
        merged: list[_ExternTarget] = []
        for targets in target_groups:
            for target in targets:
                if target not in merged:
                    merged.append(target)
        return tuple(merged)

    def _extern_targets_for_expr(self, expr: Expr, result_type: Type) -> _ExternTargets:
        """Return extern provenance for a function-valued expression result."""
        if not isinstance(result_type, FunctionType):
            return ()
        return self._extern_expr_targets.get(expr.node_id, ())

    def _current_return_extern_targets(self) -> _ExternTargets:
        """Return merged extern provenance from the active return context."""
        assert self._return_extern_targets_stack
        return tuple(self._return_extern_targets_stack[-1])

    def _record_return_extern_targets(self, targets: _ExternTargets) -> None:
        """Accumulate returned function provenance for the active function/lambda."""
        if not targets:
            return
        assert self._return_extern_targets_stack
        self._return_extern_targets_stack[-1].extend(targets)

    def _extern_targets_for_ref(self, ref: BindingRef, typ: Type) -> _ExternTargets:
        """Return extern call targets represented by a resolved value reference."""
        if ref.kind is BinderKind.function_binding and self._env.is_extern_node_id(
            ref.decl_node_id
        ):
            assert isinstance(typ, FunctionType), (
                f"extern binding {ref.name!r} has non-function type {typ!r}"
            )
            return (
                _ExternTarget(
                    name=ref.name,
                    result_type=typ.result,
                    decl_node_id=ref.decl_node_id,
                    module_id=ref.module_id,
                ),
            )
        return self._extern_binding_targets.get(ref.decl_node_id, ())

    def _extern_targets_for_function_exprs(
        self, exprs: Sequence[Expr], result_type: Type
    ) -> _ExternTargets:
        """Merge extern provenance from branches of a function-valued expression."""
        if not isinstance(result_type, FunctionType):
            return ()
        return self._merge_extern_targets(
            *(self._extern_expr_targets.get(expr.node_id, ()) for expr in exprs)
        )

    def _finalize_extern_call_obligation(self, obligation: PendingExternCallObligation) -> None:
        """Publish one concrete extern inventory record at region close."""
        if contains_inference_var(obligation.target_type):
            raise AglTypeError(
                "Cannot infer a concrete target type for this extern call.", span=obligation.span
            )
        self._append_call_site(
            CallSiteRecord(
                node_id=obligation.node_id,
                callee=obligation.callee,
                target_type=obligation.target_type,
                codec_name="extern",
                parse_policy="default",
                line=obligation.span.start_line,
                col=obligation.span.start_col,
            )
        )

    def _zonk_extern_targets(
        self, targets: _ExternTargets, engine: InferenceEngine
    ) -> _ExternTargets:
        """Zonk and validate one function-provenance target group."""
        zonked = tuple(
            replace(target, result_type=engine.zonk(target.result_type)) for target in targets
        )
        engine.assert_no_inference_vars(target.result_type for target in zonked)
        return self._merge_extern_targets(zonked)

    def _finalize_extern_provenance(self, region: _InferenceRegion) -> None:
        """Publish only the completed region's zonked extern provenance."""
        for node_id in region.added_side_table_keys.get("extern_expr_targets", set()):
            targets = self._extern_expr_targets.get(node_id)
            if targets is not None:
                self._extern_expr_targets[node_id] = self._zonk_extern_targets(
                    targets, region.engine
                )
        for node_id in region.added_side_table_keys.get("extern_binding_targets", set()):
            targets = self._extern_binding_targets.get(node_id)
            if targets is not None:
                self._extern_binding_targets[node_id] = self._zonk_extern_targets(
                    targets, region.engine
                )
        for return_targets, start in zip(
            self._return_extern_targets_stack, region.return_target_lengths, strict=True
        ):
            return_targets[start:] = self._zonk_extern_targets(
                tuple(return_targets[start:]), region.engine
            )

    def _check_call_dispatch(
        self,
        node: Call,
        *,
        expected: Type | None,
        hole_indices: Mapping[int, int],
    ) -> Type:
        # Built-in?
        if node.node_id in self._resolved.builtin_calls:
            kind = self._resolved.builtin_calls[node.node_id]
            if hole_indices:
                builtin_name = next(
                    name for name, value in BUILTIN_CALL_NAMES.items() if value is kind
                )
                raise AglTypeError(
                    f"Cannot use placeholder arguments with special builtin '{builtin_name}'; "
                    "partial application is not supported.",
                    span=node.span,
                )
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
            return self._constructors.check_constructor_callee_call(
                node, expected=expected, hole_indices=hole_indices
            )
        if (
            isinstance(node.callee, VarRef)
            and node.callee.node_id in self._resolved.qualified_constructor_refs
        ):
            if node.type_args:
                raise self._qualified_constructor_typed_call_error(node.span)
            return self._constructors.check_qualified_constructor_callee_call(
                node, expected=expected, hole_indices=hole_indices
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
                    callee_ref=callee_ref,
                    hole_indices=hole_indices,
                )
            if (
                callee_ref is not None
                and callee_ref.kind is BinderKind.constructor_binding
                and not callee_ref.module_id.is_entry
            ):
                return self._constructors.check_cross_module_constructor_call(
                    node, callee_ref, expected=expected, hole_indices=hole_indices
                )

        # Value call (lambda or higher-order).
        return self._check_value_call(node, expected=expected, hole_indices=hole_indices)

    # --- shared call-argument checker ---

    @staticmethod
    def _bind_call_args(
        params: tuple[ParamSpec, ...],
        node: Call,
        func_name: str,
    ) -> tuple[Expr | None, ...]:
        """Bind a call's positional/named args against *params* (declaration order).

        Shared by concrete checking and generic inference, so a bare-name shorthand
        in named-only territory is always matched to its parameter by NAME rather
        than by raw positional index. Raises ``AglTypeError`` on any binding violation.
        """
        return bind_call_args(
            params,
            node.args,
            node.named_args,
            call_span=node.span,
            context_desc=f"call to '{func_name}'",
        )

    def _check_bound_call_args(
        self,
        params: tuple[ParamSpec, ...],
        binding: tuple[Expr | None, ...],
    ) -> None:
        for spec, bound_expr in zip(params, binding):
            if bound_expr is None or isinstance(bound_expr, Placeholder):
                continue
            at = self._check_expr(bound_expr, expected=spec.type)
            self._assert_assignable(at, spec.type, bound_expr.span)

    def _finish_declared_call(
        self,
        node: Call,
        params: tuple[ParamSpec, ...],
        result: Type,
        binding: tuple[Expr | None, ...],
        hole_indices: Mapping[int, int],
        *,
        check_args: bool = True,
    ) -> Type:
        """Record direct-call side tables and build the result type."""
        self._record_function_call_binding(node.node_id, binding)
        self._record_function_call_param_types(node.node_id, tuple(p.type for p in params))
        if hole_indices:
            self._record_partial_call(node, binding, hole_indices)
        if check_args:
            self._check_bound_call_args(params, binding)
        return self._call_result_type(params, result, binding, hole_indices)

    # --- generic declared-name call ---

    def _check_generic_declared_call(
        self,
        node: Call,
        func_name: str,
        sig: FunctionSignature,
        binding: tuple[Expr | None, ...],
        hole_indices: Mapping[int, int],
        *,
        expected: Type | None,
    ) -> Type:
        """Check a generic declared call with fresh expression-local variables."""
        if node.type_args:
            if len(node.type_args) != len(sig.type_params):
                raise AglTypeError(
                    f"'{func_name}' requires {len(sig.type_params)} type argument(s), "
                    f"but {len(node.type_args)} were supplied.",
                    span=node.span,
                )
            subst = {
                type_param: self._env.resolve_type_expr(
                    type_arg, span=node.span, type_vars=self._current_type_vars
                )
                for type_param, type_arg in zip(sig.type_params, node.type_args, strict=True)
            }
            params = tuple(
                ParamSpec(
                    name=param.name,
                    type=substitute(param.type, subst),
                    kind=param.kind,
                    has_default=param.has_default,
                )
                for param in sig.params
            )
            return self._finish_declared_call(
                node, params, substitute(sig.result, subst), binding, hole_indices
            )

        assert self._inference_region is not None
        engine = self._inference_region.engine
        templates = (*tuple(param.type for param in sig.params), sig.result)
        instantiation = engine.instantiate(sig.type_params, templates)
        params = tuple(
            ParamSpec(
                name=param.name,
                type=instantiation.templates[index],
                kind=param.kind,
                has_default=param.has_default,
            )
            for index, param in enumerate(sig.params)
        )
        result = instantiation.templates[-1]
        for type_param in sig.type_params:
            engine.require_solved(
                instantiation.variables[type_param],
                engine.origin(
                    node.span,
                    role=ConstraintRole.EXPECTED_RESULT,
                    subject=func_name,
                    type_param=type_param,
                ),
            )

        # Check every supplied argument before allowing expected-result context
        # to fill a still-unresolved variable. Exact constraints select the
        # generic instantiation; ordinary assignability remains a post-solve
        # check for fully concrete parameter positions.
        for param, bound_expr in zip(params, binding, strict=True):
            if bound_expr is None or isinstance(bound_expr, Placeholder):
                continue
            self._constrain_argument(
                param.type,
                bound_expr,
                role=ConstraintRole.FUNCTION_ARGUMENT,
                subject=func_name,
                error_subject=f"call to '{func_name}'",
            )

        produced = self._call_result_type(params, result, binding, hole_indices)
        if expected is not None:
            engine.complete_from_context(
                produced,
                expected,
                engine.origin(node.span, role=ConstraintRole.EXPECTED_RESULT, subject=func_name),
            )
        return self._finish_declared_call(
            node, params, result, binding, hole_indices, check_args=False
        )

    # --- declared-name call ---

    def _check_declared_name_call(
        self,
        node: Call,
        func_name: str,
        *,
        expected: Type | None,
        callee_ref: BindingRef,
        hole_indices: Mapping[int, int],
    ) -> Type:
        # Use the node-id-keyed lookup populated by the function-signature pre-pass
        # (program context) and by _preregister_funcdef (module mode).  Keying
        # by the callee's globally-unique decl_node_id avoids the same-name collision
        # where two modules define functions with identical names but different
        # signatures, which would cause the name-keyed table to return the wrong
        # signature for a qualified cross-module call.
        sig = self._env.get_function_signature_by_node_id(callee_ref.decl_node_id)
        if sig is None:
            raise AglTypeError(
                f"Cannot infer return type of function '{func_name}' before it is checked. "
                "Add a return type annotation.",
                span=node.span,
            )

        binding = self._bind_call_args(sig.params, node, func_name)

        # Dispatch to the generic path when the function has type parameters.
        if sig.type_params:
            result_type = self._check_generic_declared_call(
                node,
                func_name,
                sig,
                binding,
                hole_indices,
                expected=expected,
            )
        else:
            # Non-generic path: reject unexpected explicit type args.
            if node.type_args:
                raise AglTypeError(
                    f"'{func_name}' is not a generic function and does not accept type arguments.",
                    span=node.span,
                )
            result_type = self._finish_declared_call(
                node, sig.params, sig.result, binding, hole_indices
            )

        # Calls to a known extern are recorded like ask/exec call sites, for
        # own-module AND imported (program) externs alike. A partial call
        # only builds a function value, so carry extern provenance forward and
        # record the eventual invocation site instead.
        if self._env.is_extern_node_id(callee_ref.decl_node_id):
            extern_target_type = (
                result_type.result
                if hole_indices and isinstance(result_type, FunctionType)
                else result_type
            )
            if hole_indices:
                self._set_extern_expr_targets(
                    node.node_id,
                    (
                        _ExternTarget(
                            name=func_name,
                            result_type=extern_target_type,
                            decl_node_id=callee_ref.decl_node_id,
                            module_id=callee_ref.module_id,
                        ),
                    ),
                )
            else:
                self._register_extern_call_obligation(node, func_name, extern_target_type)
        elif isinstance(result_type, FunctionType):
            self._set_extern_expr_targets(
                node.node_id, self._extern_binding_targets.get(callee_ref.decl_node_id, ())
            )

        return result_type

    # --- value call (lambda / higher-order) ---

    def _check_value_call_head(self, node: Call) -> FunctionType:
        """Validate a function-value call site and return the callee's function type.

        Rejects named arguments (accepted only at declared-function call sites),
        requires the callee to be a function value, and checks positional arity.
        """
        if node.named_args:
            raise AglTypeError(
                "Named arguments are not allowed when calling a function value; "
                "they are only allowed at declared-function call sites.",
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
        return callee_type

    def _check_value_call(
        self,
        node: Call,
        *,
        expected: Type | None,
        hole_indices: Mapping[int, int],
    ) -> Type:
        callee_type = self._check_value_call_head(node)
        binding: tuple[Expr | None, ...] = node.args
        if hole_indices:
            self._record_partial_call(node, binding, hole_indices, callee_kind="value")
        for arg, ptype in zip(node.args, callee_type.params, strict=True):
            if isinstance(arg, Placeholder):
                continue
            self._constrain_argument(
                ptype,
                arg,
                role=ConstraintRole.FUNCTION_ARGUMENT,
                subject="function value",
                error_subject="function value call",
            )
        params = tuple(
            ParamSpec(
                name=f"arg{index}",
                type=ptype,
                kind=ParamKind.STANDARD,
                has_default=False,
            )
            for index, ptype in enumerate(callee_type.params)
        )
        result_type = self._call_result_type(params, callee_type.result, binding, hole_indices)
        if expected is not None:
            assert self._inference_region is not None
            self._inference_region.engine.complete_from_context(
                result_type,
                expected,
                self._inference_region.engine.origin(
                    node.span,
                    role=ConstraintRole.EXPECTED_RESULT,
                    subject="function value",
                ),
            )
        extern_targets = self._extern_expr_targets.get(node.callee.node_id, ())
        if extern_targets:
            invocation_targets = tuple(
                _ExternTarget(
                    name=target.name,
                    result_type=callee_type.result,
                    decl_node_id=target.decl_node_id,
                    module_id=target.module_id,
                )
                for target in extern_targets
            )
            if hole_indices:
                self._set_extern_expr_targets(node.node_id, invocation_targets)
            elif isinstance(result_type, FunctionType):
                self._set_extern_expr_targets(node.node_id, invocation_targets)
            else:
                for target in invocation_targets:
                    self._register_extern_call_obligation(node, target.name, target.result_type)
        return result_type

    # --- Lambda ---

    def _check_lambda(self, node: Lambda, *, expected: Type | None) -> Type:
        # no required positional-fillable param may follow a defaulted one.
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
            with self._return_context(result_type):
                body_type = self._check_expr(node.body, expected=result_type)
            if not isinstance(body_type, BottomType):
                self._assert_assignable(body_type, result_type, node.span)
        else:
            with self._return_context(None) as collected:
                body_type = self._check_expr(node.body, expected=None)
            if isinstance(body_type, BottomType) and not collected:
                raise AglTypeError(
                    "Cannot infer return type of lambda: body always raises.",
                    span=node.span,
                )
            try:
                result_type = self._unify_branch_types(
                    [*collected, body_type], node.span, "Lambda return"
                )
            except AglTypeError as exc:
                raise AglTypeError(
                    "Cannot infer return type of lambda: return values have incompatible "
                    "types. Add a return type annotation.",
                    span=node.span,
                    related=exc.related,
                ) from exc

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

        result = self._unify_branch_types(branch_types, node.span, "If expression")
        self._set_extern_expr_targets(
            node.node_id,
            self._extern_targets_for_function_exprs(
                tuple(branch.body for branch in node.branches), result
            ),
        )
        return result

    # --- case ---

    def _check_case(self, node: Case, *, expected: Type | None) -> Type:
        subj_type = self._check_expr(node.subject, expected=None)
        branch_types: list[Type] = []
        for branch in node.branches:
            self._bind_pattern_types(branch.pattern, subj_type, branch)
            self._reconcile_provisional_pattern_references(node, branch.pattern)
            bt = self._check_expr(branch.body, expected=expected)
            branch_types.append(bt)
        result = self._unify_branch_types(branch_types, node.span, "Case expression")
        self._set_extern_expr_targets(
            node.node_id,
            self._extern_targets_for_function_exprs(
                tuple(branch.body for branch in node.branches), result
            ),
        )
        return result

    # --- loop ---

    def _check_loop(self, node: Loop) -> Type:
        if node.bound is not None:
            bound_type = self._check_expr(node.bound, expected=None)
            if not self._is_type_or_bottom(bound_type, IntType):
                raise AglTypeError(
                    f"do-loop bound must be int; got '{bound_type!r}'.",
                    span=node.bound.span,
                )
        if node.for_range_to is not None:
            # Integer-range for: for VAR in a to/downto b [by k]
            assert node.for_iter is not None
            start_type = self._check_expr(node.for_iter, expected=None)
            if not self._is_type_or_bottom(start_type, IntType):
                raise AglTypeError(
                    f"'for' range start must be int; got '{start_type!r}'.",
                    span=node.for_iter.span,
                )
            to_type = self._check_expr(node.for_range_to, expected=None)
            if not self._is_type_or_bottom(to_type, IntType):
                raise AglTypeError(
                    f"'for' range bound must be int; got '{to_type!r}'.",
                    span=node.for_range_to.span,
                )
            if node.for_range_by is not None:
                by_type = self._check_expr(node.for_range_by, expected=None)
                if not self._is_type_or_bottom(by_type, IntType):
                    raise AglTypeError(
                        f"'for' range step must be int; got '{by_type!r}'.",
                        span=node.for_range_by.span,
                    )
                # Static guard: a literal step <= 0 is always wrong.
                # IntLit(0) → zero; UnaryNeg(IntLit(k)) with k >= 1 → always negative.
                by_expr = node.for_range_by
                if isinstance(by_expr, IntLit) and by_expr.value <= 0:
                    raise AglTypeError(
                        f"loop step must be positive; got a literal step of {by_expr.value}.",
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
        handler_bodies: list[Expr] = []
        for clause in node.handlers:
            ht = self._check_catch_clause(clause, expected=expected)
            handler_types.append(ht)
            handler_bodies.append(clause.body)
        result = self._unify_branch_types(handler_types, node.span, "Try expression")
        self._set_extern_expr_targets(
            node.node_id,
            self._extern_targets_for_function_exprs((node.body, *handler_bodies), result),
        )
        return result

    def _check_catch_clause(self, clause: CatchClause, *, expected: Type | None) -> Type:
        if clause.exc_type is None or clause.exc_type == "_":
            from agm.agl.semantics.types import EXCEPTION_BASE

            exc_type: ExceptionType = EXCEPTION_BASE
        else:
            # resolve_named_type is used instead of get_type so that open-imported
            # exception types (cross-module program context) are found as well.
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

    # --- return ---

    def _check_return(self, node: Return) -> BottomType:
        if not self._return_expected_stack:
            raise AglTypeError("'return' used outside a function.", span=node.span)
        expected = self._return_expected_stack[-1]
        value_type = (
            UnitType() if node.value is None else self._check_expr(node.value, expected=expected)
        )
        targets = (
            self._extern_targets_for_expr(node.value, value_type) if node.value is not None else ()
        )
        self._set_extern_expr_targets(node.node_id, targets)
        self._record_return_extern_targets(targets)
        if expected is None:
            self._return_collected_stack[-1].append(value_type)
        else:
            self._assert_assignable(value_type, expected, node.span)
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
                    self._record_node_type(seg.expr.node_id, seg_type)
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
            self._record_node_type(expr.node_id, result)
            return result
        if isinstance(expr, DictLit):
            if not expr.entries:
                return self._check_expr(expr, expected=JsonType())
            result = self._check_template_literal(expr)
            self._record_node_type(expr.node_id, result)
            return result
        child_type = self._check_expr(expr, expected=None)
        self._assert_assignable(child_type, JsonType(), expr.span)
        return child_type

    # --- binary ops ---

    @staticmethod
    def _is_type_or_bottom(t: Type, *allowed: type[Type]) -> bool:
        """Whether ``t`` is one of ``allowed`` or the bottom type.

        A bottom-typed operand (from ``return``/``raise``) is accepted wherever a
        concrete operand type is required, since bottom is assignable to any type.
        """
        return isinstance(t, allowed) or isinstance(t, BottomType)

    def _check_binary_op(self, node: BinaryOp) -> Type:
        left_type = self._check_expr(node.left, expected=None)
        right_type = self._check_expr(node.right, expected=None)
        op = node.op

        if op in (BinOp.AND, BinOp.OR):
            op_name = "and" if op is BinOp.AND else "or"
            if not self._is_type_or_bottom(left_type, BoolType):
                raise AglTypeError(
                    f"'{op_name}' requires bool operands; left operand has type '{left_type!r}'.",
                    span=node.left.span,
                )
            if not self._is_type_or_bottom(right_type, BoolType):
                raise AglTypeError(
                    f"'{op_name}' requires bool operands; right operand has type '{right_type!r}'.",
                    span=node.right.span,
                )
            return BoolType()

        if op in (BinOp.EQ, BinOp.NEQ):
            # reject operations on bare type variables.
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
            if not (
                isinstance(left_type, BottomType)
                or isinstance(right_type, BottomType)
                or comparable_types(left_type, right_type, self._env.type_table)
            ):
                raise AglTypeError(
                    f"Equality operands must have the same type; "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return BoolType()

        if op in (BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE):
            # reject operations on bare type variables.
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
            numeric_pair = self._is_type_or_bottom(
                left_type, IntType, DecimalType
            ) and self._is_type_or_bottom(right_type, IntType, DecimalType)
            text_pair = self._is_type_or_bottom(left_type, TextType) and self._is_type_or_bottom(
                right_type, TextType
            )
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
            # reject operations on bare type variables.
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
                self._is_type_or_bottom(left_type, IntType, DecimalType)
                and self._is_type_or_bottom(right_type, IntType, DecimalType)
            ):
                raise AglTypeError(
                    f"'/' requires numeric operands; got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return DecimalType()

        raise AglTypeError(  # pragma: no cover
            f"Unknown binary operator: {op!r}",
            span=node.span,
        )

    def _check_add(self, left_type: Type, right_type: Type, span: SourceSpan) -> Type:
        # reject operations on bare type variables.
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
        if self._is_type_or_bottom(left_type, TextType) and self._is_type_or_bottom(
            right_type, TextType
        ):
            if isinstance(left_type, TextType) or isinstance(right_type, TextType):
                return TextType()
        if self._is_type_or_bottom(left_type, IntType, DecimalType) and self._is_type_or_bottom(
            right_type, IntType, DecimalType
        ):
            if isinstance(left_type, DecimalType) or isinstance(right_type, DecimalType):
                return DecimalType()
            return IntType()
        raise AglTypeError(
            f"'+' requires both operands to be text or both to be numeric; "
            f"got '{left_type!r}' and '{right_type!r}'.",
            span=span,
        )

    def _check_numeric_binop(
        self, left_type: Type, right_type: Type, span: SourceSpan, op_str: str
    ) -> Type:
        # reject operations on bare type variables.
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
            self._is_type_or_bottom(left_type, IntType, DecimalType)
            and self._is_type_or_bottom(right_type, IntType, DecimalType)
        ):
            raise AglTypeError(
                f"'{op_str}' requires numeric operands; got '{left_type!r}' and '{right_type!r}'.",
                span=span,
            )
        if isinstance(left_type, DecimalType) or isinstance(right_type, DecimalType):
            return DecimalType()
        return IntType()

    def _check_in_op(self, left_type: Type, right_type: Type, span: SourceSpan) -> Type:
        # reject operations on bare type variables.
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
        if self._is_type_or_bottom(left_type, TextType) and self._is_type_or_bottom(
            right_type, TextType
        ):
            return BoolType()
        if isinstance(right_type, ListType):
            if not is_assignable(left_type, right_type.elem):
                raise AglTypeError(
                    f"'in' element type mismatch: '{left_type!r}' in 'list[{right_type.elem!r}]'.",
                    span=span,
                )
            return BoolType()
        if isinstance(right_type, DictType) and self._is_type_or_bottom(left_type, TextType):
            return BoolType()
        if isinstance(right_type, BottomType):
            return BoolType()
        raise AglTypeError(
            f"'in' requires (text in text), (T in list[T]), or (text in dict); "
            f"got '{left_type!r}' in '{right_type!r}'.",
            span=span,
        )

    # --- unary ---

    def _check_unary_neg(self, node: UnaryNeg) -> Type:
        t = self._check_expr(node.operand, expected=None)
        # reject operations on bare type variables.
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
        # reject operations on bare type variables.
        if isinstance(expr_type, TypeVarType):
            raise AglTypeError(
                f"an abstract type variable '{expr_type.name}' cannot be tested with 'is'.",
                span=node.span,
            )
        if not isinstance(expr_type, EnumType):
            raise AglTypeError(
                f"'is' / 'is not' requires an enum-typed left-hand side; got '{expr_type!r}'.",
                span=node.span,
            )
        self._check_variant_qualification(
            qualifier=node.qualifier,
            module_qualifier=node.module_qualifier,
            enum_type=expr_type,
            span=node.span,
        )
        if node.variant not in self._env.type_table.enum_variants(expr_type):
            raise AglTypeError(
                f"Variant '{node.variant}' does not belong to enum '{expr_type.name}'.",
                span=node.span,
            )
        return BoolType()

    def _qualified_constructor_typed_call_error(self, span: SourceSpan) -> AglTypeError:
        return AglTypeError(
            "Type-qualified constructors take explicit type arguments on the type name; "
            "write 'Type[T]::Ctor(...)' instead of applying '::[T]' to the constructor.",
            span=span,
        )

    def _check_variant_qualification(
        self,
        *,
        qualifier: str | None,
        module_qualifier: Qualifier | None,
        enum_type: EnumType,
        span: SourceSpan,
    ) -> None:
        """Validate the optional enum-type qualifier on a variant reference."""
        if qualifier is not None and module_qualifier is not None:
            self._check_module_qualified_variant(module_qualifier, qualifier, enum_type, span)
        elif module_qualifier is not None:
            self._check_qualified_variant_prefix(module_qualifier, enum_type.name, enum_type, span)

    def _require_enum_owner_match(
        self,
        form: EnumOwnerForm,
        enum_type: EnumType,
        rendered_owner: str,
        span: SourceSpan,
    ) -> None:
        """Require one checked owner form to match the concrete subject enum."""
        if form.match(enum_type) is None:
            template = form.type_template
            resolved = None if template is None else template.template
            if not isinstance(resolved, EnumType):
                raise AglTypeError(
                    f"'{rendered_owner}' is not a known enum type.",
                    span=span,
                )
            raise AglTypeError(
                f"Qualifier '{rendered_owner}' resolves to enum '{resolved.name}', "
                f"but the value has enum type '{enum_type.name}'.",
                span=span,
            )

    def _check_qualified_variant_prefix(
        self,
        module_qualifier: Qualifier,
        enum_name: str,
        enum_type: EnumType,
        span: SourceSpan,
    ) -> None:
        """Validate a lone ``prefix::Variant`` qualifier."""
        if len(module_qualifier.segments) == 1:
            qualifier = module_qualifier.segments[0]
            form = self._env.resolve_unqualified_enum_owner_form(qualifier)
            handle_match = self._env.has_qualified_import_handle(module_qualifier.segments)
            if self._env.has_visible_unqualified_type_name(qualifier) and handle_match:
                raise AglTypeError(
                    f"Qualifier '{qualifier}' is both a type name and an import handle; "
                    "rename the import alias to disambiguate.",
                    span=span,
                )
            if form is not None:
                self._require_enum_owner_match(form, enum_type, qualifier, span)
                return

        self._check_module_qualified_variant(module_qualifier, enum_name, enum_type, span)

    def _check_module_qualified_variant(
        self,
        module_qualifier: Qualifier,
        enum_name: str,
        enum_type: EnumType,
        span: SourceSpan,
    ) -> None:
        """Validate a module-qualified enum-type qualifier, e.g. ``mylib::Color``."""
        kind = (
            EnumOwnerFormKind.SELF
            if not module_qualifier.segments
            else EnumOwnerFormKind.QUALIFIED_IMPORT
        )
        form = self._env.resolve_enum_owner_form(kind, enum_name, module_qualifier.segments)
        if form is not None:
            owner = (
                f"::{enum_name}"
                if not module_qualifier.segments
                else f"{'.'.join(module_qualifier.segments)}::{enum_name}"
            )
            self._require_enum_owner_match(form, enum_type, owner, span)
            return
        raise AglTypeError(
            f"'{'.'.join(module_qualifier.segments)}::{enum_name}' is not a known enum type.",
            span=span,
        )

    # --- field access ---

    def _check_field_access(self, node: FieldAccess, expected: Type | None = None) -> Type:
        obj_type = self._check_expr(node.obj, expected=None)
        # reject operations on bare type variables.
        if isinstance(obj_type, TypeVarType):
            raise AglTypeError(
                f"a value of type variable '{obj_type.name}' has no fields.",
                span=node.span,
            )
        if isinstance(obj_type, ExceptionType):
            exc_fields = self._env.type_table.exception_fields(obj_type)
            if node.field not in exc_fields:
                raise AglTypeError(
                    f"Exception type '{obj_type.name}' has no field '{node.field}'.",
                    span=node.span,
                )
            return exc_fields[node.field]
        if isinstance(obj_type, RecordType):
            record_fields = self._env.type_table.record_fields(obj_type)
            if node.field not in record_fields:
                raise AglTypeError(
                    f"Record '{obj_type.name}' has no field '{node.field}'.",
                    span=node.span,
                )
            return record_fields[node.field]
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
        # reject operations on bare type variables.
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

    def _complete_empty_literal_shape(
        self,
        expected: Type | None,
        *,
        make_type: Callable[[Type], Type],
        literal_name: str,
        span: SourceSpan,
    ) -> Type:
        """Return an empty literal's provisional container shape.

        An uncontextualized empty literal introduces an element/value variable
        that must resolve before the owning expression region closes. When a
        generic call supplies a flexible expected type, the literal also gives
        that type its container shape, leaving its element/value to be solved
        by sibling arguments or the enclosing result context.
        """
        assert self._inference_region is not None
        engine = self._inference_region.engine
        if expected is not None:
            expected = engine.zonk(expected)
        if isinstance(expected, InferenceVarType):
            element = engine.fresh()
            shape = make_type(element)
            engine.unify(
                expected,
                shape,
                engine.origin(span, role=ConstraintRole.LITERAL_ELEMENT, subject=literal_name),
            )
            return engine.zonk(expected)
        if expected is not None:
            return expected
        element = engine.fresh()
        engine.require_solved(
            element,
            engine.origin(span, role=ConstraintRole.LITERAL_ELEMENT, subject=literal_name),
        )
        return make_type(element)

    def _check_list_lit(self, node: ListLit, *, expected: Type | None) -> Type:
        if not node.elements:
            expected = self._complete_empty_literal_shape(
                expected,
                make_type=ListType,
                literal_name="empty list literal",
                span=node.span,
            )
        elem_expected = self._expected_elem_type(expected)
        if not node.elements:
            if elem_expected is None:
                raise AglTypeError(
                    "Empty list literal requires a type annotation "
                    "(e.g. 'let xs: list[text] = []').",
                    span=node.span,
                )
            return ListType(elem=elem_expected)
        if elem_expected is not None and not contains_inference_var(elem_expected):
            for elem in node.elements:
                et = self._check_expr(elem, expected=elem_expected)
                self._assert_assignable(et, elem_expected, elem.span)
            return ListType(elem=elem_expected)
        unified = self._unify_elements(node.elements, kind="List", span=node.span)
        return ListType(elem=unified)

    def _check_dict_lit(self, node: DictLit, *, expected: Type | None) -> Type:
        if not node.entries:
            expected = self._complete_empty_literal_shape(
                expected,
                make_type=DictType,
                literal_name="empty dict literal",
                span=node.span,
            )
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
        if val_expected is not None and not contains_inference_var(val_expected):
            for entry in node.entries:
                et = self._check_expr(entry.value, expected=val_expected)
                self._assert_assignable(et, val_expected, entry.span)
            return DictType(value=val_expected)
        unified = self._unify_elements(
            [entry.value for entry in node.entries], kind="Dict", span=node.span
        )
        return DictType(value=unified)

    def _unify_elements(self, elements: Sequence[Expr], *, kind: str, span: SourceSpan) -> Type:
        """Find a literal's common element type without coercing flexible variables."""
        types = [self._check_expr(e, expected=None) for e in elements]
        unified = types[0]
        for typ in types[1:]:
            provisional = self._unify_provisional_common_types(
                unified, typ, span=span, subject=f"{kind} literal elements"
            )
            if provisional is not None:
                unified = provisional
            elif is_assignable(unified, typ):
                unified = typ
            elif not is_assignable(typ, unified):
                raise AglTypeError(
                    f"{kind} literal elements have inconsistent types: "
                    f"'{unified!r}' and '{typ!r}'.",
                    span=span,
                )
        return unified

    def _unify_provisional_common_types(
        self, left: Type, right: Type, *, span: SourceSpan, subject: str
    ) -> Type | None:
        """Exactly unify provisional common-type candidates, if either has flexibles."""
        assert self._inference_region is not None
        engine = self._inference_region.engine
        left = engine.zonk(left)
        right = engine.zonk(right)
        if not (contains_inference_var(left) or contains_inference_var(right)):
            return None
        try:
            engine.unify(
                left,
                right,
                engine.origin(span, role=ConstraintRole.LITERAL_ELEMENT, subject=subject),
            )
        except InferenceError as exc:
            raise AglTypeError(
                f"{subject} have incompatible types: {exc}", span=exc.span, related=exc.related
            ) from exc
        return engine.zonk(left)

    # ------------------------------------------------------------------
    # Pattern binding helpers
    # ------------------------------------------------------------------

    def _bind_pattern_types(
        self,
        pattern: object,
        subj_type: Type,
        owner: object,
        *,
        field_name: str | None = None,
        field_names: frozenset[str] = frozenset(),
    ) -> None:
        """Type and finally classify one pattern at its matched occurrence."""
        if isinstance(pattern, WildcardPattern):
            return
        if isinstance(pattern, LiteralPattern):
            lit_type = self._check_expr(pattern.literal, expected=None)
            if not comparable_types(lit_type, subj_type, self._env.type_table):
                raise AglTypeError(
                    f"Literal pattern of type '{lit_type!r}' is incompatible with "
                    f"scrutinee of type '{subj_type!r}'.",
                    span=pattern.span,
                )
            return
        if isinstance(pattern, AsPattern):
            self._bind_pattern_types(
                pattern.pattern,
                subj_type,
                owner,
                field_name=field_name,
                field_names=field_names,
            )
            self._env.set_binding_type(pattern.node_id, subj_type)
            self._record_pattern_classification(pattern.node_id, None)
            return
        if isinstance(pattern, VarPattern):
            if field_name is None:
                self._check_top_level_bare_constructor(pattern, subj_type)
            else:
                self._check_field_bare_pattern(pattern, subj_type, field_name, field_names)
            return

        assert isinstance(pattern, ConstructorPattern), (
            f"Unexpected pattern kind: {type(pattern).__name__}"
        )
        if not isinstance(subj_type, EnumType):
            raise AglTypeError(
                f"Cannot match constructor pattern '{pattern.name}' against "
                f"non-enum type '{subj_type!r}'.",
                span=pattern.span,
            )
        self._check_variant_qualification(
            qualifier=pattern.qualifier,
            module_qualifier=pattern.module_qualifier,
            enum_type=subj_type,
            span=pattern.span,
        )
        variant_name = pattern.name
        enum_variants = self._env.type_table.enum_variants(subj_type)
        if variant_name not in enum_variants:
            raise AglTypeError(
                f"Variant '{variant_name}' does not belong to enum '{subj_type.name}'.",
                span=pattern.span,
            )
        vfields = enum_variants[variant_name]
        field_kinds = self._env.get_constructor_field_kinds(
            subj_type.name, variant_name, module_id=subj_type.module_id
        )
        assert field_kinds is not None, (
            f"field kinds not registered for {subj_type.name}.{variant_name}"
        )
        binding = bind_pattern_args(
            field_kinds,
            pattern.positional,
            pattern.named,
            call_span=pattern.span,
            context_desc=f"pattern for variant '{variant_name}'",
        )
        bound_pairs: list[tuple[str, Pattern]] = []
        constructor_field_names = frozenset(vfields)
        for (name, _), bound_pattern in zip(field_kinds, binding):
            if bound_pattern is not None:
                bound_pairs.append((name, bound_pattern))
                self._bind_pattern_types(
                    bound_pattern,
                    vfields[name],
                    pattern,
                    field_name=name,
                    field_names=constructor_field_names,
                )
        self._record_constructor_pattern_binding(pattern.node_id, tuple(bound_pairs))

    def _candidate_for_field_type(
        self, pattern: VarPattern, field_type: Type
    ) -> ConstructorRef | None:
        """Return the candidate spelling that belongs to this enum field, if any."""
        if not isinstance(field_type, EnumType):
            return None
        for candidate in self._resolved.pattern_constructor_candidates.get(pattern.node_id, ()):
            if (
                candidate.owner_module_id == field_type.module_id
                and candidate.owner_name == field_type.name
                and candidate.variant == pattern.name
            ):
                return candidate
        return None

    def _check_top_level_bare_constructor(self, pattern: VarPattern, subj_type: Type) -> None:
        """Finalize a top-level bare pattern as a nullary enum constructor."""
        if not isinstance(subj_type, EnumType):
            raise AglTypeError(
                f"Cannot match constructor pattern '{pattern.name}' against "
                f"non-enum type '{subj_type!r}'.",
                span=pattern.span,
            )
        candidate = self._candidate_for_field_type(pattern, subj_type)
        if candidate is None:
            raise AglTypeError(
                f"Variant '{pattern.name}' does not belong to enum '{subj_type.name}'.",
                span=pattern.span,
            )
        self._require_nullary_bare_constructor(pattern, subj_type)
        self._record_pattern_classification(pattern.node_id, candidate)

    def _check_field_bare_pattern(
        self,
        pattern: VarPattern,
        field_type: Type,
        field_name: str,
        field_names: frozenset[str],
    ) -> None:
        """Finalize one bare subpattern after shared field binding selected its slot."""
        candidate = self._candidate_for_field_type(pattern, field_type)
        if pattern.name == field_name:
            if candidate is not None:
                raise AglTypeError(
                    f"'{pattern.name}' is both field '{field_name}' and a constructor of its type. "
                    f"Write '{pattern.name}()' for the constructor or '_ as {pattern.name}' "
                    "for the field binder.",
                    span=pattern.span,
                )
            self._env.set_binding_type(pattern.node_id, field_type)
            self._record_pattern_classification(pattern.node_id, None)
            return
        if candidate is not None:
            self._require_nullary_bare_constructor(pattern, field_type)
            self._record_pattern_classification(pattern.node_id, candidate)
            return
        if pattern.name in field_names:
            raise AglTypeError(
                f"'{pattern.name}' names a different field, not matched field '{field_name}'. "
                f"Use '{field_name} as {pattern.name}' to rename the binding.",
                span=pattern.span,
            )
        raise AglTypeError(
            f"'{pattern.name}' is neither field '{field_name}' nor a constructor of its type.",
            span=pattern.span,
        )

    def _require_nullary_bare_constructor(self, pattern: VarPattern, enum_type: Type) -> None:
        """Require a bare constructor spelling to be a nullary matched enum variant."""
        assert isinstance(enum_type, EnumType)
        fields = self._env.type_table.enum_variants(enum_type).get(pattern.name)
        if fields is None:
            raise AglTypeError(
                f"Variant '{pattern.name}' does not belong to enum '{enum_type.name}'.",
                span=pattern.span,
            )
        if fields:
            raise AglTypeError(
                f"'{pattern.name}' has fields; write '{pattern.name}(...)' to match it.",
                span=pattern.span,
            )

    def _reconcile_provisional_pattern_references(self, case: Case, pattern: Pattern) -> None:
        """Reconcile branch references with final field-directed classifications.

        Scope gives nested bare names a temporary branch binding because their
        field type is not known yet. Several candidate spellings can share that
        temporary slot. Once checking decides which names truly bind, every
        branch reference is redirected to the unique final binder or to the
        enclosing scope when none binds.
        """
        case_scope = self._resolved.case_scopes[case.node_id]
        names: dict[str, list[VarPattern | AsPattern]] = {}
        for candidate in self._pattern_binding_candidates(pattern):
            names.setdefault(candidate.name, []).append(candidate)

        for name, candidates in names.items():
            provisional = [
                candidate
                for candidate in candidates
                if candidate.node_id in self._resolved.provisional_pattern_binders
            ]
            if not provisional:
                continue
            binders = [
                candidate
                for candidate in candidates
                if self._pattern_classifications.get(candidate.node_id) is None
            ]
            if len(binders) > 1:
                raise AglTypeError(
                    f"Name '{name}' is bound more than once in this pattern.",
                    span=binders[1].span,
                )
            provisional_ids = {candidate.node_id for candidate in provisional}
            reference_node_ids = [
                ref_node_id
                for ref_node_id, ref in self._resolved.resolution.items()
                if ref.decl_node_id in provisional_ids
            ]
            if not reference_node_ids:
                continue
            if binders:
                binder = binders[0]
                target_ref = replace(
                    self._resolved.resolution[reference_node_ids[0]],
                    decl_span=binder.span,
                    decl_node_id=binder.node_id,
                )
            else:
                target_ref = cast(BindingRef, case_scope.lookup(name))
                candidates_for_name = self._resolved.constructor_candidates.get(name, ())
                if (
                    target_ref.kind is BinderKind.constructor_binding
                    and len(candidates_for_name) != 1
                ):
                    raise AglTypeError(
                        f"'{name}' is ambiguous outside the pattern; qualify the reference.",
                        span=provisional[0].span,
                    )
            region = self._inference_region
            assert region is not None
            for ref_node_id in reference_node_ids:
                region.reconciled_resolution_originals.setdefault(
                    ref_node_id, self._resolved.resolution[ref_node_id]
                )
                region.reconciled_constructor_ref_originals.setdefault(
                    ref_node_id, self._resolved.constructor_refs.get(ref_node_id)
                )
                self._resolved.resolution[ref_node_id] = target_ref
                if target_ref.kind is BinderKind.constructor_binding:
                    constructor = self._pattern_classifications[provisional[0].node_id]
                    assert constructor is not None
                    self._resolved.constructor_refs[ref_node_id] = constructor
                else:
                    self._resolved.constructor_refs.pop(ref_node_id, None)

    @staticmethod
    def _pattern_binding_candidates(pattern: Pattern) -> tuple[VarPattern | AsPattern, ...]:
        """Return bare and explicit binding names in a pattern."""
        if isinstance(pattern, VarPattern):
            return (pattern,)
        if isinstance(pattern, AsPattern):
            return (*_Checker._pattern_binding_candidates(pattern.pattern), pattern)
        if isinstance(pattern, ConstructorPattern):
            return tuple(
                candidate
                for child in (*pattern.positional, *(field.pattern for field in pattern.named))
                for candidate in _Checker._pattern_binding_candidates(child)
            )
        return ()

    # ------------------------------------------------------------------
    # Branch unification
    # ------------------------------------------------------------------

    def _unify_branch_types(
        self,
        branch_types: list[Type],
        span: SourceSpan,
        construct: str,
    ) -> Type:
        """Find a branch common type, exactly solving provisional candidates first."""
        if self._inference_region is None:
            non_bottom = [typ for typ in branch_types if not isinstance(typ, BottomType)]
        else:
            engine = self._inference_region.engine
            non_bottom = [
                resolved
                for typ in branch_types
                if not isinstance(resolved := engine.zonk(typ), BottomType)
            ]
        if not non_bottom:
            return BottomType()
        result_type: Type = non_bottom[0]
        for branch_type in non_bottom[1:]:
            provisional = (
                self._unify_provisional_common_types(
                    result_type, branch_type, span=span, subject=f"{construct} branches"
                )
                if self._inference_region is not None
                else None
            )
            if provisional is not None:
                result_type = provisional
            elif result_type == branch_type:
                continue
            elif isinstance(result_type, IntType) and isinstance(branch_type, DecimalType):
                result_type = DecimalType()
            elif isinstance(result_type, DecimalType) and isinstance(branch_type, IntType):
                pass
            else:
                raise AglTypeError(
                    f"{construct} branches have incompatible types: "
                    f"'{result_type!r}' and '{branch_type!r}'.",
                    span=span,
                )
        return result_type

    # ------------------------------------------------------------------
    # Bool condition helper
    # ------------------------------------------------------------------

    def _require_bool_condition(self, cond_type: Type, span: SourceSpan, kw: str) -> None:
        if not self._is_type_or_bottom(cond_type, BoolType):
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

    def result(self, resolved: ModuleResolution) -> CheckedModule:
        return CheckedModule(
            resolved=resolved,
            resolution=self._resolved.resolution,
            constructor_refs=self._resolved.constructor_refs,
            qualified_constructor_refs=self._resolved.qualified_constructor_refs,
            node_types=self._node_types,
            contract_specs=self._contract_specs,
            call_sites=tuple(self._call_sites),
            warnings=tuple(self._warnings),
            type_env=self._env,
            function_signatures=self._env.all_function_signatures(),
            cast_specs=self._cast_specs,
            argument_bindings=ArgumentBindings(
                function_calls=self._function_call_bindings,
                function_param_types=self._function_call_param_types,
                constructor_calls=self._constructor_call_bindings,
                constructor_patterns=self._constructor_pattern_bindings,
                pattern_binders=frozenset(
                    node_id
                    for node_id, constructor in self._pattern_classifications.items()
                    if constructor is None
                ),
                pattern_constructors={
                    node_id: constructor
                    for node_id, constructor in self._pattern_classifications.items()
                    if constructor is not None
                },
            ),
            partial_calls=self._partial_calls,
        )


# ---------------------------------------------------------------------------
# Checked-output construction
# ---------------------------------------------------------------------------


def _check_prepared_module(
    resolved: ModuleResolution,
    capabilities: HostCapabilities,
    *,
    env: TypeEnvironment,
    module_id: ModuleId = ENTRY_ID,
    check_inhabitation: bool = True,
) -> CheckedModule:
    """Check using a prepared environment and return only finalized annotations.

    Both single-module and program callers enter here after preparing the
    namespace appropriate to their mode.  ``_Checker`` owns expression-region
    close/finalize validation, so this boundary never returns provisional
    inference state.
    """
    builder = _TypeBuilder(env, module_id=module_id)
    builder.collect(resolved.program, check_inhabitation=check_inhabitation)

    checker = _Checker(env=env, resolved=resolved, capabilities=capabilities)
    checker.check_module(resolved.program)
    checked = checker.result(resolved)
    assert_checked_module_closed(checked)
    return checked


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_module(
    resolved: ModuleResolution,
    capabilities: HostCapabilities,
    *,
    seed_env: TypeEnvironment | None = None,
) -> CheckedModule:
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
    CheckedModule
        The annotated program with type side tables and contract specs.

    Raises
    ------
    AglTypeError
        On the first static type violation (first-error abort).
    """
    env = TypeEnvironment()
    if seed_env is not None:
        env.seed_from(seed_env)
    return _check_prepared_module(resolved, capabilities, env=env)
