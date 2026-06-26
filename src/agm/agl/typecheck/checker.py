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
15. ``do-until`` — yields ``unit``; condition must be bool.
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
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, ModuleId
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    BinderKind,
    BindingRef,
    BuiltinKind,
    ConstructorRef,
    ResolvedProgram,
)
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
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
    free_type_vars,
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
    Call,
    Case,
    Cast,
    CatchClause,
    ConfigPragma,
    ConstructorPattern,
    DecimalLit,
    DictLit,
    Do,
    ElseSentinel,
    EnumDef,
    ExceptionDef,
    Expr,
    FieldAccess,
    FieldDef,
    FuncDef,
    If,
    ImportDecl,
    IndexAccess,
    IndexTarget,
    InterpSegment,
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    LiteralPattern,
    NamedArg,
    NameTarget,
    NullLit,
    ParamDecl,
    Pattern,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    StringLit,
    Template,
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
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr
from agm.agl.typecheck.env import (
    AglTypeError,
    CallSiteRecord,
    CheckedProgram,
    ConstructorSignature,
    FunctionSignature,
    GenericTypeDef,
    OutputContractSpec,
    TypeEnvironment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Built-in type names that the user may not shadow with a record/enum/alias.
_BUILTIN_TYPE_NAMES: frozenset[str] = (
    frozenset({"text", "json", "bool", "int", "decimal", "unit", "agent"})
    | BUILTIN_EXCEPTION_NAMES
    | BUILTIN_PRELUDE_TYPE_NAMES
)
_BUILTIN_NOMINAL_NAMES: frozenset[str] = BUILTIN_EXCEPTION_NAMES | BUILTIN_PRELUDE_TYPE_NAMES

# Built-in function names that user-defined defs may not shadow. Derived from
# the single source of truth in ``scope.symbols`` so the two never drift.
_BUILTIN_FUNC_NAMES: frozenset[str] = frozenset(BUILTIN_CALL_NAMES)


_IndexLike = IndexAccess | IndexTarget


def _builtin_function_signature(name: str) -> FunctionSignature | None:
    t = TypeVarType("T")
    match name:
        case "print":
            return FunctionSignature(
                params=(("value", t, False),), result=UnitType(), type_params=("T",)
            )
        case "render":
            return FunctionSignature(
                params=(("value", t, False),), result=TextType(), type_params=("T",)
            )
        case "parse_json":
            return FunctionSignature(params=(("value", TextType(), False),), result=JsonType())
        case "ask":
            return FunctionSignature(
                params=(
                    ("prompt", TextType(), False),
                    ("agent", AgentType(), True),
                    ("format", TextType(), True),
                    ("strict_json", BoolType(), True),
                    ("on_parse_error", EnumType(name="ParsePolicy", variants={}), True),
                ),
                result=t,
                type_params=("T",),
            )
        case "ask-request":
            return FunctionSignature(
                params=(("prompt", TextType(), False),),
                result=RecordType(name="AgentRequest", fields={}),
            )
        case "exec":
            return FunctionSignature(
                params=(("command", TextType(), False),),
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
            FunctionSignature(params=(("prompt", TextType(), False),), result=TextType()),
        )
    return (expected,)


def _signature_matches(actual: FunctionSignature, expected: FunctionSignature) -> bool:
    if actual.type_params != expected.type_params:
        return False
    if len(actual.params) != len(expected.params):
        return False
    for (aname, atype, adefault), (ename, etype, edefault) in zip(
        actual.params, expected.params
    ):
        if aname != ename or adefault != edefault:
            return False
        if isinstance(etype, RecordType):
            if not isinstance(atype, RecordType) or atype.name != etype.name:
                return False
        elif isinstance(etype, EnumType):
            if not isinstance(atype, EnumType) or atype.name != etype.name:
                return False
        elif atype != etype:
            return False
    if isinstance(expected.result, RecordType):
        return isinstance(actual.result, RecordType) and actual.result.name == expected.result.name
    return actual.result == expected.result


def _type_shape_matches(actual: Type, expected: Type) -> bool:
    if isinstance(expected, RecordType):
        return isinstance(actual, RecordType) and actual.fields == expected.fields
    if isinstance(expected, EnumType):
        return isinstance(actual, EnumType) and actual.variants == expected.variants
    if isinstance(expected, ExceptionType):
        return isinstance(actual, ExceptionType) and actual.fields == expected.fields
    return actual == expected


def _expected_builtin_type(name: str) -> Type | None:
    prelude = BUILTIN_PRELUDE_TYPES.get(name)
    if prelude is not None:
        return prelude
    return BUILTIN_EXCEPTIONS.get(name)


def _is_index_like(node: object) -> TypeGuard[_IndexLike]:
    return isinstance(node, (IndexAccess, IndexTarget))


# ---------------------------------------------------------------------------
# Pre-pass: collect and validate type declarations
# ---------------------------------------------------------------------------


class _TypeBuilder:
    """First pass: collect record/enum/alias declarations and validate them.

    Populates a ``TypeEnvironment`` with all user-declared types.  Raises
    ``AglTypeError`` on:
    - Duplicate type names (user vs user, or user shadowing a built-in).
    - Duplicate record fields or enum variants/fields.
    - Unknown type references inside field/variant definitions.
    - Recursive records or enums (v1: rejected).
    - Alias cycles.

    Implementation uses a two-phase approach so that type declarations are
    order-independent (design §0 "type-decl ordering"):

    Phase 1  Register all user-declared type names and alias targets.
             Empty shells (RecordType/EnumType with empty fields/variants)
             are registered immediately so that forward references within the
             same program resolve without "Unknown type" false-positives.
    Phase 2  Resolve every field and variant type, tracking the set of types
             currently under construction (``_building``) to detect direct or
             indirect recursion and report a clear diagnostic.
    """

    def __init__(self, env: TypeEnvironment, module_id: ModuleId = ENTRY_ID) -> None:
        self._env = env
        self._module_id = module_id
        # Track user-declared names → declaration span (excludes built-ins).
        self._declared: dict[str, SourceSpan] = {}
        # Index of record/enum definitions for on-demand phase-2 building.
        self._record_defs: dict[str, RecordDef] = {}
        self._enum_defs: dict[str, EnumDef] = {}
        self._exception_defs: dict[str, ExceptionDef] = {}
        # Names currently being resolved in phase 2 (cycle detection).
        self._building: set[str] = set()
        # Names that have been fully resolved in phase 2.
        self._built: set[str] = set()

    def collect(self, program: Program) -> None:
        """Scan *program* and populate ``self._env``."""
        # ----------------------------------------------------------------
        # Phase 1: Register names and empty shells (order-independent).
        # ----------------------------------------------------------------
        for item in program.body.items:
            if isinstance(item, RecordDef):
                self._register_name(item.name, item.span, is_builtin=item.is_builtin)
                self._env.unregister_name(item.name)
                self._env.register_type(
                    item.name, RecordType(name=item.name, fields={}, module_id=self._module_id)
                )
                self._record_defs[item.name] = item
            elif isinstance(item, EnumDef):
                self._register_name(item.name, item.span, is_builtin=item.is_builtin)
                self._env.unregister_name(item.name)
                self._env.register_type(
                    item.name,
                    EnumType(name=item.name, variants={}, module_id=self._module_id),
                )
                self._enum_defs[item.name] = item
            elif isinstance(item, ExceptionDef):
                self._register_name(item.name, item.span, is_builtin=item.is_builtin)
                self._env.register_type(
                    item.name,
                    ExceptionType(name=item.name, fields={}, abstract=item.base is None),
                )
                self._exception_defs[item.name] = item
            elif isinstance(item, TypeAlias):
                self._register_name(item.name, item.span)
                self._env.unregister_name(item.name)
                self._env.register_alias(
                    item.name, item.type_expr, type_params=item.type_params
                )

        # ----------------------------------------------------------------
        # Phase 2: Resolve all field/variant types with recursion detection.
        # ----------------------------------------------------------------
        for item in program.body.items:
            if isinstance(item, RecordDef):
                self._ensure_built_record(item.name)
            elif isinstance(item, EnumDef):
                self._ensure_built_enum(item.name)
            elif isinstance(item, ExceptionDef):
                self._ensure_built_exception(item.name)
            elif isinstance(item, TypeAlias):
                self._validate_alias(item)

    def collect_shells_only(self, program: Program) -> None:
        """Register only phase-1 type shells (names + empty records/enums/aliases).

        Public interface for the graph pre-pass (``graph.py``) which needs to
        register all module's type shells before resolving any body.  Equivalent
        to running only phase 1 of :meth:`collect`, without phase 2 body
        resolution.  Called by :func:`~agm.agl.typecheck.graph._build_graph_type_table`
        for each module before the cross-module topological body resolution.
        """
        for item in program.body.items:
            if isinstance(item, RecordDef):
                self._register_name(item.name, item.span, is_builtin=item.is_builtin)
                self._env.unregister_name(item.name)
                self._env.register_type(
                    item.name,
                    RecordType(name=item.name, fields={}, module_id=self._module_id),
                )
                self._record_defs[item.name] = item
            elif isinstance(item, EnumDef):
                self._register_name(item.name, item.span, is_builtin=item.is_builtin)
                self._env.unregister_name(item.name)
                self._env.register_type(
                    item.name,
                    EnumType(name=item.name, variants={}, module_id=self._module_id),
                )
                self._enum_defs[item.name] = item
            elif isinstance(item, ExceptionDef):
                self._register_name(item.name, item.span, is_builtin=item.is_builtin)
                self._env.register_type(
                    item.name,
                    ExceptionType(name=item.name, fields={}, abstract=item.base is None),
                )
                self._exception_defs[item.name] = item
            elif isinstance(item, TypeAlias):
                self._register_name(item.name, item.span)
                self._env.unregister_name(item.name)
                self._env.register_alias(item.name, item.type_expr, type_params=item.type_params)

    def ensure_built_record(self, name: str) -> None:
        """Public proxy for :meth:`_ensure_built_record`.

        Used by the graph pre-pass body-resolution step to build a named record
        type through the type builder without accessing private members.
        """
        self._ensure_built_record(name)

    def ensure_built_enum(self, name: str) -> None:
        """Public proxy for :meth:`_ensure_built_enum`.

        Used by the graph pre-pass body-resolution step to build a named enum
        type through the type builder without accessing private members.
        """
        self._ensure_built_enum(name)

    def ensure_built_exception(self, name: str) -> None:
        """Public proxy for :meth:`_ensure_built_exception`."""
        self._ensure_built_exception(name)

    def _register_name(self, name: str, span: SourceSpan, *, is_builtin: bool = False) -> None:
        if is_builtin and name not in _BUILTIN_NOMINAL_NAMES:
            raise AglTypeError(
                f"Unknown builtin type '{name}'.",
                span=span,
            )
        if name in _BUILTIN_TYPE_NAMES and not is_builtin:
            raise AglTypeError(
                f"'{name}' is a built-in type name and cannot be redeclared.",
                span=span,
            )
        if name in self._declared:
            raise AglTypeError(
                f"Type '{name}' is already declared.",
                span=span,
            )
        self._declared[name] = span

    def _ensure_built_record(self, name: str) -> None:
        """Build the record type for *name* if not already built."""
        if name in self._built:
            return
        stmt = self._record_defs[name]
        if name in self._building:
            raise AglTypeError(
                f"Record type '{name}' is directly or indirectly recursive. "
                "Recursive types are not supported in v1.",
                span=self._declared[name],
            )
        self._building.add(name)
        self._build_record(stmt)
        self._building.discard(name)
        self._built.add(name)

    def _ensure_built_enum(self, name: str) -> None:
        """Build the enum type for *name* if not already built."""
        if name in self._built:
            return
        stmt = self._enum_defs[name]
        if name in self._building:
            raise AglTypeError(
                f"Enum type '{name}' is directly or indirectly recursive. "
                "Recursive types are not supported in v1.",
                span=self._declared[name],
            )
        self._building.add(name)
        self._build_enum(stmt)
        self._building.discard(name)
        self._built.add(name)

    def _ensure_built_exception(self, name: str) -> None:
        """Build the exception type for *name* if not already built."""
        if name in self._built:
            return
        stmt = self._exception_defs[name]
        if name in self._building:
            raise AglTypeError(
                f"Exception type '{name}' is directly or indirectly recursive.",
                span=self._declared[name],
            )
        self._building.add(name)
        self._build_exception(stmt)
        self._building.discard(name)
        self._built.add(name)

    def _build_record(self, stmt: RecordDef) -> None:
        if stmt.type_params:
            self._build_generic_record(stmt)
            return
        fields: dict[str, Type] = {}
        seen_fields: dict[str, SourceSpan] = {}
        for fd in stmt.fields:
            if fd.name in seen_fields:
                raise AglTypeError(
                    f"Duplicate field '{fd.name}' in record '{stmt.name}'.",
                    span=fd.span,
                )
            seen_fields[fd.name] = fd.span
            field_type = self._resolve_field_type(fd, stmt.name)
            fields[fd.name] = field_type
        typ = RecordType(
            name=stmt.name,
            fields=fields,
            module_id=PRELUDE_ID if stmt.is_builtin else self._module_id,
        )
        self._validate_builtin_type_shape(stmt, typ)
        self._env.register_type(stmt.name, typ)

    def _build_enum(self, stmt: EnumDef) -> None:
        if stmt.type_params:
            self._build_generic_enum(stmt)
            return
        variants: dict[str, dict[str, Type]] = {}
        seen_variants: dict[str, SourceSpan] = {}
        for vd in stmt.variants:
            if vd.name in seen_variants:
                raise AglTypeError(
                    f"Duplicate variant '{vd.name}' in enum '{stmt.name}'.",
                    span=vd.span,
                )
            seen_variants[vd.name] = vd.span
            vfields: dict[str, Type] = {}
            seen_vfields: dict[str, SourceSpan] = {}
            for fd in vd.fields:
                if fd.name in seen_vfields:
                    raise AglTypeError(
                        f"Duplicate field '{fd.name}' in variant "
                        f"'{stmt.name}.{vd.name}'.",
                        span=fd.span,
                    )
                seen_vfields[fd.name] = fd.span
                vfields[fd.name] = self._resolve_field_type(fd, f"{stmt.name}.{vd.name}")
            variants[vd.name] = vfields
        typ = EnumType(
            name=stmt.name,
            variants=variants,
            module_id=PRELUDE_ID if stmt.is_builtin else self._module_id,
        )
        self._validate_builtin_type_shape(stmt, typ)
        self._env.register_type(stmt.name, typ)

    def _build_exception(self, stmt: ExceptionDef) -> None:
        fields: dict[str, Type] = {}
        if stmt.base is not None:
            if stmt.base in self._exception_defs:
                self._ensure_built_exception(stmt.base)
            base_type = self._env.resolve_named_type(stmt.base)
            if not isinstance(base_type, ExceptionType):
                raise AglTypeError(
                    f"Exception '{stmt.name}' extends unknown exception '{stmt.base}'.",
                    span=stmt.span,
                )
            fields.update(base_type.fields)
        seen_fields: dict[str, SourceSpan] = {}
        for fd in stmt.fields:
            if fd.name in fields or fd.name in seen_fields:
                raise AglTypeError(
                    f"Duplicate field '{fd.name}' in exception '{stmt.name}'.",
                    span=fd.span,
                )
            seen_fields[fd.name] = fd.span
            field_type = self._resolve_field_type(fd, stmt.name)
            fields[fd.name] = field_type
        typ = ExceptionType(
            name=stmt.name,
            fields=fields,
            abstract=stmt.base is None,
        )
        self._validate_builtin_type_shape(stmt, typ)
        self._env.register_type(stmt.name, typ)

    def _validate_builtin_type_shape(
        self, stmt: RecordDef | EnumDef | ExceptionDef, typ: Type
    ) -> None:
        if not stmt.is_builtin:
            return
        expected = _expected_builtin_type(stmt.name)
        assert expected is not None
        if not _type_shape_matches(typ, expected):
            raise AglTypeError(
                f"Builtin type '{stmt.name}' has an invalid definition.",
                span=stmt.span,
            )

    def _resolve_field_type(
        self, fd: FieldDef, owner: str, type_vars: frozenset[str] = frozenset()
    ) -> Type:
        """Resolve a field's TypeExpr to a semantic Type."""
        self._ensure_referenced_type_built(fd.type_expr)
        return self._env.resolve_type_expr(fd.type_expr, span=fd.span, type_vars=type_vars)

    def _ensure_referenced_type_built(
        self, type_expr: object, _alias_seen: frozenset[str] = frozenset()
    ) -> None:
        """Recursively ensure that all user-declared types in *type_expr* are built."""
        from agm.agl.syntax.types import AppliedT, DictT, ListT, NameT

        if isinstance(type_expr, NameT):
            name = type_expr.name
            if name in self._record_defs:
                self._ensure_built_record(name)
            elif name in self._enum_defs:
                self._ensure_built_enum(name)
            elif name in self._exception_defs:
                self._ensure_built_exception(name)
            elif self._env.get_alias_target_expr(name) is not None:
                if name not in _alias_seen:
                    self._ensure_referenced_type_built(
                        self._env.get_alias_target_expr(name),
                        _alias_seen | {name},
                    )
        elif isinstance(type_expr, ListT):
            self._ensure_referenced_type_built(type_expr.elem, _alias_seen)
        elif isinstance(type_expr, DictT):
            self._ensure_referenced_type_built(type_expr.value, _alias_seen)
        elif isinstance(type_expr, AppliedT):
            name = type_expr.name
            if name in self._record_defs:
                self._ensure_built_record(name)
            elif name in self._enum_defs:
                self._ensure_built_enum(name)
            elif name in self._exception_defs:
                self._ensure_built_exception(name)
            for arg in type_expr.args:
                self._ensure_referenced_type_built(arg, _alias_seen)

    def _build_generic_record(self, stmt: RecordDef) -> None:
        """Build a generic record type: register GenericTypeDef + ConstructorSignature."""
        type_vars = frozenset(stmt.type_params)
        fields: dict[str, Type] = {}
        seen_fields: dict[str, SourceSpan] = {}
        for fd in stmt.fields:
            if fd.name in seen_fields:
                raise AglTypeError(
                    f"Duplicate field '{fd.name}' in record '{stmt.name}'.", span=fd.span
                )
            seen_fields[fd.name] = fd.span
            self._ensure_referenced_type_built(fd.type_expr)
            field_type = self._env.resolve_type_expr(
                fd.type_expr, span=fd.span, type_vars=type_vars
            )
            fields[fd.name] = field_type
        template = RecordType(
            name=stmt.name,
            fields=fields,
            type_args=tuple(TypeVarType(p) for p in stmt.type_params),
            module_id=self._module_id,
        )
        gdef = GenericTypeDef(kind="record", type_params=stmt.type_params, template=template)
        self._env.register_generic_type(stmt.name, gdef)
        self._env.unregister_name(stmt.name)
        field_names = tuple(fields.keys())
        field_templates = tuple(fields.values())
        sig = ConstructorSignature(
            owner_name=stmt.name,
            variant=None,
            field_names=field_names,
            field_templates=field_templates,
            result_template=template,
            type_params=stmt.type_params,
        )
        self._env.register_constructor_signature(sig)

    def _build_generic_enum(self, stmt: EnumDef) -> None:
        """Build a generic enum type: register GenericTypeDef + ConstructorSignatures."""
        type_vars = frozenset(stmt.type_params)
        variants: dict[str, dict[str, Type]] = {}
        seen_variants: dict[str, SourceSpan] = {}
        for vd in stmt.variants:
            if vd.name in seen_variants:
                raise AglTypeError(
                    f"Duplicate variant '{vd.name}' in enum '{stmt.name}'.", span=vd.span
                )
            seen_variants[vd.name] = vd.span
            vfields: dict[str, Type] = {}
            seen_vfields: dict[str, SourceSpan] = {}
            for fd in vd.fields:
                if fd.name in seen_vfields:
                    raise AglTypeError(
                        f"Duplicate field '{fd.name}' in variant '{stmt.name}.{vd.name}'.",
                        span=fd.span,
                    )
                seen_vfields[fd.name] = fd.span
                self._ensure_referenced_type_built(fd.type_expr)
                vfields[fd.name] = self._env.resolve_type_expr(
                    fd.type_expr, span=fd.span, type_vars=type_vars
                )
            variants[vd.name] = vfields
        template = EnumType(
            name=stmt.name,
            variants=variants,
            type_args=tuple(TypeVarType(p) for p in stmt.type_params),
            module_id=self._module_id,
        )
        gdef = GenericTypeDef(kind="enum", type_params=stmt.type_params, template=template)
        self._env.register_generic_type(stmt.name, gdef)
        self._env.unregister_name(stmt.name)
        # Register one ConstructorSignature per variant.
        for vd in stmt.variants:
            vfields = variants[vd.name]
            field_names = tuple(vfields.keys())
            field_templates = tuple(vfields.values())
            sig = ConstructorSignature(
                owner_name=stmt.name,
                variant=vd.name,
                field_names=field_names,
                field_templates=field_templates,
                result_template=template,
                type_params=stmt.type_params,
            )
            self._env.register_constructor_signature(sig)

    def _validate_alias(self, stmt: TypeAlias) -> None:
        """Validate that the alias target resolves without cycles.

        A parameterized alias body may reference its own type parameters, so
        they are in scope as type variables during validation.
        """
        self._env.resolve_type_expr(
            stmt.type_expr, span=stmt.span, type_vars=frozenset(stmt.type_params)
        )


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

    # ------------------------------------------------------------------
    # Pre-registration of function signatures
    # ------------------------------------------------------------------

    def _preregister_funcdef(self, node: FuncDef) -> None:
        """Resolve and register the signature of a top-level ``def``."""
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
        type_vars: frozenset[str] = frozenset(node.type_params)
        params: list[tuple[str, Type, bool]] = []
        seen_required = True  # True until first defaulted param
        for p in node.params:
            pt = self._env.resolve_type_expr(p.type_expr, span=p.span, type_vars=type_vars)
            has_default = p.default is not None
            if seen_required and has_default:
                # First defaulted param: switch to "defaulted" mode
                seen_required = False
            elif not seen_required and not has_default:
                raise AglTypeError(
                    f"Parameter '{p.name}' has no default but follows a defaulted parameter. "
                    "Required parameters must come before parameters with defaults.",
                    span=p.span,
                )
            params.append((p.name, pt, has_default))

        result_type = self._env.resolve_type_expr(
            node.return_type, span=node.span, type_vars=type_vars
        )
        sig = FunctionSignature(
            params=tuple(params), result=result_type, type_params=node.type_params
        )
        if node.is_builtin:
            expected_sigs = _builtin_function_signature_alternates(node.name)
            assert expected_sigs
            if not any(_signature_matches(sig, expected_sig) for expected_sig in expected_sigs):
                raise AglTypeError(
                    f"Builtin function '{node.name}' has an invalid signature.",
                    span=node.span,
                )
        self._env.register_function_signature(node.name, sig)
        # Register by node_id so that cross-module callee signature lookups in
        # _check_declared_name_call find the correct signature even when another
        # module defines a function with the same name but a different signature.
        self._env.register_function_signature_by_node_id(node.node_id, sig)
        # Register the binding type as FunctionType (erases names/defaults).
        func_type = FunctionType(
            params=tuple(pt for _, pt, _ in params),
            result=result_type,
        )
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
        if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias, ConfigPragma)):
            return UnitType()
        if isinstance(item, AgentDecl):
            self._env.set_binding_type(item.node_id, AgentType())
            return UnitType()
        if isinstance(item, ParamDecl):
            self._check_param(item)
            return UnitType()
        if isinstance(item, ProgramDecl):
            return UnitType()
        if isinstance(item, ImportDecl):
            return UnitType()  # The graph module-system pass processes imports.
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
        assert sig is not None, f"FuncDef '{node.name}' not pre-registered"
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
            for p, (pname, ptype, has_default) in zip(node.params, sig.params):
                self._env.set_binding_type(p.node_id, ptype)
            # Check defaults against declared parameter types.
            for p, (pname, ptype, has_default) in zip(node.params, sig.params):
                if p.default is not None:
                    def_type = self._check_expr(p.default, expected=ptype)
                    self._assert_assignable(def_type, ptype, p.span)
            # Check body against declared return type.
            body_type = self._check_expr(node.body, expected=sig.result)
            if not isinstance(body_type, BottomType):
                self._assert_assignable(body_type, sig.result, node.span)
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
        if isinstance(expr, Do):
            return self._check_do(expr)
        if isinstance(expr, Try):
            return self._check_try(expr, expected=expected)
        if isinstance(expr, Raise):
            return self._infer_raise(expr)
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
        # DictLit is the last Expr union member.
        assert isinstance(expr, DictLit), f"Unexpected expression kind: {type(expr).__name__}"
        return self._check_dict_lit(expr, expected=expected)

    # --- VarRef ---

    def _check_varref(self, node: VarRef, *, expected: Type | None = None) -> Type:
        # Bare constructor reference → zero-arg construction or generic constructor as value.
        if node.node_id in self._resolved.constructor_refs:
            ctor_ref = self._resolved.constructor_refs[node.node_id]
            if ctor_ref.type_params:
                return self._check_generic_constructor_as_value(
                    ctor_ref=ctor_ref, span=node.span, expected=expected
                )
            owner = self._resolve_constructor_owner(ctor_ref, node.span)
            return self._check_constructor_as_value(
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
        if typ is None:
            raise AssertionError(
                f"Binding {ref!r} has no recorded type; checker invariant violated."
            )
        return typ

    def _check_generic_constructor_as_value(
        self,
        *,
        ctor_ref: ConstructorRef,
        span: SourceSpan,
        expected: Type | None,
    ) -> Type:
        """Handle a generic constructor used as a bare value (not in direct call position).

        For nullary variants (no fields): instantiate from the expected nominal type.
        For payload constructors: instantiate to a FunctionType from expected FunctionType.
        """
        owner_name = ctor_ref.owner_name
        variant = ctor_ref.variant
        type_params = ctor_ref.type_params
        sig = self._env.get_constructor_signature(owner_name, variant)
        # Open-imported generic constructor used as a bare value: the own-module
        # env has no signature for it; fall back to the owning module's graph
        # tables (mirrors the call-position path in _check_constructor_callee_call).
        imported_gdef: GenericTypeDef | None = None
        imported_source_name = owner_name
        if sig is None:
            # A generic constructor with no own-module signature must be open-imported
            # (the scope resolver guarantees the reference resolved to some type).
            imported = self._env.get_open_imported_generic_type(owner_name)
            assert imported is not None, (
                f"No constructor signature for {owner_name}.{variant}"
            )
            module_id, imported_source_name, imported_gdef = imported
            sig = self._env.get_ctor_sig_from_module(
                module_id, imported_source_name, variant
            )
        assert sig is not None, f"No constructor signature for {owner_name}.{variant}"

        if not sig.field_names:
            # Nullary variant: infer type args from the expected nominal enum type.
            # Match on FULL nominal identity (name AND owning module), not just the
            # name — two modules may export same-named generic enums, and borrowing
            # type args from the wrong one would mis-instantiate this constructor.
            if imported_gdef is not None:
                owner_module_id = imported_gdef.template.module_id
                nominal_name = imported_source_name
            else:
                local_gdef = self._env.get_generic_type(owner_name)
                assert local_gdef is not None, f"No generic type def for '{owner_name}'"
                owner_module_id = local_gdef.template.module_id
                nominal_name = owner_name
            subst: dict[str, Type] = {}
            if (
                expected is not None
                and isinstance(expected, EnumType)
                and expected.name == nominal_name
                and expected.module_id == owner_module_id
            ):
                for p, ta in zip(type_params, expected.type_args):
                    subst[p] = ta
            self._require_all_solved(
                type_params,
                subst,
                span=span,
                message_for=lambda p: (
                    f"Cannot infer type argument(s) for '{owner_name}': "
                    "no contextual type available. "
                    f"Add a type annotation (e.g. 'let x: {owner_name}[…] = …')."
                ),
            )
            concrete_args = tuple(subst[p] for p in type_params)
            concrete_type = (
                self._env.instantiate_from_gdef(
                    imported_source_name, imported_gdef, concrete_args
                )
                if imported_gdef is not None
                else self._env.instantiate_nominal(owner_name, concrete_args)
            )
            return self._check_constructor_call(
                owner=concrete_type, variant=variant, args=(), span=span
            )
        else:
            # Payload constructor as value: produce a FunctionType.
            subst = {}
            if expected is not None and isinstance(expected, FunctionType):
                # Match field templates against expected function params.
                for ft, ep in zip(sig.field_templates, expected.params):
                    self._match(ft, ep, subst, span=span, challenge=False)
                self._match(sig.result_template, expected.result, subst, span=span, challenge=False)
            self._require_all_solved(
                type_params,
                subst,
                span=span,
                message_for=lambda p: (
                    f"Cannot infer type argument(s) for constructor '{owner_name}': "
                    "no contextual type available. "
                    f"Add a type annotation (e.g. 'let f: ({owner_name}[…]) = …')."
                ),
            )
            concrete_params = tuple(substitute(ft, subst) for ft in sig.field_templates)
            concrete_result = substitute(sig.result_template, subst)
            return FunctionType(params=concrete_params, result=concrete_result)

    def _check_generic_constructor_call(
        self,
        *,
        node_type_args: tuple[object, ...],
        ctor_ref: ConstructorRef,
        named_args: tuple[NamedArg, ...],
        span: SourceSpan,
        expected: Type | None,
        sig: ConstructorSignature | None = None,
        gdef: GenericTypeDef | None = None,
    ) -> Type:
        """Check a generic constructor call (with inference or explicit type args).

        ``sig`` and ``gdef`` may be supplied by cross-module callers that already
        looked up these from the graph tables; when ``None``, they are looked up
        from the own-module env (the default path for same-module generic calls).
        """
        owner_name = ctor_ref.owner_name
        variant = ctor_ref.variant
        type_params = ctor_ref.type_params
        if sig is None:
            sig = self._env.get_constructor_signature(owner_name, variant)
        assert sig is not None, (
            f"No constructor signature for {owner_name}.{variant!r}; "
            "scope resolver should have caught unknown variants."
        )

        subst: dict[str, Type] = {}

        if node_type_args:
            # Explicit type arguments path.
            if len(node_type_args) != len(type_params):
                raise AglTypeError(
                    f"'{owner_name}' requires {len(type_params)} type argument(s), "
                    f"but {len(node_type_args)} were supplied.",
                    span=span,
                )
            for p, ta in zip(type_params, node_type_args):
                resolved_arg = self._env.resolve_type_expr(
                    ta, span=span, type_vars=self._current_type_vars
                )
                subst[p] = resolved_arg
        else:
            # Inference path: match field arg types against field templates.
            # A hint from the expected result type lets annotation-requiring
            # literals (e.g. an empty list field on `let b: Box[int] = Box(items: [])`)
            # resolve before the field arguments are individually checked.
            hint = self._result_hint(sig.result_template, expected, span=span)
            named_by_field = {na.name: na for na in named_args}
            for field_name, field_template in zip(sig.field_names, sig.field_templates):
                if field_name in named_by_field:
                    na = named_by_field[field_name]
                    self._infer_arg(field_template, na.value, subst, hint, span=na.span)
            # Fill remaining unsolved from expected result type.
            if expected is not None:
                self._match(sig.result_template, expected, subst, span=span, challenge=False)
            # Verify all type params were solved.
            self._require_all_solved(
                type_params,
                subst,
                span=span,
                message_for=lambda p: (
                    f"Cannot infer type argument '{p}' for constructor '{owner_name}'; "
                    f"supply it explicitly via '{owner_name}::[…]' or add a type annotation."
                ),
            )

        # Instantiate the nominal type.
        concrete_args = tuple(subst[p] for p in type_params)
        if gdef is not None:
            concrete_type = self._env.instantiate_from_gdef(
                owner_name, gdef, concrete_args, span=span
            )
        else:
            concrete_type = self._env.instantiate_nominal(owner_name, concrete_args, span=span)
        # Validate the constructor call.
        return self._check_constructor_call(
            owner=concrete_type, variant=variant, args=named_args, span=span
        )

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
                return self._check_print_call(node)
            if kind == BuiltinKind.RENDER:
                return self._check_render_call(node)
            if kind == BuiltinKind.ASK:
                return self._check_ask_call(node, expected=expected)
            if kind == BuiltinKind.ASK_REQUEST:
                return self._check_ask_request_call(node)
            if kind == BuiltinKind.PARSE_JSON:
                return self._check_parse_json_call(node)
            # EXEC
            return self._check_exec_call(node, expected=expected)

        # Constructor call?
        if (
            isinstance(node.callee, VarRef)
            and node.callee.node_id in self._resolved.constructor_refs
        ):
            return self._check_constructor_callee_call(node, expected=expected)
        if (
            isinstance(node.callee, FieldAccess)
            and node.callee.node_id in self._resolved.qualified_constructor_refs
        ):
            return self._check_qualified_constructor_callee_call(node, expected=expected)

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
                return self._check_cross_module_constructor_call(
                    node, callee_ref, expected=expected
                )

        # Value call (lambda or higher-order).
        return self._check_value_call(node, expected=expected)

    # --- print ---

    def _check_print_call(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError(
                "print() requires exactly one positional argument.",
                span=node.span,
            )
        self._check_expr(node.args[0], expected=None)
        return UnitType()

    # --- render ---

    def _check_render_call(self, node: Call) -> Type:
        if len(node.args) != 1:
            raise AglTypeError(
                "render() requires exactly one positional argument.",
                span=node.span,
            )
        allowed = {"pretty", "quote_strings"}
        for named in node.named_args:
            if named.name not in allowed:
                raise AglTypeError(
                    f"render() got unknown named argument {named.name!r}.",
                    span=named.span,
                )
            option_type = self._check_expr(named.value, expected=BoolType())
            self._assert_assignable(option_type, BoolType(), named.value.span)
        self._check_expr(node.args[0], expected=None)
        return TextType()

    # --- parse_json ---

    def _check_parse_json_call(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError(
                "parse_json() requires exactly one positional text argument.",
                span=node.span,
            )
        arg_type = self._check_expr(node.args[0], expected=TextType())
        self._assert_assignable(arg_type, TextType(), node.args[0].span)
        return JsonType()

    # --- ask ---

    _ASK_ALLOWED_NAMED_ARGS: frozenset[str] = frozenset(
        {"agent", "format", "strict_json", "on_parse_error"}
    )

    def _check_ask_call(self, node: Call, *, expected: Type | None) -> Type:
        # Target type: explicit type argument overrides context.
        explicit = self._resolve_explicit_target(node, "ask")
        target_type: Type = explicit if explicit is not None else (
            expected if expected is not None else TextType()
        )
        self._reject_type_var_target(target_type, node.span)

        # D9: reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot parse agent output into a function/agent value.",
                span=node.span,
            )

        named = {na.name: na for na in node.named_args}

        # Reject unknown named args.
        for arg_name, na in named.items():
            if arg_name not in self._ASK_ALLOWED_NAMED_ARGS:
                raise AglTypeError(
                    f"ask: unknown argument '{arg_name}'.",
                    span=na.span,
                )

        # Prompt (first positional arg — reject extra positionals).
        if not node.args:
            raise AglTypeError("ask() requires a prompt argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError(
                "ask: too many positional arguments (expected 1).",
                span=node.span,
            )
        prompt_type = self._check_expr(node.args[0], expected=TextType())
        self._assert_assignable(prompt_type, TextType(), node.args[0].span)

        # agent: named arg.
        if "agent" in named:
            agent_na = named["agent"]
            agent_type = self._check_expr(agent_na.value, expected=None)
            if not isinstance(agent_type, AgentType):
                raise AglTypeError(
                    f"'agent:' argument must be of type agent; got '{agent_type!r}'.",
                    span=agent_na.span,
                )
        else:
            if not self._caps.has_default_agent:
                raise AglTypeError(
                    "No default agent is configured; the built-in 'ask' call "
                    "cannot run. Register a default agent, or run via `agm exec`, "
                    "which provides one.",
                    span=node.span,
                )

        if isinstance(target_type, UnitType):
            self._reject_unit_parse_options(named, callee="ask")
            codec_name = "none"
            parse_policy_str = "default"
        else:
            codec_name, effective_strict, parse_policy_str = self._resolve_parse_options(
                node, target_type, named
            )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name=codec_name,
                strict_json=effective_strict,
            )
            self._contract_specs[node.node_id] = spec
        self._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee="ask",
                target_type=target_type,
                codec_name=codec_name,
                parse_policy=parse_policy_str,
                line=node.span.start_line,
                col=node.span.start_col,
            )
        )
        return target_type

    # --- ask-request ---

    def _check_ask_request_call(self, node: Call) -> Type:
        """Type-check ``ask-request(prompt, ...)`` — the side-effect-free twin of ``ask``.

        Like ``ask`` it builds an output contract from a target type and the
        parse-shaping named args (``format`` / ``strict_json`` /
        ``on_parse_error``), and accepts an ``agent:`` named arg.  But it never
        dispatches to the agent: it yields the ``AgentRequest`` record that the
        corresponding ``ask`` call would pass to ``AgentRegistry.dispatch`` on
        its first attempt.

        The target type is taken from the explicit type argument
        (``ask-request::[Review](...)``) when present, and defaults to ``text``
        otherwise (``ask-request(...)``).  Because the result type is fixed to
        ``AgentRequest``, the contextual ``expected`` type is ignored — unlike
        ``ask``, the target type is not inferred from context.
        """
        agent_request_type = self._env.get_type("AgentRequest")
        assert agent_request_type is not None, "AgentRequest prelude type missing"

        # Target type: explicit type argument, else text default.
        explicit = self._resolve_explicit_target(node, "ask-request")
        target_type = explicit if explicit is not None else TextType()
        self._reject_type_var_target(target_type, node.span)

        # D9: reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot build an output contract for a function/agent target.",
                span=node.span,
            )

        named = {na.name: na for na in node.named_args}

        # Reject unknown named args (same set as ask, minus none — agent is
        # accepted even though it only labels the request, never dispatches).
        for arg_name, na in named.items():
            if arg_name not in self._ASK_ALLOWED_NAMED_ARGS:
                raise AglTypeError(
                    f"ask-request: unknown argument '{arg_name}'.",
                    span=na.span,
                )

        # Prompt (first positional arg — reject extra positionals).
        if not node.args:
            raise AglTypeError(
                "ask-request() requires a prompt argument.", span=node.span
            )
        if len(node.args) > 1:
            raise AglTypeError(
                "ask-request: too many positional arguments (expected 1).",
                span=node.span,
            )
        prompt_type = self._check_expr(node.args[0], expected=TextType())
        self._assert_assignable(prompt_type, TextType(), node.args[0].span)

        # agent: named arg — same validation as ask (must be an agent value).
        if "agent" in named:
            agent_na = named["agent"]
            agent_type = self._check_expr(agent_na.value, expected=None)
            if not isinstance(agent_type, AgentType):
                raise AglTypeError(
                    f"'agent:' argument must be of type agent; got '{agent_type!r}'.",
                    span=agent_na.span,
                )

        # Build the same output contract spec an ``ask`` call would, so the
        # materialized contract (and thus the returned request) matches exactly.
        if isinstance(target_type, UnitType):
            self._reject_unit_parse_options(named, callee="ask-request")
            codec_name = "none"
            parse_policy_str = "default"
        else:
            codec_name, effective_strict, parse_policy_str = self._resolve_parse_options(
                node, target_type, named
            )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name=codec_name,
                strict_json=effective_strict,
            )
            self._contract_specs[node.node_id] = spec
        self._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee="ask-request",
                target_type=target_type,
                codec_name=codec_name,
                parse_policy=parse_policy_str,
                line=node.span.start_line,
                col=node.span.start_col,
            )
        )
        return agent_request_type

    # --- exec ---

    _EXEC_ALLOWED_NAMED_ARGS: frozenset[str] = frozenset(
        {"format", "strict_json", "on_parse_error"}
    )

    def _check_exec_call(self, node: Call, *, expected: Type | None) -> Type:
        if not self._caps.supports_shell_exec:
            raise AglTypeError(
                "The host does not support 'exec' (shell) calls.", span=node.span
            )

        exec_result_type = self._env.get_type("ExecResult")
        target_type: Type
        # Explicit type argument overrides context.
        explicit = self._resolve_explicit_target(node, "exec")
        if explicit is not None:
            target_type = explicit
        elif expected is not None:
            target_type = expected
        else:
            assert exec_result_type is not None
            target_type = exec_result_type
        self._reject_type_var_target(target_type, node.span)

        # D9: reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot parse exec output into a function/agent value.",
                span=node.span,
            )

        named = {na.name: na for na in node.named_args}

        # Reject unknown named args (exec has no 'agent:' argument).
        for arg_name, na in named.items():
            if arg_name not in self._EXEC_ALLOWED_NAMED_ARGS:
                raise AglTypeError(
                    f"exec: unknown argument '{arg_name}'.",
                    span=na.span,
                )

        # Command (first positional arg — reject extra positionals).
        if not node.args:
            raise AglTypeError("exec() requires a command argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError(
                "exec: too many positional arguments (expected 1).",
                span=node.span,
            )
        cmd_type = self._check_expr(node.args[0], expected=TextType())
        self._assert_assignable(cmd_type, TextType(), node.args[0].span)

        # Determine codec.
        is_exec_result = exec_result_type is not None and target_type == exec_result_type
        parse_policy_str = "default"

        if is_exec_result:
            # Structured form: reject parse-shaping options — they are meaningless
            # when exec returns the raw ExecResult record.
            for shaping_arg in ("format", "strict_json", "on_parse_error"):
                if shaping_arg in named:
                    raise AglTypeError(
                        f"exec returning ExecResult does not accept '{shaping_arg}'; "
                        "those options apply only when parsing stdout into a typed value.",
                        span=named[shaping_arg].span,
                    )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name="text",
                strict_json=None,
                structured_exec=True,
            )
        else:
            codec_name, effective_strict, parse_policy_str = self._resolve_parse_options(
                node, target_type, named
            )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name=codec_name,
                strict_json=effective_strict,
            )
        self._contract_specs[node.node_id] = spec
        self._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee="exec",
                target_type=target_type,
                codec_name=spec.codec_name,
                parse_policy=parse_policy_str,
                line=node.span.start_line,
                col=node.span.start_col,
            )
        )
        return target_type

    # --- shared explicit-target resolver for D3 ---

    def _resolve_explicit_target(self, node: Call, builtin_name: str) -> Type | None:
        """Resolve the explicit type argument of an ask/ask-request/exec call.

        Returns the resolved ``Type`` when ``node.type_args`` is non-empty, or
        ``None`` when there are no explicit type arguments (caller falls back to
        its contextual/default target logic).

        Raises ``AglTypeError`` when more than one type argument is provided
        (arity error). The D3 type-variable guard is applied by the caller to
        the *final* target type (see :meth:`_reject_type_var_target`), so it
        covers both the explicit and the contextual/inferred target paths.
        """
        if not node.type_args:
            return None
        if len(node.type_args) > 1:
            raise AglTypeError(
                f"{builtin_name} expects at most one explicit type argument; "
                f"got {len(node.type_args)}.",
                span=node.span,
            )
        return self._env.resolve_type_expr(
            node.type_args[0], span=node.span, type_vars=self._current_type_vars
        )

    def _reject_type_var_target(self, target_type: Type, span: SourceSpan) -> None:
        """D3: an ask/exec/ask-request target type may not contain a type variable.

        Applied to the final resolved target — whether it came from an explicit
        ``::[…]`` argument or was inferred from the contextual expected type
        (e.g. a generic ``def``'s return type) — so a type variable never reaches
        codec selection or schema generation (which cannot serialise one).
        """
        if contains_type_var(target_type):
            tv = next(iter(free_type_vars(target_type)))
            raise AglTypeError(
                f"agent/exec target type cannot contain a type variable ('{tv}').",
                span=span,
            )

    # --- shared parse-option handling (ask / exec) ---

    def _reject_unit_parse_options(
        self, named: dict[str, NamedArg], *, callee: str
    ) -> None:
        for option in ("format", "strict_json", "on_parse_error"):
            if option in named:
                raise AglTypeError(
                    f"{callee} returning unit does not accept '{option}'; "
                    "unit responses are ignored and have no output contract.",
                    span=named[option].span,
                )

    def _resolve_parse_options(
        self, node: Call, target_type: Type, named: dict[str, NamedArg]
    ) -> tuple[str, bool | None, str]:
        """Resolve the format/strict_json/on_parse_error named args shared by ask and exec.

        Returns ``(codec_name, effective_strict, parse_policy_str)``.
        """
        if "format" in named:
            format_na = named["format"]
            fmt_expr = format_na.value
            if not isinstance(fmt_expr, StringLit):
                raise AglTypeError(
                    "'format' must be a static text literal (codec name).",
                    span=format_na.span,
                )
            codec_name = self._validate_format_option(fmt_expr.value, target_type, format_na.span)
        else:
            codec_name = self._select_codec(target_type, node.span)

        strict_json: bool | None = None
        if "strict_json" in named:
            sj_na = named["strict_json"]
            sj_expr = sj_na.value
            if not isinstance(sj_expr, BoolLit):
                raise AglTypeError(
                    "'strict_json' must be a static bool literal.",
                    span=sj_na.span,
                )
            if codec_name != "json":
                raise AglTypeError(
                    f"'strict_json' is only valid when the codec is 'json'; "
                    f"the selected codec for this call is '{codec_name}'.",
                    span=sj_na.span,
                )
            strict_json = sj_expr.value

        parse_policy_str = "default"
        if "on_parse_error" in named:
            ope_na = named["on_parse_error"]
            parse_policy_str = self._extract_parse_policy_str(ope_na.value, ope_na.span)
            # Warn: no-op on text target.
            if isinstance(target_type, TextType):
                self._warnings.append(
                    Diagnostic(
                        message=(
                            "'on_parse_error' has no effect on a text target: a text "
                            "result never fails parsing, so the policy can never fire."
                        ),
                        line=node.span.start_line,
                        column=node.span.start_col,
                        end_line=node.span.end_line,
                        end_column=node.span.end_col,
                        severity="warning",
                    )
                )

        effective_strict = strict_json if codec_name == "json" else None
        return codec_name, effective_strict, parse_policy_str

    # --- on_parse_error policy extraction ---

    def _extract_parse_policy_str(self, arg: Expr, span: SourceSpan) -> str:
        """Extract a static ``ParsePolicy`` constructor as an inventory string."""
        if isinstance(arg, Call) and isinstance(arg.callee, FieldAccess):
            qualifier = arg.callee.obj
            if not (isinstance(qualifier, VarRef) and qualifier.name == "ParsePolicy"):
                raise AglTypeError(
                    "'on_parse_error' must be a static ParsePolicy constructor "
                    "(Abort or Retry(n: <int>)).",
                    span=span,
                )
            return self._extract_parse_policy_variant(arg.callee.field, arg.named_args, span)
        if isinstance(arg, Call) and isinstance(arg.callee, VarRef):
            return self._extract_parse_policy_variant(arg.callee.name, arg.named_args, span)
        # Bare VarRef: ``Abort`` (no parens) is also accepted as abort policy.
        if isinstance(arg, VarRef) and arg.name == "Abort":
            return "abort"
        # Bare FieldAccess: ``ParsePolicy.Abort`` (no parens) is also accepted.
        if isinstance(arg, FieldAccess) and arg.field == "Abort":
            qualifier = arg.obj
            if isinstance(qualifier, VarRef) and qualifier.name == "ParsePolicy":
                return "abort"
        raise AglTypeError(
            "'on_parse_error' must be a static ParsePolicy constructor "
            "(Abort or Retry(n: <int>)).",
            span=span,
        )

    def _extract_parse_policy_variant(
        self, name: str, named_args: tuple[NamedArg, ...], span: SourceSpan
    ) -> str:
        """Extract Abort or Retry variant from ParsePolicy call."""
        if name == "Abort":
            if named_args:
                raise AglTypeError(
                    "'on_parse_error' must be a static ParsePolicy constructor "
                    "(Abort or Retry(n: <int>)).",
                    span=span,
                )
            return "abort"
        if name == "Retry":
            n_arg = next((a for a in named_args if a.name == "n"), None)
            if n_arg is None or not isinstance(n_arg.value, IntLit):
                raise AglTypeError(
                    "'on_parse_error' must be a static ParsePolicy constructor "
                    "(Abort or Retry(n: <int>)).",
                    span=span,
                )
            return f"retry[{n_arg.value.value}]"
        raise AglTypeError(
            "'on_parse_error' must be a static ParsePolicy constructor "
            "(Abort or Retry(n: <int>)).",
            span=span,
        )

    # --- codec helpers ---

    def _select_codec(self, target_type: Type, span: SourceSpan) -> str:
        kind = target_type.kind
        for codec_name, supported_kinds in self._caps.codec_kinds.items():
            if kind in supported_kinds:
                return codec_name
        raise AglTypeError(
            f"No registered codec supports type '{target_type!r}'. "
            f"(Type kind '{kind}' is not handled by any available codec.)",
            span=span,
        )

    def _validate_format_option(
        self, format_name: str, target_type: Type, span: SourceSpan
    ) -> str:
        if format_name not in self._caps.codec_kinds:
            known = sorted(self._caps.codec_kinds)
            raise AglTypeError(
                f"Unknown codec '{format_name}' in 'format' option. "
                f"Known codecs: {known}.",
                span=span,
            )
        supported_kinds = self._caps.codec_kinds[format_name]
        if target_type.kind not in supported_kinds:
            raise AglTypeError(
                f"Codec '{format_name}' does not support target type '{target_type!r}'. "
                f"(Supported kinds: {sorted(supported_kinds)}.)",
                span=span,
            )
        return format_name

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

    def _check_call_args(
        self,
        params: tuple[tuple[str, Type, bool], ...],
        node: Call,
        func_name: str,
    ) -> None:
        """Check positional and named arguments against a concrete parameter list.

        Performs: positional arity check, positional arg type-check+assignability,
        named-arg matching (unknown / both-positional-and-named / duplicate-named),
        and required-arg presence check.  Raises ``AglTypeError`` on any violation.
        The caller is responsible for re-checking positional arg expressions during
        type-variable inference before calling this helper with the substituted params.
        """
        # Positional arity check.
        if len(node.args) > len(params):
            raise AglTypeError(
                f"Too many positional arguments: '{func_name}' takes "
                f"{len(params)} parameter(s), got {len(node.args)}.",
                span=node.span,
            )

        # Check positional args.
        positional_filled: set[str] = set()
        for i, arg in enumerate(node.args):
            pname, ptype, _ = params[i]
            at = self._check_expr(arg, expected=ptype)
            self._assert_assignable(at, ptype, arg.span)
            positional_filled.add(pname)

        # Check named args.
        named_filled: set[str] = set()
        for na in node.named_args:
            match_param: tuple[str, Type, bool] | None = None
            for p in params:
                if p[0] == na.name:
                    match_param = p
                    break
            if match_param is None:
                raise AglTypeError(
                    f"Unknown parameter '{na.name}' in call to '{func_name}'.",
                    span=na.span,
                )
            mp_name, mp_type, _ = match_param
            if mp_name in positional_filled:
                raise AglTypeError(
                    f"Parameter '{mp_name}' supplied both positionally and by name.",
                    span=na.span,
                )
            if mp_name in named_filled:
                raise AglTypeError(
                    f"Duplicate named argument '{mp_name}' in call to '{func_name}'.",
                    span=na.span,
                )
            at = self._check_expr(na.value, expected=mp_type)
            self._assert_assignable(at, mp_type, na.span)
            named_filled.add(mp_name)

        # Check all required params are supplied.
        for pname, _, has_def in params:
            if not has_def and pname not in positional_filled and pname not in named_filled:
                raise AglTypeError(
                    f"Missing required argument '{pname}' in call to '{func_name}'.",
                    span=node.span,
                )

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
            # Infer from positional args.
            for i, arg in enumerate(node.args):
                if i >= len(sig.params):
                    break
                self._infer_arg(sig.params[i][1], arg, subst, hint, span=arg.span)
            # Infer from named args.
            for na in node.named_args:
                for pname, ptype, _ in sig.params:
                    if pname == na.name:
                        self._infer_arg(ptype, na.value, subst, hint, span=na.span)
                        break
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

        # Substitute to get the concrete signature.
        sub_params = tuple((n, substitute(pt, subst), hd) for n, pt, hd in sig.params)
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
            return self._check_value_call(node, expected=expected)

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

        if not branch_types:
            return expected if expected is not None else TextType()

        return self._unify_branch_types(branch_types, node.span, "Case expression")

    # --- do ---

    def _check_do(self, node: Do) -> Type:
        self._check_expr(node.body, expected=None)
        cond_type = self._check_expr(node.condition, expected=None)
        self._require_bool_condition(cond_type, node.condition.span, "do-until")
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
            resolved = self._env.get_type(clause.exc_type)
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
            return self._check_qualified_constructor_as_value(
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

    # --- constructor ---

    def _resolve_constructor_owner(
        self, ref: ConstructorRef, span: SourceSpan
    ) -> RecordType | EnumType | ExceptionType:
        """Resolve the owner type for a constructor ref.

        Falls back to the unqualified import map for cross-module types that
        are open-imported but not registered in the local environment.
        """
        owner: Type | None = self._env.get_type(ref.owner_name)
        if owner is None:
            owner = self._env.resolve_named_type(ref.owner_name)
        if not isinstance(owner, (RecordType, EnumType, ExceptionType)):
            raise AglTypeError(
                f"'{ref.owner_name}' is not a known constructible type.",
                span=span,
            )
        return owner

    def _check_qualified_constructor_as_value(
        self,
        *,
        owner_name: str,
        variant: str,
        owner_module_id: ModuleId | None,
        span: SourceSpan,
        expected: Type | None,
    ) -> Type:
        """Type a qualified constructor (``Owner.variant``) used in value position."""
        gdef = self._env.get_generic_type(owner_name)
        if gdef is not None:
            # owner_decl_node_id is unused on the as-value path (only owner_name,
            # variant, and type_params are consumed); pass the 0 placeholder.
            ctor_ref = ConstructorRef(
                owner_name=owner_name,
                variant=variant,
                owner_decl_node_id=0,
                type_params=gdef.type_params,
            )
            return self._check_generic_constructor_as_value(
                ctor_ref=ctor_ref, span=span, expected=expected
            )
        enum_type = self._resolve_qualified_enum_owner(
            owner_name, variant, span, owner_module_id=owner_module_id
        )
        return self._check_constructor_as_value(
            owner=enum_type, variant=variant, span=span
        )

    def _resolve_qualified_enum_owner(
        self,
        owner_name: str,
        variant: str,
        span: SourceSpan,
        *,
        owner_module_id: ModuleId | None = None,
    ) -> EnumType:
        """Resolve a non-generic qualified constructor's owner to a validated enum.

        Scope records ``Owner.member`` for any declared type name without
        checking enum-ness or variant existence, so both are validated here.
        When ``owner_module_id`` is given (cross-module constructor ref), look up
        directly in the graph type table instead of the unqualified import map.
        """
        if owner_module_id is not None:
            enum_type = self._env.resolve_type_by_module_id(owner_module_id, owner_name)
        else:
            enum_type = self._env.resolve_named_type(owner_name)
        if not isinstance(enum_type, EnumType):
            raise AglTypeError(
                f"'{owner_name}' is not a known enum type.",
                span=span,
            )
        if variant not in enum_type.variants:
            raise AglTypeError(
                f"Variant '{variant}' does not exist in enum '{owner_name}'.",
                span=span,
            )
        return enum_type

    def _check_constructor_as_value(
        self,
        *,
        owner: RecordType | EnumType | ExceptionType,
        variant: str | None,
        span: SourceSpan,
    ) -> Type:
        """Type a non-generic constructor used in value position (not directly called).

        A constructor with fields becomes a ``FunctionType`` (field types →
        owner type) so it can be passed around and called positionally.  A
        zero-field record or nullary variant keeps its bare nominal value (a
        zero-arg construction).  An exception constructor is rejected — its
        construction has special trace-id semantics and is out of scope as a
        first-class value.
        """
        if isinstance(owner, ExceptionType):
            raise AglTypeError(
                "Exception constructors cannot be used as a first-class value; "
                "construct the exception directly (e.g. `Abort(message: ...)`).",
                span=span,
            )
        if isinstance(owner, EnumType):
            assert variant is not None, "variant is required for EnumType"
            fields = owner.variants[variant]
        else:
            fields = owner.fields
        if fields:
            params = tuple(fields.values())
            return FunctionType(params=params, result=owner)
        return self._check_constructor_call(
            owner=owner, variant=variant, args=(), span=span
        )

    def _resolve_qualified_constructor_and_call(
        self,
        *,
        owner_name: str,
        variant: str,
        owner_module_id: ModuleId | None = None,
        args: tuple[NamedArg, ...],
        span: SourceSpan,
        expected: Type | None = None,
        type_args: tuple[object, ...] = (),
    ) -> Type:
        """Validate and dispatch a qualified constructor (EnumName.variant)."""
        # Check if this is a generic enum type.
        gdef = (
            self._env.get_generic_type_from_module(owner_module_id, owner_name)
            if owner_module_id is not None
            else self._env.get_generic_type(owner_name)
        )
        if gdef is not None:
            sig = (
                self._env.get_ctor_sig_from_module(owner_module_id, owner_name, variant)
                if owner_module_id is not None
                else self._env.get_constructor_signature(owner_name, variant)
            )
            assert sig is not None, (
                f"Generic enum '{owner_name}' has no constructor signature for '{variant}'"
            )
            ctor_ref = ConstructorRef(
                owner_name=owner_name,
                variant=variant,
                owner_decl_node_id=0,
                type_params=gdef.type_params,
            )
            return self._check_generic_constructor_call(
                node_type_args=type_args,
                ctor_ref=ctor_ref,
                named_args=args,
                span=span,
                expected=expected,
                sig=sig,
                gdef=gdef,
            )
        if type_args:
            raise AglTypeError(
                f"'{owner_name}.{variant}' is not a generic constructor and does not accept "
                "type arguments.",
                span=span,
            )
        enum_type = self._resolve_qualified_enum_owner(
            owner_name, variant, span, owner_module_id=owner_module_id
        )
        return self._check_constructor_call(
            owner=enum_type, variant=variant, args=args, span=span
        )

    def _check_cross_module_constructor_call(
        self,
        node: Call,
        callee_ref: BindingRef,
        *,
        expected: Type | None = None,
    ) -> Type:
        """Handle a Call whose callee is a cross-module constructor VarRef.

        Used when the callee is a qualified VarRef like ``modA::Foo`` that
        resolved to a ``constructor_binding`` in a non-entry module.
        """
        assert isinstance(node.callee, VarRef)
        if node.args:
            raise AglTypeError(
                "Constructor arguments must be named; positional arguments are not allowed.",
                span=node.span,
            )
        # Cross-module generic constructor — both explicit (lib::Box[int](v:1)) and
        # inferred (lib::Box(value:1)) routes go here.
        gdef = self._env.get_generic_type_from_module(callee_ref.module_id, callee_ref.name)
        if gdef is not None:
            ctor_sig = self._env.get_ctor_sig_from_module(
                callee_ref.module_id, callee_ref.name, None
            )
            assert ctor_sig is not None, (
                f"GenericTypeDef '{callee_ref.name}' in '{callee_ref.module_id.dotted()}' "
                "has no constructor signature in the graph table"
            )
            ctor_ref = ConstructorRef(
                owner_name=callee_ref.name,
                variant=None,
                owner_decl_node_id=callee_ref.decl_node_id,
                type_params=gdef.type_params,
            )
            return self._check_generic_constructor_call(
                node_type_args=node.type_args,
                ctor_ref=ctor_ref,
                named_args=node.named_args,
                span=node.span,
                expected=expected,
                sig=ctor_sig,
                gdef=gdef,
            )
        if node.type_args:
            raise AglTypeError(
                f"'{callee_ref.name}' is not a generic type and does not accept "
                "type arguments.",
                span=node.span,
            )
        owner_type = self._env.resolve_type_by_module_id(callee_ref.module_id, callee_ref.name)
        # Invariant: scope resolver only sets constructor_binding for RecordDef/EnumDef,
        # which are always RecordType/EnumType in the graph type table.
        assert isinstance(owner_type, (RecordType, EnumType)), (
            f"constructor_binding for '{callee_ref.name}' in "
            f"'{callee_ref.module_id.dotted()}' resolved to {type(owner_type).__name__}"
        )
        return self._check_constructor_call(
            owner=owner_type, variant=None, args=node.named_args, span=node.span
        )

    def _check_constructor_callee_call(self, node: Call, *, expected: Type | None = None) -> Type:
        """Handle a Call whose callee is an unqualified constructor VarRef."""
        assert isinstance(node.callee, VarRef)
        ctor_ref = self._resolved.constructor_refs[node.callee.node_id]
        if node.args:
            raise AglTypeError(
                "Constructor arguments must be named; positional arguments are not allowed.",
                span=node.span,
            )
        if ctor_ref.type_params:
            # Generic constructor: route to generic call handler.
            gdef = None
            sig = None
            # Look up the imported generic type by its OWNER name, not the callee
            # name: for an enum variant constructor (e.g. `some`), the callee name
            # is the variant, but only the enum TYPE name (`Option`) is registered
            # in the import map (enum variants travel with their enum).
            imported = self._env.get_open_imported_generic_type(ctor_ref.owner_name)
            if imported is not None:
                module_id, source_name, gdef = imported
                sig = self._env.get_ctor_sig_from_module(
                    module_id, source_name, ctor_ref.variant
                )
            return self._check_generic_constructor_call(
                node_type_args=node.type_args,
                ctor_ref=ctor_ref,
                named_args=node.named_args,
                span=node.span,
                expected=expected,
                sig=sig,
                gdef=gdef,
            )
        if node.type_args:
            raise AglTypeError(
                f"'{ctor_ref.owner_name}' is not a generic constructor and does not accept "
                "type arguments.",
                span=node.span,
            )
        owner = self._resolve_constructor_owner(ctor_ref, node.span)
        if isinstance(owner, ExceptionType) and owner.abstract:
            raise AglTypeError(
                "The abstract 'Exception' base type is not constructible. "
                "Use a concrete exception type (e.g. 'Abort').",
                span=node.span,
            )
        return self._check_constructor_call(
            owner=owner, variant=ctor_ref.variant, args=node.named_args, span=node.span
        )

    def _check_qualified_constructor_callee_call(
        self, node: Call, *, expected: Type | None = None
    ) -> Type:
        """Handle a Call whose callee is a qualified constructor FieldAccess."""
        assert isinstance(node.callee, FieldAccess)
        owner_name, variant, owner_module_id = (
            self._resolved.qualified_constructor_refs[node.callee.node_id]
        )
        if node.args:
            raise AglTypeError(
                "Constructor arguments must be named; positional arguments are not allowed.",
                span=node.span,
            )
        return self._resolve_qualified_constructor_and_call(
            owner_name=owner_name, variant=variant, owner_module_id=owner_module_id,
            args=node.named_args, span=node.span, expected=expected, type_args=node.type_args,
        )

    def _check_constructor_call(
        self,
        *,
        owner: RecordType | EnumType | ExceptionType,
        variant: str | None,
        args: tuple[NamedArg, ...],
        span: SourceSpan,
    ) -> RecordType | EnumType | ExceptionType:
        if isinstance(owner, EnumType):
            assert variant is not None, "variant is required for EnumType"
            fields = owner.variants[variant]
            type_label = f"Variant '{owner.name}.{variant}'"
        elif isinstance(owner, RecordType):
            fields = owner.fields
            type_label = f"Record '{owner.name}'"
        else:
            fields = owner.fields
            type_label = f"Exception type '{owner.name}'"

        provided = {arg.name: arg for arg in args}

        seen_args: set[str] = set()
        for arg in args:
            if arg.name in seen_args:
                raise AglTypeError(
                    f"Duplicate argument '{arg.name}' in constructor call.",
                    span=arg.span,
                )
            seen_args.add(arg.name)

        for arg_name in provided:
            if arg_name not in fields:
                raise AglTypeError(
                    f"{type_label} has no field '{arg_name}'.",
                    span=provided[arg_name].span,
                )

        for field_name in fields:
            if isinstance(owner, ExceptionType) and field_name == "trace_id":
                continue
            if field_name not in provided:
                raise AglTypeError(
                    f"Missing field '{field_name}' in constructor call for "
                    f"{type_label}.",
                    span=span,
                )

        for arg in args:
            expected_field_type = fields[arg.name]
            arg_type = self._check_expr(arg.value, expected=expected_field_type)
            self._assert_assignable(arg_type, expected_field_type, arg.span)

        return owner

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
            seen_pf: set[str] = set()
            for pf in pattern.fields:
                if pf.name in seen_pf:
                    raise AglTypeError(
                        f"Duplicate field '{pf.name}' in pattern for variant "
                        f"'{variant_name}' — each field may appear at most once.",
                        span=pf.span,
                    )
                seen_pf.add(pf.name)
                if pf.name not in vfields:
                    raise AglTypeError(
                        f"Variant '{variant_name}' has no field '{pf.name}'.",
                        span=pf.span,
                    )
                field_type = vfields[pf.name]
                self._bind_pattern_types(pf.pattern, field_type, pf)

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
