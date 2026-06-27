"""Standalone per-module type-table builder for the AgL v2 type-checking pass.

``_TypeBuilder`` collects ``record``/``enum``/``exception``/alias shells and
bodies, and registers generic templates, into a ``TypeEnvironment``.  It was
extracted from ``typecheck/checker.py`` (where it lived as the first pass of
the bidirectional type checker) so that ``typecheck/graph.py`` can import it
without pulling in the full ``_Checker``.

Module-level helpers
--------------------
``_expected_builtin_type``
    Map a builtin type name to its canonical semantic ``Type`` object (from the
    prelude / exception tables).
``_type_shape_matches``
    Structural comparison used when validating that a ``builtin`` type
    definition matches the expected shape.
"""

from __future__ import annotations

from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, ModuleId
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
    EnumType,
    ExceptionType,
    RecordType,
    Type,
    TypeVarType,
)
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    Param,
    Program,
    RecordDef,
    TypeAlias,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import (
    AglTypeError,
    ConstructorSignature,
    GenericTypeDef,
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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


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
        self.collect_shells_only(program)

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
        self, fd: Param, owner: str, type_vars: frozenset[str] = frozenset()
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
