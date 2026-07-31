"""Standalone per-module type-table builder for the AgL type-checking pass.

``_TypeBuilder`` collects ``record``/``enum``/``exception``/alias declarations
and registers them into a ``TypeEnvironment``.  It was extracted from
``typecheck/checker.py`` (where it lived as the first pass of the
bidirectional type checker) so that ``typecheck/program.py`` can import it
without pulling in the full ``_Checker``.

Nominal types (``RecordType``/``EnumType``/``ExceptionType``) are lightweight
handles with no embedded field/variant data, so declaration order does not
matter: every reference — forward or backward — resolves to a valid handle.
Building is therefore two simple, order-free phases:

Phase 1 (``collect_shells_only``)
    Register every declared name's FINAL handle (or, for a generic
    declaration, its ``GenericTypeDef``) and every alias target.  A handle
    carries no shape, so this phase never needs revisiting.
Phase 2 (the loop in ``collect``)
    Resolve each declaration's field/variant type expressions, in source
    order, into a ``TypeDef`` registered in the shared ``TypeTable``.  An
    exception's ``TypeDef`` stores its OWN fields plus a resolved ``base``
    key (see ``semantics.type_table.TypeDef``); no ordering is required since
    the base need not be built yet to resolve the key.  A small post-pass
    (``_finalize_exceptions``) runs once every exception's own body is
    resolved to check own-vs-inherited field duplication, using
    ``TypeTable.exception_fields`` to read the (by-then fully buildable)
    flattened base chain.

Recursive nominal types (records, enums, exceptions, including mutual and
generic recursion) are legal. Because handles make forward references
trivially resolve, nothing in phase 2 itself needs to reject a declaration
that structurally contains itself; instead, once every body is resolved, an
inhabitation check (:func:`~agm.agl.semantics.analyses.compute_uninhabited`)
runs over the whole table and rejects any declaration that has no finite
value (e.g. a record whose only field is itself, with no ``array``/``dict``
or enum base-case escape). Recursive type ALIASES remain banned — an alias
is transparent and has no nominal identity to anchor a cycle — by the
existing alias-cycle check in ``typecheck/env.py``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace

from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, ModuleId, spell_scope_path
from agm.agl.semantics.analyses import compute_uninhabited, uninhabitable_message
from agm.agl.semantics.type_table import (
    BUILTIN_EXCEPTION_TYPE_DEFS,
    BUILTIN_PRELUDE_TYPE_DEFS,
    TypeDef,
)
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_PRELUDE_TYPE_NAMES,
    EnumType,
    ExceptionType,
    RecordType,
    Type,
    TypeVarType,
    reroot_type,
)
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    Item,
    Param,
    Program,
    RecordDef,
    TypeAlias,
    static_type_items,
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


def _bare_name(name: str) -> str:
    """Strip the scope prefix off a joined declaration name.

    A scoped declaration is keyed in the name tables under its full
    ``A::B::name`` spelling (see ``_TypeBuilder._static_type_items``), while
    a nominal handle's own ``name`` is always the last segment, with the path
    carried separately. This is the one place that separator is undone.
    """
    return name.rsplit("::", maxsplit=1)[-1]


# ---------------------------------------------------------------------------
# Pre-pass: collect and validate type declarations
# ---------------------------------------------------------------------------


class _TypeBuilder:
    """Collect record/enum/exception/alias declarations and validate them.

    Populates a ``TypeEnvironment`` with all user-declared types.  Raises
    ``AglTypeError`` on:
    - Duplicate type names (user vs user, or user shadowing a built-in).
    - Duplicate record and enum-variant fields.
    - Unknown type references inside field/variant definitions.
    - An uninhabited record, enum, or exception declaration (see
      :meth:`_check_inhabitation`).
    - Alias cycles.

    See the module docstring for the two-phase, order-free build strategy.
    """

    def __init__(self, env: TypeEnvironment, module_id: ModuleId = ENTRY_ID) -> None:
        self._env = env
        self._module_id = module_id
        # Track user-declared names → declaration span (excludes built-ins).
        self._declared: dict[str, SourceSpan] = {}
        # Index of record/enum/exception definitions, for phase-2 body
        # resolution and the post-pass inhabitation check.
        self._record_defs: dict[str, RecordDef] = {}
        self._enum_defs: dict[str, EnumDef] = {}
        self._exception_defs: dict[str, ExceptionDef] = {}

    def _owning_module_id(self, is_builtin: bool) -> ModuleId:
        """Module a declaration belongs to: the prelude if built-in, else this module."""
        return PRELUDE_ID if is_builtin else self._module_id

    @staticmethod
    def _static_type_items(
        items: tuple[Item, ...],
    ) -> Iterator[RecordDef | EnumDef | ExceptionDef | TypeAlias]:
        """Yield type declarations with their scope path included in their identity."""
        for item in static_type_items(items):
            if item.scope_path:
                path = tuple(segment.name for segment in item.scope_path)
                yield replace(item, name=spell_scope_path((*path, item.name)))
            else:
                yield item

    def collect(self, program: Program, *, check_inhabitation: bool = True) -> None:
        """Scan *program* and populate ``self._env``.

        Phase 1 registers every declaration's name and handle
        (:meth:`collect_shells_only`); phase 2 resolves each declaration's
        body, in source order, with no dependency ordering (see the module
        docstring); an inhabitation pass (:meth:`_check_inhabitation`) then
        rejects the first uninhabited declaration, in source order, unless
        *check_inhabitation* is ``False``; a final post-pass
        (:meth:`_finalize_exceptions`) validates own-vs-inherited exception
        field duplication now that every exception's flattened field set is
        buildable (an uninhabited ``extends`` cycle is always caught by the
        inhabitation pass first).

        *check_inhabitation* is ``False`` only for the program per-module
        re-check (``typecheck/program.py``), whose whole-program pre-pass has
        already run inhabitation over every module's declarations before any
        module's body is individually re-checked; re-running it per module
        would be redundant.
        """
        self.collect_shells_only(program)

        for item in self._static_type_items(program.body.items):
            path = tuple(segment.name for segment in item.scope_path)
            with self._env.type_scope(path):
                if isinstance(item, RecordDef):
                    self._build_record(item)
                elif isinstance(item, EnumDef):
                    self._build_enum(item)
                elif isinstance(item, ExceptionDef):
                    self._build_exception(item)
                else:
                    self._validate_alias(item)

        if check_inhabitation:
            self._check_inhabitation(program)
        self._finalize_exceptions()

    def collect_shells_only(self, program: Program) -> None:
        """Register phase-1 declarations: names, handles, and alias targets.

        Public interface for the program pre-pass (``program.py``) which needs to
        register every module's declarations before resolving any body.
        Non-generic records/enums/exceptions get their FINAL handle
        registered directly (a handle carries no shape, so there is nothing
        left to "finish" later); generic records/enums get their
        ``GenericTypeDef`` (name, type params, and a handle template stamped
        with ``TypeVarType`` args) registered instead — likewise final, since
        the template carries no shape either.  Exceptions are never generic.
        """
        for item in self._static_type_items(program.body.items):
            if isinstance(item, RecordDef):
                self._register_name(
                    item.name,
                    item.span,
                    is_builtin=item.is_builtin,
                    expected_defs=BUILTIN_PRELUDE_TYPE_DEFS,
                )
                self._env.unregister_name(item.name)
                self._register_record_or_enum_handle(item, is_enum=False)
                self._record_defs[item.name] = item
            elif isinstance(item, EnumDef):
                self._register_name(
                    item.name,
                    item.span,
                    is_builtin=item.is_builtin,
                    expected_defs=BUILTIN_PRELUDE_TYPE_DEFS,
                )
                self._env.unregister_name(item.name)
                self._register_record_or_enum_handle(item, is_enum=True)
                self._enum_defs[item.name] = item
            elif isinstance(item, ExceptionDef):
                self._register_name(
                    item.name,
                    item.span,
                    is_builtin=item.is_builtin,
                    expected_defs=BUILTIN_EXCEPTION_TYPE_DEFS,
                )
                self._env.unregister_name(item.name)
                module_id = self._owning_module_id(item.is_builtin)
                self._env.register_type(
                    item.name,
                    ExceptionType(
                        name=_bare_name(item.name),
                        module_id=module_id,
                        scope_path=tuple(segment.name for segment in item.scope_path),
                    ),
                )
                self._exception_defs[item.name] = item
            else:
                self._register_name(item.name, item.span)
                self._env.unregister_name(item.name)
                self._env.register_alias(item.name, item.type_expr, type_params=item.type_params)

    def _register_record_or_enum_handle(self, item: RecordDef | EnumDef, *, is_enum: bool) -> None:
        module_id = self._owning_module_id(item.is_builtin)
        declared_name = _bare_name(item.name)
        scope_path = tuple(segment.name for segment in item.scope_path)
        type_params = item.type_params
        if type_params:
            type_args = tuple(TypeVarType(p) for p in type_params)
            template: RecordType | EnumType = (
                EnumType(
                    name=declared_name,
                    type_args=type_args,
                    module_id=module_id,
                    scope_path=scope_path,
                )
                if is_enum
                else RecordType(
                    name=declared_name,
                    type_args=type_args,
                    module_id=module_id,
                    scope_path=scope_path,
                )
            )
            gdef = GenericTypeDef(
                kind="enum" if is_enum else "record",
                type_params=type_params,
                template=template,
            )
            self._env.register_generic_type(item.name, gdef)
        else:
            handle: RecordType | EnumType = (
                EnumType(name=declared_name, module_id=module_id, scope_path=scope_path)
                if is_enum
                else RecordType(name=declared_name, module_id=module_id, scope_path=scope_path)
            )
            self._env.register_type(item.name, handle)

    def validate_alias(self, stmt: TypeAlias) -> None:
        """Public proxy for :meth:`_validate_alias`."""
        self._validate_alias(stmt)

    def build_record(self, name: str) -> None:
        """Resolve and register the named record's body.

        Public entry point for the program pre-pass, which resolves every
        module's declarations in a fixed order (no dependency ordering —
        see the module docstring).
        """
        self._build_record(self._record_defs[name])

    def build_enum(self, name: str) -> None:
        """Resolve and register the named enum's body. See :meth:`build_record`."""
        self._build_enum(self._enum_defs[name])

    def build_exception(self, name: str) -> None:
        """Resolve and register the named exception's body. See :meth:`build_record`."""
        self._build_exception(self._exception_defs[name])

    def _register_name(
        self,
        name: str,
        span: SourceSpan,
        *,
        is_builtin: bool = False,
        expected_defs: Mapping[str, TypeDef] | None = None,
    ) -> None:
        # `name` carries its scope path joined with "::" (see
        # `_static_type_items`) when the declaration is scoped, so the check
        # strips it back to the bare canonical name a scoped `builtin`
        # declaration still has to spell — the host-known tables name
        # canonical types, never a scope path.
        #
        # Every caller that can pass `is_builtin=True` passes the table for
        # its own kind (records/enums against BUILTIN_PRELUDE_TYPE_DEFS,
        # exceptions against BUILTIN_EXCEPTION_TYPE_DEFS), so this one check
        # rejects both an entirely unknown name and a name declared under the
        # wrong kind — the latter being exactly what `_validate_builtin_shape`
        # would otherwise assume was present.
        bare_name = _bare_name(name)
        if is_builtin and expected_defs is not None and bare_name not in expected_defs:
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
            fields[fd.name] = self._resolve_field_type(fd)
        module_id = self._owning_module_id(stmt.is_builtin)
        typedef = TypeDef(
            kind="record",
            name=_bare_name(stmt.name),
            module_id=module_id,
            scope_path=tuple(segment.name for segment in stmt.scope_path),
            fields=tuple(fields.items()),
        )
        self._validate_builtin_shape(stmt, typedef, BUILTIN_PRELUDE_TYPE_DEFS)
        self._env.type_table.register(typedef)
        # Register field kinds for this record constructor, under the same
        # owning module as the TypeDef just above (PRELUDE_ID for a builtin
        # declaration, wherever it is declared).
        field_kinds = tuple((fd.name, fd.kind) for fd in stmt.fields)
        self._env.register_constructor_field_kinds(
            stmt.name, None, field_kinds, module_id=module_id
        )

    def _build_enum(self, stmt: EnumDef) -> None:
        if stmt.type_params:
            self._build_generic_enum(stmt)
            return
        variants: dict[str, dict[str, Type]] = {}
        for vd in stmt.variants:
            vfields: dict[str, Type] = {}
            seen_vfields: dict[str, SourceSpan] = {}
            for fd in vd.fields:
                if fd.name in seen_vfields:
                    raise AglTypeError(
                        f"Duplicate field '{fd.name}' in variant '{stmt.name}.{vd.name}'.",
                        span=fd.span,
                    )
                seen_vfields[fd.name] = fd.span
                vfields[fd.name] = self._resolve_field_type(fd)
            variants[vd.name] = vfields
        module_id = self._owning_module_id(stmt.is_builtin)
        typedef = TypeDef(
            kind="enum",
            name=_bare_name(stmt.name),
            module_id=module_id,
            scope_path=tuple(segment.name for segment in stmt.scope_path),
            variants=tuple((vname, tuple(vfields.items())) for vname, vfields in variants.items()),
        )
        self._validate_builtin_shape(stmt, typedef, BUILTIN_PRELUDE_TYPE_DEFS)
        self._env.type_table.register(typedef)
        # Register field kinds for each variant constructor, under the same
        # owning module as the TypeDef just above.
        for vd in stmt.variants:
            vfield_kinds = tuple((fd.name, fd.kind) for fd in vd.fields)
            self._env.register_constructor_field_kinds(
                stmt.name, vd.name, vfield_kinds, module_id=module_id
            )

    def _build_exception(self, stmt: ExceptionDef) -> None:
        """Resolve and register an exception's own ``TypeDef`` (no ordering).

        Reuses the record-path field resolution and duplicate-own-field
        check.  ``base`` is resolved to a ``(module_id, scope_path, name)`` key — no
        ordering is required to do so, since only the base's *identity* (not
        its shape) is needed here.  Own-vs-inherited field duplication and
        constructor-callability are checked later, once every exception's
        shape is buildable (see :meth:`_finalize_exceptions`).
        """
        base_key: tuple[ModuleId, tuple[str, ...], str] | None = None
        if stmt.base is not None:
            base_type = self._env.resolve_named_type(stmt.base)
            if not isinstance(base_type, ExceptionType):
                raise AglTypeError(
                    f"Exception '{stmt.name}' extends unknown exception '{stmt.base}'.",
                    span=stmt.span,
                )
            base_key = (base_type.module_id, base_type.scope_path, base_type.name)
        fields: dict[str, Type] = {}
        seen_fields: dict[str, SourceSpan] = {}
        for fd in stmt.fields:
            if not stmt.is_builtin and fd.name == "trace_id":
                raise AglTypeError(
                    "Exception field name 'trace_id' is reserved for the built-in trace id.",
                    span=fd.span,
                )
            if fd.name in seen_fields:
                raise AglTypeError(
                    f"Duplicate field '{fd.name}' in exception '{stmt.name}'.",
                    span=fd.span,
                )
            seen_fields[fd.name] = fd.span
            fields[fd.name] = self._resolve_field_type(fd)
        module_id = self._owning_module_id(stmt.is_builtin)
        # Own field kinds honor each field's declared @pos/@std/@named marker —
        # exactly like a record's fields — in declaration order, parallel to
        # ``fields`` above.  Stored as ``ParamKind.value`` strings (see
        # ``TypeDef.field_kinds``: ``semantics`` may not import ``syntax``).
        # Inheriting the base's kinds through the extends chain is
        # ``TypeTable.exception_field_kinds``'s job (walked on demand from
        # ``TypeDef.base``, no build-ordering step needed).
        typedef = TypeDef(
            kind="exception",
            name=_bare_name(stmt.name),
            module_id=module_id,
            scope_path=tuple(segment.name for segment in stmt.scope_path),
            fields=tuple(fields.items()),
            abstract=stmt.base is None,
            base=base_key,
            field_kinds=tuple(fd.kind.value for fd in stmt.fields),
        )
        self._validate_builtin_shape(stmt, typedef, BUILTIN_EXCEPTION_TYPE_DEFS)
        self._env.type_table.register(typedef)

    def _finalize_exceptions(self) -> None:
        """Post-pass: reject own-vs-inherited field name collisions.

        Deferred until every exception's own ``TypeDef`` (and therefore its
        base's, however deep the ``extends`` chain) is registered — there is
        no build-ordering step in :meth:`_build_exception` (see the module
        docstring).  A cyclic ``extends`` chain has no independent evidence to
        ever become inhabited (see :meth:`_check_inhabitation`), so it is
        always rejected before this post-pass runs — either here (single-
        module) or by the program pre-pass (``typecheck/program.py``) before
        Phase 3 re-checks this module at all — and
        :meth:`~agm.agl.semantics.type_table.TypeTable.exception_fields` never
        hits its internal cycle guard here.
        """
        for item in self._exception_defs.values():
            if item.base is None:
                continue
            module_id = self._owning_module_id(item.is_builtin)
            typedef = self._env.type_table.exception_def(
                ExceptionType(
                    name=_bare_name(item.name),
                    module_id=module_id,
                    scope_path=tuple(segment.name for segment in item.scope_path),
                )
            )
            assert typedef.base is not None
            base_handle = ExceptionType(
                name=typedef.base[2], module_id=typedef.base[0], scope_path=typedef.base[1]
            )
            base_fields = self._env.type_table.exception_fields(base_handle)
            for fd in item.fields:
                if fd.name in base_fields:
                    raise AglTypeError(
                        f"Duplicate field '{fd.name}' in exception '{item.name}'.",
                        span=fd.span,
                    )

    def _validate_builtin_shape(
        self,
        stmt: RecordDef | EnumDef | ExceptionDef,
        typedef: TypeDef,
        expected_defs: Mapping[str, TypeDef],
    ) -> None:
        """Check a ``builtin`` declaration's shape against its canonical definition.

        The canonical definitions in *expected_defs* are host-known shapes
        keyed by bare name, indifferent to where the declaration sits: a
        scoped ``builtin`` declaration names the same host type as a root one
        and must match the same shape, just at a different nominal path. The
        comparison therefore re-roots *typedef* under its own scope path
        (:meth:`_reroot_typedef`) before comparing — not just its own
        top-level ``scope_path`` (the one field *expected* to differ), but
        also every nominal reference embedded in its fields/variants/base
        that resolves under that same path, so a scoped exception hierarchy
        or a scoped record naming a sibling scoped builtin still compares
        equal to the canonical (root) shape. *bare_name* is always present in
        *expected_defs* here: ``_register_name`` already validated it against
        this same kind-appropriate table during phase 1.
        """
        if not stmt.is_builtin:
            return
        bare_name = _bare_name(stmt.name)
        expected = expected_defs[bare_name]
        if self._reroot_typedef(typedef) != expected:
            raise AglTypeError(
                f"Builtin type '{stmt.name}' has an invalid definition.",
                span=stmt.span,
            )

    @staticmethod
    def _reroot_typedef(typedef: TypeDef) -> TypeDef:
        """Re-root *typedef* onto its own scope path, for canonical comparison.

        *typedef.scope_path* itself is dropped (the field a scoped ``builtin``
        declaration is always expected to differ in), and that same path is
        also stripped, via :func:`~agm.agl.semantics.types.reroot_type`, from
        every nominal reference embedded in its fields, variant fields, and
        (for an exception) its ``extends`` base key — each of those resolves
        under the declaration's own scope path exactly when it names a
        sibling member of the same region, matching how the canonical
        (root, path ``()``) shape names its own siblings.
        """
        prefix = typedef.scope_path
        if not prefix:
            return typedef
        fields = tuple((name, reroot_type(t, prefix)) for name, t in typedef.fields)
        variants = tuple(
            (vname, tuple((fname, reroot_type(ft, prefix)) for fname, ft in vfields))
            for vname, vfields in typedef.variants
        )
        base = typedef.base
        if base is not None and base[1][: len(prefix)] == prefix:
            base = (base[0], base[1][len(prefix) :], base[2])
        return replace(typedef, scope_path=(), fields=fields, variants=variants, base=base)

    def _resolve_field_type(self, fd: Param, type_vars: frozenset[str] = frozenset()) -> Type:
        """Resolve a field's TypeExpr to a semantic Type.

        No ordering is required: every named type reference (record, enum,
        alias, or generic) was already registered as a handle in phase 1
        (``collect_shells_only``), so resolution succeeds regardless of
        declaration order.
        """
        # A field is always annotated: the grammar's ``field_def`` requires it.
        assert fd.type_expr is not None
        return self._env.resolve_type_expr(fd.type_expr, span=fd.span, type_vars=type_vars)

    def _build_generic_record(self, stmt: RecordDef) -> None:
        """Resolve a generic record's fields and register its TypeDef + constructor.

        The ``GenericTypeDef`` itself (name, type params, handle template)
        was already registered in phase 1 so that forward references to
        this generic type resolve regardless of declaration order.
        """
        type_params = stmt.type_params
        type_vars = frozenset(type_params)
        fields: dict[str, Type] = {}
        seen_fields: dict[str, SourceSpan] = {}
        for fd in stmt.fields:
            if fd.name in seen_fields:
                raise AglTypeError(
                    f"Duplicate field '{fd.name}' in record '{stmt.name}'.", span=fd.span
                )
            seen_fields[fd.name] = fd.span
            fields[fd.name] = self._resolve_field_type(fd, type_vars=type_vars)
        gdef = self._env.get_generic_type(stmt.name)
        assert gdef is not None, f"compiler bug: generic record {stmt.name!r} not pre-registered"
        template = gdef.template
        assert isinstance(template, RecordType)
        module_id = self._owning_module_id(stmt.is_builtin)
        typedef = TypeDef(
            kind="record",
            name=_bare_name(stmt.name),
            module_id=module_id,
            scope_path=tuple(segment.name for segment in stmt.scope_path),
            type_params=type_params,
            fields=tuple(fields.items()),
        )
        self._validate_builtin_shape(stmt, typedef, BUILTIN_PRELUDE_TYPE_DEFS)
        self._env.type_table.register(typedef)
        field_names = tuple(fields.keys())
        field_templates = tuple(fields.values())
        sig = ConstructorSignature(
            owner_name=stmt.name,
            variant=None,
            field_names=field_names,
            field_templates=field_templates,
            result_template=template,
            type_params=type_params,
        )
        self._env.register_constructor_signature(sig)
        # Register field kinds for the generic record constructor, under the
        # same owning module as the TypeDef just above.
        generic_record_field_kinds = tuple((fd.name, fd.kind) for fd in stmt.fields)
        self._env.register_constructor_field_kinds(
            stmt.name, None, generic_record_field_kinds, module_id=module_id
        )

    def _build_generic_enum(self, stmt: EnumDef) -> None:
        """Resolve a generic enum's variants and register its TypeDef + constructors.

        See :meth:`_build_generic_record` for why the ``GenericTypeDef``
        itself is not (re-)registered here.
        """
        type_params = stmt.type_params
        type_vars = frozenset(type_params)
        variants: dict[str, dict[str, Type]] = {}
        for vd in stmt.variants:
            vfields: dict[str, Type] = {}
            seen_vfields: dict[str, SourceSpan] = {}
            for fd in vd.fields:
                if fd.name in seen_vfields:
                    raise AglTypeError(
                        f"Duplicate field '{fd.name}' in variant '{stmt.name}.{vd.name}'.",
                        span=fd.span,
                    )
                seen_vfields[fd.name] = fd.span
                vfields[fd.name] = self._resolve_field_type(fd, type_vars=type_vars)
            variants[vd.name] = vfields
        gdef = self._env.get_generic_type(stmt.name)
        assert gdef is not None, f"compiler bug: generic enum {stmt.name!r} not pre-registered"
        template = gdef.template
        assert isinstance(template, EnumType)
        module_id = self._owning_module_id(stmt.is_builtin)
        typedef = TypeDef(
            kind="enum",
            name=_bare_name(stmt.name),
            module_id=module_id,
            scope_path=tuple(segment.name for segment in stmt.scope_path),
            type_params=type_params,
            variants=tuple((vname, tuple(vfields.items())) for vname, vfields in variants.items()),
        )
        self._validate_builtin_shape(stmt, typedef, BUILTIN_PRELUDE_TYPE_DEFS)
        self._env.type_table.register(typedef)
        # Register one ConstructorSignature and field kinds per variant, under
        # the same owning module as the TypeDef just above.
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
                type_params=type_params,
            )
            self._env.register_constructor_signature(sig)
            # Register field kinds for this generic enum variant constructor.
            vfield_kinds = tuple((fd.name, fd.kind) for fd in vd.fields)
            self._env.register_constructor_field_kinds(
                stmt.name, vd.name, vfield_kinds, module_id=module_id
            )

    def _validate_alias(self, stmt: TypeAlias) -> None:
        """Validate that the alias target resolves without cycles.

        A parameterized alias body may reference its own type parameters, so
        they are in scope as type variables during validation.
        """
        self._env.resolve_type_expr(
            stmt.type_expr,
            span=stmt.span,
            type_vars=frozenset(stmt.type_params),
        )

    # ------------------------------------------------------------------
    # Inhabitation check
    # ------------------------------------------------------------------

    def _check_inhabitation(self, program: Program) -> None:
        """Reject the first uninhabited record/enum/exception, in source order.

        Runs :func:`~agm.agl.semantics.analyses.compute_uninhabited` over the
        WHOLE shared table (this module's declarations plus anything already
        seeded from a prior REPL entry or the builtin/prelude defs — all
        trivially inhabited), then walks *program*'s own declarations in
        source order reporting the first one whose key came back
        uninhabited, at its declaration span. Reporting only THIS program's
        own declarations (rather than any uninhabited key at all) keeps
        error attribution local: a key from another module's table entry has
        no span available here anyway.
        """
        uninhabited = compute_uninhabited(self._env.type_table)
        if not uninhabited:
            return
        for item in self._static_type_items(program.body.items):
            if not isinstance(item, (RecordDef, EnumDef, ExceptionDef)):
                continue
            module_id = self._owning_module_id(item.is_builtin)
            scope_path = tuple(segment.name for segment in item.scope_path)
            declared_name = _bare_name(item.name)
            key = (module_id, scope_path, declared_name)
            if key not in uninhabited:
                continue
            typedef = self._env.type_table.get(module_id, declared_name, scope_path)
            assert typedef is not None
            raise AglTypeError(uninhabitable_message(typedef.kind, item.name), span=item.span)
        raise AssertionError(  # pragma: no cover
            "compute_uninhabited reported a key not owned by this program's own "
            "declarations — every uninhabited key in a single-module table is "
            "expected to trace back to a declaration in the program being checked"
        )
