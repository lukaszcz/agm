"""Per-module type-table builder for the AgL type-checking pass.

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
    Register every declared name's handle (or, for a generic
    declaration, its ``GenericTypeDef``) and every alias target.  A handle
    carries no shape. Inline enum-member shells use their syntactically
    captured owner parameters until an intermediate reconciliation pass
    resolves transparent aliases and finalizes every member arity.
Phase 2 (the loop in ``collect``)
    Resolve each declaration's field/variant type expressions, in source
    order, into a ``TypeDef`` registered in the shared ``TypeTable``.  An
    inline enum member's resolved body confirms the already-finalized shell.
    An
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
that structurally contains itself; instead, an inhabitation check
(:func:`~agm.agl.semantics.analyses.compute_uninhabited`) runs once,
whole-program, over the shared table ahead of every module's body resolution
(``typecheck/program.py::_build_program_type_table``), rejecting any
declaration that has no finite value (e.g. a record whose only field is
itself, with no ``array``/``dict`` or enum base-case escape). Recursive type
ALIASES remain banned — an alias is transparent and has no nominal identity
to anchor a cycle — by the existing alias-cycle check in
``typecheck/env.py``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import cast

from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.semantics.type_table import (
    TypeDef,
    source_enum_member_decl_id,
    source_nominal_decl_id,
)
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_PRELUDE_TYPE_NAMES,
    EnumType,
    ExceptionType,
    RecordType,
    Type,
    TypeVarType,
    free_type_vars,
)
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    Item,
    Param,
    Program,
    RecordDef,
    TypeAlias,
    VariantDef,
    VariantRef,
    scoped_public_name,
    static_type_items,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, NameT, TypeExpr, member_type_params
from agm.agl.typecheck.builtin_contracts import (
    BUILTIN_ENUM_CONTRACTS,
    BUILTIN_EXCEPTION_CONTRACTS,
    BUILTIN_RECORD_CONTRACTS,
    BuiltinTypeContract,
    contract_for_typedef,
)
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
    frozenset({"text", "json", "bool", "int", "decimal", "unit"})
    | BUILTIN_EXCEPTION_NAMES
    | BUILTIN_PRELUDE_TYPE_NAMES
)


def _decl_identity(
    module_id: ModuleId, scope_path: tuple[str, ...], bare_name: str, node_id: int
) -> int:
    """Return the declaration identity for a record/enum/exception declaration.

    Every parsed declaration is identified by its own AST node id, which the
    loader keeps disjoint across a program's modules. Host-known reserved ids
    identify only seeded fallback definitions used when no source declaration
    is loaded.
    """
    return source_nominal_decl_id(module_id, scope_path, bare_name, node_id)


def _bare_name(name: str) -> str:
    """Strip the scope prefix off a joined declaration name.

    A scoped declaration is keyed in the name tables under its full
    ``A::B::name`` spelling (see ``_TypeBuilder._static_type_items``), while
    a nominal handle's own ``name`` is always the last segment, with the path
    carried separately. This is the one place that separator is undone.
    """
    return name.rsplit("::", maxsplit=1)[-1]


def _member_identity(enum: EnumDef, member: VariantDef, module_id: ModuleId) -> int:
    """Return an inline member's source declaration identity."""
    return source_enum_member_decl_id(
        module_id,
        tuple(segment.name for segment in enum.scope_path),
        _bare_name(enum.name),
        member.name,
        member.node_id,
        is_builtin=enum.is_builtin,
    )


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
    - Alias cycles.

    Inhabitation (rejecting a record/enum/exception with no finite value) is
    checked whole-program, not here — see :meth:`collect`.

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
        # Source-built definitions retained independently of the shared table.
        # A source declaration can supersede a seeded fallback at the same
        # name path, so contract validation must inspect this exact object.
        self._resolved_defs: dict[str, TypeDef] = {}

    @staticmethod
    def _static_type_items(
        items: tuple[Item, ...],
    ) -> Iterator[RecordDef | EnumDef | ExceptionDef | TypeAlias]:
        """Yield type declarations with their scope path included in their identity."""
        for item in static_type_items(items):
            if item.scope_path:
                yield replace(item, name=scoped_public_name(item.scope_path, item.name))
            else:
                yield item

    def collect(self, program: Program) -> None:
        """Scan *program* and populate ``self._env``.

        Phase 1 registers every declaration's name and handle
        (:meth:`collect_shells_only`); phase 2 resolves each declaration's
        body, in source order, with no dependency ordering (see the module
        docstring); a final post-pass (:meth:`_finalize_exceptions`)
        validates own-vs-inherited exception field duplication, which needs
        every exception's flattened field set to be buildable.

        Inhabitation is checked once, whole-program, ahead of every module's
        body resolution (:func:`~agm.agl.semantics.analyses.compute_uninhabited`
        over the shared type table in
        ``typecheck/program.py::_build_program_type_table``), so this pass
        does not repeat it per module.
        """
        self.collect_shells_only(program)
        self.reconcile_inline_member_arities()

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

        self._finalize_exceptions()
        self.validate_builtin_contracts()

    def collect_shells_only(self, program: Program) -> None:
        """Register phase-1 declarations: names, handles, and alias targets.

        Public interface for the program pre-pass (``program.py``) which needs to
        register every module's declarations before resolving any body.
        Non-generic records/enums/exceptions get their handle
        registered directly (a handle carries no shape, so there is nothing
        left to "finish" later); generic records/enums get their
        ``GenericTypeDef`` (name, type params, and a handle template stamped
        with ``TypeVarType`` args) registered instead. Inline member handles
        are provisional because transparent aliases can erase syntactically
        captured owner parameters; phase 2 reconciles them. Exceptions are
        never generic.
        """
        for item in self._static_type_items(program.body.items):
            if isinstance(item, RecordDef):
                self._register_name(
                    item.name,
                    item.span,
                    is_builtin=item.is_builtin,
                    expected_contracts=BUILTIN_RECORD_CONTRACTS,
                )
                self._env.unregister_name(item.name)
                self._register_record_or_enum_handle(item, is_enum=False)
                self._record_defs[item.name] = item
            elif isinstance(item, EnumDef):
                self._register_name(
                    item.name,
                    item.span,
                    is_builtin=item.is_builtin,
                    expected_contracts=BUILTIN_ENUM_CONTRACTS,
                )
                self._env.unregister_name(item.name)
                self._clear_inline_member_names(item)
                self._register_record_or_enum_handle(item, is_enum=True)
                for member in item.members:
                    if isinstance(member, VariantDef):
                        self._register_inline_member_handle(item, member)
                self._enum_defs[item.name] = item
            elif isinstance(item, ExceptionDef):
                self._register_name(
                    item.name,
                    item.span,
                    is_builtin=item.is_builtin,
                    expected_contracts=BUILTIN_EXCEPTION_CONTRACTS,
                )
                self._env.unregister_name(item.name)
                module_id = self._module_id
                bare_name = _bare_name(item.name)
                scope_path = tuple(segment.name for segment in item.scope_path)
                self._env.register_type(
                    item.name,
                    ExceptionType(
                        name=bare_name,
                        module_id=module_id,
                        scope_path=scope_path,
                        decl_id=_decl_identity(module_id, scope_path, bare_name, item.node_id),
                    ),
                )
                self._exception_defs[item.name] = item
            else:
                self._register_name(item.name, item.span)
                self._env.unregister_name(item.name)
                self._env.register_alias(item.name, item.type_expr, type_params=item.type_params)

    def reconcile_inline_member_arities(self) -> None:
        """Finalize inline-member shells before any dependent body resolves.

        Transparent aliases can erase owner parameters mentioned in source
        syntax. Resolve only the member field types here, then replace each
        provisional shell with the parameters that remain genuinely free.
        Full enum bodies and constructor metadata are still built in phase 2.
        """
        for stmt in self._enum_defs.values():
            type_vars = frozenset(stmt.type_params)
            path = tuple(segment.name for segment in stmt.scope_path)
            with self._env.type_scope(path):
                for member in stmt.members:
                    if not isinstance(member, VariantDef):
                        continue
                    field_types = tuple(
                        self._resolve_field_type(field, type_vars=type_vars)
                        for field in member.fields
                    )
                    captured_params = tuple(
                        param
                        for param in stmt.type_params
                        if any(param in free_type_vars(field_type) for field_type in field_types)
                    )
                    self._replace_inline_member_handle(stmt, member, captured_params)

    def _clear_inline_member_names(self, enum: EnumDef) -> None:
        """Release member record names from a superseded enum declaration."""
        scope_path = (*tuple(segment.name for segment in enum.scope_path), _bare_name(enum.name))
        for typedef in self._env.type_table.entries():
            if (
                typedef.kind == "record"
                and typedef.module_id == self._module_id
                and typedef.scope_path == scope_path
                and typedef.is_inline_enum_member
            ):
                self._env.unregister_name(f"{enum.name}::{typedef.name}")

    def _register_inline_member_handle(self, enum: EnumDef, member: VariantDef) -> None:
        """Register an inline enum member as its scoped record type."""
        member_name = f"{enum.name}::{member.name}"
        type_params = member_type_params(
            (cast(TypeExpr, field.type_expr) for field in member.fields), enum.type_params
        )
        self._register_name(member_name, member.span)
        self._replace_inline_member_handle(enum, member, type_params)

    def _replace_inline_member_handle(
        self, enum: EnumDef, member: VariantDef, type_params: tuple[str, ...]
    ) -> None:
        """Make a member's type shell agree with its current captured parameters."""
        enum_name = _bare_name(enum.name)
        scope_path = (*tuple(segment.name for segment in enum.scope_path), enum_name)
        member_name = f"{enum.name}::{member.name}"
        self._env.unregister_name(member_name)
        decl_id = _member_identity(enum, member, self._module_id)
        template = RecordType(
            name=member.name,
            type_args=tuple(TypeVarType(param) for param in type_params),
            module_id=self._module_id,
            scope_path=scope_path,
            decl_id=decl_id,
        )
        if type_params:
            self._env.register_generic_type(
                member_name,
                GenericTypeDef(kind="record", type_params=type_params, template=template),
            )
        else:
            self._env.register_type(member_name, template)

    def _register_record_or_enum_handle(self, item: RecordDef | EnumDef, *, is_enum: bool) -> None:
        module_id = self._module_id
        declared_name = _bare_name(item.name)
        scope_path = tuple(segment.name for segment in item.scope_path)
        decl_id = _decl_identity(module_id, scope_path, declared_name, item.node_id)
        type_params = item.type_params
        if type_params:
            type_args = tuple(TypeVarType(p) for p in type_params)
            template: RecordType | EnumType = (
                EnumType(
                    name=declared_name,
                    type_args=type_args,
                    module_id=module_id,
                    scope_path=scope_path,
                    decl_id=decl_id,
                )
                if is_enum
                else RecordType(
                    name=declared_name,
                    type_args=type_args,
                    module_id=module_id,
                    scope_path=scope_path,
                    decl_id=decl_id,
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
                EnumType(
                    name=declared_name, module_id=module_id, scope_path=scope_path, decl_id=decl_id
                )
                if is_enum
                else RecordType(
                    name=declared_name, module_id=module_id, scope_path=scope_path, decl_id=decl_id
                )
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

    def validate_builtin_contracts(self) -> None:
        """Validate builtin shapes after every referenced type body is available."""
        declarations: tuple[
            tuple[
                RecordDef | EnumDef | ExceptionDef,
                Mapping[str, BuiltinTypeContract],
            ],
            ...,
        ] = (
            *((item, BUILTIN_RECORD_CONTRACTS) for item in self._record_defs.values()),
            *((item, BUILTIN_ENUM_CONTRACTS) for item in self._enum_defs.values()),
            *((item, BUILTIN_EXCEPTION_CONTRACTS) for item in self._exception_defs.values()),
        )
        for stmt, expected_contracts in declarations:
            if not stmt.is_builtin:
                continue
            typedef = self._resolved_defs.get(stmt.name)
            assert typedef is not None, "compiler bug: builtin type is not registered"
            base_type: ExceptionType | None = None
            if typedef.base is not None:
                base_def = self._env.type_table.get_by_id(typedef.base)
                assert base_def is not None, "compiler bug: builtin base is not registered"
                base_handle = base_def.handle()
                assert isinstance(base_handle, ExceptionType)
                base_type = base_handle
            self._validate_builtin_shape(
                stmt,
                typedef,
                expected_contracts,
                base_type=base_type,
            )

    def _register_name(
        self,
        name: str,
        span: SourceSpan,
        *,
        is_builtin: bool = False,
        expected_contracts: Mapping[str, BuiltinTypeContract] | None = None,
    ) -> None:
        # `name` carries its scope path joined with "::" (see
        # `_static_type_items`) when the declaration is scoped, so the check
        # strips it back to the bare canonical name a scoped `builtin`
        # declaration still has to spell — the host-known tables name
        # contracts, never a scope path.
        #
        # Every caller that can pass `is_builtin=True` passes the contract
        # table for its own kind,
        # so this one check rejects both an entirely unknown name and a name
        # declared under the wrong kind — the latter being exactly what
        # `_validate_builtin_shape` would otherwise assume was present.
        bare_name = _bare_name(name)
        if is_builtin and expected_contracts is not None and bare_name not in expected_contracts:
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
        module_id = self._module_id
        scope_path = tuple(segment.name for segment in stmt.scope_path)
        bare_name = _bare_name(stmt.name)
        typedef = TypeDef(
            kind="record",
            name=bare_name,
            module_id=module_id,
            scope_path=scope_path,
            fields=tuple(fields.items()),
            field_mutability=tuple(fd.mutable for fd in stmt.fields),
            is_builtin=stmt.is_builtin,
            decl_node_id=_decl_identity(module_id, scope_path, bare_name, stmt.node_id),
        )
        self._resolved_defs[stmt.name] = typedef
        self._env.type_table.register(typedef)
        # Register field kinds for this record constructor, under the same
        # owning identity as the TypeDef just above (its declaring module,
        # like every other declaration — including a builtin one).
        field_kinds = tuple((fd.name, fd.kind) for fd in stmt.fields)
        self._env.register_constructor_field_kinds(
            bare_name,
            field_kinds,
            scope_path=scope_path,
            module_id=module_id,
            decl_id=typedef.decl_node_id,
        )

    def _build_enum(self, stmt: EnumDef) -> None:
        if stmt.type_params:
            self._build_generic_enum(stmt)
            return
        typedef, member_defs = self._build_enum_members(stmt, type_vars=frozenset())
        self._resolved_defs[stmt.name] = typedef
        for member_def in member_defs:
            self._env.type_table.register(member_def)
        self._env.type_table.register(typedef)
        self._register_enum_constructor_field_kinds(stmt)

    def _build_enum_members(
        self, stmt: EnumDef, *, type_vars: frozenset[str]
    ) -> tuple[TypeDef, tuple[TypeDef, ...]]:
        """Build an enum's record members and its member-set declaration."""
        module_id = self._module_id
        scope_path = tuple(segment.name for segment in stmt.scope_path)
        bare_name = _bare_name(stmt.name)
        member_scope_path = (*scope_path, bare_name)
        member_defs: list[TypeDef] = []
        members: list[RecordType] = []
        enum_decl_id = _decl_identity(module_id, scope_path, bare_name, stmt.node_id)
        member_spans: list[SourceSpan] = []
        for member in stmt.members:
            if isinstance(member, VariantRef):
                reference = (
                    AppliedT(
                        name=member.chain.member,
                        args=member.type_args,
                        qualifier=member.chain,
                        span=member.span,
                        node_id=member.node_id,
                    )
                    if member.type_args
                    else NameT(
                        name=member.chain.member,
                        qualifier=member.chain,
                        span=member.span,
                        node_id=member.node_id,
                    )
                )
                resolved = self._env.resolve_type_expr(
                    reference, span=member.span, type_vars=type_vars
                )
                if not isinstance(resolved, RecordType):
                    raise AglTypeError(
                        f"Enum member reference '{member.chain.member}' must name a record.",
                        span=member.span,
                    )
                members.append(resolved)
                member_spans.append(member.span)
                continue
            vd = member
            fields: dict[str, Type] = {}
            seen_fields: dict[str, SourceSpan] = {}
            for fd in vd.fields:
                if fd.name in seen_fields:
                    raise AglTypeError(
                        f"Duplicate field '{fd.name}' in variant '{stmt.name}.{vd.name}'.",
                        span=fd.span,
                    )
                seen_fields[fd.name] = fd.span
                fields[fd.name] = self._resolve_field_type(fd, type_vars=type_vars)
            captured_params = tuple(
                param
                for param in stmt.type_params
                if any(param in free_type_vars(field_type) for field_type in fields.values())
            )
            decl_id = _member_identity(stmt, vd, module_id)
            member_def = TypeDef(
                kind="record",
                name=vd.name,
                module_id=module_id,
                scope_path=member_scope_path,
                type_params=captured_params,
                fields=tuple(fields.items()),
                field_mutability=tuple(fd.mutable for fd in vd.fields),
                decl_node_id=decl_id,
                is_inline_enum_member=True,
            )
            member_defs.append(member_def)
            self._replace_inline_member_handle(stmt, vd, captured_params)
            members.append(
                RecordType(
                    name=vd.name,
                    type_args=tuple(TypeVarType(param) for param in captured_params),
                    module_id=module_id,
                    scope_path=member_scope_path,
                    decl_id=decl_id,
                )
            )
            member_spans.append(vd.span)
        seen_declarations: set[int] = set()
        seen_names: set[str] = set()
        for record_member, span in zip(members, member_spans, strict=True):
            if record_member.decl_id in seen_declarations:
                raise AglTypeError(
                    "Enum members must name distinct record declarations.", span=span
                )
            if record_member.name in seen_names:
                raise AglTypeError(
                    "Enum members must have distinct terminal names; "
                    f"'{record_member.name}' is repeated.",
                    span=span,
                )
            seen_declarations.add(record_member.decl_id)
            seen_names.add(record_member.name)
        return (
            TypeDef(
                kind="enum",
                name=bare_name,
                module_id=module_id,
                scope_path=scope_path,
                type_params=stmt.type_params,
                members=tuple(members),
                is_builtin=stmt.is_builtin,
                decl_node_id=enum_decl_id,
            ),
            tuple(member_defs),
        )

    def _register_enum_constructor_field_kinds(self, stmt: EnumDef) -> None:
        """Keep the current enum-constructor surface keyed by its member names."""
        module_id = self._module_id
        scope_path = tuple(segment.name for segment in stmt.scope_path)
        bare_name = _bare_name(stmt.name)
        for member in stmt.members:
            if not isinstance(member, VariantDef):
                continue
            self._env.register_constructor_field_kinds(
                member.name,
                tuple((fd.name, fd.kind) for fd in member.fields),
                scope_path=(*scope_path, bare_name),
                module_id=module_id,
                decl_id=_member_identity(stmt, member, module_id),
            )

    def _build_exception(self, stmt: ExceptionDef) -> None:
        """Resolve and register an exception's own ``TypeDef`` (no ordering).

        Reuses the record-path field resolution and duplicate-own-field
        check.  ``base`` is resolved to the base declaration's own identity —
        no ordering is required to do so, since only the base's *identity*
        (not its shape) is needed here.  Own-vs-inherited field duplication
        and constructor-callability are checked later, once every
        exception's shape is buildable (see :meth:`_finalize_exceptions`).
        """
        base_type: ExceptionType | None = None
        if stmt.base is not None:
            resolved_base = self._env.resolve_named_type(stmt.base, span=stmt.span)
            if not isinstance(resolved_base, ExceptionType):
                raise AglTypeError(
                    f"Exception '{stmt.name}' extends unknown exception '{stmt.base}'.",
                    span=stmt.span,
                )
            base_type = resolved_base
        fields: dict[str, Type] = {}
        seen_fields: dict[str, SourceSpan] = {}
        for fd in stmt.fields:
            if fd.mutable:
                raise AglTypeError(
                    f"Exception field '{fd.name}' in '{stmt.name}' cannot be mutable.",
                    span=fd.span,
                )
            if fd.name in seen_fields:
                raise AglTypeError(
                    f"Duplicate field '{fd.name}' in exception '{stmt.name}'.",
                    span=fd.span,
                )
            seen_fields[fd.name] = fd.span
            fields[fd.name] = self._resolve_field_type(fd)
        module_id = self._module_id
        scope_path = tuple(segment.name for segment in stmt.scope_path)
        bare_name = _bare_name(stmt.name)
        # Own field kinds honor each field's declared @pos/@std/@named marker —
        # exactly like a record's fields — in declaration order, parallel to
        # ``fields`` above.  Stored as ``ParamKind.value`` strings (see
        # ``TypeDef.field_kinds``: ``semantics`` may not import ``syntax``).
        # Inheriting the base's kinds through the extends chain is
        # ``TypeTable.exception_field_kinds``'s job (walked on demand from
        # ``TypeDef.base``, no build-ordering step needed).
        typedef = TypeDef(
            kind="exception",
            name=bare_name,
            module_id=module_id,
            scope_path=scope_path,
            fields=tuple(fields.items()),
            abstract=stmt.base is None,
            base=None if base_type is None else base_type.decl_id,
            field_kinds=tuple(fd.kind.value for fd in stmt.fields),
            is_builtin=stmt.is_builtin,
            decl_node_id=_decl_identity(module_id, scope_path, bare_name, stmt.node_id),
        )
        self._resolved_defs[stmt.name] = typedef
        self._env.type_table.register(typedef)

    def _finalize_exceptions(self) -> None:
        """Post-pass: reject own-vs-inherited field name collisions.

        Deferred until every exception's own ``TypeDef`` (and therefore its
        base's, however deep the ``extends`` chain) is registered — there is
        no build-ordering step in :meth:`_build_exception` (see the module
        docstring).  A cyclic ``extends`` chain has no independent evidence to
        ever become inhabited, so it is always rejected by the whole-program
        inhabitation pre-pass (``typecheck/program.py``) before Phase 3
        re-checks this module at all, and
        :meth:`~agm.agl.semantics.type_table.TypeTable.exception_fields` never
        hits its internal cycle guard here.
        """
        for item in self._exception_defs.values():
            if item.base is None:
                continue
            module_id = self._module_id
            scope_path = tuple(segment.name for segment in item.scope_path)
            typedef = self._env.type_table.get(module_id, _bare_name(item.name), scope_path)
            assert typedef is not None, "compiler bug: exception is not registered"
            assert typedef.base is not None
            base_typedef = self._env.type_table.get_by_id(typedef.base)
            assert base_typedef is not None, (
                f"compiler bug: exception base {typedef.base!r} has no registered TypeDef"
            )
            base_handle = base_typedef.handle()
            assert isinstance(base_handle, ExceptionType)
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
        expected_contracts: Mapping[str, BuiltinTypeContract],
        *,
        base_type: ExceptionType | None = None,
    ) -> None:
        """Check a ``builtin`` declaration against its structural host contract.

        A builtin may live at any source path, but its enum members must be
        inline so host-minted values and source constructors share the member
        identities assigned by the builtin declaration. Nominal types inside
        fields and an exception's base remain contract-bearing and are
        normalized by :func:`contract_for_typedef` relative to the declaration's
        own frame.
        """
        if isinstance(stmt, EnumDef) and any(
            isinstance(member, VariantRef) for member in stmt.members
        ):
            raise AglTypeError(
                f"Builtin type '{stmt.name}' has an invalid definition.",
                span=stmt.span,
            )
        bare_name = _bare_name(stmt.name)
        expected = expected_contracts[bare_name]
        actual = contract_for_typedef(typedef, self._env.type_table, base_type=base_type)
        if actual != expected:
            raise AglTypeError(
                f"Builtin type '{stmt.name}' has an invalid definition.",
                span=stmt.span,
            )

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
        module_id = self._module_id
        scope_path = tuple(segment.name for segment in stmt.scope_path)
        bare_name = _bare_name(stmt.name)
        typedef = TypeDef(
            kind="record",
            name=bare_name,
            module_id=module_id,
            scope_path=scope_path,
            type_params=type_params,
            fields=tuple(fields.items()),
            field_mutability=tuple(fd.mutable for fd in stmt.fields),
            is_builtin=stmt.is_builtin,
            # Same identity as the handle template registered in phase 1
            # (:meth:`_register_record_or_enum_handle`), so the TypeDef and
            # every instantiated handle agree on which declaration they name.
            decl_node_id=template.decl_id,
        )
        self._resolved_defs[stmt.name] = typedef
        self._env.type_table.register(typedef)
        field_names = tuple(fields.keys())
        field_templates = tuple(fields.values())
        sig = ConstructorSignature(
            owner_name=stmt.name,
            field_names=field_names,
            field_templates=field_templates,
            result_template=template,
            type_params=type_params,
        )
        self._env.register_constructor_signature(sig)
        # Register field kinds for the generic record constructor, under the
        # same owning identity as the TypeDef just above.
        generic_record_field_kinds = tuple((fd.name, fd.kind) for fd in stmt.fields)
        self._env.register_constructor_field_kinds(
            bare_name,
            generic_record_field_kinds,
            scope_path=scope_path,
            module_id=module_id,
            decl_id=template.decl_id,
        )

    def _build_generic_enum(self, stmt: EnumDef) -> None:
        """Resolve a generic enum's member-record definitions and constructors."""
        type_params = stmt.type_params
        gdef = self._env.get_generic_type(stmt.name)
        assert gdef is not None, f"compiler bug: generic enum {stmt.name!r} not pre-registered"
        template = gdef.template
        assert isinstance(template, EnumType)
        typedef, member_defs = self._build_enum_members(stmt, type_vars=frozenset(type_params))
        typedef = replace(typedef, decl_node_id=template.decl_id)
        self._resolved_defs[stmt.name] = typedef
        for member_def in member_defs:
            self._env.type_table.register(member_def)
        self._env.type_table.register(typedef)
        for member in stmt.members:
            if not isinstance(member, VariantDef):
                continue
            member_type = next(
                member_type for member_type in typedef.members if member_type.name == member.name
            )
            fields = self._env.type_table.record_fields(member_type)
            self._env.register_constructor_signature(
                ConstructorSignature(
                    owner_name=member_type.name,
                    field_names=tuple(fields),
                    field_templates=tuple(fields.values()),
                    result_template=member_type,
                    type_params=tuple(
                        arg.name for arg in member_type.type_args if isinstance(arg, TypeVarType)
                    ),
                )
            )
        self._register_enum_constructor_field_kinds(stmt)

    def _validate_alias(self, stmt: TypeAlias) -> None:
        """Validate that the alias target resolves without cycles.

        A parameterized alias body may reference its own type parameters, so
        they are in scope as type variables during validation.
        """
        resolved = self._env.resolve_type_expr(
            stmt.type_expr,
            span=stmt.span,
            type_vars=frozenset(stmt.type_params),
        )
        self._env.freeze_alias(stmt.name, resolved, type_params=stmt.type_params)
