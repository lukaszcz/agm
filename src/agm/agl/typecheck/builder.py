"""Standalone per-module type-table builder for the AgL type-checking pass.

``_TypeBuilder`` collects ``record``/``enum``/``exception``/alias declarations
and registers them into a ``TypeEnvironment``.  It was extracted from
``typecheck/checker.py`` (where it lived as the first pass of the
bidirectional type checker) so that ``typecheck/graph.py`` can import it
without pulling in the full ``_Checker``.

Nominal types (``RecordType``/``EnumType``) are lightweight handles with no
embedded field/variant data, so declaration order does not matter: every
reference — forward or backward — resolves to a valid handle.  Building is
therefore two simple, order-free phases:

Phase 1 (``collect_shells_only``)
    Register every declared name's FINAL handle (or, for a generic
    declaration, its ``GenericTypeDef``) and every alias target.  A handle
    carries no shape, so this phase never needs revisiting.
Phase 2 (the loop in ``collect``)
    Resolve each declaration's field/variant type expressions, in source
    order, into a ``TypeDef`` registered in the shared ``TypeTable``.
    Exceptions are the one exception (no pun intended): ``ExceptionType``
    still embeds its fields directly (a later step unifies exceptions into
    the handle model), so building an exception that ``extends`` a
    not-yet-built base must build that base first — a minimal,
    exception-only ordering step, not a general one.

Recursive nominal types are not yet supported.  Because handles make forward
references trivially resolve, nothing in phase 2 itself detects a
declaration that structurally contains itself; a dedicated pass
(``_check_recursive_types``) runs after all bodies are resolved and rejects
one, replicating today's diagnostics exactly.  This is a temporary
restriction pending an inhabitation-based analysis.
"""

from __future__ import annotations

from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, ModuleId
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS, TypeDef
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPE_NAMES,
    DictType,
    EnumType,
    ExceptionType,
    ListType,
    RecordType,
    Type,
    TypeVarType,
)
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    Param,
    ParamKind,
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
# Pre-pass: collect and validate type declarations
# ---------------------------------------------------------------------------


class _TypeBuilder:
    """Collect record/enum/exception/alias declarations and validate them.

    Populates a ``TypeEnvironment`` with all user-declared types.  Raises
    ``AglTypeError`` on:
    - Duplicate type names (user vs user, or user shadowing a built-in).
    - Duplicate record fields or enum variants/fields.
    - Unknown type references inside field/variant definitions.
    - Recursive records, enums, or exceptions (temporary ban).
    - Alias cycles.

    See the module docstring for the two-phase, order-free build strategy.
    """

    def __init__(self, env: TypeEnvironment, module_id: ModuleId = ENTRY_ID) -> None:
        self._env = env
        self._module_id = module_id
        # Track user-declared names → declaration span (excludes built-ins).
        self._declared: dict[str, SourceSpan] = {}
        # Index of record/enum/exception definitions, for phase-2 body
        # resolution and the post-pass recursion check.
        self._record_defs: dict[str, RecordDef] = {}
        self._enum_defs: dict[str, EnumDef] = {}
        self._exception_defs: dict[str, ExceptionDef] = {}
        # Exceptions already given their final (base-inclusive) field set —
        # the minimal exception-only ordering step (see module docstring).
        self._exceptions_built: set[str] = set()

    def collect(self, program: Program) -> None:
        """Scan *program* and populate ``self._env``.

        Phase 1 registers every declaration's name and handle
        (:meth:`collect_shells_only`); phase 2 resolves each declaration's
        body, in source order, with no dependency ordering (see the module
        docstring); a final pass rejects any declaration that structurally
        contains itself (the temporary recursion ban).
        """
        self.collect_shells_only(program)

        for item in program.body.items:
            if isinstance(item, RecordDef):
                self._build_record(item)
            elif isinstance(item, EnumDef):
                self._build_enum(item)
            elif isinstance(item, ExceptionDef):
                self._build_exception_ordered(item.name)
            elif isinstance(item, TypeAlias):
                self._validate_alias(item)

        self._check_recursive_types(program)

    def collect_shells_only(self, program: Program) -> None:
        """Register phase-1 declarations: names, handles, and alias targets.

        Public interface for the graph pre-pass (``graph.py``) which needs to
        register every module's declarations before resolving any body.
        Non-generic records/enums get their FINAL handle registered directly
        (handles carry no shape, so there is nothing left to "finish"
        later); generic records/enums get their ``GenericTypeDef`` (name,
        type params, and a handle template stamped with ``TypeVarType``
        args) registered instead — likewise final, since the template
        carries no shape either.  Exceptions still register an empty-fields
        shell (``ExceptionType`` embeds its fields; the shell is replaced by
        the fully-built type in phase 2).
        """
        for item in program.body.items:
            if isinstance(item, RecordDef):
                self._register_name(item.name, item.span, is_builtin=item.is_builtin)
                self._env.unregister_name(item.name)
                self._register_record_or_enum_handle(item, is_enum=False)
                self._record_defs[item.name] = item
            elif isinstance(item, EnumDef):
                self._register_name(item.name, item.span, is_builtin=item.is_builtin)
                self._env.unregister_name(item.name)
                self._register_record_or_enum_handle(item, is_enum=True)
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

    def _register_record_or_enum_handle(
        self, item: RecordDef | EnumDef, *, is_enum: bool
    ) -> None:
        module_id = PRELUDE_ID if item.is_builtin else self._module_id
        if item.type_params:
            type_args = tuple(TypeVarType(p) for p in item.type_params)
            template: RecordType | EnumType = (
                EnumType(name=item.name, type_args=type_args, module_id=module_id)
                if is_enum
                else RecordType(name=item.name, type_args=type_args, module_id=module_id)
            )
            gdef = GenericTypeDef(
                kind="enum" if is_enum else "record",
                type_params=item.type_params,
                template=template,
            )
            self._env.register_generic_type(item.name, gdef)
        else:
            handle: RecordType | EnumType = (
                EnumType(name=item.name, module_id=module_id)
                if is_enum
                else RecordType(name=item.name, module_id=module_id)
            )
            self._env.register_type(item.name, handle)

    def validate_alias(self, stmt: TypeAlias) -> None:
        """Public proxy for :meth:`_validate_alias`."""
        self._validate_alias(stmt)

    def build_record(self, name: str) -> None:
        """Resolve and register the named record's body.

        Public entry point for the graph pre-pass, which resolves every
        module's declarations in a fixed order (no dependency ordering —
        see the module docstring).
        """
        self._build_record(self._record_defs[name])

    def build_enum(self, name: str) -> None:
        """Resolve and register the named enum's body. See :meth:`build_record`."""
        self._build_enum(self._enum_defs[name])

    def build_exception(self, name: str) -> None:
        """Resolve and register the named exception, building its base first."""
        self._build_exception_ordered(name)

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

    def _build_exception_ordered(
        self, name: str, _in_progress: frozenset[str] = frozenset()
    ) -> None:
        """Ensure exception *name*'s base is built before *name* itself.

        ``_build_exception`` copies its base's embedded fields, so the base
        must already be fully built.  This is the minimal, exception-only
        ordering step described in the module docstring — NOT a general
        build-ordering mechanism.  A cyclic ``extends`` chain is left
        unresolved past the point of re-entry: it is always rejected by
        :meth:`_check_recursive_types` once all bodies have been resolved;
        this guard only prevents unbounded recursion before that check runs.
        """
        if name in self._exceptions_built or name in _in_progress:
            return
        stmt = self._exception_defs[name]
        if stmt.base is not None and stmt.base in self._exception_defs:
            self._build_exception_ordered(stmt.base, _in_progress | {name})
        self._build_exception(stmt)
        self._exceptions_built.add(name)

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
        module_id = PRELUDE_ID if stmt.is_builtin else self._module_id
        typedef = TypeDef(
            kind="record",
            name=stmt.name,
            module_id=module_id,
            fields=tuple(fields.items()),
        )
        self._validate_builtin_record_or_enum_shape(stmt, typedef)
        self._env.type_table.register(typedef)
        # Register field kinds for this record constructor.
        field_kinds = tuple((fd.name, fd.kind) for fd in stmt.fields)
        self._env.register_constructor_field_kinds(stmt.name, None, field_kinds)

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
                vfields[fd.name] = self._resolve_field_type(fd)
            variants[vd.name] = vfields
        module_id = PRELUDE_ID if stmt.is_builtin else self._module_id
        typedef = TypeDef(
            kind="enum",
            name=stmt.name,
            module_id=module_id,
            variants=tuple(
                (vname, tuple(vfields.items())) for vname, vfields in variants.items()
            ),
        )
        self._validate_builtin_record_or_enum_shape(stmt, typedef)
        self._env.type_table.register(typedef)
        # Register field kinds for each variant constructor.
        for vd in stmt.variants:
            vfield_kinds = tuple((fd.name, fd.kind) for fd in vd.fields)
            self._env.register_constructor_field_kinds(stmt.name, vd.name, vfield_kinds)

    def _build_exception(self, stmt: ExceptionDef) -> None:
        fields: dict[str, Type] = {}
        if stmt.base is not None:
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
            fields[fd.name] = self._resolve_field_type(fd)
        typ = ExceptionType(
            name=stmt.name,
            fields=fields,
            abstract=stmt.base is None,
        )
        self._validate_builtin_exception_shape(stmt, typ)
        self._env.register_type(stmt.name, typ)
        # Register field kinds for this exception constructor (user fields only,
        # trace_id excluded).  Inherit base field kinds first, then add own fields.
        # Abstract exceptions (base is None) have no constructor, so nothing to register.
        if stmt.base is not None:
            base_registered = self._env.get_constructor_field_kinds(stmt.base, None)
            if base_registered is not None:
                # Base has registered field kinds (concrete exception): use them directly.
                base_inherited = base_registered
            else:
                # Base is abstract (e.g. the built-in Exception root) and has no
                # registered field kinds.  Derive NAMED_ONLY kinds from its field
                # names, excluding trace_id which is auto-filled at runtime.
                assert isinstance(base_type, ExceptionType)
                base_inherited = tuple(
                    (fname, ParamKind.NAMED_ONLY)
                    for fname in base_type.fields
                    if fname != "trace_id"
                )
            own_field_kinds = tuple((fd.name, fd.kind) for fd in stmt.fields)
            self._env.register_constructor_field_kinds(
                stmt.name, None, base_inherited + own_field_kinds
            )

    def _validate_builtin_record_or_enum_shape(
        self, stmt: RecordDef | EnumDef, typedef: TypeDef
    ) -> None:
        if not stmt.is_builtin:
            return
        expected = BUILTIN_PRELUDE_TYPE_DEFS.get(stmt.name)
        assert expected is not None
        if typedef != expected:
            raise AglTypeError(
                f"Builtin type '{stmt.name}' has an invalid definition.",
                span=stmt.span,
            )

    def _validate_builtin_exception_shape(self, stmt: ExceptionDef, typ: ExceptionType) -> None:
        if not stmt.is_builtin:
            return
        expected = BUILTIN_EXCEPTIONS.get(stmt.name)
        assert expected is not None
        if typ.fields != expected.fields:
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
        return self._env.resolve_type_expr(fd.type_expr, span=fd.span, type_vars=type_vars)

    def _build_generic_record(self, stmt: RecordDef) -> None:
        """Resolve a generic record's fields and register its TypeDef + constructor.

        The ``GenericTypeDef`` itself (name, type params, handle template)
        was already registered in phase 1 so that forward references to
        this generic type resolve regardless of declaration order.
        """
        type_vars = frozenset(stmt.type_params)
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
        self._env.type_table.register(
            TypeDef(
                kind="record",
                name=stmt.name,
                module_id=self._module_id,
                type_params=stmt.type_params,
                fields=tuple(fields.items()),
            )
        )
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
        # Register field kinds for the generic record constructor.
        generic_record_field_kinds = tuple((fd.name, fd.kind) for fd in stmt.fields)
        self._env.register_constructor_field_kinds(stmt.name, None, generic_record_field_kinds)

    def _build_generic_enum(self, stmt: EnumDef) -> None:
        """Resolve a generic enum's variants and register its TypeDef + constructors.

        See :meth:`_build_generic_record` for why the ``GenericTypeDef``
        itself is not (re-)registered here.
        """
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
                vfields[fd.name] = self._resolve_field_type(fd, type_vars=type_vars)
            variants[vd.name] = vfields
        gdef = self._env.get_generic_type(stmt.name)
        assert gdef is not None, f"compiler bug: generic enum {stmt.name!r} not pre-registered"
        template = gdef.template
        assert isinstance(template, EnumType)
        self._env.type_table.register(
            TypeDef(
                kind="enum",
                name=stmt.name,
                module_id=self._module_id,
                type_params=stmt.type_params,
                variants=tuple(
                    (vname, tuple(vfields.items())) for vname, vfields in variants.items()
                ),
            )
        )
        # Register one ConstructorSignature and field kinds per variant.
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
            # Register field kinds for this generic enum variant constructor.
            vfield_kinds = tuple((fd.name, fd.kind) for fd in vd.fields)
            self._env.register_constructor_field_kinds(stmt.name, vd.name, vfield_kinds)

    def _validate_alias(self, stmt: TypeAlias) -> None:
        """Validate that the alias target resolves without cycles.

        A parameterized alias body may reference its own type parameters, so
        they are in scope as type variables during validation.
        """
        self._env.resolve_type_expr(
            stmt.type_expr, span=stmt.span, type_vars=frozenset(stmt.type_params)
        )

    # ------------------------------------------------------------------
    # Temporary recursion ban
    # ------------------------------------------------------------------
    #
    # Recursive nominal types are not yet supported in AgL — a later step
    # replaces this blanket rejection with an inhabitation-based analysis
    # that allows well-founded recursion.  Since every reference now resolves
    # to a handle regardless of build order, nothing above detects a
    # declaration that structurally contains itself; this dedicated pass
    # runs after all bodies are resolved and replicates today's exact
    # diagnostics.

    def _check_recursive_types(self, program: Program) -> None:
        in_progress: set[str] = set()
        done: set[str] = set()

        def visit_type(t: Type) -> None:
            if isinstance(t, RecordType):
                # Type arguments are visited unconditionally (even for a
                # cross-module or not-yet-declared handle): a generic
                # argument such as ``Box[A]`` is where the cycle actually
                # lives, not in Box's own (non-recursive) template body.
                for ta in t.type_args:
                    visit_type(ta)
                if t.module_id == self._module_id and t.name in self._record_defs:
                    visit_record(t.name)
            elif isinstance(t, EnumType):
                for ta in t.type_args:
                    visit_type(ta)
                if t.module_id == self._module_id and t.name in self._enum_defs:
                    visit_enum(t.name)
            elif isinstance(t, ExceptionType):
                if t.name in self._exception_defs:
                    visit_exception(t.name)
            elif isinstance(t, ListType):
                visit_type(t.elem)
            elif isinstance(t, DictType):
                visit_type(t.value)
            # primitives, function/agent/unit/typevar/bottom types never
            # embed a user declaration.

        def visit_record(name: str) -> None:
            if name in done:
                return
            if name in in_progress:
                raise AglTypeError(
                    f"Record type '{name}' is directly or indirectly recursive. "
                    "Recursive types are not supported in AgL.",
                    span=self._declared[name],
                )
            in_progress.add(name)
            typedef = self._env.type_table.get(self._module_id, name)
            if typedef is not None:
                for _fname, ftype in typedef.fields:
                    visit_type(ftype)
            in_progress.discard(name)
            done.add(name)

        def visit_enum(name: str) -> None:
            if name in done:
                return
            if name in in_progress:
                raise AglTypeError(
                    f"Enum type '{name}' is directly or indirectly recursive. "
                    "Recursive types are not supported in AgL.",
                    span=self._declared[name],
                )
            in_progress.add(name)
            typedef = self._env.type_table.get(self._module_id, name)
            if typedef is not None:
                for _vname, vfields in typedef.variants:
                    for _fname, ftype in vfields:
                        visit_type(ftype)
            in_progress.discard(name)
            done.add(name)

        def visit_exception(name: str) -> None:
            if name in done:
                return
            if name in in_progress:
                raise AglTypeError(
                    f"Exception type '{name}' is directly or indirectly recursive.",
                    span=self._declared[name],
                )
            in_progress.add(name)
            stmt = self._exception_defs[name]
            if stmt.base is not None and stmt.base in self._exception_defs:
                visit_exception(stmt.base)
            for fd in stmt.fields:
                visit_type(self._resolve_field_type(fd))
            in_progress.discard(name)
            done.add(name)

        for item in program.body.items:
            if isinstance(item, RecordDef):
                visit_record(item.name)
            elif isinstance(item, EnumDef):
                visit_enum(item.name)
            elif isinstance(item, ExceptionDef):
                visit_exception(item.name)
