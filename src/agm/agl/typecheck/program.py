"""Program-level type-checking pass for the AgL module system.

``check_program(resolved, capabilities)`` runs the full type-checking pass
over a :class:`~agm.agl.scope.program.ResolvedProgram`, producing a
:class:`CheckedProgram`.

Algorithm
---------
1. **Program type pre-pass** — collect ALL public type declarations across every
   module and resolve their bodies (stamping each
   ``RecordType``/``EnumType``/``ExceptionType`` handle with the owning
   ``ModuleId``), building the shared ``program_type_table``.
   ``RecordType``/``EnumType``/``ExceptionType`` carry no field/variant data
   of their own (their shapes live in the shared ``TypeTable``), so a
   reference to another module's type is a valid handle whether or not that
   type's own body has been resolved yet — body resolution is therefore
   order-free.

   The pre-pass is genuinely whole-program two-phase:

   a. **Headers** — every declared name's handle (or, for a generic
      declaration, its ``GenericTypeDef``) is registered into the shared
      ``program_type_table`` first; type aliases are registered as lazy program
      alias keys because aliases are transparent and have no handle shell.
   b. **Bodies, order-free** — each declaration's body (field/variant type
      expressions) is resolved in a fixed deterministic order (sorted by
      ``(ModuleId.segments, name)``), with no dependency-ordering
      constraint — nominal cycles (same-module, mutual, or spanning any
      number of modules) are legal, since every nominal reference resolves to
      a handle regardless of build order; alias references resolve lazily and
      still reject transparent alias cycles.
   c. **Inhabitation** — once every body is resolved, the whole-program
      inhabitation fixpoint
      (:func:`~agm.agl.semantics.analyses.compute_uninhabited`) rejects the
      first declaration (across the whole program) that has no finite value.
      This is the only inhabitation check the pipeline runs; the per-module
      ``_TypeBuilder`` pass that resolves each declaration's body never
      repeats it.

2. **Program function headers** — resolve parameter and explicit return
   annotations for top-level ``FuncDef`` declarations in every module,
   producing a declaration-node-id-keyed signature table.

3. **Import-SCC candidate inference** — consume the loader's graph SCCs in
   reverse topological order. Ordinary dependency SCCs publish their closed
   signatures before their importers are considered. Each candidate function
   dependency SCC publishes only closed unannotated signatures; one resulting
   cycle builds a single cross-module function graph.

4. **Authoritative per-module type-check** — after every signature is concrete,
   recheck each module body with its module-aware
   :class:`~agm.agl.typecheck.env.TypeEnvironment`. This pass alone publishes
   checked node types, calls, contracts, bindings, and warnings.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Generic, Mapping, TypeVar, cast

from agm.agl.artifact_cache import (
    module_fingerprints,
    retain_checked_modules,
    retained_checked_modules,
    retained_module_sources,
)
from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import ImportEnv
from agm.agl.scope.program import ResolvedProgram
from agm.agl.scope.symbols import ModuleResolution
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.analyses import compute_uninhabited, uninhabitable_message
from agm.agl.semantics.type_table import (
    DeclId,
    DeclKey,
    TypeDef,
    TypeTable,
    create_seeded_type_table,
    decl_def_sort_key,
)
from agm.agl.semantics.types import EnumType, ExceptionType, RecordType, Type
from agm.agl.syntax.nodes import (
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    FuncDef,
    LetDecl,
    Program,
    RecordDef,
    TypeAlias,
    simple_let_pattern_name,
    static_function_items,
    static_items,
    static_type_items,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.builder import _TypeBuilder
from agm.agl.typecheck.checker import _check_prepared_module, prepare_module_headers
from agm.agl.typecheck.declaration_validation import (
    validate_builtin_declaration_uniqueness,
    validate_builtin_method_ownership,
    validate_method_declaration_collisions,
)
from agm.agl.typecheck.env import (
    AglTypeError,
    CheckedModule,
    ConstructorSignature,
    FunctionSignature,
    GenericAliasDef,
    GenericTypeDef,
    TypeEnvironment,
    _assert_checked_types_closed,
    assert_checked_output_closed,
)
from agm.agl.typecheck.function_inference import (
    CandidateModule,
    FunctionReturnSource,
    FunctionSignatureRecord,
    ModuleCandidateComponent,
    candidate_records_for,
    infer_module_component_candidates,
    register_method_header,
    resolve_function_header,
)
from agm.agl.zones import ParamZone

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


_T = TypeVar("_T")


class _DeclKeyDict(dict[DeclKey, _T], Generic[_T]):
    """Structured declaration map with root-key lookup compatibility."""

    @staticmethod
    def _normalize(key: object) -> object:
        if isinstance(key, tuple) and len(key) == 2:
            module_id, name = key
            return (cast(ModuleId, module_id), (), cast(str, name))
        return key

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._normalize(key))

    def __getitem__(self, key: DeclKey | tuple[ModuleId, str]) -> _T:
        normalized = self._normalize(key)
        assert isinstance(normalized, tuple)
        return super().__getitem__(normalized)


@dataclass(frozen=True, slots=True)
class CheckedProgram:
    """Immutable output of :func:`check_program`.

    ``modules``
        Maps each :class:`~agm.agl.modules.ids.ModuleId` to its
        :class:`CheckedModule`.
    ``entry_id``
        The entry module's identity, as the loaded graph keys it: its owning
        package's declared module id, or
        :data:`~agm.agl.modules.ids.ENTRY_ID` for a source with no module
        identity.
    ``program_type_table``
        Whole-program type table mapping ``(ModuleId, scope_path, name)`` to the fully-built
        :class:`~agm.agl.semantics.types.Type` object stamped with the owning
        ``module_id``.  Built in the program pre-pass; shared (read-only) across
        all per-module environments.
    ``warnings``
        All non-fatal type-check diagnostics collected across all modules, in
        module-traversal order.
    ``import_sccs``
        Loader-computed reverse-topological import components, retained for
        dependency-ordered lowering after this pass's presentation ordering.
    ``runtime_modules``
        Entry-reachable modules through explicit source imports and exports.
        Loader-injected standard-library edges do not add dry-run call sites.
    """

    modules: dict[ModuleId, CheckedModule]
    entry_id: ModuleId
    program_type_table: dict[DeclKey, Type]
    warnings: tuple[Diagnostic, ...]
    capabilities: HostCapabilities | None = None
    import_sccs: tuple[tuple[ModuleId, ...], ...] = ()
    resource_roots: Mapping[ModuleId, Path | None] = field(default_factory=dict)
    runtime_modules: frozenset[ModuleId] | None = None
    module_fingerprints: Mapping[ModuleId, bytes] = field(default_factory=dict)


def program_funcdefs(
    modules: Mapping[ModuleId, CheckedModule],
) -> Iterator[tuple[ModuleId, CheckedModule, FuncDef]]:
    """Yield every ``program def`` declaration across *modules*, with its owning module.

    The one walk over a checked program's ``program def`` declarations, so
    every table keyed on a program -- the lowerer's symbol, function and
    signature tables, and the driver's discovered declaration infos -- is
    built from the same traversal in the same order.
    """
    return (
        (module_id, checked_module, item)
        for module_id, checked_module in modules.items()
        for item in static_items(checked_module.resolved.program.body.items)
        if isinstance(item, FuncDef) and item.is_program
    )


def _assert_checked_module_closed(module: CheckedModule) -> None:
    """Assert that one program module is safe to pass to the lowerer."""
    assert_checked_output_closed(
        node_types=module.node_types,
        contract_specs=module.contract_specs,
        call_sites=module.call_sites,
        function_signatures=module.function_signatures,
        cast_specs=module.cast_specs,
        argument_bindings=module.argument_bindings,
        let_matched_types=module.let_matched_types,
        explicit_builtin_targets=module.explicit_builtin_targets,
        owner=f"checked module {module.module_id.path_str()}",
    )
    module.type_env.assert_closed()


def assert_checked_program_closed(
    checked: CheckedProgram, reused_modules: frozenset[ModuleId] = frozenset()
) -> None:
    """Assert that all program-level checked output is safe to lower.

    A module named in *reused_modules* arrived unchanged from an earlier
    compilation that already sealed it. Its artifact is the very object that
    was validated then and is immutable, so re-asserting it can only reach the
    same conclusion; the whole-program tables below are rebuilt every call and
    are always validated.
    """
    for module_id, module in checked.modules.items():
        if module_id in reused_modules:
            continue
        _assert_checked_module_closed(module)
    _assert_checked_types_closed(checked.program_type_table.values(), owner="checked module graph")
    # The remaining whole-program tables (the shared TypeTable and the generic /
    # alias / constructor maps) are the same instances on every module env, so
    # validate them once here rather than on every per-module seal.
    checked.modules[checked.entry_id].type_env.assert_shared_tables_closed()


# ---------------------------------------------------------------------------
# Program type pre-pass helpers
# ---------------------------------------------------------------------------


def _decl_key(module_id: ModuleId, item: RecordDef | EnumDef | ExceptionDef | TypeAlias) -> DeclKey:
    """Return a declaration's structured nominal identity."""
    return (module_id, tuple(segment.name for segment in item.scope_path), item.name)


def _collect_shells_only(builder: _TypeBuilder, program: object) -> None:
    """Run only phase 1 (shell registration) of ``_TypeBuilder.collect``.

    Delegates to :meth:`~agm.agl.typecheck.builder._TypeBuilder.collect_shells_only`,
    the public API added to ``_TypeBuilder`` for this purpose.
    """
    assert isinstance(program, Program)
    builder.collect_shells_only(program)


def _sync_program_env_extensions(
    mid: ModuleId,
    env: TypeEnvironment,
    program_type_table: dict[DeclKey, Type],
    program_generic_table: dict[DeclKey, GenericTypeDef],
    program_ctor_sig_table: dict[DeclKey, ConstructorSignature],
    program_ctor_field_kinds_table: dict[DeclKey, tuple[tuple[str, ParamZone], ...]],
) -> None:
    """Copy reconciled type and constructor metadata into the program tables."""
    for _type_name, typ in env.non_builtin_type_items():
        assert isinstance(typ, (RecordType, EnumType, ExceptionType))
        key = (mid, typ.scope_path, typ.name)
        program_type_table[key] = typ
        program_generic_table.pop(key, None)
    for _generic_name, gdef in env.all_generic_types().items():
        key = (mid, gdef.template.scope_path, gdef.template.name)
        program_generic_table[key] = gdef
        program_type_table.pop(key, None)
    for key, sig in env.all_constructor_sigs():
        program_ctor_sig_table[key] = sig
    for key, kinds in env.all_constructor_field_kinds():
        program_ctor_field_kinds_table[key] = kinds


def _resolve_body_for_one(
    mid: ModuleId,
    key: DeclKey,
    per_module_builders: dict[ModuleId, _TypeBuilder],
    program_type_table: dict[DeclKey, Type],
    program_generic_table: dict[DeclKey, GenericTypeDef],
    program_alias_table: dict[DeclKey, GenericAliasDef],
    program_ctor_sig_table: dict[DeclKey, ConstructorSignature],
    program_ctor_field_kinds_table: dict[DeclKey, tuple[tuple[str, ParamZone], ...]],
    resolved: ResolvedProgram,
    cross_envs: dict[ModuleId, TypeEnvironment],
) -> None:
    """Resolve the body of one structured type key and update the program table.

    Called once per key in a fixed deterministic order (no dependency
    ordering): every type reference is a handle, valid regardless of whether
    the referenced type's own body has been resolved yet.
    """
    cross_env = cross_envs[mid]
    builder = per_module_builders[mid]

    program = resolved.modules[mid].resolved.program
    assert isinstance(program, Program)

    display_name = "::".join((*key[1], key[2]))
    for item in static_type_items(program.body.items):
        if isinstance(item, RecordDef) and _decl_key(mid, item) == key:
            with cross_env.type_scope(key[1]):
                builder.build_record(display_name)
            t = cross_env.get_type(display_name)
            if t is not None:
                # Non-generic record: update the program table with the fully-built type.
                program_type_table[key] = t
            # Generic record: body registered in _generic_types (no _types entry);
            # program_type_table retains the handle from Step A.  Cross-module generic
            # constructor calls use _program_generic_table / _program_ctor_sig_table instead.
            break
        if isinstance(item, EnumDef) and _decl_key(mid, item) == key:
            with cross_env.type_scope(key[1]):
                builder.build_enum(display_name)
            t = cross_env.get_type(display_name)
            if t is not None:
                program_type_table[key] = t
            break
        if isinstance(item, ExceptionDef) and _decl_key(mid, item) == key:
            with cross_env.type_scope(key[1]):
                builder.build_exception(display_name)
            typ = cross_env.get_type(display_name)
            assert typ is not None, f"Exception type {key[2]!r} not registered"
            program_type_table[key] = typ
            break
        if isinstance(item, TypeAlias) and _decl_key(mid, item) == key:
            with cross_env.type_scope(key[1]):
                type_params = item.type_params
                if type_params:
                    builder.validate_alias(item)
                    template = cross_env.resolve_type_expr(
                        item.type_expr,
                        span=item.span,
                        type_vars=frozenset(type_params),
                    )
                    program_alias_table[key] = GenericAliasDef(
                        type_params=type_params,
                        template=template,
                    )
                else:
                    alias_type = cross_env.resolve_type_expr(item.type_expr, span=item.span)
                    program_type_table[key] = alias_type
            break
    else:
        # Unreachable: called only for keys produced by _collect_all_type_keys,
        # which iterates the same program.body.items.
        raise AssertionError(f"type '{key[2]}' not found in module '{mid}'")  # pragma: no cover

    _sync_program_env_extensions(
        mid,
        cross_env,
        program_type_table,
        program_generic_table,
        program_ctor_sig_table,
        program_ctor_field_kinds_table,
    )


def _collect_all_type_keys(
    resolved: ResolvedProgram,
) -> set[DeclKey]:
    """Collect the set of all user-declared type keys across all modules.

    Returns ``{(ModuleId, scope_path, name)}`` for every ``RecordDef``, ``EnumDef``, and
    ``TypeAlias`` in every module.  This includes type aliases whose resolved type
    is a primitive (e.g. ``type Number = int``).  Builtin-shadowing types are
    never present here because ``_collect_shells_only`` rejects them earlier.

    This set is the fixed order in which Step C below resolves every
    declaration's body (sorted by name path, see
    :func:`~agm.agl.semantics.type_table.decl_def_sort_key`). It is LARGER
    than ``program_type_table`` during the shell-collection step because
    aliases are not yet resolved to shells there — the program table is only
    populated with record/enum shells and is updated with alias resolutions
    as each body is resolved.
    """
    all_keys: set[DeclKey] = set()
    for mid, rmod in resolved.modules.items():
        program = rmod.resolved.program
        assert isinstance(program, Program)
        for item in static_type_items(program.body.items):
            # Builtin/prelude shadowing is rejected in _collect_shells_only
            # (Step A of _build_program_type_table), which is called before this
            # function. Only non-builtin types reach this point.
            all_keys.add(_decl_key(mid, item))
    return all_keys


def _find_type_decl_span(resolved: ResolvedProgram, key: DeclKey) -> SourceSpan | None:
    """Return the declaration span for *key*, or ``None`` if it cannot be found.

    Used to attach a real source span to the whole-program inhabitation error
    (see :func:`_build_program_type_table`): the resolved module ASTs are
    already in hand, so the span is a plain lookup rather than anything
    carried through the type table itself (a ``TypeDef`` has no span — it is
    a pure semantic description, the same shape ``_TypeBuilder`` produces for
    every module).
    """
    mid, _scope_path, _name = key
    rmod = resolved.modules.get(mid)
    if rmod is None:
        return None
    program = rmod.resolved.program
    assert isinstance(program, Program)
    for item in static_type_items(program.body.items):
        if isinstance(item, (RecordDef, EnumDef, ExceptionDef)) and _decl_key(mid, item) == key:
            return item.span
    return None


def _raise_first_uninhabited(
    uninhabited: frozenset[DeclId],
    type_table: TypeTable,
    resolved: ResolvedProgram,
) -> None:
    """Raise ``AglTypeError`` for the first uninhabited declaration, sorted deterministically."""
    typedefs: list[TypeDef] = []
    for decl_id in uninhabited:
        typedef = type_table.get_by_id(decl_id)
        assert typedef is not None
        typedefs.append(typedef)
    typedef = sorted(typedefs, key=decl_def_sort_key)[0]
    key = (typedef.module_id, typedef.scope_path, typedef.name)
    span = _find_type_decl_span(resolved, key)
    raise AglTypeError(uninhabitable_message(typedef.kind, typedef.name), span=span)


def _build_program_type_table(
    resolved: ResolvedProgram,
    *,
    type_table: TypeTable | None = None,
    entry_seed_env: TypeEnvironment | None = None,
    cached_modules: Mapping[ModuleId, CheckedModule] | None = None,
) -> tuple[
    dict[DeclKey, Type],
    dict[DeclKey, GenericTypeDef],
    dict[DeclKey, GenericAliasDef],
    dict[DeclKey, ConstructorSignature],
    dict[DeclKey, tuple[tuple[str, ParamZone], ...]],
]:
    """Phase 1: collect and resolve all public type declarations across all modules.

    Returns a ``program_type_table`` mapping ``(ModuleId, scope_path, name)`` → the
    ``RecordType``/``EnumType``/``ExceptionType`` handle or resolved-alias
    ``Type``.

    The pre-pass is genuinely whole-program two-phase:

    Step A: Register every declared name's handle for ALL modules.  Records,
            enums, and exceptions get their handle entered into
            ``program_type_table`` directly (a handle carries no field/variant
            data, so there is nothing left to fill in later — forward
            references within or across modules are valid immediately). Inline
            member arity is provisional until its resolved fields reveal which
            owner parameters survive transparent aliases.
            Type aliases are registered as lazy program alias keys (their target
            type is not known until the alias body is resolved, so they have
            no handle entry yet).

    Step B: Resolve every type body in a fixed deterministic order (sorted by
            ``(ModuleId.segments, name)``), with no dependency-ordering
            constraint — every nominal field/variant/element type reference is
            a handle, valid regardless of whether the referenced declaration's
            own body has been resolved yet. Transparent aliases are resolved
            lazily when referenced so alias dependencies do not impose a body
            ordering, while recursive aliases are still rejected.

    Step C: Once every body is resolved, run the inhabitation fixpoint
            (:func:`~agm.agl.semantics.analyses.compute_uninhabited`) over the
            whole shared table and reject the first uninhabited declaration
            (sorted by declaration name, see
            :func:`~agm.agl.semantics.type_table.decl_def_sort_key`), at its
            declaration span.

    Cross-module type cycles are allowed (a cycle may span any modules, the
    same as same-module mutual recursion) as long as the declarations
    involved are inhabited — e.g. modA imports modB for a Color enum used in
    modA.Foo fields, and modB imports modA for modA.Foo used in modB.Bar
    fields via an ``array``/``dict`` field or an enum base-case variant.
    """
    # Shared TypeTable: one instance dual-written by every cross-module env
    # below, so declarations from all modules land in the same table.  Step A's
    # per-module envs are transient header-only scaffolding and never
    # dual-write, so they keep their own private (default) tables.
    shared_type_table = type_table if type_table is not None else create_seeded_type_table()
    if entry_seed_env is not None:
        shared_type_table.merge_from(entry_seed_env.type_table)

    interfaces = {
        mid: cm.type_env.module_interface()
        for mid, cm in (cached_modules or {}).items()
        if cm.published_signatures is not None
    }
    for interface in interfaces.values():
        for definition in interface.definitions:
            shared_type_table.register(definition)

    # Step A: register every declared name's handle for all modules.
    # For records/enums: register the handle in both the per-module env AND
    # program_type_table.  For aliases: register in the per-module env only
    # (their entry is added to program_type_table in Step C, once resolved).
    per_module_envs: dict[ModuleId, TypeEnvironment] = {}

    for mid, rmod in resolved.modules.items():
        if mid in interfaces:
            continue
        env = TypeEnvironment(
            module_id=mid,
            local_scope_paths=frozenset(rmod.resolved.scope_nodes),
            scope_nodes=rmod.resolved.scope_nodes,
        )
        if mid == resolved.entry_id and entry_seed_env is not None:
            env.seed_from(entry_seed_env)
        # The builder is transient: it only collects headers into ``env``
        # (which bootstraps ``program_type_table`` below).  Body resolution
        # uses the cross-module builders built later, not this one.
        _collect_shells_only(
            _TypeBuilder(env, module_id=mid, param_zones=rmod.resolved.attributes.param_zones),
            rmod.resolved.program,
        )
        per_module_envs[mid] = env

    # Collect record/enum handles into the shared program type table.
    # Aliases are NOT added here — their entries will be written in Step C
    # after their target type is resolved.
    program_type_table: _DeclKeyDict[Type] = _DeclKeyDict()
    for mid, env in per_module_envs.items():
        for _name, t in env.non_builtin_type_items():
            assert isinstance(t, (RecordType, EnumType, ExceptionType))
            program_type_table[(mid, t.scope_path, t.name)] = t

    # Cross-module generic type definitions carry no field shape (a GenericTypeDef
    # is just a type-parameter count plus a TypeVarType-stamped template — the
    # same "shell" data a non-generic handle carries), so — like
    # program_type_table above — they are collected here in Step A rather than
    # gated on that module's own body-resolution order in Step C: a qualified
    # generic application (e.g. ``lib::Box[int]``) inside a field of a type
    # declared in a module that sorts before ``lib`` in the fixed body-resolution
    # order must still resolve. Inline enum-member entries are reconciled after
    # their resolved fields determine their true captured parameters. Aliases
    # need resolved targets rather than shells, so program environments resolve
    # them lazily; constructor signatures and constructor field kinds genuinely
    # need a resolved body (field/target types), so those remain filled as each
    # type body is resolved in Step C.
    program_generic_table: dict[DeclKey, GenericTypeDef] = {}
    for mid, env in per_module_envs.items():
        for _name, gdef in env.all_generic_types().items():
            program_generic_table[(mid, gdef.template.scope_path, gdef.template.name)] = gdef
    program_alias_table: dict[DeclKey, GenericAliasDef] = {}
    alias_decls: dict[DeclKey, TypeAlias] = {}
    for mid, rmod in resolved.modules.items():
        if mid in interfaces:
            continue
        program = rmod.resolved.program
        assert isinstance(program, Program)
        for item in static_type_items(program.body.items):
            if isinstance(item, TypeAlias):
                alias_decls[_decl_key(mid, item)] = item
    program_alias_keys = frozenset(alias_decls)
    program_ctor_sig_table: dict[DeclKey, ConstructorSignature] = {}
    program_ctor_field_kinds_table: dict[DeclKey, tuple[tuple[str, ParamZone], ...]] = {}

    for interface in interfaces.values():
        program_type_table.update(interface.types)
        program_generic_table.update(interface.generics)
        program_alias_table.update(interface.aliases)
        program_ctor_sig_table.update(interface.constructors)
        program_ctor_field_kinds_table.update(interface.field_kinds)

    # Build per-module cross-module-aware environments and builders for
    # body resolution.  Each env knows the full program_type_table and its own
    # module's ImportEnv so qualified and import-tail-exposed type refs resolve.
    cross_envs: dict[ModuleId, TypeEnvironment] = {}
    cross_builders: dict[ModuleId, _TypeBuilder] = {}
    resolving_aliases: set[DeclKey] = set()

    def _resolve_program_alias(key: DeclKey, span: SourceSpan | None) -> Type | None:
        alias_mid, scope_path, alias_name = key
        item = alias_decls[key]
        if key in resolving_aliases:
            rendered = "::".join((*scope_path, alias_name))
            raise AglTypeError(f"Type alias '{rendered}' is part of a cycle.", span=span)
        resolving_aliases.add(key)
        try:
            env = cross_envs[alias_mid]
            # An alias forced from elsewhere still names its target the way its
            # own scope region does, so resolution re-enters the declaring path.
            with env.type_scope(scope_path):
                type_params = item.type_params
                if type_params:
                    template = env.resolve_type_expr(
                        item.type_expr,
                        span=item.span,
                        type_vars=frozenset(type_params),
                    )
                    program_alias_table[key] = GenericAliasDef(
                        type_params=type_params,
                        template=template,
                    )
                    return None
                resolved = env.resolve_type_expr(item.type_expr, span=item.span)
            program_type_table[key] = resolved
            return resolved
        finally:
            resolving_aliases.remove(key)

    for mid, rmod in resolved.modules.items():
        if mid in interfaces:
            continue
        import_env = rmod.import_env
        cross_env = TypeEnvironment(
            program_type_table=program_type_table,
            program_generic_table=program_generic_table,
            program_alias_table=program_alias_table,
            program_alias_keys=program_alias_keys,
            program_alias_resolver=_resolve_program_alias,
            program_ctor_sig_table=program_ctor_sig_table,
            program_ctor_field_kinds_table=program_ctor_field_kinds_table,
            import_env=import_env,
            local_scope_paths=frozenset(rmod.resolved.scope_nodes),
            scope_nodes=rmod.resolved.scope_nodes,
            module_id=mid,
            type_table=shared_type_table,
        )
        if mid == resolved.entry_id and entry_seed_env is not None:
            cross_env.seed_from(entry_seed_env)
        # Seed with own type shells so bare-name local refs resolve.
        for name, t in per_module_envs[mid].non_builtin_type_items():
            cross_env.register_type(name, t)
        cross_envs[mid] = cross_env
        # Build a _TypeBuilder that uses the cross-module env and has the
        # headers and alias targets registered (for build_record/build_enum/
        # build_exception to work).
        builder = _TypeBuilder(
            cross_env, module_id=mid, param_zones=rmod.resolved.attributes.param_zones
        )
        _collect_shells_only(builder, rmod.resolved.program)
        cross_builders[mid] = builder

    # Transparent aliases can erase enum-owner parameters from inline member
    # records. Finalize every member shell before resolving any declaration
    # body, so a forward reference observes the same arity as a backward one.
    for mid, builder in cross_builders.items():
        builder.reconcile_inline_member_arities()
        _sync_program_env_extensions(
            mid,
            cross_envs[mid],
            program_type_table,
            program_generic_table,
            program_ctor_sig_table,
            program_ctor_field_kinds_table,
        )

    # Use the COMPLETE set of declared type keys (including aliases), NOT just
    # the record/enum handles in program_type_table, as the fixed resolution
    # order for Step B below.
    all_type_keys = {key for key in _collect_all_type_keys(resolved) if key[0] not in interfaces}

    def _resolve_one(key: DeclKey) -> None:
        _resolve_body_for_one(
            mid=key[0],
            key=key,
            per_module_builders=cross_builders,
            program_type_table=program_type_table,
            program_generic_table=program_generic_table,
            program_alias_table=program_alias_table,
            program_ctor_sig_table=program_ctor_sig_table,
            program_ctor_field_kinds_table=program_ctor_field_kinds_table,
            resolved=resolved,
            cross_envs=cross_envs,
        )

    # Step B: resolve every type body in a fixed deterministic order — no
    # dependency-ordering constraint of any kind, since every reference
    # (including an exception's ``extends`` base) is a handle, valid whether
    # or not the referenced declaration's own body has been resolved yet.
    def source_decl_sort_key(key: DeclKey) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        return (key[0].segments, key[1], key[2])

    body_order = sorted(all_type_keys, key=source_decl_sort_key)

    for key in body_order:
        _resolve_one(key)

    # Builtin contracts may inspect referenced enum-member record fields, so
    # validate only after every type body has been resolved.  This preserves
    # the order-free handle phase while making contract validation structural.
    for builder in cross_builders.values():
        builder.validate_builtin_contracts()

    # Step C: every body is now resolved, so the inhabitation fixpoint can
    # run over the whole shared table (this program's declarations plus the
    # builtin/prelude defs, all trivially inhabited). This is the only
    # inhabitation check the pipeline runs; Phase 4's per-module re-check
    # never repeats it.
    uninhabited = compute_uninhabited(shared_type_table)
    if uninhabited:
        _raise_first_uninhabited(uninhabited, shared_type_table, resolved)

    return (
        program_type_table,
        program_generic_table,
        program_alias_table,
        program_ctor_sig_table,
        program_ctor_field_kinds_table,
    )


# ---------------------------------------------------------------------------
# Phase 2: whole-program function-signature pre-pass
# ---------------------------------------------------------------------------


def _build_program_func_sig_table(
    resolved: ResolvedProgram,
    program_type_table: dict[DeclKey, Type],
    program_generic_table: dict[DeclKey, GenericTypeDef],
    program_alias_table: dict[DeclKey, GenericAliasDef],
    entry_seed_env: TypeEnvironment | None = None,
    cached_modules: Mapping[ModuleId, CheckedModule] | None = None,
) -> dict[int, FunctionSignatureRecord]:
    """Phase 2: collect explicit top-level function signatures across modules.

    The table initially contains only functions with declared return types.
    Import-SCC candidate discovery later adds concrete candidate records; it
    never exposes a provisional signature across a module boundary.

    The pre-pass uses temporary per-module environments seeded with the
    program-wide type table and each module's ``ImportEnv``. It resolves only
    header type expressions; no body is checked. The resulting node-id-keyed
    signatures can safely seed every module environment, making explicitly
    annotated cross-module recursion independent of module traversal order.
    """
    from agm.agl.typecheck.checker import _BUILTIN_TYPE_NAMES

    result: dict[int, FunctionSignatureRecord] = {}

    for mid, rmod in resolved.modules.items():
        cached = cached_modules.get(mid) if cached_modules is not None else None
        if cached is not None and cached.published_signatures is not None:
            result.update(cached.published_signatures)
            continue
        program = rmod.resolved.program
        assert isinstance(program, Program)

        import_env = rmod.import_env
        # Build a cross-module-aware env for this module, seeded with its own
        # types so bare-name local type refs in param annotations resolve.
        env = TypeEnvironment(
            program_type_table=program_type_table,
            program_generic_table=program_generic_table,
            program_alias_table=program_alias_table,
            import_env=import_env,
            local_scope_paths=frozenset(rmod.resolved.scope_nodes),
            scope_nodes=rmod.resolved.scope_nodes,
            module_id=mid,
        )
        if mid == resolved.entry_id and entry_seed_env is not None:
            env.seed_from(entry_seed_env)
        for (t_mid, scope_path, t_name), t in program_type_table.items():
            if t_mid == mid:
                env.register_type("::".join((*scope_path, t_name)), t)
        # Also seed the module's own generic types so bare-name local generic
        # refs in param/return annotations (e.g. `o: Option[T]`) resolve here —
        # mirroring the register_type seeding above for non-generic types.
        for (g_mid, scope_path, g_name), gdef in program_generic_table.items():
            if g_mid == mid:
                env.register_generic_type("::".join((*scope_path, g_name)), gdef)

        for item in static_function_items(program.body.items):
            if item.is_synthetic:
                continue
            receiver_owner = rmod.resolved.receiver_owner_for(mid, item)
            # A builtin declaration may reuse a builtin type name -- std spells
            # Agent and Session that way -- and such an item carries no function
            # header metadata.
            if receiver_owner is None and item.name in _BUILTIN_TYPE_NAMES:
                continue
            if item.return_type is None:
                continue

            with env.type_scope(tuple(segment.name for segment in item.scope_path)):
                signature, function_type, _receiver = resolve_function_header(
                    env,
                    item,
                    result_type=item.return_type,
                    param_zones=rmod.resolved.attributes.param_zones,
                    receiver_owner=receiver_owner,
                )
            result[item.node_id] = FunctionSignatureRecord(
                declaration_node_id=item.node_id,
                name=item.name,
                signature=signature,
                function_type=function_type,
                is_builtin=item.is_builtin,
                is_extern=item.is_extern,
                return_source=FunctionReturnSource.DECLARED,
                module_id=mid,
                declaration_span=item.span,
                scope_path=tuple(segment.name for segment in item.scope_path),
            )

    return result


def _build_program_static_let_table(
    resolved: ResolvedProgram, module_envs: Mapping[ModuleId, TypeEnvironment]
) -> dict[int, Type]:
    """Resolve annotated static ``let`` bindings for cross-module access.

    Module-root lets initialize before importers execute. An annotation makes
    their type available in the whole-program header phase, while the ordinary
    body check remains responsible for validating the initializer.
    """
    result: dict[int, Type] = {}
    for mid, loaded in resolved.modules.items():
        env = module_envs[mid]
        for item in static_items(loaded.resolved.program.body.items):
            if not isinstance(item, LetDecl) or item.type_ann is None:
                continue
            name = simple_let_pattern_name(item.pattern)
            if name is None or name == "_":
                continue
            scope_path = tuple(segment.name for segment in item.scope_path)
            with env.type_scope(scope_path):
                result[item.pattern.node_id] = env.resolve_type_expr(
                    item.type_ann, span=item.span, type_vars=frozenset()
                )
    return result


def _build_program_builtin_var_table(
    resolved: ResolvedProgram,
    module_envs: Mapping[ModuleId, TypeEnvironment],
    type_table: TypeTable,
) -> dict[int, Type]:
    """Compute binding types for every ``builtin var`` across all modules.

    Engine settings use their registry type, selecting a loaded source builtin
    identity when available. Other standard-library bindings use their declared
    type resolved in their owning module's complete type environment.
    """
    from agm.agl.modules.ids import STD_CONFIG_ID
    from agm.agl.semantics.engine_keys import get_engine_key_type

    result: dict[int, Type] = {}
    for mid, loaded in resolved.modules.items():
        env = module_envs[mid]
        for item in static_items(loaded.resolved.program.body.items):
            if not isinstance(item, BuiltinVarDecl):
                continue
            if mid == STD_CONFIG_ID and not item.scope_path:
                key_type = get_engine_key_type(item.name)
                if key_type is not None:
                    if isinstance(key_type, (RecordType, EnumType, ExceptionType)):
                        declared = type_table.standard_builtin_declaration(key_type.name)
                        if declared is not None:
                            type_args = (
                                key_type.type_args
                                if isinstance(key_type, (RecordType, EnumType))
                                else ()
                            )
                            key_type = declared.handle(type_args=type_args)
                    result[item.node_id] = key_type
                continue
            scope_path = tuple(segment.name for segment in item.scope_path)
            with env.type_scope(scope_path):
                result[item.node_id] = env.resolve_type_expr(
                    item.type_ann, span=item.span, type_vars=frozenset()
                )
    return result


# ---------------------------------------------------------------------------
# Per-module checking helper
# ---------------------------------------------------------------------------


def _module_function_signatures(
    program: Program, env: TypeEnvironment
) -> dict[str, FunctionSignature]:
    """Return checked schemes declared by *program*, keyed by their local names.

    A program environment carries imported schemes during checking, keyed by
    declaration node id.  They support resolved occurrences but are not
    declarations of the module being published.
    """
    signatures: dict[str, FunctionSignature] = {}
    for item in program.body.items:
        if not isinstance(item, FuncDef) or item.scope_path or item.is_synthetic:
            continue
        signature = env.get_function_signature_by_node_id(item.node_id)
        assert signature is not None, f"No checked signature for '{item.name}'"
        signatures[item.name] = signature
    return signatures


def _prepare_module_environment(
    mid: ModuleId,
    resolved: ModuleResolution,
    program_type_table: dict[DeclKey, Type],
    import_env_map: Mapping[ModuleId, object],
    program_func_sig_table: dict[int, FunctionSignatureRecord],
    program_generic_table: dict[DeclKey, GenericTypeDef],
    program_alias_table: dict[DeclKey, GenericAliasDef],
    program_ctor_sig_table: dict[DeclKey, ConstructorSignature],
    program_ctor_field_kinds_table: dict[DeclKey, tuple[tuple[str, ParamZone], ...]],
    type_table: TypeTable,
    entry_seed_env: TypeEnvironment | None = None,
) -> TypeEnvironment:
    """Build one module's environment before program-wide candidate discovery.

    The env is seeded with:
    - The module's own types (from ``program_type_table``).
    - The program table + import env for cross-module lookups.
    - Explicit function signatures from the whole-program pre-pass
      (``program_func_sig_table``), seeded before any body is checked. Their
      globally unique ``node_id`` keys make declared cross-module calls
      independent of per-module checking order. The candidate-inference SCC
      coordinator adds unannotated signatures after their dependency SCCs close.
    - ``type_table``: the single ``TypeTable`` instance shared by every module
      in this program (the same one built and dual-written in the type pre-pass),
      so this module's own re-check dual-writes into the same table.
    - ``entry_seed_env``: the session type env, seeded first so that prior REPL
      bindings are available. The caller supplies it for the entry module only.
    """
    import_env = import_env_map[mid]
    assert isinstance(import_env, ImportEnv)

    env = TypeEnvironment(
        program_type_table=program_type_table,
        program_generic_table=program_generic_table,
        program_alias_table=program_alias_table,
        program_ctor_sig_table=program_ctor_sig_table,
        program_ctor_field_kinds_table=program_ctor_field_kinds_table,
        import_env=import_env,
        local_scope_paths=frozenset(resolved.scope_nodes),
        scope_nodes=resolved.scope_nodes,
        module_id=mid,
        type_table=type_table,
    )

    # Seed from the REPL session type env first (for the entry module in REPL
    # program context).  Program tables override on collision, so the entry's own types
    # and function signatures always shadow any session binding with the same name.
    # ``type_table`` here is the SAME shared instance the type pre-pass
    # (``_build_program_type_table``) already seeded from this same
    # ``entry_seed_env`` and then advanced with this entry's own declarations,
    # so re-merging its name index now would regress it onto a name this
    # entry just redeclared (see ``TypeEnvironment.seed_from``).
    if entry_seed_env is not None:
        env.seed_from(entry_seed_env, merge_type_table=False)

    # Seed env with the module's own fully-resolved types so they're
    # accessible by bare name (no qualifier needed within the module).
    for (t_mid, scope_path, t_name), t in program_type_table.items():
        if t_mid == mid:
            env.register_type("::".join((*scope_path, t_name)), t)
    for (g_mid, scope_path, g_name), gdef in program_generic_table.items():
        if g_mid == mid:
            env.register_generic_type("::".join((*scope_path, g_name)), gdef)

    # Seed explicit binding types from the whole-program header collection.
    # The candidate coordinator installs only concrete signatures after their
    # function dependency SCCs close.
    #
    # Three tables are seeded:
    # - _binding_types (node_id-keyed, globally unique): used by _check_varref to
    #   look up the callee type when the callee VarRef resolves to a cross-module
    #   FuncDef's decl_node_id.  Seeding the entire pre-pass table is safe because
    #   node_ids are globally unique.
    # - _function_signatures_by_node_id (node_id-keyed, globally unique): used by
    #   _check_declared_name_call to look up the CORRECT signature for any callee
    #   by its globally-unique decl_node_id.  Unlike the name-keyed table below,
    #   this table never suffers from same-name collisions across modules.
    # - _function_signatures (name-keyed): used as a fallback by
    #   _check_declared_name_call when no node-id lookup is available (single-
    #   program path).  Same-named functions from different modules may collide
    #   here; the current module's own signatures always win because
    #   builder.collect() → _preregister_funcdef re-registers them AFTER this
    #   seeding step, overwriting any cross-module collision for bare-name calls.
    for node_id, record in program_func_sig_table.items():
        env.set_binding_type(node_id, record.function_type)
        env.register_function_signature_by_node_id(node_id, record.signature)
        env.register_function_signature(record.name, record.signature, scope_path=record.scope_path)
        if record.is_extern:
            env.register_extern_node_id(node_id)

    return env


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_program(
    resolved: ResolvedProgram,
    capabilities: HostCapabilities,
    entry_seed_env: TypeEnvironment | None = None,
    cached_checked_modules: Mapping[ModuleId, CheckedModule] | None = None,
) -> CheckedProgram:
    """Run the full type-checking pass over a :class:`ResolvedProgram`.

    Parameters
    ----------
    resolved:
        Output of :func:`~agm.agl.scope.program.resolve_program`.
    capabilities:
        Immutable host capability catalog (agents, codecs, renderers).
    entry_seed_env:
        When given, the entry module's ``TypeEnvironment`` is seeded from this
        environment before the program type table and function signatures are
        installed.  Used by the REPL program context to make prior session
        bindings available in program entries.
    cached_checked_modules:
        Checked non-entry modules from an unchanged REPL bootstrap image. Their
        bodies are immutable and can be reused after this call has rebuilt the
        whole-program declaration and signature context for the fresh entry.
        Whatever this leaves uncovered is looked up in the process-global
        artifact cache, and this pass's own results are retained there for the
        next compilation -- except under an ``entry_seed_env``, whose session
        types this call seeds the shared type table from, so its results are
        not a function of the loaded modules alone.

    Returns
    -------
    CheckedProgram
        Per-module type side tables plus the shared program type table.

    Raises
    ------
    AglTypeError
        On the first static type violation in any module (first-error abort).
    """
    # An earlier compilation in this process already checked the modules behind
    # this program; a caller-supplied image (a REPL session's) wins over it.
    retainable = retained_module_sources(resolved.graph)
    reusable: dict[ModuleId, CheckedModule] = dict(
        retained_checked_modules(retainable, capabilities)
    )
    if cached_checked_modules is not None:
        reusable.update(cached_checked_modules)
    cached_checked_modules = {
        mid: cm
        for mid, cm in reusable.items()
        if mid in resolved.modules and cm.resolved is resolved.modules[mid].resolved
    }

    # One TypeTable shared by every module in this program: the type pre-pass
    # dual-writes into it below, and Phase 4 re-checks each module's own
    # declarations against the SAME instance, so the whole program's declarations
    # land in one table regardless of per-module checking order.
    shared_type_table = create_seeded_type_table()

    # Phase 1: build the program-wide type table with all module types stamped
    # with their owning module_id.  Also collects cross-module generic type defs,
    # parameterized aliases, constructor signatures, and constructor field kinds
    # from the per-module envs built during body resolution.
    (
        program_type_table,
        program_generic_table,
        program_alias_table,
        program_ctor_sig_table,
        program_ctor_field_kinds_table,
    ) = _build_program_type_table(
        resolved,
        type_table=shared_type_table,
        entry_seed_env=entry_seed_env,
        cached_modules=cached_checked_modules if entry_seed_env is None else None,
    )

    # Phase 2: build the program-wide explicit function-signature table.
    # It resolves declared header type expressions without checking bodies;
    # unannotated functions are inferred inference SCC by inference SCC in Phase 3.
    program_func_sig_table = _build_program_func_sig_table(
        resolved,
        program_type_table,
        program_generic_table,
        program_alias_table,
        entry_seed_env=entry_seed_env,
        cached_modules=cached_checked_modules,
    )

    # Collect import envs for per-module checking.
    import_env_map: dict[ModuleId, object] = {
        mid: rmod.import_env for mid, rmod in resolved.modules.items()
    }

    # Phase 3: build every module environment before candidate inference. The
    # completed explicit headers are present in every environment, while each
    # dependency SCC later adds only its closed candidates.
    inference_sccs = resolved.graph.sccs
    ordered_mids = tuple(mid for inference_scc in inference_sccs for mid in inference_scc)
    module_envs: dict[ModuleId, TypeEnvironment] = {}
    for mid in ordered_mids:
        module_envs[mid] = _prepare_module_environment(
            mid,
            resolved.modules[mid].resolved,
            program_type_table,
            import_env_map,
            program_func_sig_table,
            program_generic_table,
            program_alias_table,
            program_ctor_sig_table,
            program_ctor_field_kinds_table,
            shared_type_table,
            entry_seed_env=entry_seed_env if mid == resolved.entry_id else None,
        )

    # Annotated static lets and builtin vars need each module's complete type
    # environment, while the environments themselves need those types only for
    # later body checks. Build the environments first, then seed their completed
    # binding tables into every module for cross-module references.
    program_static_let_table = _build_program_static_let_table(resolved, module_envs)
    program_builtin_var_table = _build_program_builtin_var_table(
        resolved, module_envs, shared_type_table
    )
    for env in module_envs.values():
        for binding_node_id, binding_type in program_static_let_table.items():
            env.set_binding_type(binding_node_id, binding_type)
        for var_node_id, var_type in program_builtin_var_table.items():
            env.set_binding_type(var_node_id, var_type)

    program_modules = {module_id: module.resolved for module_id, module in resolved.modules.items()}
    validate_builtin_method_ownership(program_modules)

    for mid in ordered_mids:
        prepare_module_headers(
            resolved.modules[mid].resolved,
            capabilities,
            env=module_envs[mid],
            module_id=mid,
        )

    for mid, retained in cached_checked_modules.items():
        env = module_envs[mid]
        for item in static_function_items(retained.resolved.program.body.items):
            owner = retained.resolved.receiver_owner_for(mid, item)
            record = (retained.published_signatures or {}).get(item.node_id)
            if item.return_type is not None or owner is None or record is None:
                continue
            with env.type_scope(tuple(segment.name for segment in item.scope_path)):
                signature, _, receiver = resolve_function_header(
                    env,
                    item,
                    result_type=record.signature.result,
                    param_zones=retained.resolved.attributes.param_zones,
                    receiver_owner=owner,
                )
            register_method_header(env, item, signature, receiver, mid)

    validate_builtin_declaration_uniqueness(program_modules, resolved.entry_id)
    validate_method_declaration_collisions(program_modules, shared_type_table)

    # Candidate discovery follows the reverse-topological dependency SCC
    # sequence. A cycle is one cross-module function graph; a dependency
    # SCC's concrete records are available before its importers are considered.
    # Each SCC's closed signatures are published only into itself and the later
    # SCCs (their potential importers); earlier SCCs are dependencies that
    # cannot reference it, so registering there would be wasted work.
    for index, inference_scc in enumerate(inference_sccs):
        if all(
            mid in cached_checked_modules
            and cached_checked_modules[mid].published_signatures is not None
            for mid in inference_scc
        ):
            continue
        candidates = tuple(
            CandidateModule(
                resolved.modules[mid].resolved,
                module_envs[mid],
                capabilities,
                mid,
            )
            for mid in inference_scc
        )
        publication_envs = tuple(
            module_envs[mid] for later_scc in inference_sccs[index:] for mid in later_scc
        )
        for record in infer_module_component_candidates(
            ModuleCandidateComponent(candidates, publication_envs)
        ):
            program_func_sig_table[record.declaration_node_id] = record

    # Phase 4: ordinary body checking is the sole source of checked artifacts.
    # The full program table mixes declared and candidate records; filter to the
    # candidate subset once and share it across every module check.
    candidate_records = candidate_records_for(program_func_sig_table)
    checked_modules: dict[ModuleId, CheckedModule] = {}
    reused_modules: set[ModuleId] = set()
    for mid in ordered_mids:
        cached = cached_checked_modules.get(mid) if cached_checked_modules is not None else None
        if cached is not None and cached.resolved is resolved.modules[mid].resolved:
            checked_modules[mid] = cached
            reused_modules.add(mid)
            continue
        rmod = resolved.modules[mid]
        cp = _check_prepared_module(
            rmod.resolved,
            capabilities,
            env=module_envs[mid],
            module_id=mid,
            prepare_headers=False,
            infer_candidates=False,
            candidate_records=candidate_records,
        )
        cm = replace(
            cp,
            module_id=mid,
            import_env=rmod.import_env,
            source_text=rmod.source_text,
            function_signatures=_module_function_signatures(rmod.resolved.program, cp.type_env),
            published_signatures={
                item.node_id: program_func_sig_table[item.node_id]
                for item in static_function_items(rmod.resolved.program.body.items)
                if item.node_id in program_func_sig_table
            },
        )
        checked_modules[mid] = cm

    # Preserve the established checked-module presentation order independently
    # from dependency-ordered inference and validation above. Warnings follow the
    # same presentation order so multi-module diagnostics stay stable.
    presentation_order = tuple(mid for mid in resolved.modules if mid != resolved.entry_id) + (
        resolved.entry_id,
    )
    runtime_modules = frozenset(resolved.graph.source_reachable_modules(resolved.entry_id))

    checked = CheckedProgram(
        modules={mid: checked_modules[mid] for mid in presentation_order},
        entry_id=resolved.entry_id,
        program_type_table=program_type_table,
        warnings=tuple(
            warning for mid in presentation_order for warning in checked_modules[mid].warnings
        ),
        capabilities=capabilities,
        import_sccs=resolved.import_sccs,
        resource_roots={mid: resolved.graph.resource_root_for(mid) for mid in presentation_order},
        runtime_modules=runtime_modules,
        module_fingerprints=module_fingerprints(retainable, capabilities)
        if entry_seed_env is None
        else {},
    )
    if self_validation_enabled():
        assert_checked_program_closed(checked, frozenset(reused_modules))
    if entry_seed_env is None:
        retain_checked_modules(retainable, capabilities, checked_modules)
    return checked
