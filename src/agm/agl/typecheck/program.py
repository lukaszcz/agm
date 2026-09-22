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

3. **Module headers and static bindings** — finalize every module's record,
   enum, and exception bodies and pre-register function headers
   (:func:`~agm.agl.typecheck.checker.prepare_module_headers`), then resolve
   every module's exported static ``let``/``var`` binding type: an annotated
   binding resolves its annotation, an unannotated one is typed from its
   constant initializer (a static-root binding in an importable module already
   requires one). Running this after headers, not before, keeps a binding's
   own diagnostic from preempting a type declaration's; running it before
   candidate inference means a published binding type never depends on an
   inferred function signature. A module served from the artifact cache
   supplies its published binding types directly, skipping the recheck.
   Every resulting binding and builtin-var type is seeded into every module's
   environment for cross-module references.

4. **Import-SCC candidate inference** — consume the loader's graph SCCs in
   reverse topological order. Ordinary dependency SCCs publish their closed
   signatures before their importers are considered. Each candidate function
   dependency SCC publishes only closed unannotated signatures; one resulting
   cycle builds a single cross-module function graph.

5. **Authoritative per-module type-check** — after every signature is concrete,
   recheck each module body with its module-aware
   :class:`~agm.agl.typecheck.env.TypeEnvironment`. This pass alone publishes
   checked node types, calls, contracts, bindings, and warnings. Each module's
   environment starts its own-facts journal
   (:meth:`~agm.agl.typecheck.env.TypeEnvironment.begin_facts`)
   immediately before this recheck, once phases 1-4 are done mutating it, so
   the journal records exactly this recheck's mutations
   (:class:`~agm.agl.typecheck.env.CheckedModuleImage`). Consequently, an
   unannotated static binding's own type error surfaces before its module's
   ordinary body-check diagnostics.

Phases 1-4 live in :func:`_prepare_program`, returning the prepared
per-module environments and tables in :class:`_PreparedProgram`;
:func:`check_program` runs phase 5 over that result.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Generic, Mapping, TypeVar, cast

from agm.agl.artifact_cache import (
    RetainedSources,
    module_fingerprints,
    retain_checked_modules,
    retain_module_headers,
    retained_checked_modules,
    retained_module_headers,
    retained_module_sources,
)
from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import ImportEnv
from agm.agl.scope.program import ResolvedProgram
from agm.agl.scope.symbols import DeclarationKey, ModuleResolution
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.analyses import compute_uninhabited, uninhabitable_message
from agm.agl.semantics.persistent import PersistentDict
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
    VarDecl,
    exported_binding_name,
    static_binding_node_id,
    static_function_items,
    static_items,
    static_type_items,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.builder import _TypeBuilder
from agm.agl.typecheck.checker import (
    _check_prepared_module,
    _Checker,
    is_constant_builtin_call,
    prepare_module_headers,
    require_static_root_constant,
)
from agm.agl.typecheck.constant_bindings import ModuleConstantBindings
from agm.agl.typecheck.declaration_validation import (
    validate_builtin_declaration_uniqueness,
    validate_method_declaration_collisions,
)
from agm.agl.typecheck.env import (
    AglTypeError,
    CheckedModule,
    CheckedModuleImage,
    ConstructorSignature,
    DeclaredHeaderSeed,
    EnvironmentFacts,
    FunctionSignature,
    GenericAliasDef,
    GenericTypeDef,
    PublishedModuleSurface,
    TypeEnvironment,
    _assert_checked_types_closed,
    assert_checked_output_closed,
    dereference_slot_constructor_ref,
)
from agm.agl.typecheck.function_inference import (
    CandidateModule,
    FunctionReturnSource,
    FunctionSignatureRecord,
    ModuleCandidateComponent,
    candidate_records_for,
    declared_records_for,
    infer_module_component_candidates,
    publish_candidate_signature,
    register_method_header,
    resolve_function_header,
)
from agm.agl.zones import ParamZone
from agm.util.graph import GraphCycleError, toposort

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


def _is_exception_root_key(key: DeclKey) -> bool:
    """Whether *key* names the standard library's own ``Exception`` declaration."""
    module_id, scope_path, name = key
    return name == "Exception" and not scope_path and module_id.is_standard_library


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


def _declaration_spans(resolved: ResolvedProgram) -> dict[DeclarationKey, SourceSpan]:
    """Index source spans for declarations available to program-wide diagnostics."""
    return {
        key: ref.decl_span
        for module in resolved.modules.values()
        for key, ref in module.resolved.declarations.items()
    }


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
    cached_modules: Mapping[ModuleId, PublishedModuleSurface] | None = None,
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
        mid: cm.interface
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
            _TypeBuilder(env, module_id=mid, attributes=rmod.resolved.attributes),
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
        builder = _TypeBuilder(cross_env, module_id=mid, attributes=rmod.resolved.attributes)
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
    # The one exception is the canonical ``Exception`` itself: an exception
    # that omits ``extends`` takes it as its base, and
    # ``TypeTable.exception_root`` reads it from the registered standard-library
    # declaration, so that declaration resolves before every other body.
    def source_decl_sort_key(
        key: DeclKey,
    ) -> tuple[bool, tuple[str, ...], tuple[str, ...], str]:
        return (not _is_exception_root_key(key), key[0].segments, key[1], key[2])

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
    cached_modules: Mapping[ModuleId, PublishedModuleSurface] | None = None,
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


def _scope_constant_bindings(
    module_resolved: ModuleResolution, module_id: ModuleId
) -> ModuleConstantBindings:
    """Classify one module's constants from scope-resolved data alone.

    No checker-time slot resolution exists yet when the static-binding
    pre-pass runs, so constancy is decided here exactly as the checker later
    decides it, from scope's own constructor classification.
    """
    return ModuleConstantBindings(
        module_resolved,
        module_id,
        is_constructor=lambda node_id: (
            dereference_slot_constructor_ref(
                node_id,
                resolution=module_resolved.resolution,
                constructor_refs=module_resolved.constructor_refs,
                slot_constructor_refs={},
            )
            is not None
        ),
        is_constant_builtin=lambda node_id: is_constant_builtin_call(module_resolved, node_id),
    )


def _constant_dependency_order(
    bindings: Mapping[int, LetDecl | VarDecl], constants: ModuleConstantBindings
) -> list[int]:
    """Order one module's static bindings so a named constant is typed before its reader.

    A constant may name another constant of its own module in either source
    direction, so an unannotated binding's inferred type can depend on a later
    one. A cyclic set keeps source order: the constant-expression check reports
    the cycle itself once the first of them is screened.
    """
    position = {node_id: index for index, node_id in enumerate(bindings)}
    deps = {
        node_id: constants.dependencies(item.value) & bindings.keys()
        for node_id, item in bindings.items()
    }
    try:
        return toposort(bindings, deps, key=position.__getitem__)
    except GraphCycleError:
        return list(bindings)


def _build_program_static_binding_table(
    resolved: ResolvedProgram,
    module_envs: Mapping[ModuleId, TypeEnvironment],
    capabilities: HostCapabilities,
    cached_modules: Mapping[ModuleId, PublishedModuleSurface] | None = None,
) -> dict[int, Type]:
    """Resolve every exported static ``let``/``var`` binding for cross-module access.

    Module-root and scoped bindings initialize before importers execute.
    Annotated: resolve the annotation directly. Unannotated: the static-root
    constant-expression rule already required of every module-level initializer
    in an importable module (``checker.require_static_root_constant``) means an
    unannotated binding's type depends only on its own initializer and on the
    constants that initializer names -- never on candidate function inference,
    which has not run yet -- so it is screened for constancy and then typed
    with a throwaway ``_Checker``'s ``infer_static_initializer_type`` over this
    module's own prepared environment (the same pattern
    ``function_inference._seed_candidate_visible_bindings`` uses), which
    installs no binding and writes nothing beyond the initializer's own
    ordinary node types. A named constant is typed first and its type installed
    into the environment, so a binding that names one resolves it there.

    A non-static-root module (the REPL, or a ``-c`` entry with no ``program
    def`` of its own) is skipped: it is never imported, so nothing reads its
    published types. A loose file with its own ``program def`` is still
    static-root despite carrying ``ENTRY_ID`` (no package identity), since
    its own defs need root bindings' types seeded just like an importable
    module's -- hence the ``static_root`` test here, not an ``ENTRY_ID`` one.

    A module served from the artifact cache is never re-checked here: its
    published binding types are read directly off its ``published_binding_types``.
    """
    result: dict[int, Type] = {}
    for mid, loaded in resolved.modules.items():
        module_resolved = loaded.resolved
        if not module_resolved.static_root:
            continue
        cached = cached_modules.get(mid) if cached_modules is not None else None
        if cached is not None and cached.published_binding_types is not None:
            result.update(cached.published_binding_types)
            continue
        env = module_envs[mid]
        constants = _scope_constant_bindings(module_resolved, mid)
        bindings = {
            static_binding_node_id(item): item
            for item in static_items(module_resolved.program.body.items)
            if isinstance(item, (LetDecl, VarDecl)) and exported_binding_name(item) is not None
        }
        checker: _Checker | None = None
        for decl_node_id in _constant_dependency_order(bindings, constants):
            item = bindings[decl_node_id]
            if item.type_ann is not None:
                scope_path = tuple(segment.name for segment in item.scope_path)
                with env.type_scope(scope_path):
                    binding_type = env.resolve_type_expr(
                        item.type_ann, span=item.span, type_vars=frozenset()
                    )
            else:
                require_static_root_constant(item.value, module_resolved, constants=constants)
                if checker is None:
                    checker = _Checker(
                        env=env,
                        resolved=module_resolved,
                        capabilities=capabilities,
                        module_id=mid,
                    )
                binding_type = checker.infer_static_initializer_type(item)
            result[decl_node_id] = binding_type
            env.set_binding_type(decl_node_id, binding_type)
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


def _module_static_binding_types(program: Program, env: TypeEnvironment) -> dict[int, Type]:
    """Return this module's own published static-binding types, keyed by node id.

    Read straight off the authoritative post-check environment -- the same
    source ``_module_function_signatures`` reads signatures from -- not off the
    static-binding pre-pass table, so a published type always matches what the
    ordinary body check installed.
    """
    result: dict[int, Type] = {}
    for item in static_items(program.body.items):
        if not isinstance(item, (LetDecl, VarDecl)) or exported_binding_name(item) is None:
            continue
        node_id = static_binding_node_id(item)
        binding_type = env.get_binding_type(node_id)
        assert binding_type is not None, f"No checked type for binding node {node_id}"
        result[node_id] = binding_type
    return result


def _declared_header_seed(
    declared_func_sig_table: Mapping[int, FunctionSignatureRecord],
) -> DeclaredHeaderSeed:
    """Collect the declared headers every module environment is seeded with.

    Mirrors the per-module registration in :func:`_prepare_module_environment`,
    including ``register_function_signature``'s root-scope-only rule for the
    name-keyed table.
    """
    binding_types: PersistentDict[int, Type] = PersistentDict()
    signatures: dict[str, FunctionSignature] = {}
    signatures_by_node_id: dict[int, FunctionSignature] = {}
    extern_node_ids: set[int] = set()
    for node_id, record in declared_func_sig_table.items():
        binding_types[node_id] = record.function_type
        signatures_by_node_id[node_id] = record.signature
        if not record.scope_path:
            signatures[record.name] = record.signature
        if record.is_extern:
            extern_node_ids.add(node_id)
    return DeclaredHeaderSeed(
        binding_types=binding_types,
        signatures=signatures,
        signatures_by_node_id=signatures_by_node_id,
        extern_node_ids=extern_node_ids,
    )


def _prepare_module_environment(
    mid: ModuleId,
    resolved: ModuleResolution,
    program_type_table: dict[DeclKey, Type],
    import_env_map: Mapping[ModuleId, object],
    declared_func_sig_table: dict[int, FunctionSignatureRecord],
    program_generic_table: dict[DeclKey, GenericTypeDef],
    program_alias_table: dict[DeclKey, GenericAliasDef],
    program_ctor_sig_table: dict[DeclKey, ConstructorSignature],
    program_ctor_field_kinds_table: dict[DeclKey, tuple[tuple[str, ParamZone], ...]],
    type_table: TypeTable,
    entry_seed_env: TypeEnvironment | None = None,
    declared_seed: DeclaredHeaderSeed | None = None,
) -> TypeEnvironment:
    """Build one module's environment before program-wide candidate discovery.

    The env is seeded with:
    - The module's own types (from ``program_type_table``).
    - The program table + import env for cross-module lookups.
    - Explicitly declared function signatures from the whole-program pre-pass
      (``declared_func_sig_table``: ``program_func_sig_table`` filtered to
      ``FunctionReturnSource.DECLARED``), seeded before any body is checked.
      Their globally unique ``node_id`` keys make declared cross-module calls
      independent of per-module checking order. Candidate (unannotated-function)
      records are deliberately excluded here: the import-SCC coordinator
      publishes each one, by the narrower rule ``publish_candidate_signature``
      enforces, only after its dependency SCC closes -- for both a freshly
      inferred candidate and one restored from a cached module's
      ``published_signatures``.
    - ``type_table``: the single ``TypeTable`` instance shared by every module
      in this program (the same one built and dual-written in the type pre-pass),
      so this module's own re-check dual-writes into the same table.
    - ``entry_seed_env``: the session type env, seeded first so that prior REPL
      bindings are available. The caller supplies it for the entry module only.
    - ``declared_seed``: the declared headers above, already collected for the
      whole program, so the constructor copies them instead of this function
      registering them one by one. Absent for a module whose tables cannot
      start from the shared copy -- the REPL entry, seeded from its session env.
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
        declared_seed=declared_seed,
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
    # Only DECLARED records reach this loop (see ``declared_func_sig_table``
    # above); a candidate is published later, once its dependency SCC closes,
    # by the narrower rule ``publish_candidate_signature`` shares with the
    # import-SCC loop's fully-cached branch below.
    #
    # Three tables are seeded, each keyed globally uniquely by node_id except
    # the name-keyed one:
    # - _binding_types: used by _check_varref to look up the callee type when
    #   the callee VarRef resolves to a cross-module FuncDef's decl_node_id.
    # - _function_signatures_by_node_id: used by _check_declared_name_call to
    #   look up the correct signature for any callee by its globally-unique
    #   decl_node_id, so same-name collisions across modules never matter here.
    # - _function_signatures (name-keyed, root scope only): every module's
    #   declared signatures land here under a shared bare name; nothing looks
    #   this table up by name, so a same-name collision across modules is
    #   inert -- Phase 4's own re-check still calls ``register_function_signature`` for
    #   this module's own declarations, but only to keep the table live for
    #   ``all_function_signatures()``, not because a later lookup depends on it.
    # ``declared_seed`` carries the same four tables this loop would build, so
    # the caller passes it wherever the constructor can copy them wholesale
    # instead.  The loop remains for the REPL entry module, whose tables start
    # from the session env above and so cannot share the program-wide seed.
    if declared_seed is None:
        for node_id, record in declared_func_sig_table.items():
            env.set_binding_type(node_id, record.function_type)
            env.register_function_signature_by_node_id(node_id, record.signature)
            env.register_function_signature(
                record.name, record.signature, scope_path=record.scope_path
            )
            if record.is_extern:
                env.register_extern_node_id(node_id)

    return env


def _prepare_headers(
    resolved: ResolvedProgram,
    capabilities: HostCapabilities,
    module_envs: Mapping[ModuleId, TypeEnvironment],
    ordered_mids: tuple[ModuleId, ...],
    retainable: RetainedSources | None,
) -> None:
    """Phase 3's header pass: build every module's types and function headers.

    Preparing one module's headers resolves every declaration body and every
    declared function header the module owns, then writes the results into its
    own environment and its methods into the program's shared type table. All
    of it is a function of the module, its import/export closure, and the
    capabilities -- exactly what *retainable* keys on -- so a retainable
    module journals its writes and a later compilation replays that journal
    instead of resolving the same declarations again. The entry module, which
    has no stable identity across programs, is never retainable and is always
    prepared from source.

    The shared table's *declarations* are deliberately not journaled: Phase 1
    resolved and registered every one of them before this pass runs, and what
    this pass registers is that same content again. The parity test over a
    replayed preparation compares the whole table, so a Phase 1 that stopped
    registering a declaration would fail there rather than silently leave a
    memo-served compilation short of one.
    """
    retained = retained_module_headers(retainable, capabilities) if retainable is not None else {}
    prepared: dict[ModuleId, EnvironmentFacts] = {}
    for mid in ordered_mids:
        env = module_envs[mid]
        facts = retained.get(mid)
        if facts is not None:
            env.replay(facts)
            continue
        journaled = retainable is not None and mid in retainable
        if journaled:
            env.begin_facts()
        prepare_module_headers(
            resolved.modules[mid].resolved,
            capabilities,
            env=env,
            module_id=mid,
        )
        if journaled:
            prepared[mid] = env.end_facts()
    if retainable is not None and prepared:
        retain_module_headers(retainable, capabilities, prepared)


@dataclass(frozen=True, slots=True)
class _PreparedProgram:
    """Per-module environments and whole-program tables ready for Phase 4.

    Returned by :func:`_prepare_program`, which runs Phases 1-3: the
    whole-program type and function-signature pre-passes, per-module
    environment preparation, and import-SCC candidate inference. Phase 4
    re-checks each module's own body against its prepared environment; a
    module named in ``cached_checked_modules`` at that point is reused
    unchecked instead.
    """

    module_envs: dict[ModuleId, TypeEnvironment]
    program_type_table: dict[DeclKey, Type]
    program_func_sig_table: dict[int, FunctionSignatureRecord]
    program_static_binding_table: dict[int, Type]
    candidate_records: dict[int, FunctionSignatureRecord]
    ordered_mids: tuple[ModuleId, ...]
    declaration_spans: dict[DeclarationKey, SourceSpan]


def _prepare_program(
    resolved: ResolvedProgram,
    capabilities: HostCapabilities,
    entry_seed_env: TypeEnvironment | None = None,
    cached_checked_modules: Mapping[ModuleId, CheckedModule | CheckedModuleImage] | None = None,
    retainable: RetainedSources | None = None,
) -> _PreparedProgram:
    """Run Phases 1-3 of :func:`check_program`: prepare, but do not check, every module.

    *cached_checked_modules* supplies checked non-entry modules whose bodies
    are immutable and can be reused; each is still prepared here like every
    other module (a cached module adds an extra method-header registration
    pass on top), because the environments this returns serve fresh modules
    and cached ones alike. Only the *cache-free* call (no
    *cached_checked_modules*, no *retainable*) reproduces exactly what a
    fresh, cache-free compilation would build for every environment -- the
    seam a test uses to obtain one for
    :meth:`~agm.agl.typecheck.env.CheckedModuleImage.rehydrate`.

    *retainable* names, per module, everything its artifacts could have been
    derived from, and enables the header memo: Phase 3's header preparation
    is journaled per module and replayed on a later compilation instead of
    re-resolving every declaration body and function header. Omitted, every
    module's headers are prepared from source.
    """
    cached_checked_modules = cached_checked_modules or {}

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
    # dependency SCC later adds only its closed candidates. ``program_func_sig_table``
    # may already carry restored candidate records here (merged in from a cached
    # module's ``published_signatures`` by Phase 2), so bulk seeding is filtered
    # down to the declared subset -- a candidate publishes only through the
    # import-SCC loop below, whichever path (fresh or cached) produced it.
    declared_func_sig_table = declared_records_for(program_func_sig_table)
    inference_sccs = resolved.graph.sccs
    ordered_mids = tuple(mid for inference_scc in inference_sccs for mid in inference_scc)
    module_envs: dict[ModuleId, TypeEnvironment] = {}
    declared_seed = _declared_header_seed(declared_func_sig_table)
    for mid in ordered_mids:
        module_envs[mid] = _prepare_module_environment(
            mid,
            resolved.modules[mid].resolved,
            program_type_table,
            import_env_map,
            declared_func_sig_table,
            program_generic_table,
            program_alias_table,
            program_ctor_sig_table,
            program_ctor_field_kinds_table,
            shared_type_table,
            entry_seed_env=entry_seed_env if mid == resolved.entry_id else None,
            declared_seed=(
                None if entry_seed_env is not None and mid == resolved.entry_id else declared_seed
            ),
        )

    program_modules = {module_id: module.resolved for module_id, module in resolved.modules.items()}
    declaration_spans = _declaration_spans(resolved)

    _prepare_headers(resolved, capabilities, module_envs, ordered_mids, retainable)

    # Static let/var bindings and builtin vars need each module's headers
    # finalized (record/enum/exception bodies, constructor arities) so that a
    # binding's own initializer diagnostic never preempts a type-declaration
    # diagnostic from the same module. They need each module's complete type
    # environment, while the environments themselves need those types only for
    # later body checks, so seed the completed binding tables into every
    # module for cross-module references only after computing them here.
    program_static_binding_table = _build_program_static_binding_table(
        resolved, module_envs, capabilities, cached_modules=cached_checked_modules
    )
    program_builtin_var_table = _build_program_builtin_var_table(
        resolved, module_envs, shared_type_table
    )
    for env in module_envs.values():
        for binding_node_id, binding_type in program_static_binding_table.items():
            env.set_binding_type(binding_node_id, binding_type)
        for var_node_id, var_type in program_builtin_var_table.items():
            env.set_binding_type(var_node_id, var_type)

    for mid, retained in cached_checked_modules.items():
        env = module_envs[mid]
        # `retained` may be a `CheckedModuleImage`, which carries no `resolved`
        # (see the module docstring's rehydration seam). This program's own
        # `resolved.modules[mid].resolved` is the identical object for a full
        # `CheckedModule` too, by the identity gate `check_program` applies
        # before including it here -- so reading it uniformly needs no
        # instance check.
        rmod = resolved.modules[mid].resolved
        for item in static_function_items(rmod.program.body.items):
            owner = rmod.receiver_owner_for(mid, item)
            record = (retained.published_signatures or {}).get(item.node_id)
            if item.return_type is not None or owner is None or record is None:
                continue
            with env.type_scope(tuple(segment.name for segment in item.scope_path)):
                signature, _, receiver = resolve_function_header(
                    env,
                    item,
                    result_type=record.signature.result,
                    param_zones=rmod.attributes.param_zones,
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
        publication_envs = tuple(
            module_envs[mid] for later_scc in inference_sccs[index:] for mid in later_scc
        )
        if all(
            mid in cached_checked_modules
            and cached_checked_modules[mid].published_signatures is not None
            for mid in inference_scc
        ):
            # Every module in this SCC was served from cache, so none of its
            # candidates are (re)inferred here -- but a restored candidate still
            # publishes through the same rule a freshly inferred one would
            # (``publish_candidate_signature``), into this SCC's own declaring
            # env by name and into this same ``publication_envs`` by node id.
            for mid in inference_scc:
                declaring_env = module_envs[mid]
                published = cached_checked_modules[mid].published_signatures or {}
                for record in candidate_records_for(published).values():
                    for env in publication_envs:
                        publish_candidate_signature(
                            env,
                            declaring_env=declaring_env,
                            declaration_node_id=record.declaration_node_id,
                            name=record.name,
                            scope_path=record.scope_path,
                            signature=record.signature,
                            function_type=record.function_type,
                        )
            continue
        candidates = tuple(
            CandidateModule(
                resolved.modules[mid].resolved,
                module_envs[mid],
                capabilities,
                mid,
                declaration_spans,
            )
            for mid in inference_scc
        )
        for record in infer_module_component_candidates(
            ModuleCandidateComponent(candidates, publication_envs)
        ):
            program_func_sig_table[record.declaration_node_id] = record

    # The full program table mixes declared and candidate records; filter to
    # the candidate subset once and share it across every module's Phase 4 check.
    candidate_records = candidate_records_for(program_func_sig_table)

    return _PreparedProgram(
        module_envs=module_envs,
        program_type_table=program_type_table,
        program_func_sig_table=program_func_sig_table,
        program_static_binding_table=program_static_binding_table,
        candidate_records=candidate_records,
        ordered_mids=ordered_mids,
        declaration_spans=declaration_spans,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_program(
    resolved: ResolvedProgram,
    capabilities: HostCapabilities,
    entry_seed_env: TypeEnvironment | None = None,
    cached_checked_modules: Mapping[ModuleId, CheckedModule | CheckedModuleImage] | None = None,
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
        Checked non-entry modules from an unchanged REPL bootstrap snapshot. Their
        bodies are immutable and can be reused after this call has rebuilt the
        whole-program declaration and signature context for the fresh entry.
        Whatever this leaves uncovered is looked up in the process-global
        artifact cache, which may answer with a full ``CheckedModule`` (reused
        as-is) or a ``CheckedModuleImage`` (rehydrated onto this call's own
        prepared environment) -- and this pass's own results are retained
        there for the next compilation -- except under an ``entry_seed_env``,
        whose session types this call seeds the shared type table from, so its
        results are not a function of the loaded modules alone.

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
    reusable: dict[ModuleId, CheckedModule | CheckedModuleImage] = dict(
        retained_checked_modules(retainable, capabilities)
    )
    if cached_checked_modules is not None:
        # A REPL full object supersedes an artifact-cache image by design: if
        # it later fails the identity gate below, the module is re-checked in
        # full even though a valid image was available -- correct, since a
        # superseded REPL module should be re-checked, not restored stale.
        reusable.update(cached_checked_modules)
    # A full object must be the exact current `ResolvedModule` -- an in-process
    # identity check. An image carries no `resolved` to compare against: it
    # reaches `reusable` only through `_served`'s own source-identity (memory)
    # or `_disk_key`'s content digest over the module and its transitive
    # import/export closure (disk), which is sound because node ids are
    # content-derived (`modules/parsed_module_cache.py`) -- membership in
    # `resolved.modules` is the only condition an image still needs here.
    cached_checked_modules = {
        mid: cm
        for mid, cm in reusable.items()
        if mid in resolved.modules
        and (not isinstance(cm, CheckedModule) or cm.resolved is resolved.modules[mid].resolved)
    }

    prepared = _prepare_program(
        resolved,
        capabilities,
        entry_seed_env=entry_seed_env,
        cached_checked_modules=cached_checked_modules,
        # A REPL compilation seeds the shared type table from its session
        # environment, so its modules' headers are not a function of the
        # loaded modules alone -- the same condition that keeps its checked
        # modules out of the artifact cache below.
        retainable=retainable if entry_seed_env is None else None,
    )
    module_envs = prepared.module_envs
    program_type_table = prepared.program_type_table
    program_func_sig_table = prepared.program_func_sig_table
    candidate_records = prepared.candidate_records
    ordered_mids = prepared.ordered_mids
    declaration_spans = prepared.declaration_spans

    # Phase 4: ordinary body checking is the sole source of checked artifacts.
    # The journal starts here, immediately before each freshly checked module's
    # own body check, once every environment above is fully prepared: it must
    # record exactly the mutations _check_prepared_module's body check makes
    # on an otherwise-complete environment, not the whole-program preparation
    # above (see agm.agl.typecheck.env.EnvironmentFacts / CheckedModuleImage).
    checked_modules: dict[ModuleId, CheckedModule] = {}
    reused_modules: set[ModuleId] = set()
    for mid in ordered_mids:
        cached = cached_checked_modules.get(mid)
        if isinstance(cached, CheckedModule) and cached.resolved is resolved.modules[mid].resolved:
            checked_modules[mid] = cached
            reused_modules.add(mid)
            continue
        if isinstance(cached, CheckedModuleImage):
            # Rehydrated, not reused: left out of `reused_modules` so
            # self-validation below covers it like a freshly checked module.
            # Storing the full object here lets `retain_checked_modules`
            # restore `_CHECKED`'s in-memory entry to a full `CheckedModule`,
            # which `retained_match_sites`'s `matches`-kind anchors need.
            checked_modules[mid] = cached.rehydrate(resolved.modules[mid], module_envs[mid])
            continue
        rmod = resolved.modules[mid]
        env = module_envs[mid]
        env.begin_facts()
        cp = _check_prepared_module(
            rmod.resolved,
            capabilities,
            env=env,
            module_id=mid,
            prepare_headers=False,
            infer_candidates=False,
            candidate_records=candidate_records,
            declaration_spans=declaration_spans,
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
            published_binding_types=_module_static_binding_types(
                rmod.resolved.program, cp.type_env
            ),
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
