"""Program-level scope resolver for the AgL module system.

This module provides :func:`resolve_program`, which runs the scope-resolution
pass over an entire :class:`~agm.agl.modules.loader.ModuleGraph`, producing
a :class:`ResolvedProgram` that contains per-module :class:`ResolvedModule`
results plus whole-program pre-pass tables.

Design
------
- **Export maps**: top-level ``def``/``record``/``enum``/``type`` names per
  module plus explicit ``export`` declarations, computed before any body is
  resolved.
- **Contribution import environment per module**: built from each module's
  import declarations against the already-loaded graph (no re-reading files).
- **Whole-program pre-pass tables**: ``all_public_funcs`` and ``all_public_types``
  collected BEFORE resolving any body, enabling cross-module mutual recursion.
- **Declaration-only enforcement**: non-entry modules may only contain
  declarations (``def``, ``record``, ``enum``, ``type``, ``infixl``/``infixr``,
  ``import``).
- **Entry-only enforcement**: ``agent``, ``param``, ``program`` only in entry.
- **Header-only imports** (non-entry): imports must appear before any
  declaration.
- **``::name`` self-reference**: resolved to the current module's own scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import ModuleGraph
from agm.agl.scope.imports import (
    EMPTY_IMPORT_ENV,
    ImportEnv,
    ImportTarget,
    NameAtom,
    PathAtom,
    QName,
    SingleTarget,
    WildcardTarget,
    build_import_env,
    resolve_alias_target,
)
from agm.agl.scope.resolver import _Resolver
from agm.agl.scope.symbols import (
    AgentKey,
    AglScopeError,
    BinderKind,
    ConstructorRef,
    ModuleResolution,
    ScopeNode,
    ScopePath,
    alias_denotes_constructible_type,
)
from agm.agl.syntax.nodes import (
    AgentDecl,
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    ExportItem,
    FuncDef,
    ImportDecl,
    Program,
    QualifierChain,
    RecordDef,
    ScopeRegion,
    TypeAlias,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, ImportMode, NameT


def _mid_sort_key(m: ModuleId) -> tuple[str, ...]:
    return m.segments


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedModule:
    """Per-module output of the program resolver.

    ``module_id``
        The :class:`~agm.agl.modules.ids.ModuleId` of this module.
    ``resolved``
        The per-module scope resolution output (resolution tables, declared
        agents, functions, etc.).
    ``import_env``
        The import environment computed from this module's import declarations.
    ``exports``
        Export map for this module: maps each exported name to its origin
        :data:`~agm.agl.scope.imports.QName`.  For locally-declared names
        the origin is ``(self_module_id, name)``; for re-exported imported names
        it is the original defining module and name, preserved through chains.
    """

    module_id: ModuleId
    resolved: ModuleResolution
    import_env: ImportEnv
    exports: dict[NameAtom, QName]
    source_text: str


@dataclass(frozen=True, slots=True)
class ResolvedProgram:
    """Output of :func:`resolve_program`.

    ``modules``
        Maps each :class:`~agm.agl.modules.ids.ModuleId` to its
        :class:`ResolvedModule`.
    ``entry_id``
        Always :data:`~agm.agl.modules.ids.ENTRY_ID`.
    ``all_public_funcs``
        Whole-program pre-pass table mapping ``(ModuleId, name)`` to the
        :class:`~agm.agl.syntax.nodes.FuncDef` node.  Contains every
        top-level function across all modules.
    ``all_public_types``
        Whole-program pre-pass table mapping ``(ModuleId, name)`` to the
        type declaration node (``RecordDef | EnumDef | TypeAlias``).
    ``entry_agents``
        Agent declarations from the entry module
        (``(scope_path, name)`` → ``AgentDecl``).
    ``import_sccs``
        Loader-computed import strongly-connected components, retained in their
        deterministic reverse-topological order for downstream program passes.
    ``warnings``
        Collected non-fatal scope-pass diagnostics from all modules.
    """

    modules: dict[ModuleId, ResolvedModule]
    entry_id: ModuleId
    all_public_funcs: dict[QName, FuncDef]
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias]
    entry_agents: dict[AgentKey, AgentDecl]
    import_sccs: tuple[tuple[ModuleId, ...], ...]
    warnings: tuple[Diagnostic, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_cross_module_constructor_candidates(
    import_env: ImportEnv,
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
    cross_module_constructor_refs: Mapping[QName, ConstructorRef],
    import_envs: Mapping[ModuleId, ImportEnv],
) -> tuple[dict[str, tuple[ConstructorRef, ...]], frozenset[str]]:
    """Build constructor candidates from open-imported types for a module.

    For each type exposed via unqualified (open) import:
    - RecordDef: add the record name as a candidate (e.g. ``Foo(x:1)``).
    - EnumDef: add each variant name as a candidate (e.g. ``Red``).
    - TypeAlias: add the alias name only when its chain provably ends at a
      constructible type, judged from its DECLARING module's environment —
      an alias can reach its target through that module's own import, not the
      consumer's — which is why *import_envs* is the whole program's table.

    A selected QName may also name an enum variant directly (e.g. an
    individually imported/renamed variant); such names are absent from
    ``all_public_types`` (which is keyed by owning-type QName), so they are
    resolved through ``cross_module_constructor_refs`` instead, which already
    carries a per-variant :class:`ConstructorRef`.

    Returns ``(candidates, type_names)`` where ``type_names`` is the set of
    open-imported type names (for qualified constructor access like ``Color::Red``).
    """
    candidates: dict[str, list[ConstructorRef]] = {}
    type_names: set[str] = set()
    seen: set[QName] = set()
    for exposed_name, qnames in import_env.unqualified.items():
        if not isinstance(exposed_name, str):
            continue
        for mid, src_name in qnames:
            key = (mid, src_name)
            if key in seen:
                continue
            seen.add(key)
            decl = all_public_types.get(key)
            if decl is None:
                variant_ref = cross_module_constructor_refs.get(key)
                if variant_ref is not None:
                    candidates.setdefault(exposed_name, []).append(variant_ref)
                continue
            type_names.add(exposed_name)
            src_path = (src_name,) if isinstance(src_name, str) else src_name
            owner_path = src_path[:-1]

            def declaring_module_lookup(
                target: str,
                qualifier: QualifierChain | None,
                *,
                declaring_module: ModuleId = mid,
                declaring_path: PathAtom = owner_path,
            ) -> RecordDef | EnumDef | ExceptionDef | TypeAlias | None:
                return resolve_alias_target(
                    target,
                    qualifier,
                    self_module_id=declaring_module,
                    import_env=import_envs.get(declaring_module, EMPTY_IMPORT_ENV),
                    all_public_types=all_public_types,
                    scope_path=declaring_path,
                )

            if isinstance(decl, (RecordDef, ExceptionDef)) or (
                isinstance(decl, TypeAlias)
                and isinstance(decl.type_expr, (NameT, AppliedT))
                and alias_denotes_constructible_type(decl, declaring_module_lookup)
            ):
                cref = ConstructorRef(
                    owner_name=decl.name,
                    variant=None,
                    owner_decl_node_id=decl.node_id,
                    type_params=decl.type_params,
                    owner_module_id=mid,
                    owner_path=owner_path,
                )
                candidates.setdefault(exposed_name, []).append(cref)
            elif isinstance(decl, EnumDef):
                for variant in decl.variants:
                    if (mid, variant.name) in all_public_types and isinstance(
                        all_public_types[(mid, variant.name)], ExceptionDef
                    ):
                        continue
                    cref = ConstructorRef(
                        owner_name=decl.name,
                        variant=variant.name,
                        owner_decl_node_id=decl.node_id,
                        type_params=decl.type_params,
                        owner_module_id=mid,
                        owner_path=owner_path,
                        can_match_bare_pattern=not variant.fields,
                    )
                    candidates.setdefault(variant.name, []).append(cref)
    return (
        {name: tuple(refs) for name, refs in candidates.items()},
        frozenset(type_names),
    )


def _atom(path: PathAtom) -> NameAtom:
    """Keep root atoms compatible while representing scoped atoms structurally."""
    return path[0] if len(path) == 1 else path


def _declaration_items(items: tuple[object, ...]) -> tuple[object, ...]:
    """Flatten static scope regions into their normalized declaration members."""
    result: list[object] = []
    for item in items:
        if isinstance(item, ScopeRegion):
            result.extend(_declaration_items(item.items))
        else:
            result.append(item)
    return tuple(result)


def _item_atom(
    item: FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias | BuiltinVarDecl,
) -> NameAtom:
    return _atom((*tuple(segment.name for segment in item.scope_path), item.name))


def _compute_local_exports(self_id: ModuleId, program: Program) -> dict[NameAtom, QName]:
    """Compute declaration paths, including members below named scopes."""
    result: dict[NameAtom, QName] = {}
    for item in _declaration_items(program.body.items):
        if isinstance(item, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            atom = _item_atom(item)
            result[atom] = (self_id, atom)
            if isinstance(item, EnumDef):
                for variant in item.variants:
                    variant_atom = _atom(
                        (
                            *tuple(segment.name for segment in item.scope_path),
                            item.name,
                            variant.name,
                        )
                    )
                    result[variant_atom] = (self_id, variant_atom)
        elif isinstance(item, BuiltinVarDecl):
            atom = _item_atom(item)
            result[atom] = (self_id, atom)
    return result


def _cross_module_constructor_refs(
    all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
) -> dict[QName, ConstructorRef]:
    """Build constructor results for publicly selected declaration paths."""
    result: dict[QName, ConstructorRef] = {}
    for (module_id, atom), declaration in all_public_types.items():
        path = (atom,) if isinstance(atom, str) else atom
        if isinstance(declaration, (RecordDef, ExceptionDef)) and path[:-1]:
            result[(module_id, atom)] = ConstructorRef(
                owner_name=declaration.name,
                variant=None,
                owner_decl_node_id=declaration.node_id,
                type_params=declaration.type_params,
                owner_module_id=module_id,
                owner_path=path[:-1],
            )
        elif isinstance(declaration, EnumDef):
            for variant in declaration.variants:
                variant_path = (*path, variant.name)
                variant_atom = _atom(variant_path)
                result[(module_id, variant_atom)] = ConstructorRef(
                    owner_name=declaration.name,
                    variant=variant.name,
                    owner_decl_node_id=declaration.node_id,
                    type_params=declaration.type_params,
                    owner_module_id=module_id,
                    owner_path=path[:-1],
                    can_match_bare_pattern=not variant.fields,
                )
    return result


def _raise_reexport_conflict(
    exposed: NameAtom, existing: QName, origin: QName, decl: ExportDecl
) -> None:
    raise AglScopeError(
        f"re-export name {exposed!r} has conflicting origins:"
        f" {existing[0].display()!r}::{existing[1]!r}"
        f" and {origin[0].display()!r}::{origin[1]!r}",
        span=decl.span,
    )


def _resolve_reexports(
    export_maps: dict[ModuleId, dict[NameAtom, QName]],
    all_targets: dict[int, ImportTarget],
    graph: ModuleGraph,
) -> None:
    """Fixed-point resolution of explicit export declarations across the program.

    Iterates until no new re-exported names are added.  For each ``ExportDecl``,
    this function propagates the target module's exported names into the
    current module's export map with their origin :data:`QName` preserved.

    Re-export name conflicts (same exposed name → different origin QNames)
    raise :class:`~agm.agl.scope.symbols.AglScopeError`.
    """
    changed = True
    while changed:
        changed = False
        for mid, loaded in graph.modules.items():
            for decl in loaded.export_decls:
                target = all_targets[decl.node_id]
                if isinstance(target, SingleTarget):
                    target_mids: list[ModuleId] = [target.module]
                else:
                    target_mids = sorted(target.modules, key=_mid_sort_key)

                for target_mid in target_mids:
                    target_exports = export_maps.get(target_mid, {})
                    additions = _compute_reexport_additions(
                        decl, target_exports, allow_missing=True
                    )
                    current_exports = export_maps[mid]
                    for exposed, qname in additions.items():
                        existing = current_exports.get(exposed)
                        if existing is None:
                            current_exports[exposed] = qname
                            changed = True
                        elif existing != qname:
                            _raise_reexport_conflict(exposed, existing, qname, decl)

    # A selection may target a re-export which is populated later in the
    # fixed point. Validate only after every reachable export has propagated.
    for _mid, loaded in graph.modules.items():
        for decl in loaded.export_decls:
            target = all_targets[decl.node_id]
            validation_targets = (
                (target.module,)
                if isinstance(target, SingleTarget)
                else tuple(sorted(target.modules, key=_mid_sort_key))
            )
            for target_mid in validation_targets:
                _compute_reexport_additions(decl, export_maps.get(target_mid, {}))


def _compute_reexport_additions(
    decl: ExportDecl,
    target_exports: dict[NameAtom, QName],
    *,
    allow_missing: bool = False,
) -> dict[NameAtom, QName]:
    """Compute names to add to the current module's exports from one ExportDecl.

    Returns a dict of ``exposed_name → origin_qname``.  This is called once
    per (module, export-decl, target-module) triple during the fixed-point.
    A region-scoped ``decl`` re-roots every forwarded atom under its own
    scope path, exactly as a ``using … as`` rename re-roots a selected atom.
    """
    result: dict[NameAtom, QName] = {}
    region_prefix = tuple(segment.name for segment in decl.scope_path)

    def item_path(item: ExportItem) -> PathAtom:
        return (*tuple(segment.name for segment in item.scope_path), item.name)

    def matches(prefix: PathAtom) -> tuple[NameAtom, ...]:
        return tuple(
            atom
            for atom in target_exports
            if ((atom,) if isinstance(atom, str) else atom)[: len(prefix)] == prefix
        )

    selected: dict[NameAtom, None] = {}
    if decl.mode is ImportMode.ALL:
        selected = dict.fromkeys(target_exports)
    else:
        for item in decl.items:
            matched = matches(item_path(item))
            if not matched:
                if allow_missing:
                    continue
                raise AglScopeError(
                    f"name {'::'.join(item_path(item))!r} is not exported by module "
                    f"{'/'.join(decl.module_path)!r}",
                    span=decl.span,
                )
            for atom in matched:
                selected[atom] = None
        if decl.mode is ImportMode.HIDING:
            selected = {atom: None for atom in target_exports if atom not in selected}

    for source in selected:
        source_path = (source,) if isinstance(source, str) else source
        exposed: NameAtom = source
        if decl.mode is ImportMode.USING:
            for item in decl.items:
                prefix = item_path(item)
                if item.rename is not None and source_path[: len(prefix)] == prefix:
                    routed = (item.rename, *source_path[len(prefix) :])
                    exposed = routed[0] if len(routed) == 1 else routed
                    break
        exposed_path = (exposed,) if isinstance(exposed, str) else exposed
        rooted = _atom(region_prefix + exposed_path) if region_prefix else exposed
        origin = target_exports[source]
        existing = result.get(rooted)
        if existing is not None and existing != origin:
            _raise_reexport_conflict(rooted, existing, origin, decl)
        result[rooted] = origin
    return result


def _decl_to_import_target(
    decl: ImportDecl | ExportDecl,
    loaded_modules: Mapping[ModuleId, object],
) -> ImportTarget:
    """Map an import/export declaration to an ImportTarget using the loaded graph.

    For single imports, returns a ``SingleTarget`` with the resolved
    ``ModuleId``.  For wildcard imports, returns a ``WildcardTarget`` with
    all matching loaded modules (excluding the entry sentinel).
    """
    if not decl.wildcard:
        mid = ModuleId(segments=tuple(decl.module_path))
        return SingleTarget(module=mid)
    # Wildcard: all loaded modules whose segments start with decl.module_path
    prefix = tuple(decl.module_path)
    matched = frozenset(
        mid for mid in loaded_modules if not mid.is_entry and mid.segments[: len(prefix)] == prefix
    )
    return WildcardTarget(modules=matched)


# ---------------------------------------------------------------------------
# Cross-module decl info type aliases
# ---------------------------------------------------------------------------

# Maps (module_id, name) → (decl_node_id, decl_span, binder_kind) for use
# when building BindingRef for cross-module references.
_DeclInfo = dict[QName, tuple[int, SourceSpan, BinderKind]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_program(
    graph: ModuleGraph,
    *,
    ambient_agents: frozenset[str] = frozenset(),
    entry_ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
    entry_ambient_type_names: frozenset[str] = frozenset(),
    entry_parent_scope: ScopeNode | None = None,
    entry_repl_session_scope: ScopeNode | None = None,
    entry_repl_session_scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
    entry_repl_session_type_paths: Mapping[ScopePath, str | None] | None = None,
) -> ResolvedProgram:
    """Run the full scope-resolution pass over a :class:`~agm.agl.modules.loader.ModuleGraph`.

    Parameters
    ----------
    graph:
        A loaded module graph from :func:`~agm.agl.modules.loader.load_graph`.
    ambient_agents:
        Agent names the host already backs (passed through to the entry
        resolver; non-entry modules never declare agents).
    entry_ambient_constructor_candidates:
        Constructor candidates from prior REPL entries.  These are merged with
        open-imported constructor candidates for the entry module.
    entry_ambient_type_names:
        Type names from prior REPL entries, used for qualified constructor
        access in the entry module.
    entry_parent_scope:
        When given, the entry module's root scope is parented to this scope
        so name lookups fall through to session bindings (REPL incremental
        mode).
    entry_repl_session_scope:
        When given, passed to the entry resolver so ``::name`` self-references
        can fall back to prior session bindings (REPL program context).
    entry_repl_session_scope_nodes:
        Named scope layers promoted by prior REPL entries. They are copied into
        the entry's resolver so qualified members remain available.
    entry_repl_session_type_paths:
        Type-owned scope paths among the retained layers, each mapped to its
        rendered alias target or to None for a nominal type. Scope needs the
        distinction to reject a method receiver in an alias scope.

    Returns
    -------
    ResolvedProgram
        The resolved graph with per-module resolution tables and whole-program
        pre-pass tables.

    Raises
    ------
    AglScopeError
        On the first static scope violation (first-error abort).
    """
    # ------------------------------------------------------------------
    # Step 1: Build local export maps (own declarations only).
    # ------------------------------------------------------------------
    export_maps: dict[ModuleId, dict[NameAtom, QName]] = {}
    for mid, loaded in graph.modules.items():
        export_maps[mid] = _compute_local_exports(mid, loaded.program)

    # ------------------------------------------------------------------
    # Step 2: Map ImportDecl and ExportDecl → ImportTarget for every module.
    # ------------------------------------------------------------------
    all_targets: dict[int, ImportTarget] = {}
    for _mid, loaded in graph.modules.items():
        for decl in loaded.imports:
            target = _decl_to_import_target(decl, graph.modules)
            all_targets[decl.node_id] = target
        for export_decl in loaded.export_decls:
            target = _decl_to_import_target(export_decl, graph.modules)
            all_targets[export_decl.node_id] = target

    # ------------------------------------------------------------------
    # Step 3: Resolve re-exports (fixed-point propagation).
    # ------------------------------------------------------------------
    _resolve_reexports(export_maps, all_targets, graph)

    # ------------------------------------------------------------------
    # Step 4: Build ImportEnv per module.
    # ------------------------------------------------------------------
    import_envs: dict[ModuleId, ImportEnv] = {}
    for mid, loaded in graph.modules.items():
        decls = loaded.imports
        # Build a targets mapping scoped to this module's declarations.
        module_targets: dict[int, ImportTarget] = {
            decl.node_id: all_targets[decl.node_id] for decl in decls
        }
        import_envs[mid] = build_import_env(decls, module_targets, export_maps)

    # ------------------------------------------------------------------
    # Step 5: Whole-program pre-pass — collect all funcs/types and
    # build decl_info for cross-module BindingRef construction.
    # ------------------------------------------------------------------
    all_public_funcs: dict[QName, FuncDef] = {}
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias] = {}

    # decl_info: (mid, name) → (node_id, span, kind) for building BindingRefs
    decl_info: _DeclInfo = {}

    for mid, loaded in graph.modules.items():
        for item in _declaration_items(loaded.program.body.items):
            if isinstance(item, FuncDef):
                key = (mid, _item_atom(item))
                all_public_funcs[key] = item
                decl_info[key] = (item.node_id, item.span, BinderKind.function_binding)
            elif isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                key = (mid, _item_atom(item))
                all_public_types[key] = item
                kind = (
                    BinderKind.constructor_binding
                    if not isinstance(item, TypeAlias)
                    or isinstance(item.type_expr, (NameT, AppliedT))
                    else BinderKind.let_binding
                )
                decl_info[key] = (item.node_id, item.span, kind)
            elif isinstance(item, BuiltinVarDecl):
                key = (mid, _item_atom(item))
                decl_info[key] = (item.node_id, item.span, BinderKind.builtin_var_binding)

    cross_module_constructor_refs = _cross_module_constructor_refs(all_public_types)
    cross_module_constructible_types = frozenset(
        qname
        for qname, declaration in all_public_types.items()
        if isinstance(declaration, (RecordDef, EnumDef, ExceptionDef))
    )

    # ------------------------------------------------------------------
    # Step 6: Resolve each module's bodies.
    # ------------------------------------------------------------------
    resolved_modules: dict[ModuleId, ResolvedModule] = {}
    all_warnings: list[Diagnostic] = []
    entry_agents: dict[AgentKey, AgentDecl] = {}

    for mid, loaded in graph.modules.items():
        is_entry = mid.is_entry
        # Build cross-module constructor candidates from open imports.
        cross_module_candidates, cross_module_type_names = (
            _build_cross_module_constructor_candidates(
                import_envs[mid], all_public_types, cross_module_constructor_refs, import_envs
            )
        )
        constructor_candidates = cross_module_candidates
        type_names = cross_module_type_names
        if is_entry:
            constructor_candidates = dict(entry_ambient_constructor_candidates or {})
            for name, refs in cross_module_candidates.items():
                constructor_candidates[name] = (*constructor_candidates.get(name, ()), *refs)
            type_names = entry_ambient_type_names | cross_module_type_names
        resolver = _Resolver(
            module_id=mid,
            import_env=import_envs[mid],
            decl_info=decl_info,
            cross_module_constructor_refs=cross_module_constructor_refs,
            cross_module_constructible_types=cross_module_constructible_types,
            cross_module_type_scopes=frozenset(all_public_types),
            all_public_types=all_public_types,
            is_entry=is_entry,
            repl_session_scope=entry_repl_session_scope if is_entry else None,
            repl_session_scope_nodes=entry_repl_session_scope_nodes if is_entry else None,
            repl_session_type_paths=entry_repl_session_type_paths if is_entry else None,
            origin_path=loaded.path,
            spaced_qualifiers=loaded.spaced_qualifiers,
        )
        resolved = resolver.run(
            loaded.program,
            parent_scope=entry_parent_scope if is_entry else None,
            ambient_agents=ambient_agents if is_entry else frozenset(),
            ambient_constructor_candidates=constructor_candidates or None,
            ambient_type_names=type_names,
        )
        all_warnings.extend(resolved.warnings)
        resolved_modules[mid] = ResolvedModule(
            module_id=mid,
            resolved=resolved,
            import_env=import_envs[mid],
            exports=export_maps[mid],
            source_text=graph.modules[mid].source_text,
        )
        if is_entry:
            entry_agents = dict(resolved.declared_agents)

    return ResolvedProgram(
        modules=resolved_modules,
        entry_id=graph.entry_id,
        all_public_funcs=all_public_funcs,
        all_public_types=all_public_types,
        entry_agents=entry_agents,
        import_sccs=graph.sccs,
        warnings=tuple(all_warnings),
    )
