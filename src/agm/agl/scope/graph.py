"""Graph-aware scope resolver for the AgL module system (M3b).

This module provides :func:`resolve_graph`, which runs the scope-resolution
pass over an entire :class:`~agm.agl.modules.loader.ModuleGraph`, producing
a :class:`ResolvedModuleGraph` that contains per-module :class:`ResolvedModule`
results plus whole-graph pre-pass tables.

Design
------
- **Export sets**: non-private top-level ``def``/``record``/``enum``/``type``
  names per module, computed before any body is resolved.
- **ImportEnv per module**: built from each module's import declarations against
  the already-loaded graph (no re-reading files).
- **Whole-graph pre-pass tables**: ``all_public_funcs`` and ``all_public_types``
  collected BEFORE resolving any body, enabling cross-module mutual recursion
  (D8).
- **Declaration-only enforcement**: non-entry modules may only contain
  declarations (``def``, ``record``, ``enum``, ``type``, ``import``).
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
    ImportEnv,
    ImportTarget,
    SingleTarget,
    WildcardTarget,
    build_import_env,
)
from agm.agl.scope.resolver import _Resolver
from agm.agl.scope.symbols import BinderKind, ConstructorRef, ResolvedProgram, ScopeNode
from agm.agl.syntax.nodes import (
    AgentDecl,
    EnumDef,
    FuncDef,
    ImportDecl,
    Program,
    RecordDef,
    TypeAlias,
)
from agm.agl.syntax.spans import SourceSpan

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedModule:
    """Per-module output of the graph resolver.

    ``module_id``
        The :class:`~agm.agl.modules.ids.ModuleId` of this module.
    ``resolved``
        The per-module scope resolution output (resolution tables, declared
        agents, functions, etc.).
    ``import_env``
        The import environment computed from this module's import declarations.
    ``exports``
        Frozenset of non-private top-level ``def``/``record``/``enum``/``type``
        names exported by this module.
    """

    module_id: ModuleId
    resolved: ResolvedProgram
    import_env: ImportEnv
    exports: frozenset[str]
    source_text: str


@dataclass(frozen=True, slots=True)
class ResolvedModuleGraph:
    """Output of :func:`resolve_graph`.

    ``modules``
        Maps each :class:`~agm.agl.modules.ids.ModuleId` to its
        :class:`ResolvedModule`.
    ``entry_id``
        Always :data:`~agm.agl.modules.ids.ENTRY_ID`.
    ``all_public_funcs``
        Whole-graph pre-pass table mapping ``(ModuleId, name)`` to the
        :class:`~agm.agl.syntax.nodes.FuncDef` node.  Contains only
        non-private top-level functions across all modules.
    ``all_public_types``
        Whole-graph pre-pass table mapping ``(ModuleId, name)`` to the
        type declaration node (``RecordDef | EnumDef | TypeAlias``).
    ``entry_agents``
        Agent declarations from the entry module (name → ``AgentDecl``).
    ``warnings``
        Collected non-fatal scope-pass diagnostics from all modules.
    """

    modules: dict[ModuleId, ResolvedModule]
    entry_id: ModuleId
    all_public_funcs: dict[tuple[ModuleId, str], FuncDef]
    all_public_types: dict[tuple[ModuleId, str], RecordDef | EnumDef | TypeAlias]
    entry_agents: dict[str, AgentDecl]
    warnings: tuple[Diagnostic, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_cross_module_constructor_candidates(
    import_env: ImportEnv,
    all_public_types: dict[tuple[ModuleId, str], RecordDef | EnumDef | TypeAlias],
) -> tuple[dict[str, tuple[ConstructorRef, ...]], frozenset[str]]:
    """Build constructor candidates from open-imported types for a module.

    For each type exposed via unqualified (open) import:
    - RecordDef: add the record name as a candidate (e.g. ``Foo(x:1)``).
    - EnumDef: add each variant name as a candidate (e.g. ``Red``).
    - TypeAlias: not constructible, skip.

    Returns ``(candidates, type_names)`` where ``type_names`` is the set of
    open-imported type names (for qualified constructor access like ``Color.Red``).
    """
    candidates: dict[str, list[ConstructorRef]] = {}
    type_names: set[str] = set()
    seen: set[tuple[ModuleId, str]] = set()
    for exposed_name, qnames in import_env.unqualified.items():
        for mid, src_name in qnames:
            key = (mid, src_name)
            if key in seen:
                continue
            seen.add(key)
            decl = all_public_types.get(key)
            if decl is None:
                continue
            type_names.add(exposed_name)
            if isinstance(decl, RecordDef):
                cref = ConstructorRef(
                    owner_name=decl.name,
                    variant=None,
                    owner_decl_node_id=decl.node_id,
                    type_params=decl.type_params,
                )
                candidates.setdefault(exposed_name, []).append(cref)
            elif isinstance(decl, EnumDef):
                for variant in decl.variants:
                    cref = ConstructorRef(
                        owner_name=decl.name,
                        variant=variant.name,
                        owner_decl_node_id=decl.node_id,
                        type_params=decl.type_params,
                    )
                    candidates.setdefault(variant.name, []).append(cref)
    return (
        {name: tuple(refs) for name, refs in candidates.items()},
        frozenset(type_names),
    )


def _compute_exports(program: Program) -> frozenset[str]:
    """Compute the export set for a module from its program items.

    Returns the set of non-private top-level ``FuncDef``, ``RecordDef``,
    ``EnumDef``, and ``TypeAlias`` names.
    """
    result: set[str] = set()
    for item in program.body.items:
        if isinstance(item, (FuncDef, RecordDef, EnumDef, TypeAlias)):
            if not item.is_private:
                result.add(item.name)
    return frozenset(result)


def _decl_to_import_target(
    decl: ImportDecl,
    graph_modules: Mapping[ModuleId, object],
) -> ImportTarget:
    """Map an ImportDecl to an ImportTarget using the already-loaded graph.

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
        mid
        for mid in graph_modules
        if not mid.is_entry and mid.segments[: len(prefix)] == prefix
    )
    return WildcardTarget(modules=matched)


# ---------------------------------------------------------------------------
# Cross-module decl info type aliases
# ---------------------------------------------------------------------------

# Maps (module_id, name) → (decl_node_id, decl_span, binder_kind) for use
# when building BindingRef for cross-module references.
_DeclInfo = dict[tuple[ModuleId, str], tuple[int, SourceSpan, BinderKind]]

# Maps (module_id, name) → True for private decls (for private-access error messages).
_PrivateInfo = dict[tuple[ModuleId, str], bool]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_graph(
    graph: ModuleGraph,
    *,
    ambient_agents: frozenset[str] = frozenset(),
    entry_parent_scope: ScopeNode | None = None,
    entry_repl_session_scope: ScopeNode | None = None,
) -> ResolvedModuleGraph:
    """Run the full scope-resolution pass over a :class:`~agm.agl.modules.loader.ModuleGraph`.

    Parameters
    ----------
    graph:
        A loaded module graph from :func:`~agm.agl.modules.loader.load_graph`.
    ambient_agents:
        Agent names the host already backs (passed through to the entry
        resolver; non-entry modules never declare agents).
    entry_parent_scope:
        When given, the entry module's root scope is parented to this scope
        so name lookups fall through to session bindings (REPL incremental
        mode, M6).
    entry_repl_session_scope:
        When given, passed to the entry resolver so ``::name`` self-references
        can fall back to prior session bindings (REPL graph mode, M6).

    Returns
    -------
    ResolvedModuleGraph
        The resolved graph with per-module resolution tables and whole-graph
        pre-pass tables.

    Raises
    ------
    AglScopeError
        On the first static scope violation (first-error abort).
    """
    # ------------------------------------------------------------------
    # Step 1: Build export sets for every module.
    # ------------------------------------------------------------------
    exports: dict[ModuleId, frozenset[str]] = {}
    for mid, loaded in graph.modules.items():
        exports[mid] = _compute_exports(loaded.program)

    # ------------------------------------------------------------------
    # Step 2: Map ImportDecl → ImportTarget for every module.
    # ------------------------------------------------------------------
    all_targets: dict[int, ImportTarget] = {}
    for _mid, loaded in graph.modules.items():
        for decl in loaded.imports:
            target = _decl_to_import_target(decl, graph.modules)
            all_targets[decl.node_id] = target

    # ------------------------------------------------------------------
    # Step 3: Build ImportEnv per module.
    # ------------------------------------------------------------------
    import_envs: dict[ModuleId, ImportEnv] = {}
    for mid, loaded in graph.modules.items():
        decls = loaded.imports
        # Build a targets mapping scoped to this module's declarations.
        module_targets: dict[int, ImportTarget] = {
            decl.node_id: all_targets[decl.node_id] for decl in decls
        }
        import_envs[mid] = build_import_env(mid, decls, module_targets, exports)

    # ------------------------------------------------------------------
    # Step 4: Whole-graph pre-pass — collect public funcs/types and
    # build decl_info for cross-module BindingRef construction.
    # ------------------------------------------------------------------
    all_public_funcs: dict[tuple[ModuleId, str], FuncDef] = {}
    all_public_types: dict[tuple[ModuleId, str], RecordDef | EnumDef | TypeAlias] = {}

    # decl_info: (mid, name) → (node_id, span, kind) for building BindingRefs
    decl_info: _DeclInfo = {}
    # private_info: (mid, name) → True for private decls (for error msgs)
    private_info: _PrivateInfo = {}

    for mid, loaded in graph.modules.items():
        for item in loaded.program.body.items:
            if isinstance(item, FuncDef):
                key = (mid, item.name)
                if item.is_private:
                    private_info[key] = True
                else:
                    all_public_funcs[key] = item
                    decl_info[key] = (item.node_id, item.span, BinderKind.function_binding)
            elif isinstance(item, (RecordDef, EnumDef, TypeAlias)):
                key = (mid, item.name)
                if item.is_private:
                    private_info[key] = True
                else:
                    all_public_types[key] = item
                    # RecordDef/EnumDef use constructor_binding so cross-module
                    # field access (mylib::Color.Red) is detected as type-qualified.
                    # TypeAlias uses let_binding (it's not constructible).
                    kind = (
                        BinderKind.let_binding
                        if isinstance(item, TypeAlias)
                        else BinderKind.constructor_binding
                    )
                    decl_info[key] = (item.node_id, item.span, kind)

    # ------------------------------------------------------------------
    # Step 5: Resolve each module's bodies.
    # ------------------------------------------------------------------
    resolved_modules: dict[ModuleId, ResolvedModule] = {}
    all_warnings: list[Diagnostic] = []
    entry_agents: dict[str, AgentDecl] = {}

    for mid, loaded in graph.modules.items():
        is_entry = mid.is_entry
        # Build cross-module constructor candidates from open imports.
        cross_module_candidates, cross_module_type_names = (
            _build_cross_module_constructor_candidates(import_envs[mid], all_public_types)
        )
        resolver = _Resolver(
            module_id=mid,
            import_env=import_envs[mid],
            decl_info=decl_info,
            private_info=private_info,
            is_entry=is_entry,
            repl_session_scope=entry_repl_session_scope if is_entry else None,
        )
        resolved = resolver.run(
            loaded.program,
            parent_scope=entry_parent_scope if is_entry else None,
            ambient_agents=ambient_agents if is_entry else frozenset(),
            ambient_constructor_candidates=cross_module_candidates or None,
            ambient_type_names=cross_module_type_names,
        )
        all_warnings.extend(resolved.warnings)
        resolved_modules[mid] = ResolvedModule(
            module_id=mid,
            resolved=resolved,
            import_env=import_envs[mid],
            exports=exports[mid],
            source_text=graph.modules[mid].source_text,
        )
        if is_entry:
            entry_agents = dict(resolved.declared_agents)

    return ResolvedModuleGraph(
        modules=resolved_modules,
        entry_id=graph.entry_id,
        all_public_funcs=all_public_funcs,
        all_public_types=all_public_types,
        entry_agents=entry_agents,
        warnings=tuple(all_warnings),
    )
