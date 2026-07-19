"""Program-level scope resolver for the AgL module system.

This module provides :func:`resolve_program`, which runs the scope-resolution
pass over an entire :class:`~agm.agl.modules.loader.ModuleGraph`, producing
a :class:`ResolvedProgram` that contains per-module :class:`ResolvedModule`
results plus whole-program pre-pass tables.

Design
------
- **Export maps**: non-private top-level ``def``/``record``/``enum``/``type``
  names per module plus explicit ``export`` declarations, computed before any
  body is resolved.
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
    ImportEnv,
    ImportTarget,
    QName,
    SingleTarget,
    WildcardTarget,
    build_import_env,
)
from agm.agl.scope.resolver import _Resolver
from agm.agl.scope.symbols import (
    AglScopeError,
    BinderKind,
    ConstructorRef,
    ModuleResolution,
    ScopeNode,
)
from agm.agl.syntax.nodes import (
    AgentDecl,
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    FuncDef,
    ImportDecl,
    Program,
    RecordDef,
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
        :data:`~agm.agl.scope.imports.QName`.  For locally-defined public names
        the origin is ``(self_module_id, name)``; for re-exported imported names
        it is the original defining module and name, preserved through chains.
    """

    module_id: ModuleId
    resolved: ModuleResolution
    import_env: ImportEnv
    exports: dict[str, QName]
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
        :class:`~agm.agl.syntax.nodes.FuncDef` node.  Contains only
        non-private top-level functions across all modules.
    ``all_public_types``
        Whole-program pre-pass table mapping ``(ModuleId, name)`` to the
        type declaration node (``RecordDef | EnumDef | TypeAlias``).
    ``entry_agents``
        Agent declarations from the entry module (name → ``AgentDecl``).
    ``warnings``
        Collected non-fatal scope-pass diagnostics from all modules.
    """

    modules: dict[ModuleId, ResolvedModule]
    entry_id: ModuleId
    all_public_funcs: dict[tuple[ModuleId, str], FuncDef]
    all_public_types: dict[tuple[ModuleId, str], RecordDef | EnumDef | ExceptionDef | TypeAlias]
    entry_agents: dict[str, AgentDecl]
    warnings: tuple[Diagnostic, ...]
    private_info: Mapping[tuple[ModuleId, str], bool]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_cross_module_constructor_candidates(
    import_env: ImportEnv,
    all_public_types: dict[tuple[ModuleId, str], RecordDef | EnumDef | ExceptionDef | TypeAlias],
) -> tuple[dict[str, tuple[ConstructorRef, ...]], frozenset[str]]:
    """Build constructor candidates from open-imported types for a module.

    For each type exposed via unqualified (open) import:
    - RecordDef: add the record name as a candidate (e.g. ``Foo(x:1)``).
    - EnumDef: add each variant name as a candidate (e.g. ``Red``).
    - TypeAlias: not constructible, skip.

    Returns ``(candidates, type_names)`` where ``type_names`` is the set of
    open-imported type names (for qualified constructor access like ``Color::Red``).
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
            if isinstance(decl, (RecordDef, ExceptionDef)):
                cref = ConstructorRef(
                    owner_name=decl.name,
                    variant=None,
                    owner_decl_node_id=decl.node_id,
                    type_params=decl.type_params,
                    owner_module_id=mid,
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
                    )
                    candidates.setdefault(variant.name, []).append(cref)
    return (
        {name: tuple(refs) for name, refs in candidates.items()},
        frozenset(type_names),
    )


def _compute_local_exports(self_id: ModuleId, program: Program) -> dict[str, QName]:
    """Compute the local export map for a module from its own declarations.

    Returns a dict mapping each non-private top-level name to the QName
    ``(self_id, name)``.  Re-exported names are NOT included here;
    they are added by :func:`_resolve_reexports` in a subsequent pass.
    """
    result: dict[str, QName] = {}
    for item in program.body.items:
        if isinstance(item, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            if not item.is_private:
                result[item.name] = (self_id, item.name)
        elif isinstance(item, BuiltinVarDecl):
            # The resolver admits these only in ``std/config``; its engine
            # settings are public so callers can use ``std/config::name``.
            result[item.name] = (self_id, item.name)
    return result


def _resolve_reexports(
    export_maps: dict[ModuleId, dict[str, QName]],
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
                    additions = _compute_reexport_additions(decl, target_exports)
                    current_exports = export_maps[mid]
                    for exposed, qname in additions.items():
                        existing = current_exports.get(exposed)
                        if existing is None:
                            current_exports[exposed] = qname
                            changed = True
                        elif existing != qname:
                            raise AglScopeError(
                                f"re-export name {exposed!r} has conflicting origins:"
                                f" {existing[0].path_str()!r}::{existing[1]!r}"
                                f" and {qname[0].path_str()!r}::{qname[1]!r}",
                                span=decl.span,
                            )


def _compute_reexport_additions(
    decl: ExportDecl,
    target_exports: dict[str, QName],
) -> dict[str, QName]:
    """Compute names to add to the current module's exports from one ExportDecl.

    Returns a dict of ``exposed_name → origin_qname``.  This is called once
    per (module, export-decl, target-module) triple during the fixed-point.
    """
    result: dict[str, QName] = {}

    if decl.mode == ImportMode.HIDING:
        hidden = frozenset(item.name for item in decl.items)
        selected = frozenset(target_exports.keys()) - hidden
    elif decl.mode == ImportMode.USING:
        selected = frozenset(item.name for item in decl.items)
    else:
        selected = frozenset(target_exports.keys())

    rename_map: dict[str, str] = {}
    if decl.mode == ImportMode.USING:
        for item in decl.items:
            if item.rename is not None:
                rename_map[item.name] = item.rename

    for src_name in selected:
        origin = target_exports.get(src_name)
        if origin is None:
            continue  # name not yet propagated; fixed-point will retry
        exposed = rename_map.get(src_name, src_name)
        result[exposed] = origin

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
_DeclInfo = dict[tuple[ModuleId, str], tuple[int, SourceSpan, BinderKind]]

# Maps (module_id, name) → True for private decls (for private-access error messages).
_PrivateInfo = dict[tuple[ModuleId, str], bool]


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
    export_maps: dict[ModuleId, dict[str, QName]] = {}
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
    # Step 5: Whole-program pre-pass — collect public funcs/types and
    # build decl_info for cross-module BindingRef construction.
    # ------------------------------------------------------------------
    all_public_funcs: dict[tuple[ModuleId, str], FuncDef] = {}
    all_public_types: dict[
        tuple[ModuleId, str], RecordDef | EnumDef | ExceptionDef | TypeAlias
    ] = {}

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
            elif isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                key = (mid, item.name)
                if item.is_private:
                    private_info[key] = True
                else:
                    all_public_types[key] = item
                    # A syntactically nominal alias can transparently name a
                    # record, enum, or another alias. Scope admits that shape
                    # as a possible constructor owner and leaves semantic
                    # target validation to type checking. Container,
                    # function, and primitive aliases remain definitively
                    # non-constructible at this boundary.
                    kind = (
                        BinderKind.constructor_binding
                        if not isinstance(item, TypeAlias)
                        or isinstance(item.type_expr, (NameT, AppliedT))
                        else BinderKind.let_binding
                    )
                    decl_info[key] = (item.node_id, item.span, kind)
            elif isinstance(item, BuiltinVarDecl):
                # The resolver admits this declaration only in ``std/config``.
                # Record its mutable kind so qualified reads and writes resolve.
                key = (mid, item.name)
                decl_info[key] = (item.node_id, item.span, BinderKind.builtin_var_binding)

    # ------------------------------------------------------------------
    # Step 6: Resolve each module's bodies.
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
            private_info=private_info,
            is_entry=is_entry,
            repl_session_scope=entry_repl_session_scope if is_entry else None,
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
        warnings=tuple(all_warnings),
        private_info=private_info,
    )
