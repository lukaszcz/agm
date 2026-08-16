"""Program-level scope resolver for the AgL module system.

This module provides :func:`resolve_program`, which runs the scope-resolution
pass over an entire :class:`~agm.agl.modules.loader.ModuleGraph`, producing
a :class:`ResolvedProgram` that contains per-module :class:`ResolvedModule`
results plus whole-program pre-pass tables.

Design
------
- **Public surfaces**: declaration export maps and separate named-scope
  identity maps per module, including explicit ``export`` declarations,
  computed before any body is resolved.
- **Contribution import environment per module**: built from each module's
  import declarations against the already-loaded graph (no re-reading files).
- **Whole-program pre-pass tables**: ``all_public_funcs`` and ``all_public_types``
  collect source declarations BEFORE resolving any body, enabling cross-module
  mutual recursion without publishing host-only synthetic entries.
- **Static module roots**: every file-backed module permits declarations,
  parameters, and bindings but rejects root assignments and bare expressions;
  the incremental REPL is the executable-root host.
- **Header-only imports** (every module root): imports must appear before any
  declaration.
- **``::name`` self-reference**: resolved to the current module's own scope.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agm.agl.modules.ids import ModuleId

if TYPE_CHECKING:
    from agm.agl.modules.loader import ModuleGraph
from agm.agl.scope.imports import (
    EMPTY_IMPORT_ENV,
    ImportEnv,
    ImportTarget,
    NameAtom,
    PathAtom,
    QName,
    ScopeOrigins,
    SingleTarget,
    WildcardTarget,
    build_import_env,
    resolve_alias_target,
    sibling_qname,
)
from agm.agl.scope.resolver import _Resolver
from agm.agl.scope.symbols import (
    AglScopeError,
    BinderKind,
    ConstructorRef,
    ModuleResolution,
    ResolvedUseTarget,
    ScopeNode,
    ScopePath,
    alias_denotes_constructible_type,
)
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.scope.symbols import to_bare_path as _path
from agm.agl.syntax.nodes import (
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    ExportItem,
    FuncDef,
    ImportDecl,
    LetDecl,
    Program,
    QualifierChain,
    RecordDef,
    ScopeRegion,
    TypeAlias,
    VarDecl,
    static_items,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, NameT


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
        functions and related resolution data).
    ``import_env``
        The import environment computed from this module's import declarations.
    ``exports``
        Declaration export map for this module: maps each exported name to its
        origin :data:`~agm.agl.scope.imports.QName`.
    ``scope_exports``
        Named-scope export map. Scope identities are separate from declaration
        exports because an empty scope is public without denoting a value.
        Re-exports preserve and merge the scope's original module/path origins.
    """

    module_id: ModuleId
    resolved: ModuleResolution
    import_env: ImportEnv
    exports: dict[NameAtom, QName]
    scope_exports: dict[NameAtom, ScopeOrigins]
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
        :class:`~agm.agl.syntax.nodes.FuncDef` node. Contains every source-level
        function across all modules; host-only synthetic entries are excluded.
    ``all_public_types``
        Whole-program pre-pass table mapping ``(ModuleId, name)`` to the
        type declaration node (``RecordDef | EnumDef | TypeAlias``).
    ``import_sccs``
        The graph's import strongly-connected components, in their
        deterministic reverse-topological order for downstream program passes.
    ``graph``
        The loaded module graph, retained so consumers that need module
        reachability use its import/export adjacency rather than scope data.
    """

    modules: dict[ModuleId, ResolvedModule]
    entry_id: ModuleId
    all_public_funcs: dict[QName, FuncDef]
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias]
    graph: ModuleGraph

    @property
    def import_sccs(self) -> tuple[tuple[ModuleId, ...], ...]:
        """Return the loaded graph's import strongly-connected components."""

        return self.graph.sccs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _constructor_ref_for_type(
    qname: QName,
    declaration: RecordDef | EnumDef | ExceptionDef | TypeAlias,
    import_envs: Mapping[ModuleId, ImportEnv],
    all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
) -> ConstructorRef | None:
    """Return the declaration's constructor identity when its type is constructible."""
    module_id, atom = qname
    path = (atom,) if isinstance(atom, str) else atom

    def declaring_module_lookup(
        target: str, qualifier: QualifierChain | None
    ) -> RecordDef | EnumDef | ExceptionDef | TypeAlias | None:
        return resolve_alias_target(
            target,
            qualifier,
            self_module_id=module_id,
            import_env=import_envs.get(module_id, EMPTY_IMPORT_ENV),
            all_public_types=all_public_types,
            scope_path=path[:-1],
        )

    if not isinstance(declaration, (RecordDef, ExceptionDef)) and not (
        isinstance(declaration, TypeAlias)
        and isinstance(declaration.type_expr, (NameT, AppliedT))
        and alias_denotes_constructible_type(declaration, declaring_module_lookup)
    ):
        return None
    return ConstructorRef(
        owner_name=declaration.name,
        variant=None,
        owner_decl_node_id=declaration.node_id,
        type_params=declaration.type_params,
        owner_module_id=module_id,
        owner_path=path[:-1],
    )


def _build_cross_module_constructor_candidates(
    import_env: ImportEnv,
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
    cross_module_constructor_refs: Mapping[QName, ConstructorRef],
    import_envs: Mapping[ModuleId, ImportEnv],
) -> tuple[dict[str, tuple[ConstructorRef, ...]], frozenset[str]]:
    """Build constructor candidates from types exposed by import tails for a module.

    For each type exposed unqualified by an import tail:
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
    import-tail-exposed type names (for qualified constructor access like ``Color::Red``).
    """
    candidates: dict[str, list[ConstructorRef]] = {}
    type_names: set[str] = set()
    exposed_qnames = frozenset(
        qname for qnames in import_env.unqualified.values() for qname in qnames
    )
    seen_candidates: set[tuple[str, ConstructorRef]] = set()

    def add_candidate(name: str, ref: ConstructorRef) -> None:
        candidate = (name, ref)
        if candidate not in seen_candidates:
            seen_candidates.add(candidate)
            candidates.setdefault(name, []).append(ref)

    for exposed_name, qnames in import_env.unqualified.items():
        if not isinstance(exposed_name, str):
            continue
        for mid, src_name in qnames:
            key = (mid, src_name)
            decl = all_public_types.get(key)
            if decl is None:
                variant_ref = cross_module_constructor_refs.get(key)
                if variant_ref is not None:
                    add_candidate(exposed_name, variant_ref)
                continue
            type_names.add(exposed_name)
            src_path = (src_name,) if isinstance(src_name, str) else src_name
            constructor = _constructor_ref_for_type(key, decl, import_envs, all_public_types)
            if constructor is not None:
                add_candidate(exposed_name, constructor)
            elif isinstance(decl, EnumDef):
                for variant in decl.variants:
                    exception_qname = sibling_qname(key, variant.name)
                    if exception_qname in exposed_qnames and isinstance(
                        all_public_types.get(exception_qname), ExceptionDef
                    ):
                        continue
                    variant_qname = (mid, _atom((*src_path, variant.name)))
                    if variant_qname not in exposed_qnames:
                        continue
                    cref = ConstructorRef(
                        owner_name=decl.name,
                        variant=variant.name,
                        owner_decl_node_id=decl.node_id,
                        type_params=decl.type_params,
                        owner_module_id=mid,
                        owner_path=src_path[:-1],
                        can_match_bare_pattern=not variant.fields,
                    )
                    add_candidate(variant.name, cref)
    return (
        {name: tuple(refs) for name, refs in candidates.items()},
        frozenset(type_names),
    )


def _item_atom(
    item: FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias | BuiltinVarDecl,
) -> NameAtom:
    return _atom((*tuple(segment.name for segment in item.scope_path), item.name))


def _compute_local_scope_exports(
    self_id: ModuleId, program: Program
) -> dict[NameAtom, ScopeOrigins]:
    """Collect public named-scope identities from regions and shorthand paths."""
    result: dict[NameAtom, ScopeOrigins] = {}

    def add_path(path: PathAtom) -> None:
        for length in range(1, len(path) + 1):
            atom = _atom(path[:length])
            result[atom] = frozenset({(self_id, atom)})

    def collect_regions(items: Iterable[object], parent: PathAtom) -> None:
        for item in items:
            if not isinstance(item, ScopeRegion):
                continue
            path = (*parent, item.segment.name)
            add_path(path)
            collect_regions(item.items, path)

    collect_regions(program.body.items, ())
    for item in static_items(program.body.items):
        if isinstance(
            item,
            (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias, LetDecl, VarDecl),
        ):
            add_path(tuple(segment.name for segment in item.scope_path))
    return result


def _compute_local_exports(self_id: ModuleId, program: Program) -> dict[NameAtom, QName]:
    """Compute declaration paths, including members below named scopes."""
    result: dict[NameAtom, QName] = {}
    for item in static_items(program.body.items):
        if isinstance(item, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            if isinstance(item, FuncDef) and item.is_synthetic:
                continue
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
    import_envs: Mapping[ModuleId, ImportEnv],
) -> dict[QName, ConstructorRef]:
    """Build constructor results for publicly selected declaration paths."""
    result: dict[QName, ConstructorRef] = {}
    for (module_id, atom), declaration in all_public_types.items():
        path = (atom,) if isinstance(atom, str) else atom
        constructor = _constructor_ref_for_type(
            (module_id, atom), declaration, import_envs, all_public_types
        )
        if constructor is not None:
            result[(module_id, atom)] = constructor
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


def _raise_reexport_scope_conflict(exposed: NameAtom, decl: ExportDecl) -> None:
    raise AglScopeError(
        f"Name {exposed!r} cannot be both an ordinary declaration and a scope.",
        span=decl.span,
    )


def _resolve_reexports(
    export_maps: dict[ModuleId, dict[NameAtom, QName]],
    scope_export_maps: dict[ModuleId, dict[NameAtom, ScopeOrigins]],
    type_origins: frozenset[QName],
    all_targets: dict[int, ImportTarget],
    graph: ModuleGraph,
) -> None:
    """Fixed-point resolution of explicit export declarations across the program.

    Iterates until no new re-exported names are added.  For each ``ExportDecl``,
    this function propagates the target module's exported names into the
    current module's export map with their origin :data:`QName` preserved.

    Re-export name conflicts (same exposed name → different origin QNames)
    raise :class:`~agm.agl.scope.symbols.AglScopeError`.

    Propagation is monotone: a contribution that does not revisit an export
    declaration crosses at most ``declaration_count`` declarations, and one
    additional pass observes convergence. Continued growth after that can
    only depend on reapplying a declaration through a cycle.
    """

    def propagate() -> tuple[bool, ExportDecl | None]:
        changed = False
        changed_decl: ExportDecl | None = None
        for mid, loaded in graph.modules.items():
            for decl in loaded.export_decls:
                target = all_targets[decl.node_id]
                if isinstance(target, SingleTarget):
                    target_mids: list[ModuleId] = [target.module]
                else:
                    target_mids = sorted(target.modules, key=_mid_sort_key)

                for target_mid in target_mids:
                    additions, scope_additions = _compute_reexport_additions(
                        decl,
                        export_maps.get(target_mid, {}),
                        scope_export_maps.get(target_mid, {}),
                        allow_missing=True,
                    )
                    for exposed, qname in additions.items():
                        if exposed in scope_export_maps[mid] and qname not in type_origins:
                            _raise_reexport_scope_conflict(exposed, decl)
                        existing = export_maps[mid].get(exposed)
                        if existing is None:
                            export_maps[mid][exposed] = qname
                            changed = True
                            changed_decl = decl
                        elif existing != qname:
                            _raise_reexport_conflict(exposed, existing, qname, decl)
                    for exposed, origins in scope_additions.items():
                        existing = export_maps[mid].get(exposed)
                        if existing is not None and existing not in type_origins:
                            _raise_reexport_scope_conflict(exposed, decl)
                        existing_origins = scope_export_maps[mid].get(exposed, frozenset())
                        merged_origins = existing_origins | origins
                        if merged_origins != existing_origins:
                            scope_export_maps[mid][exposed] = merged_origins
                            changed = True
                            changed_decl = decl
        return changed, changed_decl

    declaration_count = sum(len(loaded.export_decls) for loaded in graph.modules.values())
    last_changed_decl: ExportDecl | None = None
    for _ in range(declaration_count + 1):
        changed, changed_decl = propagate()
        if changed_decl is not None:
            last_changed_decl = changed_decl
        if not changed:
            break
    else:
        raise AglScopeError(
            "cyclic re-export expansion does not converge",
            span=last_changed_decl.span if last_changed_decl else None,
        )

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
                _compute_reexport_additions(
                    decl,
                    export_maps.get(target_mid, {}),
                    scope_export_maps.get(target_mid, {}),
                )


def _compute_reexport_additions(
    decl: ExportDecl,
    target_exports: Mapping[NameAtom, QName],
    target_scopes: Mapping[NameAtom, ScopeOrigins],
    *,
    allow_missing: bool = False,
) -> tuple[dict[NameAtom, QName], dict[NameAtom, ScopeOrigins]]:
    """Compute declaration and scope identities forwarded by one export."""
    result: dict[NameAtom, QName] = {}
    scope_result: dict[NameAtom, ScopeOrigins] = {}
    region_prefix = tuple(segment.name for segment in decl.scope_path)

    def item_path(item: ExportItem) -> PathAtom:
        return (*tuple(segment.name for segment in item.scope_path), item.name)

    def matches(surface: Mapping[NameAtom, object], prefix: PathAtom) -> tuple[NameAtom, ...]:
        return tuple(atom for atom in surface if _path(atom)[: len(prefix)] == prefix)

    matched_items: list[tuple[ExportItem, tuple[NameAtom, ...], tuple[NameAtom, ...]]] = []
    for item in (*decl.items, *decl.hidden):
        prefix = item_path(item)
        declarations = matches(target_exports, prefix)
        scopes = matches(target_scopes, prefix)
        if not declarations and not scopes and not allow_missing:
            raise AglScopeError(
                f"name {'::'.join(prefix)!r} is not exported by module "
                f"{'/'.join(decl.module_path)!r}",
                span=decl.span,
            )
        matched_items.append((item, declarations, scopes))

    hidden_declarations = {
        source
        for _item, declarations, _scopes in matched_items[len(decl.items) :]
        for source in declarations
    }
    hidden_scopes = {
        source
        for _item, _declarations, scopes in matched_items[len(decl.items) :]
        for source in scopes
    }

    def add(
        source: NameAtom,
        exposed: NameAtom,
        origins: Mapping[NameAtom, QName],
        destination: dict[NameAtom, QName],
    ) -> None:
        exposed_path = _path(exposed)
        rooted = _atom(region_prefix + exposed_path) if region_prefix else exposed
        origin = origins[source]
        existing = destination.get(rooted)
        if existing is not None and existing != origin:
            _raise_reexport_conflict(rooted, existing, origin, decl)
        destination[rooted] = origin

    if not decl.items:
        for source in target_exports:
            if source not in hidden_declarations:
                add(source, source, target_exports, result)
        for source, origins in target_scopes.items():
            if source not in hidden_scopes:
                exposed_path = _path(source)
                rooted = _atom(region_prefix + exposed_path) if region_prefix else source
                scope_result[rooted] = scope_result.get(rooted, frozenset()) | origins
        return result, scope_result

    for item, declarations, scopes in matched_items[: len(decl.items)]:
        prefix = item_path(item)
        for source in declarations:
            source_path = _path(source)
            exposed = (
                source if item.rename is None else _atom((item.rename, *source_path[len(prefix) :]))
            )
            add(source, exposed, target_exports, result)
        for source in scopes:
            source_path = _path(source)
            exposed = (
                source if item.rename is None else _atom((item.rename, *source_path[len(prefix) :]))
            )
            exposed_path = _path(exposed)
            rooted = _atom(region_prefix + exposed_path) if region_prefix else exposed
            scope_result[rooted] = scope_result.get(rooted, frozenset()) | target_scopes[source]
    return result, scope_result


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

# Maps (module_id, name) → (decl_node_id, decl_span, binder_kind, is_builtin)
# for building BindingRef values for cross-module references.
_DeclInfo = dict[QName, tuple[int, SourceSpan, BinderKind, bool]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_program(
    graph: ModuleGraph,
    *,
    entry_ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
    entry_ambient_type_names: frozenset[str] = frozenset(),
    entry_parent_scope: ScopeNode | None = None,
    entry_repl_session_scope: ScopeNode | None = None,
    entry_repl_session_scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
    entry_repl_session_type_paths: Mapping[ScopePath, str | None] | None = None,
    entry_retained_use_targets: Mapping[int, ResolvedUseTarget] | None = None,
    _entry_use_targets_only: bool = False,
) -> ResolvedProgram:
    """Run the full scope-resolution pass over a :class:`~agm.agl.modules.loader.ModuleGraph`.

    Parameters
    ----------
    graph:
        A loaded module graph from :func:`~agm.agl.modules.loader.load_graph`.
    entry_ambient_constructor_candidates:
        Constructor candidates from prior REPL entries.  These are merged with
        import-tail-exposed constructor candidates for the entry module.
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
    scope_export_maps: dict[ModuleId, dict[NameAtom, ScopeOrigins]] = {}
    type_origins: set[QName] = set()
    for mid, loaded in graph.modules.items():
        export_maps[mid] = _compute_local_exports(mid, loaded.program)
        scope_export_maps[mid] = _compute_local_scope_exports(mid, loaded.program)
        type_origins.update(
            (mid, _item_atom(item))
            for item in static_items(loaded.program.body.items)
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias))
        )

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
    _resolve_reexports(export_maps, scope_export_maps, frozenset(type_origins), all_targets, graph)

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
        import_envs[mid] = build_import_env(decls, module_targets, export_maps, scope_export_maps)

    # ------------------------------------------------------------------
    # Step 5: Whole-program pre-pass — collect all funcs/types and
    # build decl_info for cross-module BindingRef construction.
    # ------------------------------------------------------------------
    all_public_funcs: dict[QName, FuncDef] = {}
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias] = {}

    # decl_info: (mid, name) → (node_id, span, kind) for building BindingRefs
    decl_info: _DeclInfo = {}

    for mid, loaded in graph.modules.items():
        for item in static_items(loaded.program.body.items):
            if isinstance(item, FuncDef):
                if item.is_synthetic:
                    continue
                key = (mid, _item_atom(item))
                all_public_funcs[key] = item
                decl_info[key] = (
                    item.node_id,
                    item.span,
                    BinderKind.function_binding,
                    item.is_builtin,
                )
            elif isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                key = (mid, _item_atom(item))
                all_public_types[key] = item
                kind = (
                    BinderKind.constructor_binding
                    if not isinstance(item, TypeAlias)
                    or isinstance(item.type_expr, (NameT, AppliedT))
                    else BinderKind.let_binding
                )
                decl_info[key] = (item.node_id, item.span, kind, False)
            elif isinstance(item, BuiltinVarDecl):
                key = (mid, _item_atom(item))
                decl_info[key] = (
                    item.node_id,
                    item.span,
                    BinderKind.builtin_var_binding,
                    False,
                )

    cross_module_constructor_refs = _cross_module_constructor_refs(all_public_types, import_envs)
    cross_module_constructible_types = frozenset(
        qname
        for qname, declaration in all_public_types.items()
        if isinstance(declaration, (RecordDef, EnumDef, ExceptionDef))
    )
    # ------------------------------------------------------------------
    # Step 6: Resolve each module's bodies.
    # ------------------------------------------------------------------
    resolved_modules: dict[ModuleId, ResolvedModule] = {}

    for mid, loaded in graph.modules.items():
        is_entry = mid.is_entry
        # Build cross-module constructor candidates from unqualified import tails.
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
            allow_root_statements=is_entry and entry_parent_scope is not None,
            repl_session_scope=entry_repl_session_scope if is_entry else None,
            repl_session_scope_nodes=entry_repl_session_scope_nodes if is_entry else None,
            repl_session_type_paths=entry_repl_session_type_paths if is_entry else None,
            retained_use_targets=entry_retained_use_targets if is_entry else None,
            origin_path=loaded.path,
            spaced_qualifiers=loaded.spaced_qualifiers,
        )
        resolved = resolver.run(
            loaded.program,
            parent_scope=entry_parent_scope if is_entry else None,
            ambient_constructor_candidates=constructor_candidates or None,
            ambient_type_names=type_names,
            use_targets_only=is_entry and _entry_use_targets_only,
        )
        resolved_modules[mid] = ResolvedModule(
            module_id=mid,
            resolved=resolved,
            import_env=import_envs[mid],
            exports=export_maps[mid],
            scope_exports=scope_export_maps[mid],
            source_text=graph.modules[mid].source_text,
        )

    return ResolvedProgram(
        modules=resolved_modules,
        entry_id=graph.entry_id,
        all_public_funcs=all_public_funcs,
        all_public_types=all_public_types,
        graph=graph,
    )
