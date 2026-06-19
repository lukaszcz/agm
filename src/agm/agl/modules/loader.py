"""AgL module-graph loader.

This module provides :func:`load_graph`, which drives the full load-and-graph
phase of the AgL module system:

1. Parse the entry source (inline ``-c`` or a file on disk).
2. Extract top-level :class:`~agm.agl.syntax.nodes.ImportDecl` nodes.
3. BFS over transitive imports, resolving each module id to its canonical file
   via :func:`~agm.agl.modules.resolver.resolve_module` (or
   :func:`~agm.agl.modules.resolver.expand_wildcard` for ``.*`` imports),
   parsing each file with a monotonically growing ``start_id`` seed so that
   **node ids are disjoint across all modules in the graph**.
4. Terminate traversal when a module id is already loaded — this makes cycles
   finite and safe (D8).
5. Reject any import whose canonical file identity equals the entry file (D9).
6. Compute Strongly-Connected Components (SCCs) via Tarjan's algorithm for
   diagnostics.

The result is a :class:`ModuleGraph` keyed by :data:`~agm.agl.modules.ids.ENTRY_ID`
for the entry plus a :class:`~agm.agl.modules.ids.ModuleId` per library module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import agm.agl.syntax as syntax
from agm.agl.modules.errors import ImportEntryError
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.resolver import expand_wildcard, resolve_module
from agm.agl.modules.roots import RootSet
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.syntax.nodes import ImportDecl
from agm.agl.syntax.spans import SourceId


@dataclass(frozen=True, slots=True)
class LoadedModule:
    """A parsed AgL module and its metadata.

    Attributes
    ----------
    module_id:
        The logical identifier of this module.  For the entry program this is
        :data:`~agm.agl.modules.ids.ENTRY_ID`.
    program:
        The ``Program`` AST produced by parsing this module's source text.
    path:
        Canonical absolute file path.  ``None`` for an inline/``-c`` entry.
    source:
        The :class:`~agm.agl.syntax.spans.SourceId` stamped on every span in
        ``program``.
    imports:
        Top-level :class:`~agm.agl.syntax.nodes.ImportDecl` nodes extracted
        from ``program.body.items``.
    """

    module_id: ModuleId
    program: syntax.Program
    path: Path | None
    source: SourceId
    imports: tuple[ImportDecl, ...]


@dataclass(frozen=True, slots=True)
class ModuleGraph:
    """The fully-loaded module graph for an AgL program.

    Attributes
    ----------
    modules:
        ``{ModuleId: LoadedModule}`` for every reachable module (entry +
        library imports).  The entry is keyed by
        :data:`~agm.agl.modules.ids.ENTRY_ID`.
    entry_id:
        Always :data:`~agm.agl.modules.ids.ENTRY_ID`.
    sccs:
        Strongly-connected components of the *import graph*, computed by
        Tarjan's algorithm.  Each SCC is a tuple of :class:`ModuleId` values;
        the outer tuple is in **reverse topological order** (a module whose
        imports have no back-edges is last).  Retained for diagnostics; has no
        semantic effect on loading.
    """

    modules: dict[ModuleId, LoadedModule]
    entry_id: ModuleId
    sccs: tuple[tuple[ModuleId, ...], ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_imports(program: syntax.Program) -> tuple[ImportDecl, ...]:
    """Return the top-level ImportDecl nodes from *program*.

    Only nodes at the top level of ``program.body.items`` are included;
    imports inside nested blocks are not valid (D4) and are ignored here
    (the scope pass enforces the restriction).
    """
    return tuple(
        item for item in program.body.items if isinstance(item, ImportDecl)
    )


# ---------------------------------------------------------------------------
# Tarjan's SCC algorithm
# ---------------------------------------------------------------------------


def _mid_sort_key(mid: ModuleId) -> tuple[str, ...]:
    """Key function for sorting :class:`ModuleId` values by segments."""
    return mid.segments


def _pair_sort_key(pair: tuple[ModuleId, ImportDecl]) -> tuple[str, ...]:
    """Key function for sorting ``(ModuleId, ImportDecl)`` pairs by module id."""
    return pair[0].segments


def _tarjan_sccs(
    graph: dict[ModuleId, list[ModuleId]],
) -> tuple[tuple[ModuleId, ...], ...]:
    """Compute SCCs of *graph* using Tarjan's algorithm.

    Parameters
    ----------
    graph:
        Adjacency list mapping each :class:`ModuleId` to its direct
        dependencies (import targets that are in the loaded set).

    Returns
    -------
    tuple[tuple[ModuleId, ...], ...]
        SCCs in **reverse topological order** (sinks first, roots last).
    """
    index_counter = [0]
    stack: list[ModuleId] = []
    on_stack: set[ModuleId] = set()
    index: dict[ModuleId, int] = {}
    lowlink: dict[ModuleId, int] = {}
    sccs: list[tuple[ModuleId, ...]] = []

    def strongconnect(node: ModuleId) -> None:
        idx = index_counter[0]
        index[node] = idx
        lowlink[node] = idx
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)

        for neighbour in graph.get(node, []):
            if neighbour not in index:
                strongconnect(neighbour)
                lowlink[node] = min(lowlink[node], lowlink[neighbour])
            elif neighbour in on_stack:
                lowlink[node] = min(lowlink[node], index[neighbour])

        if lowlink[node] == index[node]:
            scc: list[ModuleId] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == node:
                    break
            scc.sort(key=_mid_sort_key)
            sccs.append(tuple(scc))

    nodes = sorted(graph.keys(), key=_mid_sort_key)
    for node in nodes:
        if node not in index:
            strongconnect(node)

    return tuple(sccs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_graph(
    entry_source: str,
    *,
    entry_path: Path | None,
    roots: RootSet,
) -> ModuleGraph:
    """Parse and load the full transitive module graph.

    Parameters
    ----------
    entry_source:
        The AgL source text of the entry program.
    entry_path:
        Canonical file path of the entry program, or ``None`` for an inline
        ``-c`` invocation.  When supplied, its canonical form is used to
        detect and reject any import that resolves to the same file (D9).
    roots:
        The assembled :class:`~agm.agl.modules.roots.RootSet` to search.

    Returns
    -------
    ModuleGraph
        The fully loaded module graph.

    Raises
    ------
    ModuleNotFound
        When a non-wildcard import cannot be resolved.
    AmbiguousModule
        When a module id (or a wildcard-expanded id) resolves to ≥2 distinct
        canonical files.
    ModulePrefixNotFound
        When a wildcard import prefix matches no module.
    ImportEntryError
        When an import resolves to the entry file's canonical identity (D9).
    agm.agl.parser.errors.AglSyntaxError
        When any module's source text fails to parse.
    """
    # Canonical entry path (for rejection checks in D9).
    canonical_entry_path: Path | None = (
        entry_path.resolve() if entry_path is not None else None
    )

    # Entry source identity.
    if canonical_entry_path is not None:
        entry_source_id = SourceId(label=str(canonical_entry_path))
    else:
        entry_source_id = SourceId(label="<command>")

    # Parse the entry.
    entry_program, next_id = parse_program_seeded(
        entry_source,
        start_id=0,
        source=entry_source_id,
    )
    entry_imports = _extract_imports(entry_program)

    entry_loaded = LoadedModule(
        module_id=ENTRY_ID,
        program=entry_program,
        path=canonical_entry_path,
        source=entry_source_id,
        imports=entry_imports,
    )

    # Accumulated modules: module_id → LoadedModule.
    modules: dict[ModuleId, LoadedModule] = {ENTRY_ID: entry_loaded}

    # BFS queue: module ids to visit next.
    # We use a deterministic BFS by always sorting the discovered ids before
    # enqueuing; this ensures the traversal order — and therefore the
    # start_id seed assignments — are stable regardless of dict/set ordering.
    queue: deque[tuple[ModuleId, ImportDecl]] = deque()

    def _enqueue_imports(imports: tuple[ImportDecl, ...]) -> None:
        """Resolve and enqueue all new module ids discovered from *imports*."""
        # Collect new (module_id, decl) pairs in sorted order for determinism.
        new_pairs: list[tuple[ModuleId, ImportDecl]] = []

        for decl in imports:
            if decl.wildcard:
                # Expand foo.* → dict[ModuleId, Path]
                matched = expand_wildcard(
                    tuple(decl.module_path), roots, span=decl.span
                )
                for mid in matched:
                    if mid not in modules:
                        new_pairs.append((mid, decl))
            else:
                mid = ModuleId(segments=tuple(decl.module_path))
                if mid not in modules:
                    new_pairs.append((mid, decl))

        # Sort for deterministic BFS order.
        new_pairs.sort(key=_pair_sort_key)
        queue.extend(new_pairs)

    _enqueue_imports(entry_imports)

    while queue:
        mid, decl = queue.popleft()

        # Already loaded (cycle or shared dep) — terminate this branch.
        if mid in modules:
            continue

        # Resolve the id to its canonical file path.
        canon_path = resolve_module(mid, roots, span=decl.span)

        # D9: reject any import that resolves to the entry file.
        if canonical_entry_path is not None and canon_path == canonical_entry_path:
            raise ImportEntryError(mid, canonical_entry_path, span=decl.span)

        # Read and parse the file.
        source_text = canon_path.read_text(encoding="utf-8")
        file_source_id = SourceId(label=str(canon_path))
        program, next_id = parse_program_seeded(
            source_text,
            start_id=next_id,
            source=file_source_id,
        )
        imports = _extract_imports(program)

        loaded = LoadedModule(
            module_id=mid,
            program=program,
            path=canon_path,
            source=file_source_id,
            imports=imports,
        )
        modules[mid] = loaded

        _enqueue_imports(imports)

    # Build the import-graph adjacency list for SCC computation.
    # Every import target was resolved and loaded during the BFS above, so all
    # target ids are guaranteed to be present in ``modules``.
    adj: dict[ModuleId, list[ModuleId]] = {mid: [] for mid in modules}
    for mid, loaded in modules.items():
        for decl in loaded.imports:
            if decl.wildcard:
                # Wildcard: re-expand to get the set of resolved ids.
                # All matched ids are in ``modules`` (loaded during BFS).
                matched = expand_wildcard(tuple(decl.module_path), roots)
                for target_id in matched:
                    adj[mid].append(target_id)
            else:
                target_id = ModuleId(segments=tuple(decl.module_path))
                adj[mid].append(target_id)

    sccs = _tarjan_sccs(adj)

    return ModuleGraph(
        modules=modules,
        entry_id=ENTRY_ID,
        sccs=sccs,
    )
