"""AgL module-graph loader.

This module provides :func:`load_graph`, which drives the full load-and-graph
phase of the AgL module system:

1. Parse the entry source (inline ``-c`` or a file on disk).
2. Extract top-level import/export declarations.
3. BFS over transitive import and export declarations, resolving each module id
   to its canonical file via :func:`~agm.agl.modules.resolver.resolve_module` (or
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
from agm.agl.modules.ids import ENTRY_ID, STD_CORE_ID, ModuleId
from agm.agl.modules.resolver import expand_wildcard, resolve_module
from agm.agl.modules.roots import RootSet
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.syntax.nodes import ExportDecl, ImportDecl
from agm.agl.syntax.spans import SourceId, SourceSpan
from agm.agl.syntax.types import ImportMode
from agm.core import fs
from agm.util.graph import sccs as _compute_sccs
from agm.util.text import normalize_newlines


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
    export_decls:
        Top-level :class:`~agm.agl.syntax.nodes.ExportDecl` nodes extracted
        from ``program.body.items``.
    """

    module_id: ModuleId
    program: syntax.Program
    path: Path | None
    source: SourceId
    imports: tuple[ImportDecl, ...]
    export_decls: tuple[ExportDecl, ...]
    source_text: str


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


def _extract_exports(program: syntax.Program) -> tuple[ExportDecl, ...]:
    """Return the top-level ExportDecl nodes from *program*.

    Only nodes at the top level of ``program.body.items`` are included;
    exports inside nested blocks are not valid and are ignored here (the scope
    pass enforces the restriction).
    """
    return tuple(
        item for item in program.body.items if isinstance(item, ExportDecl)
    )


def _synthetic_stdlib_import(node_id: int) -> ImportDecl:
    span = SourceSpan(
        start_line=0,
        start_col=0,
        end_line=0,
        end_col=0,
        start_offset=0,
        end_offset=0,
        source=SourceId(label="<stdlib-import>"),
    )
    return ImportDecl(
        module_path=STD_CORE_ID.segments,
        wildcard=False,
        qualified=False,
        alias=None,
        mode=ImportMode.ALL,
        items=(),
        span=span,
        node_id=node_id,
    )


def _with_default_stdlib_import(
    program: syntax.Program,
    *,
    import_node_id: int,
) -> syntax.Program:
    std_import = _synthetic_stdlib_import(import_node_id)
    body = syntax.Block(
        items=(std_import, *program.body.items),
        span=program.body.span,
        node_id=program.body.node_id,
    )
    return syntax.Program(body=body, span=program.span, node_id=program.node_id)


# ---------------------------------------------------------------------------
# Tarjan's SCC algorithm
# ---------------------------------------------------------------------------


def _mid_sort_key(mid: ModuleId) -> tuple[str, ...]:
    """Key function for sorting :class:`ModuleId` values by segments."""
    return mid.segments


_ModuleDependencyDecl = ImportDecl | ExportDecl


def _pair_sort_key(pair: tuple[ModuleId, _ModuleDependencyDecl]) -> tuple[str, ...]:
    """Key function for sorting ``(ModuleId, decl)`` pairs by module id."""
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
    return _compute_sccs(graph, key=_mid_sort_key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_into_graph(
    entry_loaded: LoadedModule,
    *,
    roots: RootSet,
    canonical_entry_path: Path | None,
    seed_modules: dict[ModuleId, LoadedModule],
    start_id: int,
    default_stdlib: bool,
) -> tuple[ModuleGraph, int, dict[ModuleId, LoadedModule]]:
    """BFS the transitive module graph from *entry_loaded*.

    Shared core of :func:`load_graph` and :func:`build_repl_graph`.  *seed_modules*
    are already-loaded library modules reused without re-parsing (empty for a
    fresh whole-program load; the REPL cache otherwise).  Newly-discovered
    modules are parsed with node ids seeded from *start_id* so ids stay disjoint
    across the graph.

    The import-graph adjacency list is captured during traversal: every wildcard
    is expanded exactly once (feeding both the BFS queue and the adjacency list),
    so no module is re-resolved when SCCs are computed.

    Returns the assembled :class:`ModuleGraph`, the next free node id, and the
    dict of modules loaded during this call (those not in *seed_modules*).
    """
    modules: dict[ModuleId, LoadedModule] = dict(seed_modules)
    modules[ENTRY_ID] = entry_loaded
    newly_loaded: dict[ModuleId, LoadedModule] = {}
    adj: dict[ModuleId, list[ModuleId]] = {}
    next_id = start_id

    # BFS queue: (module id, the import/export decl that discovered it).  We sort each
    # batch of newly-discovered ids before enqueuing so the traversal order —
    # and therefore the start_id seed assignments — are stable regardless of
    # dict/set ordering.
    queue: deque[tuple[ModuleId, _ModuleDependencyDecl]] = deque()

    def _resolve_dependencies(
        source: ModuleId,
        decls: tuple[_ModuleDependencyDecl, ...],
    ) -> None:
        """Record *source*'s module dependencies in ``adj`` and enqueue new ones."""
        targets: list[ModuleId] = []
        new_pairs: list[tuple[ModuleId, _ModuleDependencyDecl]] = []
        for decl in decls:
            if decl.wildcard:
                matched = expand_wildcard(tuple(decl.module_path), roots, span=decl.span)
                target_ids: list[ModuleId] = list(matched)
            else:
                target_ids = [ModuleId(segments=tuple(decl.module_path))]
            for mid in target_ids:
                targets.append(mid)
                if mid not in modules:
                    new_pairs.append((mid, decl))
        adj[source] = targets
        new_pairs.sort(key=_pair_sort_key)
        queue.extend(new_pairs)

    _resolve_dependencies(ENTRY_ID, (*entry_loaded.imports, *entry_loaded.export_decls))

    while queue:
        mid, decl = queue.popleft()

        # Already loaded (cycle, shared dep, or cached) — terminate this branch.
        if mid in modules:
            continue

        canon_path = resolve_module(mid, roots, span=decl.span)

        # D9: reject any import that resolves to the entry file.
        if canonical_entry_path is not None and canon_path == canonical_entry_path:
            raise ImportEntryError(mid, canonical_entry_path, span=decl.span)

        file_source_id = SourceId(label=str(canon_path))
        source_text = normalize_newlines(fs.read_text(canon_path))
        program, next_id = parse_program_seeded(
            source_text,
            start_id=next_id,
            source=file_source_id,
        )
        if default_stdlib:
            program = _with_default_stdlib_import(program, import_node_id=next_id)
            next_id += 1
        loaded = LoadedModule(
            module_id=mid,
            program=program,
            path=canon_path,
            source=file_source_id,
            imports=_extract_imports(program),
            export_decls=_extract_exports(program),
            source_text=source_text,
        )
        modules[mid] = loaded
        newly_loaded[mid] = loaded
        _resolve_dependencies(mid, (*loaded.imports, *loaded.export_decls))

    # Seeded (cached) modules are reused as-is and were never re-walked above;
    # record their adjacency so SCCs cover the whole graph.  Their import targets
    # were all loaded when they were first discovered, so nothing new is queued.
    for mid, loaded in modules.items():
        if mid not in adj:
            _resolve_dependencies(mid, (*loaded.imports, *loaded.export_decls))

    sccs = _tarjan_sccs(adj)
    graph = ModuleGraph(modules=modules, entry_id=ENTRY_ID, sccs=sccs)
    return graph, next_id, newly_loaded


def load_graph(
    entry_source: str,
    *,
    entry_path: Path | None,
    roots: RootSet,
    default_stdlib: bool = True,
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
    label = str(canonical_entry_path) if canonical_entry_path is not None else "<command>"
    entry_source_id = SourceId(label=label)

    entry_program, next_id = parse_program_seeded(
        entry_source,
        start_id=0,
        source=entry_source_id,
    )
    if default_stdlib:
        entry_program = _with_default_stdlib_import(
            entry_program,
            import_node_id=next_id,
        )
        next_id += 1
    entry_loaded = LoadedModule(
        module_id=ENTRY_ID,
        program=entry_program,
        path=canonical_entry_path,
        source=entry_source_id,
        imports=_extract_imports(entry_program),
        export_decls=_extract_exports(entry_program),
        source_text=normalize_newlines(entry_source),
    )

    graph, _next_id, _newly_loaded = _load_into_graph(
        entry_loaded,
        roots=roots,
        canonical_entry_path=canonical_entry_path,
        seed_modules={},
        start_id=next_id,
        default_stdlib=default_stdlib,
    )
    return graph


def build_repl_graph(
    program: syntax.Program,
    next_start_id: int,
    *,
    path: Path | None,
    cached: dict[ModuleId, LoadedModule],
    roots: RootSet,
) -> tuple[ModuleGraph, int, dict[ModuleId, LoadedModule]]:
    """Build a module graph from an already-parsed entry program.

    Unlike :func:`load_graph`, this function accepts an already-parsed
    ``Program`` AST (from the REPL's per-entry parse) and performs BFS loading
    only for library modules that are not already cached.  Node ids in
    newly-loaded modules are seeded from *next_start_id* so they remain
    disjoint from the entry and from any previously loaded modules.

    Parameters
    ----------
    program:
        The already-parsed entry ``Program`` AST.
    next_start_id:
        The next node id to use for newly-loaded library modules.
    path:
        Canonical file path of the entry, or ``None`` for inline/REPL.
    cached:
        Already-loaded library modules from prior REPL entries (by module id).
        These are reused without re-parsing.
    roots:
        The assembled :class:`~agm.agl.modules.roots.RootSet` to search.

    Returns
    -------
    tuple[ModuleGraph, int, dict[ModuleId, LoadedModule]]
        - The full :class:`ModuleGraph` (entry + all library modules).
        - The updated ``next_start_id`` after loading any new modules.
        - A dict of newly-loaded modules (not in *cached*) for promotion.
    """
    canonical_entry_path: Path | None = path.resolve() if path is not None else None
    label = str(canonical_entry_path) if canonical_entry_path is not None else "<repl>"
    entry_source_id = SourceId(label=label)

    seed_modules = dict(cached)
    program = _with_default_stdlib_import(program, import_node_id=next_start_id)
    next_start_id += 1

    entry_loaded = LoadedModule(
        module_id=ENTRY_ID,
        program=program,
        path=canonical_entry_path,
        source=entry_source_id,
        imports=_extract_imports(program),
        export_decls=_extract_exports(program),
        source_text="",
    )

    return _load_into_graph(
        entry_loaded,
        roots=roots,
        canonical_entry_path=canonical_entry_path,
        seed_modules=seed_modules,
        start_id=next_start_id,
        default_stdlib=True,
    )
