"""AgL module-graph loader.

This module provides :func:`load_graph`, which drives the full load-and-graph
phase of the AgL module system:

1. Parse the entry source (inline ``-c`` or a file on disk).
2. Extract import, use, and export declarations from the module and its named scope regions.
3. BFS over transitive import and export declarations, resolving each module id
   to its canonical file via :func:`~agm.agl.modules.resolver.resolve_module` (or
   :func:`~agm.agl.modules.resolver.expand_wildcard` for ``/*`` imports),
   parsing each file with a monotonically growing ``start_id`` seed so that
   **node ids are disjoint across all modules in the graph**.
4. Terminate traversal when a module id is already loaded — this makes cycles
   finite and safe.
5. Reject any import whose canonical file identity equals the entry file.
6. Compute Strongly-Connected Components (SCCs) via Tarjan's algorithm for
   diagnostics.

The result is a :class:`ModuleGraph` keyed by :data:`~agm.agl.modules.ids.ENTRY_ID`
for the entry plus a :class:`~agm.agl.modules.ids.ModuleId` per library module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import agm.agl.syntax as syntax
from agm.agl.lexer import spaced_qualifier_collector
from agm.agl.modules.errors import (
    ImportEntryError,
    MissingExternCompanion,
    PackageImportVisibilityError,
)
from agm.agl.modules.ids import ENTRY_ID, STD_CORE_ID, ModuleId
from agm.agl.modules.resolver import expand_wildcard, resolve_module
from agm.agl.modules.roots import RootSet
from agm.agl.parser import AglSyntaxError
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.syntax.advisories import SpacedQualifier
from agm.agl.syntax.nodes import ExportDecl, FuncDef, ImportDecl, UseDecl, static_items
from agm.agl.syntax.spans import SourceId, SourceSpan
from agm.core import fs
from agm.packages.model import owning_package
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
        :class:`~agm.agl.syntax.nodes.ImportDecl` nodes extracted from the
        module root and named scope regions.
    uses:
        :class:`~agm.agl.syntax.nodes.UseDecl` nodes extracted from the module
        root and named scope regions. They do not create module-graph edges.
    export_decls:
        :class:`~agm.agl.syntax.nodes.ExportDecl` nodes extracted from the
        module root and named scope regions.
    spaced_qualifiers:
        Lexical advisories for qualifier runs this module's source separated
        from their ``::`` by whitespace — see
        :class:`~agm.agl.syntax.advisories.SpacedQualifier`.
    companion_path:
        The canonical Python companion file path (this module's path with its
        suffix replaced by ``.py``) when ``program`` declares at least one
        ``extern def``; ``None`` otherwise. Always ``None`` for an
        inline/REPL module (``path is None``) since such a module can never
        declare an extern (the scope pass rejects it for lack of a backing
        file).  Verified to exist at load time — see
        :class:`~agm.agl.modules.errors.MissingExternCompanion`.
    """

    module_id: ModuleId
    program: syntax.Program
    path: Path | None
    source: SourceId
    imports: tuple[ImportDecl, ...]
    uses: tuple[UseDecl, ...]
    export_decls: tuple[ExportDecl, ...]
    source_text: str
    spaced_qualifiers: tuple[SpacedQualifier, ...]
    companion_path: Path | None


class EntryParseSyntaxError(AglSyntaxError):
    """An entry parse failure retaining lexical advisories emitted before it."""

    def __init__(
        self, error: AglSyntaxError, spaced_qualifiers: tuple[SpacedQualifier, ...]
    ) -> None:
        super().__init__(str(error), span=error.source_span)
        self.spaced_qualifiers = spaced_qualifiers


@dataclass(frozen=True, slots=True)
class ParsedEntryModule:
    """The parsed entry and metadata shared by entry-loading paths."""

    program: syntax.Program
    next_id: int
    canonical_path: Path | None
    source_id: SourceId
    spaced_qualifiers: tuple[SpacedQualifier, ...]


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
        Strongly-connected components of the dependency graph, computed by
        Tarjan's algorithm. Each SCC is a tuple of :class:`ModuleId` values;
        the outer tuple is in **reverse topological order**.
    adjacency:
        Direct dependency edges for every loaded module. This includes imports
        and exports after wildcard expansion; uses do not create edges. It is
        the authoritative reachability relation for graph consumers.
    """

    modules: dict[ModuleId, LoadedModule]
    entry_id: ModuleId
    sccs: tuple[tuple[ModuleId, ...], ...]
    adjacency: dict[ModuleId, tuple[ModuleId, ...]]
    roots: RootSet = field(default_factory=lambda: RootSet(roots=frozenset()))

    def resource_root_for(self, module_id: ModuleId) -> Path | None:
        """Return the filesystem anchor used by a module's resource calls."""
        path = self.modules[module_id].path
        if path is None:
            return None
        package = owning_package(path, self.roots.packages)
        if package is not None:
            return package.root
        return next(
            (root for root in self.roots.stdlib_roots if path.resolve().is_relative_to(root)),
            path.parent,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_imports(
    program: syntax.Program,
) -> tuple[tuple[ImportDecl, ...], tuple[UseDecl, ...]]:
    """Return a module's imports and uses, including region-nested declarations.

    Named scope regions are transparent to this walk. Imports create
    module-graph edges; uses are retained for consumers that resolve their
    targets against those imports. Declarations inside ordinary nested blocks
    are not valid and are ignored here (the scope pass enforces the
    restriction).
    """
    declarations = tuple(static_items(program.body.items))
    return (
        tuple(item for item in declarations if isinstance(item, ImportDecl)),
        tuple(item for item in declarations if isinstance(item, UseDecl)),
    )


def _extract_exports(program: syntax.Program) -> tuple[ExportDecl, ...]:
    """Return the module's ExportDecl nodes, including region-nested ones.

    Named scope regions are transparent to this walk, so a scoped ``export``
    is discovered exactly like a root one. Exports inside an ordinary nested
    block are not valid and are ignored here (the scope pass enforces the
    restriction).
    """
    return tuple(item for item in static_items(program.body.items) if isinstance(item, ExportDecl))


def _companion_path_for(
    module_id: ModuleId, program: syntax.Program, path: Path | None
) -> Path | None:
    """Derive and verify *module_id*'s Python companion path, if it needs one.

    A module needs a companion iff it declares at least one ``extern def``.
    Scoped externs select their unqualified member names from that same
    companion. Returns ``None`` for modules with no extern declarations and
    for inline/REPL modules (``path is None``) — the scope pass rejects
    ``extern def`` in a module with no backing file, so such a module never
    needs a companion.

    :raises MissingExternCompanion: when the module declares an extern but the
        derived ``.py`` sibling file does not exist.
    """
    if path is None:
        return None
    externs: list[FuncDef] = []

    def collect_extern(node: object) -> None:
        if isinstance(node, FuncDef) and node.is_extern:
            externs.append(node)

    syntax.walk(program, collect_extern)
    if not externs:
        return None
    companion_path = path.with_suffix(".py")
    if not fs.is_file(companion_path):
        raise MissingExternCompanion(module_id, companion_path, span=externs[0].span)
    return companion_path


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
        alias=None,
        tail=(),
        hidden=(),
        span=span,
        node_id=node_id,
    )


def _with_default_stdlib_import(
    program: syntax.Program,
    *,
    import_node_id: int,
) -> syntax.Program:
    imports, _uses = _extract_imports(program)
    if any(
        decl.module_path == STD_CORE_ID.segments
        or (decl.wildcard and STD_CORE_ID.segments[: len(decl.module_path)] == decl.module_path)
        for decl in imports
    ):
        return program
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


_ResolvedDependency = tuple[ModuleId, _ModuleDependencyDecl, Path]


def _pair_sort_key(pair: _ResolvedDependency) -> tuple[str, ...]:
    """Key function for sorting resolved dependency triples by module id."""
    return pair[0].segments


def _check_package_import_visibility(
    source_path: Path | None,
    target_path: Path,
    target_id: ModuleId,
    *,
    roots: RootSet,
    span: SourceSpan,
) -> None:
    """Reject package-to-package edges missing a manifest dependency.

    Canonical paths establish ownership, keeping ad-hoc entry and loose-root
    modules open even when they import mounted packages. A dependency key is
    not enough: the resolved target must actually be owned by its mounted
    package, so a loose module cannot borrow package visibility by sharing a
    declared dependency's leading path segment.
    """
    if source_path is None:
        return
    source_package = owning_package(source_path, roots.packages)
    if source_package is None:
        return
    if roots.is_standard_library_path(target_path):
        return
    target_package = owning_package(target_path, roots.packages)
    if target_package == source_package and target_id.segments[0] == source_package.manifest.name:
        return
    if target_package is None or target_package == source_package:
        raise PackageImportVisibilityError(
            source_package.manifest.name, target_id.segments[0], span=span
        )
    target_name = target_package.manifest.name
    if target_id.segments[0] == target_name and target_name in source_package.manifest.dependencies:
        return
    raise PackageImportVisibilityError(source_package.manifest.name, target_name, span=span)


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

    # BFS queue: (module id, dependency decl, canonical target path). We sort each
    # batch of newly-discovered ids before enqueuing so the traversal order —
    # and therefore the start_id seed assignments — are stable regardless of
    # dict/set ordering.
    queue: deque[_ResolvedDependency] = deque()

    def _resolve_dependencies(
        source: ModuleId,
        decls: tuple[_ModuleDependencyDecl, ...],
    ) -> None:
        """Record *source*'s module dependencies in ``adj`` and enqueue new ones."""
        targets: list[ModuleId] = []
        new_pairs: list[_ResolvedDependency] = []
        source_path = modules[source].path
        for decl in decls:
            if decl.wildcard:
                targets_by_id = expand_wildcard(tuple(decl.module_path), roots, span=decl.span)
            else:
                target_id = ModuleId(segments=tuple(decl.module_path))
                targets_by_id = {target_id: resolve_module(target_id, roots, span=decl.span)}
            for mid, target_path in targets_by_id.items():
                _check_package_import_visibility(
                    source_path, target_path, mid, roots=roots, span=decl.span
                )
                targets.append(mid)
                if mid not in modules:
                    new_pairs.append((mid, decl, target_path))
        adj[source] = targets
        new_pairs.sort(key=_pair_sort_key)
        queue.extend(new_pairs)

    _resolve_dependencies(ENTRY_ID, (*entry_loaded.imports, *entry_loaded.export_decls))

    while queue:
        mid, decl, canon_path = queue.popleft()

        # Already loaded (cycle, shared dep, or cached) — terminate this branch.
        if mid in modules:
            continue

        # Reject any import that resolves to the entry file.
        if canonical_entry_path is not None and canon_path == canonical_entry_path:
            raise ImportEntryError(mid, canonical_entry_path, span=decl.span)

        file_source_id = SourceId(label=str(canon_path))
        source_text = normalize_newlines(fs.read_text(canon_path))
        with spaced_qualifier_collector() as spaced_sink:
            program, next_id = parse_program_seeded(
                source_text,
                start_id=next_id,
                source=file_source_id,
            )
        if default_stdlib and mid != STD_CORE_ID:
            program = _with_default_stdlib_import(program, import_node_id=next_id)
            next_id += 1
        imports, uses = _extract_imports(program)
        loaded = LoadedModule(
            module_id=mid,
            program=program,
            path=canon_path,
            source=file_source_id,
            imports=imports,
            uses=uses,
            export_decls=_extract_exports(program),
            source_text=source_text,
            spaced_qualifiers=tuple(spaced_sink),
            companion_path=_companion_path_for(mid, program, canon_path),
        )
        modules[mid] = loaded
        newly_loaded[mid] = loaded
        _resolve_dependencies(mid, (*loaded.imports, *loaded.export_decls))

    # Seeded (cached) modules are reused as-is and were never re-walked above;
    # record their adjacency so SCCs cover the whole program.  Their import targets
    # were all loaded when they were first discovered, so nothing new is queued.
    for mid, loaded in modules.items():
        if mid not in adj:
            _resolve_dependencies(mid, (*loaded.imports, *loaded.export_decls))

    sccs = _tarjan_sccs(adj)
    graph = ModuleGraph(
        modules=modules,
        entry_id=ENTRY_ID,
        sccs=sccs,
        adjacency={mid: tuple(targets) for mid, targets in adj.items()},
        roots=roots,
    )
    return graph, next_id, newly_loaded


def entry_source_id(
    entry_path: Path | None,
    *,
    default_label: str = "<command>",
) -> tuple[Path | None, SourceId]:
    """Return an entry's canonical path and the source id its label implies.

    A source without a backing file uses *default_label*; every entry-loading
    path derives its diagnostic identity here.
    """
    canonical_path = entry_path.resolve() if entry_path is not None else None
    label = str(canonical_path) if canonical_path is not None else default_label
    return canonical_path, SourceId(label=label)


def parse_entry_module(
    entry_source: str,
    *,
    entry_path: Path | None,
) -> ParsedEntryModule:
    """Parse an entry source and collect its lexical advisories.

    Syntax failures intentionally propagate to let callers choose their own
    raising or diagnostic-capturing policy.
    """
    canonical_path, source_id = entry_source_id(entry_path)
    with spaced_qualifier_collector() as spaced_sink:
        try:
            program, next_id = parse_program_seeded(entry_source, start_id=0, source=source_id)
        except AglSyntaxError as error:
            raise EntryParseSyntaxError(error, tuple(spaced_sink)) from error
    return ParsedEntryModule(
        program=program,
        next_id=next_id,
        canonical_path=canonical_path,
        source_id=source_id,
        spaced_qualifiers=tuple(spaced_sink),
    )


def _build_entry_loaded_module(
    program: syntax.Program,
    next_id: int,
    *,
    canonical_entry_path: Path | None,
    entry_source_id: SourceId,
    default_stdlib: bool,
    spaced_qualifiers: tuple[SpacedQualifier, ...],
    source_text: str,
) -> tuple[LoadedModule, int]:
    """Build the entry :class:`LoadedModule` from an already-parsed program.

    Shared by :func:`load_graph` (parses the entry itself first) and
    :func:`build_repl_graph` (given an already-parsed entry from the REPL's
    own per-entry parse). Injects the ``std/core`` prelude import when
    *default_stdlib* is set, consuming one more node id, and derives the
    companion path for a declared extern. Returns the built module together
    with the next free node id.
    """
    if default_stdlib:
        program = _with_default_stdlib_import(program, import_node_id=next_id)
        next_id += 1
    imports, uses = _extract_imports(program)
    entry_loaded = LoadedModule(
        module_id=ENTRY_ID,
        program=program,
        path=canonical_entry_path,
        source=entry_source_id,
        imports=imports,
        uses=uses,
        export_decls=_extract_exports(program),
        source_text=source_text,
        spaced_qualifiers=spaced_qualifiers,
        companion_path=_companion_path_for(ENTRY_ID, program, canonical_entry_path),
    )
    return entry_loaded, next_id


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
        detect and reject any import that resolves to the same file.
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
        When an import resolves to the entry file's canonical identity.
    agm.agl.parser.errors.AglSyntaxError
        When any module's source text fails to parse.
    """
    parsed_entry = parse_entry_module(entry_source, entry_path=entry_path)
    entry_loaded, next_id = _build_entry_loaded_module(
        parsed_entry.program,
        parsed_entry.next_id,
        canonical_entry_path=parsed_entry.canonical_path,
        entry_source_id=parsed_entry.source_id,
        default_stdlib=default_stdlib,
        spaced_qualifiers=parsed_entry.spaced_qualifiers,
        source_text=normalize_newlines(entry_source),
    )

    graph, _next_id, _newly_loaded = _load_into_graph(
        entry_loaded,
        roots=roots,
        canonical_entry_path=parsed_entry.canonical_path,
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
    default_stdlib: bool = True,
    spaced_qualifiers: tuple[SpacedQualifier, ...] = (),
    default_label: str = "<repl>",
    source_text: str = "",
) -> tuple[ModuleGraph, int, dict[ModuleId, LoadedModule]]:
    """Build a module graph from an already-parsed entry program.

    Unlike :func:`load_graph`, this function accepts an already-parsed
    ``Program`` AST (from the REPL's per-entry parse, or from a host-side
    pipeline step that needs the parsed entry before it loads the rest of the
    graph) and performs BFS loading only for library modules that are not
    already cached. Node ids in newly-loaded modules are seeded from
    *next_start_id* so they remain disjoint from the entry and from any
    previously loaded modules.

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
    default_stdlib:
        Whether to inject the standard-library prelude into the entry and
        newly loaded library modules.
    spaced_qualifiers:
        Spaced-qualifier advisories collected while *program* was lexed.
    default_label:
        The entry source label used when *path* is ``None`` (inline entry).
        Defaults to ``"<repl>"`` for the REPL's own incremental sessions;
        callers outside the REPL (e.g. an ``exec -c`` style host pipeline)
        pass their own label so diagnostics match :func:`load_graph`.
    source_text:
        The entry's normalized source text, recorded on the entry
        ``LoadedModule`` for deep IR-validation span checks and runtime
        source slicing.  Defaults to ``""``, which is what the REPL wants:
        its own lowering step supplies the per-entry text directly (see
        ``agm.agl.lower.repl``), so the loader's copy is never consulted. A
        caller outside the REPL passes the real entry source here.

    Returns
    -------
    tuple[ModuleGraph, int, dict[ModuleId, LoadedModule]]
        - The full :class:`ModuleGraph` (entry + all library modules).
        - The updated ``next_start_id`` after loading any new modules.
        - A dict of newly-loaded modules (not in *cached*) for promotion.
    """
    canonical_entry_path, source_id = entry_source_id(path, default_label=default_label)

    seed_modules = dict(cached)
    entry_loaded, next_start_id = _build_entry_loaded_module(
        program,
        next_start_id,
        canonical_entry_path=canonical_entry_path,
        entry_source_id=source_id,
        default_stdlib=default_stdlib,
        spaced_qualifiers=spaced_qualifiers,
        source_text=source_text,
    )

    return _load_into_graph(
        entry_loaded,
        roots=roots,
        canonical_entry_path=canonical_entry_path,
        seed_modules=seed_modules,
        start_id=next_start_id,
        default_stdlib=default_stdlib,
    )
