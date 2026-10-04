"""AgL module-graph loader.

:func:`parse_entry_module` parses an entry source (inline ``-c`` or a file on
disk), and :func:`build_repl_graph` drives the rest of the load-and-graph phase
of the AgL module system from the parsed entry:

1. Inject the standard-library prelude import into the entry.
2. Extract import and export declarations from the module and its named scope regions.
3. BFS over transitive import and export declarations, resolving each module id
   to its canonical file via :func:`~agm.agl.modules.resolver.resolve_module` (or
   :func:`~agm.agl.modules.resolver.expand_wildcard` for ``/*`` imports),
   parsing each file with a monotonically growing ``start_id`` seed so that
   **node ids are disjoint across all modules in the graph**. A module under a
   standard-library root comes instead from the process-global cache in
   :mod:`agm.agl.modules.parsed_module_cache`, whose reserved id band keeps it
   disjoint from every graph it is served into.
4. Terminate traversal when a module id is already loaded — this makes cycles
   finite and safe.
5. Reject any import that resolves to the entry file under an id other than
   the entry's own.
6. Compute Strongly-Connected Components (SCCs) via Tarjan's algorithm for
   diagnostics.

The result is a :class:`ModuleGraph` keyed by one
:class:`~agm.agl.modules.ids.ModuleId` per module. The entry is keyed by the
module id its owning package declares for its file, and by
:data:`~agm.agl.modules.ids.ENTRY_ID` when the source has no module identity
(inline ``-c`` source, a REPL splice, or a file no package owns).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import agm.agl.syntax as syntax
from agm.agl.lexer import spaced_qualifier_collector
from agm.agl.modules.errors import (
    AmbiguousModule,
    ImportEntryError,
    MissingExternCompanion,
    ModuleNotFound,
    PackageImportVisibilityError,
)
from agm.agl.modules.ids import (
    ENTRY_ID,
    STD_PRELUDE_ID,
    ModuleId,
)
from agm.agl.modules.parsed_module_cache import cached_parsed_module
from agm.agl.modules.resolver import expand_wildcard, resolve_module
from agm.agl.modules.roots import RootSet
from agm.agl.parser import AglSyntaxError
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.parser.wrap import wrap_inline_program
from agm.agl.syntax.advisories import SpacedQualifier
from agm.agl.syntax.nodes import (
    ExportDecl,
    FuncDef,
    ImportDecl,
    static_items,
)
from agm.agl.syntax.spans import SourceId, SourceSpan
from agm.core import fs
from agm.packages.model import owning_package
from agm.util.graph import sccs as _compute_sccs


@dataclass(frozen=True, slots=True)
class LoadedModule:
    """A parsed AgL module and its metadata.

    Attributes
    ----------
    module_id:
        The logical identifier of this module.  The entry program carries the
        module id its package declares for its file, or
        :data:`~agm.agl.modules.ids.ENTRY_ID` when it has no module identity.
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
    export_decls: tuple[ExportDecl, ...]
    source_text: str
    spaced_qualifiers: tuple[SpacedQualifier, ...]
    companion_path: Path | None


class EntryParseSyntaxError(AglSyntaxError):
    """An entry parse failure retaining lexical advisories emitted before it."""

    def __init__(
        self, error: AglSyntaxError, spaced_qualifiers: tuple[SpacedQualifier, ...]
    ) -> None:
        super().__init__(str(error), span=error.span)
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
        library imports), the entry keyed by :attr:`entry_id`.
    entry_id:
        The entry module's identity: the module id its owning package declares
        for its file, or :data:`~agm.agl.modules.ids.ENTRY_ID` when the entry
        source has no module identity. Ask for the entry by this id rather
        than by the sentinel.
    sccs:
        Strongly-connected components of the dependency graph, computed by
        Tarjan's algorithm. Each SCC is a tuple of :class:`ModuleId` values;
        the outer tuple is in **reverse topological order**.
    adjacency:
        Direct dependency edges for every loaded module. This includes imports
        and exports after wildcard expansion; uses do not create edges. It is
        the authoritative reachability relation for graph consumers.
    source_adjacency:
        The subset of direct dependency edges authored in source. Loader
        injections are absent, retaining provenance for runtime inventories.
    """

    modules: dict[ModuleId, LoadedModule]
    entry_id: ModuleId
    sccs: tuple[tuple[ModuleId, ...], ...]
    adjacency: dict[ModuleId, tuple[ModuleId, ...]]
    source_adjacency: dict[ModuleId, tuple[ModuleId, ...]] = field(default_factory=dict)
    roots: RootSet = field(default_factory=lambda: RootSet(roots=frozenset()))
    # Memo for dependency_closures. The graph is frozen, so a closure computed
    # once holds for its lifetime; it takes no part in equality or repr, and
    # being init-less it is never carried over by `replace`, which is free to
    # hand back a graph with different modules or edges.
    _closures: dict[ModuleId, tuple[LoadedModule, ...]] = field(
        default_factory=dict, compare=False, repr=False, init=False
    )
    # Memo for source_reachable_modules, held under the same terms as
    # _closures. Filled one queried module at a time: a source closure is
    # ordered by the walk that produced it, and such an order cannot be
    # assembled from the closures of a module's neighbours, so entries a
    # caller never asks for are never walked.
    _source_closures: dict[ModuleId, tuple[ModuleId, ...]] = field(
        default_factory=dict, compare=False, repr=False, init=False
    )

    def dependency_closures(self) -> Mapping[ModuleId, tuple[LoadedModule, ...]]:
        """Return each loaded module's transitive dependencies, itself included.

        Closures hold the loaded modules themselves, ordered by module id so
        two graphs list the same modules alike. Computed for the whole graph at
        once on first use: :attr:`sccs` lists components dependencies-first, so
        a component's outside neighbours already have their closures when it is
        reached.
        """
        if not self._closures:
            reached: dict[ModuleId, frozenset[ModuleId]] = {
                module_id: frozenset((module_id,)) for module_id in self.modules
            }
            for component in self.sccs:
                members = frozenset(component)
                closure = members.union(
                    *(
                        reached[neighbour]
                        for member in component
                        for neighbour in self.adjacency.get(member, ())
                        if neighbour not in members
                    )
                )
                for member in component:
                    reached[member] = closure
            self._closures.update(
                (
                    module_id,
                    tuple(
                        self.modules[reached_id]
                        for reached_id in sorted(
                            reached[module_id] & self.modules.keys(), key=_mid_sort_key
                        )
                    ),
                )
                for module_id in self.modules
            )
        return self._closures

    def source_reachable_modules(self, module_id: ModuleId) -> tuple[ModuleId, ...]:
        """Return modules reachable through source-authored import/export edges.

        ``source_adjacency`` records edge provenance at load time, excluding
        loader-injected standard-library edges. The walk is memoized per
        module: the graph is frozen, so one module's closure is the same on
        every later ask, however many programs it declares.
        """
        closure = self._source_closures.get(module_id)
        if closure is not None:
            return closure
        adjacency = self.source_adjacency or self.adjacency
        reachable: list[ModuleId] = []
        seen: set[ModuleId] = set()
        pending = [module_id]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            reachable.append(current)
            pending.extend(reversed(adjacency[current]))
        closure = tuple(reachable)
        self._source_closures[module_id] = closure
        return closure

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


def _extract_imports(program: syntax.Program) -> tuple[ImportDecl, ...]:
    """Return a module's imports, including region-nested declarations.

    Named scope regions are transparent to this walk. Imports create
    module-graph edges; ``use`` declarations do not, and the scope pass reads
    them straight from the AST. Declarations inside ordinary nested blocks are
    not valid and are ignored here (the scope pass enforces the restriction).
    """
    return tuple(item for item in static_items(program.body.items) if isinstance(item, ImportDecl))


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
        module_path=STD_PRELUDE_ID.segments,
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
    imports = _extract_imports(program)
    if any(
        decl.module_path == STD_PRELUDE_ID.segments
        or (decl.wildcard and STD_PRELUDE_ID.segments[: len(decl.module_path)] == decl.module_path)
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


def load_parsed_module(
    module_id: ModuleId, path: Path, *, default_stdlib: bool = True
) -> LoadedModule:
    """Serve one module's parse from the shared cache, parsing only on a miss.

    The single entry point for a caller that needs a module's AST without
    building a graph around it — package command discovery is one. It keys the
    same cache the graph loader keys, so a module parsed here is reused there
    and the reverse, and neither parses source the other already has.
    """
    build = partial(_parse_imported_module, module_id, path, default_stdlib=default_stdlib)
    return cached_parsed_module(module_id, path, default_stdlib=default_stdlib, build=build)


def _parse_imported_module(
    module_id: ModuleId,
    path: Path,
    start_id: int,
    source_text: str,
    *,
    default_stdlib: bool,
) -> tuple[LoadedModule, int]:
    """Parse one imported module, seeding its node ids from *start_id*.

    Returns the module together with the first node id it did not consume.
    This is the whole per-module load step, so the parsed-module cache can own
    an identical module without the loader duplicating it; the cache reads the
    file and passes the text in, so the source is read exactly once.
    """
    file_source_id = SourceId(label=str(path))
    with spaced_qualifier_collector() as spaced_sink:
        program, next_id = parse_program_seeded(
            source_text,
            start_id=start_id,
            source=file_source_id,
        )
    if default_stdlib and module_id != STD_PRELUDE_ID:
        program = _with_default_stdlib_import(program, import_node_id=next_id)
        next_id += 1
    loaded = LoadedModule(
        module_id=module_id,
        program=program,
        path=path,
        source=file_source_id,
        imports=_extract_imports(program),
        export_decls=_extract_exports(program),
        source_text=source_text,
        spaced_qualifiers=tuple(spaced_sink),
        companion_path=_companion_path_for(module_id, program, path),
    )
    return loaded, next_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_into_graph(
    entry_loaded: LoadedModule,
    *,
    entry_id: ModuleId,
    roots: RootSet,
    canonical_entry_path: Path | None,
    seed_modules: dict[ModuleId, LoadedModule],
    start_id: int,
    default_stdlib: bool,
) -> tuple[ModuleGraph, int, dict[ModuleId, LoadedModule]]:
    """BFS the transitive module graph from *entry_loaded*.

    Core of :func:`build_repl_graph`.  *seed_modules*
    are already-loaded library modules reused without re-parsing (empty for a
    fresh whole-program load; the REPL cache otherwise).  Newly-discovered
    modules are parsed with node ids seeded from *start_id* so ids stay disjoint
    across the graph.

    The import-graph adjacency list is captured during traversal: every wildcard
    is expanded exactly once (feeding both the BFS queue and the adjacency list),
    so no module is re-resolved when SCCs are computed.

    Returns the assembled :class:`ModuleGraph`, the next free node id, and the
    dict of modules loaded during this call (those not in *seed_modules*).  A named
    entry is among them: it was loaded now, under a module id anything else in
    the graph may name, so a caller asking what this call made available must
    see it exactly as it would had an import reached the same file.  An
    anonymous entry is not, since no id names it.
    """
    modules: dict[ModuleId, LoadedModule] = dict(seed_modules)
    modules[entry_id] = entry_loaded
    newly_loaded: dict[ModuleId, LoadedModule] = {}
    if not entry_id.is_entry:
        newly_loaded[entry_id] = entry_loaded
    adj: dict[ModuleId, list[ModuleId]] = {}
    source_adj: dict[ModuleId, list[ModuleId]] = {}
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
        source_targets: list[ModuleId] = []
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
                if decl.span.source.label != "<stdlib-import>":
                    source_targets.append(mid)
                if mid not in modules:
                    new_pairs.append((mid, decl, target_path))
        adj[source] = targets
        source_adj[source] = source_targets
        new_pairs.sort(key=_pair_sort_key)
        queue.extend(new_pairs)

    _resolve_dependencies(entry_id, (*entry_loaded.imports, *entry_loaded.export_decls))

    # Cached modules were loaded in an earlier REPL compilation. Rebuild their
    # adjacency before draining the queue so any dependency absent from this
    # invocation's cache is loaded normally.
    for mid, loaded in modules.items():
        if mid != entry_id:
            _resolve_dependencies(mid, (*loaded.imports, *loaded.export_decls))

    while queue:
        mid, decl, canon_path = queue.popleft()

        # Already loaded (cycle, shared dep, or cached) — terminate this branch.
        if mid in modules:
            continue

        # Reject any import that resolves to the entry file.
        if canonical_entry_path is not None and canon_path == canonical_entry_path:
            raise ImportEntryError(mid, canonical_entry_path, span=decl.span)

        build = partial(_parse_imported_module, mid, canon_path, default_stdlib=default_stdlib)
        loaded = cached_parsed_module(mid, canon_path, default_stdlib=default_stdlib, build=build)
        modules[mid] = loaded
        newly_loaded[mid] = loaded
        _resolve_dependencies(mid, (*loaded.imports, *loaded.export_decls))

    sccs = _tarjan_sccs(adj)
    graph = ModuleGraph(
        modules=modules,
        entry_id=entry_id,
        sccs=sccs,
        adjacency={mid: tuple(targets) for mid, targets in adj.items()},
        source_adjacency={mid: tuple(targets) for mid, targets in source_adj.items()},
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
    inline_command: bool = False,
) -> ParsedEntryModule:
    """Parse an entry source and collect its lexical advisories.

    *inline_command* applies the ``agm exec -c`` synthetic-entry wrap. Syntax
    failures intentionally propagate to let callers choose their own raising or
    diagnostic-capturing policy.
    """
    canonical_path, source_id = entry_source_id(entry_path)
    with spaced_qualifier_collector() as spaced_sink:
        try:
            program, next_id = parse_program_seeded(entry_source, start_id=0, source=source_id)
        except AglSyntaxError as error:
            raise EntryParseSyntaxError(error, tuple(spaced_sink)) from error
    if inline_command:
        program, next_id = wrap_inline_program(program, next_node_id=next_id)
    return ParsedEntryModule(
        program=program,
        next_id=next_id,
        canonical_path=canonical_path,
        source_id=source_id,
        spaced_qualifiers=tuple(spaced_sink),
    )


def _entry_module_id(canonical_entry_path: Path | None, roots: RootSet) -> ModuleId:
    """Return the identity the graph's entry module is keyed by.

    A mounted package's manifest declares a module tree, so a file inside one
    is that package's module however it was reached: it keeps its declared id,
    and the rest of the package may import it back.  Every other entry — inline
    ``-c`` source, a REPL splice, a file under a loose root — has no module
    identity and takes :data:`~agm.agl.modules.ids.ENTRY_ID`.

    An identity is only worth having when it is the one everything else would
    reach the file by, so the derived id is kept only if resolving it against
    the same roots leads back to this very file.  Where it cannot — the id is
    unspellable, resolves nowhere, resolves to several files, or resolves to a
    different one — the file simply has no identity, exactly like a loose one.
    """
    if canonical_entry_path is None:
        return ENTRY_ID
    derived = roots.package_module_id_for(canonical_entry_path)
    if derived is None or not _resolves_back_to(derived, canonical_entry_path, roots):
        return ENTRY_ID
    return derived


def _resolves_back_to(module_id: ModuleId, path: Path, roots: RootSet) -> bool:
    """Return whether *module_id* is spellable and *roots* resolve it to *path*."""
    try:
        ModuleId.from_path(module_id.path_str())
    except ValueError:
        return False
    try:
        return resolve_module(module_id, roots) == path
    except (AmbiguousModule, ModuleNotFound):
        return False


def _build_entry_loaded_module(
    program: syntax.Program,
    next_id: int,
    *,
    entry_id: ModuleId,
    canonical_entry_path: Path | None,
    entry_source_id: SourceId,
    default_stdlib: bool,
    spaced_qualifiers: tuple[SpacedQualifier, ...],
    source_text: str,
) -> tuple[LoadedModule, int]:
    """Build the entry :class:`LoadedModule` from an already-parsed program.

    Used by :func:`build_repl_graph`. Injects the ``std/prelude`` prelude import when
    *default_stdlib* is set, consuming one more node id, and derives the
    companion path for a declared extern. The prelude never imports itself, so
    an entry that *is* ``std/prelude`` is exempt exactly as the library path
    is. Returns the built module together with the next free node id.
    """
    if default_stdlib and entry_id != STD_PRELUDE_ID:
        program = _with_default_stdlib_import(program, import_node_id=next_id)
        next_id += 1
    imports = _extract_imports(program)
    entry_loaded = LoadedModule(
        module_id=entry_id,
        program=program,
        path=canonical_entry_path,
        source=entry_source_id,
        imports=imports,
        export_decls=_extract_exports(program),
        source_text=source_text,
        spaced_qualifiers=spaced_qualifiers,
        companion_path=_companion_path_for(entry_id, program, canonical_entry_path),
    )
    return entry_loaded, next_id


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

    Accepts an already-parsed ``Program`` AST (from the REPL's per-entry
    parse, or from :func:`parse_entry_module` in a host pipeline) and performs
    BFS loading only for library modules that are not already cached. Node ids
    in newly-loaded modules are seeded from *next_start_id* so they remain
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
    default_stdlib:
        Whether to inject the standard-library prelude into the entry and
        newly loaded library modules.
    spaced_qualifiers:
        Spaced-qualifier advisories collected while *program* was lexed.
    default_label:
        The entry source label used when *path* is ``None`` (inline entry).
        Defaults to ``"<repl>"`` for the REPL's own incremental sessions;
        callers outside the REPL (e.g. an ``exec -c`` style host pipeline)
        pass the label :func:`parse_entry_module` used so diagnostics match.
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
        - A dict of newly-loaded modules (not in *cached*) for promotion,
          including the entry when a package names it.
    """
    canonical_entry_path, source_id = entry_source_id(path, default_label=default_label)

    seed_modules = dict(cached)
    entry_id = _entry_module_id(canonical_entry_path, roots)
    entry_loaded, next_start_id = _build_entry_loaded_module(
        program,
        next_start_id,
        entry_id=entry_id,
        canonical_entry_path=canonical_entry_path,
        entry_source_id=source_id,
        default_stdlib=default_stdlib,
        spaced_qualifiers=spaced_qualifiers,
        source_text=source_text,
    )

    return _load_into_graph(
        entry_loaded,
        entry_id=entry_id,
        roots=roots,
        canonical_entry_path=canonical_entry_path,
        seed_modules=seed_modules,
        start_id=next_start_id,
        default_stdlib=default_stdlib,
    )
