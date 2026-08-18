"""AgL module-graph loader.

This module provides :func:`load_graph`, which drives the full load-and-graph
phase of the AgL module system:

1. Parse the entry source (inline ``-c`` or a file on disk).
2. Extract top-level import/export declarations.
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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import agm.agl.syntax as syntax
from agm.agl.lexer import spaced_qualifier_collector
from agm.agl.modules.errors import (
    ImportEntryError,
    MissingExternCompanion,
    ModuleNotFound,
    PackageImportVisibilityError,
)
from agm.agl.modules.ids import ENTRY_ID, STD_BUILTIN_METHODS_ID, STD_CORE_ID, ModuleId
from agm.agl.modules.resolver import expand_wildcard, resolve_module
from agm.agl.modules.roots import RootSet
from agm.agl.parser import AglSyntaxError, build_infix_operator_table, resolve_infix_chains
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.parser.transform import resolve_infix_fixity
from agm.agl.syntax.advisories import SpacedQualifier
from agm.agl.syntax.nodes import (
    ExportDecl,
    ExportItem,
    FuncDef,
    ImportDecl,
    ImportItem,
    InfixDecl,
    OpenDecl,
    ScopeRegion,
    static_items,
)
from agm.agl.syntax.spans import SourceId, SourceSpan
from agm.agl.syntax.types import ImportMode
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
        Top-level :class:`~agm.agl.syntax.nodes.ImportDecl` nodes extracted
        from ``program.body.items``.
    export_decls:
        Top-level :class:`~agm.agl.syntax.nodes.ExportDecl` nodes extracted
        from ``program.body.items``.
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
        Direct dependency edges for every loaded module. This includes both
        imports and exports, after wildcard expansion, and is the authoritative
        reachability relation for graph consumers.
    source_adjacency:
        The subset of direct dependency edges authored in source. Loader
        injections are absent, retaining provenance for runtime inventories.
    ambient_modules:
        Standard-library modules reached from the optional builtin-method
        registry when the loader, rather than source imports, introduced it.
        They are linked and initialized for every selected program,
        but contribute no names to an entry's import environment. They are
        virtual dependencies only for :attr:`inference_sccs`; :attr:`adjacency`
        and :attr:`sccs` retain source import/export graph semantics.
    """

    modules: dict[ModuleId, LoadedModule]
    entry_id: ModuleId
    sccs: tuple[tuple[ModuleId, ...], ...]
    adjacency: dict[ModuleId, tuple[ModuleId, ...]]
    source_adjacency: dict[ModuleId, tuple[ModuleId, ...]] = field(default_factory=dict)
    ambient_modules: frozenset[ModuleId] = frozenset()
    roots: RootSet = field(default_factory=lambda: RootSet(roots=frozenset()))
    # Unambiguous root-level user fixities visible while assembling the entry.
    # REPL promotion uses this to retain declarations with relative priorities
    # without retaining imported operators as session declarations.
    entry_infix_ambient: dict[str, tuple[int, syntax.InfixAssoc]] = field(default_factory=dict)

    @property
    def inference_sccs(self) -> tuple[tuple[ModuleId, ...], ...]:
        """Return dependency-ordered SCCs with ambient methods available first.

        Ambient builtin-method modules do not form source import edges: adding
        them to :attr:`adjacency` would expose their routes and free functions
        to user scope and would alter ordinary graph consumers. Candidate
        inference nevertheless needs their closed method signatures before it
        processes any consuming module. This derived graph adds those ordering-
        only edges while retaining every real edge, so any resulting cycle is
        still inferred as one component.
        """
        if not self.ambient_modules:
            return self.sccs
        ambient = tuple(sorted(self.ambient_modules, key=_mid_sort_key))
        inference_adjacency = {
            mid: [
                *targets,
                *(ambient if mid not in self.ambient_modules else ()),
            ]
            for mid, targets in self.adjacency.items()
        }
        return _tarjan_sccs(inference_adjacency)

    def source_reachable_modules(self, module_id: ModuleId) -> tuple[ModuleId, ...]:
        """Return modules reachable through source-authored import/export edges.

        ``source_adjacency`` records edge provenance at load time, excluding
        loader-injected standard-library and builtin-method-registry edges.
        Thus a module reached through a source import remains reachable even
        if it also belongs to the ambient registry closure.
        """
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
        return tuple(reachable)

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
    """Return the module's ImportDecl nodes, including region-nested ones.

    Named scope regions are transparent to this walk, so a scoped ``import``
    is discovered as a module-graph edge exactly like a root one. Imports
    inside an ordinary nested block are not valid and are ignored here (the
    scope pass enforces the restriction).
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
        module_path=STD_CORE_ID.segments,
        wildcard=False,
        is_open=True,
        alias=None,
        mode=ImportMode.ALL,
        items=(),
        span=span,
        node_id=node_id,
    )


def _ambient_builtin_methods_import() -> ImportDecl:
    """Return the loader-only edge to the optional builtin-method registry."""
    span = SourceSpan(
        start_line=0,
        start_col=0,
        end_line=0,
        end_col=0,
        start_offset=0,
        end_offset=0,
        source=SourceId(label="<builtin-method-registry>"),
    )
    return ImportDecl(
        module_path=STD_BUILTIN_METHODS_ID.segments,
        wildcard=False,
        is_open=False,
        alias=None,
        mode=ImportMode.ALL,
        items=(),
        span=span,
        node_id=-1,
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
# Infix resolution
# ---------------------------------------------------------------------------


_OperatorOrigin = tuple[ModuleId, str]
_InfixFixity = tuple[int, syntax.InfixAssoc]


def _operator_declarations(program: syntax.Program) -> tuple[InfixDecl, ...]:
    """Return one module root's operator declarations in source order."""
    return tuple(item for item in program.body.items if isinstance(item, InfixDecl))


def _dependency_targets(
    decl: ImportDecl | ExportDecl, modules: Mapping[ModuleId, LoadedModule]
) -> tuple[ModuleId, ...]:
    """Return loaded targets for one import or export declaration."""
    if not decl.wildcard:
        return (ModuleId(segments=tuple(decl.module_path)),)
    prefix = tuple(decl.module_path)
    return tuple(
        sorted(
            (mid for mid in modules if not mid.is_entry and mid.segments[: len(prefix)] == prefix),
            key=ModuleId.path_str,
        )
    )


_OperatorPath = tuple[str, ...]
_OperatorExports = Mapping[_OperatorPath, set[_OperatorOrigin]]


def _operator_item_path(item: ImportItem | ExportItem) -> _OperatorPath:
    """Return an import/export selection item's complete source path."""
    return (*(segment.name for segment in item.scope_path), item.name)


def _operator_decl_scope_path(decl: ImportDecl | ExportDecl) -> _OperatorPath:
    """Return the enclosing lexical path of one import/export declaration."""
    return tuple(segment.name for segment in decl.scope_path)


def _selected_operator_exports(
    decl: ImportDecl | ExportDecl, exports: _OperatorExports
) -> dict[_OperatorPath, set[_OperatorOrigin]]:
    """Apply selection, renaming, and scoped re-exporting to operator paths."""
    selected: dict[_OperatorPath, set[_OperatorOrigin]]
    if decl.mode is ImportMode.ALL:
        selected = dict(exports)
    else:
        selected = {}
        for item in decl.items:
            prefix = _operator_item_path(item)
            for path, origins in exports.items():
                if path[: len(prefix)] == prefix:
                    selected[path] = origins
        if decl.mode is ImportMode.HIDING:
            selected = {path: origins for path, origins in exports.items() if path not in selected}

    result: dict[_OperatorPath, set[_OperatorOrigin]] = {}
    for path, origins in selected.items():
        exposed = path
        if decl.mode is ImportMode.USING:
            for item in decl.items:
                prefix = _operator_item_path(item)
                if item.rename is not None and path[: len(prefix)] == prefix:
                    exposed = (item.rename, *path[len(prefix) :])
                    break
        if isinstance(decl, ExportDecl):
            exposed = (*_operator_decl_scope_path(decl), *exposed)
        result.setdefault(exposed, set()).update(origins)
    return result


def _operator_export_maps(
    graph: ModuleGraph,
) -> dict[ModuleId, dict[_OperatorPath, set[_OperatorOrigin]]]:
    """Build operator export maps, preserving paths and re-export origins."""
    exports: dict[ModuleId, dict[_OperatorPath, set[_OperatorOrigin]]] = {
        mid: {(decl.name,): {(mid, decl.name)} for decl in _operator_declarations(loaded.program)}
        for mid, loaded in graph.modules.items()
    }

    def propagate() -> tuple[ExportDecl, ...]:
        changed_decls: list[ExportDecl] = []
        for mid, loaded in graph.modules.items():
            for decl in loaded.export_decls:
                for target in _dependency_targets(decl, graph.modules):
                    for path, origins in _selected_operator_exports(
                        decl, exports.get(target, {})
                    ).items():
                        before = len(exports[mid].setdefault(path, set()))
                        exports[mid][path].update(origins)
                        if len(exports[mid][path]) != before:
                            changed_decls.append(decl)
        return tuple(changed_decls)

    # As in scope's export resolver, an acyclic propagation path traverses at
    # most every export declaration once; the following pass observes its
    # fixed point. Further growth therefore means a scoped re-export cycle is
    # continually extending its exposed path rather than converging.
    declaration_count = sum(len(loaded.export_decls) for loaded in graph.modules.values())
    for _ in range(declaration_count + 1):
        changed_decls = propagate()
        if not changed_decls:
            return exports

    raise AglSyntaxError(
        "cyclic re-export expansion does not converge",
        span=changed_decls[-1].span,
    )


def _operator_open_declarations(
    program: syntax.Program,
) -> tuple[tuple[OpenDecl, _OperatorPath], ...]:
    """Return each ``open`` declaration with its enclosing lexical scope path."""
    opens: list[tuple[OpenDecl, _OperatorPath]] = []

    def collect(items: tuple[syntax.Item, ...], scope_path: _OperatorPath) -> None:
        for item in items:
            if isinstance(item, ScopeRegion):
                collect(item.items, (*scope_path, item.segment.name))
            elif isinstance(item, OpenDecl):
                opens.append((item, scope_path))

    collect(program.body.items, ())
    return tuple(opens)


def _local_scope_paths(program: syntax.Program) -> set[_OperatorPath]:
    """Return every named scope path declared by one module."""
    paths: set[_OperatorPath] = {()}

    def collect(items: tuple[syntax.Item, ...], scope_path: _OperatorPath) -> None:
        for item in items:
            if isinstance(item, ScopeRegion):
                path = (*scope_path, item.segment.name)
                paths.add(path)
                collect(item.items, path)

    collect(program.body.items, ())
    return paths


def _operator_import_routes(decl: ImportDecl, target: ModuleId) -> tuple[_OperatorPath, ...]:
    """Return the qualifier routes one import makes available for *target*."""
    if decl.alias is not None:
        return ((decl.alias,),)
    return tuple(target.segments[index:] for index in range(len(target.segments)))


def _select_opened_operator_members(
    decl: OpenDecl, members: Mapping[_OperatorPath, set[_OperatorOrigin]]
) -> dict[_OperatorPath, set[_OperatorOrigin]]:
    """Apply an ``open`` declaration's selection to relative operator members."""
    if decl.mode is ImportMode.ALL:
        return {path: set(origins) for path, origins in members.items()}

    selected: dict[_OperatorPath, set[_OperatorOrigin]] = {}
    for item in decl.items:
        prefix = _operator_item_path(item)
        for path, origins in members.items():
            if path[: len(prefix)] != prefix:
                continue
            exposed = (item.rename, *path[len(prefix) :]) if item.rename is not None else path
            selected.setdefault(exposed, set()).update(origins)
    if decl.mode is ImportMode.USING:
        return selected
    return {
        path: set(origins)
        for path, origins in members.items()
        if not any(
            path[: len(_operator_item_path(item))] == _operator_item_path(item)
            for item in decl.items
        )
    }


def _has_local_open_target(
    scope_paths: set[_OperatorPath],
    scope_path: _OperatorPath,
    requested_path: _OperatorPath,
) -> bool:
    """Whether an unrouted ``open`` resolves to a local scope first."""
    current = scope_path
    while True:
        if (*current, *requested_path) in scope_paths:
            return True
        if not current:
            return False
        current = current[:-1]


def _operator_open_members(
    module_id: ModuleId,
    decl: OpenDecl,
    scope_path: _OperatorPath,
    graph: ModuleGraph,
    exports: Mapping[ModuleId, _OperatorExports],
    local_scopes: set[_OperatorPath],
) -> dict[_OperatorPath, set[_OperatorOrigin]]:
    """Return relative operator members made bare by one scope ``open``."""
    requested_path = tuple(segment.name for segment in decl.scope_ref.scope_path)
    if not decl.scope_ref.module_route and _has_local_open_target(
        local_scopes, scope_path, requested_path
    ):
        return {}

    route = decl.scope_ref.module_route
    target_path = requested_path
    if not route and len(requested_path) > 1:
        route, target_path = (requested_path[0],), requested_path[1:]

    members: dict[_OperatorPath, set[_OperatorOrigin]] = {}
    for import_decl in graph.modules[module_id].imports:
        if not (import_decl.is_open or import_decl.mode is ImportMode.USING) and not route:
            continue
        if not route and scope_path[
            : len(_operator_decl_scope_path(import_decl))
        ] != _operator_decl_scope_path(import_decl):
            continue
        for target in _dependency_targets(import_decl, graph.modules):
            if route and route not in _operator_import_routes(import_decl, target):
                continue
            for path, origins in _selected_operator_exports(
                import_decl, exports.get(target, {})
            ).items():
                if path[: len(target_path)] == target_path and len(path) > len(target_path):
                    relative = path[len(target_path) :]
                    members.setdefault(relative, set()).update(origins)
    return _select_opened_operator_members(decl, members)


def _operator_bare_layers(
    module_id: ModuleId,
    graph: ModuleGraph,
    exports: Mapping[ModuleId, _OperatorExports],
) -> dict[_OperatorPath, dict[str, set[_OperatorOrigin]]]:
    """Build bare operator contributions keyed by their lexical scope layer."""
    loaded = graph.modules[module_id]
    layers: dict[_OperatorPath, dict[str, set[_OperatorOrigin]]] = {}

    def contribute(
        scope_path: _OperatorPath, members: Mapping[_OperatorPath, set[_OperatorOrigin]]
    ) -> None:
        layer = layers.setdefault(scope_path, {})
        for path, origins in members.items():
            if len(path) == 1:
                layer.setdefault(path[0], set()).update(origins)

    for decl in loaded.imports:
        if not (decl.is_open or decl.mode is ImportMode.USING):
            continue
        members: dict[_OperatorPath, set[_OperatorOrigin]] = {}
        for target in _dependency_targets(decl, graph.modules):
            for path, origins in _selected_operator_exports(decl, exports.get(target, {})).items():
                members.setdefault(path, set()).update(origins)
        contribute(_operator_decl_scope_path(decl), members)

    local_scopes = _local_scope_paths(loaded.program)
    for open_decl, scope_path in _operator_open_declarations(loaded.program):
        contribute(
            scope_path,
            _operator_open_members(module_id, open_decl, scope_path, graph, exports, local_scopes),
        )
    return layers


def _visible_operator_origins(
    scope_path: _OperatorPath,
    bare_layers: Mapping[_OperatorPath, Mapping[str, set[_OperatorOrigin]]],
) -> dict[str, set[_OperatorOrigin]]:
    """Return operators visible through the nearest bare-contribution layers."""
    visible: dict[str, set[_OperatorOrigin]] = {}
    current = scope_path
    while True:
        for name, origins in bare_layers.get(current, {}).items():
            visible.setdefault(name, set(origins))
        if not current:
            return visible
        current = current[:-1]


def _raw_chain_scope_paths(program: syntax.Program) -> dict[int, _OperatorPath]:
    """Map every raw infix chain to the named scope that lexically owns it."""
    paths: dict[int, _OperatorPath] = {}

    def collect(item: object, scope_path: _OperatorPath) -> None:
        if isinstance(item, syntax.ScopeRegion):
            nested_path = (*scope_path, item.segment.name)
            for child in item.items:
                collect(child, nested_path)
            return
        if (
            isinstance(
                item,
                (
                    syntax.BuiltinVarDecl,
                    syntax.EnumDef,
                    syntax.ExceptionDef,
                    syntax.FuncDef,
                    syntax.LetDecl,
                    syntax.ParamDecl,
                    syntax.RecordDef,
                    syntax.TypeAlias,
                    syntax.VarDecl,
                ),
            )
            and item.scope_path
        ):
            scope_path = tuple(segment.name for segment in item.scope_path)

        def record(node: object) -> None:
            if isinstance(node, syntax.RawInfixChain):
                paths[node.node_id] = scope_path

        syntax.walk(item, record)

    for root_item in program.body.items:
        collect(root_item, ())
    return paths


def _resolve_graph_infix(
    graph: ModuleGraph,
    session_infix: Mapping[str, _InfixFixity] | None = None,
) -> ModuleGraph:
    """Resolve each raw module chain with the fixity visible at that module."""
    declarations = {
        mid: _operator_declarations(loaded.program) for mid, loaded in graph.modules.items()
    }
    exports = _operator_export_maps(graph)
    bare_layers = {mid: _operator_bare_layers(mid, graph, exports) for mid in graph.modules}
    origin_fixities: dict[_OperatorOrigin, _InfixFixity] = {}
    session_origins: dict[str, _OperatorOrigin] = {}
    if session_infix is not None:
        for name, fixity in session_infix.items():
            origin = (ENTRY_ID, f"<session:{name}>")
            session_origins[name] = origin
            origin_fixities[origin] = fixity
    unresolved: AglSyntaxError | None = None

    def visible_at(mid: ModuleId, scope_path: _OperatorPath) -> dict[str, set[_OperatorOrigin]]:
        visible = _visible_operator_origins(scope_path, bare_layers[mid])
        if mid == ENTRY_ID:
            for name, origin in session_origins.items():
                visible.setdefault(name, set()).add(origin)
        return visible

    # Fixities normally resolve locally, but this fixed point also lets a
    # declaration express its priority relative to a bare-visible imported operator.
    changed = True
    while changed:
        changed = False
        unresolved = None
        for mid, decls in declarations.items():
            fixity_ambient: dict[str, _InfixFixity] = {}
            for name, origins in visible_at(mid, ()).items():
                values = {
                    origin_fixities[origin] for origin in origins if origin in origin_fixities
                }
                if len(values) == 1:
                    fixity_ambient[name] = next(iter(values))
            try:
                resolved = resolve_infix_fixity(decls, fixity_ambient)
            except AglSyntaxError as error:
                unresolved = error
                continue
            for decl in decls:
                fixity = resolved[decl.name]
                origin = (mid, decl.name)
                if origin_fixities.get(origin) != fixity:
                    origin_fixities[origin] = fixity
                    changed = True
    if unresolved is not None:
        raise unresolved

    modules: dict[ModuleId, LoadedModule] = {}
    entry_infix_ambient: dict[str, _InfixFixity] = {}
    for mid, loaded in graph.modules.items():
        chain_tables: dict[int, dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]]] = {}
        chain_conflicts: dict[int, frozenset[str]] = {}
        own_names = {decl.name for decl in declarations[mid]}
        for chain_id, scope_path in _raw_chain_scope_paths(loaded.program).items():
            ambient: dict[str, _InfixFixity] = {}
            conflicts: set[str] = set()
            for name, origins in visible_at(mid, scope_path).items():
                values = {origin_fixities[origin] for origin in origins}
                if len(values) == 1:
                    ambient[name] = next(iter(values))
                else:
                    conflicts.add(name)
            conflicts.difference_update(own_names)
            chain_tables[chain_id] = build_infix_operator_table(declarations[mid], ambient)
            chain_conflicts[chain_id] = frozenset(conflicts)

        root_ambient: dict[str, _InfixFixity] = {}
        root_conflicts: set[str] = set()
        for name, origins in visible_at(mid, ()).items():
            values = {origin_fixities[origin] for origin in origins}
            if len(values) == 1:
                root_ambient[name] = next(iter(values))
            else:
                root_conflicts.add(name)
        root_conflicts.difference_update(own_names)
        if mid == ENTRY_ID:
            entry_infix_ambient = root_ambient
        resolved_program = resolve_infix_chains(
            loaded.program,
            build_infix_operator_table(declarations[mid], root_ambient),
            conflicting_operators=frozenset(root_conflicts),
            operator_tables=chain_tables,
            conflicting_operators_by_chain=chain_conflicts,
        )
        modules[mid] = (
            loaded
            if resolved_program == loaded.program
            else replace(loaded, program=resolved_program)
        )
    return replace(graph, modules=modules, entry_infix_ambient=entry_infix_ambient)


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
    session_infix: Mapping[str, _InfixFixity] | None = None,
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
    dict of modules loaded during this call (those not in *seed_modules*), after
    their source infix chains are resolved to match the returned graph.
    """
    modules: dict[ModuleId, LoadedModule] = dict(seed_modules)
    modules[ENTRY_ID] = entry_loaded
    newly_loaded: dict[ModuleId, LoadedModule] = {}
    adj: dict[ModuleId, list[ModuleId]] = {}
    source_adj: dict[ModuleId, list[ModuleId]] = {}
    next_id = start_id

    # BFS queue: (module id, dependency decl, canonical target path). We sort each
    # batch of newly-discovered ids before enqueuing so the traversal order —
    # and therefore the start_id seed assignments — are stable regardless of
    # dict/set ordering.
    queue: deque[_ResolvedDependency] = deque()
    loader_injected_registry = False

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
                if decl.span.source.label not in {"<stdlib-import>", "<builtin-method-registry>"}:
                    source_targets.append(mid)
                if mid not in modules:
                    new_pairs.append((mid, decl, target_path))
        adj[source] = targets
        source_adj[source] = source_targets
        new_pairs.sort(key=_pair_sort_key)
        queue.extend(new_pairs)

    _resolve_dependencies(ENTRY_ID, (*entry_loaded.imports, *entry_loaded.export_decls))

    # Cached modules were loaded in an earlier REPL compilation. Rebuild their
    # adjacency before draining the queue so any dependency absent from this
    # invocation's cache is loaded normally.
    for mid, loaded in modules.items():
        if mid != ENTRY_ID:
            _resolve_dependencies(mid, (*loaded.imports, *loaded.export_decls))

    if default_stdlib:
        registry_decl = _ambient_builtin_methods_import()
        try:
            registry_path = resolve_module(STD_BUILTIN_METHODS_ID, roots, span=registry_decl.span)
        except ModuleNotFound:
            # The registry was introduced after existing standard libraries.
            # Its absence deliberately preserves their current module set.
            pass
        else:
            queue.append((STD_BUILTIN_METHODS_ID, registry_decl, registry_path))
            loader_injected_registry = True

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
                resolve_infix=False,
            )
        if default_stdlib and mid != STD_CORE_ID:
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
            spaced_qualifiers=tuple(spaced_sink),
            companion_path=_companion_path_for(mid, program, canon_path),
        )
        modules[mid] = loaded
        newly_loaded[mid] = loaded
        _resolve_dependencies(mid, (*loaded.imports, *loaded.export_decls))

    sccs = _tarjan_sccs(adj)
    # The registry is ambient only when the loader alone introduced it. An
    # explicit source import (including one in a standard-library module)
    # keeps its normal route, visibility, and runtime-dependency semantics.
    ambient_roots = (
        {STD_BUILTIN_METHODS_ID}
        if loader_injected_registry
        and not any(STD_BUILTIN_METHODS_ID in targets for targets in adj.values())
        else set()
    )
    ambient_modules: set[ModuleId] = set()
    pending = list(ambient_roots)
    while pending:
        current = pending.pop()
        if current in ambient_modules:
            continue
        ambient_modules.add(current)
        pending.extend(adj[current])
    graph = ModuleGraph(
        modules=modules,
        entry_id=ENTRY_ID,
        sccs=sccs,
        adjacency={mid: tuple(targets) for mid, targets in adj.items()},
        source_adjacency={mid: tuple(targets) for mid, targets in source_adj.items()},
        ambient_modules=frozenset(ambient_modules),
        roots=roots,
    )
    resolved_graph = _resolve_graph_infix(graph, session_infix)
    resolved_newly_loaded = {mid: resolved_graph.modules[mid] for mid in newly_loaded}
    return resolved_graph, next_id, resolved_newly_loaded


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
            program, next_id = parse_program_seeded(
                entry_source, start_id=0, source=source_id, resolve_infix=False
            )
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
    entry_loaded = LoadedModule(
        module_id=ENTRY_ID,
        program=program,
        path=canonical_entry_path,
        source=entry_source_id,
        imports=_extract_imports(program),
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
    session_infix: Mapping[str, _InfixFixity] | None = None,
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
        session_infix=session_infix,
    )
