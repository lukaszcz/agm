"""Contribution-based import environments and qualified resolution."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TypeAlias

from agm.agl.modules.ids import ModuleId, spell_declaration
from agm.agl.scope.symbols import MissRepair, UnknownMemberError
from agm.agl.scope.symbols import import_item_path as _item_path
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.scope.symbols import to_bare_path as _path
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    ImportDecl,
    ImportItem,
    RecordDef,
)
from agm.agl.syntax.nodes import TypeAlias as TypeAliasDecl
from agm.agl.syntax.spans import SourceSpan

__all__ = [
    "EMPTY_IMPORT_ENV",
    "Exposure",
    "ItemDeclaration",
    "ImportEnv",
    "ImportTarget",
    "BareRoute",
    "ModuleContribution",
    "NameAtom",
    "PathAtom",
    "QName",
    "ScopeOrigins",
    "SingleTarget",
    "WildcardTarget",
    "alias_prefix",
    "build_import_env",
    "validate_import_items",
    "target_modules",
    "contribution_routes",
    "matching_atoms",
    "qualifier_candidates",
    "qualifier_hides",
    "qualifier_exposures",
    "qualifier_member_decls",
    "qualifier_members",
    "qualifier_scope_paths",
    "unqualified_exposures",
    "declares_bare_constructor",
]

PathAtom: TypeAlias = tuple[str, ...]
# Root members retain their historical string representation at this boundary;
# scoped members use a structured tuple.  All policy operations normalize it.
NameAtom: TypeAlias = str | PathAtom
QName: TypeAlias = tuple[ModuleId, NameAtom]
ScopeOrigins: TypeAlias = frozenset[QName]
Exposure: TypeAlias = tuple[NameAtom, QName, frozenset[int]]
"""A path imports expose, what it names, and the import declarations contributing it."""
BareRoute: TypeAlias = tuple[ModuleId, PathAtom]


def declares_bare_constructor(
    qnames: Iterable[QName],
    all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAliasDecl],
) -> bool:
    """Whether one of *qnames* is a record or exception declaring its bare name.

    An enum member's bare spelling is an injected convenience that stays
    reachable qualified, so on an import surface it yields to a same-named
    record or exception constructor, whichever module declares it.
    """
    return any(
        isinstance(all_public_types.get(qname), (RecordDef, ExceptionDef)) for qname in qnames
    )


def _path_sort_key(atom: NameAtom) -> str:
    return "::".join(_path(atom))


@dataclass(frozen=True, slots=True)
class SingleTarget:
    """The import resolves to exactly one module."""

    module: ModuleId


@dataclass(frozen=True, slots=True)
class WildcardTarget:
    """The wildcard import expands to these modules."""

    modules: frozenset[ModuleId]


ImportTarget = SingleTarget | WildcardTarget


def _frozen_routes(
    routes: Mapping[tuple[str, ...], set[ModuleId]],
) -> Mapping[tuple[str, ...], tuple[ModuleId, ...]]:
    return MappingProxyType(
        {
            qualifier: tuple(sorted(modules, key=ModuleId.path_str))
            for qualifier, modules in routes.items()
        }
    )


@dataclass(frozen=True, slots=True)
class RouteSurface:
    """What one import route -- an alias, or the module path -- brings from a module.

    ``decls`` are the import declarations forming the route, ``member_decls``
    those contributing each member; ``hidden`` holds the declarations and
    scopes a ``hiding`` removed from the route that none of them brings.
    """

    members: Mapping[NameAtom, QName] = field(default_factory=dict)
    scope_paths: frozenset[NameAtom] = frozenset()
    member_decls: Mapping[NameAtom, frozenset[int]] = field(default_factory=dict)
    decls: frozenset[int] = frozenset()
    hidden: frozenset[NameAtom] = frozenset()


_NO_SURFACE = RouteSurface()


@dataclass(frozen=True, slots=True)
class ModuleContribution:
    """One imported module's route-keyed declaration and named-scope contribution.

    ``routes`` maps each alias, and ``None`` for the module path, to what it
    brings; ``exports`` is everything the module exports, hidden or not.
    """

    module: ModuleId
    members: Mapping[NameAtom, QName]
    path_enabled: bool
    aliases: frozenset[str]
    routes: Mapping[str | None, RouteSurface] = field(default_factory=dict)
    exports: Mapping[NameAtom, QName] = field(default_factory=dict)

    def __post_init__(self) -> None:
        members: Mapping[NameAtom, QName] = MappingProxyType(
            {atom: self.members[atom] for atom in sorted(self.members, key=_path_sort_key)}
        )
        ordered: list[str | None] = [None] if None in self.routes else []
        ordered.extend(sorted(route for route in self.routes if route is not None))
        routes: Mapping[str | None, RouteSurface] = MappingProxyType(
            {
                route: replace(
                    self.routes[route],
                    members=MappingProxyType(
                        {
                            atom: self.routes[route].members[atom]
                            for atom in sorted(self.routes[route].members, key=_path_sort_key)
                        }
                    ),
                )
                for route in ordered
            }
        )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "routes", routes)


@dataclass(frozen=True, slots=True)
class ItemDeclaration:
    """The declaration one tail or ``hiding`` item names in one imported *module*.

    ``declaration`` is the export the item's path, or its longest prefix
    naming an exported alias, selects; ``beneath`` is the rest of the path,
    read beneath that alias's target by scope. ``item`` and ``span`` spell
    a path naming nothing.
    """

    module: ModuleId
    item: PathAtom
    declaration: QName
    beneath: PathAtom
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class ImportEnv:
    """Pure contribution environment, including structured public paths.

    ``decl_bare`` holds, per region-scoped import declaration (keyed by its
    ``node_id``), exactly the atoms *that declaration alone* contributes
    bare. It is kept separate from ``unqualified`` -- the module-wide bare
    table a root-position tailed import feeds -- so a scoped import's bare
    names can be snapshotted onto its own region instead of leaking to the
    whole module; its qualifier route still flows through ``contributions``.
    Each atom maps to
    every origin it draws from a wildcard's expansion, exactly like
    ``unqualified``: two modules exposing the same bare name is deferred to
    the name's first use, not raised here. ``decl_scope_routes`` carries,
    per tailed declaration at the root or in a region, the parallel
    namespace-only contribution for scopes that have no declaration member
    to put in a bare table.
    ``decl_hidden`` holds, per tailed declaration, the bare atoms -- of
    declarations and scopes alike -- its ``hiding`` or the imported module's
    export ``hiding`` removed from its tail; ``unqualified_hidden`` holds the
    declarations a root-position import's ``hiding`` removed.
    ``unqualified_decls`` names the declarations contributing each root bare
    atom's origins, as :attr:`ModuleContribution.path_member_decls` does a
    route's members; ``decl_hiding`` holds the declarations each tailed or
    routed declaration's ``hiding`` names, which scope removes from every
    spelling it contributes. ``decl_tail_beneath`` holds, per tailed
    declaration, each spelling a tail item naming a path beneath an exported
    alias exposes, with the items; scope reads the path beneath the target.
    ``decl_withheld`` holds, per declaration, the declarations the imported
    module's export ``hiding`` removes beneath each export it brings, by the
    export's declaration.
    """

    contributions: Mapping[ModuleId, ModuleContribution]
    unqualified: Mapping[NameAtom, frozenset[QName]]
    decl_bare: Mapping[int, Mapping[NameAtom, frozenset[QName]]] = field(default_factory=dict)
    unqualified_routes: Mapping[NameAtom, frozenset[BareRoute]] = field(default_factory=dict)
    decl_bare_routes: Mapping[int, Mapping[NameAtom, frozenset[BareRoute]]] = field(
        default_factory=dict
    )
    decl_scope_routes: Mapping[int, Mapping[NameAtom, frozenset[BareRoute]]] = field(
        default_factory=dict
    )
    decl_hidden: Mapping[int, frozenset[NameAtom]] = field(default_factory=dict)
    unqualified_hidden: frozenset[QName] = frozenset()
    unqualified_decls: Mapping[NameAtom, Mapping[QName, frozenset[int]]] = field(
        default_factory=dict
    )
    decl_hiding: Mapping[int, tuple[ItemDeclaration, ...]] = field(default_factory=dict)
    decl_tail_beneath: Mapping[int, Mapping[NameAtom, frozenset[ItemDeclaration]]] = field(
        default_factory=dict
    )
    decl_withheld: Mapping[int, Mapping[QName, frozenset[QName]]] = field(default_factory=dict)
    scope_origins_by_route: Mapping[BareRoute, ScopeOrigins] = field(default_factory=dict)
    suffix_routes: Mapping[tuple[str, ...], tuple[ModuleId, ...]] = field(
        init=False, repr=False, compare=False
    )
    anchored_routes: Mapping[tuple[str, ...], tuple[ModuleId, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        contributions: Mapping[ModuleId, ModuleContribution] = MappingProxyType(
            {
                module: self.contributions[module]
                for module in sorted(self.contributions, key=ModuleId.path_str)
            }
        )
        unqualified: Mapping[NameAtom, frozenset[QName]] = MappingProxyType(
            {atom: self.unqualified[atom] for atom in sorted(self.unqualified, key=_path_sort_key)}
        )
        decl_bare: Mapping[int, Mapping[NameAtom, frozenset[QName]]] = MappingProxyType(
            {
                node_id: MappingProxyType(dict(members))
                for node_id, members in self.decl_bare.items()
            }
        )
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "unqualified", unqualified)
        unqualified_routes: Mapping[NameAtom, frozenset[BareRoute]] = MappingProxyType(
            dict(self.unqualified_routes)
        )
        decl_bare_routes: Mapping[int, Mapping[NameAtom, frozenset[BareRoute]]] = MappingProxyType(
            {
                node_id: MappingProxyType(dict(routes))
                for node_id, routes in self.decl_bare_routes.items()
            }
        )
        decl_scope_routes: Mapping[int, Mapping[NameAtom, frozenset[BareRoute]]] = MappingProxyType(
            {
                node_id: MappingProxyType(dict(routes))
                for node_id, routes in self.decl_scope_routes.items()
            }
        )
        object.__setattr__(self, "decl_bare", decl_bare)
        object.__setattr__(self, "unqualified_routes", unqualified_routes)
        object.__setattr__(self, "decl_bare_routes", decl_bare_routes)
        object.__setattr__(self, "decl_scope_routes", decl_scope_routes)
        decl_hidden: Mapping[int, frozenset[NameAtom]] = MappingProxyType(dict(self.decl_hidden))
        object.__setattr__(self, "decl_hidden", decl_hidden)
        scope_origins_by_route: Mapping[BareRoute, ScopeOrigins] = MappingProxyType(
            dict(self.scope_origins_by_route)
        )
        object.__setattr__(self, "scope_origins_by_route", scope_origins_by_route)
        suffix: dict[tuple[str, ...], set[ModuleId]] = {}
        anchored: dict[tuple[str, ...], set[ModuleId]] = {}
        for module, contribution in contributions.items():
            if contribution.path_enabled:
                for index in range(len(module.segments)):
                    suffix.setdefault(module.segments[index:], set()).add(module)
                anchored.setdefault(module.segments, set()).add(module)
            for alias in contribution.aliases:
                suffix.setdefault((alias,), set()).add(module)
        object.__setattr__(self, "suffix_routes", _frozen_routes(suffix))
        object.__setattr__(self, "anchored_routes", _frozen_routes(anchored))


# A module with no imports at all shares this single empty environment rather
# than each caller allocating its own throwaway ``ImportEnv()``. Safe to share:
# the dataclass is frozen and every mapping field is frozen in ``__post_init__``,
# so nothing can mutate ``contributions``/``unqualified`` through a reference.
EMPTY_IMPORT_ENV = ImportEnv(contributions={}, unqualified={})


@dataclass(slots=True)
class _RouteAccumulator:
    members: dict[NameAtom, QName] = field(default_factory=dict)
    scope_paths: set[NameAtom] = field(default_factory=set)
    member_decls: dict[NameAtom, set[int]] = field(default_factory=dict)
    decls: set[int] = field(default_factory=set)
    hidden: set[NameAtom] = field(default_factory=set)

    def freeze(self) -> RouteSurface:
        """The surface accumulated; hidden atoms the route brings anyway are not hidden."""
        return RouteSurface(
            self.members,
            frozenset(self.scope_paths),
            _frozen_decls(self.member_decls),
            frozenset(self.decls),
            frozenset(self.hidden - self.members.keys() - self.scope_paths),
        )


@dataclass(slots=True)
class _ContributionAccumulator:
    members: dict[NameAtom, QName] = field(default_factory=dict)
    path_enabled: bool = False
    routes: dict[str | None, _RouteAccumulator] = field(default_factory=dict)


def matching_atoms(surface: Iterable[NameAtom], prefix: PathAtom) -> tuple[NameAtom, ...]:
    """Return every atom of *surface* that a selection *prefix* reaches."""
    return tuple(atom for atom in surface if _path(atom)[: len(prefix)] == prefix)


def _selected_public_atoms(
    items: tuple[ImportItem, ...],
    exports: Iterable[NameAtom],
    scope_exports: Iterable[NameAtom],
) -> tuple[tuple[NameAtom, ...], tuple[NameAtom, ...]]:
    """Expand selected declaration atoms and independent scope identities.

    An item naming nothing selects nothing (:func:`validate_import_items`).
    """
    matched_exports: dict[NameAtom, None] = {}
    matched_scopes: dict[NameAtom, None] = {}
    for item in items:
        prefix = _item_path(item)
        for atom in matching_atoms(exports, prefix):
            matched_exports[atom] = None
        for atom in matching_atoms(scope_exports, prefix):
            matched_scopes[atom] = None
    return tuple(matched_exports), tuple(matched_scopes)


def validate_import_items(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
    exports: Mapping[ModuleId, Mapping[NameAtom, QName]],
    scope_exports: Mapping[ModuleId, Mapping[NameAtom, ScopeOrigins]],
    aliases: Collection[QName],
) -> None:
    """Reject the first ``hiding`` or tail item of *decls* naming nothing its module exports.

    An item beneath one of the exported *aliases* names a path beneath the
    alias's target, which scope reads.
    """
    for decl in decls:
        for module in target_modules(targets[decl.node_id]):
            module_exports = exports[module]
            for item in (*decl.hidden, *(decl.tail or ())):
                prefix = _item_path(item)
                if (
                    not matching_atoms(module_exports, prefix)
                    and not matching_atoms(scope_exports[module], prefix)
                    and alias_prefix(prefix, module_exports, aliases) is None
                ):
                    raise UnknownMemberError(
                        spell_declaration(module, prefix),
                        span=decl.span,
                        repair=MissRepair.NOT_EXPORTED,
                    )


def alias_prefix(
    path: PathAtom, exports: Mapping[NameAtom, QName], aliases: Collection[QName]
) -> tuple[QName, PathAtom] | None:
    """The exported alias the longest proper prefix of *path* names, and the rest; if any."""
    for end in range(len(path) - 1, 0, -1):
        qname = exports.get(_atom(path[:end]))
        if qname is not None and qname in aliases:
            return qname, path[end:]
    return None


def _item_declarations(
    items: tuple[ImportItem, ...],
    module: ModuleId,
    exports: Mapping[NameAtom, QName],
    aliases: Collection[QName],
    span: SourceSpan,
) -> tuple[ItemDeclaration, ...]:
    """The declarations *items* name in *module*: an export, or a path beneath an alias.

    An item naming only a named scope names no declaration.
    """
    named: list[ItemDeclaration] = []
    for item in items:
        path = _item_path(item)
        exported = exports.get(_atom(path))
        found = (exported, ()) if exported is not None else alias_prefix(path, exports, aliases)
        if found is not None:
            named.append(ItemDeclaration(module, path, *found, span))
    return tuple(named)


def _tail_beneath_exposures(
    items: tuple[ImportItem, ...],
    module: ModuleId,
    exports: Mapping[NameAtom, QName],
    aliases: Collection[QName],
    span: SourceSpan,
) -> dict[NameAtom, set[ItemDeclaration]]:
    """The spellings tail *items* naming a path beneath an alias *module* exports expose.

    Each exposes its path, and a renamed one its rename too.
    """
    exposures: dict[NameAtom, set[ItemDeclaration]] = {}
    for item in items:
        path = _item_path(item)
        if matching_atoms(exports, path):
            continue
        beneath = alias_prefix(path, exports, aliases)
        if beneath is None:
            continue
        named = ItemDeclaration(module, path, *beneath, span)
        spellings: list[PathAtom] = [path] if item.rename is None else [path, (item.rename,)]
        for exposed in spellings:
            exposures.setdefault(_atom(exposed), set()).add(named)
    return exposures


def _tail_exposures(
    decl: ImportDecl, selected: tuple[NameAtom, ...]
) -> tuple[tuple[NameAtom, NameAtom], ...]:
    """Return the bare spellings a tailed import exposes of *selected*, paired with their sources.

    Serves both namespaces: declarations look their ``QName`` up from the
    module's export map, named scopes carry only the surface path.
    """
    if decl.tail is None:
        return ()
    result: dict[tuple[NameAtom, NameAtom], None] = {}
    for source in selected:
        result[(source, source)] = None
        source_path = _path(source)
        for item in decl.tail:
            prefix = _item_path(item)
            if item.rename is not None and source_path[: len(prefix)] == prefix:
                exposed = _atom((item.rename, *source_path[len(prefix) :]))
                result[(exposed, source)] = None
    return tuple(result)


def target_modules(target: ImportTarget) -> tuple[ModuleId, ...]:
    """The modules an import of *target* names."""
    return (
        (target.module,)
        if isinstance(target, SingleTarget)
        else tuple(sorted(target.modules, key=ModuleId.path_str))
    )


_NOTHING_WITHHELD: Mapping[ModuleId, Mapping[NameAtom, frozenset[QName]]] = MappingProxyType({})


def build_import_env(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
    exports: Mapping[ModuleId, Mapping[NameAtom, QName]],
    scope_exports: Mapping[ModuleId, Mapping[NameAtom, ScopeOrigins]],
    aliases: Collection[QName] = (),
    withheld: Mapping[ModuleId, Mapping[NameAtom, frozenset[QName]]] = _NOTHING_WITHHELD,
) -> ImportEnv:
    """Build route and implicit-tail contributions for import declarations.

    A region-scoped declaration still contributes qualifier routes module-wide,
    but its implicit tail's bare atoms are recorded in ``decl_bare`` so the
    scope pass can narrow them to that region. *aliases* are the program's
    type aliases, beneath whose exports a ``hiding`` item may name a path.
    *withheld* holds, per module, what its export ``hiding`` removes beneath
    each re-exported atom; a declaration reaching one export through several
    atoms withholds only what each does. An atom there the module does not
    export is one its export ``hiding`` removed, with its declarations: the
    import hides it like one of its own ``hiding`` items.
    """
    accumulators: dict[ModuleId, _ContributionAccumulator] = {}
    root_bare: dict[NameAtom, dict[QName, set[int]]] = {}
    decl_bare: dict[int, dict[NameAtom, set[QName]]] = {}
    root_bare_routes: dict[NameAtom, set[BareRoute]] = {}
    decl_bare_routes: dict[int, dict[NameAtom, set[BareRoute]]] = {}
    decl_scope_routes: dict[int, dict[NameAtom, set[BareRoute]]] = {}
    decl_hidden: dict[int, set[NameAtom]] = {}
    decl_hiding: dict[int, list[ItemDeclaration]] = {}
    decl_tail_beneath: dict[int, dict[NameAtom, set[ItemDeclaration]]] = {}
    decl_withheld: dict[int, dict[QName, frozenset[QName]]] = {}
    root_hidden: set[QName] = set()
    scope_origins_by_route: dict[BareRoute, ScopeOrigins] = {}
    for decl in decls:
        target = targets[decl.node_id]
        modules = target_modules(target)
        for module in modules:
            module_exports = exports[module]
            module_scopes = scope_exports[module]
            hidden_exports, hidden_scopes = _selected_public_atoms(
                decl.hidden, module_exports, module_scopes
            )
            module_withheld = withheld.get(module, {})
            unexported = {
                source: origins
                for source, origins in module_withheld.items()
                if source not in module_exports
            }
            named = (
                *_item_declarations(decl.hidden, module, module_exports, aliases, decl.span),
                *(
                    ItemDeclaration(module, _path(source), origin, (), decl.span)
                    for source, origins in unexported.items()
                    for origin in origins
                    if origin in aliases
                ),
            )
            if named:
                decl_hiding.setdefault(decl.node_id, []).extend(named)
            surface = (*module_exports, *unexported)
            if decl.tail is None:
                selected_exports: tuple[NameAtom, ...] = ()
                selected_scopes: tuple[NameAtom, ...] = ()
            elif not decl.tail:
                selected_exports = surface
                selected_scopes = tuple(module_scopes)
            else:
                selected_exports, selected_scopes = _selected_public_atoms(
                    decl.tail, surface, module_scopes
                )
                beneath = _tail_beneath_exposures(
                    decl.tail, module, module_exports, aliases, decl.span
                )
                for exposed, items in beneath.items():
                    decl_tail_beneath.setdefault(decl.node_id, {}).setdefault(
                        exposed, set()
                    ).update(items)
            hidden = {*hidden_exports, *unexported}
            hidden_scope_paths = set(hidden_scopes)
            if not decl.scope_path:
                root_hidden.update(module_exports[source] for source in hidden_exports)
            acc = accumulators.setdefault(module, _ContributionAccumulator())
            if decl.alias is None:
                acc.path_enabled = True
            brought = acc.routes.setdefault(decl.alias, _RouteAccumulator())
            brought.hidden.update(hidden, hidden_scope_paths)
            brought.decls.add(decl.node_id)
            reached_withheld = decl_withheld.setdefault(decl.node_id, {})
            for source, qname in module_exports.items():
                if source not in hidden:
                    acc.members[source] = qname
                    brought.members[source] = qname
                    brought.member_decls.setdefault(source, set()).add(decl.node_id)
                    kept = module_withheld.get(source, frozenset())
                    reached_withheld[qname] = reached_withheld.get(qname, kept) & kept
            visible_scope_paths = tuple(
                source for source in module_scopes if source not in hidden_scope_paths
            )
            brought.scope_paths.update(visible_scope_paths)
            for source in visible_scope_paths:
                scope_origins_by_route[(module, _path(source))] = module_scopes[source]
            exposures = _tail_exposures(decl, selected_exports)
            scope_exposures = _tail_exposures(decl, selected_scopes)
            removed = [
                *(exposed for exposed, source in exposures if source in hidden),
                *(exposed for exposed, source in scope_exposures if source in hidden_scope_paths),
            ]
            if removed:
                decl_hidden.setdefault(decl.node_id, set()).update(removed)
            for exposed, source in exposures:
                if source in hidden:
                    continue
                qname = module_exports[source]
                route = (module, _path(source))
                if decl.scope_path:
                    decl_bare.setdefault(decl.node_id, {}).setdefault(exposed, set()).add(qname)
                    decl_bare_routes.setdefault(decl.node_id, {}).setdefault(exposed, set()).add(
                        route
                    )
                else:
                    root_bare.setdefault(exposed, {}).setdefault(qname, set()).add(decl.node_id)
                    root_bare_routes.setdefault(exposed, set()).add(route)
            for exposed, source in scope_exposures:
                if source in hidden_scope_paths:
                    continue
                decl_scope_routes.setdefault(decl.node_id, {}).setdefault(exposed, set()).add(
                    (module, _path(source))
                )

    contributions: dict[ModuleId, ModuleContribution] = {}
    for module, acc in accumulators.items():
        contributions[module] = ModuleContribution(
            module,
            acc.members,
            acc.path_enabled,
            frozenset(route for route in acc.routes if route is not None),
            {route: surface.freeze() for route, surface in acc.routes.items()},
            exports[module],
        )
    return ImportEnv(
        contributions=contributions,
        unqualified={name: frozenset(qnames) for name, qnames in root_bare.items()},
        unqualified_decls={name: _frozen_decls(qnames) for name, qnames in root_bare.items()},
        decl_hiding={node_id: tuple(named) for node_id, named in decl_hiding.items()},
        decl_tail_beneath={
            node_id: {exposed: frozenset(items) for exposed, items in exposures.items()}
            for node_id, exposures in decl_tail_beneath.items()
        },
        decl_withheld={
            node_id: {qname: removed for qname, removed in reached.items() if removed}
            for node_id, reached in decl_withheld.items()
        },
        decl_bare={
            node_id: {atom: frozenset(qnames) for atom, qnames in members.items()}
            for node_id, members in decl_bare.items()
        },
        unqualified_routes={atom: frozenset(routes) for atom, routes in root_bare_routes.items()},
        decl_bare_routes={
            node_id: {atom: frozenset(routes) for atom, routes in members.items()}
            for node_id, members in decl_bare_routes.items()
        },
        decl_scope_routes={
            node_id: {atom: frozenset(routes) for atom, routes in members.items()}
            for node_id, members in decl_scope_routes.items()
        },
        decl_hidden={node_id: frozenset(atoms) for node_id, atoms in decl_hidden.items()},
        unqualified_hidden=frozenset(root_hidden),
        scope_origins_by_route=scope_origins_by_route,
    )


def _frozen_decls[K](decls: Mapping[K, set[int]]) -> dict[K, frozenset[int]]:
    return {key: frozenset(ids) for key, ids in decls.items()}


def qualifier_candidates(
    env: ImportEnv, qualifier: tuple[str, ...], *, anchored: bool
) -> tuple[ModuleId, ...]:
    return (env.anchored_routes if anchored else env.suffix_routes).get(qualifier, ())


def contribution_routes(
    contribution: ModuleContribution,
) -> tuple[tuple[tuple[str, ...], bool], ...]:
    routes: list[tuple[tuple[str, ...], bool]] = [
        ((alias,), False) for alias in sorted(contribution.aliases)
    ]
    if contribution.path_enabled:
        routes.extend(
            (contribution.module.segments[index:], False)
            for index in range(len(contribution.module.segments))
        )
        routes.append((contribution.module.segments, True))
    return tuple(routes)


def _matching_contribution_routes(
    contribution: ModuleContribution,
    qualifier: tuple[str, ...],
    *,
    anchored: bool,
) -> tuple[str | None, ...]:
    """Return matching alias names, using ``None`` for the module-path route."""
    routes: list[str | None] = []
    if not anchored and len(qualifier) == 1 and qualifier[0] in contribution.routes:
        routes.append(qualifier[0])
    path_matches = (
        qualifier == contribution.module.segments
        if anchored
        else any(
            contribution.module.segments[index:] == qualifier
            for index in range(len(contribution.module.segments))
        )
    )
    if contribution.path_enabled and path_matches:
        routes.append(None)
    return tuple(routes)


def _qualifier_routes(
    env: ImportEnv, qualifier: tuple[str, ...], *, anchored: bool
) -> Iterator[tuple[ModuleId, tuple[RouteSurface, ...]]]:
    """Yield each module *qualifier* names with the surfaces of its matching import routes."""
    for module in qualifier_candidates(env, qualifier, anchored=anchored):
        contribution = env.contributions[module]
        routes = _matching_contribution_routes(contribution, qualifier, anchored=anchored)
        yield module, tuple(contribution.routes.get(route, _NO_SURFACE) for route in routes)


def qualifier_members(
    env: ImportEnv, qualifier: tuple[str, ...], *, anchored: bool = False
) -> tuple[tuple[ModuleId, Mapping[NameAtom, QName]], ...]:
    """Return each imported route's public members without choosing a route."""
    return tuple(
        (
            module,
            MappingProxyType(
                {atom: qname for surface in surfaces for atom, qname in surface.members.items()}
            ),
        )
        for module, surfaces in _qualifier_routes(env, qualifier, anchored=anchored)
        if surfaces
    )


def qualifier_member_decls(
    env: ImportEnv, qualifier: tuple[str, ...], member: NameAtom, *, anchored: bool = False
) -> dict[QName, frozenset[int]]:
    """Return what each import route *qualifier* names reaches as *member*.

    Each with the import declarations contributing it on those routes.
    """
    found: dict[QName, frozenset[int]] = {}
    for _module, surfaces in _qualifier_routes(env, qualifier, anchored=anchored):
        for surface in surfaces:
            qname = surface.members.get(member)
            if qname is not None:
                found[qname] = found.get(qname, frozenset()) | surface.member_decls[member]
    return found


def qualifier_exposures(
    env: ImportEnv, qualifier: tuple[str, ...], *, anchored: bool = False
) -> Iterator[Exposure]:
    """Yield every member the import routes *qualifier* names reach."""
    exposed = {
        member: qualifier_member_decls(env, qualifier, member, anchored=anchored)
        for _module, members in qualifier_members(env, qualifier, anchored=anchored)
        for member in members
    }
    for member, reached in exposed.items():
        for qname, decls in reached.items():
            yield member, qname, decls


def unqualified_exposures(env: ImportEnv) -> Iterator[Exposure]:
    """Yield every root bare atom the root-position import tails expose."""
    for atom, reached in env.unqualified_decls.items():
        for qname, decls in reached.items():
            yield atom, qname, decls


def qualifier_scope_paths(
    env: ImportEnv, qualifier: tuple[str, ...], *, anchored: bool = False
) -> tuple[tuple[ModuleId, frozenset[NameAtom], frozenset[int]], ...]:
    """Return named-scope identities visible through each matching import route.

    Each with the import declarations of those routes.
    """
    return tuple(
        (
            module,
            frozenset(path for surface in surfaces for path in surface.scope_paths),
            frozenset(node_id for surface in surfaces for node_id in surface.decls),
        )
        for module, surfaces in _qualifier_routes(env, qualifier, anchored=anchored)
    )


def qualifier_hides(
    env: ImportEnv, qualifier: tuple[str, ...], member: NameAtom, *, anchored: bool = False
) -> bool:
    """Whether a ``hiding`` removed *member* from every route *qualifier* names that had it."""
    return any(
        member in surface.hidden
        for _module, surfaces in _qualifier_routes(env, qualifier, anchored=anchored)
        for surface in surfaces
    )
