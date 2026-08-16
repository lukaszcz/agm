"""Contribution-based import environments and qualified resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import AglScopeError
from agm.agl.scope.symbols import import_item_path as _item_path
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.scope.symbols import to_bare_path as _path
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    ImportDecl,
    ImportItem,
    QualifierChain,
    RecordDef,
)
from agm.agl.syntax.nodes import TypeAlias as TypeAliasDecl
from agm.agl.syntax.spans import SourceSpan

__all__ = [
    "EMPTY_IMPORT_ENV",
    "ImportEnv",
    "ImportTarget",
    "BareRoute",
    "ModuleContribution",
    "NameAtom",
    "PathAtom",
    "QName",
    "QualResolution",
    "QualResolutionAmbiguous",
    "QualResolutionFound",
    "QualResolutionMissingMember",
    "QualResolutionUnknownQualifier",
    "ScopeOrigins",
    "SingleTarget",
    "WildcardTarget",
    "ambiguous_qualification_message",
    "build_import_env",
    "contribution_routes",
    "qualification_repair_guidance",
    "qualifier_candidates",
    "qualifier_contributes",
    "qualifier_members",
    "render_qualifier",
    "resolve_alias_target",
    "resolve_qualified",
    "resolve_qualified_member",
    "try_resolve_qualified_member",
]

PathAtom: TypeAlias = tuple[str, ...]
# Root members retain their historical string representation at this boundary;
# scoped members use a structured tuple.  All policy operations normalize it.
NameAtom: TypeAlias = str | PathAtom
QName: TypeAlias = tuple[ModuleId, NameAtom]
ScopeOrigins: TypeAlias = frozenset[QName]
BareRoute: TypeAlias = tuple[ModuleId, PathAtom]


def _path_sort_key(atom: NameAtom) -> str:
    return "::".join(_path(atom))


def render_qualifier(qualifier: tuple[str, ...], *, anchored: bool = False) -> str:
    """Render a source qualifier with its slash route and optional anchor."""
    return ("/" if anchored else "") + "/".join(qualifier)


@dataclass(frozen=True, slots=True)
class SingleTarget:
    """The import resolves to exactly one module."""

    module: ModuleId


@dataclass(frozen=True, slots=True)
class WildcardTarget:
    """The wildcard import expands to these modules."""

    modules: frozenset[ModuleId]


ImportTarget = SingleTarget | WildcardTarget


def qualification_repair_guidance() -> str:
    """Return the common, source-level repairs for a qualifier ambiguity."""
    return (
        "Use a :: anchor to select the current module, hiding to remove a conflicting member, "
        "a longer suffix or a /-anchored path to select a module, or as to give one import "
        "a distinct name."
    )


def ambiguous_qualification_message(
    qualifier: tuple[str, ...],
    member: NameAtom,
    candidates: tuple[ModuleId, ...],
    *,
    anchored: bool = False,
) -> str:
    """Render the common repair-oriented diagnostic for a shared verdict."""
    rendered = render_qualifier(qualifier, anchored=anchored)
    paths = ", ".join(module.display() for module in candidates)
    name = "::".join(_path(member))
    message = f"'{rendered}::{name}' is ambiguous across imported modules: {paths}."
    return f"{message} {qualification_repair_guidance()}"


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
class ModuleContribution:
    """One imported module's route-keyed public contribution."""

    module: ModuleId
    members: Mapping[NameAtom, QName]
    bare_names: frozenset[NameAtom]
    path_enabled: bool
    aliases: frozenset[str]
    path_members: Mapping[NameAtom, QName] = field(default_factory=dict)
    alias_members: Mapping[str, Mapping[NameAtom, QName]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        members: Mapping[NameAtom, QName] = MappingProxyType(
            {atom: self.members[atom] for atom in sorted(self.members, key=_path_sort_key)}
        )
        path_members: Mapping[NameAtom, QName] = MappingProxyType(
            {
                atom: self.path_members[atom]
                for atom in sorted(self.path_members, key=_path_sort_key)
            }
        )
        alias_members: Mapping[str, Mapping[NameAtom, QName]] = MappingProxyType(
            {
                alias: MappingProxyType(
                    {
                        atom: self.alias_members[alias][atom]
                        for atom in sorted(self.alias_members[alias], key=_path_sort_key)
                    }
                )
                for alias in sorted(self.alias_members)
            }
        )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "path_members", path_members)
        object.__setattr__(self, "alias_members", alias_members)


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
    the name's first use, not raised here. ``unqualified_scope_routes`` and
    ``decl_bare_scope_routes`` carry the parallel namespace-only contribution
    for scopes that have no declaration member to put in a bare table.
    """

    contributions: Mapping[ModuleId, ModuleContribution]
    unqualified: Mapping[NameAtom, frozenset[QName]]
    decl_bare: Mapping[int, Mapping[NameAtom, frozenset[QName]]] = field(default_factory=dict)
    unqualified_routes: Mapping[NameAtom, frozenset[BareRoute]] = field(default_factory=dict)
    decl_bare_routes: Mapping[int, Mapping[NameAtom, frozenset[BareRoute]]] = field(
        default_factory=dict
    )
    unqualified_scope_routes: Mapping[NameAtom, frozenset[BareRoute]] = field(default_factory=dict)
    decl_bare_scope_routes: Mapping[int, Mapping[NameAtom, frozenset[BareRoute]]] = field(
        default_factory=dict
    )
    facade_aliases: Mapping[str, frozenset[ModuleId]] = field(default_factory=dict)
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
        unqualified_scope_routes: Mapping[NameAtom, frozenset[BareRoute]] = MappingProxyType(
            dict(self.unqualified_scope_routes)
        )
        decl_bare_scope_routes: Mapping[int, Mapping[NameAtom, frozenset[BareRoute]]] = (
            MappingProxyType(
                {
                    node_id: MappingProxyType(dict(routes))
                    for node_id, routes in self.decl_bare_scope_routes.items()
                }
            )
        )
        object.__setattr__(self, "decl_bare", decl_bare)
        object.__setattr__(self, "unqualified_routes", unqualified_routes)
        object.__setattr__(self, "decl_bare_routes", decl_bare_routes)
        object.__setattr__(self, "unqualified_scope_routes", unqualified_scope_routes)
        object.__setattr__(self, "decl_bare_scope_routes", decl_bare_scope_routes)
        facade_aliases: Mapping[str, frozenset[ModuleId]] = MappingProxyType(
            {alias: self.facade_aliases[alias] for alias in sorted(self.facade_aliases)}
        )
        object.__setattr__(self, "facade_aliases", facade_aliases)
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


@dataclass(frozen=True, slots=True)
class QualResolutionFound:
    module: ModuleId
    qname: QName


@dataclass(frozen=True, slots=True)
class QualResolutionUnknownQualifier:
    qualifier: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualResolutionMissingMember:
    qualifier: tuple[str, ...]
    member: NameAtom
    candidates: tuple[ModuleId, ...]


@dataclass(frozen=True, slots=True)
class QualResolutionAmbiguous:
    qualifier: tuple[str, ...]
    member: NameAtom
    candidates: tuple[ModuleId, ...]


QualResolution = (
    QualResolutionFound
    | QualResolutionUnknownQualifier
    | QualResolutionMissingMember
    | QualResolutionAmbiguous
)


@dataclass(slots=True)
class _ContributionAccumulator:
    members: dict[NameAtom, QName]
    bare_names: set[NameAtom]
    path_enabled: bool
    aliases: set[str]
    path_members: dict[NameAtom, QName]
    alias_members: dict[str, dict[NameAtom, QName]]


def _matching_atoms(exports: Mapping[NameAtom, object], prefix: PathAtom) -> tuple[NameAtom, ...]:
    return tuple(atom for atom in exports if _path(atom)[: len(prefix)] == prefix)


def _selected_public_atoms(
    items: tuple[ImportItem, ...],
    module: ModuleId,
    exports: Mapping[NameAtom, QName],
    scope_exports: Mapping[NameAtom, ScopeOrigins],
    span: SourceSpan,
) -> tuple[tuple[NameAtom, ...], tuple[NameAtom, ...]]:
    """Expand selected declaration atoms and independent scope identities."""
    matched_exports: dict[NameAtom, None] = {}
    matched_scopes: dict[NameAtom, None] = {}
    for item in items:
        prefix = _item_path(item)
        declarations = _matching_atoms(exports, prefix)
        scopes = _matching_atoms(scope_exports, prefix)
        if not declarations and not scopes:
            rendered = "::".join(prefix)
            raise AglScopeError(
                f"name {rendered!r} is not exported by module {module.display()!r}", span=span
            )
        for atom in declarations:
            matched_exports[atom] = None
        for atom in scopes:
            matched_scopes[atom] = None
    return tuple(matched_exports), tuple(matched_scopes)


def _tail_exposures(
    decl: ImportDecl,
    exports: Mapping[NameAtom, QName],
    hidden: set[NameAtom],
    selected: tuple[NameAtom, ...],
) -> tuple[tuple[NameAtom, QName, NameAtom], ...]:
    """Return the implicit use-style bare declaration contribution of a tailed import."""
    if decl.tail is None:
        return ()
    result: list[tuple[NameAtom, QName, NameAtom]] = []
    for source in selected:
        if source in hidden:
            continue
        qname = exports[source]
        result.append((source, qname, source))
        source_path = _path(source)
        for item in decl.tail:
            prefix = _item_path(item)
            if item.rename is not None and source_path[: len(prefix)] == prefix:
                result.append((_atom((item.rename, *source_path[len(prefix) :])), qname, source))
    return tuple(result)


def _tail_scope_exposures(
    decl: ImportDecl,
    hidden: set[NameAtom],
    selected: tuple[NameAtom, ...],
) -> tuple[tuple[NameAtom, NameAtom], ...]:
    """Return bare scope spellings paired with their module-surface paths."""
    if decl.tail is None:
        return ()
    result: dict[tuple[NameAtom, NameAtom], None] = {}
    for source in selected:
        if source in hidden:
            continue
        result[(source, source)] = None
        source_path = _path(source)
        for item in decl.tail:
            prefix = _item_path(item)
            if item.rename is not None and source_path[: len(prefix)] == prefix:
                exposed = _atom((item.rename, *source_path[len(prefix) :]))
                result[(exposed, source)] = None
    return tuple(result)


def _targets(target: ImportTarget) -> tuple[ModuleId, ...]:
    return (
        (target.module,)
        if isinstance(target, SingleTarget)
        else tuple(sorted(target.modules, key=ModuleId.path_str))
    )


def _merge_member(members: dict[NameAtom, QName], exposed: NameAtom, qname: QName) -> None:
    members[exposed] = qname


def build_import_env(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
    exports: Mapping[ModuleId, Mapping[NameAtom, QName]],
    scope_exports: Mapping[ModuleId, Mapping[NameAtom, ScopeOrigins]] | None = None,
) -> ImportEnv:
    """Build route and implicit-tail contributions for import declarations.

    A region-scoped declaration still contributes qualifier routes module-wide,
    but its implicit tail's bare atoms are recorded in ``decl_bare`` so the
    scope pass can narrow them to that region.
    """
    accumulators: dict[ModuleId, _ContributionAccumulator] = {}
    root_bare: dict[NameAtom, set[QName]] = {}
    decl_bare: dict[int, dict[NameAtom, set[QName]]] = {}
    root_bare_routes: dict[NameAtom, set[BareRoute]] = {}
    decl_bare_routes: dict[int, dict[NameAtom, set[BareRoute]]] = {}
    root_scope_routes: dict[NameAtom, set[BareRoute]] = {}
    decl_scope_routes: dict[int, dict[NameAtom, set[BareRoute]]] = {}
    facade_aliases: dict[str, set[ModuleId]] = {}
    public_scopes = scope_exports or {}
    for decl in decls:
        target = targets[decl.node_id]
        modules = _targets(target)
        if decl.alias is not None and (decl.wildcard_origin or isinstance(target, WildcardTarget)):
            facade_aliases.setdefault(decl.alias, set()).update(modules)
        for module in modules:
            module_exports = exports.get(module, {})
            module_scopes = public_scopes.get(module, {})
            hidden_exports, hidden_scopes = _selected_public_atoms(
                decl.hidden, module, module_exports, module_scopes, decl.span
            )
            if decl.tail is None:
                selected_exports: tuple[NameAtom, ...] = ()
                selected_scopes: tuple[NameAtom, ...] = ()
            elif not decl.tail:
                selected_exports = tuple(module_exports)
                selected_scopes = tuple(module_scopes)
            else:
                selected_exports, selected_scopes = _selected_public_atoms(
                    decl.tail, module, module_exports, module_scopes, decl.span
                )
            hidden = set(hidden_exports)
            acc = accumulators.setdefault(
                module, _ContributionAccumulator({}, set(), False, set(), {}, {})
            )
            route_members = (
                acc.path_members
                if decl.alias is None
                else acc.alias_members.setdefault(decl.alias, {})
            )
            if decl.alias is None:
                acc.path_enabled = True
            else:
                acc.aliases.add(decl.alias)
            for source, qname in module_exports.items():
                if source not in hidden:
                    _merge_member(acc.members, source, qname)
                    _merge_member(route_members, source, qname)
            for exposed, qname, source in _tail_exposures(
                decl, module_exports, hidden, selected_exports
            ):
                route = (module, _path(source))
                if decl.scope_path:
                    decl_bare.setdefault(decl.node_id, {}).setdefault(exposed, set()).add(qname)
                    decl_bare_routes.setdefault(decl.node_id, {}).setdefault(exposed, set()).add(
                        route
                    )
                else:
                    root_bare.setdefault(exposed, set()).add(qname)
                    root_bare_routes.setdefault(exposed, set()).add(route)
                acc.bare_names.add(exposed)
            for exposed, source in _tail_scope_exposures(decl, set(hidden_scopes), selected_scopes):
                route = (module, _path(source))
                destination = (
                    decl_scope_routes.setdefault(decl.node_id, {})
                    if decl.scope_path
                    else root_scope_routes
                )
                destination.setdefault(exposed, set()).add(route)

    contributions: dict[ModuleId, ModuleContribution] = {}
    for module, acc in accumulators.items():
        contributions[module] = ModuleContribution(
            module,
            acc.members,
            frozenset(acc.bare_names),
            acc.path_enabled,
            frozenset(acc.aliases),
            acc.path_members,
            acc.alias_members,
        )
    return ImportEnv(
        contributions=contributions,
        unqualified={name: frozenset(qnames) for name, qnames in root_bare.items()},
        decl_bare={
            node_id: {atom: frozenset(qnames) for atom, qnames in members.items()}
            for node_id, members in decl_bare.items()
        },
        unqualified_routes={atom: frozenset(routes) for atom, routes in root_bare_routes.items()},
        decl_bare_routes={
            node_id: {atom: frozenset(routes) for atom, routes in members.items()}
            for node_id, members in decl_bare_routes.items()
        },
        unqualified_scope_routes={
            atom: frozenset(routes) for atom, routes in root_scope_routes.items()
        },
        decl_bare_scope_routes={
            node_id: {atom: frozenset(routes) for atom, routes in members.items()}
            for node_id, members in decl_scope_routes.items()
        },
        facade_aliases={alias: frozenset(modules) for alias, modules in facade_aliases.items()},
    )


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


def _member_qname(
    contribution: ModuleContribution,
    qualifier: tuple[str, ...],
    member: NameAtom,
    *,
    anchored: bool,
) -> QName | None:
    """Find a member through the declaration routes named by *qualifier*."""
    candidates: list[Mapping[NameAtom, QName]] = []
    if not anchored and len(qualifier) == 1:
        alias_members = contribution.alias_members.get(qualifier[0])
        if alias_members is not None:
            candidates.append(alias_members)
    path_matches = (
        qualifier == contribution.module.segments
        if anchored
        else any(
            contribution.module.segments[index:] == qualifier
            for index in range(len(contribution.module.segments))
        )
    )
    if contribution.path_enabled and path_matches:
        candidates.append(contribution.path_members)
    member_path = _path(member)
    qnames = {
        qname
        for members in candidates
        for exposed, qname in members.items()
        if _path(exposed) == member_path
    }
    return next(iter(qnames)) if len(qnames) == 1 else None


def qualifier_members(
    env: ImportEnv, qualifier: tuple[str, ...], *, anchored: bool = False
) -> tuple[tuple[ModuleId, Mapping[NameAtom, QName]], ...]:
    """Return each imported route's public members without choosing a route."""
    members: list[tuple[ModuleId, Mapping[NameAtom, QName]]] = []
    for module in qualifier_candidates(env, qualifier, anchored=anchored):
        contribution = env.contributions[module]
        maps: list[Mapping[NameAtom, QName]] = []
        if not anchored and len(qualifier) == 1:
            alias_members = contribution.alias_members.get(qualifier[0])
            if alias_members is not None:
                maps.append(alias_members)
        path_matches = (
            qualifier == contribution.module.segments
            if anchored
            else any(
                contribution.module.segments[index:] == qualifier
                for index in range(len(contribution.module.segments))
            )
        )
        if contribution.path_enabled and path_matches:
            maps.append(contribution.path_members)
        merged = {atom: qname for member_map in maps for atom, qname in member_map.items()}
        if maps:
            members.append((module, MappingProxyType(merged)))
    return tuple(members)


def qualifier_contributes(
    env: ImportEnv, qualifier: tuple[str, ...], member: NameAtom, *, anchored: bool = False
) -> bool:
    return any(
        _member_qname(env.contributions[module], qualifier, member, anchored=anchored) is not None
        for module in qualifier_candidates(env, qualifier, anchored=anchored)
    )


def try_resolve_qualified_member(
    env: ImportEnv, qualifier: tuple[str, ...], member: NameAtom, *, anchored: bool = False
) -> QName | None:
    """Resolve ``qualifier::member``, returning ``None`` for any non-unique verdict."""
    result = resolve_qualified(env, qualifier, member, anchored=anchored)
    return result.qname if isinstance(result, QualResolutionFound) else None


def resolve_alias_target(
    name: str,
    qualifier: QualifierChain | None,
    *,
    self_module_id: ModuleId | None,
    import_env: ImportEnv,
    all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAliasDecl],
    scope_path: PathAtom = (),
) -> RecordDef | EnumDef | ExceptionDef | TypeAliasDecl | None:
    """Resolve one type-alias target reference through a module's import environment.

    Shared by the scope resolver and the program-level cross-module constructor
    pre-pass so both judge a type alias's constructibility (see
    :func:`~agm.agl.scope.symbols.alias_denotes_constructible_type`) the same
    way for a target reached through an import rather than a same-module
    declaration.

    For an unqualified *name*, tries *self_module_id*'s own declaration under
    *scope_path* first (when *self_module_id* is given — a caller that
    already checked richer local state passes ``self_module_id=None`` to skip
    this step), then the unqualified name exposed by *import_env*'s import
    tails. For a qualified *name*, resolves through the ordinary
    qualified-member route.

    Returns ``None`` for anything it cannot resolve — an ambiguous
    unqualified name or an unknown route — which the caller treats as
    "presumed constructible".
    """
    if qualifier is None or not qualifier.segments:
        if self_module_id is not None:
            local = all_public_types.get((self_module_id, _atom((*scope_path, name))))
            if local is not None:
                return local
        qnames = import_env.unqualified.get(name)
        if qnames is None or len(qnames) != 1:
            return None
        (qname,) = qnames
        return all_public_types.get(qname)
    qualified = try_resolve_qualified_member(
        import_env,
        qualifier.route_segments,
        name,
        anchored=qualifier.anchored,
    )
    if qualified is None:
        return None
    return all_public_types.get(qualified)


def resolve_qualified_member(
    env: ImportEnv,
    qualifier: tuple[str, ...],
    member: NameAtom,
    *,
    anchored: bool = False,
    unknown_qualifier: Callable[[str], Exception],
    missing_member: Callable[[str], Exception],
    ambiguous: Callable[[str], Exception],
) -> QName:
    result = resolve_qualified(env, qualifier, member, anchored=anchored)
    if isinstance(result, QualResolutionFound):
        return result.qname
    rendered = render_qualifier(qualifier, anchored=anchored)
    if isinstance(result, QualResolutionUnknownQualifier):
        raise unknown_qualifier(rendered)
    if isinstance(result, QualResolutionMissingMember):
        raise missing_member(rendered)
    raise ambiguous(
        ambiguous_qualification_message(qualifier, member, result.candidates, anchored=anchored)
    )


def resolve_qualified(
    env: ImportEnv, qualifier: tuple[str, ...], member: NameAtom, *, anchored: bool = False
) -> QualResolution:
    candidates = qualifier_candidates(env, qualifier, anchored=anchored)
    route_members = tuple(
        (module, qname)
        for module in candidates
        if (qname := _member_qname(env.contributions[module], qualifier, member, anchored=anchored))
        is not None
    )
    bare_atom = _atom((*qualifier, *_path(member)))
    bare_qnames = frozenset() if anchored else env.unqualified.get(bare_atom, frozenset())

    if route_members:
        route_qnames = {qname for _module, qname in route_members}
        if len(route_qnames | bare_qnames) > 1:
            modules = {module for module, _qname in route_members}
            modules.update(module for module, _atom in bare_qnames)
            return QualResolutionAmbiguous(
                qualifier, member, tuple(sorted(modules, key=ModuleId.path_str))
            )
        module, qname = route_members[0]
        return QualResolutionFound(module, qname)
    if bare_qnames:
        if len(bare_qnames) > 1:
            return QualResolutionAmbiguous(
                qualifier,
                member,
                tuple(sorted((module for module, _atom in bare_qnames), key=ModuleId.path_str)),
            )
        qname = next(iter(bare_qnames))
        return QualResolutionFound(qname[0], qname)
    if candidates:
        return QualResolutionMissingMember(qualifier, member, candidates)
    return QualResolutionUnknownQualifier(qualifier)
