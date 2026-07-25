"""Contribution-based import environments and qualified resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, TypeAlias

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import ImportDecl, ImportItem
from agm.agl.syntax.types import ImportMode

__all__ = [
    "ImportEnv",
    "ImportTarget",
    "ModuleContribution",
    "NameAtom",
    "PathAtom",
    "QName",
    "QualResolution",
    "QualResolutionAmbiguous",
    "QualResolutionFound",
    "QualResolutionMissingMember",
    "QualResolutionUnknownQualifier",
    "SingleTarget",
    "WildcardTarget",
    "ambiguous_qualification_message",
    "build_import_env",
    "contribution_routes",
    "private_missing_member_module",
    "qualification_repair_guidance",
    "qualifier_candidates",
    "qualifier_contributes",
    "render_qualifier",
    "resolve_qualified",
    "resolve_qualified_member",
    "try_resolve_qualified_member",
]

PathAtom: TypeAlias = tuple[str, ...]
# Root members retain their historical string representation at this boundary;
# scoped members use a structured tuple.  All policy operations normalize it.
NameAtom: TypeAlias = str | PathAtom
QName: TypeAlias = tuple[ModuleId, NameAtom]


def _path(atom: NameAtom) -> PathAtom:
    return (atom,) if isinstance(atom, str) else atom


def _atom(path: PathAtom) -> NameAtom:
    return path[0] if len(path) == 1 else path


def _item_path(item: ImportItem) -> PathAtom:
    return (*tuple(segment.name for segment in item.scope_path), item.name)


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
    """One imported module's selected, path-keyed contribution."""

    module: ModuleId
    members: Mapping[NameAtom, QName]
    bare_names: frozenset[NameAtom]
    path_enabled: bool
    aliases: frozenset[str]
    complete_public_set: bool = False

    def __post_init__(self) -> None:
        members: Mapping[NameAtom, QName] = MappingProxyType(
            {atom: self.members[atom] for atom in sorted(self.members, key=_path_sort_key)}
        )
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class ImportEnv:
    """Pure contribution environment, including structured public paths."""

    contributions: Mapping[ModuleId, ModuleContribution]
    unqualified: Mapping[NameAtom, frozenset[QName]]
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
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "unqualified", unqualified)
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
    complete_public_set: bool


def _matching_atoms(exports: Mapping[NameAtom, QName], prefix: PathAtom) -> tuple[NameAtom, ...]:
    return tuple(atom for atom in exports if _path(atom)[: len(prefix)] == prefix)


def _selected_atoms(
    decl: ImportDecl, module: ModuleId, exports: Mapping[NameAtom, QName]
) -> tuple[NameAtom, ...]:
    """Expand selection atoms to public paths, including complete subtrees."""
    if decl.mode is ImportMode.ALL:
        return tuple(exports)
    matched: dict[NameAtom, None] = {}
    for item in decl.items:
        paths = _matching_atoms(exports, _item_path(item))
        if not paths:
            rendered = "::".join(_item_path(item))
            raise AglScopeError(
                f"name {rendered!r} is not exported by module {module.display()!r}", span=decl.span
            )
        for atom in paths:
            matched[atom] = None
    if decl.mode is ImportMode.USING:
        return tuple(matched)
    return tuple(atom for atom in exports if atom not in matched)


def _exposed_atom(source: NameAtom, decl: ImportDecl) -> NameAtom:
    """Apply the matching ``using … as`` path re-rooting, if any."""
    source_path = _path(source)
    for item in decl.items:
        prefix = _item_path(item)
        if item.rename is not None and source_path[: len(prefix)] == prefix:
            return _atom((item.rename, *source_path[len(prefix) :]))
    return source


def _targets(target: ImportTarget) -> tuple[ModuleId, ...]:
    return (
        (target.module,)
        if isinstance(target, SingleTarget)
        else tuple(sorted(target.modules, key=ModuleId.path_str))
    )


def _add_member(
    acc: _ContributionAccumulator, exposed: NameAtom, qname: QName, decl: ImportDecl
) -> None:
    existing = acc.members.get(exposed)
    if existing is not None and existing != qname:
        raise AglScopeError(
            f"conflicting origins for exposed name {_path_sort_key(exposed)!r}", span=decl.span
        )
    acc.members[exposed] = qname


def build_import_env(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
    exports: Mapping[ModuleId, Mapping[NameAtom, QName]],
) -> ImportEnv:
    """Build contributions. Selection, hiding, and renaming operate on paths."""
    accumulators: dict[ModuleId, _ContributionAccumulator] = {}
    for decl in decls:
        for module in _targets(targets[decl.node_id]):
            acc = accumulators.setdefault(
                module, _ContributionAccumulator({}, set(), False, set(), False)
            )
            if decl.alias is None:
                acc.path_enabled = True
            else:
                acc.aliases.add(decl.alias)
            acc.complete_public_set |= decl.mode is ImportMode.ALL
            for source in _selected_atoms(decl, module, exports.get(module, {})):
                exposed = _exposed_atom(source, decl) if decl.mode is ImportMode.USING else source
                _add_member(acc, exposed, exports[module][source], decl)
                if decl.is_open or decl.mode is ImportMode.USING:
                    acc.bare_names.add(exposed)

    contributions: dict[ModuleId, ModuleContribution] = {}
    unqualified: dict[NameAtom, set[QName]] = {}
    for module, acc in accumulators.items():
        contribution = ModuleContribution(
            module,
            acc.members,
            frozenset(acc.bare_names),
            acc.path_enabled,
            frozenset(acc.aliases),
            acc.complete_public_set,
        )
        contributions[module] = contribution
        for name in contribution.bare_names:
            unqualified.setdefault(name, set()).add(contribution.members[name])
    return ImportEnv(
        contributions, {name: frozenset(qnames) for name, qnames in unqualified.items()}
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


def _member_qname(contribution: ModuleContribution, member: NameAtom) -> QName | None:
    member_path = _path(member)
    return next(
        (qname for exposed, qname in contribution.members.items() if _path(exposed) == member_path),
        None,
    )


def qualifier_contributes(
    env: ImportEnv, qualifier: tuple[str, ...], member: NameAtom, *, anchored: bool = False
) -> bool:
    return any(
        _member_qname(env.contributions[module], member) is not None
        for module in qualifier_candidates(env, qualifier, anchored=anchored)
    )


def private_missing_member_module(
    result: QualResolutionMissingMember, private_info: Mapping[QName, bool]
) -> ModuleId | None:
    requested = _path(result.member)
    for module in result.candidates:
        for (owner, atom), private in private_info.items():
            if owner == module and private and requested[: len(_path(atom))] == _path(atom):
                return module
    return None


def try_resolve_qualified_member(
    env: ImportEnv, qualifier: tuple[str, ...], member: NameAtom, *, anchored: bool = False
) -> QName | None:
    result = resolve_qualified(env, qualifier, member, anchored=anchored)
    return result.qname if isinstance(result, QualResolutionFound) else None


def resolve_qualified_member(
    env: ImportEnv,
    qualifier: tuple[str, ...],
    member: NameAtom,
    private_info: Mapping[QName, bool],
    *,
    anchored: bool = False,
    unknown_qualifier: Callable[[str], Exception],
    private_member: Callable[[ModuleId], Exception],
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
        module = private_missing_member_module(result, private_info)
        if module is not None:
            raise private_member(module)
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
        if (qname := _member_qname(env.contributions[module], member)) is not None
    )
    bare_atom = _atom((*qualifier, *_path(member)))
    bare_qnames = frozenset() if anchored else env.unqualified.get(bare_atom, frozenset())

    if route_members:
        route_qnames = {qname for _module, qname in route_members}
        if len(route_members) > 1 or len(route_qnames | bare_qnames) > 1:
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
