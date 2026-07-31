"""Contribution-based import environments and qualified resolution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    ImportDecl,
    ImportItem,
    QualifierChain,
    RecordDef,
)
from agm.agl.syntax.nodes import TypeAlias as TypeAliasDecl
from agm.agl.syntax.types import ImportMode

__all__ = [
    "EMPTY_IMPORT_ENV",
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
    "qualification_repair_guidance",
    "qualifier_candidates",
    "qualifier_contributes",
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
    """Pure contribution environment, including structured public paths.

    ``decl_bare`` holds, per region-scoped import declaration (keyed by its
    ``node_id``), exactly the atoms *that declaration alone* contributes
    bare. It is kept separate from ``unqualified`` -- the module-wide bare
    table a root-position ``open import`` feeds -- so a scoped ``import``'s
    bare names can be snapshotted onto its own region instead of leaking to
    the whole module; the module-wide qualifier route it also establishes is
    unaffected and still flows through ``contributions``. Each atom maps to
    every origin it draws from a wildcard's expansion, exactly like
    ``unqualified``: two modules exposing the same bare name is deferred to
    the name's first use, not raised here.
    """

    contributions: Mapping[ModuleId, ModuleContribution]
    unqualified: Mapping[NameAtom, frozenset[QName]]
    decl_bare: Mapping[int, Mapping[NameAtom, frozenset[QName]]] = field(default_factory=dict)
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
        object.__setattr__(self, "decl_bare", decl_bare)
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


def _merge_member(
    members: dict[NameAtom, QName], exposed: NameAtom, qname: QName, decl: ImportDecl
) -> None:
    existing = members.get(exposed)
    if existing is not None and existing != qname:
        raise AglScopeError(
            f"conflicting origins for exposed name {_path_sort_key(exposed)!r}", span=decl.span
        )
    members[exposed] = qname


def build_import_env(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
    exports: Mapping[ModuleId, Mapping[NameAtom, QName]],
) -> ImportEnv:
    """Build contributions. Selection, hiding, and renaming operate on paths.

    A region-scoped declaration (``decl.scope_path``) still contributes its
    members to the module-wide qualifier route like any other import, but its
    bare names are kept out of the shared ``unqualified`` table and recorded
    per declaration in ``decl_bare`` instead, so the scope pass can narrow
    them to their own region rather than exposing them module-wide.
    """
    accumulators: dict[ModuleId, _ContributionAccumulator] = {}
    decl_bare: dict[int, dict[NameAtom, set[QName]]] = {}
    for decl in decls:
        contributes_bare = decl.is_open or decl.mode is ImportMode.USING
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
                qname = exports[module][source]
                _merge_member(acc.members, exposed, qname, decl)
                if contributes_bare:
                    if decl.scope_path:
                        decl_bare.setdefault(decl.node_id, {}).setdefault(exposed, set()).add(qname)
                    else:
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
        contributions,
        {name: frozenset(qnames) for name, qnames in unqualified.items()},
        {
            node_id: {atom: frozenset(qnames) for atom, qnames in members.items()}
            for node_id, members in decl_bare.items()
        },
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
    import_env: ImportEnv | None,
    all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAliasDecl] | None,
    scope_path: PathAtom = (),
) -> RecordDef | EnumDef | ExceptionDef | TypeAliasDecl | None:
    """Resolve one type-alias target reference through a module's import environment.

    Shared by the scope resolver and the program-level cross-module constructor
    pre-pass so both judge a type alias's constructibility (see
    :func:`~agm.agl.scope.symbols.alias_denotes_constructible_type`) the same
    way for a target reached through an import rather than a same-module
    declaration.

    For an unqualified *name*, tries *self_module_id*'s own declaration under
    *scope_path* first (when both *self_module_id* and *all_public_types* are
    given — a caller that already checked richer local state passes
    ``self_module_id=None`` to skip this step), then the
    unqualified name exposed by *import_env*'s open imports. For a qualified
    *name*, resolves through the ordinary qualified-member route.

    Returns ``None`` for anything it cannot resolve — no import environment or
    public-types table, an ambiguous unqualified name, or an unknown route —
    which the caller treats as "presumed constructible".
    """
    if qualifier is None or not qualifier.segments:
        if self_module_id is not None and all_public_types is not None:
            local = all_public_types.get((self_module_id, _atom((*scope_path, name))))
            if local is not None:
                return local
        if import_env is None or all_public_types is None:
            return None
        qnames = import_env.unqualified.get(name)
        if qnames is None or len(qnames) != 1:
            return None
        (qname,) = qnames
        return all_public_types.get(qname)
    if import_env is None or all_public_types is None:
        return None
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
