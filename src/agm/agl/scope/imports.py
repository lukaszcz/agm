"""Contribution-based import environments and qualified resolution."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import ImportDecl
from agm.agl.syntax.types import ImportMode

QName = tuple[ModuleId, str]


@dataclass(frozen=True, slots=True)
class SingleTarget:
    """The import resolves to exactly one module."""

    module: ModuleId


@dataclass(frozen=True, slots=True)
class WildcardTarget:
    """The wildcard import expands to these modules."""

    modules: frozenset[ModuleId]


ImportTarget = SingleTarget | WildcardTarget


def _module_path(module: ModuleId) -> str:
    return module.path_str()


def _module_sort_key(module: ModuleId) -> tuple[str, ...]:
    return module.segments


def render_qualifier(qualifier: tuple[str, ...], *, anchored: bool = False) -> str:
    """Render a source qualifier with its slash route and optional anchor."""
    return ("/" if anchored else "") + "/".join(qualifier)


def qualification_repair_guidance() -> str:
    """Return the common, source-level repairs for a qualifier ambiguity."""
    return (
        "Use hiding to remove a conflicting member, a longer suffix or a /-anchored path "
        "to select a module, or as to give one import a distinct name."
    )


def ambiguous_qualification_message(
    qualifier: tuple[str, ...],
    member: str,
    candidates: tuple[ModuleId, ...],
    *,
    anchored: bool = False,
) -> str:
    """Render the common repair-oriented diagnostic for a shared verdict."""
    rendered = render_qualifier(qualifier, anchored=anchored)
    paths = ", ".join(module.path_str() for module in candidates)
    message = f"'{rendered}::{member}' is ambiguous across imported modules: {paths}."
    return f"{message} {qualification_repair_guidance()}"


def _build_rename_map(decl: ImportDecl) -> dict[str, str]:
    """Map selected source names to their canonical exposed names."""
    if decl.mode is not ImportMode.USING:
        return {}
    return {item.name: item.rename for item in decl.items if item.rename is not None}


def _exposed_name(source_name: str, rename_map: Mapping[str, str]) -> str:
    """Return the canonical name exposed for *source_name*."""
    return rename_map.get(source_name, source_name)


# ---------------------------------------------------------------------------
# Contribution import model
# ---------------------------------------------------------------------------


def _immutable_members(members: Mapping[str, QName]) -> Mapping[str, QName]:
    """Copy *members* into a deterministically ordered immutable mapping."""
    return MappingProxyType({name: qname for name, qname in sorted(members.items())})


def _immutable_unqualified(
    unqualified: Mapping[str, frozenset[QName]],
) -> Mapping[str, frozenset[QName]]:
    """Copy bare-name candidates into a deterministically ordered immutable mapping."""
    return MappingProxyType(
        {name: frozenset(qnames) for name, qnames in sorted(unqualified.items())}
    )


def _immutable_contributions(
    contributions: Mapping[ModuleId, "ModuleContribution"],
) -> Mapping[ModuleId, "ModuleContribution"]:
    """Copy module contributions into a deterministic immutable mapping."""
    return MappingProxyType(
        {module: contributions[module] for module in sorted(contributions, key=_module_sort_key)}
    )


def _immutable_names(names: frozenset[str]) -> frozenset[str]:
    """Copy a set of names into an immutable set."""
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class ModuleContribution:
    """The merged contribution of one imported module.

    ``members`` is the module's unioned contributed set S, keyed by its
    canonical exposed names. ``bare_names`` is the subset injected into the
    unqualified namespace by ``using`` or an ``open`` import. ``path_enabled``
    records whether a plain import makes the module available to suffix and
    anchored qualification; aliases are instead listed in ``aliases``.
    """

    module: ModuleId
    members: Mapping[str, QName]
    bare_names: frozenset[str]
    path_enabled: bool
    aliases: frozenset[str]

    def __post_init__(self) -> None:
        members = _immutable_members(self.members)
        bare_names = _immutable_names(self.bare_names)
        aliases = _immutable_names(self.aliases)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "bare_names", bare_names)
        object.__setattr__(self, "aliases", aliases)


@dataclass(frozen=True, slots=True)
class ImportEnv:
    """Pure contribution import environment.

    Contributions are indexed by imported module so repeated declarations union
    their selected sets. The unqualified map retains lazy-clash candidates.
    """

    contributions: Mapping[ModuleId, ModuleContribution]
    unqualified: Mapping[str, frozenset[QName]]

    def __post_init__(self) -> None:
        contributions = _immutable_contributions(self.contributions)
        unqualified = _immutable_unqualified(self.unqualified)
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "unqualified", unqualified)


@dataclass(frozen=True, slots=True)
class QualResolutionFound:
    """A qualified reference resolved to one contributed member."""

    module: ModuleId
    qname: QName


@dataclass(frozen=True, slots=True)
class QualResolutionUnknownQualifier:
    """No imported route matches the qualifier."""

    qualifier: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualResolutionMissingMember:
    """The qualifier matches imports, but none contributes the requested member."""

    qualifier: tuple[str, ...]
    member: str
    candidates: tuple[ModuleId, ...]


@dataclass(frozen=True, slots=True)
class QualResolutionAmbiguous:
    """Several matching modules contribute the requested member."""

    qualifier: tuple[str, ...]
    member: str
    candidates: tuple[ModuleId, ...]


QualResolution = (
    QualResolutionFound
    | QualResolutionUnknownQualifier
    | QualResolutionMissingMember
    | QualResolutionAmbiguous
)


@dataclass(slots=True)
class _ContributionAccumulator:
    """Mutable builder state for one module's merged contribution."""

    members: dict[str, QName]
    bare_names: set[str]
    path_enabled: bool
    aliases: set[str]


def _compute_set(
    decl: ImportDecl,
    module: ModuleId,
    module_exports: Mapping[str, QName],
) -> frozenset[str]:
    """Select one declaration's S for one target-module expansion.

    Wildcard imports expand per module, so a ``using`` or ``hiding`` item
    must be public in every expanded module just as it must be
    for a non-wildcard import.
    """
    exported_names = frozenset(module_exports)
    if decl.mode == ImportMode.ALL:
        return exported_names

    selected_names = frozenset(item.name for item in decl.items)
    for name in selected_names:
        if name not in exported_names:
            raise AglScopeError(
                f"name {name!r} is not exported by module {_module_path(module)!r}",
                span=decl.span,
            )
    if decl.mode == ImportMode.USING:
        return selected_names
    return exported_names - selected_names


def _targets(target: ImportTarget) -> tuple[ModuleId, ...]:
    """Return a deterministic per-module expansion of an import target."""
    if isinstance(target, SingleTarget):
        return (target.module,)
    return tuple(sorted(target.modules, key=_module_sort_key))


def _add_member(
    accumulator: _ContributionAccumulator,
    exposed_name: str,
    qname: QName,
    decl: ImportDecl,
) -> None:
    """Merge a canonical exposed member, rejecting distinct origins."""
    existing = accumulator.members.get(exposed_name)
    if existing is not None and existing != qname:
        raise AglScopeError(
            f"conflicting origins for exposed name {exposed_name!r}"
            f" ({_module_path(existing[0])!r} and {_module_path(qname[0])!r})",
            span=decl.span,
        )
    accumulator.members[exposed_name] = qname


def build_import_env(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
    exports: Mapping[ModuleId, Mapping[str, QName]],
) -> ImportEnv:
    """Build the contribution environment from resolved imports.

    Plain imports contribute qualified members only; ``using`` and ``open``
    declarations inject their selected members into the bare namespace.
    """
    accumulators: dict[ModuleId, _ContributionAccumulator] = {}

    for decl in decls:
        is_open = decl.is_open
        if is_open and decl.mode == ImportMode.USING:
            raise AglScopeError("open imports cannot also use a using clause", span=decl.span)

        for module in _targets(targets[decl.node_id]):
            accumulator = accumulators.setdefault(
                module,
                _ContributionAccumulator(
                    members={}, bare_names=set(), path_enabled=False, aliases=set()
                ),
            )
            if decl.alias is None:
                accumulator.path_enabled = True
            else:
                accumulator.aliases.add(decl.alias)

            module_exports = exports.get(module, {})
            selected = _compute_set(decl, module, module_exports)
            rename_map = _build_rename_map(decl)
            inject_bare = is_open or decl.mode == ImportMode.USING
            for source_name in sorted(selected):
                exposed_name = _exposed_name(source_name, rename_map)
                _add_member(accumulator, exposed_name, module_exports[source_name], decl)
                if inject_bare:
                    accumulator.bare_names.add(exposed_name)

    contributions: dict[ModuleId, ModuleContribution] = {}
    unqualified_acc: dict[str, set[QName]] = {}
    for module in sorted(accumulators, key=_module_sort_key):
        accumulator = accumulators[module]
        contribution = ModuleContribution(
            module=module,
            members=dict(accumulator.members),
            bare_names=frozenset(accumulator.bare_names),
            path_enabled=accumulator.path_enabled,
            aliases=frozenset(accumulator.aliases),
        )
        contributions[module] = contribution
        for name in sorted(contribution.bare_names):
            unqualified_acc.setdefault(name, set()).add(contribution.members[name])

    return ImportEnv(
        contributions=contributions,
        unqualified={name: frozenset(unqualified_acc[name]) for name in sorted(unqualified_acc)},
    )


def qualifier_candidates(
    env: ImportEnv,
    qualifier: tuple[str, ...],
    *,
    anchored: bool,
) -> tuple[ModuleId, ...]:
    """Collect every route matching a qualifier, without applying precedence."""
    candidates: set[ModuleId] = set()
    for module, contribution in env.contributions.items():
        matches_path = contribution.path_enabled and (
            module.segments == qualifier
            if anchored
            else bool(qualifier) and module.segments[-len(qualifier) :] == qualifier
        )
        matches_alias = (
            not anchored and len(qualifier) == 1 and qualifier[0] in contribution.aliases
        )
        if matches_path or matches_alias:
            candidates.add(module)
    return tuple(sorted(candidates, key=_module_sort_key))


def resolve_qualified(
    env: ImportEnv,
    qualifier: tuple[str, ...],
    member: str,
    *,
    anchored: bool = False,
) -> QualResolution:
    """Resolve one qualified reference with suffix/anchor semantics.

    Routes are collected before their members are considered. Member filtering
    can therefore rescue a shared suffix or alias facade, while multiple
    surviving candidates always remain an ambiguity: there is no route, suffix,
    alias, or declaration-order preference.
    """
    candidates = qualifier_candidates(env, qualifier, anchored=anchored)
    if not candidates:
        return QualResolutionUnknownQualifier(qualifier)

    member_candidates = tuple(
        module for module in candidates if member in env.contributions[module].members
    )
    if not member_candidates:
        return QualResolutionMissingMember(qualifier, member, candidates)
    if len(member_candidates) > 1:
        return QualResolutionAmbiguous(qualifier, member, member_candidates)

    module = member_candidates[0]
    return QualResolutionFound(module, env.contributions[module].members[member])
