"""Contribution-based import environments and qualified resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, TypeVar

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import ImportDecl
from agm.agl.syntax.types import ImportMode, render_qualifier

__all__ = [
    "ImportEnv",
    "ImportTarget",
    "ModuleContribution",
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
]

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


_K = TypeVar("_K")
_V = TypeVar("_V")


def _by_name(name: str) -> str:
    """Order a string-keyed mapping by its own keys."""
    return name


def _frozen_sorted(mapping: Mapping[_K, _V], key: Callable[[_K], str]) -> Mapping[_K, _V]:
    """Copy *mapping* into a deterministically ordered read-only mapping."""
    ordered: dict[_K, _V] = {entry: mapping[entry] for entry in sorted(mapping, key=key)}
    return MappingProxyType(ordered)


def _frozen_routes(
    routes: Mapping[tuple[str, ...], set[ModuleId]],
) -> Mapping[tuple[str, ...], tuple[ModuleId, ...]]:
    """Freeze a route index, ordering each candidate list for stable diagnostics."""
    return MappingProxyType(
        {
            qualifier: tuple(sorted(modules, key=ModuleId.path_str))
            for qualifier, modules in routes.items()
        }
    )


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
        members = _frozen_sorted(self.members, _by_name)
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class ImportEnv:
    """Pure contribution import environment.

    Contributions are indexed by imported module so repeated declarations union
    their selected sets. The unqualified map retains lazy-clash candidates.
    """

    contributions: Mapping[ModuleId, ModuleContribution]
    unqualified: Mapping[str, frozenset[QName]]
    # Route indexes derived from the contributions, so qualified resolution is a
    # dict lookup instead of a scan over every imported module.
    suffix_routes: Mapping[tuple[str, ...], tuple[ModuleId, ...]] = field(
        init=False, repr=False, compare=False
    )
    anchored_routes: Mapping[tuple[str, ...], tuple[ModuleId, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        contributions = _frozen_sorted(self.contributions, ModuleId.path_str)
        object.__setattr__(self, "contributions", contributions)
        unqualified = _frozen_sorted(self.unqualified, _by_name)
        object.__setattr__(self, "unqualified", unqualified)

        suffix: dict[tuple[str, ...], set[ModuleId]] = {}
        anchored: dict[tuple[str, ...], set[ModuleId]] = {}
        for module, contribution in contributions.items():
            if contribution.path_enabled:
                segments = module.segments
                for index in range(len(segments)):
                    suffix.setdefault(segments[index:], set()).add(module)
                anchored.setdefault(segments, set()).add(module)
            for alias in contribution.aliases:
                suffix.setdefault((alias,), set()).add(module)
        object.__setattr__(self, "suffix_routes", _frozen_routes(suffix))
        object.__setattr__(self, "anchored_routes", _frozen_routes(anchored))


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
                f"name {name!r} is not exported by module {module.path_str()!r}",
                span=decl.span,
            )
    if decl.mode == ImportMode.USING:
        return selected_names
    return exported_names - selected_names


def _targets(target: ImportTarget) -> tuple[ModuleId, ...]:
    """Return a deterministic per-module expansion of an import target."""
    if isinstance(target, SingleTarget):
        return (target.module,)
    return tuple(sorted(target.modules, key=ModuleId.path_str))


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
            f" ({existing[0].path_str()!r} and {qname[0].path_str()!r})",
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

        rename_map = _build_rename_map(decl)
        inject_bare = is_open or decl.mode == ImportMode.USING

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
            for source_name in sorted(selected):
                exposed_name = _exposed_name(source_name, rename_map)
                _add_member(accumulator, exposed_name, module_exports[source_name], decl)
                if inject_bare:
                    accumulator.bare_names.add(exposed_name)

    contributions: dict[ModuleId, ModuleContribution] = {}
    unqualified_acc: dict[str, set[QName]] = {}
    for module, accumulator in accumulators.items():
        contribution = ModuleContribution(
            module=module,
            members=accumulator.members,
            bare_names=frozenset(accumulator.bare_names),
            path_enabled=accumulator.path_enabled,
            aliases=frozenset(accumulator.aliases),
        )
        contributions[module] = contribution
        for name in contribution.bare_names:
            unqualified_acc.setdefault(name, set()).add(contribution.members[name])

    return ImportEnv(
        contributions=contributions,
        unqualified={name: frozenset(qnames) for name, qnames in unqualified_acc.items()},
    )


def qualifier_candidates(
    env: ImportEnv,
    qualifier: tuple[str, ...],
    *,
    anchored: bool,
) -> tuple[ModuleId, ...]:
    """Collect every route matching a qualifier, without applying precedence."""
    index = env.anchored_routes if anchored else env.suffix_routes
    return index.get(qualifier, ())


def contribution_routes(
    contribution: ModuleContribution,
) -> tuple[tuple[tuple[str, ...], bool], ...]:
    """Enumerate every ``(qualifier, anchored)`` route that reaches *contribution*.

    This is the writable inverse of :func:`qualifier_candidates`: a plain import
    exposes every suffix of its path plus the ``/``-anchored full path, and each
    alias exposes a single-segment route.
    """
    routes: list[tuple[tuple[str, ...], bool]] = [
        ((alias,), False) for alias in sorted(contribution.aliases)
    ]
    if contribution.path_enabled:
        segments = contribution.module.segments
        routes.extend((segments[index:], False) for index in range(len(segments)))
        routes.append((segments, True))
    return tuple(routes)


def qualifier_contributes(
    env: ImportEnv, qualifier: tuple[str, ...], member: str, *, anchored: bool = False
) -> bool:
    """Return whether some route under *qualifier* contributes *member*.

    True for both a unique resolution and an ambiguity: the qualifier does name
    the member either way, which is what qualifier-shadowing checks care about.
    """
    return isinstance(
        resolve_qualified(env, qualifier, member, anchored=anchored),
        (QualResolutionFound, QualResolutionAmbiguous),
    )


def private_missing_member_module(
    result: QualResolutionMissingMember,
    private_info: Mapping[QName, bool],
) -> ModuleId | None:
    """Return the candidate module that declares the missing member private.

    A privacy miss is reported in preference to a plain "not in the imported
    set" miss, because it names the actual reason the member is unreachable.
    """
    for module in result.candidates:
        if private_info.get((module, result.member)):
            return module
    return None


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
