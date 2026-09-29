"""The declarations a type name selects where it is written.

One selection serves every consumer that must agree on it: annotations,
alias targets, which resolve where their alias is declared, and value names
that denote no value. A name selects, in order: this module's nearest
lexical declaration of it; for an unqualified name, the type contributions
(``use`` members and region import tails) of the nearest layer contributing
it, ranked equally at the module root with root import tails; for an
unanchored qualified path, the nearest layer's contributions of that path;
then a qualified name's module route.

A leading (single-segment) name -- a bare type name, or a qualifier chain's
own first segment -- additionally shares the type/scope namespace with scope
regions: :func:`leading_name_reading` is the one nearest-level lookup a bare
type name, a qualifier's leading segment, and a method receiver all resolve
through, so a nearer region always stops a farther type from merging in.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeGuard

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    NameAtom,
    QName,
    QualResolutionAmbiguous,
    QualResolutionFound,
    resolve_qualified,
)
from agm.agl.scope.symbols import ScopePath, TypeOwner
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.syntax.nodes import QualifierAnchor, QualifierChain
from agm.agl.syntax.qualifiers import enclosing_scope_bases
from agm.agl.syntax.types import AppliedT, NameT, TypeExpr

__all__ = [
    "ContributionLayer",
    "LeadingReading",
    "MemberHidden",
    "MemberReferenced",
    "MemberSelection",
    "OwnerRoute",
    "TypeContributions",
    "TypeNameSite",
    "bare_type_selection",
    "imported_member_selection",
    "is_nominal_type_expr",
    "leading_name_reading",
    "nominal_selection",
    "owner_member_selection",
    "owner_type_expr",
    "routed_qualifier_and_member",
    "type_name_selection",
]


class ContributionLayer(enum.Enum):
    """Which layer a leading name's nearest reading came from.

    ``DECLARED`` is this module's own lexical declaration; ``USE`` a ``use``
    contribution (local or region-scoped); ``IMPORTED`` a bare import tail or
    scope route, reachable only at the module root.
    """

    DECLARED = enum.auto()
    USE = enum.auto()
    IMPORTED = enum.auto()


@dataclass(frozen=True, slots=True)
class LeadingReading:
    """A leading name's nearest-level reading, across the type/scope namespace.

    ``is_region`` marks a scope region's reading: decisive on its own, so
    ``types`` is always empty then -- a farther level's type never merges
    into a nearer region.
    """

    layer: ContributionLayer
    is_region: bool
    types: frozenset[QName] = frozenset()


TypeContributions = Callable[[NameAtom], tuple[ScopePath, frozenset[QName]] | None]
"""The nearest layer contributing a type spelling, with that layer's path and selections."""

LeadingContributions = Callable[[NameAtom], LeadingReading | None]
"""The nearest layer's :class:`LeadingReading` of a leading name, region and type alike."""


def _no_region(_path: ScopePath) -> bool:
    """The default ``declares_region``: a site that never sees one."""
    return False


def _no_leading_contribution(_name: NameAtom) -> LeadingReading | None:
    """The default ``leading_contributions``: a site that never sees one."""
    return None


@dataclass(frozen=True, slots=True)
class TypeNameSite:
    """The lexical layer a type name is written in and what it sees there.

    ``declares`` reports this module's own type declaration at a scope path;
    ``is_type`` whether an import-tail selection denotes a type;
    ``type_params`` the type parameters in scope, which shadow every declaration.
    ``declares_region`` reports this module's own scope-region declaration at
    a path; ``leading_contributions`` is :attr:`contributions`' counterpart
    for :func:`leading_name_reading` -- both default to "no region" for a
    site that never sees one (typecheck's own, region-free site).
    """

    module_id: ModuleId
    scope_path: ScopePath
    import_env: ImportEnv
    declares: Callable[[ScopePath], bool]
    contributions: TypeContributions
    is_type: Callable[[QName], bool]
    type_params: frozenset[str] = frozenset()
    declares_region: Callable[[ScopePath], bool] = _no_region
    leading_contributions: LeadingContributions = _no_leading_contribution


def bare_type_selection(site: TypeNameSite, name: NameAtom) -> frozenset[QName]:
    """Return what a bare *name* selects beyond own declarations."""
    layer = site.contributions(name)
    contributed = frozenset() if layer is None else layer[1]
    if layer is not None and layer[0]:
        return contributed
    imported = frozenset(
        qname for qname in site.import_env.unqualified.get(name, ()) if site.is_type(qname)
    )
    return contributed | imported


def leading_name_reading(
    site: TypeNameSite, name: str, *, rooted: bool = False
) -> LeadingReading | None:
    """Return the nearest level's reading of leading *name*, across the type/scope namespace.

    Checked nearest first: this module's own lexical declaration -- a scope
    region or a type -- at each enclosing scope, the module root alone when
    *rooted*; then, unless *rooted* (a current-module anchor never falls
    back further), the nearest layer *site* contributes it through -- a scope
    region is decisive wherever it is found, so a farther level's type never
    merges into a nearer region -- falling back to :func:`bare_type_selection`
    when *site* carries no region-aware layer of its own (the default for a
    site that never sees one, as typecheck's own does). Shared by a bare
    type name (:func:`type_name_selection`), a qualifier chain's own leading
    segment, and a method receiver's owner, so the three agree on the same
    nearest level.
    """
    for base in enclosing_scope_bases(site.scope_path, rooted=rooted):
        path = (*base, name)
        if site.declares_region(path):
            return LeadingReading(ContributionLayer.DECLARED, True)
        if site.declares(path):
            return LeadingReading(
                ContributionLayer.DECLARED, False, frozenset({(site.module_id, _atom(path))})
            )
    if rooted:
        return None
    contributed = site.leading_contributions(_atom((name,)))
    if contributed is not None:
        return contributed
    bare = bare_type_selection(site, name)
    return None if not bare else LeadingReading(ContributionLayer.IMPORTED, False, bare)


def type_name_selection(site: TypeNameSite, type_expr: NameT | AppliedT) -> frozenset[QName]:
    """Return every declaration *type_expr*'s name selects at *site*; several are ambiguous."""
    return _type_name_selection(site, type_expr)[0]


def _type_name_selection(
    site: TypeNameSite, type_expr: NameT | AppliedT
) -> tuple[frozenset[QName], bool]:
    """Return *type_expr*'s selection at *site*, and whether it was reached indirectly.

    A direct hit (the site's own nearest lexical declaration) carries no
    ``hiding``: the second element is ``False`` only then. Every other route
    -- a leading name's reading or root import tails, an unanchored
    qualified path's contributions, or a qualified name's module route --
    already reflects whatever ``hiding`` applies there, so the second element
    is ``True``.
    """
    qualifier = type_expr.qualifier
    if qualifier is None:
        return _leading_type_name_selection(site, type_expr.name, None, None)
    anchor = qualifier.anchor
    segments = qualifier.route_segments
    if not segments:
        return _leading_type_name_selection(site, type_expr.name, qualifier, anchor)
    if anchor is not QualifierAnchor.MODULE:
        for base in enclosing_scope_bases(
            site.scope_path, rooted=anchor is QualifierAnchor.CURRENT_MODULE
        ):
            path = (*base, *segments, type_expr.name)
            if site.declares(path):
                return frozenset({(site.module_id, _atom(path))}), False
    if anchor is None:
        layer = site.contributions(_atom((*segments, type_expr.name)))
        if layer is not None:
            return layer[1], True
    return _routed_selection(site, qualifier, (type_expr.name,)), True


def _leading_type_name_selection(
    site: TypeNameSite,
    name: str,
    qualifier: QualifierChain | None,
    anchor: QualifierAnchor | None,
) -> tuple[frozenset[QName], bool]:
    """Return a leading name's selection: a bare name, or a qualifier's own leading segment.

    Delegates to :func:`leading_name_reading`, the one nearest-level lookup
    shared with a qualifier chain's leading segment and a method receiver's
    owner. A module-anchored leading segment is a route only, never a local
    or contributed reading; a scope-region reading selects no type.
    """
    if anchor is QualifierAnchor.MODULE and qualifier is not None:
        return _routed_selection(site, qualifier, (name,)), True
    reading = leading_name_reading(site, name, rooted=anchor is QualifierAnchor.CURRENT_MODULE)
    if reading is None:
        return frozenset(), True
    return reading.types, reading.layer is not ContributionLayer.DECLARED


def imported_member_selection(
    site: TypeNameSite, owner: NameT | AppliedT, member: str
) -> frozenset[QName]:
    """Return what path ``owner::member`` selects through *site*'s imports, *owner* as written.

    The spelling reaches a member only through an import surface exposing its
    complete path, which ``hiding`` filters: the nearest layer contributing it,
    a bare owner's root import tails, else the owner's module route.
    """
    qualifier = owner.qualifier
    segments = () if qualifier is None else qualifier.route_segments
    path = _atom((*segments, owner.name, member))
    if qualifier is None or qualifier.anchor is None:
        layer = site.contributions(path)
        if layer is not None:
            return layer[1]
    if qualifier is None:
        return site.import_env.unqualified.get(path, frozenset())
    return _routed_selection(site, qualifier, (owner.name, member))


def routed_qualifier_and_member(
    qualifier: QualifierChain, tail: tuple[str, ...]
) -> tuple[tuple[str, ...], NameAtom]:
    """Return the module route and member atom *tail* resolves against the qualifier's lead."""
    return (
        tuple(qualifier.segments[0].name.split("/")),
        _atom((*(segment.name for segment in qualifier.segments[1:]), *tail)),
    )


def _routed_selection(
    site: TypeNameSite, qualifier: QualifierChain, tail: tuple[str, ...]
) -> frozenset[QName]:
    """Return what *qualifier*'s module route selects for *tail* below its later segments.

    A route and a bare compound spelling of the same path are checked
    together, as :func:`~agm.agl.scope.imports.resolve_qualified` does for
    every qualified lookup; a size above one reports ambiguity, matching
    every other selection this module returns, rather than collapsing it
    away.
    """
    if qualifier.anchor is QualifierAnchor.CURRENT_MODULE or not qualifier.segments:
        return frozenset()
    route, member = routed_qualifier_and_member(qualifier, tail)
    result = resolve_qualified(site.import_env, route, member, anchored=qualifier.anchored)
    if isinstance(result, QualResolutionFound):
        return frozenset({result.qname})
    if isinstance(result, QualResolutionAmbiguous):
        return frozenset((module, member) for module in result.candidates)
    return frozenset()


def is_nominal_type_expr(type_expr: TypeExpr, site: TypeNameSite) -> TypeGuard[NameT | AppliedT]:
    """Whether *type_expr* is a type name :func:`nominal_selection` can select at *site*.

    False for anything structural: not a type name, or the bare name of one
    of *site*'s type parameters.
    """
    return isinstance(type_expr, (NameT, AppliedT)) and not (
        isinstance(type_expr, NameT)
        and type_expr.qualifier is None
        and type_expr.name in site.type_params
    )


def nominal_selection(
    site: TypeNameSite, type_expr: TypeExpr
) -> tuple[frozenset[QName], bool] | None:
    """Return what *type_expr* selects at *site*, and whether indirectly.

    ``None`` when *type_expr* is not :func:`is_nominal_type_expr`. See
    :func:`_type_name_selection` for the elements of a non-``None`` result.
    """
    if not is_nominal_type_expr(type_expr, site):
        return None
    return _type_name_selection(site, type_expr)


def owner_type_expr(qualifier: QualifierChain) -> NameT | AppliedT:
    """Return the type expression a non-empty qualifier's last segment names as an owner."""
    owner_segment = qualifier.segments[-1]
    prefix_segments = qualifier.segments[:-1]
    owner_qualifier = (
        None
        if not prefix_segments and qualifier.anchor is None
        else QualifierChain(
            anchor=qualifier.anchor,
            segments=prefix_segments,
            member=owner_segment.name,
            span=qualifier.span,
            node_id=qualifier.node_id,
        )
    )
    if owner_segment.type_args is None:
        return NameT(
            name=owner_segment.name,
            qualifier=owner_qualifier,
            span=owner_segment.span,
            node_id=owner_segment.node_id,
        )
    return AppliedT(
        name=owner_segment.name,
        args=owner_segment.type_args,
        qualifier=owner_qualifier,
        span=owner_segment.span,
        node_id=owner_segment.node_id,
    )


@dataclass(frozen=True, slots=True)
class OwnerRoute:
    """How a use site spells an owner: resolved at *site* through *owner_expr*."""

    site: TypeNameSite
    owner_expr: NameT | AppliedT


@dataclass(frozen=True, slots=True)
class MemberReferenced:
    """*owner*'s enum only references ``member``; it selects nothing at *owner*'s own path."""


@dataclass(frozen=True, slots=True)
class MemberHidden:
    """*owner* declares ``member``, but no route at the site currently reaches it."""


MemberSelection = MemberReferenced | MemberHidden | None
"""The verdict for ``owner::member``: referenced-only, hidden, or ``None`` (reachable or absent).

An absent member raises no route diagnostic here: it is left to the
caller's own "unknown member" reporting, which already needs a second,
identity-based lookup (``TypeOwner.select`` or ``members.get``) to build the
actual result or reject it, so this function would gain nothing by
repeating that lookup only to relabel it.
"""


def owner_member_selection(
    owner: TypeOwner, member: str, route: OwnerRoute | None
) -> MemberSelection:
    """Return what ``owner::member`` selects: *owner* being what the spelling selects, by identity.

    Declared members come from *owner* alone: ``members`` (inline members, or
    an alias's already-filtered reachable projection), ``hidden`` (what that
    projection subtracted), and ``referenced`` (names the enum only
    references). An alias's ``hidden`` set already encodes its own use-site
    filter, so it is never re-filtered here. A nominal (non-alias) enum
    reached indirectly -- through a ``use`` contribution or an import route --
    carries no such set of its own, so *route*, when given, filters it fresh
    at the current site through :func:`imported_member_selection`; a direct
    lexical hit (``route`` reporting no indirection) is never filtered. A
    direct owner's referenced name declared at its own path
    (``owner.own_path_referenced``) selects like a declared member instead;
    an alias never carries that set (see :class:`~agm.agl.scope.symbols.TypeOwner`),
    so the same name stays referenced through one.
    """
    if member in owner.referenced:
        if member in owner.own_path_referenced:
            return None
        return MemberReferenced()
    if member in owner.hidden:
        return MemberHidden()
    constructor = owner.members.get(member)
    if constructor is None:
        return None
    if route is not None and owner.alias is None:
        reached = nominal_selection(route.site, route.owner_expr)
        if (
            reached is not None
            and reached[1]
            and constructor.qname
            not in imported_member_selection(route.site, route.owner_expr, member)
        ):
            return MemberHidden()
    return None
