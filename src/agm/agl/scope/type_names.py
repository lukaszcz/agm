"""The declarations a type name selects where it is written.

One selection serves every consumer that must agree on it: annotations,
alias targets, which resolve where their alias is declared, and value names
that denote no value. A name selects, in order: this module's nearest
lexical declaration of it; for an unqualified name, the type contributions
(``use`` members and region import tails) of the nearest layer contributing
it, ranked equally at the module root with root import tails; for an
unanchored qualified path, the nearest layer's contributions of that path;
then a qualified name's module route.
"""

from __future__ import annotations

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
    "MemberHidden",
    "MemberReferenced",
    "MemberSelection",
    "OwnerRoute",
    "TypeContributions",
    "TypeNameSite",
    "bare_type_selection",
    "imported_member_selection",
    "is_nominal_type_expr",
    "nominal_selection",
    "owner_member_selection",
    "owner_type_expr",
    "routed_qualifier_and_member",
    "type_name_selection",
]

TypeContributions = Callable[[NameAtom], tuple[ScopePath, frozenset[QName]] | None]
"""The nearest layer contributing a type spelling, with that layer's path and selections."""


@dataclass(frozen=True, slots=True)
class TypeNameSite:
    """The lexical layer a type name is written in and what it sees there.

    ``declares`` reports this module's own type declaration at a scope path;
    ``is_type`` whether an import-tail selection denotes a type;
    ``type_params`` the type parameters in scope, which shadow every declaration.
    """

    module_id: ModuleId
    scope_path: ScopePath
    import_env: ImportEnv
    declares: Callable[[ScopePath], bool]
    contributions: TypeContributions
    is_type: Callable[[QName], bool]
    type_params: frozenset[str] = frozenset()


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


def type_name_selection(site: TypeNameSite, type_expr: NameT | AppliedT) -> frozenset[QName]:
    """Return every declaration *type_expr*'s name selects at *site*; several are ambiguous."""
    return _type_name_selection(site, type_expr)[0]


def _type_name_selection(
    site: TypeNameSite, type_expr: NameT | AppliedT
) -> tuple[frozenset[QName], bool]:
    """Return *type_expr*'s selection at *site*, and whether it was reached indirectly.

    A direct hit (the site's own nearest lexical declaration) carries no
    ``hiding``: the second element is ``False`` only then. Every other route
    -- a bare name's contributions or root import tails, an unanchored
    qualified path's contributions, or a qualified name's module route --
    already reflects whatever ``hiding`` applies there, so the second element
    is ``True``.
    """
    qualifier = type_expr.qualifier
    anchor = None if qualifier is None else qualifier.anchor
    segments = () if qualifier is None else qualifier.route_segments
    if anchor is not QualifierAnchor.MODULE:
        for base in enclosing_scope_bases(
            site.scope_path, rooted=anchor is QualifierAnchor.CURRENT_MODULE
        ):
            path = (*base, *segments, type_expr.name)
            if site.declares(path):
                return frozenset({(site.module_id, _atom(path))}), False
    if qualifier is None:
        return bare_type_selection(site, type_expr.name), True
    if anchor is None:
        layer = site.contributions(_atom((*segments, type_expr.name)))
        if layer is not None:
            return layer[1], True
    return _routed_selection(site, qualifier, (type_expr.name,)), True


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
