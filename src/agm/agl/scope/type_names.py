"""Type-name lookups shared by scope and the type-owner index.

:func:`imported_member_selection` is what an alias's member path reaches
through the import surfaces its target is spelled through, so ``hiding``
filters an alias's members exactly as it filters its target's.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TypeGuard

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    NameAtom,
    QName,
    QualResolutionFound,
    resolve_qualified,
)
from agm.agl.scope.symbols import DeclarationKey, ScopePath, TypeOwner
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.syntax.nodes import QualifierAnchor, QualifierChain
from agm.agl.syntax.qualifiers import enclosing_scope_bases
from agm.agl.syntax.types import AppliedT, NameT, TypeExpr

__all__ = [
    "MemberHidden",
    "MemberReferenced",
    "MemberSelection",
    "TypeContributions",
    "imported_member_selection",
    "is_nominal_type_expr",
    "owner_member_selection",
    "owner_type_expr",
    "selection_node_id",
    "spells_own_declaration",
]


TypeContributions = Callable[[ScopePath], frozenset[QName]]
"""The types contributions make a bare path spelled at one site."""


def imported_member_selection(
    import_env: ImportEnv,
    contributions: TypeContributions,
    owner: NameT | AppliedT,
    member: str,
) -> frozenset[QName]:
    """Return what path ``owner::member`` selects through *import_env*, *owner* as written.

    The spelling reaches a member only through an import surface exposing its
    complete path, which ``hiding`` filters: the nearest layer *contributions*
    reports for it, a bare owner's root import tails, else the owner's module
    route.
    """
    qualifier = owner.qualifier
    segments = () if qualifier is None else qualifier.route_segments
    path = (*segments, owner.name, member)
    if qualifier is None or qualifier.anchor is None:
        contributed = contributions(path)
        if contributed:
            return contributed
    if qualifier is None:
        return import_env.unqualified.get(_atom(path), frozenset())
    return _routed_selection(import_env, qualifier, (owner.name, member))


def _routed_qualifier_and_member(
    qualifier: QualifierChain, tail: tuple[str, ...]
) -> tuple[tuple[str, ...], NameAtom]:
    """Return the module route and member atom *tail* resolves against the qualifier's lead."""
    return (
        qualifier.leading_route,
        _atom((*(segment.name for segment in qualifier.segments[1:]), *tail)),
    )


def _routed_selection(
    import_env: ImportEnv, qualifier: QualifierChain, tail: tuple[str, ...]
) -> frozenset[QName]:
    """Return what *qualifier*'s module route selects for *tail* below its later segments.

    A route and a bare compound spelling of the same path are checked
    together, as :func:`~agm.agl.scope.imports.resolve_qualified` does; the
    owner's own spelling selected one declaration, so its member paths are
    never ambiguous.
    """
    route, member = _routed_qualifier_and_member(qualifier, tail)
    result = resolve_qualified(import_env, route, member, anchored=qualifier.anchored)
    return frozenset({result.qname}) if isinstance(result, QualResolutionFound) else frozenset()


def is_nominal_type_expr(
    type_expr: TypeExpr, type_params: Collection[str]
) -> TypeGuard[NameT | AppliedT]:
    """Whether *type_expr* is a type name, one scope selects a declaration for.

    False for anything structural: not a type name, or the bare name of one
    of the enclosing declaration's *type_params*.
    """
    return isinstance(type_expr, (NameT, AppliedT)) and not (
        isinstance(type_expr, NameT)
        and type_expr.qualifier is None
        and type_expr.name in type_params
    )


def selection_node_id(type_expr: NameT | AppliedT) -> int:
    """Return the node id scope records *type_expr*'s selected declaration under.

    A qualified or ``::``-anchored name's is its qualifier's; a bare name's
    is its own.
    """
    qualifier = type_expr.qualifier
    return type_expr.node_id if qualifier is None else qualifier.node_id


def spells_own_declaration(
    module_id: ModuleId, scope_path: ScopePath, type_expr: NameT | AppliedT, key: DeclarationKey
) -> bool:
    """Whether *key*, selected for *type_expr* at *scope_path*, is reached directly.

    Direct means this module's own declaration at the path the spelling
    names from an enclosing scope: no import surface lies between, so no
    ``hiding`` filters its members. A selection through a ``use``
    contribution, an import tail, or a module route is indirect.
    """
    qualifier = type_expr.qualifier
    anchor = None if qualifier is None else qualifier.anchor
    if key[0] != module_id or anchor is QualifierAnchor.MODULE:
        return False
    segments = () if qualifier is None else qualifier.route_segments
    rooted = anchor is QualifierAnchor.CURRENT_MODULE
    return any(
        (*key[1], key[2]) == (*base, *segments, type_expr.name)
        for base in enclosing_scope_bases(scope_path, rooted=rooted)
    )


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


def owner_member_selection(owner: TypeOwner, member: str) -> MemberSelection:
    """Return what ``owner::member`` selects: *owner* being what the spelling selects, by identity.

    Declared members come from *owner* alone: ``hidden`` (what an alias's
    already-filtered reachable projection subtracted) and ``referenced``
    (names the enum only references). A nominal owner's own member hidden at
    the site is the caller's to decide, from what the site's contributions
    select. A direct owner's referenced name declared at its own path
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
    return None
