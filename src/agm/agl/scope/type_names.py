"""Type-name spellings shared by scope and the type-owner index.

:func:`member_chain` spells a path beneath an alias's target
``<target>::path`` as its target is written, so the one whole-path lookup at
the alias's site decides which of the target's paths the alias reaches.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TypeGuard

from agm.agl.scope.symbols import ScopePath, TypeOwner
from agm.agl.syntax.nodes import QualifierChain, QualifierSegment, TypeAlias, VariantRef
from agm.agl.syntax.types import AppliedT, NameT, TypeExpr

__all__ = [
    "MemberHidden",
    "MemberReferenced",
    "MemberSelection",
    "is_nominal_type_expr",
    "member_chain",
    "owner_member_selection",
    "owner_type_expr",
    "renames_target",
    "selection_node_id",
]


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


def renames_target(alias: TypeAlias) -> bool:
    """Whether *alias* is another name for its target: it passes its type parameters through.

    ``type A = m::B`` and ``type A[T] = m::B[T]`` rename ``m::B``; ``type A =
    m::B[int]`` and ``type A[T] = m::B`` are types of their own.
    """
    target = alias.type_expr
    if isinstance(target, NameT):
        return not alias.type_params
    return isinstance(target, AppliedT) and alias.type_params == tuple(
        arg.name if isinstance(arg, NameT) and arg.qualifier is None else None
        for arg in target.args
    )


def selection_node_id(spelling: NameT | AppliedT | VariantRef) -> int:
    """Return the node id scope records *spelling*'s selected declaration under.

    A qualified or ``::``-anchored name's -- an enum member reference's
    included -- is its qualifier's; a bare name's is its own.
    """
    qualifier = spelling.chain if isinstance(spelling, VariantRef) else spelling.qualifier
    return spelling.node_id if qualifier is None else qualifier.node_id


def member_chain(owner: NameT | AppliedT, path: ScopePath) -> QualifierChain:
    """Return the chain spelling ``owner::path``, *owner* as written.

    The inverse of :func:`owner_type_expr` for a one-name *path*.
    """
    qualifier = owner.qualifier
    segments = (
        QualifierSegment(
            owner.name,
            owner.args if isinstance(owner, AppliedT) else None,
            span=owner.span,
            node_id=owner.node_id,
        ),
        *(
            QualifierSegment(name, None, span=owner.span, node_id=owner.node_id)
            for name in path[:-1]
        ),
    )
    return QualifierChain(
        anchor=None if qualifier is None else qualifier.anchor,
        segments=(*(() if qualifier is None else qualifier.segments), *segments),
        member=path[-1],
        span=owner.span,
        node_id=owner.node_id,
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
    if (member,) in owner.hidden:
        return MemberHidden()
    return None
