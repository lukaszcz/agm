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

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import ImportEnv, NameAtom, QName, try_resolve_qualified_member
from agm.agl.scope.symbols import ScopePath
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.syntax.nodes import QualifierAnchor, QualifierChain
from agm.agl.syntax.qualifiers import enclosing_scope_bases
from agm.agl.syntax.types import AppliedT, NameT, TypeExpr

__all__ = [
    "TypeContributions",
    "TypeNameSite",
    "bare_type_selection",
    "imported_member_selection",
    "nominal_selection",
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


def bare_type_selection(
    site: TypeNameSite, name: NameAtom
) -> tuple[frozenset[QName], frozenset[QName]]:
    """Return what a bare *name* selects beyond own declarations, and the nearest layer's share."""
    layer = site.contributions(name)
    contributed = frozenset() if layer is None else layer[1]
    if layer is not None and layer[0]:
        return contributed, contributed
    imported = frozenset(
        qname for qname in site.import_env.unqualified.get(name, ()) if site.is_type(qname)
    )
    return contributed | imported, contributed


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
        return bare_type_selection(site, type_expr.name)[0], True
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


def _routed_selection(
    site: TypeNameSite, qualifier: QualifierChain, tail: tuple[str, ...]
) -> frozenset[QName]:
    """Return what *qualifier*'s module route selects for *tail* below its later segments."""
    if qualifier.anchor is QualifierAnchor.CURRENT_MODULE or not qualifier.segments:
        return frozenset()
    routed = try_resolve_qualified_member(
        site.import_env,
        tuple(qualifier.segments[0].name.split("/")),
        _atom((*(segment.name for segment in qualifier.segments[1:]), *tail)),
        anchored=qualifier.anchored,
    )
    return frozenset() if routed is None else frozenset({routed})


def nominal_selection(
    site: TypeNameSite, type_expr: TypeExpr
) -> tuple[frozenset[QName], bool] | None:
    """Return what *type_expr* selects at *site* and whether that was indirect.

    ``None`` when *type_expr* is structural: not a type name, or the bare name
    of one of the site's type parameters. See :func:`_type_name_selection` for
    the second element.
    """
    if not isinstance(type_expr, (NameT, AppliedT)) or (
        isinstance(type_expr, NameT)
        and type_expr.qualifier is None
        and type_expr.name in site.type_params
    ):
        return None
    return _type_name_selection(site, type_expr)
