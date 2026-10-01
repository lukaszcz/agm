"""Reading ``use`` declarations: what each exposes, read where it is written, whenever used.

A use names a target path, read as written where the use is -- only the
uses written before it visible meanwhile -- and exposes paths beneath it in
its region and the regions nested in it. :class:`UseReader` answers what a
module's uses expose for the module's path reads (:mod:`agm.agl.scope.lookup`),
which in turn read the uses: :class:`UseSources` is what it reads of them.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Protocol

from agm.agl.modules.ids import ModuleId, render_route_member
from agm.agl.scope.lookup import (
    Candidate,
    LookupKind,
    PathSources,
    QualifiedTarget,
    lookup_origins,
    lookup_reached,
    lookup_steps,
)
from agm.agl.scope.symbols import (
    BindingRef,
    ConstructorRef,
    ContributionLayer,
    DeclarationKey,
    MissRepair,
    QName,
    ScopeNode,
    ScopePath,
    UnknownMemberError,
    UnknownQualifierError,
    atom_under_prefix,
    contribution_origin,
)
from agm.agl.scope.symbols import declaration_qname as _key_qname
from agm.agl.scope.symbols import import_item_path as _item_path
from agm.agl.scope.symbols import qname_declaration as _qname_decl_key
from agm.agl.scope.symbols import to_bare_atom as _bare_atom
from agm.agl.scope.type_owners import TypeOwnerIndex
from agm.agl.syntax.nodes import QualifierAnchor, QualifierChain, QualifierSegment, UseDecl
from agm.agl.syntax.spans import SourceSpan

# What one outermost read has learned each ``use`` reaches, by declaration, path and kind.
type _UseReads = dict[tuple[int, ScopePath, LookupKind], tuple[Candidate, ...]]


def _use_target(decl: UseDecl) -> ScopePath:
    """The path *decl* names as its target, beneath its anchor."""
    return tuple(segment.name for segment in decl.target)


def spelled_through(decl: UseDecl, earlier: UseDecl) -> bool:
    """Whether *decl* spells its target through the name *earlier* introduces.

    Such a use reads its target through *earlier*, so it never replaces it.
    """
    return (
        earlier.alias is not None
        and not decl.anchored
        and not decl.current_module
        and decl.target[0].name == earlier.alias
    )


def _use_route(decl: UseDecl, names: ScopePath) -> tuple[str, ...] | None:
    """The module route *names*, beneath *decl*'s anchor, spell alone; ``None`` for a path."""
    if len(names) == 1 and not decl.current_module and (decl.anchored or "/" in names[0]):
        return tuple(names[0].split("/"))
    return None


def _use_chain(decl: UseDecl, names: ScopePath) -> QualifierChain:
    """Spell *names* beneath *decl*'s anchor as a chain written where *decl* is."""
    anchor = (
        QualifierAnchor.CURRENT_MODULE
        if decl.current_module
        else QualifierAnchor.MODULE
        if decl.anchored
        else None
    )
    return QualifierChain(
        anchor,
        tuple(QualifierSegment(name, None, decl.span, decl.node_id) for name in names[:-1]),
        names[-1],
        decl.span,
        decl.node_id,
    )


def _use_target_spelling(decl: UseDecl, member_path: ScopePath = ()) -> str:
    """Spell *decl*'s target, then *member_path* beneath it, as written."""
    names = (*(segment.name for segment in decl.target), *member_path)
    if decl.current_module:
        return render_route_member((), names)
    return render_route_member(tuple(names[0].split("/")), names[1:], anchored=decl.anchored)


class UseSources(PathSources, Protocol):
    """What reading a module's uses needs of the module, besides its path reads."""

    def fits(self, target: QualifiedTarget, kind: LookupKind) -> bool:
        """Whether *target* is a declaration of the kind a position takes."""
        ...

    def module_route_origins(
        self, route: tuple[str, ...], path: ScopePath, *, anchored: bool
    ) -> frozenset[QName]:
        """The scopes and types module *route* reaches as *path* beneath it; itself for none."""
        ...

    def variant_binding_ref(self, constructor: ConstructorRef, span: SourceSpan) -> BindingRef:
        """The binding an enum member *constructor*, injected bare at *span*, is read through."""
        ...

    def enum_members_named(self, name: str) -> Mapping[QName, ConstructorRef]:
        """The enums, of any module this one reads, with a member named *name*, and that member."""
        ...


class UseReader:
    """The uses one module writes, and those a REPL session retained, as the module reads them.

    *scope_entity_kinds* is the kind of each of the module's own declarations.
    """

    def __init__(
        self,
        sources: UseSources,
        module_id: ModuleId,
        scope_nodes: Mapping[ScopePath, ScopeNode],
        scope_entity_kinds: Mapping[DeclarationKey, str],
        type_owners: TypeOwnerIndex,
    ) -> None:
        self._sources = sources
        self._module_id = module_id
        self._scope_nodes = scope_nodes
        self._scope_entity_kinds = scope_entity_kinds
        self._type_owners = type_owners
        # The uses this entry writes; the others a REPL session retained.
        self._entry_ids: set[int] = set()
        # While a use's target is read, only the uses written before it are
        # visible (``_reading_use``); ``None`` when every use is.
        self._horizon: int | None = None
        # What the outermost read in progress has learned about each use.
        self._reads: _UseReads | None = None
        # Whether each use is a single-item rename, the target it names, and
        # the declarations its ``hiding`` removes.
        self._renamed: dict[int, bool] = {}
        self._identities: dict[int, frozenset[QName]] = {}
        self._hidden_by: dict[int, frozenset[DeclarationKey]] = {}

    @property
    def reads_every_use(self) -> bool:
        """Whether a read now sees every use, as no use's own read is in progress."""
        return self._horizon is None

    def write(self, layer: ScopeNode, decl: UseDecl) -> None:
        """Record *decl*, which this entry writes in *layer*."""
        layer.uses.append(decl)
        self._entry_ids.add(decl.node_id)

    @contextmanager
    def _reading_use(self, decl: UseDecl) -> Iterator[_UseReads]:
        """Read what *decl* reaches: only the uses written before it are visible meanwhile.

        What one outermost read learns about every use is kept until it ends.
        """
        horizon, reads = self._horizon, self._reads
        current: _UseReads = {} if reads is None else reads
        self._horizon, self._reads = decl.node_id, current
        try:
            yield current
        finally:
            self._horizon, self._reads = horizon, reads

    @contextmanager
    def view(self, every_use: bool) -> Iterator[None]:
        """Read with every use visible, or, unless *every_use*, those the read in progress sees.

        The type-owner index, which keeps its answers, reads a declaration's
        with every use; a retained alias's current reach, which it does not
        keep, as the read asking for it.
        """
        horizon = self._horizon
        if every_use:
            self._horizon = None
        try:
            yield
        finally:
            self._horizon = horizon

    def visible(self, layer: ScopeNode) -> Iterator[UseDecl]:
        """Yield *layer*'s uses a read now sees.

        Within a use's read, only those written before it. A use an earlier
        REPL entry retained yields to one this entry writes in the same
        region replacing it (:meth:`_replacing`).
        """
        horizon = self._horizon
        for decl in layer.uses:
            if horizon is not None and decl.node_id >= horizon:
                continue
            if decl.node_id in self._entry_ids or not self._use_replaced(layer.scope_path, decl):
                yield decl

    def _use_replaced(self, site: ScopePath, decl: UseDecl) -> bool:
        """Whether this entry writes a visible use in region *site* replacing *decl*."""
        return bool(self._replacing(site, decl))

    def _replacing(self, site: ScopePath, decl: UseDecl) -> frozenset[int]:
        """The visible uses this entry writes in region *site* replacing *decl*.

        A rename replaces a rename to its name; any other use, one of the same
        target. Neither replaces a use it is spelled through (:func:`spelled_through`).
        """
        horizon = self._horizon
        written = [
            use
            for use in self._scope_nodes[site].uses
            if use.node_id in self._entry_ids
            and (horizon is None or use.node_id < horizon)
            and use.alias == decl.alias
            and not spelled_through(use, decl)
        ]
        if not written or decl.alias is not None:
            return frozenset(use.node_id for use in written)
        target = self.named_target(site, decl)
        return frozenset(use.node_id for use in written if self.named_target(site, use) == target)

    def replacements(self, layers: Iterable[ScopeNode]) -> dict[int, frozenset[int]]:
        """Each retained use of *layers* this entry replaces, with the uses replacing it.

        A REPL session drops a retained use once one of them is promoted.
        """
        return {
            decl.node_id: replacing
            for layer in layers
            for decl in layer.uses
            if decl.node_id not in self._entry_ids
            and (replacing := self._replacing(layer.scope_path, decl))
        }

    def named_target(self, site: ScopePath, decl: UseDecl) -> frozenset[QName]:
        """The scopes and types *decl*, a use renaming nothing written in region *site*, names."""
        found = self._identities.get(decl.node_id)
        if found is None:
            with self._reading_use(decl):
                found = self._use_path_origins(site, decl, _use_target(decl))
            self._identities[decl.node_id] = found
        return found

    def _use_renames(self, site: ScopePath, decl: UseDecl) -> bool:
        """Whether *decl*, written in region *site*, is a single-item rename.

        ``use P::m as A`` is ``use P::{m as A}`` when ``P::m`` is a
        declaration, and ``use T as A`` when ``T`` is a type; otherwise the
        alias names the whole target.
        """
        if decl.alias is None:
            return False
        found = self._renamed.get(decl.node_id)
        if found is None:
            target = _use_target(decl)
            kinds = (LookupKind.VALUE, LookupKind.TYPE) if len(target) > 1 else (LookupKind.TYPE,)
            with self._reading_use(decl):
                found = any(self._use_path_reached(site, decl, target, kind) for kind in kinds) or (
                    len(target) > 1 and self._pending_at(site, decl, target)
                )
            self._renamed[decl.node_id] = found
        return found

    def _pending_at(self, site: ScopePath, decl: UseDecl, names: ScopePath) -> bool:
        """Whether *names*, as *decl* in region *site* spells them, is an own member not yet bound.

        A non-static module installs a scoped let/var only where the walk
        reaches it.
        """
        if decl.current_module:
            bases: Iterable[ScopePath] = ((),)
        elif decl.anchored:
            return False
        else:
            bases = lookup_steps(site)
        return any(
            self._scope_entity_kinds.get((self._module_id, (*base, *names[:-1]), names[-1]))
            == "ordinary"
            for base in bases
        )

    def _use_rests(
        self, site: ScopePath, decl: UseDecl, relative: ScopePath
    ) -> tuple[ScopePath, ...]:
        """The paths beneath *decl*'s target it exposes as *relative* in region *site*.

        An alias stands for the target, a single-item rename adds the item
        under its own name, a glob exposes every path its ``hiding`` leaves,
        and a tail each path under an item, a renamed item under its rename
        too.
        """
        rests: dict[ScopePath, None] = {}
        if decl.tail is None:
            if relative[0] == decl.alias:
                rests[relative[1:]] = None
            target = _use_target(decl)
            if len(target) > 1 and relative[0] == target[-1] and self._use_renames(site, decl):
                rests[relative[1:]] = None
        elif not decl.tail:
            atom = _bare_atom(relative)
            if not any(atom_under_prefix(atom, _item_path(item)) for item in decl.hidden):
                rests[relative] = None
        else:
            for item in decl.tail:
                path = _item_path(item)
                if atom_under_prefix(_bare_atom(relative), path):
                    rests[relative] = None
                if relative[0] == item.rename:
                    rests[(*path, *relative[1:])] = None
        return tuple(rests)

    def _use_reached(
        self, site: ScopePath, decl: UseDecl, relative: ScopePath, kind: LookupKind
    ) -> tuple[Candidate, ...]:
        """What *decl*, written in region *site*, reaches of *kind* as *relative*.

        Each path beneath the target it exposes there is read as the use
        site reads the target spelled with it. Only a type the target's path
        reaches projects its member table there: a type the use exposes
        projects where the exposed path is read, a renamed one included.
        """
        with self._reading_use(decl) as reads:
            key = (decl.node_id, relative, kind)
            found = reads.get(key)
            if found is None:
                target = _use_target(decl)
                owners_within = len(target) - (1 if self._use_renames(site, decl) else 0)
                found = tuple(
                    candidate
                    for rest in self._use_rests(site, decl, relative)
                    for candidate in self._use_path_reached(
                        site, decl, (*target, *rest), kind, owners_within
                    )
                )
                reads[key] = found
            return found

    def exposure(
        self, site: ScopePath, decl: UseDecl, relative: ScopePath, kind: LookupKind
    ) -> Iterator[Candidate]:
        """What *decl*, written in region *site*, contributes of *kind* as *relative*.

        What it reaches, and a bare enum member an enum it exposes injects.
        """
        reached: Iterable[Candidate] = self._use_reached(site, decl, relative, kind)
        if kind is not LookupKind.TYPE and len(relative) == 1:
            reached = itertools.chain(reached, self._use_injected(site, decl, relative[0]))
        for candidate in reached:
            used = self._as_used(site, decl, candidate, alone=len(relative) == 1)
            if self._sources.fits(used.target, kind):
                yield used

    def _as_used(
        self, site: ScopePath, decl: UseDecl, candidate: Candidate, *, alone: bool
    ) -> Candidate:
        """*candidate* as *decl*, in region *site*, contributes it, exposed *alone* or owned.

        An alias segment stands for its target's path, so a member an alias
        renaming its target selects that the use exposes *alone* -- no
        longer spelled beneath the alias -- is the target's own; one an
        alias applying its target selects stays at the alias's type
        arguments. Each way it was reached also removes what the use's
        ``hiding`` names.
        """
        target = candidate.target
        declaration = candidate.origin.declaration
        key = target.key
        constructor = target.constructor
        member = None
        if alone and key is not None and constructor is not None and constructor.member is not None:
            owner = _qname_decl_key((key[0], _bare_atom(key[1])))
            if not self._sources.applies(owner):
                member = self._type_owners.owner_member(owner, key[2])
        if member is not None:
            target = QualifiedTarget(
                (member.owner_module_id, member.owner_path, member.owner_name), None, member
            )
            declaration = member.qname
        layer = ContributionLayer.USE
        removed = self._use_hidden(site, decl)
        hiding = (
            frozenset(way | removed for way in candidate.hiding) if removed else candidate.hiding
        )
        return Candidate(target, layer, contribution_origin(declaration, layer), hiding)

    def _use_hidden(self, site: ScopePath, decl: UseDecl) -> frozenset[DeclarationKey]:
        """The declarations *decl*'s ``hiding``, read in region *site*, names, by identity."""
        found = self._hidden_by.get(decl.node_id)
        if found is None:
            target = _use_target(decl)
            with self._reading_use(decl):
                found = frozenset(
                    self._sources.identity(key)
                    for item in decl.hidden
                    for kind in LookupKind
                    for candidate in self._use_path_reached(
                        site, decl, (*target, *_item_path(item)), kind, len(target)
                    )
                    if (key := candidate.target.key) is not None
                )
            self._hidden_by[decl.node_id] = found
        return found

    def _use_injected(self, site: ScopePath, decl: UseDecl, name: str) -> Iterator[Candidate]:
        """Yield the enum member *name* each enum *decl*, in region *site*, exposes injects.

        An inline member its ``hiding`` removed, or whose name a record or
        exception the use exposes owns, is not injected.
        """
        spellings = [] if decl.alias is None else [decl.alias]
        spellings.extend(
            item.rename
            for item in decl.tail or ()
            if item.rename is not None and not item.scope_path
        )
        layer = ContributionLayer.USE
        for qname, constructor in self._sources.enum_members_named(name).items():
            inline = constructor.inline_enum_owner_decl_node_id is not None
            key = _qname_decl_key(qname)
            exposed = any(
                any(
                    candidate.target.key == key
                    for candidate in self._use_reached(site, decl, (spelling,), LookupKind.TYPE)
                )
                and (not inline or self._use_rests(site, decl, (spelling, name)))
                for spelling in (key[2], *spellings)
            )
            if not exposed or (inline and self._exposes_standalone(site, decl, name)):
                continue
            yield Candidate(
                QualifiedTarget(
                    _qname_decl_key(constructor.selected_qname),
                    self._sources.variant_binding_ref(constructor, decl.span),
                    constructor,
                ),
                layer,
                contribution_origin(constructor.selected_qname, layer),
            )

    def _exposes_standalone(self, site: ScopePath, decl: UseDecl, name: str) -> bool:
        """Whether *decl*, in region *site*, exposes a record or exception as *name*."""
        for candidate in self._use_reached(site, decl, (name,), LookupKind.VALUE):
            target = candidate.target
            if (
                target.key is None
                or target.constructor is None
                or target.constructor.inline_enum_owner_decl_node_id is not None
            ):
                continue
            owner = self._type_owners.owner(_key_qname(target.key))
            if owner is not None and owner.alias is None and owner.constructor is not None:
                return True
        return False

    def origins(self, site: ScopePath, decl: UseDecl, relative: ScopePath) -> frozenset[QName]:
        """The scopes and types *decl*, written in region *site*, exposes as *relative*.

        A path holding a tail item names what the target's path there names.
        """
        with self._reading_use(decl):
            target = _use_target(decl)
            paths = [(*target, *rest) for rest in self._use_rests(site, decl, relative)]
            if any(
                len(path := _item_path(item)) > len(relative) and path[: len(relative)] == relative
                for item in decl.tail or ()
            ):
                paths.append((*target, *relative))
            return frozenset().union(*(self._use_path_origins(site, decl, path) for path in paths))

    def _use_path_reached(
        self,
        site: ScopePath,
        decl: UseDecl,
        names: ScopePath,
        kind: LookupKind,
        owners_within: int | None = None,
    ) -> tuple[Candidate, ...]:
        """What *names*, spelled beneath *decl*'s anchor in region *site*, reaches of *kind*.

        With *owners_within*, only a type its first that many names reach projects.
        """
        if _use_route(decl, names) is not None:
            return ()
        return lookup_reached(
            self._sources,
            _use_chain(decl, names),
            site,
            kind,
            owners_within=owners_within,
        )

    def _use_path_origins(
        self, site: ScopePath, decl: UseDecl, names: ScopePath
    ) -> frozenset[QName]:
        """The scopes and types *names*, spelled beneath *decl*'s anchor in region *site*, name."""
        route = _use_route(decl, names)
        if route is not None:
            return self._sources.module_route_origins(route, (), anchored=decl.anchored)
        return lookup_origins(self._sources, _use_chain(decl, names), site)

    def _names_qualifier(
        self, site: ScopePath, decl: UseDecl, names: ScopePath, owners_within: int | None = None
    ) -> bool:
        """Whether *names*, spelled beneath *decl*'s anchor in region *site*, names a qualifier.

        With *owners_within*, only a type its first that many names reach projects.
        """
        return bool(self._use_path_origins(site, decl, names)) or bool(
            self._use_path_reached(site, decl, names, LookupKind.TYPE, owners_within)
        )

    def validate(self, site: ScopePath, decl: UseDecl) -> None:
        """Check that *decl*, written in region *site*, names a qualifier, each item a path beneath.

        A single-item rename's target is the declaration it renames. An item
        is a path declared beneath the target: a type the target reaches
        projects its member table, and an alias inside the item stands for
        its target's path.
        """
        target = _use_target(decl)
        with self._reading_use(decl):
            if not self._use_renames(site, decl) and not self._names_qualifier(site, decl, target):
                raise UnknownQualifierError(
                    _use_target_spelling(decl), span=decl.span, repair=MissRepair.IMPORT_MODULE
                )
            for item in (*(decl.tail or ()), *decl.hidden):
                item_path = _item_path(item)
                path = (*target, *item_path)
                if not (
                    self._names_qualifier(site, decl, path, len(target))
                    or self._use_path_reached(site, decl, path, LookupKind.VALUE, len(target))
                    or self._pending_at(site, decl, path)
                ):
                    raise UnknownMemberError(_use_target_spelling(decl, item_path), span=decl.span)
