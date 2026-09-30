"""Full-path lookup: which declaration a spelling selects where it is written.

A scope path is part of a declaration's name. A spelling ``p`` written inside
``scope S1::S2`` is looked up as ``S1::S2::p``, then ``S1::p``, then ``p``
(:func:`lookup_steps`); each step reads the module's own declarations at that
full path and the contributions anchored at or above the step, and the first
step selecting something decides. The module's own declaration wins;
otherwise one distinct contributed declaration is selected, and several are
ambiguous. ``::p`` reads the own root alone, and a module-anchored or slash
route that module alone. A type a written prefix selects also selects through
its own member table (an alias's projection, a record's own spelling).

The data each step reads is the caller's (:class:`PathSources`); this module
owns the order and the verdicts.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Protocol, TypeAlias

from agm.agl.diagnostics import AglError, HiddenMemberError
from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousQualificationError,
    BindingRef,
    ConstructorRef,
    ContributionLayer,
    DeclarationKey,
    DeclarationSelection,
    OwnerMemberSelection,
    QName,
    QualificationOrigin,
    ScopePath,
    TypeArgumentsError,
    TypeSelection,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.syntax.nodes import QualifierAnchor, QualifierChain
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import render_qualified_name, render_qualifier_path

__all__ = [
    "NOT_HIDDEN",
    "Candidate",
    "Hiding",
    "LookupKind",
    "Misfit",
    "PathSources",
    "QualifiedTarget",
    "Reading",
    "lookup_bare",
    "lookup_declared",
    "lookup_origins",
    "lookup_qualified",
    "lookup_reached",
    "lookup_steps",
    "is_removed",
    "removes",
]


class LookupKind(enum.Enum):
    """The declarations a position takes.

    ``TYPE`` for type positions and receivers, ``VALUE`` for values and
    constructors in value position, ``CONSTRUCTOR`` for patterns and ``is``.
    """

    TYPE = enum.auto()
    VALUE = enum.auto()
    CONSTRUCTOR = enum.auto()


#: What a qualified spelling finding nothing of its position's kind is read as instead.
_OTHER_KINDS: dict[LookupKind, tuple[LookupKind, ...]] = {
    LookupKind.TYPE: (LookupKind.VALUE,),
    LookupKind.VALUE: (LookupKind.TYPE,),
    LookupKind.CONSTRUCTOR: (LookupKind.VALUE, LookupKind.TYPE),
}


@dataclass(frozen=True, slots=True)
class QualifiedTarget:
    """What a spelling selects, whatever its position.

    ``key`` is the selected declaration's identity -- ``None`` only for an
    enum member a module qualifier's surface injects, which has no path of
    its own under that qualifier. ``ref`` is the binding the spelling reads
    as a value, and ``None`` for a type or a member only a type owner's own
    table selects. ``constructor`` is the constructor it names, if any.
    ``owner`` is set when the selection is an inline member of the type the
    last qualifier segment selects.
    """

    key: DeclarationKey | None
    ref: BindingRef | None
    constructor: ConstructorRef | None
    owner: OwnerMemberSelection | None = None

    @property
    def selection(self) -> TypeSelection | None:
        """What scope records for a spelling selecting this; ``None`` without a key."""
        if self.owner is not None:
            return self.owner
        return None if self.key is None else DeclarationSelection(self.key)


@dataclass(frozen=True, slots=True)
class Misfit:
    """A qualified spelling selecting only a declaration of another kind than its position's."""

    target: QualifiedTarget


Hiding: TypeAlias = frozenset[frozenset[DeclarationKey]]
"""What the ``hiding`` on each way a declaration is reached removes, by identity.

A declaration every way removes -- it or one above it -- is reached no way.
"""

#: Reached one way, hiding nothing.
NOT_HIDDEN: Hiding = frozenset({frozenset()})


def removes(
    hiding: Hiding, key: DeclarationKey, identity: Callable[[DeclarationKey], DeclarationKey]
) -> bool:
    """Whether every way of *hiding* removes the declaration *key* names, or one above it.

    *identity* names it (:meth:`PathSources.identity`).
    """
    if hiding == NOT_HIDDEN:
        return False
    module, path, name = identity(key)
    full = (*path, name)
    return all(
        any(
            hidden[0] == module and full[: len(hidden[1]) + 1] == (*hidden[1], hidden[2])
            for hidden in way
        )
        for way in hiding
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    """A declaration one source reaches, with the layer and origin that made it visible.

    ``hiding`` is what the ways that reached it hide; a path its owner table
    selects beneath it is reached the same ways.
    """

    target: QualifiedTarget
    layer: ContributionLayer
    origin: QualificationOrigin
    hiding: Hiding = NOT_HIDDEN


def is_removed(candidate: Candidate, identity: Callable[[DeclarationKey], DeclarationKey]) -> bool:
    """Whether every way that reached *candidate* removes its declaration (:func:`removes`)."""
    key = candidate.target.key
    return key is not None and removes(candidate.hiding, key, identity)


@dataclass(frozen=True, slots=True)
class Reading:
    """What sources reach at one full path: candidates, and owner-table refusals.

    A refusal (a member an owner's table only references, or its alias's
    import hides) is a verdict only when nothing is selected.
    """

    candidates: tuple[Candidate, ...] = ()
    refusals: tuple[AglError, ...] = ()

    def __add__(self, other: Reading) -> Reading:
        return Reading(self.candidates + other.candidates, self.refusals + other.refusals)


class PathSources(Protocol):
    """The declarations and contributions a lookup reads, by full path."""

    def own_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """The module's own declarations of *kind* at full *path*."""
        ...

    def contributed_at(self, step: ScopePath, path: ScopePath, kind: LookupKind) -> Reading:
        """What contributions anchored at or above *step* reach at full *path*."""
        ...

    def routed_at(self, chain: QualifierChain, path: ScopePath, kind: LookupKind) -> Reading:
        """What *chain*'s leading module route alone reaches at *path* beneath it."""
        ...

    def projected(
        self,
        owner: DeclarationKey,
        layer: ContributionLayer,
        rest: ScopePath,
        chain: QualifierChain,
        kind: LookupKind,
    ) -> Reading:
        """What type *owner*, made visible by *layer*, selects for *rest* in a position of *kind*.

        *rest* is the tail of *chain*'s names after the owner's. The owner's
        own member table decides, and an alias's target path stands beneath
        an alias.
        """
        ...

    def surface_injected(self, chain: QualifierChain, member: str) -> Reading:
        """The enum member module qualifier *chain*'s surface injects as *member*."""
        ...

    def inline_arity(self, owner: DeclarationKey, member: str, written: str) -> int | None:
        """The type parameters type *owner*, spelled *written*, takes when it owns *member* inline.

        A type owns its inline enum members inline, and a record's or
        exception's own constructor spellings (``Box::Box``). An alias's
        are those of its target it reaches. ``None`` when *member* is none.
        """
        ...

    def applies(self, key: DeclarationKey) -> bool:
        """Whether type *key* is an alias applying its target to type arguments of its own."""
        ...

    def hidden_at(self, step: ScopePath, path: ScopePath) -> bool:
        """Whether a ``hiding`` visible at *step* removed full *path* from its contribution."""
        ...

    def routed_hidden(self, chain: QualifierChain, path: ScopePath) -> bool:
        """Whether a ``hiding`` removed *path* from *chain*'s leading module route."""
        ...

    def own_origins(self, path: ScopePath) -> frozenset[QName]:
        """Full *path* when it is one of the module's own scope paths or types."""
        ...

    def identity(self, key: DeclarationKey) -> DeclarationKey:
        """The declaration *key* names: a renaming alias's is its target's."""
        ...

    def denotes(self, key: DeclarationKey) -> object:
        """What *key* names in an ambiguity: its :meth:`identity`, or the type an alias denotes.

        Two aliases denoting one type are one, whatever their declarations.
        """
        ...

    def aliases(self, key: DeclarationKey) -> bool:
        """Whether *key* declares a type alias."""
        ...

    def contributed_origins(self, step: ScopePath, path: ScopePath) -> frozenset[QName]:
        """The scopes and types contributions anchored at or above *step* reach as *path*.

        Only a path a contribution reaches as a qualifier -- a scope above
        what it reaches, or a type -- has any.
        """
        ...

    def routed_origins(self, chain: QualifierChain, path: ScopePath) -> frozenset[QName]:
        """The scopes and types *chain*'s leading module route reaches as *path* beneath it."""
        ...


def lookup_steps(scope_path: ScopePath) -> tuple[ScopePath, ...]:
    """Return the steps a spelling written in *scope_path* is tried at, nearest first."""
    return tuple(scope_path[:end] for end in range(len(scope_path), -1, -1))


@dataclass(frozen=True, slots=True)
class _Step:
    """One step: the path spellings are read under, and what reads them.

    *owners* reads the types a written prefix selects as the owner of the
    segments after it. The first *start* written segments form a module
    route rather than selecting anything themselves.
    """

    path: ScopePath
    read: Callable[[ScopePath, LookupKind], Reading]
    owners: Callable[[ScopePath], Reading]
    start: int = 0


@dataclass(frozen=True, slots=True)
class _Anchor:
    """Where a spelling is read: its steps, and how a miss there is named."""

    steps: tuple[_Step, ...]
    hidden: Callable[[ScopePath], bool]
    visible: Callable[[ScopePath], frozenset[QName]]
    route: tuple[str, ...] = ()


def lookup_bare(
    sources: PathSources,
    name: str,
    scope_path: ScopePath,
    kind: LookupKind,
    *,
    span: SourceSpan,
    local_to: ModuleId,
) -> QualifiedTarget | AglError | None:
    """Return what bare *name*, written in *scope_path*, selects; ``None`` when nothing.

    *span* locates an ambiguity.
    """
    anchor = _anchor(sources, None, scope_path)
    return _Walk(sources, anchor.steps, (), None, (name,), local_to, span).find(kind)


def lookup_declared(
    sources: PathSources,
    path: ScopePath,
    written: QualifierChain | None,
    kind: LookupKind,
    *,
    span: SourceSpan,
    local_to: ModuleId,
) -> QualifiedTarget | AglError | None:
    """Return the declaration of *kind* at full *path*, read at its parent step alone.

    A declaration at *path* is what a declaring path (a receiver's) names, so
    no step further out is tried. *written* is the qualifier chain spelling
    the last names of *path*, if any; the types its prefixes select add what
    their own member tables select. Finding nothing is then an owner-table
    refusal, a hidden member when a ``hiding`` removed *path*, else an
    unknown member of it, since every prefix of a declaring path is a scope
    path of the module's own. *span* locates a bare spelling's ambiguity.
    """
    names = path[len(path) - (1 if written is None else len(written.segments) + 1) :]
    parent = _step(sources, path[:-1])
    step = replace(parent, path=path[: len(path) - len(names)])
    walk = _Walk(sources, (step,), (), written, names, local_to, span)
    found = walk.find(kind)
    if found is not None or written is None:
        return found
    refusal = walk.refusal()
    if refusal is not None:
        return refusal
    if sources.hidden_at(parent.path, path):
        return HiddenMemberError(render_qualifier_path(written), path[-1], span=written.span)
    return UnknownMemberError(render_qualified_name(written, path[-1]), span=written.span)


def _nowhere(_path: ScopePath) -> bool:
    return False


def lookup_reached(
    sources: PathSources,
    chain: QualifierChain,
    scope_path: ScopePath,
    kind: LookupKind,
    *,
    local_to: ModuleId,
    owners_within: int | None = None,
) -> tuple[Candidate, ...]:
    """Return the declarations of *kind* that *chain*, written in *scope_path*, reaches.

    Those the first step reaching any finds, its own ones alone when it has
    some: several distinct ones are ambiguous where the spelling is used. A
    module qualifier's surface injects no enum member here. *chain* spells
    more than a module route. With *owners_within*, only a type its first
    that many names reach, or an alias, projects its member table: the rest
    of the path must be declared, an alias standing for its target's path.
    """
    names = (*(segment.name for segment in chain.segments), chain.member)
    anchor = _anchor(sources, chain, scope_path)
    walk = _Walk(sources, anchor.steps, anchor.route, chain, names, local_to, chain.span)
    return walk.reached(kind, len(names) if owners_within is None else owners_within)


def lookup_origins(
    sources: PathSources, chain: QualifierChain, scope_path: ScopePath
) -> frozenset[QName]:
    """Return the scopes and types *chain*'s full path, written in *scope_path*, names.

    Those of every step: a qualifier names each scope it reaches.
    """
    return _anchor(sources, chain, scope_path).visible(
        (*(segment.name for segment in chain.segments), chain.member)
    )


def lookup_qualified(
    sources: PathSources,
    chain: QualifierChain,
    member: str,
    scope_path: ScopePath,
    kind: LookupKind,
    *,
    span: SourceSpan,
    local_to: ModuleId,
) -> QualifiedTarget | Misfit | AglError:
    """Return what *chain*``::``*member*, written in *scope_path*, selects, or why nothing.

    Finding nothing of *kind* but a declaration of another kind is a
    :class:`Misfit`. *span* locates a ``::name`` miss.
    """
    names = (*(segment.name for segment in chain.segments), member)
    anchor = _anchor(sources, chain, scope_path)
    walk = _Walk(sources, anchor.steps, anchor.route, chain, names, local_to, span)
    found = walk.find(kind)
    if found is not None:
        return found
    refusal = walk.refusal()
    if refusal is not None:
        return refusal
    for other in _OTHER_KINDS[kind]:
        misfit = _Walk(sources, anchor.steps, anchor.route, chain, names, local_to, span).find(
            other
        )
        if isinstance(misfit, QualifiedTarget):
            return Misfit(misfit)
        if misfit is not None:
            return misfit
    if not chain.segments:
        return UnknownMemberError(render_qualified_name(chain, member), span=span)
    if anchor.hidden(names):
        return HiddenMemberError(render_qualifier_path(chain), member, span=chain.span)

    def selects(prefix: QualifierChain, kind: LookupKind) -> QualifiedTarget | AglError | None:
        written = names[: len(prefix.segments) + 1]
        return _Walk(sources, anchor.steps, anchor.route, prefix, written, local_to, span).find(
            kind
        )

    return _unknown(chain, names, anchor.visible, selects)


def _anchor(sources: PathSources, chain: QualifierChain | None, scope_path: ScopePath) -> _Anchor:
    """Return where *chain*, written in *scope_path*, is read."""
    if chain is not None and (
        chain.anchor is QualifierAnchor.MODULE or (chain.segments and "/" in chain.segments[0].name)
    ):
        routed = chain
        return _Anchor(
            (
                _Step(
                    (),
                    lambda path, kind: sources.routed_at(routed, path[1:], kind),
                    lambda path: sources.routed_at(routed, path[1:], LookupKind.TYPE),
                    1,
                ),
            ),
            lambda path: sources.routed_hidden(routed, path[1:]),
            lambda path: sources.routed_origins(routed, path[1:]),
            chain.leading_route,
        )
    if chain is not None and chain.anchor is QualifierAnchor.CURRENT_MODULE:
        own = _Step((), sources.own_at, lambda path: sources.own_at(path, LookupKind.TYPE))
        return _Anchor((own,), _nowhere, sources.own_origins)
    steps = lookup_steps(scope_path)
    return _Anchor(
        tuple(_step(sources, step) for step in steps),
        lambda path: any(sources.hidden_at(step, (*step, *path)) for step in steps),
        lambda path: frozenset().union(
            *(
                sources.own_origins((*step, *path))
                | sources.contributed_origins(step, (*step, *path))
                for step in steps
            )
        ),
    )


def _step(sources: PathSources, step: ScopePath) -> _Step:
    """Return *step* reading own declarations and contributions.

    An own declaration at a full path wins it, so the contributions there are
    read only when there is none. Every type a prefix reaches owns what its
    member table selects, the contributed ones beside an own one included.
    """

    def read(path: ScopePath, kind: LookupKind) -> Reading:
        own = sources.own_at(path, kind)
        return own if own.candidates else sources.contributed_at(step, path, kind)

    def owners(path: ScopePath) -> Reading:
        return sources.own_at(path, LookupKind.TYPE) + sources.contributed_at(
            step, path, LookupKind.TYPE
        )

    return _Step(step, read, owners)


class _Walk:
    """One spelling's walk over its anchor's steps for one kind.

    *route* is the module route the spelling leads with, if any.
    """

    def __init__(
        self,
        sources: PathSources,
        steps: tuple[_Step, ...],
        route: tuple[str, ...],
        chain: QualifierChain | None,
        names: ScopePath,
        local_to: ModuleId,
        span: SourceSpan,
    ) -> None:
        self._sources = sources
        self._steps = steps
        self._route = route
        self._chain = chain
        self._names = names
        self._local_to = local_to
        self._span = span
        self._refusals: list[AglError] = []

    def find(self, kind: LookupKind) -> QualifiedTarget | AglError | None:
        """Return what the first step selecting a declaration of *kind* selects."""
        for step in self._steps:
            found = self._decide(step, kind)
            if found is not None:
                return found
        return None

    def refusal(self) -> AglError | None:
        """The owner-table verdict the walk met, hidden first."""
        return next(
            (error for error in self._refusals if isinstance(error, HiddenMemberError)),
            next(iter(self._refusals), None),
        )

    def reached(self, kind: LookupKind, owners_within: int) -> tuple[Candidate, ...]:
        """Return what the first step reaching a declaration of *kind* reaches; own ones alone.

        Only a type the first *owners_within* names reach, or an alias,
        projects its member table.
        """
        for step in self._steps:
            reading = self._reading(step, kind, injects=False, owners_within=owners_within)
            candidates = self._kept(reading).candidates
            if candidates:
                own = tuple(
                    candidate
                    for candidate in candidates
                    if candidate.layer is ContributionLayer.DECLARED
                )
                return own or candidates
        return ()

    def _decide(self, step: _Step, kind: LookupKind) -> QualifiedTarget | AglError | None:
        """Decide the full path at *step*: own first, then one distinct contribution."""
        reading = self._kept(
            self._reading(step, kind, injects=True, owners_within=len(self._names))
        )
        self._refusals.extend(reading.refusals)
        selected = _decided(reading.candidates, self._sources.denotes)
        chain = self._chain
        if selected is None:
            return None
        if not isinstance(selected, Candidate):
            return self._ambiguous(
                selected, self._names[step.start :], self._span if chain is None else chain.span
            )
        if chain is None:
            return selected.target
        return self._owned(chain, step, selected.target)

    def _reading(
        self, step: _Step, kind: LookupKind, *, injects: bool, owners_within: int
    ) -> Reading:
        """Read the full path at *step*.

        Every type a written prefix of at most *owners_within* names reaches,
        and every alias a longer one reaches, adds what its own member table
        selects for the rest of the path. When
        *injects*, a module qualifier's surface adds the enum member it injects.
        """
        full = (*step.path, *self._names)
        reading = step.read(full, kind)
        chain = self._chain
        if chain is None:
            return reading
        reading = sum(
            (
                _reached_as(
                    self._sources.projected(key, owner.layer, full[end:], chain, kind), owner
                )
                for end in range(len(step.path) + step.start + 1, len(full))
                for owner in step.owners(full[:end]).candidates
                if (key := owner.target.key) is not None
                and (end <= len(step.path) + owners_within or self._sources.aliases(key))
            ),
            reading,
        )
        if (
            injects
            and kind is not LookupKind.TYPE
            and not reading.candidates
            and _is_module_qualifier(chain, step)
        ):
            reading += self._sources.surface_injected(chain, self._names[-1])
        return reading

    def _kept(self, reading: Reading) -> Reading:
        """*reading* without the candidates every way that reached them removes.

        A qualified spelling reaching only removed ones is hidden.
        """
        kept = tuple(
            candidate
            for candidate in reading.candidates
            if not is_removed(candidate, self._sources.identity)
        )
        chain = self._chain
        if len(kept) == len(reading.candidates) or chain is None:
            return Reading(kept, reading.refusals)
        hidden = HiddenMemberError(render_qualifier_path(chain), self._names[-1], span=chain.span)
        return Reading(kept, (*reading.refusals, hidden))

    def _owned(
        self, chain: QualifierChain, step: _Step, target: QualifiedTarget
    ) -> QualifiedTarget | AglError:
        """Return *target* with the type owning it inline, or why a segment's type arguments fail.

        A segment owns what follows it -- the next segment's selection, or
        *target* after the last -- when a type its full path reaches declares
        that as an inline member (an alias's projected member is declared
        beneath the alias). A segment carries type arguments only when the
        type its full path selects (own first, else the one contributed; two
        are ambiguous) owns what follows, as many as it takes. A segment
        selecting an alias that applies its target carries that alias's.
        """
        segments = chain.segments
        owner: DeclarationKey | None = None
        for index in range(step.start, len(segments)):
            segment = segments[index]
            last = index == len(segments) - 1
            prefix = (*step.path, *self._names[: index + 1])
            owners = self._kept(step.owners(prefix)).candidates
            selected = _decided(owners, self._sources.denotes)
            applied = segment.type_args is not None or (
                isinstance(selected, Candidate)
                and selected.target.key is not None
                and self._sources.applies(selected.target.key)
            )
            if not applied and not last:
                continue
            member = self._names[index + 1]
            following = (
                target
                if last
                else _decided(
                    self._kept(step.read((*prefix, member), LookupKind.TYPE)).candidates,
                    self._sources.denotes,
                )
            )
            owner, arity = next(
                (
                    (key, arity)
                    for candidate in owners
                    if (key := candidate.target.key) is not None
                    and _is_beneath(following, key)
                    and (arity := self._sources.inline_arity(key, member, segment.name)) is not None
                ),
                (None, None),
            )
            if not applied:
                continue
            if isinstance(selected, tuple):
                return self._ambiguous(selected, self._names[step.start : index + 1], segment.span)
            if selected is None or selected.target.key != owner:
                arity = None
            if arity is None or (segment.type_args is not None and arity != len(segment.type_args)):
                return TypeArgumentsError(segment.name, arity, span=segment.span)
        return (
            target
            if owner is None
            else replace(target, owner=OwnerMemberSelection(owner, self._names[-1]))
        )

    def _ambiguous(
        self, origins: tuple[QualificationOrigin, ...], names: ScopePath, span: SourceSpan
    ) -> AmbiguousQualificationError:
        """Return the error for spelling *names*, at *span*, selecting several declarations."""
        chain = self._chain
        return AmbiguousQualificationError.for_origins(
            self._route,
            names,
            origins,
            anchored=chain is not None and chain.anchored,
            span=span,
            local_to=self._local_to,
        )


def _unknown(
    chain: QualifierChain,
    names: ScopePath,
    visible: Callable[[ScopePath], frozenset[QName]],
    selects: Callable[[QualifierChain, LookupKind], QualifiedTarget | AglError | None],
) -> AglScopeError:
    """An unknown member of the longest prefix naming something, else an unknown qualifier.

    A prefix names something when it is *visible*, or *selects* a type as
    written: through an alias, the path it stands for. A prefix naming a
    value but no qualifier ends the search: a function, binding or injected
    enum member is never a qualifier.
    """
    for length in range(len(chain.segments), 0, -1):
        prefix = replace(chain, segments=chain.segments[: length - 1])
        if visible(names[:length]) or isinstance(selects(prefix, LookupKind.TYPE), QualifiedTarget):
            written = replace(chain, segments=chain.segments[:length])
            return UnknownMemberError(
                render_qualified_name(written, names[length]), span=chain.span
            )
        if selects(prefix, LookupKind.VALUE) is not None:
            break
    return UnknownQualifierError(render_qualifier_path(chain), span=chain.span)


def _reached_as(reading: Reading, owner: Candidate) -> Reading:
    """*reading*, what *owner*'s member table selects, reached the ways *owner* was."""
    if owner.hiding == NOT_HIDDEN:
        return reading
    return Reading(
        tuple(replace(candidate, hiding=owner.hiding) for candidate in reading.candidates),
        reading.refusals,
    )


def _decided(
    candidates: Iterable[Candidate], identity: Callable[[DeclarationKey], object]
) -> Candidate | tuple[QualificationOrigin, ...] | None:
    """The one candidate selected, own first; the origins when several distinct ones compete.

    Candidates are distinct when they name distinct declarations by
    *identity* (:meth:`PathSources.denotes`): an alias renaming a declaration
    is that declaration, and aliases denoting one type are one. A
    declaration reached several ways is one candidate -- the one spelling it
    directly, else the first by path -- yet an ambiguity it takes part in
    names every way it was reached.
    """
    pool = list(candidates)
    own = [candidate for candidate in pool if candidate.layer is ContributionLayer.DECLARED]
    competing = own or pool
    if len(competing) == 1:
        # A lone candidate is selected without reading what it names.
        return competing[0]
    distinct: dict[object, list[Candidate]] = {}
    for candidate in competing:
        target = candidate.target
        key = target.constructor if target.key is None else identity(target.key)
        distinct.setdefault(key, []).append(candidate)
    if len(distinct) > 1:
        return tuple(candidate.origin for reached in distinct.values() for candidate in reached)
    if not distinct:
        return None
    ((named, reached),) = distinct.items()

    def spelled_first(
        candidate: Candidate,
    ) -> tuple[bool, tuple[tuple[str, ...], ScopePath, str] | None]:
        key = candidate.target.key
        return key != named, None if key is None else (key[0].segments, key[1], key[2])

    return min(reached, key=spelled_first)


def _is_beneath(
    target: QualifiedTarget | Candidate | tuple[QualificationOrigin, ...] | None,
    owner: DeclarationKey,
) -> bool:
    """Whether *target* is declared directly beneath type *owner*."""
    if isinstance(target, Candidate):
        target = target.target
    if not isinstance(target, QualifiedTarget) or target.key is None:
        return False
    module, path, name = owner
    return target.key[0] == module and target.key[1] == (*path, name)


def _is_module_qualifier(chain: QualifierChain, step: _Step) -> bool:
    """Whether *chain*, read at *step*, is a module qualifier: ``::`` alone or one route."""
    if step.path:
        return False
    if chain.anchor is QualifierAnchor.CURRENT_MODULE:
        return not chain.segments
    return len(chain.segments) == 1
