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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol, TypeAlias

from agm.agl.diagnostics import AglError, HiddenMemberError
from agm.agl.modules.ids import Reader
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousQualificationError,
    BindingRef,
    ConstructorRef,
    ContributionLayer,
    DeclarationKey,
    DeclarationSelection,
    Layers,
    OwnerMemberSelection,
    QName,
    QualificationOrigin,
    ScopePath,
    TypeArgumentsError,
    TypeSelection,
    UnknownMemberError,
    UnknownQualifierError,
    add_layers,
    contribution_origin,
)
from agm.agl.syntax.nodes import QualifierAnchor, QualifierChain, QualifierSegment
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import render_qualified_name, render_qualifier_path

__all__ = [
    "NOT_HIDDEN",
    "Candidate",
    "DeclarationNames",
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


class DeclarationNames(Protocol):
    """What declaration a key names."""

    def identity(self, key: DeclarationKey) -> DeclarationKey:
        """The declaration *key* names: a renaming alias's is its target's."""
        ...

    def denotes(self, key: DeclarationKey) -> object:
        """What *key* names in an ambiguity: its :meth:`identity`, or the type an alias denotes.

        Two aliases denoting one type are one, whatever their declarations.
        """
        ...

    def placement(self, key: DeclarationKey) -> DeclarationKey:
        """The full path *key*'s declaration is declared at.

        One its module writes beneath another module's type, directly or
        through an alias, is declared at that type's path there.
        """
        ...

    def scopes_of(self, key: DeclarationKey) -> frozenset[DeclarationKey]:
        """The scopes type *key* stands for beside its path: an alias of a built-in type's."""
        ...


def removes(hiding: Hiding, key: DeclarationKey, names: DeclarationNames) -> bool:
    """Whether every way of *hiding* removes the declaration *key* names, or one above it.

    *names* names it: an alias denoting the type a removed alias denotes is
    removed too, and so is a declaration placed beneath a removed type (or
    a scope a removed alias of a built-in type stands for), however spelled.
    """
    if hiding == NOT_HIDDEN:
        return False
    named = names.identity(key)
    placed = names.placement(key)
    denoted = names.denotes(key)
    return all(
        any(
            _beneath(named, hidden)
            or any(_beneath(placed, above) for above in (hidden, *names.scopes_of(hidden)))
            or (denoted != named and names.denotes(hidden) == denoted)
            for hidden in way
        )
        for way in hiding
    )


def _beneath(key: DeclarationKey, above: DeclarationKey) -> bool:
    """Whether declaration *key* is *above*, or lies beneath it."""
    module, path, name = key
    return above[0] == module and (*path, name)[: len(above[1]) + 1] == (*above[1], above[2])


@dataclass(frozen=True, slots=True)
class Candidate:
    """A declaration one source reaches, with the layer and origin that made it visible.

    ``hiding`` is what the ways that reached it hide; a path its owner table
    selects beneath it is reached the same ways. ``routed`` when only a
    module route reached it: its member table is read as that module's alone.
    """

    target: QualifiedTarget
    layer: ContributionLayer
    origin: QualificationOrigin
    hiding: Hiding = NOT_HIDDEN
    routed: bool = False


def is_removed(candidate: Candidate, names: DeclarationNames) -> bool:
    """Whether every way that reached *candidate* removes its declaration (:func:`removes`)."""
    key = candidate.target.key
    return key is not None and removes(candidate.hiding, key, names)


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


class AliasTarget(NamedTuple):
    """The type an alias stands for, and the path naming it in the module declaring it.

    A built-in type has no declaration (``None``); its path is its name.
    """

    declared: QName | None
    path: ScopePath


class Application(NamedTuple):
    """The type an applied segment stands for, and the type arguments the segment takes."""

    target: DeclarationKey
    arity: int


class PathSources(DeclarationNames, Protocol):
    """The declarations and contributions a lookup reads, by full path."""

    def own_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """The module's own declarations of *kind* at full *path*."""
        ...

    def own_root_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """The module's own declarations of *kind* at *path* from its root (``::p``).

        Those :meth:`own_at` reads, and those it declares at *path* beneath
        another module's type (``def Geo::m`` with ``type Geo = Base``, at
        ``::Base::m``).
        """
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
        *,
        routed: bool,
    ) -> Reading:
        """What type *owner*, made visible by *layer*, selects for *rest* in a position of *kind*.

        *rest* is the tail of *chain*'s names after the owner's. The owner's
        own member table decides, and an alias's target path stands beneath
        an alias. An owner only a module route reached (*routed*) is read as
        that module's alone.
        """
        ...

    def surface_injected(self, chain: QualifierChain, member: str) -> Reading:
        """The enum member module qualifier *chain*'s surface injects as *member*."""
        ...

    def injected_at(self, step: ScopePath, name: str) -> Reading:
        """The enum members injected as bare *name* at *step*.

        An enum injects its members at its own step: the module's own enums'
        members are its own, and any other is contributed.
        """
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

    def beneath_applied(
        self,
        applied: DeclarationKey,
        layer: ContributionLayer,
        rest: ScopePath,
        chain: QualifierChain,
        kind: LookupKind,
        *,
        routed: bool,
    ) -> Reading:
        """What type *applied*, an applied segment of *chain* stands for, selects for *rest*.

        Read as beneath an alias of *applied* (:meth:`projected`).
        """
        ...

    def application(
        self, key: DeclarationKey, segment: QualifierSegment, site: ScopePath
    ) -> Application | None:
        """What *segment*, selecting type *key* and written in *site*, stands for applied.

        An alias standing for one of its type parameters (``type Id[T] = T``)
        stands for the type its argument there names. ``None`` for any other.
        """
        ...

    def hidden_at(self, step: ScopePath, path: ScopePath) -> bool:
        """Whether a ``hiding`` visible at *step* removed full *path* from its contribution."""
        ...

    def nearest_scope(self, path: ScopePath) -> ScopePath:
        """The nearest scope path of the module's own at or above *path*.

        A path spelled through an alias, a module route or a contributed type
        is that one's path, never a scope path of the module's own.
        """
        ...

    def declaring_beside(self, path: ScopePath) -> tuple[ScopePath, ...]:
        """The other scope paths of the module's own declaring what its scope path *path* does.

        Own scope paths declaring one path are one path: what one anchors,
        a declaring path beneath any of them reaches.
        """
        ...

    def stands_for(self, key: DeclarationKey) -> tuple[AliasTarget, ...]:
        """What alias *key* stands for; nothing for another type or an alias of no named type."""
        ...

    def names_only(self, target: AliasTarget, named: Reading) -> bool:
        """Whether a path reaching the types *named* names only *target*'s type, or names for it.

        A built-in type's name always names it.
        """
        ...

    def routed_hidden(self, chain: QualifierChain, path: ScopePath) -> bool:
        """Whether a ``hiding`` removed *path* from *chain*'s leading module route."""
        ...

    def own_origins(self, path: ScopePath) -> frozenset[QName]:
        """Full *path* when it is one of the module's own scope paths or types."""
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

    def projected_origins(self, alias: DeclarationKey, rest: ScopePath) -> frozenset[QName]:
        """The scopes and types type *alias*'s own member table reaches as *rest* beneath it.

        Those *rest* stands for beneath its target (:meth:`projected`), unless
        hidden there; none when *alias* is no alias.
        """
        ...

    def reader(self) -> Reader:
        """The module spellings are written in, as an ambiguity spells its declarations."""
        ...


def lookup_steps(scope_path: ScopePath) -> tuple[ScopePath, ...]:
    """Return the steps a spelling written in *scope_path* is tried at, nearest first."""
    return tuple(scope_path[:end] for end in range(len(scope_path), -1, -1))


@dataclass(frozen=True, slots=True)
class _Step:
    """One step: the path spellings are read under, and what reads them.

    *owners* reads the types a written prefix selects as the owner of the
    segments after it; *hidden* tells whether a ``hiding`` removed a full
    path from what the step reads. The first *start* written segments form
    a module route rather than selecting anything themselves.
    """

    path: ScopePath
    read: Callable[[ScopePath, LookupKind], Reading]
    owners: Callable[[ScopePath], Reading]
    hidden: Callable[[ScopePath], bool]
    start: int = 0


@dataclass(frozen=True, slots=True)
class _Anchor:
    """Where a spelling is read: its steps, and what a full path names at one."""

    steps: tuple[_Step, ...]
    origins: Callable[[_Step, ScopePath], frozenset[QName]]
    route: tuple[str, ...] = ()


class _Standing(NamedTuple):
    """An alias a written prefix reaches at *step*, and what it stands for."""

    step: _Step
    owner: Candidate
    key: DeclarationKey
    target: AliasTarget


def lookup_bare(
    sources: PathSources,
    name: str,
    scope_path: ScopePath,
    kind: LookupKind,
    *,
    span: SourceSpan,
    contributions: bool = True,
    constructors: Callable[[Mapping[ConstructorRef, Layers]], AglError] | None = None,
) -> QualifiedTarget | AglError | None:
    """Return what bare *name*, written in *scope_path*, selects; ``None`` when nothing.

    A value spelling also reads, at each step, the enum members injected
    there (:meth:`PathSources.injected_at`) once no own declaration at the
    step's full path claims the name. Without *contributions*, only the
    module's own declarations and injections are read. *span* locates an
    ambiguity; *constructors*, when given, reports one among constructors
    alone.
    """
    steps = tuple(
        _step(sources, step, injects=kind is LookupKind.VALUE, contributions=contributions)
        for step in lookup_steps(scope_path)
    )
    walk = _Walk(sources, scope_path, steps, (), None, (name,), span, constructors=constructors)
    return walk.find(kind)


def lookup_declared(
    sources: PathSources,
    path: ScopePath,
    written: QualifierChain | None,
    kind: LookupKind,
    *,
    span: SourceSpan,
) -> QualifiedTarget | AglError | None:
    """Return the declaration of *kind* at full *path*, read at its parent step alone.

    A declaration at *path* is what a declaring path (a receiver's) names, so
    no step further out is tried (:func:`_declaring_step`). *written* is the
    qualifier chain spelling the last names of *path*, if any; the types its
    prefixes select add what their own member tables select, and an alias
    among them stands for its target's path, read at that path's own parent
    step. Finding nothing is then an owner-table refusal, a hidden member
    when a ``hiding`` removed *path*, else an unknown member of it. A bare
    value spelling (no *written*) also reads the enum members injected at
    the parent step, as :func:`lookup_bare` does. *span* locates a bare
    spelling's ambiguity.
    """
    names = path[len(path) - (1 if written is None else len(written.segments) + 1) :]
    step = _declaring_step(
        sources,
        path[: len(path) - len(names)],
        injects=written is None and kind is LookupKind.VALUE,
    )
    walk = _Walk(sources, step.path, (step,), (), written, names, span)
    found = walk.find(kind)
    if found is not None or written is None:
        return found
    refusal = walk.refusal()
    if refusal is not None:
        return refusal
    if step.hidden(path):
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
    walk = _Walk(sources, scope_path, anchor.steps, anchor.route, chain, names, chain.span)
    return walk.reached(kind, len(names) if owners_within is None else owners_within)


def lookup_origins(
    sources: PathSources, chain: QualifierChain, scope_path: ScopePath
) -> frozenset[QName]:
    """Return the scopes and types *chain*'s full path, written in *scope_path*, names.

    Those of every step: a qualifier names each scope it reaches.
    """
    names = (*(segment.name for segment in chain.segments), chain.member)
    anchor = _anchor(sources, chain, scope_path)
    walk = _Walk(sources, scope_path, anchor.steps, anchor.route, chain, names, chain.span)
    return walk.named(anchor.origins)


def lookup_qualified(
    sources: PathSources,
    chain: QualifierChain,
    member: str,
    scope_path: ScopePath,
    kind: LookupKind,
    *,
    span: SourceSpan,
) -> QualifiedTarget | Misfit | AglError:
    """Return what *chain*``::``*member*, written in *scope_path*, selects, or why nothing.

    Finding nothing of *kind* but a declaration of another kind is a
    :class:`Misfit`. *span* locates a ``::name`` miss.
    """
    names = (*(segment.name for segment in chain.segments), member)
    anchor = _anchor(sources, chain, scope_path)
    walk = _Walk(sources, scope_path, anchor.steps, anchor.route, chain, names, span)
    found = walk.find(kind)
    if found is not None:
        return found
    refusal = walk.refusal()
    if refusal is not None:
        return refusal
    for other in _OTHER_KINDS[kind]:
        misfit = _Walk(sources, scope_path, anchor.steps, anchor.route, chain, names, span).find(
            other
        )
        if isinstance(misfit, QualifiedTarget):
            return Misfit(misfit)
        if misfit is not None:
            return misfit
    if not chain.segments:
        return UnknownMemberError(render_qualified_name(chain, member), span=span)
    if any(step.hidden((*step.path, *names)) for step in anchor.steps):
        return HiddenMemberError(render_qualifier_path(chain), member, span=chain.span)

    def written(prefix: QualifierChain) -> _Walk:
        spelled = names[: len(prefix.segments) + 1]
        return _Walk(sources, scope_path, anchor.steps, anchor.route, prefix, spelled, span)

    return _unknown(chain, names, anchor.origins, written)


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
                    lambda path: sources.routed_hidden(routed, path[1:]),
                    1,
                ),
            ),
            lambda _step, path: sources.routed_origins(routed, path[1:]),
            chain.leading_route,
        )
    if chain is not None and chain.anchor is QualifierAnchor.CURRENT_MODULE:
        own = _Step(
            (),
            sources.own_root_at,
            lambda path: sources.own_root_at(path, LookupKind.TYPE),
            _nowhere,
        )
        return _Anchor((own,), lambda _step, path: sources.own_origins(path))
    return _Anchor(
        tuple(_step(sources, step) for step in lookup_steps(scope_path)),
        lambda step, path: sources.own_origins(path) | sources.contributed_origins(step.path, path),
    )


def _step(
    sources: PathSources, step: ScopePath, *, injects: bool = False, contributions: bool = True
) -> _Step:
    """Return *step* reading own declarations and contributions.

    An own declaration at a full path wins it, so the contributions there are
    read only when there is none -- and, when *injects*, the enum members
    injected at *step*, an own one winning like an own declaration. Without
    *contributions*, only the own ones are read. Every type a prefix reaches
    owns what its member table selects, the contributed ones beside an own
    one included: those anchored above the prefix, which may itself lie at or
    above *step* (:func:`lookup_declared`).
    """

    def read(path: ScopePath, kind: LookupKind) -> Reading:
        own = sources.own_at(path, kind)
        if own.candidates:
            return own
        injected = sources.injected_at(step, path[-1]) if injects else Reading()
        if contributions:
            return sources.contributed_at(step, path, kind) + injected
        return Reading(
            tuple(c for c in injected.candidates if c.layer is ContributionLayer.DECLARED)
        )

    def owners(path: ScopePath) -> Reading:
        return sources.own_at(path, LookupKind.TYPE) + sources.contributed_at(
            step[: len(path) - 1], path, LookupKind.TYPE
        )

    return _Step(step, read, owners, lambda path: sources.hidden_at(step, path))


def _declaring_step(sources: PathSources, base: ScopePath, *, injects: bool) -> _Step:
    """Return the step the declaring paths written beneath *base* are read at.

    Each full path is read at its parent step alone. A parent no scope path
    of the module's own spells -- one written through an alias, a module
    route or a contributed type -- anchors nothing itself: its step is the
    nearest own scope path above it, which injects nothing beneath it. A
    parent that is one also reads what every other own scope path declaring
    its path anchors (:meth:`PathSources.declaring_beside`). When *injects*,
    a parent step reads the enum members injected at it.
    """

    def parent(path: ScopePath) -> _Step:
        at = sources.nearest_scope(path[:-1])
        return _step(sources, at, injects=injects and at == path[:-1])

    def read(path: ScopePath, kind: LookupKind) -> Reading:
        reading = parent(path).read(path, kind)
        for beside in sources.declaring_beside(path[:-1]):
            reading += _step(sources, beside).read((*beside, path[-1]), kind)
        return reading

    return _Step(
        base,
        read,
        lambda path: parent(path).owners(path),
        lambda path: parent(path).hidden(path),
    )


class _Walk:
    """One spelling's walk over its anchor's steps for one kind.

    *site* is the scope the spelling is written in; *route* is the module
    route it leads with, if any; *constructors*, when given, reports an
    ambiguity among constructors alone.
    """

    def __init__(
        self,
        sources: PathSources,
        site: ScopePath,
        steps: tuple[_Step, ...],
        route: tuple[str, ...],
        chain: QualifierChain | None,
        names: ScopePath,
        span: SourceSpan,
        *,
        constructors: Callable[[Mapping[ConstructorRef, Layers]], AglError] | None = None,
    ) -> None:
        self._sources = sources
        self._site = site
        self._constructors = constructors
        self._steps = steps
        self._route = route
        self._chain = chain
        self._names = names
        self._span = span
        self._refusals: list[AglError] = []
        self._reached: dict[tuple[ScopePath, int], tuple[Candidate, ...]] = {}
        self._stands: dict[int, tuple[_Standing, ...]] = {}

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
            candidates = self._unremoved(reading).candidates
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
        reading = self._unremoved(
            self._reading(step, kind, injects=True, owners_within=len(self._names))
        )
        self._refusals.extend(reading.refusals)
        selected = _decided(reading.candidates, self._sources.denotes)
        chain = self._chain
        if selected is None:
            return None
        if not isinstance(selected, Candidate):
            competing = _by_constructor(selected)
            if self._constructors is not None and competing is not None:
                return self._constructors(competing)
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
        and every alias a longer one reaches, adds what it selects for the
        rest of the path (:meth:`_beneath`). A prefix reaching no type at
        *step* stands for the path of each alias it reaches at another
        (:meth:`_beside`). When *injects*, a module qualifier's surface adds
        the enum member it injects.
        """
        reading = step.read((*step.path, *self._names), kind)
        chain = self._chain
        if chain is None:
            return reading
        reading += self._through_prefixes(step, chain, kind, owners_within)
        if (
            injects
            and kind is not LookupKind.TYPE
            and not reading.candidates
            and _is_module_qualifier(chain, step)
        ):
            reading += self._sources.surface_injected(chain, self._names[-1])
        return reading

    def _through_prefixes(
        self, step: _Step, chain: QualifierChain, kind: LookupKind, owners_within: int
    ) -> Reading:
        """What each written prefix selects beneath it for the rest of the path.

        As :meth:`_reading` reads it, *owners_within* as there.
        """
        reading = Reading()
        for count in range(step.start + 1, len(self._names)):
            owners = self._owners(step, count)
            for owner in owners:
                key = owner.target.key
                if key is not None and (count <= owners_within or self._sources.aliases(key)):
                    reading += _reached_as(
                        self._beneath(step, owner, key, count, chain, kind), owner
                    )
            if not owners:
                for standing in self._standing(count):
                    reading += _reached_as(
                        self._beside(step, standing, count, chain, kind), standing.owner
                    )
        return reading

    def _owners(self, step: _Step, count: int) -> tuple[Candidate, ...]:
        """The types the first *count* written names reach at *step*."""
        at = (step.path, count)
        owners = self._reached.get(at)
        if owners is None:
            prefix = (*step.path, *self._names[:count])
            owners = self._reached[at] = step.owners(prefix).candidates
        return owners

    def _standing(self, count: int) -> tuple[_Standing, ...]:
        """The aliases the first *count* written names reach at any step, standing for a path.

        One only a module route reached stands for none: it is read as that
        module's alone.
        """
        stands = self._stands.get(count)
        if stands is None:
            sources = self._sources
            stands = self._stands[count] = tuple(
                _Standing(step, owner, key, target)
                for step in self._steps
                if count > step.start
                for owner in self._owners(step, count)
                if (key := owner.target.key) is not None and not owner.routed
                for target in sources.stands_for(key)
            )
        return stands

    def _beneath(
        self,
        step: _Step,
        owner: Candidate,
        key: DeclarationKey,
        count: int,
        chain: QualifierChain,
        kind: LookupKind,
    ) -> Reading:
        """What type *key*, which *owner* reached as the first *count* names at *step*, selects.

        Its own member table selects for the rest of the names. Where an
        alias's selects nothing, the path it stands for is read in its place
        (:meth:`_beside`). An owner only a module route reached is read as
        that module's alone.
        """
        sources = self._sources
        rest = self._names[count:]
        applied = self._applied(chain, key, count - 1)
        if applied is not None:
            return sources.beneath_applied(
                applied.target, owner.layer, rest, chain, kind, routed=owner.routed
            )
        reading = sources.projected(key, owner.layer, rest, chain, kind, routed=owner.routed)
        if reading.candidates or owner.routed:
            return reading
        return sum(
            (
                self._beside(step, _Standing(step, owner, key, target), count, chain, kind)
                for target in sources.stands_for(key)
            ),
            reading,
        )

    def _beside(
        self,
        step: _Step,
        standing: _Standing,
        count: int,
        chain: QualifierChain,
        kind: LookupKind,
    ) -> Reading:
        """What the path alias *standing* stands for reaches at *step* as the first *count* names.

        Where that path names only the alias's target at *step*, what
        contributions reach beneath it there is reached as the alias is (the
        alias's own member table reads this module's declarations beneath
        the target), and a path a ``hiding`` removed there is refused. Where
        it names no type at *step*, this module's own declarations there are
        a scope of its own named so, which the table never reads: they merge
        with it. An alias applying its target selects what its own member
        table does of what is reached. An alias named as its target's path
        adds nothing: the walk reads that path.
        """
        sources, names = self._sources, self._names
        _, owner, key, target = standing
        if target.path == names[:count]:
            return Reading()
        spelled = (*step.path, *target.path)
        beside = (*spelled, *names[count:])
        named = step.owners(spelled)
        reached = tuple(
            candidate
            if candidate.layer is ContributionLayer.DECLARED
            else replace(
                candidate,
                layer=owner.layer,
                origin=contribution_origin(candidate.origin.declaration, owner.layer),
            )
            for candidate in step.read(beside, kind).candidates
            if not candidate.routed
            and (candidate.layer is not ContributionLayer.DECLARED or not named.candidates)
        )
        hidden = not reached and step.hidden(beside)
        if not (reached or hidden) or not sources.names_only(target, named):
            return Reading()
        if hidden:
            return Reading(refusals=(self._hidden(chain),))
        if sources.applies(key):
            selected = sources.projected(key, owner.layer, names[count:], chain, kind, routed=False)
            if selected.candidates:
                return selected
        return Reading(reached)

    def named(self, origins: Callable[[_Step, ScopePath], frozenset[QName]]) -> frozenset[QName]:
        """The scopes and types the walk's full path names at every step, by *origins*.

        An alias a written prefix reaches adds what its own member table
        reaches for the rest of the path, and what the path it stands for
        names in its place, as :meth:`_reading` reads both.
        """
        names, sources = self._names, self._sources
        found: set[QName] = set()
        for step in self._steps:
            found |= origins(step, (*step.path, *names))
            for count in range(step.start + 1, len(names)):
                for key in filter(None, (owner.target.key for owner in self._owners(step, count))):
                    found |= sources.projected_origins(key, names[count:])
                stands = self._standing(count)
                if self._owners(step, count):
                    stands = tuple(standing for standing in stands if standing.step is step)
                for standing in stands:
                    target = standing.target
                    spelled = (*step.path, *target.path)
                    if target.path != names[:count] and sources.names_only(
                        target, step.owners(spelled)
                    ):
                        found |= origins(step, (*spelled, *names[count:]))
        return frozenset(found)

    def _hidden(self, chain: QualifierChain) -> HiddenMemberError:
        """The refusal of the walk's spelling, *chain*, as hidden."""
        return HiddenMemberError(render_qualifier_path(chain), self._names[-1], span=chain.span)

    def _applied(
        self, chain: QualifierChain, key: DeclarationKey, index: int
    ) -> Application | None:
        """What *chain*'s segment *index*, selecting type *key*, stands for applied.

        See :meth:`PathSources.application`.
        """
        segment = chain.segments[index]
        if segment.type_args is None:
            return None
        return self._sources.application(key, segment, self._site)

    def _unremoved(self, reading: Reading) -> Reading:
        """*reading* without the candidates every way that reached them removes.

        A qualified spelling reaching only removed ones is hidden.
        """
        kept = tuple(
            candidate
            for candidate in reading.candidates
            if not is_removed(candidate, self._sources)
        )
        chain = self._chain
        if len(kept) == len(reading.candidates) or chain is None:
            return Reading(kept, reading.refusals)
        return Reading(kept, (*reading.refusals, self._hidden(chain)))

    def _owned(
        self, chain: QualifierChain, step: _Step, target: QualifiedTarget
    ) -> QualifiedTarget | AglError:
        """Return *target* with the type owning it inline, or why a segment's type arguments fail.

        A segment owns what follows it -- the next segment's selection, or
        *target* after the last -- when a type its full path reaches declares
        that as an inline member (an alias's projected member is declared
        beneath the alias); where it reaches none, an alias another step
        reaches so does. A segment carries type arguments only when the
        type its full path selects (own first, else the one contributed; two
        are ambiguous) owns what follows, as many as it takes. A segment
        selecting an alias that applies its target carries that alias's.
        """
        segments = chain.segments
        owner: DeclarationKey | None = None
        for index in range(step.start, len(segments)):
            segment = segments[index]
            last = index == len(segments) - 1
            owners = self._selecting(
                chain,
                step,
                index + 1,
                self._owners(step, index + 1)
                or tuple(standing.owner for standing in self._standing(index + 1)),
            )
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
                    self._selecting(
                        chain,
                        step,
                        index + 2,
                        step.read(
                            (*step.path, *self._names[: index + 2]), LookupKind.TYPE
                        ).candidates,
                    ),
                    self._sources.denotes,
                )
            )
            owner, arity = next(
                (
                    (key, arity if application is None else application.arity)
                    for candidate in owners
                    if (key := candidate.target.key) is not None
                    and _is_beneath(
                        following,
                        reached := (
                            key
                            if (application := self._applied(chain, key, index)) is None
                            else application.target
                        ),
                    )
                    and (arity := self._sources.inline_arity(reached, member, segment.name))
                    is not None
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

    def _selecting(
        self, chain: QualifierChain, step: _Step, count: int, selecting: tuple[Candidate, ...]
    ) -> tuple[Candidate, ...]:
        """The types the first *count* names of *chain* select at *step*, unremoved.

        *selecting*, those they reach; failing that, what an alias a shorter
        prefix past the step's module route reaches selects beneath it, as
        for its target's spelling. Where no shorter prefix reaches a type,
        nothing is beneath one.
        """
        if not selecting and any(
            self._owners(step, shorter) or self._standing(shorter)
            for shorter in range(step.start + 1, count)
        ):
            prefix = replace(
                chain, segments=chain.segments[: count - 1], member=self._names[count - 1]
            )
            walk = _Walk(
                self._sources,
                self._site,
                (step,),
                self._route,
                prefix,
                self._names[:count],
                self._span,
            )
            # The prefixes it reads are this walk's own.
            walk._reached = self._reached
            selecting = walk._through_prefixes(
                step, prefix, LookupKind.TYPE, owners_within=step.start
            ).candidates
        return self._unremoved(Reading(selecting)).candidates

    def _ambiguous(
        self, candidates: tuple[Candidate, ...], names: ScopePath, span: SourceSpan
    ) -> AmbiguousQualificationError:
        """Return the error for spelling *names*, at *span*, selecting several *candidates*."""
        chain = self._chain
        return AmbiguousQualificationError.for_origins(
            self._route,
            names,
            (candidate.origin for candidate in candidates),
            anchored=chain is not None and chain.anchored,
            span=span,
            reader=self._sources.reader(),
        )


def _unknown(
    chain: QualifierChain,
    names: ScopePath,
    origins: Callable[[_Step, ScopePath], frozenset[QName]],
    written: Callable[[QualifierChain], _Walk],
) -> AglScopeError:
    """An unknown member of the longest prefix naming something, else an unknown qualifier.

    A prefix names something when its walk (*written*) names a scope or type
    by *origins*, or reaches a type -- through an alias, the path it stands
    for -- whatever its reading's verdict, as a visible path is. A prefix
    naming a value but no qualifier ends the search: a function, binding or
    injected enum member is never a qualifier.
    """
    for length in range(len(chain.segments), 0, -1):
        walk = written(replace(chain, segments=chain.segments[: length - 1]))
        if walk.named(origins) or walk.find(LookupKind.TYPE) is not None:
            spelled = replace(chain, segments=chain.segments[:length])
            return UnknownMemberError(
                render_qualified_name(spelled, names[length]), span=chain.span
            )
        if walk.find(LookupKind.VALUE) is not None:
            break
    return UnknownQualifierError(render_qualifier_path(chain), span=chain.span)


def _reached_as(reading: Reading, owner: Candidate) -> Reading:
    """*reading*, what lies beneath *owner*, reached the ways *owner* and each candidate were."""
    if owner.hiding == NOT_HIDDEN:
        return reading
    return Reading(
        tuple(
            replace(
                candidate,
                hiding=frozenset(way | also for way in owner.hiding for also in candidate.hiding),
            )
            for candidate in reading.candidates
        ),
        reading.refusals,
    )


def _decided(
    candidates: Iterable[Candidate], identity: Callable[[DeclarationKey], object]
) -> Candidate | tuple[Candidate, ...] | None:
    """The one candidate selected, own first; every competing one when several distinct ones do.

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
    # Each declaration is read once, however often it is reached.
    identities = {
        key: identity(key)
        for key in {candidate.target.key for candidate in competing}
        if key is not None
    }
    distinct: dict[object, list[Candidate]] = {}
    for candidate in competing:
        target = candidate.target
        key = target.constructor if target.key is None else identities[target.key]
        distinct.setdefault(key, []).append(candidate)
    if len(distinct) > 1:
        return tuple(candidate for reached in distinct.values() for candidate in reached)
    if not distinct:
        return None
    ((named, reached),) = distinct.items()

    def spelled_first(
        candidate: Candidate,
    ) -> tuple[bool, tuple[tuple[str, ...], ScopePath, str] | None]:
        key = candidate.target.key
        return key != named, None if key is None else (key[0].segments, key[1], key[2])

    return min(reached, key=spelled_first)


def _by_constructor(candidates: Iterable[Candidate]) -> dict[ConstructorRef, Layers] | None:
    """*candidates*' constructors, each with every layer reaching it; ``None`` if one names none."""
    constructors: dict[ConstructorRef, Layers] = {}
    for candidate in candidates:
        constructor = candidate.target.constructor
        if constructor is None:
            return None
        add_layers(constructors, constructor, (candidate.layer,))
    return constructors


def _is_beneath(
    target: QualifiedTarget | Candidate | tuple[Candidate, ...] | None,
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
