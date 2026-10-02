"""Constructor owners of type paths, resolved by declaration identity.

A type path qualifies constructors (``Owner::Name``). What it selects is read
from the declaration it names; an alias is followed to its target: the
declaration scope selects for the target spelling in type position at the
alias's own declaration (:data:`AliasTargets`), never where the alias is
used. A structural target -- not a type name, or the bare name of one of the
alias's own type parameters -- selects nothing; a target selecting no
declaration is presumed constructible, leaving the verdict to typecheck.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import NamedTuple, Protocol

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    QName,
)
from agm.agl.scope.symbols import (
    BareAtom,
    ConstructorRef,
    DeclarationKey,
    DeclarationSelection,
    ScopePath,
    TypeOwner,
    TypeSelection,
    TypeTarget,
    dedupe_constructor_candidates,
)
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.scope.symbols import to_bare_path as _path
from agm.agl.scope.type_names import (
    is_nominal_type_expr,
)
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    Item,
    RecordDef,
    TypeAlias,
    VariantDef,
    VariantRef,
    static_type_items,
)
from agm.agl.syntax.types import (
    AppliedT,
    ArrayT,
    DictT,
    FuncT,
    NameT,
    TypeExpr,
    named_builtin_type,
    substitute_type_names,
)

__all__ = [
    "DeclaredBeneath",
    "ReachedPaths",
    "TypeOwnerIndex",
    "beneath",
    "declared_member_scopes",
    "injected_members",
    "owned_constructors",
    "retired_member_scopes",
    "root_type_names",
]

AliasTargets = Callable[[QName, TypeAlias, NameT | AppliedT], TypeSelection | None]
"""What scope selects for alias *qname*'s nominal target *spelling* where declared.

``None`` when scope selects none: a built-in type name.
"""


class CurrentTypeSelection(Protocol):
    """What a type name or member reference spelled in one module scope selects now, if anything.

    Every use is read when *every_use*; otherwise those the read in progress sees.
    """

    def __call__(
        self,
        module_id: ModuleId,
        scope_path: ScopePath,
        spelling: NameT | AppliedT | VariantRef,
        *,
        every_use: bool,
    ) -> TypeSelection | None: ...


class ReachedPaths(Protocol):
    """The paths among *paths* beneath alias *qname*'s target *spelling* reaches where declared.

    A path is reached unless a ``hiding`` visible at the alias's site removed
    ``<spelling>::path``'s whole path. Every use is read when *every_use*;
    otherwise those the read in progress sees.
    """

    def __call__(
        self,
        qname: QName,
        spelling: NameT | AppliedT,
        paths: Collection[ScopePath],
        *,
        every_use: bool,
    ) -> frozenset[ScopePath]: ...


DeclaredBeneath = Callable[[QName], Collection[ScopePath]]
"""The paths, relative to type or scope *qname*, of the declarations its module makes beneath it."""

DeclaredPaths = Callable[[ModuleId], Mapping[BareAtom, QName]]
"""Map each declaration module *module_id* writes beneath another path to the full path there.

``def Geo::m`` with ``type Geo = Base`` (``Base`` another module's) is declared at ``Base``'s
``Base::m``.
"""


ReadView = Callable[[ModuleId], object]
"""What reads made now in module *module_id* see of its uses and recorded scope paths.

Reads with equal views see the same of both.
"""


class BuiltinScopes(Protocol):
    """The scope paths built-in type *name* names as a qualifier where alias *qname* is declared.

    Every use is read when *every_use*; otherwise those the read in progress sees.
    """

    def __call__(self, qname: QName, name: str, *, every_use: bool) -> frozenset[QName]: ...


Denoted = tuple[object, ...] | TypeExpr
"""A normalized denoted type (:meth:`TypeOwnerIndex.denotation`): a scalar type expression,
or a tagged tuple of normalized parts."""

AliasSelection = tuple[QName | None, TypeExpr]
"""An alias's selected target declaration, and the type expression the alias stands for.

No declaration for a structural expression, nor for a type name scope
selects none for.
"""


class AliasReach(NamedTuple):
    """What a path beneath an alias stands for (:meth:`TypeOwnerIndex.alias_reach`).

    ``paths`` are the full paths it stands for; ``sites`` the modules
    declaring the aliases on the way, the nearest first, each of which may
    write a declaration there (:meth:`TypeOwnerIndex.written_beneath`);
    ``hidden`` whether a ``hiding`` at one of those aliases' sites removed it.
    """

    paths: tuple[QName, ...]
    sites: tuple[ModuleId, ...]
    hidden: bool


class TypeOwnerIndex:
    """Memoized :class:`TypeOwner` of every program type path.

    Record, exception, and inline enum member paths own their own
    constructors; an enum path owns only the inline members its scope
    declares, a referenced member staying at its own path; an alias path owns
    its own constructor and selects through the owner of its target.
    *alias_targets* answers scope's decision for an alias's target;
    *declared_beneath* which declarations lie beneath a type's path and
    *reached_paths* which of them an alias of it reaches; *builtin_scopes*
    the scopes an alias of a built-in type reads paths beneath;
    *current_selection* what a retained alias's spelling or an enum's member
    reference selects now; *declared_paths* the declarations a module
    writes beneath another path, and the full paths they are declared at;
    *read_view* what tells a read of a module from a later one seeing
    otherwise. *retained* supplies the owners of *retained_module*'s paths
    that earlier REPL entries declared, already resolved against the
    declarations they saw.
    """

    def __init__(
        self,
        *,
        all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
        constructor_refs: Mapping[QName, ConstructorRef],
        alias_targets: AliasTargets,
        declared_beneath: DeclaredBeneath,
        reached_paths: ReachedPaths,
        builtin_scopes: BuiltinScopes,
        current_selection: CurrentTypeSelection,
        declared_paths: DeclaredPaths,
        read_view: ReadView,
        retained_module: ModuleId | None = None,
        retained: Mapping[ScopePath, TypeOwner] | None = None,
    ) -> None:
        self._all_public_types = all_public_types
        self._constructor_refs = constructor_refs
        self._decided_targets = alias_targets
        self._declared_beneath = declared_beneath
        self._reached_paths = reached_paths
        self._builtin_scopes = builtin_scopes
        self._current_selection = current_selection
        self._declared_paths = declared_paths
        self._read_view = read_view
        self._retained_module = retained_module
        self._retained = retained or {}
        self._owners: dict[QName, TypeOwner] = {}
        self._alias_targets: dict[QName, AliasSelection] = {}
        self._referenced_members: dict[tuple[ModuleId, int], tuple[ConstructorRef, ...]] = {}
        # Aliases and enums being resolved (:meth:`settled`).
        self._resolving: set[QName] = set()
        # Every type path resolved, in order (:meth:`resolved_since`).
        self._resolved: list[QName] = []
        # What the outermost read of a retained alias in progress projected,
        # by alias and the view it was projected under (:meth:`_view`).
        self._projected: dict[tuple[QName, object], TypeOwner] | None = None

    def with_retained(
        self, module_id: ModuleId, retained: Mapping[ScopePath, TypeOwner]
    ) -> TypeOwnerIndex:
        """Return an index over the same program that also sees *module_id*'s retained paths."""
        return TypeOwnerIndex(
            all_public_types=self._all_public_types,
            constructor_refs=self._constructor_refs,
            alias_targets=self._decided_targets,
            declared_beneath=self._declared_beneath,
            reached_paths=self._reached_paths,
            builtin_scopes=self._builtin_scopes,
            current_selection=self._current_selection,
            declared_paths=self._declared_paths,
            read_view=self._read_view,
            retained_module=module_id,
            retained=retained,
        )

    def forget(self, modules: Collection[ModuleId]) -> None:
        """Forget what paths of *modules* select: their headers are prepared again."""
        self._owners = {q: o for q, o in self._owners.items() if q[0] not in modules}
        self._alias_targets = {q: t for q, t in self._alias_targets.items() if q[0] not in modules}
        self._referenced_members = {
            key: refs for key, refs in self._referenced_members.items() if key[0] not in modules
        }

    def is_declared(self, qname: QName) -> bool:
        """Whether *qname* names a type or an inline enum member."""
        return (
            qname in self._all_public_types
            or self._retained_owner(qname) is not None
            or self._member_constructor(qname) is not None
        )

    def _retained_owner(self, qname: QName) -> TypeOwner | None:
        """Return the owner an earlier REPL entry resolved for *qname*, if it declared it."""
        return self._retained.get(_path(qname[1])) if qname[0] == self._retained_module else None

    def _member_constructor(self, qname: QName) -> ConstructorRef | None:
        """Return the constructor of the record, exception, or inline member at *qname*.

        An inline member of an enum an earlier REPL entry declared is read from
        that enum's retained owner; an alias's projected member declares nothing.
        """
        constructor = self._constructor_refs.get(qname)
        if constructor is not None or qname[0] != self._retained_module:
            return constructor
        path = _path(qname[1])
        enum = self._retained.get(path[:-1])
        return None if enum is None or enum.alias is not None else enum.members.get(path[-1])

    def referenced_member_refs(
        self, qname: QName, member: VariantRef
    ) -> tuple[ConstructorRef, ...]:
        """Return every record constructor member reference *member* transparently denotes.

        *member* belongs to the enum declared at *qname* and selects what
        scope selects for its spelling there.
        """
        key = (qname[0], member.node_id)
        if key not in self._referenced_members:
            selection = self._current_selection(
                qname[0], _path(qname[1])[:-1], member, every_use=True
            )
            target = None if selection is None else self.declared_path(selection)
            self._referenced_members[key] = (
                () if target is None else self._constructors_through(target)
            )
        return self._referenced_members[key]

    def owner(self, qname: QName) -> TypeOwner | None:
        """Return what type path *qname* selects, or ``None`` when it names no type.

        A retained alias's own declaration and target identity
        (:attr:`TypeOwner.target`) are never re-selected: they are exactly
        what the declaring entry resolved, forever, however later entries
        redeclare or import around them -- a later declaration at the same
        path is a distinct declaration (:attr:`TypeOwner.decl_node_id`),
        never mistaken for it. What a retained alias's spelling selects now
        follows :meth:`_current_retained_owner`.
        """
        declaration = self._all_public_types.get(qname)
        if declaration is not None:
            return self.declared_owner(qname, declaration)
        retained = self._retained_owner(qname)
        if retained is not None:
            return self._current_retained_owner(qname, retained)
        member = self._member_constructor(qname)
        return (
            None
            if member is None
            else TypeOwner(member, member.owner_decl_node_id, frozenset({member.owner_name}))
        )

    def _current_retained_owner(self, qname: QName, retained: TypeOwner) -> TypeOwner:
        """What a retained owner selects now.

        A non-alias *retained*, or one whose target cannot be resolved
        (record, enum, structural, or a same-entry cycle), stands unchanged;
        an alias of a built-in type reads the scopes its target's name names
        at this entry's site. Otherwise, while its target declaration stands
        and its own spelling still selects exactly that target at this
        entry's site (*current_selection*), the target's current members are
        re-projected through that spelling; once the target is gone (a later
        entry retired or redeclared its path) or the spelling selects anything
        else (shadowed, ambiguous, or the route is gone), *retained* freezes
        at its declaration-time members/hidden. The alias's site is read with
        the uses the read asking for it sees: a use's own, only those written
        before it.
        """
        alias = retained.alias
        if alias is None:
            return retained
        with self._projecting() as projected:
            key = (qname, self._view(qname[0]))
            found = projected.get(key)
            if found is None:
                found = projected[key] = self._projected_owner(qname, retained, alias)
            return found

    @contextmanager
    def _projecting(self) -> Iterator[dict[tuple[QName, object], TypeOwner]]:
        """Keep what one outermost read of a retained alias projects until it ends.

        Projecting an alias reads each path beneath its target through the
        alias before it on its chain, which projects that one again.
        """
        projected = self._projected
        if projected is not None:
            yield projected
            return
        self._projected = projected = {}
        try:
            yield projected
        finally:
            self._projected = None

    def _view(self, module_id: ModuleId) -> object:
        """What a read made now in *module_id* sees that another of the same outermost read may not.

        The module's uses and recorded scope paths (*read_view*), and which
        types are resolved and which presumed (:meth:`settled`): types
        resolve in order and those being resolved nest, so the two counts
        name both. One outermost read changes nothing else a projection
        reads, and a count once passed never returns: a projection is found
        again only where reading it afresh gives the same.
        """
        return self._read_view(module_id), len(self._resolved), len(self._resolving)

    def _projected_owner(self, qname: QName, retained: TypeOwner, alias: TypeAlias) -> TypeOwner:
        """Project retained alias *alias* at *qname* (:meth:`_current_retained_owner`)."""
        target = retained.target
        if target is None:
            return self._builtin_owner(qname, retained, every_use=False)
        current = self.owner(target.qname)
        if current is None or current.decl_node_id != target.decl_node_id:
            return retained
        spelling = retained.stands_for
        module_id, atom = qname
        if (
            not is_nominal_type_expr(spelling, alias.type_params)
            or (
                selection := self._current_selection(
                    module_id, _path(atom)[:-1], spelling, every_use=False
                )
            )
            is None
            or self.declared_path(selection) != target.qname
        ):
            return retained
        reachable, hidden = self._projection(
            qname, spelling, target.qname, current, every_use=False
        )
        return replace(retained, members=reachable, hidden=hidden, scopes=current.scopes)

    def _projection(
        self,
        qname: QName,
        spelling: NameT | AppliedT,
        target_qname: QName,
        target_owner: TypeOwner,
        *,
        every_use: bool,
    ) -> tuple[Mapping[str, ConstructorRef], frozenset[ScopePath]]:
        """Return alias *qname*'s reachable ``members``/``hidden``, projected from its target.

        *target_owner* is what the target, at *target_qname*, selects. Keeps
        the paths beneath the target that *spelling* -- the alias's own
        nominal target spelling -- reaches where the alias is declared
        (*reached_paths*, read with every use when *every_use*); a path the target itself cannot
        reach stays hidden.
        """
        final = self.final_target(target_qname)
        beneath = () if final is None else self._declared_beneath(final)
        candidates = [path for path in beneath if path not in target_owner.hidden]
        reached = self._reached_paths(qname, spelling, candidates, every_use=every_use)
        hidden = target_owner.hidden | frozenset(candidates).difference(reached)
        reachable = {
            name: member for name, member in target_owner.members.items() if (name,) not in hidden
        }
        return reachable, hidden

    def _alias_chain(self, qname: QName) -> Iterator[tuple[QName, TypeOwner]]:
        """Yield *qname* and each alias target along its chain, with what each declares.

        The chain ends at a path naming no type, an alias with no nominal
        target, a target a later REPL entry redeclared (a retained alias's
        frozen target), or a path it passed before. A retained alias's
        declaration and target are what it declared (:meth:`owner`): what
        its spelling selects now is not read.
        """
        seen: set[QName] = set()
        current, expected = qname, None
        while current not in seen:
            seen.add(current)
            retained = None if current in self._all_public_types else self._retained_owner(current)
            owner = self.owner(current) if retained is None else retained
            if owner is None or (expected is not None and owner.decl_node_id != expected):
                return
            yield current, owner
            if owner.target is None:
                return
            current, expected = owner.target.qname, owner.target.decl_node_id

    def cyclic(self, qname: QName) -> bool:
        """Whether alias *qname*'s chain (:meth:`_alias_chain`) leads back to *qname*."""
        last = None
        for _current, last in self._alias_chain(qname):
            pass
        return last is not None and last.target is not None and last.target.qname == qname

    def final_target(self, qname: QName) -> QName | None:
        """Return the type path *qname*'s alias chain ends at: *qname* itself when no alias.

        ``None`` when the chain (:meth:`_alias_chain`) ends at no type or an
        alias, a cycle included.
        """
        final = None
        for current, owner in self._alias_chain(qname):
            final = current if owner.alias is None else None
        return final

    def alias_reach(self, alias: QName, table: TypeOwner, path: ScopePath) -> AliasReach:
        """Return what *path* beneath alias *alias*, which selects *table*, stands for.

        Beneath an alias of a built-in type, that path beneath each of its
        :attr:`~TypeOwner.scopes`, its own module's first. Any other alias
        stands for its target's path (:meth:`_target_beneath`), where the
        modules declaring the aliases on the way write declarations too;
        nothing when an alias names no nominal target.
        """
        if table.builtin_name is not None:
            own = sorted((scope for scope in table.scopes if scope[0] == alias[0]), key=str)
            scopes = (*own, *sorted(table.scopes.difference(own), key=str))
            return AliasReach(
                tuple((module_id, _atom((*_path(atom), *path))) for module_id, atom in scopes),
                (),
                table.hides(path),
            )
        beneath = self._target_beneath(alias, table, path)
        if beneath is None:
            return AliasReach((), (), False)
        qname, hidden, sites = beneath
        return AliasReach((qname,), sites, hidden)

    def _target_beneath(
        self, alias: QName, table: TypeOwner, path: ScopePath
    ) -> tuple[QName, bool, tuple[ModuleId, ...]] | None:
        """Return the full path *path* beneath alias *alias* stands for, if it is hidden, its sites.

        *table* is what *alias* selects. *path* is read beneath the type the
        alias chain ends at, and a prefix of it there naming another alias
        stands for that alias's target in turn. It is hidden when a ``hiding``
        at one of those aliases' sites removed it or a prefix of it. The sites
        are the modules declaring those aliases, the first passed first.
        ``None`` when an alias names no nominal target.
        """
        hidden = False
        sites: dict[ModuleId, None] = {}
        while True:
            hidden = hidden or table.hides(path)
            final = self.final_target(alias)
            if final is None:
                return None
            for current, owner in self._alias_chain(alias):
                if owner.alias is not None:
                    sites[current[0]] = None
            module_id, atom = final
            base = _path(atom)
            for end in range(1, len(path)):
                inner = (module_id, _atom((*base, *path[:end])))
                inner_table = self.owner(inner)
                if inner_table is not None and inner_table.alias is not None:
                    alias, table, path = inner, inner_table, path[end:]
                    break
            else:
                return (module_id, _atom((*base, *path))), hidden, tuple(sites)

    def written_beneath(self, module_id: ModuleId, qname: QName) -> Iterator[ScopePath]:
        """Yield where *module_id* writes a declaration otherwise, relative to full path *qname*.

        ``def Geo::m`` with ``type Geo = Base`` is declared at ``Base::m``:
        at ``m`` beneath ``Base``, and at no path beneath ``Base::m``.
        """
        target, atom = qname
        base = _path(atom)
        for declared, declared_atom in self._declared_paths(module_id).values():
            path = _path(declared_atom)
            if declared == target and path[: len(base)] == base:
                yield path[len(base) :]

    def placement(self, qname: QName) -> QName:
        """Return the full path the declaration at *qname* is declared at (:meth:`declaration`).

        One its module writes beneath another module's type, directly or
        through an alias, is declared at that type's path there.
        """
        return self._declared_paths(qname[0]).get(qname[1]) or self.declaration(qname)

    def scopes_of(self, qname: QName) -> frozenset[QName]:
        """Return the scopes type *qname* stands for beside its path (:attr:`TypeOwner.scopes`)."""
        owner = self.owner(qname)
        return frozenset() if owner is None else owner.scopes

    def path_target(self, qname: QName) -> QName:
        """Return the full path *qname* stands for: beneath an alias, its target's path there."""
        module_id, atom = qname
        path = _path(atom)
        for end in range(1, len(path)):
            alias = (module_id, _atom(path[:end]))
            owner = self.owner(alias)
            if owner is not None and owner.alias is not None:
                beneath = self._target_beneath(alias, owner, path[end:])
                return qname if beneath is None else beneath[0]
        return qname

    def identity(self, qname: QName) -> QName:
        """Return the declaration type path *qname* names: a renaming alias's is its target's.

        An alias passing its type parameters through to its target
        (:attr:`TypeOwner.renames`) is another name for it, along the chain
        (:meth:`_alias_chain`); any other alias is a type of its own.
        """
        named = qname
        for named, owner in self._alias_chain(qname):
            if owner.alias is not None and not owner.renames:
                break
        return named

    def denotation(self, qname: QName) -> Denoted | None:
        """The type the alias declared at *qname* denotes, unless it renames its target.

        Normalized: each named head is the declaration it selects where it is
        spelled, read through aliases with their parameters substituted
        (``Box[path]`` is ``Box[text]``), and the alias's own parameters are
        positions. ``None`` for any other path.
        """
        declaration = self._all_public_types.get(qname)
        if (
            not isinstance(declaration, TypeAlias)
            or self.declared_owner(qname, declaration).renames
        ):
            return None
        return self._denoted(qname, declaration, frozenset({qname}))

    def _denoted(
        self,
        qname: QName,
        alias: TypeAlias,
        passed: frozenset[QName],
        arguments: tuple[Denoted, ...] | None = None,
    ) -> Denoted:
        """What alias *alias* at *qname* denotes, applied to *arguments*.

        Its own parameters' positions by default. A head naming nothing, or
        an alias *passed* on the way (an ill-founded chain), is unresolved.
        """
        positions = tuple(("parameter", index) for index in range(len(alias.type_params)))
        bound: dict[str, Denoted] = dict(
            zip(alias.type_params, positions if arguments is None else arguments, strict=False)
        )

        def denoted(expr: TypeExpr) -> Denoted:
            if isinstance(expr, NameT) and expr.qualifier is None and expr.name in bound:
                return bound[expr.name]
            if isinstance(expr, (NameT, AppliedT)):
                args = tuple(map(denoted, expr.args)) if isinstance(expr, AppliedT) else ()
                selection = self._decided_targets(qname, alias, expr)
                head = None if selection is None else self.declared_path(selection)
                target = None if head is None else self._all_public_types.get(head)
                if head is None or head in passed:
                    return ("unresolved", qname, expr)
                if isinstance(target, TypeAlias):
                    return self._denoted(head, target, passed | {head}, args)
                return ("applied", head, args)
            if isinstance(expr, ArrayT):
                return ("array", denoted(expr.elem))
            if isinstance(expr, DictT):
                return ("dict", denoted(expr.key), denoted(expr.value))
            if isinstance(expr, FuncT):
                return ("function", tuple(map(denoted, expr.params)), denoted(expr.result))
            return expr

        return denoted(alias.type_expr)

    def declaration(self, qname: QName) -> QName:
        """Return the declaration full path *qname* names: its :meth:`path_target`'s identity."""
        return self.identity(self.path_target(qname))

    def constructor_identity(self, constructor: ConstructorRef) -> ConstructorRef:
        """Return the constructor *constructor* names: a renaming alias's is its target's.

        A member an alias selects is the target's member.
        """
        named = self.identity(constructor.qname)
        owner = self.owner(named)
        if owner is not None and constructor.member is not None:
            return owner.members[constructor.member]
        if owner is None or named == constructor.qname or owner.constructor is None:
            return constructor
        return owner.constructor

    def declared_owner(
        self, qname: QName, declaration: RecordDef | EnumDef | ExceptionDef | TypeAlias
    ) -> TypeOwner:
        """Return what *declaration*, declared at type path *qname*, selects."""
        owner = self._owners.get(qname)
        if owner is None:
            owner = self._resolve(qname, declaration)
            self._owners[qname] = owner
            self._resolved.append(qname)
        return owner

    @property
    def resolving(self) -> bool:
        """Whether a type is being resolved; reads made meanwhile presume it (:meth:`settled`)."""
        return bool(self._resolving)

    def settled(self, qname: QName) -> bool:
        """Whether reading *qname* now gives what it finally selects.

        An alias being resolved is presumed, naming no target yet; an enum
        whose member references are being selected, with its inline members
        alone. While one is, resolving another may read it: only one already
        resolved is read.
        """
        return qname not in self._resolving and (
            not self._resolving or qname in self._owners or qname not in self._all_public_types
        )

    def resolved_since(self, count: int) -> Sequence[QName]:
        """The type paths resolved after the first *count* were, in order.

        While a type is being resolved, one not :meth:`settled` becomes so
        only once it is resolved.
        """
        return self._resolved[count:]

    def alias_constructor(self, alias: TypeAlias, qname: QName) -> ConstructorRef | None:
        """Return alias *qname*'s constructor, unless it denotes a structural type or an enum."""
        owner = self.declared_owner(qname, alias)
        return owner.constructor if owner.constructible else None

    def _resolve(
        self, qname: QName, declaration: RecordDef | EnumDef | ExceptionDef | TypeAlias
    ) -> TypeOwner:
        module_id, atom = qname
        path = _path(atom)
        arity = len(declaration.type_param_slots)
        if isinstance(declaration, (RecordDef, ExceptionDef)):
            return TypeOwner(
                self._constructor_refs[qname],
                declaration.node_id,
                frozenset({declaration.name}),
                arity=arity,
            )
        if isinstance(declaration, EnumDef):
            members = self._enum_members(module_id, path, declaration)
            # Selecting a member reference may read this enum's own path
            # (``E::Item``): it sees the inline members alone.
            self._owners[qname] = TypeOwner(None, declaration.node_id, members=members, arity=arity)
            self._resolving.add(qname)
            referenced = [
                (member, self.referenced_member_refs(qname, member))
                for member in declaration.members
                if isinstance(member, VariantRef)
            ]
            self._resolving.discard(qname)
            return TypeOwner(
                None,
                declaration.node_id,
                members=members,
                referenced=frozenset(
                    name
                    for member, refs in referenced
                    if refs
                    for name in (member.chain.member, *(ref.owner_name for ref in refs))
                ),
                injected=dedupe_constructor_candidates(
                    ref for _, refs in referenced for ref in refs
                ),
                # A reference nesting its record directly beneath the enum's
                # own path selects through it like an inline member.
                own_path_referenced=frozenset(
                    ref.owner_name
                    for _, refs in referenced
                    for ref in refs
                    if ref.owner_module_id == module_id and ref.owner_path == path
                ),
                arity=arity,
            )
        constructor = ConstructorRef.for_alias(declaration, module_id, path[:-1])
        # Typecheck judges a target scope selects no declaration for, so the
        # alias is presumed constructible; an alias cycle meets it that way.
        presumed = TypeOwner(
            constructor,
            declaration.node_id,
            frozenset({declaration.name}),
            alias=declaration,
            arity=arity,
            stands_for=declaration.type_expr,
        )
        self._owners[qname] = presumed
        self._resolving.add(qname)
        owner = self._resolve_alias(qname, declaration, presumed)
        self._resolving.discard(qname)
        return owner

    def _resolve_alias(
        self, qname: QName, declaration: TypeAlias, presumed: TypeOwner
    ) -> TypeOwner:
        """Return what alias *declaration* at *qname* selects, *presumed* meanwhile."""
        constructor, arity = presumed.constructor, presumed.arity
        target_qname, type_expr = self._alias_selection(qname, declaration)
        if not is_nominal_type_expr(type_expr, declaration.type_params):
            structural = TypeOwner(
                None, declaration.node_id, alias=declaration, arity=arity, stands_for=type_expr
            )
            return self._builtin_owner(qname, structural, every_use=True)
        target = None if target_qname is None else self.owner(target_qname)
        if target_qname is None or target is None:
            return replace(presumed, stands_for=type_expr)
        reachable, hidden = self._projection(qname, type_expr, target_qname, target, every_use=True)
        return TypeOwner(
            constructor,
            declaration.node_id,
            target.names | {declaration.name} if target.names else frozenset(),
            reachable,
            target.referenced,
            declaration,
            hidden=hidden,
            target=TypeTarget(target_qname, target.decl_node_id),
            own_path_referenced=target.own_path_referenced,
            arity=arity,
            builtin=_applied_builtin(target, type_expr),
            scopes=target.scopes,
            stands_for=type_expr,
        )

    def _builtin_owner(self, qname: QName, owner: TypeOwner, *, every_use: bool) -> TypeOwner:
        """Return *owner* of the alias at *qname*, with its scopes when it stands for a built-in.

        Those the built-in type's name names as a qualifier where the alias
        is declared, read with every use when *every_use*, and that module's own path spelled
        so, however its declarations beneath are spelled; a path beneath them
        a ``hiding`` there removed is hidden. *owner* stands for any other
        target.
        """
        builtin = owner.stands_for
        spelling = None if builtin is None else named_builtin_type(builtin)
        if spelling is None:
            return owner
        scopes = self._builtin_scopes(qname, spelling.name, every_use=every_use) | {
            (qname[0], _atom((spelling.name,)))
        }
        beneath = {path for scope in scopes for path in self._declared_beneath(scope)}
        reached = self._reached_paths(qname, spelling, beneath, every_use=every_use)
        return replace(
            owner, builtin=builtin, scopes=scopes, hidden=frozenset(beneath).difference(reached)
        )

    def _alias_selection(self, qname: QName, alias: TypeAlias) -> AliasSelection:
        """Return the declaration alias *qname*'s target denotes, as scope selected it.

        Scope's selection (*alias_targets*) may spell a member through its
        owner's alias (``O::Member``); the target is the member declaration
        it denotes. See :data:`AliasSelection`.
        """
        if qname not in self._alias_targets:
            self._alias_targets[qname] = self._spelled_selection(qname, alias, alias.type_expr)
        return self._alias_targets[qname]

    def _spelled_selection(
        self, qname: QName, alias: TypeAlias, spelling: TypeExpr
    ) -> AliasSelection:
        """Return what *spelling*, written in alias *alias* at *qname*, denotes and stands for.

        An alias standing for one of its type parameters (``type Id[T] = T``),
        applied, stands for its argument there.
        """
        if not is_nominal_type_expr(spelling, alias.type_params):
            return None, spelling
        selection = self._decided_targets(qname, alias, spelling)
        target = None if selection is None else self.declared_path(selection)
        parameter = None if target is None else self.projected_parameter(target)
        if (
            parameter is not None
            and isinstance(spelling, AppliedT)
            and parameter[0] < len(spelling.args)
        ):
            return self._spelled_selection(qname, alias, spelling.args[parameter[0]])
        return target, spelling

    def projected_parameter(self, qname: QName) -> tuple[int, int] | None:
        """The position of the type parameter alias *qname* stands for, and how many it takes.

        ``None`` unless it stands for one: ``type Id[T] = T``, or another name
        for such an alias.
        """
        owner = self.owner(self.identity(qname))
        declaration = None if owner is None else owner.alias
        spelling = None if owner is None else owner.stands_for
        if declaration is None or not isinstance(spelling, NameT) or spelling.qualifier is not None:
            return None
        params = declaration.type_params
        return (params.index(spelling.name), len(params)) if spelling.name in params else None

    def declared_path(self, selection: TypeSelection) -> QName | None:
        """Return the declaration *selection* denotes: its own path, or an owner's inline member.

        ``None`` for a declaration that is no type (a value a type spelling
        selects instead).
        """
        if isinstance(selection, DeclarationSelection):
            module_id, path, name = selection.key
            qname = (module_id, _atom((*path, name)))
            return qname if self.is_declared(qname) else None
        member = self.owner_member(selection.owner, selection.member)
        return (
            None
            if member is None
            else (member.owner_module_id, _atom((*member.owner_path, member.owner_name)))
        )

    def owner_member(self, owner: DeclarationKey, name: str) -> ConstructorRef | None:
        """Return the member type *owner* selects as *name*: an alias's is its target's member."""
        module_id, path, owner_name = owner
        table = self.owner((module_id, _atom((*path, owner_name))))
        return None if table is None else table.members.get(name)

    def _enum_members(
        self, module_id: ModuleId, path: ScopePath, declaration: EnumDef
    ) -> dict[str, ConstructorRef]:
        """Return the members an enum's scope declares -- its inline members -- by name."""
        return {
            member.name: self._constructor_refs[(module_id, _atom((*path, member.name)))]
            for member in declaration.members
            if isinstance(member, VariantDef)
        }

    def _constructors_through(self, qname: QName) -> tuple[ConstructorRef, ...]:
        """Follow aliases from *qname* to the record constructor at the end of the chain.

        A retained path follows the target identity an earlier REPL entry
        resolved for it (:attr:`TypeOwner.target`), never re-selecting it.
        """
        seen: set[QName] = set()
        current: QName | None = qname
        while current is not None and current not in seen:
            seen.add(current)
            constructor = self._member_constructor(current)
            if constructor is not None:
                return (constructor,)
            declaration = self._all_public_types.get(current)
            if declaration is None:
                retained = self._retained_owner(current)
                if retained is None:
                    return ()
                if retained.alias is None:
                    return () if retained.constructor is None else (retained.constructor,)
                current = None if retained.target is None else retained.target.qname
                continue
            if not isinstance(declaration, TypeAlias):
                return ()
            current = self._alias_selection(current, declaration)[0]
        return ()


def _applied_builtin(target: TypeOwner, spelling: NameT | AppliedT) -> TypeExpr | None:
    """The built-in type alias *target*, spelled *spelling*, is: its own, with those arguments."""
    if target.builtin is None or target.alias is None:
        return None
    arguments = spelling.args if isinstance(spelling, AppliedT) else ()
    return substitute_type_names(
        target.builtin, dict(zip(target.alias.type_params, arguments, strict=False))
    )


def declared_member_scopes(items: tuple[Item, ...]) -> dict[ScopePath, frozenset[str]]:
    """Map each type path *items* declare to the inline member names it owns scopes for."""
    return {
        (*(segment.name for segment in item.scope_path), item.name): frozenset(
            member.name for member in item.members if isinstance(member, VariantDef)
        )
        if isinstance(item, EnumDef)
        else frozenset()
        for item in static_type_items(items)
    }


def retired_member_scopes(
    retained: Mapping[ScopePath, TypeOwner], declared: Mapping[ScopePath, frozenset[str]]
) -> frozenset[ScopePath]:
    """Return the member scopes that redeclaring the *declared* type paths retires.

    A replaced enum's inline member keeps its scope only while the
    replacement declares an inline member of that name again (see
    :func:`declared_member_scopes`).
    """
    return frozenset(
        (*path, name)
        for path, names in declared.items()
        if (prior := retained.get(path)) is not None and prior.alias is None
        for name in prior.members.keys() - names
    )


def root_type_names(paths: Mapping[ScopePath, TypeOwner]) -> frozenset[str]:
    """Return the root-declared type names among *paths*'s keys.

    Shared by the REPL session (ambient type names across promoted entries)
    and the resolver (a REPL entry's own retained paths): a root path is a
    bare qualifier segment, so a later entry's ``Owner::member`` reaches it
    without importing it.
    """
    return frozenset(path[0] for path in paths if len(path) == 1)


def beneath(path: ScopePath, scopes: Iterable[ScopePath]) -> bool:
    """Whether *path* is one of *scopes* or lies inside one."""
    return any(path[: len(scope)] == scope for scope in scopes)


def owned_constructors(
    module_id: ModuleId, owners: Mapping[ScopePath, TypeOwner]
) -> Iterator[tuple[str, ConstructorRef, ScopePath, bool]]:
    """Yield ``(name, constructor, scope_path, bare)`` for module *module_id*'s type *owners*.

    A record, exception, or constructible alias is a candidate in its
    declaring scope, an enum's inline members in the enum's scope, and either
    is also bare when declared at the root. A root enum injects its referenced
    members bare, each only while it is the declaration *owners* hold at its
    own path.
    """
    for path, owner in owners.items():
        bare = len(path) == 1
        if owner.constructor is None:
            if owner.alias is None:
                for name, member in owner.members.items():
                    yield name, member, path, bare
                if bare:
                    for injected in _current_injected(module_id, owners, owner):
                        yield injected.owner_name, injected, (), True
        elif owner.constructible:
            yield path[-1], owner.constructor, path[:-1], bare


def injected_members(
    module_id: ModuleId, owners: Mapping[ScopePath, TypeOwner]
) -> Iterator[tuple[ScopePath, str, ConstructorRef]]:
    """Yield ``(step, name, constructor)`` for each member module *module_id*'s enums inject.

    An enum at path ``P`` injects its inline members, and its referenced
    members while it is the declaration *owners* hold at their paths, as
    bare names at its own step ``P[:-1]``.
    """
    for path, owner in owners.items():
        if owner.constructor is None and owner.alias is None:
            for name, member in owner.members.items():
                yield path[:-1], name, member
            for injected in _current_injected(module_id, owners, owner):
                yield path[:-1], injected.owner_name, injected


def _current_injected(
    module_id: ModuleId, owners: Mapping[ScopePath, TypeOwner], owner: TypeOwner
) -> Iterator[ConstructorRef]:
    """Yield enum *owner*'s referenced members still current in *owners* (:func:`_is_current`)."""
    return (injected for injected in owner.injected if _is_current(module_id, owners, injected))


def _is_current(
    module_id: ModuleId, owners: Mapping[ScopePath, TypeOwner], constructor: ConstructorRef
) -> bool:
    """Whether *constructor* is still what *owners* declare at its own path.

    Another module's declaration always is.
    """
    if constructor.owner_module_id != module_id:
        return True
    at_path = owners.get((*constructor.owner_path, constructor.owner_name))
    if at_path is not None:
        current = at_path.constructor
    else:
        enum = owners.get(constructor.owner_path)
        current = None if enum is None else enum.members.get(constructor.owner_name)
    return current is not None and current.owner_decl_node_id == constructor.owner_decl_node_id
