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

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import NamedTuple, Protocol

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    QName,
)
from agm.agl.scope.symbols import (
    ConstructorRef,
    DeclarationKey,
    DeclarationSelection,
    ScopePath,
    TypeOwner,
    TypeSelection,
    TypeTarget,
    declaration_qname,
    dedupe_constructor_candidates,
    passes_parameters,
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
    QualifierAnchor,
    QualifierChain,
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
    member_type_params,
    named_builtin_type,
    substitute_type_names,
)

__all__ = [
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
    """The member paths among *paths* alias *qname*'s target *spelling* reaches where declared.

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


ReadView = Callable[[ModuleId], object]
"""What reads made now in module *module_id* see of its uses.

Reads with equal views see the same of them.
"""


Denoted = tuple[object, ...] | TypeExpr
"""A normalized denoted type (:meth:`TypeOwnerIndex.denotation`): a scalar type expression,
or a tagged tuple of normalized parts."""


class AliasSelection(NamedTuple):
    """What an alias's target expression denotes, and what it reaches that through.

    ``target`` is the declaration it denotes and ``stands_for`` the type
    expression it stands for (:attr:`TypeOwner.stands_for`); ``through`` is
    the declaration a type name selects where the alias is declared, with
    that name: the alias reaches its paths. The two declarations differ
    where the alias applies a generic alias that is a type of its own. No
    declaration for a structural expression, nor for a type name scope
    selects none for.
    """

    target: QName | None
    stands_for: TypeExpr
    through: tuple[QName, NameT | AppliedT] | None


class TypeOwnerIndex:
    """Memoized :class:`TypeOwner` of every program type path.

    Record, exception, and inline enum member paths own their own
    constructors; an enum path owns only the inline members its scope
    declares, a referenced member staying at its own path; an alias path owns
    its own constructor and selects through the owner of its target.
    *alias_targets* answers scope's decision for an alias's target;
    *reached_paths* which of its target's members an alias reaches;
    *current_selection* what a retained alias's spelling or an enum's member
    reference selects now; *read_view* what tells a read of a module from a
    later one seeing otherwise. *retained* supplies the owners of *retained_module*'s paths
    that earlier REPL entries declared, already resolved against the
    declarations they saw.
    """

    def __init__(
        self,
        *,
        all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
        constructor_refs: Mapping[QName, ConstructorRef],
        alias_targets: AliasTargets,
        reached_paths: ReachedPaths,
        current_selection: CurrentTypeSelection,
        read_view: ReadView,
        retained_module: ModuleId | None = None,
        retained: Mapping[ScopePath, TypeOwner] | None = None,
    ) -> None:
        self._all_public_types = all_public_types
        self._constructor_refs = constructor_refs
        self._decided_targets = alias_targets
        self._reached_paths = reached_paths
        self._current_selection = current_selection
        self._read_view = read_view
        self._retained_module = retained_module
        self._retained = retained or {}
        self._owners: dict[QName, TypeOwner] = {}
        self._alias_targets: dict[QName, AliasSelection] = {}
        self._referenced_members: dict[tuple[ModuleId, int], tuple[ConstructorRef, ...]] = {}
        # Aliases and enums being resolved: a read meanwhile presumes an alias
        # names no target yet, and an enum has its inline members alone.
        self._resolving: set[QName] = set()
        # How many type paths have resolved (:meth:`_view`).
        self._resolved = 0
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
            reached_paths=self._reached_paths,
            current_selection=self._current_selection,
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
        (record, enum, built-in, structural, or a same-entry cycle), stands
        unchanged. Otherwise, while its target declaration stands
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

        The module's uses (*read_view*), and which
        types are resolved and which presumed (being resolved): types
        resolve in order and those being resolved nest, so the two counts
        name both. One outermost read changes nothing else a projection
        reads, and a count once passed never returns: a projection is found
        again only where reading it afresh gives the same.
        """
        return self._read_view(module_id), self._resolved, len(self._resolving)

    def _projected_owner(self, qname: QName, retained: TypeOwner, alias: TypeAlias) -> TypeOwner:
        """Project retained alias *alias* at *qname* (:meth:`_current_retained_owner`)."""
        target = retained.target
        if target is None:
            return retained
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
        reachable, hidden = self._projection(qname, spelling, current, every_use=False)
        return replace(retained, members=reachable, hidden=hidden)

    def _projection(
        self, qname: QName, spelling: NameT | AppliedT, target_owner: TypeOwner, *, every_use: bool
    ) -> tuple[Mapping[str, ConstructorRef], frozenset[ScopePath]]:
        """Return alias *qname*'s reachable ``members``/``hidden``, projected from its target.

        *target_owner* is what the target selects. Keeps the members of the
        target that *spelling* -- the alias's own nominal target spelling --
        reaches where the alias is declared (*reached_paths*, read with every
        use when *every_use*); a member the target itself cannot reach stays
        hidden.
        """
        candidates = [(name,) for name in target_owner.members]
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

    def identity(self, qname: QName) -> QName:
        """Return the declaration type path *qname* names: a renaming alias's is its target's.

        An alias passing its type parameters through to its target
        (:attr:`TypeOwner.renames`) is another name for it, along the chain
        (:meth:`_alias_chain`); any other alias is a type of its own.
        """
        named = qname
        for current, owner in self._alias_chain(qname):
            named = current
            if owner.alias is not None and not owner.renames:
                break
        return named

    def denotation(self, qname: QName) -> Denoted | None:
        """The type the alias *qname* names (:meth:`identity`) denotes, unless renaming its target.

        Normalized: each named head is the declaration it selects where it is
        spelled, read through aliases with their parameters substituted
        (``Box[path]`` is ``Box[text]``), and the alias's own parameters are
        positions. A member such an alias selects is that member at the
        alias's arguments: ``O::Som`` with ``type O = Opt[int]`` is
        ``("member", <O's denotation>, "Som")``. ``None`` for any other path.
        """
        named = self.identity(qname)
        declaration = self._all_public_types.get(named)
        if isinstance(declaration, TypeAlias):
            if self.declared_owner(named, declaration).renames:
                return None
            return self._denoted(named, declaration, frozenset({named}))
        module_id, atom = qname
        path = _path(atom)
        if len(path) < 2:
            return None
        alias = self.identity((module_id, _atom(path[:-1])))
        declaration = self._all_public_types.get(alias)
        if (
            not isinstance(declaration, TypeAlias)
            or path[-1] not in self.declared_owner(alias, declaration).members
        ):
            return None
        return ("member", self._denoted(alias, declaration, frozenset({alias})), path[-1])

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

    def constructor_identity(self, constructor: ConstructorRef) -> ConstructorRef:
        """Return the constructor *constructor* names: a renaming alias's is its target's.

        A member an alias renaming an enum selects is the enum's member; one
        an alias applying it selects stays the alias's (:meth:`denotation`).
        """
        named = self.identity(constructor.qname)
        owner = self.owner(named)
        if owner is not None and constructor.member is not None:
            return constructor if owner.alias is not None else owner.members[constructor.member]
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
            self._resolved += 1
        return owner

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
        denoted, type_expr, through = self._alias_selection(qname, declaration)
        if not is_nominal_type_expr(type_expr, declaration.type_params):
            structural = TypeOwner(
                None, declaration.node_id, alias=declaration, arity=arity, stands_for=type_expr
            )
            return _builtin_owner(structural)
        target = None if denoted is None else self.owner(denoted)
        reached = None if through is None else self.owner(through[0])
        if denoted is None or target is None or through is None or reached is None:
            return replace(presumed, stands_for=type_expr)
        reachable, hidden = self._projection(qname, through[1], reached, every_use=True)
        return TypeOwner(
            constructor,
            declaration.node_id,
            reached.names | {declaration.name} if reached.names else frozenset(),
            reachable,
            reached.referenced,
            declaration,
            hidden=hidden,
            target=TypeTarget(denoted, target.decl_node_id),
            own_path_referenced=reached.own_path_referenced,
            arity=arity,
            builtin=_applied_builtin(reached, through[1]),
            stands_for=type_expr,
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
        applied, stands for its argument there; any other generic alias that
        is a type of its own (:meth:`_applied_alias`), for what that stands
        for with its arguments.
        """
        if not is_nominal_type_expr(spelling, alias.type_params):
            return AliasSelection(None, self._normalized(qname, alias, spelling), None)
        selection = self._decided_targets(qname, alias, spelling)
        target = None if selection is None else self.declared_path(selection)
        if target is None:
            return AliasSelection(None, spelling, None)
        if isinstance(spelling, AppliedT):
            parameter = self.projected_parameter(target)
            if parameter is not None and parameter[0] < len(spelling.args):
                return self._spelled_selection(qname, alias, spelling.args[parameter[0]])
            normalized = replace(
                spelling, args=tuple(self._normalized(qname, alias, arg) for arg in spelling.args)
            )
            applied = self._applied_alias(target, alias, normalized)
            if applied is not None:
                return AliasSelection(*applied, (target, spelling))
            return AliasSelection(target, normalized, (target, spelling))
        return AliasSelection(target, spelling, (target, spelling))

    def _normalized(self, qname: QName, alias: TypeAlias, spelling: TypeExpr) -> TypeExpr:
        """*spelling*, written in alias *alias* at *qname*, with each type name what it stands for.

        ``Box[Id[T]]`` with ``type Id[T] = T`` is ``Box[T]`` (:meth:`_spelled_selection`).
        """
        if is_nominal_type_expr(spelling, alias.type_params):
            return self._spelled_selection(qname, alias, spelling).stands_for
        if isinstance(spelling, ArrayT):
            return replace(spelling, elem=self._normalized(qname, alias, spelling.elem))
        if isinstance(spelling, DictT):
            return replace(
                spelling,
                key=self._normalized(qname, alias, spelling.key),
                value=self._normalized(qname, alias, spelling.value),
            )
        if isinstance(spelling, FuncT):
            return replace(
                spelling,
                params=tuple(self._normalized(qname, alias, param) for param in spelling.params),
                result=self._normalized(qname, alias, spelling.result),
            )
        return spelling

    def _applied_alias(
        self, qname: QName, alias: TypeAlias, spelling: AppliedT
    ) -> tuple[QName | None, TypeExpr] | None:
        """What *spelling*, applying generic alias *qname* in alias *alias*, denotes and stands for.

        What the applied alias does, with *spelling*'s arguments for its
        parameters: ``P[int]`` with ``type P[T] = Plain`` is ``Plain``. A
        name it stands for spelled like a parameter of *alias* is spelled
        anchored (``::X``), so it reads as no parameter. ``None`` unless
        *qname* is a generic alias that is a type of its own; also when
        *alias* passes it its own parameters, and so renames it.
        """
        applied = self.owner(self.identity(qname))
        generic = None if applied is None else applied.alias
        stands_for = None if applied is None else applied.stands_for
        if (
            applied is None
            or generic is None
            or stands_for is None
            or not generic.type_params
            or applied.renames
            or passes_parameters(alias.type_params, spelling)
        ):
            return None
        span, node_id = spelling.span, spelling.node_id
        bound: dict[str, TypeExpr] = {
            name: NameT(
                name,
                span,
                node_id,
                QualifierChain(QualifierAnchor.CURRENT_MODULE, (), name, span, node_id),
            )
            for name in set(member_type_params((stands_for,), alias.type_params)).difference(
                generic.type_params
            )
        }
        bound |= zip(generic.type_params, spelling.args, strict=False)
        target = applied.target
        return None if target is None else target.qname, substitute_type_names(stands_for, bound)

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
            qname = declaration_qname(selection.key)
            return qname if self.is_declared(qname) else None
        member = self.owner_member(selection.owner, selection.member)
        return None if member is None else member.qname

    def owner_member(self, owner: DeclarationKey, name: str) -> ConstructorRef | None:
        """Return the member type *owner* selects as *name*: an alias's is its target's member."""
        table = self.owner(declaration_qname(owner))
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
            current = self._alias_selection(current, declaration).target
        return ()


def _builtin_owner(owner: TypeOwner) -> TypeOwner:
    """Return *owner* of an alias, with the built-in type it stands for, if any."""
    builtin = owner.stands_for
    if builtin is None or named_builtin_type(builtin) is None:
        return owner
    return replace(owner, builtin=builtin)


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
