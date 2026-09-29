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

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    NameAtom,
    QName,
    try_resolve_qualified_member,
)
from agm.agl.scope.symbols import (
    ConstructorRef,
    DeclarationKey,
    ScopePath,
    TypeOwner,
    TypeTarget,
    dedupe_constructor_candidates,
)
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.scope.symbols import to_bare_path as _path
from agm.agl.scope.type_names import (
    imported_member_selection,
    is_nominal_type_expr,
    spells_own_declaration,
)
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    Item,
    QualifierAnchor,
    RecordDef,
    TypeAlias,
    VariantDef,
    VariantRef,
    static_type_items,
)
from agm.agl.syntax.types import AppliedT, NameT

__all__ = [
    "ModuleTypeContributions",
    "TypeOwnerIndex",
    "beneath",
    "declared_member_scopes",
    "owned_constructors",
    "retired_member_scopes",
    "root_type_names",
]

ModuleTypeContributions = Callable[
    [ModuleId, ScopePath, NameAtom, Callable[[QName], bool]],
    tuple[ScopePath, frozenset[QName]] | None,
]
"""The nearest layer above one module scope contributing a spelling, restricted to types."""

AliasTargets = Callable[[QName, TypeAlias, NameT | AppliedT], DeclarationKey | None]
"""The declaration scope selects for alias *qname*'s nominal target *spelling* where declared.

``None`` when scope selects none: a built-in type name.
"""

CurrentTypeSelection = Callable[[ModuleId, ScopePath, NameT | AppliedT], DeclarationKey | None]
"""The declaration a type name spelled in one module scope selects now, if any."""

AliasSelection = tuple[QName | None, bool, NameT | AppliedT] | None
"""An alias's selected target declaration, whether reached indirectly, and its spelling.

``None`` for a structural target; a ``None`` declaration for a target scope
selects none. A target reached through an import surface (``use``, import
tail, or module route) is indirect: its ``hiding`` filters the members the
alias reaches.
"""


class TypeOwnerIndex:
    """Memoized :class:`TypeOwner` of every program type path.

    Record, exception, and inline enum member paths own their own
    constructors; an enum path owns only the inline members its scope
    declares, a referenced member staying at its own path; an alias path owns
    its own constructor and selects through the owner of its target.
    *contributions* answers what a module's lexical layers contribute, so a
    target spelling's member paths see ``use`` declarations. *alias_targets*
    answers scope's decision for an alias's target; *current_selection* what
    a retained alias's spelling selects now. *retained* supplies the owners
    of *retained_module*'s paths that earlier REPL entries declared, already
    resolved against the declarations they saw.
    """

    def __init__(
        self,
        *,
        all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
        constructor_refs: Mapping[QName, ConstructorRef],
        import_envs: Mapping[ModuleId, ImportEnv],
        contributions: ModuleTypeContributions,
        alias_targets: AliasTargets,
        current_selection: CurrentTypeSelection,
        retained_module: ModuleId | None = None,
        retained: Mapping[ScopePath, TypeOwner] | None = None,
    ) -> None:
        self._all_public_types = all_public_types
        self._constructor_refs = constructor_refs
        self._import_envs = import_envs
        self._contributions = contributions
        self._decided_targets = alias_targets
        self._current_selection = current_selection
        self._retained_module = retained_module
        self._retained = retained or {}
        self._owners: dict[QName, TypeOwner] = {}
        self._alias_targets: dict[QName, AliasSelection] = {}
        self._referenced_members: dict[tuple[ModuleId, int], tuple[ConstructorRef, ...]] = {}

    def with_retained(
        self, module_id: ModuleId, retained: Mapping[ScopePath, TypeOwner]
    ) -> TypeOwnerIndex:
        """Return an index over the same program that also sees *module_id*'s retained paths."""
        return TypeOwnerIndex(
            all_public_types=self._all_public_types,
            constructor_refs=self._constructor_refs,
            import_envs=self._import_envs,
            contributions=self._contributions,
            alias_targets=self._decided_targets,
            current_selection=self._current_selection,
            retained_module=module_id,
            retained=retained,
        )

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
        that enum's retained owner.
        """
        constructor = self._constructor_refs.get(qname)
        if constructor is not None or qname[0] != self._retained_module:
            return constructor
        path = _path(qname[1])
        enum = self._retained.get(path[:-1])
        return None if enum is None else enum.members.get(path[-1])

    def referenced_member_refs(
        self, module_id: ModuleId, member: VariantRef
    ) -> tuple[ConstructorRef, ...]:
        """Return every record constructor enum member reference *member* transparently denotes."""
        key = (module_id, member.node_id)
        if key not in self._referenced_members:
            self._referenced_members[key] = self._resolve_referenced_member(module_id, member)
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
        (record, enum, structural, or a same-entry cycle), stands unchanged.
        Otherwise its target is looked up fresh: if it is gone (a later entry
        retired or redeclared its path), *retained* stands at its
        declaration-time members/hidden. If the target is itself an alias,
        its current (recursively re-derived) members/hidden are inherited
        unfiltered, exactly as a fresh chain link does. Otherwise, a direct
        hit on the (non-alias) target stands as-is -- the same declaration
        has the same members -- while an indirect hit is re-checked: if its
        own spelling still selects the target at this entry's site
        (*current_selection*), the target's current members are re-projected
        through that spelling; if the spelling no longer selects exactly the
        target (shadowed, ambiguous, or the route is gone), *retained*
        freezes at its declaration-time members/hidden.
        """
        alias, target = retained.alias, retained.target
        if alias is None or target is None:
            return retained
        current = self.owner(target.qname)
        if current is None or current.decl_node_id != target.decl_node_id:
            return retained
        if current.alias is not None:
            return replace(retained, members=current.members, hidden=current.hidden)
        spelling = alias.type_expr
        if not retained.indirect or not is_nominal_type_expr(spelling, alias.type_params):
            return retained
        module_id, atom = qname
        path = _path(atom)
        key = self._current_selection(module_id, path[:-1], spelling)
        if key is None or self._declared_path(key) != target.qname:
            return retained
        reachable, hidden = self._filtered_projection(module_id, path, spelling, current)
        return replace(retained, members=reachable, hidden=hidden)

    def _filtered_projection(
        self,
        module_id: ModuleId,
        path: ScopePath,
        type_expr: NameT | AppliedT,
        target_owner: TypeOwner,
    ) -> tuple[Mapping[str, ConstructorRef], frozenset[str]]:
        """Return an alias's reachable ``members``/``hidden``, filtered from *target_owner*.

        Narrows *target_owner*'s members through *type_expr* -- the alias's
        own nominal target spelling -- via :func:`imported_member_selection`
        evaluated at *path*'s site. Shared by a fresh declaration
        (:meth:`_resolve`) and a retained alias (:meth:`_current_retained_owner`),
        both of which call this only when the target is not itself an alias --
        whose own projection, already filtered at its own site, applies
        unfiltered instead.
        """
        import_env = self._import_envs[module_id]

        def contributions(name: NameAtom) -> tuple[ScopePath, frozenset[QName]] | None:
            return self._contributions(module_id, path[:-1], name, self.is_declared)

        reachable = {
            name: member
            for name, member in target_owner.members.items()
            if (member.owner_module_id, _atom((*member.owner_path, member.owner_name)))
            in imported_member_selection(import_env, contributions, type_expr, name)
        }
        hidden = target_owner.hidden | (target_owner.members.keys() - reachable.keys())
        return reachable, hidden

    def declared_owner(
        self, qname: QName, declaration: RecordDef | EnumDef | ExceptionDef | TypeAlias
    ) -> TypeOwner:
        """Return what *declaration*, declared at type path *qname*, selects."""
        owner = self._owners.get(qname)
        if owner is None:
            owner = self._resolve(qname, declaration)
            self._owners[qname] = owner
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
        if isinstance(declaration, (RecordDef, ExceptionDef)):
            return TypeOwner(
                self._constructor_refs[qname], declaration.node_id, frozenset({declaration.name})
            )
        if isinstance(declaration, EnumDef):
            return TypeOwner(
                None,
                declaration.node_id,
                members=self._enum_members(module_id, path, declaration),
                referenced=self._referenced_names(module_id, declaration),
                injected=dedupe_constructor_candidates(
                    constructor
                    for member in declaration.members
                    if isinstance(member, VariantRef)
                    for constructor in self.referenced_member_refs(module_id, member)
                ),
                own_path_referenced=self._own_path_referenced_names(module_id, path, declaration),
            )
        constructor = ConstructorRef.for_alias(declaration, module_id, path[:-1])
        # None of an alias's TypeOwner constructions below pass
        # own_path_referenced, so it stays empty: an alias never selects a
        # referenced member at its own path, only at its target's.
        # Typecheck judges a target scope selects no declaration for, so the
        # alias is presumed constructible; an alias cycle meets it that way.
        presumed = TypeOwner(
            constructor, declaration.node_id, frozenset({declaration.name}), alias=declaration
        )
        self._owners[qname] = presumed
        selection = self._alias_selection(qname, declaration)
        if selection is None:
            return TypeOwner(None, declaration.node_id, alias=declaration)
        target_qname, indirect, type_expr = selection
        target = None if target_qname is None else self.owner(target_qname)
        if target_qname is None or target is None:
            return presumed
        # A fresh declaration's target spelling trivially "still selects" the
        # target it was just resolved against, so filtering is gated only by
        # indirection and the target not itself being an alias (whose own
        # projection, filtered at its own site, already applies unfiltered) --
        # the same rule ``_current_retained_owner`` applies when the target
        # it re-checks turns out to be an alias itself.
        if indirect and target.alias is None:
            reachable, hidden = self._filtered_projection(module_id, path, type_expr, target)
        else:
            reachable, hidden = target.members, target.hidden
        return TypeOwner(
            constructor,
            declaration.node_id,
            target.names | {declaration.name} if target.names else frozenset(),
            reachable,
            target.referenced,
            declaration,
            hidden=hidden,
            indirect=indirect,
            target=TypeTarget(target_qname, target.decl_node_id),
        )

    def _alias_selection(self, qname: QName, alias: TypeAlias) -> AliasSelection:
        """Return the declaration alias *qname*'s target denotes, as scope selected it.

        Scope's selection (*alias_targets*) may spell a member through its
        owner's alias (``O::Member``); the target is the member declaration
        it denotes. See :data:`AliasSelection`.
        """
        if qname not in self._alias_targets:
            spelling = alias.type_expr
            if not is_nominal_type_expr(spelling, alias.type_params):
                self._alias_targets[qname] = None
            else:
                key = self._decided_targets(qname, alias, spelling)
                self._alias_targets[qname] = (
                    None if key is None else self._declared_path(key),
                    key is not None
                    and not spells_own_declaration(qname[0], _path(qname[1])[:-1], spelling, key),
                    spelling,
                )
        return self._alias_targets[qname]

    def _declared_path(self, key: DeclarationKey) -> QName | None:
        """Return the declaration *key* denotes: its own path, or a member through its owner."""
        module_id, path, name = key
        qname = (module_id, _atom((*path, name)))
        if self.is_declared(qname):
            return qname
        owner = self.owner((module_id, _atom(path))) if path else None
        member = None if owner is None else owner.members.get(name)
        return (
            None
            if member is None
            else (member.owner_module_id, _atom((*member.owner_path, member.owner_name)))
        )

    def _enum_members(
        self, module_id: ModuleId, path: ScopePath, declaration: EnumDef
    ) -> dict[str, ConstructorRef]:
        """Return the members an enum's scope declares -- its inline members -- by name."""
        return {
            member.name: self._constructor_refs[(module_id, _atom((*path, member.name)))]
            for member in declaration.members
            if isinstance(member, VariantDef)
        }

    def _referenced_names(self, module_id: ModuleId, declaration: EnumDef) -> frozenset[str]:
        """Return the names spelling an enum's resolved referenced members and their records."""
        return frozenset(
            name
            for member in declaration.members
            if isinstance(member, VariantRef)
            and (refs := self.referenced_member_refs(module_id, member))
            for name in (member.chain.member, *(ref.owner_name for ref in refs))
        )

    def _own_path_referenced_names(
        self, module_id: ModuleId, path: ScopePath, declaration: EnumDef
    ) -> frozenset[str]:
        """Return each own-path referenced member's terminal name.

        ``enum Owner = ... | Owner::Name`` nests ``Name``'s separate
        declaration directly beneath *path*, unlike a member referenced from
        elsewhere: such a name selects through this owner like a declared one
        (see :attr:`TypeOwner.own_path_referenced`).
        """
        return frozenset(
            ref.owner_name
            for member in declaration.members
            if isinstance(member, VariantRef)
            for ref in self.referenced_member_refs(module_id, member)
            if ref.owner_module_id == module_id and ref.owner_path == path
        )

    def _resolve_referenced_member(
        self, module_id: ModuleId, member: VariantRef
    ) -> tuple[ConstructorRef, ...]:
        chain = member.chain
        local = (module_id, _atom((*chain.route_segments, chain.member)))
        qnames: tuple[QName, ...] = ()
        if self.is_declared(local):
            qnames = (local,)
        elif chain.segments and chain.anchor is not QualifierAnchor.CURRENT_MODULE:
            qname = try_resolve_qualified_member(
                self._import_envs[module_id],
                tuple(chain.segments[0].name.split("/")),
                _atom((*(segment.name for segment in chain.segments[1:]), chain.member)),
                anchored=chain.anchored,
            )
            if qname is not None:
                qnames = (qname,)
        return dedupe_constructor_candidates(
            constructor for qname in qnames for constructor in self._constructors_through(qname)
        )

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
            selection = self._alias_selection(current, declaration)
            current = None if selection is None else selection[0]
        return ()


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
                    for injected in owner.injected:
                        if _is_current(module_id, owners, injected):
                            yield injected.owner_name, injected, (), True
        elif owner.constructible:
            yield path[-1], owner.constructor, path[:-1], bare


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
