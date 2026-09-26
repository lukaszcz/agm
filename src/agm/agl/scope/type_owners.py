"""Constructor owners of type paths, resolved by declaration identity.

A type path qualifies constructors (``Owner::Name``). What it selects is read
from the declaration it names; an alias is followed to its target, which
selects exactly what the same type name would in an annotation at the
alias's own declaration (:mod:`agm.agl.scope.type_names`), never where the
alias is used. A structural target -- not a type name, or the bare name of
one of the alias's own type parameters -- selects nothing; a target selecting
no declaration is presumed constructible, leaving the verdict to typecheck.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.imports import (
    ImportEnv,
    NameAtom,
    QName,
    try_resolve_qualified_member,
)
from agm.agl.scope.symbols import (
    ConstructorRef,
    ScopePath,
    TypeOwner,
    dedupe_constructor_candidates,
)
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.scope.symbols import to_bare_path as _path
from agm.agl.scope.type_names import TypeNameSite, nominal_selection
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    QualifierAnchor,
    RecordDef,
    TypeAlias,
    VariantDef,
    VariantRef,
)

__all__ = ["ModuleTypeContributions", "TypeOwnerIndex"]

ModuleTypeContributions = Callable[
    [ModuleId, ScopePath, NameAtom, Callable[[QName], bool]],
    tuple[ScopePath, frozenset[QName]] | None,
]
"""The nearest layer above one module scope contributing a spelling, restricted to types."""


class TypeOwnerIndex:
    """Memoized :class:`TypeOwner` of every program type path.

    Record, exception, and inline enum member paths own their own
    constructors; an enum path owns only the inline members its scope
    declares, a referenced member staying at its own path; an alias path owns
    its own constructor and selects through the owner of its target.
    *contributions* answers what a module's lexical layers contribute, so
    alias targets see ``use`` declarations. *retained* supplies the owners of
    *retained_module*'s paths that earlier REPL entries declared, already
    resolved against the declarations they saw.
    """

    def __init__(
        self,
        *,
        all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
        constructor_refs: Mapping[QName, ConstructorRef],
        import_envs: Mapping[ModuleId, ImportEnv],
        contributions: ModuleTypeContributions,
        retained_module: ModuleId | None = None,
        retained: Mapping[ScopePath, TypeOwner] | None = None,
    ) -> None:
        self._all_public_types = all_public_types
        self._constructor_refs = constructor_refs
        self._import_envs = import_envs
        self._contributions = contributions
        self._retained_module = retained_module
        self._retained = retained or {}
        self._owners: dict[QName, TypeOwner] = {}
        self._alias_targets: dict[QName, frozenset[QName] | None] = {}
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
            retained_module=module_id,
            retained=retained,
        )

    def is_declared(self, qname: QName) -> bool:
        """Whether *qname* names a type or an inline enum member."""
        return (
            qname in self._all_public_types
            or qname in self._constructor_refs
            or (qname[0] == self._retained_module and _path(qname[1]) in self._retained)
        )

    def site(
        self, module_id: ModuleId, scope_path: ScopePath, type_params: Iterable[str] = ()
    ) -> TypeNameSite:
        """Return the site of a type name written in *module_id* at *scope_path*."""
        return TypeNameSite(
            module_id=module_id,
            scope_path=scope_path,
            import_env=self._import_envs[module_id],
            declares=lambda path: self.is_declared((module_id, _atom(path))),
            contributions=lambda name: self._contributions(
                module_id, scope_path, name, self.is_declared
            ),
            is_type=self.is_declared,
            type_params=frozenset(type_params),
        )

    def referenced_member_refs(
        self, module_id: ModuleId, member: VariantRef
    ) -> tuple[ConstructorRef, ...]:
        """Return every record constructor enum member reference *member* transparently denotes."""
        key = (module_id, member.node_id)
        if key not in self._referenced_members:
            self._referenced_members[key] = self._resolve_referenced_member(module_id, member)
        return self._referenced_members[key]

    def owner(self, qname: QName) -> TypeOwner | None:
        """Return what type path *qname* selects, or ``None`` when it names no type."""
        declaration = self._all_public_types.get(qname)
        if declaration is not None:
            return self.declared_owner(qname, declaration)
        path = _path(qname[1])
        if qname[0] == self._retained_module and path in self._retained:
            return self._retained[path]
        member = self._constructor_refs.get(qname)
        return None if member is None else TypeOwner(member, frozenset({member.owner_name}))

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
        return owner.constructor if owner.names else None

    def _resolve(
        self, qname: QName, declaration: RecordDef | EnumDef | ExceptionDef | TypeAlias
    ) -> TypeOwner:
        module_id, atom = qname
        path = _path(atom)
        if isinstance(declaration, (RecordDef, ExceptionDef)):
            return TypeOwner(self._constructor_refs[qname], frozenset({declaration.name}))
        if isinstance(declaration, EnumDef):
            return TypeOwner(
                None,
                members=self._enum_members(module_id, path, declaration),
                referenced=self._referenced_names(module_id, declaration),
            )
        constructor = ConstructorRef.for_alias(declaration, module_id, path[:-1])
        # Typecheck judges a target scope selects no declaration for, so the
        # alias is presumed constructible; an alias cycle meets it that way.
        presumed = TypeOwner(constructor, frozenset({declaration.name}), alias=declaration)
        self._owners[qname] = presumed
        targets = self._alias_selection(qname, declaration)
        if targets is None:
            return TypeOwner(None, alias=declaration)
        target = self.owner(next(iter(targets))) if len(targets) == 1 else None
        if target is None:
            return presumed
        return TypeOwner(
            constructor,
            target.names | {declaration.name} if target.names else frozenset(),
            target.members,
            target.referenced,
            declaration,
        )

    def _alias_selection(self, qname: QName, alias: TypeAlias) -> frozenset[QName] | None:
        """Return what alias *qname*'s target selects where declared; ``None`` if structural."""
        if qname not in self._alias_targets:
            site = self.site(qname[0], _path(qname[1])[:-1], alias.type_params)
            self._alias_targets[qname] = nominal_selection(site, alias.type_expr)
        return self._alias_targets[qname]

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

    def _resolve_referenced_member(
        self, module_id: ModuleId, member: VariantRef
    ) -> tuple[ConstructorRef, ...]:
        chain = member.chain
        local = (module_id, _atom((*chain.route_segments, chain.member)))
        qnames: tuple[QName, ...] = ()
        if local in self._all_public_types or local in self._constructor_refs:
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
        """Follow aliases from *qname* to the record constructor at the end of the chain."""
        seen: set[QName] = set()
        current: QName | None = qname
        while current is not None and current not in seen:
            seen.add(current)
            constructor = self._constructor_refs.get(current)
            if constructor is not None:
                return (constructor,)
            declaration = self._all_public_types.get(current)
            if not isinstance(declaration, TypeAlias):
                return ()
            targets = self._alias_selection(current, declaration)
            current = next(iter(targets)) if targets is not None and len(targets) == 1 else None
        return ()
