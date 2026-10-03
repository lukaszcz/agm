"""What one module's path lookups read: its own declarations, imports and uses, by full path.

:class:`ModuleSources` implements :class:`~agm.agl.scope.lookup.PathSources`
for the module ``_Resolver`` resolves, and records the full path each own
scope path declares. It reads the declaration and constructor tables the
resolver collects and the lexical layers its walk binds: :class:`SourcesHost`
declares what it reads of them.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Protocol

from agm.agl.diagnostics import (
    AglError,
    HiddenMemberError,
    ReferencedMemberError,
)
from agm.agl.modules.ids import (
    ModuleId,
    Reader,
    spell_declaration,
)
from agm.agl.scope.imports import (
    BareRoute,
    Exposure,
    ImportEnv,
    ItemDeclaration,
    NameAtom,
    QName,
    ScopeOrigins,
    declares_bare_constructor,
    qualifier_candidates,
    qualifier_decls,
    qualifier_exposures,
    qualifier_hides,
    qualifier_member_decls,
    qualifier_members,
    qualifier_scope_paths,
    unqualified_exposures,
)
from agm.agl.scope.lookup import (
    NOT_HIDDEN,
    AliasTarget,
    Application,
    Candidate,
    Hiding,
    LookupKind,
    QualifiedTarget,
    Reading,
    lookup_declared,
    removes,
)
from agm.agl.scope.symbols import (
    AmbiguousConstructorError,
    BinderKind,
    BindingRef,
    ConstructorRef,
    ContributionLayer,
    DeclarationKey,
    DeclInfo,
    ImportedModuleOrigin,
    Layers,
    MissRepair,
    ScopeNode,
    ScopePath,
    TypeOwner,
    TypeSelection,
    UnknownMemberError,
    anchored_layers,
    atom_under_prefix,
    contribution_origin,
    contribution_origins,
    layered,
    relative_under,
)
from agm.agl.scope.symbols import binding_qname as _ref_qname
from agm.agl.scope.symbols import declaration_qname as _key_qname
from agm.agl.scope.symbols import import_item_path as _item_path
from agm.agl.scope.symbols import qname_declaration as _qname_decl_key
from agm.agl.scope.symbols import to_bare_atom as _bare_atom
from agm.agl.scope.symbols import to_bare_path as _bare_path
from agm.agl.scope.type_names import (
    MemberReferenced,
    is_nominal_type_expr,
    owner_member_selection,
)
from agm.agl.scope.type_owners import AliasReach, TypeOwnerIndex
from agm.agl.scope.uses import UseReader
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    Program,
    QualifierAnchor,
    QualifierChain,
    QualifierSegment,
    RecordDef,
    TypeAlias,
    UseDecl,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import (
    AppliedT,
    NameT,
    TypeExpr,
    render_qualified_name,
    render_qualifier_path,
)

type _Declaring = tuple[QName | NameAtom, str | None]
"""A full path (or the path alone of ones in other modules) declared, and a name beneath it."""

type _Route = tuple[tuple[str, ...], bool]
"""A module route's spelling, and whether it is anchored."""

type _StandingFor = tuple[QName, ScopePath, ContributionLayer, LookupKind, _Route | None]
"""A path beneath an alias, read as a declaration of a kind made visible by a layer.

Through a module route when only that route reached the alias.
"""


def constructor_binding(name: str, constructor: ConstructorRef) -> BindingRef:
    """The value binding bare *name* reads *constructor* through where no declaration spells it."""
    return BindingRef(
        name=name,
        mutable=False,
        decl_span=SourceSpan(
            start_line=0, start_col=0, end_line=0, end_col=0, start_offset=0, end_offset=0
        ),
        decl_node_id=constructor.owner_decl_node_id,
        kind=BinderKind.constructor_binding,
        module_id=constructor.owner_module_id,
    )


def constructor_candidate_sort_key(
    candidate: ConstructorRef,
) -> tuple[tuple[str, ...], ScopePath, str, int]:
    """Order constructor candidates by their declaration identity, never by set order."""
    return (
        candidate.owner_module_id.segments,
        candidate.owner_path,
        candidate.owner_name,
        candidate.owner_decl_node_id,
    )


def is_root_inline_member(constructor: ConstructorRef) -> bool:
    """Whether *constructor* is an inline member of an enum declared at its module root."""
    return (
        constructor.inline_enum_owner_decl_node_id is not None and len(constructor.owner_path) == 1
    )


def _route_root(source: ScopePath, relative: ScopePath) -> ScopePath:
    """Return *source* with a target-relative suffix trimmed back off its end."""
    return source[: len(source) - len(relative)] if relative else source


def scope_path_sort_key(path: ScopePath) -> tuple[int, ScopePath]:
    """Order scope paths by depth, then lexical spelling."""
    return (len(path), path)


def _type_qnames(refs: Iterable[BindingRef]) -> Iterator[QName]:
    """The full paths of *refs* that may name a type; an injected enum member never does."""
    return (_ref_qname(ref) for ref in refs if ref.contributes_a_type)


class SourcesHost(Protocol):
    """What reading a module's paths needs of the resolver collecting and walking it.

    The module's program, import environment and the whole program's tables;
    the declarations, scope layers and constructor candidates it collects;
    the root layer its walk binds; and its uses.
    """

    _module_id: ModuleId
    _program: Program
    _import_env: ImportEnv
    _all_public_types: dict[
        tuple[ModuleId, NameAtom], RecordDef | EnumDef | ExceptionDef | TypeAlias
    ]
    _type_owners: TypeOwnerIndex
    _decl_info: dict[tuple[ModuleId, NameAtom], DeclInfo]
    _cross_module_constructor_refs: Mapping[tuple[ModuleId, NameAtom], ConstructorRef]
    _repl_session_type_paths: dict[ScopePath, TypeOwner]
    _repl_session_root_type_names: frozenset[str]
    _root_scope: ScopeNode
    _scope_nodes: dict[ScopePath, ScopeNode]
    _declarations: dict[DeclarationKey, BindingRef]
    _scope_entity_kinds: dict[DeclarationKey, str]
    _import_decl_scope_paths: dict[int, ScopePath]
    _type_declarations: list[tuple[RecordDef | EnumDef | ExceptionDef | TypeAlias, ScopePath]]
    _scoped_constructor_candidates: dict[tuple[ScopePath, str], list[ConstructorRef]]
    _constructor_candidates: dict[str, list[ConstructorRef]]
    _injected_constructors: dict[tuple[ScopePath, str], list[ConstructorRef]]
    _uses: UseReader

    def _route_injected_members(
        self, chain: QualifierChain, name: str
    ) -> dict[ConstructorRef, str]:
        """Map each root enum inline member one-segment route *chain* injects as *name*."""
        ...

    def type_name_selection_at(
        self, scope_path: ScopePath, spelling: NameT | AppliedT, *, every_use: bool
    ) -> TypeSelection | None:
        """Return what type name *spelling*, at *scope_path*, selects now."""
        ...


class ModuleSources(SourcesHost):
    """The one lookup's reads for one module, by full path (:class:`PathSources`)."""

    def __init__(self, placements: Mapping[DeclarationKey, QName]) -> None:
        # Own declarations placed beneath another module's path, which their
        # own path's spelling may not reach any more: decided where they were
        # declared (an earlier REPL entry, or this module before its
        # declarations were keyed by their declared paths).
        self._placements = placements
        self._placed: dict[QName, list[ScopePath]] = {}
        for (_module_id, path, name), placement in placements.items():
            self._placed.setdefault(placement, []).append((*path, name))
        # This module as diagnostics spell declarations for it (:meth:`reader`).
        self._reader: Reader | None = None
        # Every enum this module reads by the names of its members, built on
        # first use.
        self._enum_member_index: dict[str, dict[QName, ConstructorRef]] | None = None
        # Import declaration id -> the declarations its ``hiding`` removes, by identity.
        self._hidden_by: dict[int, frozenset[DeclarationKey]] = {}
        # Whether what an alias reaches is being read (:meth:`_reaching`).
        self._reads_reach = False
        # The full path each own scope path declares (an alias segment stands
        # for its target's path); recorded as lookups first read them, once
        # every module's headers are prepared, since reading an alias needs
        # them (``_declare_scope_paths``).
        self._declared_paths: dict[ScopePath, QName] = {}
        # The own scope paths to read, shorter first (a heap of sort keys);
        # those waiting on the type each selects, and how many resolved types
        # they were released for; the one being read, if any; and how many
        # readings recorded another path than was.
        self._undeclared_scope_paths: list[tuple[int, ScopePath]] = []
        self._waiting_scope_paths: dict[QName, list[ScopePath]] = {}
        self._resolved_seen = 0
        self._recording_scope_path: ScopePath | None = None
        self._scope_paths_settled = False
        self._recorded_changes = 0
        # The own scope paths, shorter first, by what each declares; those
        # declared otherwise than spelled also by each name spelled beneath
        # them (``_declaring_spellings``); and those names, collected on first use.
        self._declaring_named: dict[_Declaring, list[ScopePath]] = {}
        self._names_beneath: dict[ScopePath, set[str]] | None = None
        # The scope paths whose reading read each entry of the two recorded
        # tables: each is read again when the entry changes.
        self._declared_readers: dict[ScopePath, set[ScopePath]] = {}
        self._declaring_readers: dict[_Declaring, set[ScopePath]] = {}
        # Once the tables the resolver collects are complete (:meth:`_keep_readings`),
        # the contributions at each full path, the own types, and what each
        # path beneath an alias stands for, by what they are read with.
        self._keeping_readings = False
        self._kept_contributions: dict[tuple[ScopePath, ScopePath, LookupKind], Reading] = {}
        self._kept_own_types: dict[ScopePath, Reading] = {}
        self._kept_stood_for: dict[_StandingFor, tuple[AliasReach, Reading]] = {}

    def _own_scope_paths(self) -> list[tuple[int, ScopePath]]:
        """Return every own scope path to read, shorter first (a heap of sort keys)."""
        return sorted(scope_path_sort_key(path) for path in self._scope_nodes if path)

    def _declare_scope_paths(self) -> None:
        """Record the full path each own scope path declares.

        A scope path whose whole spelling selects a type declares that type's
        path: through an alias, its target's (``def Geo::m`` with ``type Geo
        = Base`` declares ``Base::m``). Any other declares its parent's path
        and its own name -- through an alias of a built-in type, the
        built-in's name (``def T::f`` with ``type T = text`` declares
        ``text::f``). Own scope paths declaring one path are one path: a
        spelling of any reaches what the others declare (:meth:`own_at`).

        The first lookup reading them reads the paths shorter first, each
        reading what is recorded so far, as does any lookup made meanwhile. A
        path beneath one not recorded yet is read once that one is; one
        selecting a type not :meth:`~TypeOwnerIndex.settled` yet waits, for a
        later lookup to read it once the type is resolved. A path is read
        again whenever an entry it read of the recorded tables changes, and
        recorded anew when it declares another path then.

        Own scope paths never change after collection, so once none waits,
        the recorded tables are final: later calls return at once rather than
        re-reading every own type again.
        """
        if self._recording_scope_path is not None or self._scope_paths_settled:
            return
        owners = self._type_owners
        settled = not owners.resolving
        if settled:
            # A lookup reading a path would resolve an own alias reading the
            # paths half recorded; while a type resolves, the path waits on it.
            self._declared_type_owners()
        resolved = owners.resolved_since(self._resolved_seen)
        self._resolved_seen += len(resolved)
        for qname in tuple(self._waiting_scope_paths) if settled else resolved:
            for path in self._waiting_scope_paths.pop(qname, ()):
                heapq.heappush(self._undeclared_scope_paths, scope_path_sort_key(path))
        self._record_scope_paths()
        self._scope_paths_settled = not self._waiting_scope_paths

    def _declared_type_owners(self) -> dict[ScopePath, TypeOwner]:
        """Return the owner each type this module declares resolves to."""
        return {
            (*path, declaration.name): self._type_owners.declared_owner(
                (self._module_id, _bare_atom((*path, declaration.name))), declaration
            )
            for declaration, path in self._type_declarations
        }

    def _record_scope_paths(self) -> None:
        """Read the own scope paths to read, shorter first, recording what each declares."""
        unread = self._undeclared_scope_paths
        while unread:
            _, path = heapq.heappop(unread)
            self._recording_scope_path = path
            parent = path[:-1]
            self._note_read(self._declared_readers, parent)
            if parent and parent not in self._declared_paths:
                continue
            found = lookup_declared(
                self,
                path,
                None,
                LookupKind.TYPE,
                span=self._program.span,
            )
            key = found.key if isinstance(found, QualifiedTarget) else None
            if key is not None and not self._type_owners.settled(_key_qname(key)):
                self._waiting_scope_paths.setdefault(_key_qname(key), []).append(path)
                continue
            owner = None if key is None else self._type_owners.owner(_key_qname(key))
            builtin = None if owner is None else owner.builtin_name
            if key is not None and builtin is None:
                qname = _key_qname(self.identity(key))
            else:
                module_id, atom = self._declared_paths.get(parent, (self._module_id, ()))
                qname = (module_id, _bare_atom((*_bare_path(atom), builtin or path[-1])))
            self._record_scope_path(path, qname)
        self._recording_scope_path = None

    def _record_scope_path(self, path: ScopePath, declared: QName) -> None:
        """Record that own scope path *path* declares *declared*.

        Every scope path that read an entry this changes is read again.
        """
        recorded = self._declared_paths.get(path)
        if recorded == declared:
            return
        self._declared_paths[path] = declared
        self._recorded_changes += 1
        readers = self._declared_readers.pop(path, set())
        for key in self._declaring_keys(path, recorded):
            self._declaring_named[key].remove(path)
            readers |= self._declaring_readers.pop(key, set())
        for key in self._declaring_keys(path, declared):
            bisect.insort(self._declaring_named.setdefault(key, []), path, key=scope_path_sort_key)
            readers |= self._declaring_readers.pop(key, set())
        for reader in readers:
            heapq.heappush(self._undeclared_scope_paths, scope_path_sort_key(reader))

    def _declaring_keys(self, path: ScopePath, declared: QName | None) -> Iterator[_Declaring]:
        """The entries listing own scope path *path* when it declares *declared*.

        None for a path not recorded; under no name alone for one declaring
        its own spelling (:meth:`_declaring_spellings`).
        """
        if declared is None:
            return
        yield declared, None
        if declared == (self._module_id, _bare_atom(path)):
            return
        module_id, atom = declared
        for name in self._names_spelled_beneath(path):
            yield declared, name
            if module_id != self._module_id:
                yield atom, name

    def _names_spelled_beneath(self, path: ScopePath) -> Collection[str]:
        """Every name a declaration or scope path is spelled with beneath own scope path *path*.

        Those this module collects, which claim a scoped ``let`` or ``var``
        before the walk binds it, and those earlier REPL entries retained.
        """
        names = self._names_beneath
        if names is None:
            names = self._names_beneath = {}
            retained = self._repl_session_type_paths
            spelled = (
                *((*scope, name) for _module_id, scope, name in self._scope_entity_kinds),
                *(
                    (*scope, name)
                    for scope, node in self._scope_nodes.items()
                    for name in node.members
                ),
                *retained,
                *((*scope, name) for scope, owner in retained.items() for name in owner.members),
            )
            for spelling in spelled:
                names.setdefault(spelling[:-1], set()).add(spelling[-1])
        return names.get(path, ())

    def _declaring_spellings(
        self, declared: QName | NameAtom, name: str | None
    ) -> tuple[ScopePath, ...]:
        """The own scope paths spelled otherwise declaring *declared*, with *name* spelled beneath.

        *declared* is a full path, or the path alone of ones in other
        modules. Under no *name*, every own scope path declaring full path
        *declared*. Shorter spellings first.
        """
        key = (declared, name)
        self._note_read(self._declaring_readers, key)
        return tuple(self._declaring_named.get(key, ()))

    def _note_read[K](self, readers: dict[K, set[ScopePath]], key: K) -> None:
        """Note in *readers* that the scope path being read, if any, read recorded entry *key*."""
        reader = self._recording_scope_path
        if reader is None:
            return
        noted = readers.get(key)
        if noted is None:
            readers[key] = {reader}
        else:
            noted.add(reader)

    def read_view(self) -> object:
        """What reads made now see of the uses and the recorded scope paths.

        Reads with equal views see the same of both, and are noted for the
        same scope path being read (:meth:`_note_read`).
        """
        return self._uses.reading, self._recorded_changes, self._recording_scope_path

    def declared_scope(self, path: ScopePath) -> QName | None:
        """The full path own scope path *path* declares, when it or a scope above declares another.

        ``scope Geo`` with ``type Geo = Base`` declares ``Base``, in ``Base``'s
        module. ``None`` for a scope path declaring its own spelling.
        """
        self._declare_scope_paths()
        for end in range(len(path), 0, -1):
            self._note_read(self._declared_readers, path[:end])
            declared = self._declared_paths.get(path[:end])
            if declared is not None and declared != (self._module_id, _bare_atom(path[:end])):
                module_id, atom = declared
                return module_id, _bare_atom((*_bare_path(atom), *path[end:]))
        return None

    def declared_path(self, path: ScopePath) -> QName | None:
        """The full path own declaration *path* is declared at, when a scope above declares another.

        ``def Geo::m`` with ``type Geo = Base`` is declared at ``Base::m``, in
        ``Base``'s module. ``None`` for a declaration at its own spelling.
        A declaration's recorded placement wins.
        """
        placed = self._placements.get((self._module_id, path[:-1], path[-1]))
        if placed is not None:
            return placed
        scope = self.declared_scope(path[:-1])
        if scope is None:
            return None
        module_id, atom = scope
        return module_id, _bare_atom((*_bare_path(atom), path[-1]))

    def nearest_scope(self, path: ScopePath) -> ScopePath:
        """See :meth:`~agm.agl.scope.lookup.PathSources.nearest_scope`."""
        while path not in self._scope_nodes:
            path = path[:-1]
        return path

    def declaring_beside(self, path: ScopePath) -> tuple[ScopePath, ...]:
        """See :meth:`~agm.agl.scope.lookup.PathSources.declaring_beside`.

        The read is noted under what *path* declares, or under *path* while
        it is not recorded.
        """
        self._declare_scope_paths()
        declared = self._declared_paths.get(path)
        if declared is None:
            self._note_read(self._declared_readers, path)
            return ()
        return tuple(
            beside for beside in self._declaring_spellings(declared, None) if beside != path
        )

    @staticmethod
    def _owner_member_error(
        owner: TypeOwner, chain: QualifierChain, count: int, member: str
    ) -> AglError | None:
        """Return why *member* is unreachable through *owner*'s own member table.

        *chain*'s first *count* segments spell the owner. A
        :class:`ReferencedMemberError` for a member *owner* only references,
        a :class:`HiddenMemberError` for one its alias's import hides;
        ``None`` when the member is neither.
        """
        selection = owner_member_selection(owner, member)
        if selection is None:
            return None
        spelling = render_qualifier_path(replace(chain, segments=chain.segments[:count]))
        if isinstance(selection, MemberReferenced):
            return ReferencedMemberError(spelling, member, span=chain.span)
        return HiddenMemberError(spelling, member, span=chain.span)

    def _reachable_decl_contributions[T](
        self, table: Mapping[int, Mapping[NameAtom, frozenset[T]]], path: ScopePath
    ) -> Iterator[tuple[int, Mapping[NameAtom, frozenset[T]]]]:
        """Yield each import declaration's entry in *table* reaching *path*.

        A region-scoped ``import`` contributes bare names to its own region
        and everything nested inside it, so its declaration path must be a
        prefix of *path*; a root import (empty path) reaches everywhere. The
        one definition of that reach, shared by every consumer of
        ``ImportEnv``'s per-declaration tables -- bare declarations, their
        route provenance, and scope-identity routes alike.
        """
        for node_id, contributed in table.items():
            decl_scope_path = self._import_decl_scope_paths.get(node_id, ())
            if path[: len(decl_scope_path)] == decl_scope_path:
                yield node_id, contributed

    def _cross_module_binding_ref(self, qname: QName) -> BindingRef:
        """Build a bare/member ``BindingRef`` for one exposed atom's origin ``QName``.

        Shared by every site that turns an origin reached through
        an import into a reference: promotes the ordinary cross-module
        ``BindingRef`` to a constructor binding -- with the constructor's own
        declaration id, kind, and owner path -- whenever *qname* names a
        record, exception, or enum variant. An alias's binding already is its
        constructor binding, whether or not the alias constructs.
        """
        ref = self._make_cross_module_ref(qname)
        constructor = self._cross_module_constructor_refs.get(qname)
        if constructor is None:
            return ref
        return replace(
            ref,
            decl_node_id=constructor.owner_decl_node_id,
            kind=BinderKind.constructor_binding,
            scope_path=constructor.owner_path,
        )

    def _cross_module_constructor(self, qname: QName) -> ConstructorRef | None:
        """Return the constructor an imported *qname* names, if any."""
        constructor = self._cross_module_constructor_refs.get(qname)
        declaration = self._all_public_types.get(qname)
        if constructor is None and isinstance(declaration, TypeAlias):
            return self._type_owners.alias_constructor(declaration, qname)
        return constructor

    def enum_members_named(self, name: str) -> Mapping[QName, ConstructorRef]:
        """The enums, of any module this one reads, with a member named *name*, and that member."""
        if self._enum_member_index is None:
            self._enum_member_index = self._build_enum_member_index()
        return self._enum_member_index.get(name, {})

    def _build_enum_member_index(self) -> dict[str, dict[QName, ConstructorRef]]:
        """Index every enum this module reads, and each alias renaming one, by its members' names.

        An inline member wins its name over an injected one; a current
        declaration supersedes a retained enum at its path. An alias's paths
        are its target's, so it brings the target's members: an alias
        applying an enum, each as its own path beneath the alias selects it.
        """
        owners: dict[QName, TypeOwner] = {
            (self._module_id, _bare_atom(path)): retained
            for path, retained in self._repl_session_type_paths.items()
            if retained.alias is None and retained.constructor is None
        }
        owners.update(
            (qname, self._type_owners.declared_owner(qname, declaration))
            for qname, declaration in self._all_public_types.items()
            if isinstance(declaration, EnumDef)
        )
        for item, path in self._type_declarations:
            if isinstance(item, EnumDef):
                qname = (self._module_id, _bare_atom((*path, item.name)))
                owners[qname] = self._type_owners.declared_owner(qname, item)
        owners.update(
            (qname, owners[target])
            for qname, declaration in self._all_public_types.items()
            if isinstance(declaration, TypeAlias)
            and (target := self._type_owners.identity(qname)) in owners
        )
        index: dict[str, dict[QName, ConstructorRef]] = {}
        for qname, owner in owners.items():
            for member_name, constructor in (
                *owner.members.items(),
                *((injected.owner_name, injected) for injected in owner.injected),
            ):
                index.setdefault(member_name, {}).setdefault(qname, constructor)
        applying = (
            (qname, applied)
            for qname, declaration in self._all_public_types.items()
            if isinstance(declaration, TypeAlias)
            and qname not in owners
            and (applied := self._type_owners.owner(qname)) is not None
        )
        for qname, applied in applying:
            for member_name, constructor in applied.alias_members().items():
                index.setdefault(member_name, {}).setdefault(qname, constructor)
        return index

    def _scope_route_origins(self, route: BareRoute) -> ScopeOrigins:
        """Return the declarations scope route *route* reaches, through any number of re-exports."""
        module, path = route
        return self._import_env.scope_origins_by_route.get(
            route, frozenset({(module, _bare_atom(path))})
        )

    def _is_value_member(self, ref: BindingRef) -> bool:
        """Whether a named-scope member denotes a value rather than only a type."""
        return ref.kind is not BinderKind.constructor_binding or bool(
            self._scoped_constructor_candidates.get((ref.scope_path, ref.name))
        )

    def _ambiguous_constructor(
        self,
        spelling: str,
        candidates: Mapping[ConstructorRef, Layers],
        repair: str,
        span: SourceSpan,
    ) -> AmbiguousConstructorError:
        """Report *spelling* as ambiguous among *candidates*, each from every contributing layer.

        *repair* selects the first candidate.
        """
        return AmbiguousConstructorError.for_constructor_origins(
            spelling,
            (
                origin
                for candidate, layers in candidates.items()
                for origin in contribution_origins(candidate.selected_qname, layers)
            ),
            repair=repair,
            span=span,
            reader=self.reader(),
        )

    def _keep_readings(self) -> None:
        """Keep what each full path's contributions and own types read from now on.

        The resolver calls this once the tables it collects are complete, its
        own types are resolved, and its walk binds nothing a path read sees.
        A type's resolution reads only its declaring module, so no read from
        now on presumes a type being resolved. A read made while a use's own
        read is in progress, which sees only the uses written before it, is
        not kept.
        """
        self._keeping_readings = True

    def _kept[K, V](self, kept: dict[K, V], key: K, read: Callable[[], V]) -> V:
        """Return what *read* reads, kept in *kept* under *key* when it is final."""
        if not (self._keeping_readings and self._uses.reads_every_use):
            return read()
        found = kept.get(key)
        if found is None:
            found = kept[key] = read()
        return found

    def own_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """This module's own declarations of *kind* at full *path*.

        Those spelled *path*, and those beneath every own scope path spelled
        otherwise that declares the path its parent does.
        """
        if kind is LookupKind.TYPE:
            return self._kept(self._kept_own_types, path, lambda: self._own_at(path, kind))
        return self._own_at(path, kind)

    def _own_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """Read :meth:`own_at`."""
        reading = self._own_spelled_at(path, kind)
        parent = path[:-1]
        if not parent:
            return reading
        self._declare_scope_paths()
        self._note_read(self._declared_readers, parent)
        declared = self._declared_paths.get(parent, (self._module_id, _bare_atom(parent)))
        return sum(
            (
                self._own_spelled_at((*spelling, path[-1]), kind)
                for spelling in self._declaring_spellings(declared, path[-1])
                if spelling != parent
            ),
            reading,
        )

    def own_root_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """See :meth:`~agm.agl.scope.lookup.PathSources.own_root_at`.

        Beneath a parent no own scope path spells, the own scope paths
        declaring another module's path spelled so declare *path*.
        """
        reading = self.own_at(path, kind)
        parent = path[:-1]
        if not parent:
            return reading
        self._declare_scope_paths()
        self._note_read(self._declared_readers, parent)
        if parent in self._declared_paths:
            return reading
        return sum(
            (
                self._own_spelled_at((*spelling, path[-1]), kind)
                for spelling in self._declaring_spellings(_bare_atom(parent), path[-1])
            ),
            reading,
        )

    def _own_spelled_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """This module's own declaration of *kind* spelled *path*."""
        if kind is LookupKind.TYPE:
            qname = (self._module_id, _bare_atom(path))
            target = (
                QualifiedTarget(_qname_decl_key(qname), None, None)
                if self._type_owners.is_declared(qname)
                else None
            )
        elif len(path) == 1:
            target = self._own_root_value(path[0])
        else:
            target = self._own_scoped_value(path[:-1], path[-1])
        if target is None or not self.fits(target, kind):
            return Reading()
        layer = ContributionLayer.DECLARED
        origin = contribution_origin((self._module_id, _bare_atom(path)), layer)
        return Reading((Candidate(target, layer, origin),))

    def _own_root_value(self, name: str) -> QualifiedTarget | None:
        """This module's own root value *name*, or the root constructor it declares so."""
        ref = self._own_level_value(self._layer_chain(self._root_scope), name)
        if ref is None:
            return None
        key = (self._module_id, (), name)
        if ref.kind is not BinderKind.constructor_binding:
            return QualifiedTarget(key, ref, None)
        declared = next(
            (
                candidate
                for candidate in self._constructor_candidates.get(name, ())
                if candidate.owner_module_id == self._module_id and not candidate.owner_path
            ),
            None,
        )
        return None if declared is None else QualifiedTarget(key, ref, declared)

    def _own_scoped_value(self, scope_path: ScopePath, name: str) -> QualifiedTarget | None:
        """This module's own member *name* of named scope *scope_path*, as a value."""
        node = self._scope_nodes.get(scope_path)
        ref = None if node is None else node.members.get(name)
        if ref is None or not self._is_value_member(ref):
            return None
        # One declaration owns each scoped spelling, so it has at most one constructor.
        constructors = self._scoped_constructor_candidates.get((scope_path, name), ())
        return QualifiedTarget(
            (self._module_id, scope_path, name), ref, constructors[0] if constructors else None
        )

    def contributed_at(self, step: ScopePath, path: ScopePath, kind: LookupKind) -> Reading:
        """What the contributions anchored at or above *step* reach at full *path*.

        Import tails' snapshots, and what each visible ``use`` exposes there.
        """
        return self._kept(
            self._kept_contributions,
            (step, path, kind),
            lambda: self._contributed_at(step, path, kind),
        )

    def _contributed_at(self, step: ScopePath, path: ScopePath, kind: LookupKind) -> Reading:
        """Read :meth:`contributed_at`."""
        imported = (
            Candidate(
                self._contributed_target(ref, ()),
                layer,
                contribution_origin(_ref_qname(ref), layer),
                hiding,
                routed,
            )
            for ref, (layers, hiding, routed) in self._imported(step, path).items()
            for layer in layered(layers)
        )
        used = (candidate for candidate, _decl in self._use_exposures(step, path, kind))
        return Reading(
            (*(candidate for candidate in imported if self.fits(candidate.target, kind)), *used)
        )

    def injected_at(self, step: ScopePath, name: str) -> Reading:
        """The enum members injected as bare *name* at *step*.

        This module's own enums inject their members at their own step, a
        member another module declares being contributed; so does each
        enum an import reaches (:meth:`_imports_inject`).
        """
        injected = self._injected_constructors.get((step, name), [])
        other = [c for c in injected if c.owner_module_id != self._module_id]
        other.extend(c for c in self._imports_inject(step, name) if c not in other)
        return Reading(
            tuple(
                Candidate(
                    QualifiedTarget(
                        _qname_decl_key(c.selected_qname), constructor_binding(name, c), c
                    ),
                    layer,
                    contribution_origin(c.selected_qname, layer),
                )
                for layer, constructors in (
                    (
                        ContributionLayer.DECLARED,
                        [c for c in injected if c.owner_module_id == self._module_id],
                    ),
                    (ContributionLayer.IMPORTED, other),
                )
                for c in constructors
            )
        )

    def routed_at(self, chain: QualifierChain, path: ScopePath, kind: LookupKind) -> Reading:
        """What *chain*'s leading module route alone reaches at *path* beneath it."""
        return self._route_reaches((chain.leading_route, chain.anchored), path, kind)

    def _route_reaches(self, route: _Route, path: ScopePath, kind: LookupKind) -> Reading:
        """What module *route* alone reaches at *path* beneath it."""
        spelled, anchored = route
        candidates = (
            Candidate(
                self._contributed_target(self._cross_module_binding_ref(qname), ()),
                ContributionLayer.IMPORTED,
                ImportedModuleOrigin(qname),
                self._hiding(decls, qname),
                routed=True,
            )
            for qname, decls in qualifier_member_decls(
                self._import_env, spelled, _bare_atom(path), anchored=anchored
            ).items()
        )
        return Reading(tuple(c for c in candidates if self.fits(c.target, kind)))

    def surface_injected(self, chain: QualifierChain, member: str) -> Reading:
        """The root enum inline member module qualifier *chain* injects as *member*.

        A module qualifier is ``::`` alone (this module's own root) or one
        import route. Its surface injects the terminal name of its root
        enums' inline members; a referenced member keeps its own path and is
        never injected. Two injected members are ambiguous, repaired by the
        first in declaration order, and a name only a root enum references
        is refused.
        """
        roots: Iterable[tuple[str, QName]]
        if chain.segments:
            ref = None
            layer = ContributionLayer.IMPORTED
            injected = self._route_injected_members(chain, member)
            roots = (
                (atom, origin)
                for _module, members in qualifier_members(
                    self._import_env, chain.leading_route, anchored=chain.anchored
                )
                for atom, origin in members.items()
                if isinstance(atom, str)
            )
        else:
            ref = self._level_value(self._layer_chain(self._root_scope), member)
            layer = ContributionLayer.DECLARED
            injected = {
                candidate: render_qualified_name(chain, f"{candidate.owner_path[0]}::{member}")
                for candidate in self._constructor_candidates.get(member, ())
                if candidate.owner_module_id == self._module_id and is_root_inline_member(candidate)
            }
            # Earlier REPL entries' root types, then this entry's.
            roots = (
                (root, (self._module_id, root))
                for root in (
                    *self._repl_session_root_type_names,
                    *(item.name for item, path in self._type_declarations if not path),
                )
            )
        if len(injected) > 1:
            ambiguous = self._ambiguous_constructor(
                render_qualified_name(chain, member),
                dict.fromkeys(injected, frozenset({layer})),
                injected[min(injected, key=constructor_candidate_sort_key)],
                chain.span,
            )
            return Reading(refusals=(ambiguous,))
        if injected:
            (constructor,) = injected
            origin = contribution_origin(constructor.qname, layer)
            return Reading((Candidate(QualifiedTarget(None, ref, constructor), layer, origin),))
        referenced = (
            ReferencedMemberError(render_qualified_name(chain, root), member, span=chain.span)
            for root, qname in roots
            if (owner := self._type_owners.owner(qname)) is not None
            and owner.constructor is None
            and owner.alias is None
            and member in owner.referenced
        )
        return Reading(refusals=tuple(itertools.islice(referenced, 1)))

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
        """What type *owner*, made visible by *layer*, selects for *rest* by its own member table.

        Each name of *rest* but the last must name a type declared beneath
        the one before; one the owner so far only references or hides is
        refused. The owner reached last decides the member: a referenced or
        hidden member is refused, and a type it declares (an inline enum
        member or a nested type) is not selected: only a contribution reaching
        its full path selects it, so a ``hiding`` removes exactly that path. An
        alias's projection, or a record's own spelling, selects. Any other
        path beneath an alias is its target's (:meth:`_beneath_alias`), read
        as a declaration of *kind*.

        This module's own declarations beneath an owner another module
        declares, or the type its alias chain ends at, are read beneath every
        own scope path declaring its path (``def Geo::m`` with ``Geo`` an alias
        of an imported ``Base``), and win it, unless only a module route
        reached the owner (*routed*), which reads that module's view alone;
        an own owner's spellings are own scope paths, which :meth:`own_at`
        reads. What an own alias reaches is own (*layer*), as its target's
        spelling there would be.
        """
        if not routed:
            self._declare_scope_paths()
            declared_types = tuple(
                declared
                for declared in self._declared_types(owner)
                if declared[0] != self._module_id
            )
            own = sum(
                (
                    *(
                        self.own_at((*spelling, *rest), kind)
                        for declared in declared_types
                        for spelling in self._declaring_spellings(declared, rest[0])
                    ),
                    *(
                        self._own_spelled_at(path, kind)
                        for module_id, atom in declared_types
                        for path in self._placed.get(
                            (module_id, _bare_atom((*_bare_path(atom), *rest))), ()
                        )
                    ),
                ),
                Reading(),
            )
            if own.candidates:
                return own
        return self._selected_by_table(owner, layer, rest, chain, kind, routed=routed)

    def _declared_types(self, key: DeclarationKey) -> tuple[QName, ...]:
        """The types whose paths type *key* stands for: its final declared type's, renamed first.

        An alias that is another name for an alias of its own stands for
        that one's own path too (:attr:`~agm.agl.scope.type_owners.AliasChain.renamed`).
        """
        _named, renamed, declared = self._declared_type(key)
        return (declared,) if renamed is None or renamed == declared else (renamed, declared)

    def _declared_type(self, key: DeclarationKey) -> tuple[QName, QName | None, QName]:
        """Return the type *key* names (:meth:`identity`), the alias it renames, and its final type.

        The renamed alias is the alias of its own the named one is
        (:attr:`~agm.agl.scope.type_owners.AliasChain.renamed`); the final
        type is the one its alias chain ends at, else the named type itself.
        """
        chain = self._type_owners.chain(self._identity_path(key))
        return chain.identity, chain.renamed, chain.final or chain.identity

    def stands_for(self, key: DeclarationKey) -> tuple[AliasTarget, ...]:
        """See :meth:`~agm.agl.scope.lookup.PathSources.stands_for`.

        Nothing while what an alias reaches is read (:meth:`_reaching`): that
        is what its sites declare, and a step reads its own contributions.
        """
        owner = None if self._reads_reach else self._type_owners.owner(_key_qname(key))
        if owner is None or owner.alias is None:
            return ()
        builtin = owner.builtin_name
        if builtin is not None:
            return (AliasTarget(None, (builtin,)),)
        return tuple(
            AliasTarget(declared, _bare_path(declared[1]))
            for declared in self._declared_types(key)
            if declared != _key_qname(key)
        )

    def names_only(self, target: AliasTarget, named: Reading) -> bool:
        """See :meth:`~agm.agl.scope.lookup.PathSources.names_only`."""
        return target.declared is None or all(
            target.declared in (identity, declared)
            for identity, _renamed, declared in (
                self._declared_type(candidate.target.key)
                for candidate in named.candidates
                if candidate.target.key is not None
            )
        )

    def _selected_by_table(
        self,
        owner: DeclarationKey,
        layer: ContributionLayer,
        rest: ScopePath,
        chain: QualifierChain,
        kind: LookupKind,
        *,
        routed: bool,
    ) -> Reading:
        """What type *owner*'s own member table selects for *rest* (:meth:`projected`)."""
        segments = chain.segments
        start = len(segments) + 1 - len(rest)
        current = _key_qname(owner)
        for index, name in enumerate(rest, start):
            reached = self._type_owners.owner(current)
            if reached is None:
                return Reading()
            table = reached
            error = self._owner_member_error(table, chain, index, name)
            if error is not None:
                return Reading(refusals=(error,))
            if (table.target is not None or table.builtin is not None) and not (
                index == len(segments)
                and (name in table.members or table.select(name, segments[-1].name) is not None)
            ):
                return self._beneath_alias(
                    current, table, rest[index - start :], layer, chain, kind, routed=routed
                )
            current = (current[0], _bare_atom((*_bare_path(current[1]), name)))
        if (table.alias is None and name in table.members) or self._type_owners.is_declared(
            current
        ):
            return Reading()
        return self._selected_constructor(table, current, layer, chain)

    @staticmethod
    def _selected_constructor(
        table: TypeOwner, member: QName, layer: ContributionLayer, chain: QualifierChain
    ) -> Reading:
        """The constructor *table* selects for its member path *member*, which *chain* spells."""
        constructor = table.select(_bare_path(member[1])[-1], chain.segments[-1].name)
        if constructor is None:
            return Reading()
        origin = contribution_origin(member, layer)
        target = QualifiedTarget(_qname_decl_key(member), None, constructor)
        return Reading((Candidate(target, layer, origin),))

    def _beneath_alias(
        self,
        alias: QName,
        table: TypeOwner,
        path: ScopePath,
        layer: ContributionLayer,
        chain: QualifierChain,
        kind: LookupKind,
        *,
        routed: bool,
    ) -> Reading:
        """What *path* beneath *alias* (whose owner is *table*) selects as a declaration of *kind*.

        An alias segment stands for its target's path, where the modules
        declaring the aliases on the way reach what they declare too, and the
        target's member table selects for a path of several names as for the
        target's spelling (``Geo::In::In`` as ``Base::In::In``) -- for one
        name, the alias's own table, holding its target's members, names and
        refusals, has decided (:meth:`_selected_by_table`); a path a
        ``hiding`` at the alias's site removed is refused. An alias only a module route reached
        (*routed*) reads what that route reaches at the target's path too
        (``al::Geo::u`` as ``al::Base::u``, re-exports included).
        An alias of a built-in type stands for each of its
        :attr:`~TypeOwner.scopes`, and, unless only a module route reached it
        (*routed*), for this module's own path its name spells, however this
        module spells the declarations beneath.
        """
        route = (chain.leading_route, chain.anchored) if routed else None
        reach, stood = self._kept(
            self._kept_stood_for,
            (alias, path, layer, kind, route),
            lambda: self._stood_for(alias, table, path, layer, kind, route),
        )
        if reach.hidden:
            return self._hidden_beneath(chain, path)
        if reach.target is None or len(path) == 1:
            return stood
        return stood + self._selected_by_table(
            _qname_decl_key(reach.target), layer, path, chain, kind, routed=routed
        )

    def _stood_for(
        self,
        alias: QName,
        table: TypeOwner,
        path: ScopePath,
        layer: ContributionLayer,
        kind: LookupKind,
        route: _Route | None,
    ) -> tuple[AliasReach, Reading]:
        """What *path* beneath *alias* stands for, and the declarations of *kind* there.

        As :meth:`_beneath_alias` reads them, before the target's member
        table: those declared at each path it stands for, and what module
        *route*, when only it reached the alias, reaches there
        (:meth:`_route_reads_beneath`). Nothing when hidden.
        """
        reach = self._alias_reach(alias, table, path, routed=route is not None)
        if reach.hidden:
            return reach, Reading()
        declared = sum(
            (self._declared_at(qname, layer, kind, sites=reach.sites) for qname in reach.paths),
            Reading(),
        )
        if route is None:
            return reach, declared
        routed = sum(
            (self._route_reaches(route, _bare_path(qname[1]), kind) for qname in reach.paths),
            Reading(),
        )
        if routed.candidates and self._route_reads_beneath(route, alias):
            return reach, declared + routed
        return reach, declared

    def _route_reads_beneath(self, route: _Route, alias: QName) -> bool:
        """Whether module *route* reaches beneath alias *alias* what it reaches beneath its types.

        Unless its spelling of a type the alias stands for names another type,
        whose paths are not the alias's.
        """
        return all(
            self.names_only(target, self._route_reaches(route, target.path, LookupKind.TYPE))
            for target in (
                AliasTarget(declared, _bare_path(declared[1]))
                for declared in self._declared_types(_qname_decl_key(alias))
            )
        )

    def _alias_reach(
        self, alias: QName, table: TypeOwner, path: ScopePath, *, routed: bool
    ) -> AliasReach:
        """What *path* beneath *alias* (whose owner is *table*) stands for (:meth:`_beneath_alias`).

        Beneath an alias of a built-in type, this module's own path its name
        spells too, unless only a module route reached it (*routed*).
        """
        reach = self._type_owners.alias_reach(alias, table, path)
        name = table.builtin_name
        if name is None or routed:
            return reach
        own = (self._module_id, _bare_atom((name, *path)))
        return reach._replace(paths=(*(qname for qname in reach.paths if qname != own), own))

    def projected_origins(self, alias: DeclarationKey, rest: ScopePath) -> frozenset[QName]:
        """See :meth:`~agm.agl.scope.lookup.PathSources.projected_origins`."""
        qname = _key_qname(alias)
        table = self._type_owners.owner(qname)
        if table is None or table.alias is None:
            return frozenset()
        reach = self._alias_reach(qname, table, rest, routed=False)
        if reach.hidden:
            return frozenset()
        return frozenset(
            path
            for path in reach.paths
            if (
                self.own_origins(_bare_path(path[1]))
                if path[0] == self._module_id
                else self._type_owners.names_qualifier(path, reach.sites)
            )
        )

    @staticmethod
    def _hidden_beneath(chain: QualifierChain, path: ScopePath) -> Reading:
        """The refusal of *path*, which *chain* spells beneath an alias, as hidden."""
        spelling = render_qualifier_path(chain)
        return Reading(refusals=(HiddenMemberError(spelling, path[-1], span=chain.span),))

    def _declared_at(
        self,
        qname: QName,
        layer: ContributionLayer,
        kind: LookupKind,
        *,
        sites: Sequence[ModuleId] = (),
    ) -> Reading:
        """The declarations at full path *qname*, as ones of *kind*, made visible by *layer*.

        This module's own declaration at *qname* wins, whether *qname* names
        its own type or an imported one. Failing that: another module's --
        the one it declares there, however spelled, and the one each of
        *sites* writes declared there (:meth:`declared_path`), including a
        foreign module writing a member beneath a type this module owns.
        """
        module_id, atom = qname
        if module_id == self._module_id:
            own = Reading(
                tuple(
                    replace(candidate, layer=layer, origin=contribution_origin(qname, layer))
                    for candidate in self.own_at(_bare_path(atom), kind).candidates
                )
            )
            if own.candidates or not sites:
                return own
        declared = [qname] if qname in self._decl_info else []
        # A site writing a declaration at *qname*'s path, however it is
        # spelled there, keys it at that one declared path in its own module.
        declared.extend(
            (site, atom)
            for site in sites
            if site not in (module_id, self._module_id)
            and () in self._type_owners.written_beneath(site, qname)
        )
        candidates = (
            Candidate(
                self._contributed_target(self._cross_module_binding_ref(declaration), ()),
                layer,
                contribution_origin(declaration, layer),
            )
            for declaration in declared
        )
        return Reading(tuple(c for c in candidates if self.fits(c.target, kind)))

    def inline_arity(self, owner: DeclarationKey, member: str, written: str) -> int | None:
        """The arity of type *owner*, spelled *written*, when it owns *member* inline."""
        reached = self._type_owners.owner(_key_qname(owner))
        if reached is None or (
            member not in reached.members and reached.select(member, written) is None
        ):
            return None
        return reached.arity

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

        See :meth:`~agm.agl.scope.lookup.PathSources.beneath_applied`.
        """
        qname = _key_qname(applied)
        table = self._type_owners.owner(qname)
        if table is not None and table.alias is not None:
            return self.projected(applied, layer, rest, chain, kind, routed=routed)
        module_id, atom = qname
        path = (module_id, _bare_atom((*_bare_path(atom), *rest)))
        declared = self._declared_at(path, layer, kind)
        if table is None or declared.candidates or len(rest) > 1:
            return declared
        return self._selected_constructor(table, path, layer, chain)

    def application(
        self, key: DeclarationKey, segment: QualifierSegment, site: ScopePath
    ) -> Application | None:
        """What *segment*, selecting type *key* and written in *site*, stands for applied.

        See :meth:`~agm.agl.scope.lookup.PathSources.application`.
        """
        owners = self._type_owners
        qname, arguments, application = _key_qname(key), segment.type_args, None
        while (
            (parameter := owners.projected_parameter(qname)) is not None
            and arguments is not None
            and parameter[0] < len(arguments)
            and is_nominal_type_expr(argument := arguments[parameter[0]], ())
            and (selection := self.type_name_selection_at(site, argument, every_use=False))
            is not None
            and (projected := owners.declared_path(selection)) is not None
        ):
            qname = projected
            arguments = argument.args if isinstance(argument, AppliedT) else None
            arity = parameter[1] if application is None else application.arity
            application = Application(_qname_decl_key(projected), arity)
        return application

    def structural(self, key: DeclarationKey) -> TypeExpr | None:
        """The structural type type *key* is an alias of, if it is one; it hosts no paths."""
        owners = self._type_owners
        reached = owners.owner(owners.identity(_key_qname(key)))
        if (
            reached is None
            or reached.alias is None
            or reached.target is not None
            or reached.builtin is not None
        ):
            return None
        return reached.stands_for

    def applies(self, key: DeclarationKey) -> bool:
        """Whether type *key* is an alias applying its target to type arguments of its own."""
        owners = self._type_owners
        reached = owners.owner(owners.identity(_key_qname(key)))
        return reached is not None and reached.applies

    @contextmanager
    def _reaching(self) -> Iterator[None]:
        """Read what an alias reaches: no path this module declares is hidden.

        Its walk may yet have to bind the declaration, which a lookup then
        misses; the path is the declaration's all the same.
        """
        previous, self._reads_reach = self._reads_reach, True
        try:
            yield
        finally:
            self._reads_reach = previous

    def _declares_ordinary(self, path: ScopePath) -> bool:
        """Whether this module declares a function or binding at full *path*."""
        return self._scope_entity_kinds.get((self._module_id, path[:-1], path[-1])) == "ordinary"

    def hidden_at(self, step: ScopePath, path: ScopePath) -> bool:
        """Whether a ``hiding`` of a contribution anchored at or above *step* removed *path*.

        None did for a read of what an alias reaches (:meth:`_reaching`) when
        this module declares *path*.
        """
        if self._reads_reach and self._declares_ordinary(path):
            return False
        for layer in self._layer_chain(self._scope_nodes[step]):
            atom = _bare_atom(path[len(layer.scope_path) :])
            if any(
                atom_under_prefix(atom, _item_path(item))
                for decl in self._uses.visible(layer)
                for item in decl.hidden
            ):
                return True
        env = self._import_env
        for node_id in {*env.decl_hidden, *env.decl_hiding}:
            anchor = self._import_decl_scope_paths.get(node_id, ())
            relative = path[len(anchor) :]
            if step[: len(anchor)] == anchor and (
                _bare_atom(relative) in env.decl_hidden.get(node_id, ())
                or self._hides_beneath_alias(node_id, relative)
            ):
                return True
        return (
            len(path) > 1 and self._route_hides((path[0],), path[1:], anchored=False)
        ) or self._withheld(unqualified_exposures(env), path)

    def _withheld(self, exposed: Iterable[Exposure], path: ScopePath) -> bool:
        """Whether the imports exposing the nearest qualifier of *path* remove what it names.

        *exposed* is what they expose. Their own ``hiding`` or the imported
        module's export ``hiding`` removed the declaration the rest of *path*
        names beneath the longest prefix of *path* they expose: every import
        exposing that prefix itself, else any exposing a path beneath it,
        which none reaches the declaration by.
        """
        exposures = [(_bare_path(atom), qname, decls) for atom, qname, decls in exposed]
        for end in range(len(path) - 1, 0, -1):
            reaching = [exposure for exposure in exposures if exposure[0][:end] == path[:end]]
            exact: dict[QName, frozenset[int]] = {}
            for exposed_path, qname, decls in reaching:
                if len(exposed_path) == end:
                    exact[qname] = exact.get(qname, frozenset()) | decls
            ways = [(qname, qname, decls) for qname, decls in exact.items()] or [
                (
                    (qname[0], _bare_atom(_bare_path(qname[1])[: end - len(exposed_path)])),
                    qname,
                    frozenset({node_id}),
                )
                for exposed_path, qname, decls in reaching
                for node_id in decls
            ]
            if ways:
                return any(
                    removes(
                        self._hiding(decls, entry),
                        self._declaration_beneath(above, path[end:]),
                        self,
                    )
                    for above, entry, decls in ways
                )
        return False

    def _hides_beneath_alias(self, node_id: int, path: ScopePath) -> bool:
        """Whether import *node_id*'s ``hiding`` names an alias above *path*, declared beneath.

        The item removed the alias's spelling, and with its target every
        path its target declares beneath.
        """
        return any(
            not named.beneath
            and named.item == path[: (size := len(named.item))]
            and size < len(path)
            and self.aliases(_qname_decl_key(named.declaration))
            and self._named_beneath(named.declaration, path, path[size:], named.span)
            for named in self._import_env.decl_hiding.get(node_id, ())
        )

    def routed_hidden(self, chain: QualifierChain, path: ScopePath) -> bool:
        """Whether a ``hiding`` removed *path* from *chain*'s leading module route."""
        return self._route_hides(chain.leading_route, path, anchored=chain.anchored)

    def _route_hides(self, route: tuple[str, ...], path: ScopePath, *, anchored: bool) -> bool:
        """Whether a ``hiding`` removed *path* from module *route*: it, or an alias above it.

        Beneath an alias the route reaches, so did one removing the target's
        path it stands for there (``al::Geo::w`` as ``al::Base::w``), where it
        reads beneath the alias (:meth:`_route_reads_beneath`).
        """
        env = self._import_env
        return (
            qualifier_hides(env, route, _bare_atom(path), anchored=anchored)
            or any(
                self._hides_beneath_alias(node_id, path)
                for node_id in qualifier_decls(env, route, anchored=anchored)
            )
            or self._withheld(qualifier_exposures(env, route, anchored=anchored), path)
            or any(
                self._route_hides(route, _bare_path(target[1]), anchored=anchored)
                for end in range(1, len(path))
                for qname in qualifier_member_decls(
                    env, route, _bare_atom(path[:end]), anchored=anchored
                )
                if (table := self._type_owners.owner(qname)) is not None
                and table.alias is not None
                and self._route_reads_beneath((route, anchored), qname)
                for target in self._alias_reach(qname, table, path[end:], routed=True).paths
            )
        )

    def reader(self) -> Reader:
        """This module, with the segments of every path it declares -- retained ones included."""
        found = self._reader
        if found is None:
            paths = (
                *self._scope_nodes,
                *self._repl_session_type_paths,
                *((*path, name) for _module_id, path, name in self._declarations),
            )
            found = self._reader = Reader(
                self._module_id, frozenset(segment for path in paths for segment in path)
            )
        return found

    def own_origins(self, path: ScopePath) -> frozenset[QName]:
        """Full *path* when it is one of this module's own scope paths or types."""
        qname = (self._module_id, _bare_atom(path))
        if path in self._scope_nodes or self._type_owners.is_declared(qname):
            return frozenset({qname})
        return frozenset()

    def identity(self, key: DeclarationKey) -> DeclarationKey:
        """The declaration *key* names: a renaming alias's is its target's.

        A path beneath an alias is its target's path there, unless this
        module declares it so.
        """
        return _qname_decl_key(self._type_owners.identity(self._identity_path(key)))

    def _identity_path(self, key: DeclarationKey) -> QName:
        """The full path whose identity *key*'s is (:meth:`identity`)."""
        module_id, path, name = key
        qname = _key_qname(key)
        node = self._scope_nodes.get(path) if module_id == self._module_id else None
        if node is None or name not in node.members:
            return self._type_owners.path_target(qname)
        return qname

    def denotes(self, key: DeclarationKey) -> object:
        """What *key* names in an ambiguity: its identity, or what an alias denotes there."""
        denoted = self._type_owners.denotation(_key_qname(key))
        return self.identity(key) if denoted is None else denoted

    def placement(self, key: DeclarationKey) -> DeclarationKey:
        """See :meth:`~agm.agl.scope.lookup.DeclarationNames.placement`.

        As :meth:`identity`, an own declaration is placed by this module's
        scope paths alone.
        """
        module_id, path, name = key
        node = self._scope_nodes.get(path) if module_id == self._module_id else None
        if node is None or name not in node.members:
            return _qname_decl_key(self._type_owners.placement(_key_qname(key)))
        declared = self.declared_path((*path, name))
        return key if declared is None else _qname_decl_key(declared)

    def scopes_of(self, key: DeclarationKey) -> frozenset[DeclarationKey]:
        """See :meth:`~agm.agl.scope.lookup.DeclarationNames.scopes_of`."""
        return frozenset(map(_qname_decl_key, self._type_owners.scopes_of(_key_qname(key))))

    def aliases(self, key: DeclarationKey) -> bool:
        """Whether *key* declares a type alias."""
        owner = self._type_owners.owner(_key_qname(key))
        return owner is not None and owner.alias is not None

    def contributed_origins(self, step: ScopePath, path: ScopePath) -> frozenset[QName]:
        """The scopes and types contributions anchored at or above *step* reach as *path*.

        A contributed function, binding or injected enum member is none.
        """
        found: set[QName] = set()
        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            relative = _bare_path(atom)
            for exposed, refs in layer.bare_contributions.items():
                found |= self._exposed_origins(
                    exposed, relative, [_ref_qname(ref) for ref in refs], _type_qnames(refs)
                )
            for decl in self._uses.visible(layer):
                found |= self._uses.origins(layer.scope_path, decl, relative)
        env = self._import_env
        for exposed, qnames in env.unqualified.items():
            found |= self._exposed_origins(exposed, path, qnames, qnames)
        scope_routes = (
            ((), env.unqualified_scope_routes),
            *(
                (self._import_decl_scope_paths.get(node_id, ()), routes)
                for node_id, routes in self._reachable_decl_contributions(
                    env.decl_bare_scope_routes, step
                )
            ),
        )
        for anchor, routes in scope_routes:
            for exposed, sources in routes.items():
                rest = relative_under(exposed, path[len(anchor) :])
                if rest is not None:
                    found.update(
                        origin
                        for module, source in sources
                        for origin in self._scope_route_origins((module, _route_root(source, rest)))
                    )
        return frozenset(found | self.module_route_origins((path[0],), path[1:], anchored=False))

    def routed_origins(self, chain: QualifierChain, path: ScopePath) -> frozenset[QName]:
        """The scopes and types *chain*'s leading module route reaches as *path* beneath it."""
        return self.module_route_origins(chain.leading_route, path, anchored=chain.anchored)

    def module_route_origins(
        self, route: tuple[str, ...], path: ScopePath, *, anchored: bool
    ) -> frozenset[QName]:
        """The scopes and types module *route* reaches as *path* beneath it; itself for none."""
        env = self._import_env
        if not path:
            return frozenset(
                (module, ()) for module in qualifier_candidates(env, route, anchored=anchored)
            )
        found: set[QName] = set()
        for _module, members in qualifier_members(env, route, anchored=anchored):
            for exposed, qname in members.items():
                found |= self._exposed_origins(exposed, path, (qname,), (qname,))
        for module, scope_paths in qualifier_scope_paths(env, route, anchored=anchored):
            if any(relative_under(atom, path) is not None for atom in scope_paths):
                found |= self._scope_route_origins((module, path))
        return frozenset(found)

    def _exposed_origins(
        self,
        exposed: NameAtom,
        path: ScopePath,
        declarations: Iterable[QName],
        types: Iterable[QName],
    ) -> frozenset[QName]:
        """The scopes and types contributed *exposed* makes *path*: a scope above it, or its type.

        *declarations* are what *exposed* names; *types* those that may be types.
        """
        rest = relative_under(exposed, path)
        if rest is None:
            return frozenset()
        if rest:
            return frozenset(
                (module, _bare_atom(_bare_path(atom)[: -len(rest)]))
                for module, atom in declarations
            )
        return frozenset(qname for qname in types if self._type_owners.is_declared(qname))

    def _imported(
        self, step: ScopePath, path: ScopePath
    ) -> dict[BindingRef, tuple[Layers, Hiding, bool]]:
        """Return what import tails anchored at or above *step* bind at full *path*.

        Every layer from *step* outward contributes the path relative to its
        own; the module root's import tails and the module route spelled by
        its leading name contribute it whole. A binding several contribute
        keeps every one's tag, what the ``hiding`` of each declaration
        contributing it removes, and whether only a module route reached it.
        """
        env = self._import_env
        reached: dict[BindingRef, tuple[Layers, set[frozenset[DeclarationKey]], bool]] = {}

        def add(
            ref: BindingRef,
            layers: Layers,
            decls: Iterable[int],
            entry: QName,
            *,
            routed: bool = False,
        ) -> None:
            found, ways, only_routed = reached.setdefault(ref, (frozenset(), set(), True))
            reached[ref] = found | layers, ways, only_routed and routed
            ways.update(self._hiding(decls, entry))

        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            for ref, layers in layer.bare_contributions.get(atom, {}).items():
                qname = _ref_qname(ref)
                add(
                    ref,
                    layers,
                    (
                        node_id
                        for node_id, members in env.decl_bare.items()
                        if self._import_decl_scope_paths.get(node_id) == layer.scope_path
                        and qname in members.get(atom, ())
                    ),
                    qname,
                )
        for node_id, exposures in self._reachable_decl_contributions(env.decl_tail_beneath, step):
            relative = path[len(self._import_decl_scope_paths.get(node_id, ())) :]
            for exposed, items in exposures.items():
                size = len(_bare_path(exposed))
                if _bare_atom(relative[:size]) != exposed:
                    continue
                for named in items:
                    rest = relative[size:]
                    for key in self._named_beneath(
                        named.declaration, (*named.item, *rest), (*named.beneath, *rest), named.span
                    ):
                        add(
                            self._cross_module_binding_ref(_key_qname(key)),
                            frozenset({ContributionLayer.IMPORTED}),
                            (node_id,),
                            named.declaration,
                        )
        imported = dict(env.unqualified_decls.get(_bare_atom(path), {}))
        routed: set[QName] = set()
        if path[1:]:
            for qname, decls in qualifier_member_decls(
                env, (path[0],), _bare_atom(path[1:])
            ).items():
                if qname not in imported:
                    routed.add(qname)
                imported[qname] = imported.get(qname, frozenset()) | decls
        for qname, decls in imported.items():
            add(
                self._cross_module_binding_ref(qname),
                frozenset({ContributionLayer.IMPORTED}),
                decls,
                qname,
                routed=qname in routed,
            )
        return {
            ref: (layers, frozenset(ways), only_routed)
            for ref, (layers, ways, only_routed) in reached.items()
        }

    def _hiding(self, decls: Iterable[int], entry: QName) -> Hiding:
        """What each import declaration of *decls* removes from export *entry*; none without any.

        Its own ``hiding``'s declarations, and those the imported module's
        export ``hiding`` withholds beneath *entry*.
        """
        withheld = self._import_env.decl_withheld
        return (
            frozenset(
                self._import_hidden(node_id)
                | {_qname_decl_key(q) for q in withheld.get(node_id, {}).get(entry, ())}
                for node_id in decls
            )
            or NOT_HIDDEN
        )

    def _import_hidden(self, node_id: int) -> frozenset[DeclarationKey]:
        """The declarations import declaration *node_id*'s ``hiding`` removes, by identity."""
        found = self._hidden_by.get(node_id)
        if found is None:
            found = self._hidden_by[node_id] = frozenset(
                key
                for named in self._import_env.decl_hiding.get(node_id, ())
                for key in self._named_by(named)
            )
        return found

    def _named_by(self, named: ItemDeclaration) -> frozenset[DeclarationKey]:
        """The declarations import item *named* names, by identity.

        A path it names beneath an exported alias must name a declaration
        there (:meth:`_named_beneath`).
        """
        if not named.beneath:
            return frozenset({self.identity(_qname_decl_key(named.declaration))})
        keys = self._named_beneath(named.declaration, named.item, named.beneath, named.span)
        if not keys:
            raise UnknownMemberError(
                spell_declaration(named.module, named.item),
                span=named.span,
                repair=MissRepair.NOT_EXPORTED,
            )
        return keys

    def _named_beneath(
        self, alias: QName, written: ScopePath, beneath: ScopePath, span: SourceSpan
    ) -> frozenset[DeclarationKey]:
        """The declarations of any kind the last names of *written* name beneath *alias*.

        *beneath* are those names, *alias* the exported alias the ones before
        spell, *span* where they are written. They are read as the alias's
        module route reads them (:meth:`projected`): what the alias reaches
        where it is declared, never this module's own declarations.
        """
        chain = self._probe_chain(written[:-1], written[-1], span, anchored=False)
        return frozenset(
            self.identity(key)
            for kind in LookupKind
            for candidate in self._selected_by_table(
                _qname_decl_key(alias),
                ContributionLayer.IMPORTED,
                beneath,
                chain,
                kind,
                routed=True,
            ).candidates
            if (key := candidate.target.key) is not None
        )

    def _probe_chain(
        self, qualifier: tuple[str, ...], member: str, span: SourceSpan, *, anchored: bool
    ) -> QualifierChain:
        """The spelling ``qualifier::member`` at *span*, looked up but never recorded."""
        node_id = self._program.node_id
        return QualifierChain(
            QualifierAnchor.MODULE if anchored else None,
            tuple(QualifierSegment(segment, None, span, node_id) for segment in qualifier),
            member,
            span,
            node_id,
        )

    def _declaration_beneath(self, qname: QName, path: ScopePath) -> DeclarationKey:
        """The declaration full path *qname* then *path* names (:meth:`identity`)."""
        module_id, atom = qname
        return self.identity(_qname_decl_key((module_id, _bare_atom((*_bare_path(atom), *path)))))

    def tail_removes(self, exposed: NameAtom, qname: QName, declaration: QName) -> bool:
        """Whether every root import tail exposing *qname* as *exposed* removes *declaration*.

        *declaration* is *qname*'s own or a member's beneath it.
        """
        decls = self._import_env.unqualified_decls.get(exposed, {}).get(qname, ())
        return removes(self._hiding(decls, qname), _qname_decl_key(declaration), self)

    def _imported_bindings(self, step: ScopePath, path: ScopePath) -> dict[BindingRef, Layers]:
        """Return what import tails anchored at or above *step* bind at full *path*, with layers.

        A binding every declaration contributing it hides is none.
        """
        return {
            ref: layers
            for ref, (layers, hiding, _routed) in self._imported(step, path).items()
            if not removes(hiding, (ref.module_id, ref.scope_path, ref.name), self)
        }

    def _use_exposures(
        self, step: ScopePath, path: ScopePath, kind: LookupKind
    ) -> Iterator[tuple[Candidate, UseDecl]]:
        """Yield what each use visible at or above *step* exposes of *kind* at full *path*."""
        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            for decl in self._uses.visible(layer):
                for candidate in self._uses.exposure(
                    layer.scope_path, decl, _bare_path(atom), kind
                ):
                    yield candidate, decl

    def _value_binding(self, target: QualifiedTarget, span: SourceSpan) -> BindingRef | None:
        """The binding *target*, selected at *span*, is read through, if any.

        A member only its type's own table selects binds as a variant.
        """
        constructor = target.constructor
        if target.ref is not None or constructor is None:
            return target.ref
        return self.variant_binding_ref(constructor, span)

    def fits(self, target: QualifiedTarget, kind: LookupKind) -> bool:
        """Whether *target* is a declaration of the kind a position takes.

        A type is a declared type contributed as one; a constructor names
        one; a value is a constructor or an ordinary binding.
        """
        ref = target.ref
        if kind is LookupKind.TYPE:
            return (
                target.key is not None
                and (ref is None or ref.contributes_a_type)
                and self._type_owners.is_declared(_key_qname(target.key))
            )
        if kind is LookupKind.CONSTRUCTOR or target.constructor is not None:
            return target.constructor is not None
        return ref is not None and ref.kind is not BinderKind.constructor_binding

    def _contributed_target(
        self, ref: BindingRef, constructors: Collection[ConstructorRef]
    ) -> QualifiedTarget:
        """Return the target contributed *ref* names, with its constructor, if any.

        A local declaration's constructor is the contributing layers' own
        candidate with the same identity; an imported one's is its module's.
        """
        key = (ref.module_id, ref.scope_path, ref.name)
        if ref.module_id != self._module_id:
            return QualifiedTarget(key, ref, self._cross_module_constructor(_ref_qname(ref)))
        constructor = next(
            (
                candidate
                for candidate in constructors
                if (candidate.owner_module_id, candidate.owner_path, candidate.owner_name) == key
            ),
            None,
        )
        return QualifiedTarget(key, ref, constructor)

    def _owner_less(self, key: DeclarationKey) -> bool:
        """Whether declaration *key* is selected directly, not as a type owner's member."""
        module_id, path, _name = key
        return not path or self._type_owners.owner((module_id, _bare_atom(path))) is None

    def variant_binding_ref(self, constructor: ConstructorRef, span: SourceSpan) -> BindingRef:
        """Build the bare-exposed variant convenience binding for *constructor*.

        Marked ``is_variant_member`` so no consumer mistakes the injected
        binding for a type contribution: variant expansion offers a
        constructor and pattern candidate only, never a re-exported type.
        """
        return BindingRef(
            name=constructor.owner_name,
            mutable=False,
            decl_span=span,
            decl_node_id=constructor.owner_decl_node_id,
            kind=BinderKind.constructor_binding,
            module_id=constructor.owner_module_id,
            scope_path=constructor.owner_path,
            is_variant_member=True,
        )

    @staticmethod
    def _layer_chain(layer: ScopeNode | None) -> tuple[ScopeNode, ...]:
        """Return *layer* and every layer enclosing it, innermost first."""
        chain: list[ScopeNode] = []
        while layer is not None:
            chain.append(layer)
            layer = layer.parent
        return tuple(chain)

    def _own_level_value(self, level: tuple[ScopeNode, ...], name: str) -> BindingRef | None:
        """Return this module's own value binding *name* at *level*.

        The module root's binding for a constructor only other modules
        declare stands for their candidates, which are contributions.
        """
        ref = self._level_value(level, name)
        return None if ref is None or self._is_imported_constructor_binding(ref) else ref

    @staticmethod
    def _level_value(level: tuple[ScopeNode, ...], name: str) -> BindingRef | None:
        """Return the first of *level*'s lexical or root layers' own bindings *name*."""
        return next((ref for layer in level if (ref := layer.bindings.get(name)) is not None), None)

    def _is_imported_constructor_binding(self, ref: BindingRef) -> bool:
        """Whether *ref* is the module root's binding for a constructor other modules declare."""
        return ref.kind is BinderKind.constructor_binding and ref.module_id != self._module_id

    def _make_cross_module_ref(self, qname: QName) -> BindingRef:
        """Build a ``BindingRef`` for the declaration *qname* names in its owning module."""
        owning_module, src_name = qname
        info = self._decl_info[qname]
        path = (src_name,) if isinstance(src_name, str) else src_name
        return BindingRef(
            name=path[-1],
            # Only ``var``/``builtin var`` bindings are mutable across a module
            # boundary; every other exported binding (functions, constructors,
            # exported ``let``s, …) is immutable at the reference site.
            mutable=info.kind in (BinderKind.builtin_var_binding, BinderKind.var_binding),
            decl_span=info.decl_span,
            decl_node_id=info.decl_node_id,
            kind=info.kind,
            module_id=owning_module,
            scope_path=path[:-1],
            is_builtin=info.is_builtin,
            is_method=info.is_method,
            is_param=info.is_param,
        )

    def _imports_inject(self, step: ScopePath, name: str) -> Iterator[ConstructorRef]:
        """Yield the other modules' enum members imports inject as bare *name* at *step*.

        An enum injects its members at its own step. At the module root,
        those are the constructors the import tails make bare. In a named
        scope, each enum an import reaches at a full path directly beneath
        *step* injects its member *name* -- unless the ways reaching it
        remove the enum or the member, or, for an inline member, an import
        reaches a record or exception at the member's own full path there.
        """
        if not step:
            yield from (
                candidate
                for candidate in self._constructor_candidates.get(name, ())
                if candidate.owner_module_id != self._module_id
            )
            return
        standalone = declares_bare_constructor(
            map(_ref_qname, self._imported_bindings(step, (*step, name))), self._all_public_types
        )
        for qname, member in self.enum_members_named(name).items():
            if qname[0] == self._module_id or (
                standalone and member.inline_enum_owner_decl_node_id is not None
            ):
                continue
            key = _qname_decl_key(qname)
            member_key = _qname_decl_key(member.selected_qname)
            if any(
                (ref.module_id, ref.scope_path, ref.name) == key
                and not removes(hiding, key, self)
                and not removes(hiding, member_key, self)
                for ref, (_layers, hiding, _routed) in self._imported(step, (*step, key[2])).items()
            ):
                yield member
