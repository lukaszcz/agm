"""What one module's path lookups read: its own declarations, imports and uses, by full path.

:class:`ModuleSources` implements :class:`~agm.agl.scope.lookup.PathSources`
for the module ``_Resolver`` resolves, and records the full path each own
scope path declares. It reads the declaration and constructor tables the
resolver collects and the lexical layers its walk binds: :class:`SourcesHost`
declares what it reads of them.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
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
    ImportEnv,
    ItemDeclaration,
    NameAtom,
    QName,
    ScopeOrigins,
    declares_bare_constructor,
    qualifier_candidates,
    qualifier_decls,
    qualifier_hides,
    qualifier_member_decls,
    qualifier_members,
    qualifier_scope_paths,
)
from agm.agl.scope.lookup import (
    NOT_HIDDEN,
    Candidate,
    Hiding,
    LookupKind,
    QualifiedTarget,
    Reading,
    is_removed,
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
    UnknownMemberError,
    add_layers,
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
    MemberHidden,
    MemberReferenced,
    applies_target,
    owner_member_selection,
)
from agm.agl.scope.type_owners import TypeOwnerIndex
from agm.agl.scope.uses import UseReader
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    Program,
    QualifierChain,
    RecordDef,
    TypeAlias,
    UseDecl,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import (
    render_qualified_name,
    render_qualifier_path,
)


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


class ModuleSources(SourcesHost):
    """The one lookup's reads for one module, by full path (:class:`PathSources`)."""

    def __init__(self) -> None:
        # This module as diagnostics spell declarations for it (:meth:`reader`).
        self._reader: Reader | None = None
        # Every enum this module reads by the names of its members, built on
        # first use.
        self._enum_member_index: dict[str, dict[QName, ConstructorRef]] | None = None
        # Import declaration id -> the declarations its ``hiding`` removes, by identity.
        self._hidden_by: dict[int, frozenset[DeclarationKey]] = {}
        # The full path each own scope path spelled otherwise declares (an
        # alias segment stands for its target's path), and those declaring
        # each; recorded as lookups first read them, once every module's
        # headers are prepared, since reading an alias needs them
        # (``_declare_scope_paths``).
        self._declared_paths: dict[ScopePath, QName] = {}
        self._declaring_paths: dict[QName, tuple[ScopePath, ...]] = {}
        self._undeclared_scope_paths: list[ScopePath] = []
        self._recording_scope_paths = False
        # Once the tables the resolver collects are complete (:meth:`_keep_readings`),
        # the contributions at each full path, and the own types, by what they are read with.
        self._keeping_readings = False
        self._kept_contributions: dict[tuple[ScopePath, ScopePath, LookupKind], Reading] = {}
        self._kept_own_types: dict[ScopePath, Reading] = {}

    def _own_scope_paths(self) -> list[ScopePath]:
        """Return every own scope path, the shorter ones last."""
        return sorted(
            (path for path in self._scope_nodes if path), key=scope_path_sort_key, reverse=True
        )

    def _declare_scope_paths(self) -> None:
        """Record the full path each own scope path declares.

        A scope path whose whole spelling selects a type declares that type's
        path: through an alias, its target's (``def Geo::m`` with ``type Geo
        = Base`` declares ``Base::m``). Any other declares its parent's path
        and its own name -- through an alias of a built-in type, the
        built-in's name (``def T::f`` with ``type T = text`` declares
        ``text::f``). Own scope paths declaring one path are one path: a
        spelling of any reaches what the others declare (:meth:`own_at`).

        The first lookup reading them records them, in rounds: a round
        records the paths shorter first, reading which paths the previous
        rounds recorded as declaring one path, as does any lookup made while
        it runs. A round recording a path declared otherwise, which may
        change what another selects, is followed by one recording every path
        again, until a round changes nothing. A path beneath one not recorded
        yet waits for it; so does one selecting a type not
        :meth:`~TypeOwnerIndex.settled` yet, until a later lookup.
        """
        if self._recording_scope_paths:
            return
        self._recording_scope_paths = True
        declaring = dict(self._declaring_paths)
        self._record_scope_paths(declaring)
        while declaring != self._declaring_paths:
            self._declaring_paths = declaring
            self._declared_paths = {}
            self._undeclared_scope_paths = self._own_scope_paths()
            declaring = {}
            self._record_scope_paths(declaring)
        self._recording_scope_paths = False

    def _record_scope_paths(self, declaring: dict[QName, tuple[ScopePath, ...]]) -> None:
        """Record the waiting scope paths, adding those declared otherwise to *declaring*."""
        waiting: list[ScopePath] = []
        while self._undeclared_scope_paths:
            path = self._undeclared_scope_paths.pop()
            parent = path[:-1]
            if parent and parent not in self._declared_paths:
                waiting.append(path)
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
                waiting.append(path)
                continue
            owner = None if key is None else self._type_owners.owner(_key_qname(key))
            builtin = None if owner is None else owner.builtin_name
            if key is not None and builtin is None:
                qname = _key_qname(self.identity(key))
            else:
                module_id, atom = self._declared_paths.get(parent, (self._module_id, ()))
                qname = (module_id, _bare_atom((*_bare_path(atom), builtin or path[-1])))
            self._declared_paths[path] = qname
            if qname != (self._module_id, _bare_atom(path)):
                declaring[qname] = (*declaring.get(qname, ()), path)
        self._undeclared_scope_paths.extend(reversed(waiting))

    def declared_path(self, path: ScopePath) -> QName | None:
        """The full path own declaration *path* is declared at, when a scope above declares another.

        ``def Geo::m`` with ``type Geo = Base`` is declared at ``Base::m``, in
        ``Base``'s module. ``None`` for a declaration at its own spelling.
        """
        self._declare_scope_paths()
        for end in range(len(path) - 1, 0, -1):
            declared = self._declared_paths.get(path[:end])
            if declared is not None and declared != (self._module_id, _bare_atom(path[:end])):
                module_id, atom = declared
                return module_id, _bare_atom((*_bare_path(atom), *path[end:]))
        return None

    def _declaring(self) -> Mapping[QName, tuple[ScopePath, ...]]:
        """Map each full path that own scope paths spelled otherwise declare to those paths."""
        self._declare_scope_paths()
        return self._declaring_paths

    def _declares(self, qname: QName) -> bool:
        """Whether a declaration of any kind stands at full path *qname*."""
        return any(
            self._declared_at(qname, ContributionLayer.DECLARED, kind).candidates
            for kind in LookupKind
        )

    @staticmethod
    def _owner_member_error(
        owner: TypeOwner, spelling: str, member: str, span: SourceSpan | None
    ) -> AglError | None:
        """Return why ``spelling::member`` is unreachable through *owner*'s own member table.

        A :class:`ReferencedMemberError` for a member *owner* only
        references, a :class:`HiddenMemberError` for one its alias's import
        hides; ``None`` when the member is neither.
        """
        selection = owner_member_selection(owner, member)
        if isinstance(selection, MemberReferenced):
            return ReferencedMemberError(spelling, member, span=span)
        if isinstance(selection, MemberHidden):
            return HiddenMemberError(spelling, member, span=span)
        return None

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
        """Index every enum this module reads by the names of its members.

        An inline member wins its name over an injected one; a current
        declaration supersedes a retained enum at its path.
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
        index: dict[str, dict[QName, ConstructorRef]] = {}
        for qname, owner in owners.items():
            for member_name, constructor in (
                *owner.members.items(),
                *((injected.owner_name, injected) for injected in owner.injected),
            ):
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
                for origin in contribution_origins(candidate.qname, layers)
            ),
            repair=repair,
            span=span,
            reader=self.reader(),
        )

    def _keep_readings(self) -> None:
        """Keep what each full path's contributions and own types read from now on.

        The resolver calls this once the tables it collects are complete and
        its walk binds nothing a path read sees. A read made while a use's
        own read is in progress, which sees only the uses written before it,
        or while an alias is resolved, which it presumes, is not kept.
        """
        self._keeping_readings = True

    def _kept[K](self, kept: dict[K, Reading], key: K, read: Callable[[], Reading]) -> Reading:
        """Return what *read* reads, kept in *kept* under *key* when it is final."""
        if not (
            self._keeping_readings
            and self._uses.reads_every_use
            and not self._type_owners.resolving
        ):
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
        declaring = self._declaring()
        declared = self._declared_paths.get(parent, (self._module_id, _bare_atom(parent)))
        return sum(
            (
                self._own_spelled_at((*spelling, path[-1]), kind)
                for spelling in declaring.get(declared, ())
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
        declaring = self._declaring()
        if parent in self._declared_paths:
            return reading
        return sum(
            (
                self._own_spelled_at((*spelling, path[-1]), kind)
                for (module_id, atom), spellings in declaring.items()
                if module_id != self._module_id and atom == _bare_atom(parent)
                for spelling in spellings
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
                        (c.owner_module_id, c.owner_path, c.owner_name),
                        constructor_binding(name, c),
                        c,
                    ),
                    layer,
                    contribution_origin(c.qname, layer),
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
        candidates = (
            Candidate(
                self._contributed_target(self._cross_module_binding_ref(qname), ()),
                ContributionLayer.IMPORTED,
                ImportedModuleOrigin(qname),
                self._hiding(decls, qname),
                routed=True,
            )
            for qname, decls in qualifier_member_decls(
                self._import_env, chain.leading_route, _bare_atom(path), anchored=chain.anchored
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
        hidden member is refused, and so is a type it declares (an inline enum
        member or a nested type), which only a contribution reaching its full
        path selects -- so a ``hiding`` removes exactly that path. An
        alias's projection, or a record's own spelling, selects. Any other
        path beneath an alias is its target's (:meth:`_beneath_alias`), read
        as a declaration of *kind*.

        This module's own declarations beneath an owner another module
        declares are read beneath every own scope path declaring its path
        (``def Geo::m`` with ``Geo`` an alias of an imported ``Base``), and win
        it, unless only a module route reached the owner (*routed*), which
        reads that module's view alone; an own owner's spellings are own scope
        paths, which :meth:`own_at` reads. What an own alias reaches is own
        (*layer*), as its target's spelling there would be.
        """
        declared = _key_qname(self.identity(owner))
        if not routed and declared[0] != self._module_id:
            own = sum(
                (
                    self.own_at((*spelling, *rest), kind)
                    for spelling in self._declaring().get(declared, ())
                ),
                Reading(),
            )
            if own.candidates:
                return own
        return self._selected_by_table(owner, layer, rest, chain, kind, routed=routed)

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
            spelling = render_qualifier_path(replace(chain, segments=segments[:index]))
            error = self._owner_member_error(table, spelling, name, chain.span)
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
            return Reading(refusals=(HiddenMemberError(spelling, name, span=chain.span),))
        constructor = table.select(name, segments[-1].name)
        if constructor is None:
            return Reading()
        key = _qname_decl_key(current)
        origin = contribution_origin(current, layer)
        return Reading((Candidate(QualifiedTarget(key, None, constructor), layer, origin),))

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
        declaring the aliases on the way reach what they declare too; a path
        a ``hiding`` at the alias's site removed is refused. An alias of a
        built-in type stands for each of its :attr:`~TypeOwner.scopes`, and,
        unless only a module route reached it (*routed*), for this module's
        own path its name spells, however this module spells the declarations
        beneath.
        """
        name = table.builtin_name
        if name is not None:
            if table.hides(path):
                return self._hidden_beneath(chain, path)
            own = frozenset() if routed else {(self._module_id, _bare_atom((name,)))}
            scopes = table.scopes | own
            return sum(
                (
                    self._declared_at(
                        (module_id, _bare_atom((*_bare_path(atom), *path))), layer, kind
                    )
                    for module_id, atom in scopes
                ),
                Reading(),
            )
        beneath = self._type_owners.beneath_alias(alias, table, path)
        if beneath is None:
            return Reading()
        qname, hidden = beneath
        if hidden:
            return self._hidden_beneath(chain, path)
        return self._declared_at(qname, layer, kind, sites=self._type_owners.alias_sites(alias))

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
        sites: Collection[ModuleId] = (),
    ) -> Reading:
        """The declarations at full path *qname*, as ones of *kind*, made visible by *layer*.

        Another module's: the one it declares there, however spelled, and
        the one each of *sites* writes declared there (:meth:`declared_path`).
        """
        module_id, atom = qname
        if module_id == self._module_id:
            return Reading(
                tuple(
                    replace(candidate, layer=layer, origin=contribution_origin(qname, layer))
                    for candidate in self.own_at(_bare_path(atom), kind).candidates
                )
            )
        declared = [qname] if qname in self._decl_info else []
        for site in (module_id, *sorted(set(sites).difference({module_id}), key=str)):
            written = self._type_owners.declared_at(site, qname)
            if site != self._module_id and written is not None:
                declared.append(written)
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

    def applies(self, key: DeclarationKey) -> bool:
        """Whether type *key* is an alias applying its target to type arguments of its own."""
        owners = self._type_owners
        reached = owners.owner(owners.identity(_key_qname(key)))
        return reached is not None and reached.alias is not None and applies_target(reached.alias)

    def hidden_at(self, step: ScopePath, path: ScopePath) -> bool:
        """Whether a ``hiding`` of a contribution anchored at or above *step* removed *path*."""
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
        ) or self._withheld(lambda atom: env.unqualified_decls.get(atom, {}), path)

    def _withheld(
        self, exposed: Callable[[NameAtom], Mapping[QName, frozenset[int]]], path: ScopePath
    ) -> bool:
        """Whether the imports exposing a prefix of *path* (*exposed*) all remove what it names.

        The imported module's export ``hiding`` withheld the declaration the
        rest of *path* names beneath that prefix's.
        """
        return any(
            removes(
                self._hiding(decls, qname),
                self._declaration_beneath(qname, path[end:]),
                self,
            )
            for end in range(1, len(path))
            for qname, decls in exposed(_bare_atom(path[:end])).items()
        )

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
            and self._declares(
                _key_qname(self._declaration_beneath(named.declaration, path[size:]))
            )
            for named in self._import_env.decl_hiding.get(node_id, ())
        )

    def routed_hidden(self, chain: QualifierChain, path: ScopePath) -> bool:
        """Whether a ``hiding`` removed *path* from *chain*'s leading module route."""
        return self._route_hides(chain.leading_route, path, anchored=chain.anchored)

    def _route_hides(self, route: tuple[str, ...], path: ScopePath, *, anchored: bool) -> bool:
        """Whether a ``hiding`` removed *path* from module *route*: it, or an alias above it."""
        env = self._import_env
        return (
            qualifier_hides(env, route, _bare_atom(path), anchored=anchored)
            or any(
                self._hides_beneath_alias(node_id, path)
                for node_id in qualifier_decls(env, route, anchored=anchored)
            )
            or self._withheld(
                lambda atom: qualifier_member_decls(env, route, atom, anchored=anchored), path
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
        owners = self._type_owners
        module_id, path, name = key
        qname = _key_qname(key)
        node = self._scope_nodes.get(path) if module_id == self._module_id else None
        if node is None or name not in node.members:
            return _qname_decl_key(owners.declaration(qname))
        return _qname_decl_key(owners.identity(qname))

    def denotes(self, key: DeclarationKey) -> object:
        """What *key* names in an ambiguity: its identity, or the type an alias denotes."""
        identity = self.identity(key)
        denoted = self._type_owners.denotation(_key_qname(identity))
        return identity if denoted is None else denoted

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
                    qname = _key_qname(
                        self._declaration_beneath(
                            _key_qname(self._named_by(named)), relative[size:]
                        )
                    )
                    if self._declares(qname):
                        add(
                            self._cross_module_binding_ref(qname),
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
        """The declarations import declaration *node_id*'s ``hiding`` removes, by identity.

        A path an item names beneath an exported alias is its target's, and
        must name a declaration there.
        """
        found = self._hidden_by.get(node_id)
        if found is None:
            found = self._hidden_by[node_id] = frozenset(
                self._named_by(named) for named in self._import_env.decl_hiding.get(node_id, ())
            )
        return found

    def _named_by(self, named: ItemDeclaration) -> DeclarationKey:
        """The declaration import item *named* names, by identity.

        A path it names beneath an exported alias is its target's, and must
        name a declaration there.
        """
        key = self._declaration_beneath(named.declaration, named.beneath)
        if named.beneath and not self._declares(_key_qname(key)):
            raise UnknownMemberError(
                spell_declaration(named.module, named.item),
                span=named.span,
                repair=MissRepair.NOT_EXPORTED,
            )
        return key

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

    def _contributed_bindings(
        self, step: ScopePath, path: ScopePath, kind: LookupKind
    ) -> dict[BindingRef, Layers]:
        """Return what contributions anchored at or above *step* bind at full *path*, with layers.

        *kind* is the position's, which decides what a ``use`` exposes. A
        member only its type's own table selects binds as a variant.
        """
        bindings = self._imported_bindings(step, path)
        exposed = (
            self._value_binding(candidate.target, decl.span)
            for candidate, decl in self._use_exposures(step, path, kind)
            if not is_removed(candidate, self)
        )
        for ref in filter(None, exposed):
            add_layers(bindings, ref, (ContributionLayer.USE,))
        return bindings

    def _value_binding(self, target: QualifiedTarget, span: SourceSpan) -> BindingRef | None:
        """The binding *target*, selected at *span*, is read through, if any.

        A member only its type's own table selects binds as a variant.
        """
        constructor = target.constructor
        if target.ref is not None or constructor is None:
            return target.ref
        return self.variant_binding_ref(constructor, span)

    def _contributed_constructors(
        self, step: ScopePath, path: ScopePath, kind: LookupKind
    ) -> dict[ConstructorRef, Layers]:
        """Return the constructor candidates layers anchored at or above *step* give full *path*.

        *kind* is the position's, as for :meth:`_contributed_bindings`.
        """
        constructors: dict[ConstructorRef, Layers] = {}
        for candidate, _decl in self._use_exposures(step, path, kind):
            if candidate.target.constructor is not None and not is_removed(candidate, self):
                add_layers(constructors, candidate.target.constructor, (ContributionLayer.USE,))
        return constructors

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
            member_key = (member.owner_module_id, member.owner_path, member.owner_name)
            if any(
                (ref.module_id, ref.scope_path, ref.name) == key
                and not removes(hiding, key, self)
                and not removes(hiding, member_key, self)
                for ref, (_layers, hiding, _routed) in self._imported(step, (*step, key[2])).items()
            ):
                yield member
