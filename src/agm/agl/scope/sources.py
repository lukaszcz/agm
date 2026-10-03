"""What one module's path lookups read: its own declarations, imports and uses, by full path.

:class:`ModuleSources` implements :class:`~agm.agl.scope.lookup.PathSources`
for the module ``_Resolver`` resolves, and records the full path each own
scope path declares. It reads the declaration and constructor tables the
resolver collects and the lexical layers its walk binds: :class:`SourcesHost`
declares what it reads of them.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import Protocol

from agm.agl.diagnostics import (
    AglError,
    HiddenMemberError,
    ReferencedMemberError,
)
from agm.agl.modules.ids import (
    ModuleId,
    Reader,
    render_route_member,
    spell_declaration,
)
from agm.agl.scope.imports import (
    BareRoute,
    ImportEnv,
    ItemDeclaration,
    NameAtom,
    QName,
    ScopeOrigins,
    contribution_routes,
    declares_bare_constructor,
    qualifier_candidates,
    qualifier_exposures,
    qualifier_hides,
    qualifier_member_decls,
    qualifier_members,
    qualifier_scope_paths,
    unqualified_exposures,
)
from agm.agl.scope.lookup import (
    NOT_HIDDEN,
    Application,
    Candidate,
    Hiding,
    LookupKind,
    QualifiedTarget,
    Reading,
    lookup_origins,
    lookup_qualified,
    lookup_steps,
    lookup_through,
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
    ModuleResolution,
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
    member_chain,
    owner_member_selection,
    selection_node_id,
)
from agm.agl.scope.type_owners import TypeOwnerIndex
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
    named_builtin_type,
    render_qualified_name,
    render_qualifier_path,
)

type _Route = tuple[tuple[str, ...], bool]
"""A module route's spelling, and whether it is anchored."""
type _Reach = tuple[QualifierAnchor | None, tuple[QualifierSegment, ...]]
"""How a spelling reads the paths beneath an alias it reached: its anchor and leading route."""

type _Exposed = tuple[ScopePath, QName, frozenset[int]]
"""A path imports expose, what it reaches there, and the import declarations exposing it."""

type _Through = tuple[QName, ScopePath, LookupKind | None, int]
"""A path beneath an alias read as its target as written: of a kind, or as a qualifier.

With how many names past the alias project their member tables.
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


def _target_spelling(alias: TypeAlias) -> NameT | AppliedT | None:
    """The type name *alias*'s target is written as: a built-in type's own; none if structural."""
    target = alias.type_expr
    if is_nominal_type_expr(target, alias.type_params):
        return target
    return named_builtin_type(target)


def _type_qnames(refs: Iterable[BindingRef]) -> Iterator[QName]:
    """The full paths of *refs* that may name a type; an injected enum member never does."""
    return (_ref_qname(ref) for ref in refs if ref.contributes_a_type)


class SourcesHost(Protocol):
    """What reading a module's paths needs of the resolver collecting and walking it.

    The module's program, import environment and the whole program's tables;
    the declarations, scope layers and constructor candidates it collects;
    the root layer its walk binds; its uses; and every module's sources.
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
    _scope_nodes: Mapping[ScopePath, ScopeNode]
    _declarations: Mapping[DeclarationKey, BindingRef]
    _scope_entity_kinds: Mapping[DeclarationKey, str]
    _import_decl_scope_paths: Mapping[int, ScopePath]
    _type_declarations: Sequence[tuple[RecordDef | EnumDef | ExceptionDef | TypeAlias, ScopePath]]
    _scoped_constructor_candidates: Mapping[tuple[ScopePath, str], Sequence[ConstructorRef]]
    _constructor_candidates: Mapping[str, Sequence[ConstructorRef]]
    _injected_constructors: Mapping[tuple[ScopePath, str], Sequence[ConstructorRef]]
    _uses: UseReader
    # What each module of the program reads where its aliases are declared.
    _site_sources: Callable[[ModuleId], ModuleSources]

    def type_name_selection_at(
        self, scope_path: ScopePath, spelling: NameT | AppliedT, *, every_use: bool
    ) -> TypeSelection | None:
        """Return what type name *spelling*, at *scope_path*, selects now."""
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
        # The paths beneath this module's aliases being read (:meth:`_reading_through`).
        self._through: set[_Through] = set()
        # What the imports a qualifier route names expose -- the root-position
        # tails' under ``None`` -- by the first segment of each exposed path.
        self._exposures: dict[_Route | None, dict[str, list[_Exposed]]] = {}
        # Once the tables the resolver collects are complete (:meth:`_keep_readings`),
        # the contributions at each full path, the own types and whether a
        # ``hiding`` removed a path, by what they are read with.
        self._keeping_readings = False
        self._kept_contributions: dict[tuple[ScopePath, ScopePath, LookupKind], Reading] = {}
        self._kept_own_types: dict[ScopePath, Reading] = {}
        self._kept_hidden: dict[tuple[ScopePath, ScopePath], bool] = {}
        # Alias -> the steps its target is read at here (:meth:`target_steps`).
        self._target_steps: dict[QName, tuple[ScopePath, ...] | None] = {}

    def _declared_type_owners(self) -> dict[ScopePath, TypeOwner]:
        """Return the owner each type this module declares resolves to."""
        return {
            (*path, declaration.name): self._type_owners.declared_owner(
                (self._module_id, _bare_atom((*path, declaration.name))), declaration
            )
            for declaration, path in self._type_declarations
        }

    def read_view(self) -> object:
        """What reads made now see of the uses: reads with equal views see the same of them."""
        return self._uses.reading

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
        """This module's own declarations of *kind* at full *path*."""
        if kind is LookupKind.TYPE:
            return self._kept(self._kept_own_types, path, lambda: self._own_spelled_at(path, kind))
        return self._own_spelled_at(path, kind)

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
            )
            for ref, (layers, hiding) in self._imported(step, path).items()
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
            )
            for qname, decls in qualifier_member_decls(
                self._import_env, spelled, _bare_atom(path), anchored=anchored
            ).items()
        )
        return Reading(tuple(c for c in candidates if self.fits(c.target, kind)))

    def surface_injected(self, chain: QualifierChain, member: str, site: ScopePath) -> Reading:
        """The root enum inline member module qualifier *chain*, in *site*, injects as *member*.

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
            injected = self._route_injected_members(chain, member, site)
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

    def _route_injected_members(
        self, chain: QualifierChain, name: str, site: ScopePath
    ) -> dict[ConstructorRef, str]:
        """Map each root enum inline member one-segment route *chain* injects as *name*.

        A re-exported enum's members are injected too. Each maps to its
        owner-qualified spelling where *chain* is written, in *site*: through
        *chain* when it matches one module, else through a route selecting
        only its exposing module.
        """
        surfaces = qualifier_members(self._import_env, chain.leading_route, anchored=chain.anchored)
        injected: dict[ConstructorRef, str] = {}
        for _module, members in surfaces:
            for atom, origin in members.items():
                path = _bare_path(atom)
                constructor = self._cross_module_constructor_refs.get(origin)
                if (
                    path[1:] == (name,)
                    and constructor is not None
                    and is_root_inline_member(constructor)
                ):
                    injected.setdefault(
                        constructor,
                        render_qualified_name(chain, "::".join(path))
                        if len(surfaces) == 1
                        else self._routed_spelling(constructor, origin, chain.span, site),
                    )
        return injected

    def _spelling_selects(
        self,
        qualifier: tuple[str, ...],
        member: str,
        candidate: ConstructorRef,
        span: SourceSpan,
        site: ScopePath,
        *,
        anchored: bool = False,
    ) -> bool:
        """Whether ``qualifier::member``, written at *span* in *site*, selects *candidate*."""
        found = lookup_qualified(
            self,
            self._probe_chain(qualifier, member, span, anchored=anchored),
            member,
            site,
            LookupKind.VALUE,
            span=span,
        )
        return isinstance(found, QualifiedTarget) and found.constructor == candidate

    def _routed_spelling(
        self, candidate: ConstructorRef, origin: QName, span: SourceSpan, site: ScopePath
    ) -> str:
        """Spell *candidate*, imported as *origin*, by its shortest route selecting it at *span*.

        In *site*.

        Each import route exposing *origin* is tried, anchored ones included;
        without one, *candidate* is spelled by its declaration path.
        """
        spellings = (
            render_route_member(route, written, anchored=anchored)
            for contribution in self._import_env.contributions.values()
            for atom, qname in contribution.members.items()
            if qname == origin
            for written in (_bare_path(atom),)
            for route, anchored in contribution_routes(contribution)
            if self._spelling_selects(
                ("/".join(route), *written[:-1]),
                written[-1],
                candidate,
                span,
                site,
                anchored=anchored,
            )
        )
        return min(spellings, key=len, default=None) or spell_declaration(
            origin[0], _bare_path(origin[1]), reader=self.reader()
        )

    def projected(
        self,
        owner: DeclarationKey,
        layer: ContributionLayer,
        rest: ScopePath,
        chain: QualifierChain,
        kind: LookupKind,
        *,
        owners_within: int,
        written: ScopePath,
    ) -> Reading:
        """What type *owner*, made visible by *layer*, selects for *rest* by its own member table.

        Each name of *rest* but the last must name a type declared beneath
        the one before; one the owner so far only references or hides is
        refused. The owner reached last decides the member: a referenced or
        hidden member is refused, and a type it declares (an inline enum
        member or a nested type) is not selected: only a contribution reaching
        its full path selects it, so a ``hiding`` removes exactly that path. A
        path beneath an alias is its target's as written, read where the alias
        is declared (:meth:`_beneath_alias`), or as the anchor or module route
        that reached the owner, *written*, reads its paths there
        (:meth:`_reach`); the alias's projection of its target's member, or a
        record's own spelling, stands for what that reaches of it
        (:meth:`_standing_for`).
        """
        segments = chain.segments
        start = len(segments) + 1 - len(rest)
        current = _key_qname(owner)
        reach = partial(self._reach, current, layer, chain, written)
        for index, name in enumerate(rest, start):
            reached = self._type_owners.owner(current)
            if reached is None:
                return Reading()
            table = reached
            error = self._owner_member_error(table, chain, index, name)
            if error is not None:
                return Reading(refusals=(error,))
            if table.alias is not None:
                through = self._beneath_alias(
                    current,
                    table.alias,
                    rest[index - start :],
                    layer,
                    chain,
                    kind,
                    owners_within - index + start,
                    reach,
                )
                if index < len(segments):
                    return through
                member = (current[0], _bare_atom((*_bare_path(current[1]), name)))
                return self._standing_for(
                    self._selected_constructor(table, member, layer, chain), through
                )
            current = (current[0], _bare_atom((*_bare_path(current[1]), name)))
        if name in table.members or self._type_owners.is_declared(current):
            return Reading()
        return self._selected_constructor(table, current, layer, chain)

    def _standing_for(self, selected: Reading, through: Reading) -> Reading:
        """An alias's own selection *selected*, standing for its target's member table.

        *through* is what its target as written reaches there: what a type's
        member table selects there is the target's, which *selected*
        projects; any declaration there is reached as it is, and decides
        alone when no member table selects beside it.
        """
        declared = tuple(
            candidate for candidate in through.candidates if not self._table_selected(candidate)
        )
        if declared and len(declared) == len(through.candidates):
            return through
        return Reading((*selected.candidates, *declared), through.refusals)

    def _table_selected(self, candidate: Candidate) -> bool:
        """Whether *candidate* is what a type's member table selects: a member or its own name."""
        key = candidate.target.key
        table = None if key is None else self._type_owners.owner((key[0], _bare_atom(key[1])))
        return (
            key is not None
            and table is not None
            and (key[2] in table.members or table.select(key[2], key[1][-1]) is not None)
        )

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

    def _reach(
        self, owner: QName, layer: ContributionLayer, chain: QualifierChain, written: ScopePath
    ) -> _Reach | None:
        """How *chain* reads the paths beneath *owner*, reached as *written*; ``None`` as declared.

        ``::`` reads them in this module's own root; a module route, the
        leading one *chain* spells, or the one *written* starts with when
        *owner* is what that route alone exports there.
        """
        if chain.anchor is QualifierAnchor.CURRENT_MODULE:
            return chain.anchor, ()
        if chain.routed:
            return chain.anchor, chain.segments[:1]
        if (
            layer is ContributionLayer.IMPORTED
            and len(written) > 1
            and owner
            in qualifier_member_decls(self._import_env, written[:1], _bare_atom(written[1:]))
        ):
            return None, (QualifierSegment(written[0], None, chain.span, chain.node_id),)
        return None

    def _beneath_alias(
        self,
        alias: QName,
        declaration: TypeAlias,
        path: ScopePath,
        layer: ContributionLayer,
        chain: QualifierChain,
        kind: LookupKind,
        owners_within: int,
        reach: Callable[[], _Reach | None],
    ) -> Reading:
        """What *path* beneath *alias*, which *declaration* declares, selects as one of *kind*.

        What its target as written, then *path*, reaches where the alias is
        declared (:meth:`read_through`, *owners_within* as there): a
        declaration the alias's module reaches is reached as the alias is
        (*layer*), and a ``hiding`` there removing the path refuses it as
        *chain* spells it. A target not leading with a module route names its
        paths where the alias is declared, and what *reach* tells reads them
        when it tells anything (:meth:`_reached_through`).
        """
        spelling = _target_spelling(declaration)
        if spelling is None:
            return Reading()
        site = self._site_sources(alias[0])
        steps = site.target_steps(alias, spelling)
        reached = None if steps is None else reach()
        if steps is not None and reached is not None:
            read = self._reached_through(
                alias, path, kind, owners_within, reached, steps, spelling, chain.span
            )
            return Reading(read.candidates, self._respelled(read, chain, path))
        read = site.read_through(
            alias, spelling, path, kind, owners_within, chain.span, every_use=site is not self
        )
        refusals = self._respelled(read, chain, path)
        if site is self:
            return Reading(
                tuple(replace(candidate, hiding=NOT_HIDDEN) for candidate in read.candidates),
                refusals,
            )
        return Reading(
            tuple(
                Candidate(
                    self._reached_target(candidate.target),
                    layer,
                    contribution_origin(candidate.origin.declaration, layer),
                )
                for candidate in read.candidates
            ),
            refusals,
        )

    def target_steps(
        self, alias: QName, spelling: NameT | AppliedT
    ) -> tuple[ScopePath, ...] | None:
        """The steps this module's alias *alias* reads its target, *spelling*, at, nearest first.

        ``None`` when *spelling* leads with a module route: its paths are that
        route's, not this module's.
        """
        if alias not in self._target_steps:
            self._target_steps[alias] = self._read_target_steps(alias, spelling)
        return self._target_steps[alias]

    def _read_target_steps(
        self, alias: QName, spelling: NameT | AppliedT
    ) -> tuple[ScopePath, ...] | None:
        """Read :meth:`target_steps`."""
        chain = member_chain(spelling, ())
        if chain.routed or (
            chain.segments
            and qualifier_candidates(self._import_env, (chain.segments[0].name,), anchored=False)
        ):
            return None
        if chain.anchor is QualifierAnchor.CURRENT_MODULE:
            return ((),)
        return lookup_steps(_bare_path(alias[1])[:-1])

    def _reached_through(
        self,
        alias: QName,
        path: ScopePath,
        kind: LookupKind,
        owners_within: int,
        reach: _Reach,
        steps: tuple[ScopePath, ...],
        spelling: NameT | AppliedT,
        span: SourceSpan,
    ) -> Reading:
        """What *reach* reads at alias *alias*'s target, *spelling*, then *path*, nearest first.

        As one of *kind*, *owners_within* as :meth:`read_through` takes it,
        at each of *steps*, those the target is read at where it is declared
        (:meth:`target_steps`). Reaching nothing, the nearest refusal.
        """
        anchor, route = reach
        spelled = replace(member_chain(spelling, path), anchor=anchor, span=span)
        within = len(spelled.segments) + 1 - len(path) + max(owners_within, 0)
        refusals: tuple[AglError, ...] = ()
        with self._reading_through((alias, path, kind, within), every_use=False) as reads:
            for step in steps if reads else ():
                lead = (
                    *route,
                    *(QualifierSegment(name, None, span, spelled.node_id) for name in step),
                )
                read = lookup_through(
                    self,
                    replace(spelled, segments=(*lead, *spelled.segments)),
                    (),
                    kind,
                    owners_within=len(lead) + within,
                )
                if read.candidates:
                    return read
                refusals = refusals or read.refusals
        return Reading(refusals=refusals)

    def _respelled(
        self, read: Reading, chain: QualifierChain, path: ScopePath
    ) -> tuple[AglError, ...]:
        """*read*'s refusals of *path* read for *chain*: a hidden path as *chain* spells it."""
        return tuple(
            self._hidden_beneath(chain, path) if isinstance(refusal, HiddenMemberError) else refusal
            for refusal in read.refusals
        )

    def _reached_target(self, target: QualifiedTarget) -> QualifiedTarget:
        """*target*, which another module reached, with the binding this module reads it through."""
        ref = target.ref
        if ref is None or ref.module_id == self._module_id:
            return target
        return replace(target, ref=self._cross_module_binding_ref(_ref_qname(ref)))

    @contextmanager
    def _reading_through(self, through: _Through, *, every_use: bool) -> Iterator[bool]:
        """Read *through*, a path beneath an alias, here unless already reading it.

        Yields whether to read it: an alias whose target as written leads back
        to it reaches nothing more there. Every use is read when *every_use*,
        as another module reads it; otherwise those the read in progress sees.
        """
        if through in self._through:
            yield False
            return
        self._through.add(through)
        try:
            with self._uses.view(every_use):
                yield True
        finally:
            self._through.discard(through)

    def read_through(
        self,
        alias: QName,
        spelling: NameT | AppliedT,
        path: ScopePath,
        kind: LookupKind,
        owners_within: int,
        span: SourceSpan,
        *,
        every_use: bool,
    ) -> Reading:
        """What *spelling*, alias *alias*'s target as written, then *path*, reaches here.

        Read in the alias's region, at *span*, as a declaration of *kind*
        (:func:`lookup_through`): only a type the target or *owners_within*
        more names reach, or an alias, projects its member table. Every use is
        read when *every_use* (:meth:`_reading_through`).
        """
        chain = replace(member_chain(spelling, path), span=span)
        within = len(chain.segments) + 1 - len(path) + max(owners_within, 0)
        with self._reading_through((alias, path, kind, within), every_use=every_use) as reads:
            if not reads:
                return Reading()
            return lookup_through(
                self, chain, _bare_path(alias[1])[:-1], kind, owners_within=within
            )

    def origins_through(
        self, alias: QName, spelling: NameT | AppliedT, path: ScopePath, *, every_use: bool
    ) -> frozenset[QName]:
        """The scopes and types *spelling*, alias *alias*'s target as written, then *path*, names.

        Read in the alias's region, as :meth:`read_through` reads it.
        """
        with self._reading_through((alias, path, None, 0), every_use=every_use) as reads:
            if not reads:
                return frozenset()
            return lookup_origins(self, member_chain(spelling, path), _bare_path(alias[1])[:-1])

    def exported_through(
        self,
        alias: QName,
        path: ScopePath,
        exports: Mapping[NameAtom, QName],
        declared: Callable[[QName], Collection[ScopePath]],
    ) -> dict[ScopePath, QName]:
        """The declarations *path* beneath this module's alias *alias* reaches, and beneath them.

        Each keyed by its path relative to *path*: what *exports*, those of
        the module exporting the alias, hold there (:meth:`exported_beneath`);
        for a target leading with a module route, what it as written, then
        that path, reaches where the alias is declared, read with every use,
        the first declaration of any kind there, *declared* telling the paths
        declared beneath a scope or type the target then *path* names.
        """
        spelling = self.alias_spelling(alias)
        if spelling is None:
            return {}
        exported = self.exported_beneath(alias, spelling, path, exports)
        if exported is not None:
            return exported
        span = self._all_public_types[alias].span
        relatives = {
            (): None,
            **{
                relative: None
                for origin in self.origins_through(alias, spelling, path, every_use=True)
                for relative in declared(origin)
            },
        }
        reached: dict[ScopePath, QName] = {}
        for relative in relatives:
            beneath = (*path, *relative)
            for kind in LookupKind:
                read = self.read_through(
                    alias, spelling, beneath, kind, len(beneath), span, every_use=True
                )
                for candidate in read.candidates:
                    reached.setdefault(relative, candidate.origin.declaration)
        return reached

    def exported_beneath(
        self,
        alias: QName,
        spelling: NameT | AppliedT,
        path: ScopePath,
        exports: Mapping[NameAtom, QName],
    ) -> dict[ScopePath, QName] | None:
        """What *exports* hold at and beneath this module's alias *alias*'s target, then *path*.

        *spelling* is the target as written. Keyed by their paths relative to
        that, at the nearest step holding any (:meth:`target_steps`);
        ``None`` when the target leads with a module route.
        """
        steps = self.target_steps(alias, spelling)
        if steps is None:
            return None
        target = member_chain(spelling, ())
        names = (*(segment.name for segment in target.segments), target.member, *path)
        for step in steps:
            prefix = (*step, *names)
            found = {
                written[len(prefix) :]: origin
                for atom, origin in exports.items()
                if (written := _bare_path(atom))[: len(prefix)] == prefix
            }
            if found:
                return found
        return {}

    def projected_origins(self, alias: DeclarationKey, rest: ScopePath) -> frozenset[QName]:
        """See :meth:`~agm.agl.scope.lookup.PathSources.projected_origins`."""
        qname = _key_qname(alias)
        spelling = self.alias_spelling(qname)
        if spelling is None:
            return frozenset()
        site = self._site_sources(qname[0])
        return site.origins_through(qname, spelling, rest, every_use=site is not self)

    def alias_spelling(self, qname: QName) -> NameT | AppliedT | None:
        """The target of type *qname* as written, when it is an alias naming one."""
        table = self._type_owners.owner(qname)
        return None if table is None or table.alias is None else _target_spelling(table.alias)

    def reached_beneath(self, alias: QName) -> frozenset[QName]:
        """The scopes and types a path beneath type *alias* is read in.

        Those its target as written names where it is declared, read with
        every use, and those beneath each alias among them; none for no alias.
        """
        found: set[QName] = set()
        pending = [alias]
        while pending:
            current = pending.pop()
            spelling = self.alias_spelling(current)
            if spelling is not None:
                site = self._site_sources(current[0])
                reached = site.origins_through(current, spelling, (), every_use=True) - found
                found |= reached
                pending.extend(reached)
        return frozenset(found)

    def removed_with(self, qname: QName) -> frozenset[DeclarationKey]:
        """What a ``hiding`` naming *qname* removes, by identity.

        The declaration itself, and the scopes and types a path beneath it is
        read in (:meth:`reached_beneath`).
        """
        return frozenset(
            self.identity(_qname_decl_key(reached))
            for reached in (qname, *self.reached_beneath(qname))
        )

    @staticmethod
    def _hidden_beneath(chain: QualifierChain, path: ScopePath) -> HiddenMemberError:
        """The refusal of *path*, which *chain* spells beneath an alias, as hidden."""
        return HiddenMemberError(render_qualifier_path(chain), path[-1], span=chain.span)

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
        applied: Application,
        rest: ScopePath,
        chain: QualifierChain,
        kind: LookupKind,
        *,
        site: ScopePath,
        owners_within: int,
    ) -> Reading:
        """What *applied*, an applied segment of *chain* written in *site*, selects for *rest*.

        See :meth:`~agm.agl.scope.lookup.PathSources.beneath_applied`; read as
        an alias's target is (:meth:`read_through`).
        """
        spelled = replace(member_chain(applied.spelling, rest), span=chain.span)
        within = len(spelled.segments) + 1 - len(rest) + max(owners_within, 0)
        read = lookup_through(self, spelled, site, kind, owners_within=within)
        return Reading(read.candidates, self._respelled(read, chain, rest))

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
            application = Application(_qname_decl_key(projected), arity, argument)
        return application

    def applies(self, key: DeclarationKey) -> bool:
        """Whether type *key* is an alias applying its target to type arguments of its own."""
        owners = self._type_owners
        reached = owners.owner(owners.identity(_key_qname(key)))
        return reached is not None and reached.applies

    def hidden_at(self, step: ScopePath, path: ScopePath) -> bool:
        """Whether a ``hiding`` of a contribution anchored at or above *step* removed *path*."""
        return self._kept(self._kept_hidden, (step, path), lambda: self._hidden_at(step, path))

    def _hidden_at(self, step: ScopePath, path: ScopePath) -> bool:
        """Read :meth:`hidden_at`."""
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
            if step[: len(anchor)] == anchor and _bare_atom(relative) in env.decl_hidden.get(
                node_id, ()
            ):
                return True
        return (
            len(path) > 1 and self._route_hides((path[0],), path[1:], anchored=False)
        ) or self._withheld(self._exposed(None), path)

    def _exposed(self, route: _Route | None) -> dict[str, list[_Exposed]]:
        """What the imports *route* names expose, by first segment; root tails' at ``None``."""
        found = self._exposures.get(route)
        if found is None:
            env = self._import_env
            exposed = (
                unqualified_exposures(env)
                if route is None
                else qualifier_exposures(env, route[0], anchored=route[1])
            )
            found = self._exposures[route] = {}
            for atom, qname, decls in exposed:
                exposed_path = _bare_path(atom)
                found.setdefault(exposed_path[0], []).append((exposed_path, qname, decls))
        return found

    def _withheld(self, exposed: dict[str, list[_Exposed]], path: ScopePath) -> bool:
        """Whether the imports exposing the nearest qualifier of *path* remove what it names.

        *exposed* is what they expose (:meth:`_exposed`). Their own ``hiding`` or the imported
        module's export ``hiding`` removed the declaration the rest of *path*
        names beneath the longest prefix of *path* they expose: every import
        exposing that prefix itself, else any exposing a path beneath it,
        which none reaches the declaration by.
        """
        exposures = exposed.get(path[0], ())
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

    def routed_hidden(self, chain: QualifierChain, path: ScopePath) -> bool:
        """Whether a ``hiding`` removed *path* from *chain*'s leading module route."""
        return self._route_hides(chain.leading_route, path, anchored=chain.anchored)

    def _route_hides(self, route: tuple[str, ...], path: ScopePath, *, anchored: bool) -> bool:
        """Whether a ``hiding`` removed *path* from module *route*: it, or an alias above it."""
        env = self._import_env
        return qualifier_hides(env, route, _bare_atom(path), anchored=anchored) or self._withheld(
            self._exposed((route, anchored)), path
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

    def _named(self, key: DeclarationKey) -> QName:
        """The full path *key* names: the type's, for the constructor a type selects by its name."""
        module_id, path, name = key
        if path:
            owner = (module_id, _bare_atom(path))
            table = self._type_owners.owner(owner)
            if (
                table is not None
                and name not in table.members
                and table.select(name, path[-1]) is not None
            ):
                return owner
        return _key_qname(key)

    def identity(self, key: DeclarationKey) -> DeclarationKey:
        """The declaration *key* names: a renaming alias's is its target's, a member its own."""
        module_id, path, name = key
        table = self._type_owners.owner((module_id, _bare_atom(path))) if path else None
        member = None if table is None else table.members.get(name)
        if member is not None:
            return (member.owner_module_id, member.owner_path, member.owner_name)
        return _qname_decl_key(self._type_owners.identity(self._named(key)))

    def denotes(self, key: DeclarationKey) -> object:
        """What *key* names in an ambiguity: its identity, or what an alias denotes there."""
        denoted = self._type_owners.denotation(self._named(key))
        return self.identity(key) if denoted is None else denoted

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
    ) -> dict[BindingRef, tuple[Layers, Hiding]]:
        """Return what import tails anchored at or above *step* bind at full *path*.

        Every layer from *step* outward contributes the path relative to its
        own; the module root's import tails and the module route spelled by
        its leading name contribute it whole. A binding several contribute
        keeps every one's tag, and what the ``hiding`` of each declaration
        contributing it removes.
        """
        env = self._import_env
        reached: dict[BindingRef, tuple[Layers, set[frozenset[DeclarationKey]]]] = {}

        def add(ref: BindingRef, layers: Layers, decls: Iterable[int], entry: QName) -> None:
            found, ways = reached.setdefault(ref, (frozenset(), set()))
            reached[ref] = found | layers, ways
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
                    for key in self._named_beneath(named, rest):
                        add(
                            self._cross_module_binding_ref(_key_qname(key)),
                            frozenset({ContributionLayer.IMPORTED}),
                            (node_id,),
                            named.declaration,
                        )
        imported = dict(env.unqualified_decls.get(_bare_atom(path), {}))
        if path[1:]:
            for qname, decls in qualifier_member_decls(
                env, (path[0],), _bare_atom(path[1:])
            ).items():
                imported[qname] = imported.get(qname, frozenset()) | decls
        for qname, decls in imported.items():
            add(
                self._cross_module_binding_ref(qname),
                frozenset({ContributionLayer.IMPORTED}),
                decls,
                qname,
            )
        return {ref: (layers, frozenset(ways)) for ref, (layers, ways) in reached.items()}

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

        An alias names too what a path beneath it reaches
        (:meth:`removed_with`). A path it names beneath an exported alias
        must name a declaration there (:meth:`_named_beneath`).
        """
        if not named.beneath:
            return self.removed_with(named.declaration)
        keys = self._named_beneath(named, ())
        if not keys:
            raise UnknownMemberError(
                spell_declaration(named.module, named.item),
                span=named.span,
                repair=MissRepair.NOT_EXPORTED,
            )
        return keys

    def _named_beneath(self, named: ItemDeclaration, rest: ScopePath) -> frozenset[DeclarationKey]:
        """The declarations of any kind import item *named*, then *rest*, names beneath its alias.

        What its module exports there (:meth:`exported_beneath`); for a
        target leading with a module route, what the alias reads there
        (:meth:`projected`): its target as written where it is declared.
        """
        alias, beneath = named.declaration, (*named.beneath, *rest)
        site = self._site_sources(alias[0])
        spelling = site.alias_spelling(alias)
        if spelling is None:
            return frozenset()
        exported = site.exported_beneath(
            alias, spelling, beneath, self._import_env.contributions[named.module].exports
        )
        if exported is not None:
            origin = exported.get(())
            return frozenset(() if origin is None else (self.identity(_qname_decl_key(origin)),))
        written = (*named.item, *rest)
        chain = self._probe_chain(written[:-1], written[-1], named.span, anchored=False)
        return frozenset(
            self.identity(key)
            for kind in LookupKind
            for candidate in self.projected(
                _qname_decl_key(alias),
                ContributionLayer.IMPORTED,
                beneath,
                chain,
                kind,
                owners_within=len(beneath),
                written=(),
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
            for ref, (layers, hiding) in self._imported(step, path).items()
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
                for ref, (_layers, hiding) in self._imported(step, (*step, key[2])).items()
            ):
                yield member


class ResolvedSources(ModuleSources):
    """What a module an earlier compilation resolved reads, by full path (:class:`ModuleSources`).

    Read from its retained *resolved* tables and import environment, over
    this compilation's program tables, as where its aliases are declared.
    """

    def __init__(
        self,
        module_id: ModuleId,
        resolved: ModuleResolution,
        import_env: ImportEnv,
        *,
        all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
        type_owners: TypeOwnerIndex,
        decl_info: dict[QName, DeclInfo],
        cross_module_constructor_refs: Mapping[QName, ConstructorRef],
        site_sources: Callable[[ModuleId], ModuleSources],
    ) -> None:
        super().__init__()
        self._module_id = module_id
        self._program = resolved.program
        self._import_env = import_env
        self._all_public_types = all_public_types
        self._type_owners = type_owners
        self._decl_info = decl_info
        self._cross_module_constructor_refs = cross_module_constructor_refs
        self._repl_session_type_paths = {}
        self._repl_session_root_type_names = frozenset()
        self._root_scope = resolved.root_scope
        self._scope_nodes = resolved.scope_nodes
        self._declarations = resolved.declarations
        self._scope_entity_kinds = resolved.scope_entity_kinds
        self._import_decl_scope_paths = resolved.import_decl_scope_paths
        self._type_declarations = resolved.type_declarations
        self._scoped_constructor_candidates = resolved.scoped_constructor_candidates
        self._constructor_candidates = resolved.constructor_candidates
        self._injected_constructors = resolved.injected_constructors
        self._site_sources = site_sources
        self._owner_declarations = resolved.owner_declarations
        self._uses = UseReader(
            self, module_id, self._scope_nodes, self._scope_entity_kinds, type_owners
        )
        self._keep_readings()

    def type_name_selection_at(
        self, scope_path: ScopePath, spelling: NameT | AppliedT, *, every_use: bool
    ) -> TypeSelection | None:
        """What *spelling* selected when the module was resolved: what it selects now."""
        return self._owner_declarations.get(selection_node_id(spelling))
