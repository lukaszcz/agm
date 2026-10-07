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
from enum import Enum, auto
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
from agm.agl.scope.hiding import (
    NOT_HIDDEN,
    Hiding,
    Origin,
    beneath_hiding,
    hidden_keys,
    removed,
    removes,
    unremoved,
)
from agm.agl.scope.imports import (
    BareRoute,
    ImportEnv,
    ImportWay,
    ItemDeclaration,
    NameAtom,
    QName,
    ScopeOrigins,
    contribution_routes,
    qualifier_candidates,
    qualifier_exposures,
    qualifier_member_ways,
    qualifier_members,
    qualifier_scope_paths,
    unqualified_exposures,
)
from agm.agl.scope.lookup import (
    Application,
    Candidate,
    LookupKind,
    Own,
    PathReader,
    QualifiedTarget,
    Reading,
    Route,
    Via,
    hidden_member,
    lookup_bare,
    lookup_constructors,
    lookup_qualified,
    lookup_reached,
    lookup_reached_origins,
    lookup_through,
    read_steps,
    shadowed_by_type_parameter,
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
    OwnerMemberSelection,
    ScopeNode,
    ScopePath,
    TypeOwner,
    TypeSelection,
    UnknownMemberError,
    add_layers,
    anchored_layers,
    contribution_origin,
    contribution_origins,
    layered,
    relative_under,
)
from agm.agl.scope.symbols import binding_qname as _ref_qname
from agm.agl.scope.symbols import declaration_qname as _key_qname
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
from agm.agl.scope.type_owners import TypeOwnerIndex, is_current, root_type_names
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

type _Spelling = tuple[tuple[str, ...], str, QualifierAnchor | None]
"""A qualified spelling: its qualifier segments, its member and its anchor."""

type _Exposed = tuple[ScopePath, QName, frozenset[ImportWay]]
"""A path imports expose, what it reaches there, and the ways exposing it."""


class _Read(Enum):
    """A read beneath an alias's target as written that is no member lookup of a kind."""

    ORIGINS = auto()
    """The scopes and types the path names (:meth:`ModuleSources.origins_through`)."""
    HEAD = auto()
    """The target head's own reading (:meth:`ModuleSources.leads_with_route`)."""


type _Through = tuple[QName, LookupKind | _Read, Via | None]
"""A read beneath an alias's target as written: of a kind or a :class:`_Read`, via a route."""


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


def render_spelling(qualifier: tuple[str, ...], member: str, anchor: QualifierAnchor | None) -> str:
    """Render ``qualifier::member`` as written under *anchor*."""
    path = "::".join((*qualifier, member))
    if anchor is None:
        return path
    return f"/{path}" if anchor is QualifierAnchor.MODULE else f"::{path}"


def _spelling_cost(spelling: _Spelling) -> tuple[int, int]:
    """Order spellings by segments, then characters."""
    return len(spelling[0]) + 1, len(render_spelling(*spelling))


def _way_order(way: ImportWay) -> int:
    """Sort key: ways in the order of their import declarations."""
    return way.node_id


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
    _uses: UseReader
    # What each module of the program reads where its aliases are declared.
    _site_sources: Callable[[ModuleId], ModuleSources]

    def type_name_selection_at(
        self, scope_path: ScopePath, spelling: NameT | AppliedT, *, every_use: bool
    ) -> TypeSelection | None:
        """Return what type name *spelling*, at *scope_path*, selects now."""
        ...


def _owner_path(selection: TypeSelection) -> QName | None:
    """The owner *selection* names the constructor of, as the type it stands for; else ``None``.

    The owner is where no inline member of it is declared: an alias's own
    constructor, spelled through the alias (``type A = A::B`` in a scope
    declaring it, spelled ``A::B`` there).
    """
    return _key_qname(selection.owner) if isinstance(selection, OwnerMemberSelection) else None


def _within(chain: QualifierChain, path: ScopePath, owners_within: int) -> int:
    """Return *owners_within* re-based from *path*, a tail of *chain*'s full path, to all of it."""
    return len(chain.segments) + 1 - len(path) + max(owners_within, 0)


class ModuleSources(SourcesHost):
    """The one lookup's reads for one module, by full path (:class:`PathSources`)."""

    def __init__(self) -> None:
        # This module as diagnostics spell declarations for it (:meth:`reader`).
        self._reader: Reader | None = None
        # Every enum this module reads by the names of its members, built on
        # first use.
        self._enum_member_index: dict[str, dict[QName, ConstructorRef]] | None = None
        # Each of those enums, by the names that can reach it (:meth:`_spellings_of`), built on
        # first use.
        self._spellings: dict[QName, tuple[str, ...]] | None = None
        # Every path this module reads a type by (:meth:`_bindings`), built on first use.
        self._name_bindings: tuple[list[tuple[ScopePath, QName]], list[str]] | None = None
        # (region path, exposed atom, exposed declaration) -> the ways import declarations of the
        # region reach it by; built on first use (:meth:`_region_import_ways`).
        self._region_ways: dict[tuple[ScopePath | None, NameAtom, QName], set[ImportWay]] | None = (
            None
        )
        # Import declaration id -> the declarations its ``hiding`` removes, by identity.
        self._hidden_by: dict[int, frozenset[DeclarationKey]] = {}
        # What an export ``hiding`` withholds -> the declarations that remove, by identity.
        self._withheld_keys: dict[frozenset[QName], frozenset[DeclarationKey]] = {}
        # The aliases of this module whose targets are being read beneath, by what is read and
        # the route it is reached through, with the paths of the reads in progress
        # (:meth:`_reading_through`).
        self._through: dict[_Through, list[ScopePath]] = {}
        # What the imports a qualifier route names expose -- the root-position
        # tails' under ``None`` -- by the first segment of each exposed path.
        self._exposures: dict[Route | None, dict[str, list[_Exposed]]] = {}
        # Once the tables the resolver collects are complete (:meth:`_keep_readings`),
        # the contributions at each full path and the own types, by what they are read with.
        self._keeping_readings = False
        self._kept_contributions: dict[tuple[ScopePath, ScopePath, LookupKind], Reading] = {}
        self._kept_own_types: dict[ScopePath, Reading] = {}

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
        return self._enum_members().get(name, {})

    def _enum_members(self) -> dict[str, dict[QName, ConstructorRef]]:
        """Every enum this module reads, by the names of its members; built on first use."""
        if self._enum_member_index is None:
            self._enum_member_index = self._build_enum_member_index()
        return self._enum_member_index

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
                *(
                    (injected.owner_name, injected)
                    for injected in owner.injected
                    if is_current(self._module_id, self._owner_at, injected)
                ),
            ):
                index.setdefault(member_name, {}).setdefault(qname, constructor)
        return index

    def _owner_at(self, path: ScopePath) -> TypeOwner | None:
        """This module's current type owner at *path*."""
        return self._type_owners.owner((self._module_id, _bare_atom(path)))

    def _spellings_of(self, enum: QName) -> tuple[str, ...]:
        """The names that can reach *enum* here: its own, each alias's and each rename's."""
        if self._spellings is None:
            self._spellings = self._build_spellings()
        return self._spellings.get(enum, ())

    def _build_spellings(self) -> dict[QName, tuple[str, ...]]:
        """Index each enum :meth:`enum_members_named` reads by the names that can reach it.

        Its own name, and every other name this module reads for it
        (:meth:`_bindings`), in declaration order, and any ``use`` rename (which
        may rename whatever it exposes).
        """
        enums = {qname for by_enum in self._enum_members().values() for qname in by_enum}
        names: dict[QName, dict[str, None]] = {
            qname: {_bare_path(qname[1])[-1]: None} for qname in enums
        }
        bindings, renames = self._bindings()
        for path, qname in bindings:
            enum = self._type_owners.enum_behind(qname)
            if len(path) == 1 and enum in names:
                names[enum][path[0]] = None
        return {
            qname: (*found, *(r for r in renames if r not in found))
            for qname, found in names.items()
        }

    def _bindings(self) -> tuple[list[tuple[ScopePath, QName]], list[str]]:
        """Every path this module reads a type or declaration by, with what it reaches there.

        An alias declared anywhere, an import item's or route surface's name, a
        contribution of any scope; in declaration order. Also the names ``use``
        declarations rename to, which may rename whatever they expose.
        """
        if self._name_bindings is not None:
            return self._name_bindings
        bindings: list[tuple[ScopePath, QName]] = []
        for qname, declaration in self._all_public_types.items():
            if isinstance(declaration, TypeAlias):
                bindings.append(((declaration.name,), qname))
        for item, path in self._type_declarations:
            if isinstance(item, TypeAlias):
                qname = (self._module_id, _bare_atom((*path, item.name)))
                bindings.append(((item.name,), qname))
                if path:
                    bindings.append(((*path, item.name), qname))
        for path in self._repl_session_type_paths:
            bindings.append(((path[-1],), (self._module_id, _bare_atom(path))))
        env = self._import_env
        for atom, qnames in env.unqualified.items():
            bindings.extend((_bare_path(atom), qname) for qname in qnames)
        for contribution in env.contributions.values():
            for surface in contribution.routes.values():
                bindings.extend(
                    (_bare_path(atom), qname) for atom, qname in surface.members.items()
                )
        renames: dict[str, None] = {}
        for layer in (self._root_scope, *self._scope_nodes.values()):
            for atom, refs in layer.bare_contributions.items():
                bindings.extend((_bare_path(atom), _ref_qname(ref)) for ref in refs)
            for decl in layer.uses:
                if decl.alias is not None:
                    renames[decl.alias] = None
                for imported in decl.tail or ():
                    if imported.rename is not None:
                        renames[imported.rename] = None
        self._name_bindings = bindings, list(renames)
        return self._name_bindings

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
        repair: str | None,
        span: SourceSpan,
    ) -> AmbiguousConstructorError:
        """Report *spelling* as ambiguous among *candidates*, each from every contributing layer.

        *repair* selects the first candidate, if any spelling does.
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
        ref = self._own_level_value(tuple(self._root_scope.enclosing()), name)
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
                via,
            )
            for ref, (layers, hiding, via) in self._imported(step, path).items()
            for layer in layered(layers)
        )
        used = (candidate for candidate, _decl in self._use_exposures(step, path, kind))
        return Reading(
            (*(candidate for candidate in imported if self.fits(candidate.target, kind)), *used)
        )

    def injected(
        self,
        reached: PathReader,
        step: ScopePath,
        name: str,
        *,
        via: Via | None = None,
    ) -> Reading:
        """The enum members injected as bare *name* at *step*, by the types *reached* reads.

        Each enum with a member *name* injects it once a spelling of the enum
        reaches it at ``(*step, spelling)``: an own alias of it or a rename
        included, an alias applying it as the enum itself. The member is its
        own declaration -- an applied alias's arguments are inferred -- reached
        the ways the type was, and in the layer of the module declaring it.
        Through a module qualifier (*via*) only the inline members of enums
        declared at a module's root inject, and ``::`` falls back on a
        built-in enum's when none does.
        """
        found: list[Candidate] = []
        for enum, member in self.enum_members_named(name).items():
            if via is not None and not is_root_inline_member(member):
                continue
            for spelling in self._spellings_of(enum):
                for reaching in reached((*step, spelling), LookupKind.TYPE).candidates:
                    candidate = self._injected(reaching, enum, member, name)
                    if candidate is not None:
                        found.append(candidate)
        if isinstance(via, Own) and not found:
            found = [
                self._constructor_candidate(member, name, NOT_HIDDEN, via)
                for member in self.enum_members_named(name).values()
                if member.is_builtin and is_root_inline_member(member)
            ]
        elif via is None and not step:
            found.extend(self._seeded_constructors(name, found))
        return Reading(tuple(found))

    def _seeded_constructors(self, name: str, found: Sequence[Candidate]) -> Iterator[Candidate]:
        """Yield the host-seeded constructors bare *name* reaches at the root.

        A standard-library declaration of the same built-in enum, among
        *found*, displaces its seeded one.
        """
        declared = {
            (c.target.constructor.owner_path)
            for c in found
            if c.target.constructor is not None and c.target.constructor.is_builtin
        }
        for constructor in self._constructor_candidates.get(name, ()):
            if constructor.owner_module_id != self._module_id and not (
                constructor.is_builtin and constructor.owner_path in declared
            ):
                yield self._constructor_candidate(constructor, name, NOT_HIDDEN, None)

    def _injected(
        self, reaching: Candidate, enum: QName, member: ConstructorRef, name: str
    ) -> Candidate | None:
        """The candidate for *member*, *name*d bare, when type *reaching* reaches *enum*'s.

        ``None`` unless the type is *enum*, or an alias that leads to it and
        does not hide the member.
        """
        key = reaching.target.key
        owners = self._type_owners
        owner = None if key is None else owners.owner(_key_qname(key))
        if (
            key is None
            or owner is None
            or owners.enum_behind(_key_qname(key)) != enum
            or (owner.alias is not None and (name,) in owner.hidden)
        ):
            return None
        hiding = beneath_hiding(
            reaching.hiding, key, NOT_HIDDEN, _qname_decl_key(member.qname), self
        )
        return self._constructor_candidate(member, name, hiding, reaching.via, reaching.layer)

    def _constructor_candidate(
        self,
        member: ConstructorRef,
        name: str,
        hiding: Hiding,
        via: Via | None,
        reached: ContributionLayer = ContributionLayer.DECLARED,
    ) -> Candidate:
        """The candidate for *member*, spelled bare *name*, with its type's ways' *hiding*.

        It lies in the layer its type was reached in; one reached as this module's
        own is the member's module's, so an own alias of an imported enum reaches
        it as the imported member's.
        """
        layer = (
            ContributionLayer.IMPORTED
            if reached is ContributionLayer.DECLARED and member.owner_module_id != self._module_id
            else reached
        )
        return Candidate(
            QualifiedTarget(
                _qname_decl_key(member.qname), constructor_binding(name, member), member
            ),
            layer,
            contribution_origin(member.qname, layer),
            hiding,
            via,
            member.owner_module_id == self._module_id,
        )

    def routed_at(self, route: Route, path: ScopePath, kind: LookupKind) -> Reading:
        """What module *route* alone reaches at *path* beneath it."""
        candidates = (
            Candidate(
                self._contributed_target(self._cross_module_binding_ref(qname), ()),
                ContributionLayer.IMPORTED,
                ImportedModuleOrigin(qname),
                self._hiding(ways),
            )
            for qname, ways in qualifier_member_ways(
                self._import_env, route.route, _bare_atom(path), anchored=route.anchored
            ).items()
        )
        return Reading(tuple(c for c in candidates if self.fits(c.target, kind)))

    def referenced_refusal(self, chain: QualifierChain, member: str) -> Reading:
        """The refusal of *member* beneath module qualifier *chain* when a root enum references it.

        A member keeps its own path: a name only a root enum references,
        under ``::`` or the route *chain* leads with, is refused.
        """
        roots: Iterable[tuple[str, QName]]
        if chain.segments:
            roots = (
                (atom, origin)
                for _module, members in qualifier_members(
                    self._import_env, chain.leading_route, anchored=chain.anchored
                )
                for atom, origin in members.items()
                if isinstance(atom, str)
            )
        else:
            # Earlier REPL entries' root types, then this entry's.
            roots = (
                (root, (self._module_id, root))
                for root in (
                    *self._repl_session_root_type_names,
                    *(item.name for item, path in self._type_declarations if not path),
                )
            )
        referenced = (
            ReferencedMemberError(render_qualified_name(chain, root), member, span=chain.span)
            for root, qname in roots
            if (owner := self._type_owners.owner(qname)) is not None
            and owner.constructor is None
            and owner.alias is None
            and member in owner.referenced
        )
        return Reading(refusals=tuple(itertools.islice(referenced, 1)))

    def constructor_declaration(self, constructor: ConstructorRef) -> DeclarationKey:
        """The declaration *constructor* constructs: a renaming alias's is its target's.

        A member an alias of an enum selects is that member of the enum behind it.
        Another alias is a declaration of its own here, though aliases denoting
        one type construct one (:meth:`one_per_declaration`).
        """
        named = self._type_owners.constructor_identity(constructor)
        if named.member is None:
            return named.key
        enum = self._type_owners.enum_behind(named.qname) or named.qname
        return enum[0], _bare_path(enum[1]), named.member

    def pattern_constructors(self, name: str, site: ScopePath) -> tuple[ConstructorRef, ...]:
        """Return the constructor candidates a bare pattern or ``is`` spelling *name* reaches.

        Its scrutinee selects among them: every constructor so spelled at
        every step of *site* -- own, contributed and injected -- that no
        ``hiding`` removed. A same-named record or exception does not make a
        member yield here.
        """
        found: dict[ConstructorRef, Layers] = {}
        for candidate in lookup_constructors(self, name, site):
            constructor = candidate.target.constructor
            if constructor is not None and not removed(
                candidate.hiding, candidate.target.key, self
            ):
                add_layers(found, constructor, (candidate.layer,))
        return tuple(self.one_per_declaration(found))

    def one_per_declaration(
        self, candidates: Mapping[ConstructorRef, Layers]
    ) -> dict[ConstructorRef, Layers]:
        """*candidates*, one per declaration they construct, with every layer reaching it.

        A renaming alias's constructor is its target's
        (:meth:`TypeOwnerIndex.constructor_identity`), and aliases denoting one
        type construct one, as do the members aliases applying one enum alike
        select (:meth:`TypeOwnerIndex.denotation`): the candidate
        naming the declaration directly stands for it, else its first by path.
        """
        grouped: dict[object, dict[ConstructorRef, Layers]] = {}
        for candidate, layers in candidates.items():
            named = self._type_owners.constructor_identity(candidate)
            denoted = self._type_owners.denotation(named.qname)
            grouped.setdefault(named if denoted is None else denoted, {})[candidate] = layers
        return {
            (
                named if named in reached else min(reached, key=constructor_candidate_sort_key)
            ): frozenset().union(*reached.values())
            for named, reached in grouped.items()
        }

    def selects_constructor(
        self,
        qualifier: tuple[str, ...],
        member: str,
        decl: DeclarationKey,
        span: SourceSpan,
        site: ScopePath,
        *,
        anchor: QualifierAnchor | None = None,
        kind: LookupKind = LookupKind.CONSTRUCTOR,
    ) -> bool:
        """Whether ``qualifier::member``, written at *span* in *site* as *kind*, selects *decl*."""
        found = lookup_qualified(
            self,
            self._probe_chain(qualifier, member, span, anchor=anchor),
            site,
            kind,
            span=span,
        )
        return (
            isinstance(found, QualifiedTarget)
            and found.constructor is not None
            and self.constructor_declaration(found.constructor) == decl
        )

    def spell_constructor(
        self,
        decl: DeclarationKey,
        site: ScopePath,
        span: SourceSpan,
        *,
        type_params: Collection[str] = (),
        by_scrutinee: bool = False,
        kind: LookupKind = LookupKind.CONSTRUCTOR,
    ) -> str | None:
        """Spell constructor *decl* by the shortest spelling that selects it where written.

        Written at *span* in *site* in a *kind* position, with *type_params* in scope there, which
        shadow a spelling's leading segment. Tried by fewest segments, then
        fewest characters, then in the order :meth:`_spellings_reaching` gives;
        the first the real lookup there selects *decl* by wins. Where *by_scrutinee*,
        a bare spelling is a pattern whose scrutinee selects among its candidates.
        ``None`` when no spelling selects it.
        """
        spellings = sorted(self._spellings_reaching(decl), key=_spelling_cost)
        for qualifier, member, anchor in spellings:
            if not qualifier and anchor is None:
                selects = self._selects_bare(member, decl, span, site, by_scrutinee, kind)
            else:
                selects = not shadowed_by_type_parameter(
                    anchor, qualifier, type_params
                ) and self.selects_constructor(
                    qualifier, member, decl, span, site, anchor=anchor, kind=kind
                )
            if selects:
                return render_spelling(qualifier, member, anchor)
        return None

    def _selects_bare(
        self,
        name: str,
        decl: DeclarationKey,
        span: SourceSpan,
        site: ScopePath,
        by_scrutinee: bool,
        kind: LookupKind,
    ) -> bool:
        """Whether bare *name* written at *span* in *site* selects constructor *decl*."""
        if by_scrutinee:
            return any(
                self.constructor_declaration(candidate) == decl
                for candidate in self.pattern_constructors(name, site)
            )
        found = lookup_bare(self, name, site, kind, span=span)
        return (
            isinstance(found, QualifiedTarget)
            and found.constructor is not None
            and self.constructor_declaration(found.constructor) == decl
        )

    def _spellings_reaching(self, decl: DeclarationKey) -> dict[_Spelling, None]:
        """Every spelling that may select constructor *decl*, ties in the order they settle.

        The name of the type *decl* is, or of the enum it is a member of --
        its own, each alias's, each rename's (:meth:`_bindings`), every import
        route to one, and the declaration's own path -- then the member under
        it. Whether one selects *decl* is for the lookup to say.
        """
        module_id, path, name = decl
        qname = (module_id, _bare_atom((*path, name)))
        owner_qname = (module_id, _bare_atom(path)) if path else None
        owner = None if owner_qname is None else self._type_owners.owner(owner_qname)
        target = (
            owner_qname
            if owner_qname is not None and owner is not None and name in owner.members
            else qname
        )
        member = None if target == qname else name
        spellings: dict[_Spelling, None] = {}

        def add(written: ScopePath, anchor: QualifierAnchor | None = None) -> None:
            """Spell the type by *written*, then its member (the type itself if a record)."""
            if member is None:
                spellings[(written[:-1], written[-1], anchor)] = None
            else:
                spellings[(written, member, anchor)] = None

        def reaches(exposed: QName) -> bool:
            return self._type_behind(exposed) == target

        if member is not None:
            spellings[((), member, None)] = None
        add((_bare_path(target[1])[-1],))
        bindings, renames = self._bindings()
        for written, exposed in bindings:
            if reaches(exposed):
                add(written)
        for rename in renames:
            add((rename,))
        for contribution in self._import_env.contributions.values():
            for atom, exposed in contribution.members.items():
                written = _bare_path(atom)
                for route, anchored in contribution_routes(contribution):
                    leading = "/".join(route)
                    anchor = QualifierAnchor.MODULE if anchored else None
                    if reaches(exposed):
                        add((leading, *written), anchor)
                    elif exposed == qname:
                        spellings[((leading, *written[:-1]), written[-1], anchor)] = None
        if module_id == self._module_id:
            add((*path, name) if member is None else path)
            add((*path, name) if member is None else path, QualifierAnchor.CURRENT_MODULE)
        return spellings

    def _type_behind(self, qname: QName) -> QName | None:
        """The enum, record or exception type path *qname* is, or a renaming alias of it names."""
        owners = self._type_owners
        enum = owners.enum_behind(qname)
        if enum is not None:
            return enum
        named = owners.identity(qname)
        owner = owners.owner(named)
        return named if owner is not None and owner.alias is None and owner.constructor else None

    def projected(
        self,
        owner: DeclarationKey,
        layer: ContributionLayer,
        rest: ScopePath,
        chain: QualifierChain,
        kind: LookupKind,
        *,
        owners_within: int,
        via: Via | None,
    ) -> Reading:
        """What type *owner*, made visible by *layer*, selects for *rest* by its own member table.

        Each name of *rest* but the last must name a type declared beneath
        the one before; one the owner so far only references or hides is
        refused. The owner reached last decides the member: a referenced or
        hidden member is refused, and a type it declares (an inline enum
        member or a nested type) is not selected: only a contribution reaching
        its full path selects it, so a ``hiding`` removes exactly that path. A
        path beneath an alias is its target's as written, read where the alias
        is declared, through *via*, how the owner was reached
        (:meth:`_beneath_alias`); the alias's projection of its target's
        member, or a record's own spelling, stands for what that reaches of it
        (:meth:`_standing_for`).
        """
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
            if table.alias is not None:
                through = self._beneath_alias(
                    current,
                    table.alias,
                    rest[index - start :],
                    layer,
                    chain,
                    kind,
                    owners_within - index + start,
                    via,
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

    def _beneath_alias(
        self,
        alias: QName,
        declaration: TypeAlias,
        path: ScopePath,
        layer: ContributionLayer,
        chain: QualifierChain,
        kind: LookupKind,
        owners_within: int,
        via: Via | None,
    ) -> Reading:
        """What *path* beneath *alias*, which *declaration* declares, selects as one of *kind*.

        What its target as written, then *path*, reaches where the alias is
        declared (:meth:`read_through`, *owners_within* as there): a
        declaration the alias's module reaches is reached as the alias is
        (*layer*), and a ``hiding`` there removing the path refuses it as
        *chain* spells it. When the alias was reached through *via* -- here,
        as the reader -- its target's paths are read through it, unless the
        target leads with a module route of its own.
        """
        spelling = _target_spelling(declaration)
        if spelling is None:
            return Reading()
        site = self._site_sources(alias[0])
        if via is not None and site.leads_with_route(alias, spelling):
            via = None
        reader = site if via is None else self
        read = reader.read_through(
            alias,
            spelling,
            path,
            kind,
            owners_within,
            chain.span,
            via,
            every_use=reader is not self,
        )
        refusals = self._respelled(read, chain, path)
        if via is not None:
            return Reading(read.candidates, refusals)
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

    def leads_with_route(self, alias: QName, spelling: NameT | AppliedT) -> bool:
        """Whether alias *alias*'s target, *spelling*, is reached here through a module route.

        The target head's own reading where the alias is declared, removed
        declarations included.
        """
        head = member_chain(spelling, ())
        with self._reading_through((alias, _Read.HEAD, None), (), every_use=False) as reads:
            if not reads:
                return False
            found = lookup_reached(
                self, head, _bare_path(alias[1])[:-1], LookupKind.TYPE, keep_removed=True
            )
        return any(isinstance(candidate.via, Route) for candidate in found)

    def _respelled(
        self, read: Reading, chain: QualifierChain, path: ScopePath
    ) -> tuple[AglError, ...]:
        """*read*'s refusals of *path* read for *chain*: a hidden path as *chain* spells it."""
        return tuple(
            hidden_member(chain, path[-1]) if isinstance(refusal, HiddenMemberError) else refusal
            for refusal in read.refusals
        )

    def _reached_target(self, target: QualifiedTarget) -> QualifiedTarget:
        """*target*, which another module reached, with the binding this module reads it through."""
        ref = target.ref
        if ref is None or ref.module_id == self._module_id:
            return target
        return replace(target, ref=self._cross_module_binding_ref(_ref_qname(ref)))

    @contextmanager
    def _reading_through(
        self, through: _Through, path: ScopePath, *, every_use: bool
    ) -> Iterator[bool]:
        """Yield whether to read *through* beneath an alias at *path*: not when already reading it.

        A read whose target as written leads back to the alias reaches nothing
        more once an in-progress read's path is *path* or a suffix of it; a
        different path is a different read. Every use is read when *every_use*, as
        another module reads it; otherwise those the read in progress sees.
        """
        reading = self._through.setdefault(through, [])
        if any(
            len(outer) <= len(path) and path[len(path) - len(outer) :] == outer for outer in reading
        ):
            yield False
            return
        reading.append(path)
        try:
            with self._uses.view(every_use):
                yield True
        finally:
            reading.pop()

    def read_through(
        self,
        alias: QName,
        spelling: NameT | AppliedT,
        path: ScopePath,
        kind: LookupKind,
        owners_within: int,
        span: SourceSpan,
        via: Via | None,
        *,
        every_use: bool,
    ) -> Reading:
        """What *spelling*, alias *alias*'s target as written, then *path*, reaches here.

        Read in the alias's region, at *span*, as a declaration of *kind*
        (:func:`lookup_through`), through *via*: only a type the target or
        *owners_within* more names reach, or an alias, projects its member
        table. Every use is read when *every_use* (:meth:`_reading_through`).
        """
        chain = replace(member_chain(spelling, path), span=span)
        within = _within(chain, path, owners_within)
        with self._reading_through((alias, kind, via), path, every_use=every_use) as reads:
            if not reads:
                return Reading()
            return lookup_through(
                self,
                chain,
                _bare_path(alias[1])[:-1],
                kind,
                owners_within=within,
                via=via,
                sealed=len(chain.segments) + 1 - len(path),
            )

    def origins_through(
        self, alias: QName, spelling: NameT | AppliedT, path: ScopePath, *, every_use: bool
    ) -> frozenset[Origin]:
        """The scopes and types *spelling*, alias *alias*'s target as written, then *path*, names.

        Read in the alias's region, as :meth:`read_through` reads it; removed ones included.
        """
        with self._reading_through(
            (alias, _Read.ORIGINS, None), path, every_use=every_use
        ) as reads:
            if not reads:
                return frozenset()
            return lookup_reached_origins(
                self, member_chain(spelling, path), _bare_path(alias[1])[:-1]
            )

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
                for origin in unremoved(
                    self.origins_through(alias, spelling, path, every_use=True), self
                )
                for relative in declared(origin)
            },
        }
        reached: dict[ScopePath, QName] = {}
        for relative in relatives:
            beneath = (*path, *relative)
            for kind in LookupKind:
                read = self.read_through(
                    alias, spelling, beneath, kind, len(beneath), span, None, every_use=True
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
        that, at the nearest step of the alias's region holding any (``::``
        reads the root alone); ``None`` when the target leads with a module
        route (:meth:`leads_with_route`).
        """
        if self.leads_with_route(alias, spelling):
            return None
        target = member_chain(spelling, ())
        names = (*(segment.name for segment in target.segments), target.member, *path)
        own = target.anchor is QualifierAnchor.CURRENT_MODULE
        for step in read_steps(own, _bare_path(alias[1])[:-1]):
            prefix = (*step, *names)
            found = {
                rest: origin
                for atom, origin in exports.items()
                if (rest := relative_under(atom, prefix)) is not None
            }
            if found:
                return found
        return {}

    def projected_origins(self, alias: DeclarationKey, rest: ScopePath) -> frozenset[Origin]:
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
                reached = (
                    unremoved(site.origins_through(current, spelling, (), every_use=True), self)
                    - found
                )
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
        within = _within(spelled, rest, owners_within)
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
            and (projected := owners.declared_path(selection) or _owner_path(selection)) is not None
        ):
            qname = projected
            arguments = argument.args if isinstance(argument, AppliedT) else None
            arity = parameter[1] if application is None else application.arity
            application = Application(_qname_decl_key(projected), arity, argument)
        return application

    def _exposed(self, route: Route | None) -> dict[str, list[_Exposed]]:
        """What the imports *route* names expose, by first segment; root tails' at ``None``."""
        found = self._exposures.get(route)
        if found is None:
            env = self._import_env
            exposed = (
                unqualified_exposures(env)
                if route is None
                else qualifier_exposures(env, route.route, anchored=route.anchored)
            )
            found = self._exposures[route] = {}
            for atom, qname, ways in exposed:
                exposed_path = _bare_path(atom)
                found.setdefault(exposed_path[0], []).append((exposed_path, qname, ways))
        return found

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

    def own_origins(self, path: ScopePath) -> frozenset[Origin]:
        """Full *path* when it is one of this module's own scope paths or types."""
        qname = (self._module_id, _bare_atom(path))
        if path in self._scope_nodes or self._type_owners.is_declared(qname):
            return frozenset({Origin(qname)})
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
            return member.key
        return _qname_decl_key(self._type_owners.identity(self._named(key)))

    def denotes(self, key: DeclarationKey) -> object:
        """What *key* names in an ambiguity: its identity, or what an alias denotes there."""
        denoted = self._type_owners.denotation(self._named(key))
        return self.identity(key) if denoted is None else denoted

    def aliases(self, key: DeclarationKey) -> bool:
        """Whether *key* declares a type alias."""
        owner = self._type_owners.owner(_key_qname(key))
        return owner is not None and owner.alias is not None

    def declares(self, key: DeclarationKey) -> bool:
        """Whether this module declares *key*."""
        return key[0] == self._module_id

    def contributed_origins(self, step: ScopePath, path: ScopePath) -> frozenset[Origin]:
        """The scopes and types contributions anchored at or above *step* reach as *path*.

        A contributed function, binding or injected enum member is none.
        """
        found: set[Origin] = set()
        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            relative = _bare_path(atom)
            for exposed, refs in layer.bare_contributions.items():
                for ref in refs:
                    qname = _ref_qname(ref)
                    found |= self._exposed_origin(
                        exposed,
                        relative,
                        qname,
                        self._region_import_ways(layer, exposed, qname),
                        typed=ref.contributes_a_type,
                    )
            for decl in self._uses.visible(layer):
                found |= self._uses.origins(layer.scope_path, decl, relative)
        env = self._import_env
        for exposed, qname, ways in self._exposed(None).get(path[0], ()):
            found |= self._exposed_origin(exposed, path, qname, ways, typed=True)
        for node_id, routes in self._reachable_decl_contributions(env.decl_scope_routes, step):
            relative = path[len(self._import_decl_scope_paths.get(node_id, ())) :]
            for exposed, sources in routes.items():
                rest = relative_under(exposed, relative)
                if rest is not None:
                    for module, source in sources:
                        found |= self._scope_origins(
                            (module, _route_root(source, rest)), (ImportWay(node_id),)
                        )
        return frozenset(found | self.module_route_origins((path[0],), path[1:], anchored=False))

    def routed_origins(self, route: Route, path: ScopePath) -> frozenset[Origin]:
        """The scopes and types module *route* reaches as *path* beneath it."""
        return self.module_route_origins(route.route, path, anchored=route.anchored)

    def module_route_origins(
        self, route: tuple[str, ...], path: ScopePath, *, anchored: bool
    ) -> frozenset[Origin]:
        """The scopes and types module *route* reaches as *path* beneath it; itself for none."""
        env = self._import_env
        if not path:
            return frozenset(
                Origin((module, ()))
                for module in qualifier_candidates(env, route, anchored=anchored)
            )
        found: set[Origin] = set()
        for exposed, qname, ways in self._exposed(Route(route, anchored)).get(path[0], ()):
            found |= self._exposed_origin(exposed, path, qname, ways, typed=True)
        for module, scope_paths, ways in qualifier_scope_paths(env, route, anchored=anchored):
            if any(relative_under(atom, path) is not None for atom in scope_paths):
                found |= self._scope_origins((module, path), ways)
        return frozenset(found)

    def _scope_origins(self, route: BareRoute, ways: Collection[ImportWay]) -> frozenset[Origin]:
        """The scopes scope *route* reaches that import declarations expose by *ways*.

        Each with what every one of *ways* removes (:meth:`_hiding`).
        """
        hiding = self._hiding(ways)
        return frozenset(Origin(origin, hiding) for origin in self._scope_route_origins(route))

    def _exposed_origin(
        self,
        exposed: NameAtom,
        path: ScopePath,
        qname: QName,
        ways: Collection[ImportWay],
        *,
        typed: bool,
    ) -> frozenset[Origin]:
        """The scope or type contributed *exposed*, naming *qname*, makes *path*.

        A scope above it, or its type when it is *typed*, with what every one
        of *ways* it is exposed by removes (:meth:`_hiding`).
        """
        rest = relative_under(exposed, path)
        if rest is None:
            return frozenset()
        module, atom = qname
        if rest:
            origin = (module, _bare_atom(_bare_path(atom)[: -len(rest)]))
        elif typed and self._type_owners.is_declared(qname):
            origin = qname
        else:
            return frozenset()
        return frozenset({Origin(origin, self._hiding(ways))})

    def _imported(
        self, step: ScopePath, path: ScopePath
    ) -> dict[BindingRef, tuple[Layers, Hiding, Via | None]]:
        """Return what import tails anchored at or above *step* bind at full *path*.

        Every layer from *step* outward contributes the path relative to its
        own; the module root's import tails and the module route spelled by
        its leading name contribute it whole. A binding several contribute
        keeps every one's tag, what the ``hiding`` of each declaration
        contributing it removes, and the route reaching it when its leading
        name's route does.
        """
        env = self._import_env
        reached: dict[BindingRef, tuple[Layers, set[frozenset[DeclarationKey]], Via | None]] = {}

        def add(
            ref: BindingRef, layers: Layers, ways: Iterable[ImportWay], via: Via | None = None
        ) -> None:
            found, hidings, routed = reached.setdefault(ref, (frozenset(), set(), None))
            reached[ref] = found | layers, hidings, routed or via
            hidings.update(self._hiding(ways))

        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            for ref, layers in layer.bare_contributions.get(atom, {}).items():
                qname = _ref_qname(ref)
                add(ref, layers, self._region_import_ways(layer, atom, qname))
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
                            (ImportWay(node_id, named.withheld),),
                        )
        imported = dict(env.unqualified_ways.get(_bare_atom(path), {}))
        routed: dict[QName, Via] = {}
        if path[1:]:
            route = Route((path[0],), anchored=False)
            for qname, ways in qualifier_member_ways(
                env, route.route, _bare_atom(path[1:])
            ).items():
                imported[qname] = imported.get(qname, frozenset()) | ways
                routed[qname] = route
        for qname, ways in imported.items():
            add(
                self._cross_module_binding_ref(qname),
                frozenset({ContributionLayer.IMPORTED}),
                ways,
                routed.get(qname),
            )
        return {ref: (layers, frozenset(ways), via) for ref, (layers, ways, via) in reached.items()}

    def _region_import_ways(
        self, layer: ScopeNode, atom: NameAtom, qname: QName
    ) -> frozenset[ImportWay]:
        """The ways import declarations of region *layer* expose *qname* there as *atom* by."""
        index = self._region_ways
        if index is None:
            index = self._region_ways = {}
            for node_id, members in self._import_env.decl_bare_ways.items():
                region = self._import_decl_scope_paths.get(node_id)
                for exposed, by_qname in members.items():
                    for exposing, ways in by_qname.items():
                        index.setdefault((region, exposed, exposing), set()).update(ways)
        return frozenset(index.get((layer.scope_path, atom, qname), ()))

    def _hiding(self, ways: Iterable[ImportWay]) -> Hiding:
        """What each of *ways* removes; none without any.

        Its import declaration's own ``hiding``'s declarations, and those the
        imported module's export ``hiding`` withholds on the way.
        """
        return (
            frozenset(
                self._import_hidden(way.node_id) | self._withheld_removed(way.withheld)
                for way in sorted(ways, key=_way_order)
            )
            or NOT_HIDDEN
        )

    def _ways_remove(self, ways: Iterable[ImportWay], declaration: QName) -> bool:
        """Whether every one of *ways* removes *declaration* (:meth:`_hiding`)."""
        return removes(self._hiding(ways), _qname_decl_key(declaration), self)

    def _withheld_removed(self, withheld: frozenset[QName]) -> frozenset[DeclarationKey]:
        """The declarations an export ``hiding`` withholding *withheld* removes, by identity."""
        found = self._withheld_keys.get(withheld)
        if found is None:
            found = self._withheld_keys[withheld] = hidden_keys(
                withheld, lambda qname: (qname,), self.removed_with
            )
        return found

    def _import_hidden(self, node_id: int) -> frozenset[DeclarationKey]:
        """The declarations import declaration *node_id*'s ``hiding`` removes, by identity."""
        found = self._hidden_by.get(node_id)
        if found is None:
            found = self._hidden_by[node_id] = hidden_keys(
                self._import_env.decl_hiding.get(node_id, ()),
                self._named_declarations,
                self.removed_with,
            )
        return found

    def _named_declarations(self, named: ItemDeclaration) -> tuple[QName, ...]:
        """The declarations import item *named* names; none if its alias path names none."""
        if not named.beneath:
            return (named.declaration,)
        return tuple(_key_qname(key) for key in self._named_beneath(named, ()))

    def _reject_unnamed(self, named: ItemDeclaration) -> None:
        """Reject import item *named* if its path beneath an exported alias names nothing."""
        if named.beneath and not self._named_beneath(named, ()):
            raise UnknownMemberError(
                spell_declaration(named.module, named.item),
                span=named.span,
                repair=MissRepair.NOT_EXPORTED,
            )

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
        chain = self._probe_chain(written[:-1], written[-1], named.span, anchor=None)
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
                via=None,
            ).candidates
            if (key := candidate.target.key) is not None
        )

    def _probe_chain(
        self,
        qualifier: tuple[str, ...],
        member: str,
        span: SourceSpan,
        *,
        anchor: QualifierAnchor | None,
    ) -> QualifierChain:
        """The spelling ``qualifier::member`` at *span*, looked up but never recorded."""
        node_id = self._program.node_id
        return QualifierChain(
            anchor,
            tuple(QualifierSegment(segment, None, span, node_id) for segment in qualifier),
            member,
            span,
            node_id,
        )

    def tail_exposed(self, exposed: NameAtom) -> tuple[QName, ...]:
        """The declarations the root import tails expose as *exposed* that none removes."""
        ways = self._import_env.unqualified_ways.get(exposed, {})
        return tuple(
            qname
            for qname in self._import_env.unqualified.get(exposed, ())
            if not self._ways_remove(ways.get(qname, ()), qname)
        )

    def reachable_exports(self) -> Iterator[QName]:
        """Yield the exports of every contributing route no import's ``hiding`` removes."""
        for contribution in self._import_env.contributions.values():
            for surface in contribution.routes.values():
                for atom, qname in surface.members.items():
                    if not self._ways_remove(surface.member_ways[atom], qname):
                        yield qname

    def tail_type_names(self) -> frozenset[str]:
        """The names of the types the root import tails expose and no ``hiding`` removes."""
        return frozenset(
            name
            for name in self._import_env.unqualified
            if isinstance(name, str)
            and any(qname in self._all_public_types for qname in self.tail_exposed(name))
        )

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
            (candidate for candidate in constructors if candidate.key == key),
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
        path = _bare_path(src_name)
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
        repl_session_type_paths: Mapping[ScopePath, TypeOwner] | None = None,
    ) -> None:
        super().__init__()
        self._module_id = module_id
        self._program = resolved.program
        self._import_env = import_env
        self._all_public_types = all_public_types
        self._type_owners = type_owners
        self._decl_info = decl_info
        self._cross_module_constructor_refs = cross_module_constructor_refs
        self._repl_session_type_paths = dict(repl_session_type_paths or {})
        self._repl_session_root_type_names = root_type_names(self._repl_session_type_paths)
        self._root_scope = resolved.root_scope
        self._scope_nodes = resolved.scope_nodes
        self._declarations = resolved.declarations
        self._scope_entity_kinds = resolved.scope_entity_kinds
        self._import_decl_scope_paths = resolved.import_decl_scope_paths
        self._type_declarations = resolved.type_declarations
        self._scoped_constructor_candidates = resolved.scoped_constructor_candidates
        self._constructor_candidates = resolved.constructor_candidates
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
