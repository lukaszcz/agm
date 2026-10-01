"""Program-level scope resolver for the AgL module system.

This module provides :func:`resolve_program`, which runs the scope-resolution
pass over an entire :class:`~agm.agl.modules.loader.ModuleGraph`, producing
a :class:`ResolvedProgram` that contains per-module :class:`ResolvedModule`
results plus whole-program pre-pass tables.

Design
------
- **Public surfaces**: declaration export maps covering top-level declarations
  and simple ``let``/``var`` bindings, plus separate named-scope identity maps
  per module, including explicit ``export`` declarations, all computed before
  any body is resolved.
- **Contribution import environment per module**: built from each module's
  import declarations against the already-loaded graph (no re-reading files).
- **Whole-program pre-pass tables**: ``all_public_funcs`` and ``all_public_types``
  collect source declarations BEFORE resolving any body, enabling cross-module
  mutual recursion without publishing host-only synthetic entries.
- **Static module roots**: every file-backed module permits declarations,
  parameters, and bindings but rejects root assignments and bare expressions;
  the incremental REPL is the executable-root host.
- **Header-only imports** (every module root): imports must appear before any
  declaration.
- **``::name`` self-reference**: resolved to the current module's own scope.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Protocol, cast

from agm.agl.artifact_cache import (
    retain_resolved_modules,
    retained_module_sources,
    retained_resolved_modules,
)
from agm.agl.attributes import is_param_declaration
from agm.agl.modules.ids import (
    ModuleId,
    expand_module_wildcard,
    render_route_member,
    spell_declaration,
    spell_scope_path,
)

if TYPE_CHECKING:
    from agm.agl.modules.loader import LoadedModule, ModuleGraph
from agm.agl.scope.imports import (
    ImportEnv,
    ImportTarget,
    NameAtom,
    PathAtom,
    QName,
    ScopeOrigins,
    SingleTarget,
    WildcardTarget,
    alias_prefix,
    build_import_env,
    declares_bare_constructor,
    matching_atoms,
    validate_import_items,
)
from agm.agl.scope.resolver import _Resolver
from agm.agl.scope.symbols import (
    AglScopeError,
    BinderKind,
    ConstructorRef,
    DeclarationKey,
    DeclInfo,
    DuplicateDeclarationError,
    MissRepair,
    ModuleResolution,
    ReceiverOwner,
    ScopeNode,
    ScopePath,
    TypeOwner,
    TypeSelection,
    UnknownMemberError,
    builtin_type_static_kind,
    dedupe_constructor_candidates,
)
from agm.agl.scope.symbols import import_item_path as _item_path
from agm.agl.scope.symbols import to_bare_atom as _atom
from agm.agl.scope.symbols import to_bare_path as _path
from agm.agl.scope.type_names import selection_node_id
from agm.agl.scope.type_owners import (
    TypeOwnerIndex,
    beneath,
    declared_member_scopes,
    retired_member_scopes,
)
from agm.agl.semantics.type_table import source_enum_member_decl_id, source_nominal_decl_id
from agm.agl.syntax.nodes import (
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    ExportItem,
    FuncDef,
    ImportDecl,
    Item,
    LetDecl,
    Program,
    RecordDef,
    ScopeRegion,
    ScopeSegment,
    TypeAlias,
    VarDecl,
    VariantDef,
    VariantRef,
    exported_binding_name,
    static_binding_node_id,
    static_items,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, NameT, TypeExpr, member_type_params


def _mid_sort_key(m: ModuleId) -> tuple[str, ...]:
    return m.segments


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedModule:
    """Per-module output of the program resolver.

    ``module_id``
        The :class:`~agm.agl.modules.ids.ModuleId` of this module.
    ``resolved``
        The per-module scope resolution output (resolution tables, declared
        functions and related resolution data).
    ``import_env``
        The import environment computed from this module's import declarations.
    ``exports``
        Declaration export map for this module: maps each exported name to its
        origin :data:`~agm.agl.scope.imports.QName`.
    ``scope_exports``
        Named-scope export map. Scope identities are separate from declaration
        exports because an empty scope is public without denoting a value.
        Re-exports preserve and merge the scope's original module/path origins.
    ``loaded_program``
        The program as loaded: ``resolved.program`` keys each declaration
        written otherwise at the path it is declared at.
    """

    module_id: ModuleId
    resolved: ModuleResolution
    import_env: ImportEnv
    exports: dict[NameAtom, QName]
    scope_exports: dict[NameAtom, ScopeOrigins]
    source_text: str
    loaded_program: Program


@dataclass(frozen=True, slots=True)
class ResolvedProgram:
    """Output of :func:`resolve_program`.

    ``modules``
        Maps each :class:`~agm.agl.modules.ids.ModuleId` to its
        :class:`ResolvedModule`.
    ``entry_id``
        The graph's entry module identity — read from :attr:`graph` so the two
        can never name different modules. It is the module id the entry file's
        owning package declares, or
        :data:`~agm.agl.modules.ids.ENTRY_ID` for a source with no module
        identity.
    ``all_public_funcs``
        Whole-program pre-pass table mapping ``(ModuleId, name)`` to the
        :class:`~agm.agl.syntax.nodes.FuncDef` node. Contains every source-level
        function across all modules; host-only synthetic entries are excluded.
    ``all_public_types``
        Whole-program pre-pass table mapping ``(ModuleId, name)`` to the
        type declaration node (``RecordDef | EnumDef | TypeAlias``).
    ``import_sccs``
        The graph's import strongly-connected components, in their
        deterministic reverse-topological order for downstream program passes.
    ``graph``
        The loaded module graph, retained so consumers that need module
        reachability use its import/export adjacency rather than scope data.
    ``retired_member_scopes``
        Entry-only: the enum inline-member scopes this resolution retires
        relative to ``entry_repl_session_type_paths`` (see
        :func:`~agm.agl.scope.type_owners.retired_member_scopes`). Empty for a
        non-REPL resolution. The REPL promotes only the scopes of this same
        set that lie under a type path it also promotes, rather than
        recomputing them.
    """

    modules: dict[ModuleId, ResolvedModule]
    all_public_funcs: dict[QName, FuncDef]
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias]
    graph: ModuleGraph
    retired_member_scopes: frozenset[ScopePath] = frozenset()

    @property
    def entry_id(self) -> ModuleId:
        """Return the entry module's identity, as the loaded graph keys it."""

        return self.graph.entry_id

    @property
    def import_sccs(self) -> tuple[tuple[ModuleId, ...], ...]:
        """Return the loaded graph's import strongly-connected components."""

        return self.graph.sccs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_cross_module_constructor_candidates(
    import_env: ImportEnv,
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
    cross_module_constructor_refs: Mapping[QName, ConstructorRef],
    type_owners: TypeOwnerIndex,
    tail_removes: Callable[[NameAtom, QName, QName], bool],
) -> dict[str, tuple[ConstructorRef, ...]]:
    """Build constructor candidates from types exposed by import tails for a module.

    For each type exposed unqualified by an import tail:
    - RecordDef: add the record name as a candidate (e.g. ``Foo(x:1)``).
    - EnumDef: add each variant name as a candidate (e.g. ``Red``), and each
      member it references unless an import hides that member's declaration.
    - TypeAlias: add the alias name unless *type_owners* resolves it to an
      enum, following each alias of the chain where it is declared.

    A selected QName may also name an enum variant directly (e.g. an
    individually imported/renamed variant); such names are absent from
    ``all_public_types`` (which is keyed by owning-type QName), so they are
    resolved through ``cross_module_constructor_refs`` instead, which already
    carries a per-variant :class:`ConstructorRef`.

    A declaration every tail exposing it removes by ``hiding``
    (*tail_removes*) adds none.
    """
    candidates: dict[str, list[ConstructorRef]] = {}
    exposed_qnames = frozenset(
        qname for qnames in import_env.unqualified.values() for qname in qnames
    )
    seen_candidates: set[tuple[str, ConstructorRef]] = set()

    def hidden_here(ref: ConstructorRef) -> bool:
        """Whether an import hides *ref*'s declaration, which no other exposes."""
        qname = ref.qname
        return qname in import_env.unqualified_hidden and qname not in exposed_qnames

    def add_candidate(name: str, ref: ConstructorRef) -> None:
        candidate = (name, ref)
        if candidate not in seen_candidates:
            seen_candidates.add(candidate)
            candidates.setdefault(name, []).append(ref)

    def add_members(
        exposed_name: str, key: QName, enum_qname: QName, enum: EnumDef, *, through_alias: bool
    ) -> None:
        """Add the members enum *enum* at *enum_qname*, exposed as *key*, injects bare."""
        mid, src_name = enum_qname
        for member in enum.members:
            if isinstance(member, VariantRef):
                for referenced_cref in type_owners.referenced_member_refs(enum_qname, member):
                    if not hidden_here(referenced_cref) and not tail_removes(
                        exposed_name, key, referenced_cref.qname
                    ):
                        add_candidate(referenced_cref.owner_name, referenced_cref)
                continue
            if declares_bare_constructor(
                import_env.unqualified.get(member.name, ()), all_public_types
            ):
                continue
            member_qname = (mid, _atom((*_path(src_name), member.name)))
            if (through_alias or member_qname in exposed_qnames) and not tail_removes(
                exposed_name, key, member_qname
            ):
                add_candidate(member.name, cross_module_constructor_refs[member_qname])

    for exposed_name, qnames in import_env.unqualified.items():
        if not isinstance(exposed_name, str):
            continue
        for mid, src_name in qnames:
            key = (mid, src_name)
            if tail_removes(exposed_name, key, key):
                continue
            decl = all_public_types.get(key)
            if decl is None:
                variant_ref = cross_module_constructor_refs.get(key)
                if variant_ref is not None:
                    add_candidate(exposed_name, variant_ref)
                continue
            if isinstance(decl, (RecordDef, ExceptionDef)):
                cref = cross_module_constructor_refs[key]
                add_candidate(exposed_name, cref)
            elif isinstance(decl, TypeAlias):
                alias_ref = type_owners.alias_constructor(decl, key)
                target = type_owners.identity(key)
                enum = all_public_types[target]
                if alias_ref is not None:
                    add_candidate(exposed_name, alias_ref)
                elif isinstance(enum, EnumDef):
                    # The alias's paths are its target's: its members come with it.
                    add_members(exposed_name, key, target, enum, through_alias=True)
            else:
                add_members(exposed_name, key, key, decl, through_alias=False)
    return {name: dedupe_constructor_candidates(refs) for name, refs in candidates.items()}


def _import_tail_type_names(
    import_env: ImportEnv,
    all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
) -> frozenset[str]:
    """Return the type names import tails expose unqualified, for ``Owner::member`` access."""
    return frozenset(
        name
        for name, qnames in import_env.unqualified.items()
        if isinstance(name, str) and any(qname in all_public_types for qname in qnames)
    )


def _item_atom(
    item: FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias | BuiltinVarDecl,
) -> NameAtom:
    return _atom((*tuple(segment.name for segment in item.scope_path), item.name))


def _inline_members(enum: EnumDef, path: ScopePath) -> Iterator[tuple[VariantDef, NameAtom]]:
    """Yield each inline member of *enum*, declared at *path*, with its name atom."""
    for member in enum.members:
        if isinstance(member, VariantDef):
            yield member, _atom((*path, member.name))


def _static_binding_atom(item: LetDecl | VarDecl) -> NameAtom | None:
    """Return the exported name atom for a ``let``/``var``, or ``None``.

    The ``_`` wildcard is never exported.
    """
    name = exported_binding_name(item)
    if name is None:
        return None
    return _atom((*tuple(segment.name for segment in item.scope_path), name))


def _compute_local_scope_exports(
    self_id: ModuleId, program: Program
) -> dict[NameAtom, ScopeOrigins]:
    """Collect public named-scope identities from regions and shorthand paths."""
    result: dict[NameAtom, ScopeOrigins] = {}

    def add_path(path: PathAtom) -> None:
        for length in range(1, len(path) + 1):
            atom = _atom(path[:length])
            result[atom] = frozenset({(self_id, atom)})

    def collect_regions(items: Iterable[object], parent: PathAtom) -> None:
        for item in items:
            if not isinstance(item, ScopeRegion):
                continue
            path = (*parent, item.segment.name)
            add_path(path)
            collect_regions(item.items, path)

    collect_regions(program.body.items, ())
    for item in static_items(program.body.items):
        if isinstance(
            item,
            (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias, LetDecl, VarDecl),
        ):
            scope_path = tuple(segment.name for segment in item.scope_path)
            add_path(scope_path)
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                add_path((*scope_path, item.name))
    return result


def _compute_local_exports(self_id: ModuleId, program: Program) -> dict[NameAtom, QName]:
    """Compute declaration paths, including members below named scopes."""
    result: dict[NameAtom, QName] = {}
    for item in static_items(program.body.items):
        if isinstance(item, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            if isinstance(item, FuncDef) and item.is_synthetic:
                continue
            atom = _item_atom(item)
            result[atom] = (self_id, atom)
            if isinstance(item, EnumDef):
                for _member, variant_atom in _inline_members(item, _path(atom)):
                    result[variant_atom] = (self_id, variant_atom)
        elif isinstance(item, BuiltinVarDecl):
            atom = _item_atom(item)
            result[atom] = (self_id, atom)
        elif isinstance(item, (LetDecl, VarDecl)):
            binding_atom = _static_binding_atom(item)
            if binding_atom is not None:
                result[binding_atom] = (self_id, binding_atom)
    return result


def _public_type_owners(
    all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
) -> dict[QName, ReceiverOwner]:
    """Index public nominal declarations and inline enum members by QName."""
    result: dict[QName, ReceiverOwner] = {}
    for (module_id, atom), declaration in all_public_types.items():
        path = _path(atom)
        if isinstance(declaration, (RecordDef, EnumDef, ExceptionDef)):
            result[module_id, atom] = ReceiverOwner(module_id, path)
        if isinstance(declaration, EnumDef):
            for member, member_atom in _inline_members(declaration, path):
                result[module_id, member_atom] = ReceiverOwner(module_id, (*path, member.name))
    return result


def _member_record_constructor_refs(
    all_public_types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
) -> dict[QName, ConstructorRef]:
    """Build the one constructor metadata record for every member record.

    A record, exception, and inline enum member each declares the nominal
    record a constructor produces.  Local resolution, import injection, and
    later REPL retention all reuse these objects rather than recreating a
    variant-shaped view of the source declaration.
    """
    result: dict[QName, ConstructorRef] = {}
    for (module_id, atom), declaration in all_public_types.items():
        path = _path(atom)
        if isinstance(declaration, (RecordDef, ExceptionDef)):
            result[(module_id, atom)] = ConstructorRef(
                owner_name=declaration.name,
                owner_decl_node_id=source_nominal_decl_id(
                    module_id, path[:-1], declaration.name, declaration.node_id
                ),
                type_params=declaration.type_params,
                owner_module_id=module_id,
                owner_path=path[:-1],
                is_builtin=declaration.is_builtin,
            )
        elif isinstance(declaration, EnumDef):
            for member, variant_atom in _inline_members(declaration, path):
                result[(module_id, variant_atom)] = ConstructorRef(
                    owner_name=member.name,
                    owner_decl_node_id=source_enum_member_decl_id(
                        module_id,
                        path[:-1],
                        declaration.name,
                        member.name,
                        member.node_id,
                        is_builtin=declaration.is_builtin,
                    ),
                    type_params=member_type_params(
                        (cast(TypeExpr, field.type_expr) for field in member.fields),
                        declaration.type_params,
                    ),
                    owner_module_id=module_id,
                    owner_path=path,
                    can_match_bare_pattern=not member.fields,
                    is_builtin=declaration.is_builtin,
                    inline_enum_owner_decl_node_id=source_nominal_decl_id(
                        module_id, path[:-1], declaration.name, declaration.node_id
                    ),
                )
    return result


def _builtin_static_decl_node_ids(
    functions: Mapping[QName, FuncDef],
    types: Mapping[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias],
) -> frozenset[int]:
    """Return static declarations owned by a builtin record that registers statics."""
    result: set[int] = set()
    for (module_id, _atom_name), function in functions.items():
        owner_path = tuple(segment.name for segment in function.scope_path)
        if builtin_type_static_kind(module_id, owner_path, function.name) is None:
            continue
        owner_atom = _atom(owner_path)
        owner = types.get((module_id, owner_atom))
        if isinstance(owner, RecordDef) and owner.is_builtin:
            result.add(function.node_id)
    return frozenset(result)


def _raise_reexport_conflict(
    exposed: NameAtom, existing: QName, origin: QName, decl: ExportDecl
) -> None:
    raise DuplicateDeclarationError(
        spell_scope_path(_path(exposed)),
        reexports=tuple(
            spell_declaration(module_id, _path(atom)) for module_id, atom in (existing, origin)
        ),
        span=decl.span,
    )


def _raise_reexport_scope_conflict(exposed: NameAtom, decl: ExportDecl) -> None:
    raise DuplicateDeclarationError(spell_scope_path(_path(exposed)), span=decl.span)


Through = Callable[[ModuleId, PathAtom], Mapping[PathAtom, QName]]
"""The declarations a path beneath an alias module *ModuleId* exports reaches.

Its target's declaration at that path and every one beneath it, keyed by
their paths relative to it; none when the path lies beneath no exported
alias or the alias's module is not resolved yet.
"""


class Placements(Protocol):
    """What declaration a full path names, and where (:class:`TypeOwnerIndex`)."""

    def declaration(self, qname: QName) -> QName:
        """The declaration full path *qname* names."""
        ...

    def placement(self, qname: QName) -> QName:
        """The full path the declaration at *qname* is declared at."""
        ...

    def scopes_of(self, qname: QName) -> frozenset[QName]:
        """The scopes type *qname* stands for beside its path: an alias of a built-in type's."""
        ...


Withheld = dict[ModuleId, dict[NameAtom, frozenset[QName]]]
"""Per module, the declarations each re-exported atom's export ``hiding`` removes beneath it.

An atom absent withholds nothing.
"""


def _reaches_nothing(_module: ModuleId, _path: PathAtom) -> Mapping[PathAtom, QName]:
    """A :data:`Through` before any alias is resolvable."""
    return {}


class _Unresolved:
    """:class:`Placements` before any alias is resolvable: each path names itself."""

    def declaration(self, qname: QName) -> QName:
        return qname

    def placement(self, qname: QName) -> QName:
        return qname

    def scopes_of(self, qname: QName) -> frozenset[QName]:
        return frozenset()


def _is_beneath_any(qname: QName, removed: Collection[QName]) -> bool:
    """Whether *qname* is one of *removed* or lies beneath one."""
    module, atom = qname
    path = _path(atom)
    return any(
        other[0] == module and path[: len(_path(other[1]))] == _path(other[1]) for other in removed
    )


def _resolve_reexports(
    export_maps: dict[ModuleId, dict[NameAtom, QName]],
    scope_export_maps: dict[ModuleId, dict[NameAtom, ScopeOrigins]],
    withheld: Withheld,
    local_atoms: Mapping[ModuleId, frozenset[NameAtom]],
    type_origins: frozenset[QName],
    all_targets: dict[int, ImportTarget],
    graph: ModuleGraph,
    component: tuple[ModuleId, ...],
    through: Through,
    placements: Placements,
    *,
    validate: bool,
) -> None:
    """Fixed-point resolution of *component*'s explicit export declarations.

    *component* is one strongly-connected component of the module graph whose
    dependencies' export maps are final. Iterates until no new re-exported
    names are added. For each ``ExportDecl``, this function propagates the
    target module's exported names into the current module's export map with
    their origin :data:`QName` preserved; an item written through an alias the
    target exports names the declarations *through* reaches beneath its
    target. When *validate*, an item naming nothing is then an error. What
    each re-exported atom's export ``hiding`` removes, by *placements*, is
    recorded in *withheld*:
    an atom several export declarations forward withholds only what each
    does, and a module's own (*local_atoms*) nothing.

    Re-export name conflicts (same exposed name → different origin QNames)
    raise :class:`~agm.agl.scope.symbols.AglScopeError`.

    Propagation is monotone: a contribution that does not revisit an export
    declaration crosses at most ``declaration_count`` declarations, and one
    additional pass observes convergence. Continued growth after that can
    only depend on reapplying a declaration through a cycle.
    """

    def targets(decl: ExportDecl) -> tuple[ModuleId, ...]:
        target = all_targets[decl.node_id]
        if isinstance(target, SingleTarget):
            return (target.module,)
        return tuple(sorted(target.modules, key=_mid_sort_key))

    def additions(decl: ExportDecl, target_mid: ModuleId, *, allow_missing: bool) -> _Additions:
        return _compute_reexport_additions(
            decl,
            export_maps[target_mid],
            scope_export_maps[target_mid],
            withheld[target_mid],
            lambda path: through(target_mid, path),
            placements,
            allow_missing=allow_missing,
        )

    def propagate() -> SourceSpan | None:
        changed: SourceSpan | None = None
        for mid in component:
            removed: dict[NameAtom, frozenset[QName]] = dict.fromkeys(local_atoms[mid], frozenset())
            decls = graph.modules[mid].export_decls
            for decl in decls:
                for target_mid in targets(decl):
                    found = additions(decl, target_mid, allow_missing=True)
                    for exposed, qname in found.declarations.items():
                        if exposed in scope_export_maps[mid] and qname not in type_origins:
                            _raise_reexport_scope_conflict(exposed, decl)
                        existing = export_maps[mid].get(exposed)
                        if existing is None:
                            export_maps[mid][exposed] = qname
                            changed = decl.span
                        elif existing != qname:
                            _raise_reexport_conflict(exposed, existing, qname, decl)
                        kept = found.withheld.get(exposed, frozenset())
                        removed[exposed] = removed.get(exposed, kept) & kept
                    for exposed, origins in found.scopes.items():
                        existing = export_maps[mid].get(exposed)
                        if existing is not None and existing not in type_origins:
                            _raise_reexport_scope_conflict(exposed, decl)
                        existing_origins = scope_export_maps[mid].get(exposed, frozenset())
                        merged_origins = existing_origins | origins
                        if merged_origins != existing_origins:
                            scope_export_maps[mid][exposed] = merged_origins
                            changed = decl.span
            recorded = {exposed: hidden for exposed, hidden in removed.items() if hidden}
            if recorded != withheld[mid]:
                # Only *mid*'s exports withhold, so it has one.
                withheld[mid] = recorded
                changed = decls[-1].span
        return changed

    _converge(propagate, _export_count(graph, component))

    # A selection may target a re-export which is populated later in the
    # fixed point. Validate only after every reachable export has propagated.
    if not validate:
        return
    for mid in component:
        for decl in graph.modules[mid].export_decls:
            for target_mid in targets(decl):
                additions(decl, target_mid, allow_missing=False)


def _export_count(graph: ModuleGraph, component: tuple[ModuleId, ...]) -> int:
    """How many export declarations *component*'s modules make."""
    return sum(len(graph.modules[mid].export_decls) for mid in component)


def _converge(step: Callable[[], SourceSpan | None], declarations: int) -> None:
    """Repeat *step* until it changes nothing.

    *step* returns the span of the declaration it last changed through, or
    ``None`` when it changed nothing. A cycle's *declarations* bound how
    often a change can propagate; one further step observes convergence.
    """
    for _ in range(declarations + 1):
        changed = step()
        if changed is None:
            return
    raise AglScopeError("cyclic re-export expansion does not converge", span=changed)


@dataclass(frozen=True, slots=True)
class _Additions:
    """What one export forwards: declarations, what each withholds beneath it, and scopes."""

    declarations: dict[NameAtom, QName]
    withheld: dict[NameAtom, frozenset[QName]]
    scopes: dict[NameAtom, ScopeOrigins]


def _compute_reexport_additions(
    decl: ExportDecl,
    target_exports: Mapping[NameAtom, QName],
    target_scopes: Mapping[NameAtom, ScopeOrigins],
    target_withheld: Mapping[NameAtom, frozenset[QName]],
    through: Callable[[PathAtom], Mapping[PathAtom, QName]],
    placements: Placements,
    *,
    allow_missing: bool = False,
) -> _Additions:
    """Compute declaration and scope identities forwarded by one export.

    An item matching nothing the target exports names what *through* reaches
    beneath an alias the target exports, less what the target withholds
    beneath it, and forwards it. A ``hiding`` item removes the declarations
    it names, every one beneath them and beneath the scopes an alias of a
    built-in type among them stands for, whatever atom spells them or
    wherever it is placed (by *placements*), and each
    forwarded atom withholds them beneath it too. With *allow_missing*, a
    selected item matching nothing forwards nothing, and a ``hiding`` item
    matching nothing withholds the whole export.
    """
    result: dict[NameAtom, QName] = {}
    withheld_result: dict[NameAtom, frozenset[QName]] = {}
    scope_result: dict[NameAtom, ScopeOrigins] = {}
    region_prefix = tuple(segment.name for segment in decl.scope_path)

    def beneath_any(origin: QName, removed: Collection[QName]) -> bool:
        """Whether *origin*'s declaration, or where it is placed, is or lies beneath *removed*."""
        return _is_beneath_any(placements.declaration(origin), removed) or _is_beneath_any(
            placements.placement(origin), removed
        )

    def withheld_through(prefix: PathAtom) -> frozenset[QName]:
        """What the target withholds beneath the exported alias a prefix of *prefix* spells."""
        return next(
            (
                target_withheld.get(_atom(prefix[:end]), frozenset())
                for end in range(len(prefix) - 1, 0, -1)
                if _atom(prefix[:end]) in target_exports
            ),
            frozenset(),
        )

    def match(
        item: ExportItem,
    ) -> tuple[tuple[NameAtom, ...], tuple[NameAtom, ...], Mapping[PathAtom, QName]]:
        """Expand one selection item over the target's declaration and scope surfaces."""
        prefix = _item_path(item)
        declarations = matching_atoms(target_exports, prefix)
        scopes = matching_atoms(target_scopes, prefix)
        reached: Mapping[PathAtom, QName] = {}
        if not declarations and not scopes:
            withheld = withheld_through(prefix)
            reached = {
                relative: origin
                for relative, origin in through(prefix).items()
                if not beneath_any(origin, withheld)
            }
        if not declarations and not scopes and not reached and not allow_missing:
            raise UnknownMemberError(
                render_route_member(decl.module_path, prefix),
                span=decl.span,
                repair=MissRepair.NOT_EXPORTED,
            )
        return declarations, scopes, reached

    selected_items = [(item, *match(item)) for item in decl.items]
    hidden_items = [match(item) for item in decl.hidden]
    if not all(any(found) for found in hidden_items):
        # Resolution is partial: withholding everything keeps it beneath the
        # final one, which a later pass reaches.
        return _Additions(result, withheld_result, scope_result)
    removed = frozenset(
        path
        for declarations, _scopes, reached in hidden_items
        for origin in (*(target_exports[source] for source in declarations), *reached.values())
        for path in (placements.declaration(origin), *placements.scopes_of(origin))
    )
    hidden_scopes = {source for _declarations, scopes, _named in hidden_items for source in scopes}

    def rooted_atom(path: PathAtom) -> NameAtom:
        return _atom(region_prefix + path) if region_prefix else _atom(path)

    def exposed_for(item: ExportItem, source: NameAtom) -> NameAtom:
        """Spell one matched source under the item's rename, if it has one."""
        if item.rename is None:
            return source
        return _atom((item.rename, *_path(source)[len(_item_path(item)) :]))

    def add(exposed: NameAtom, origin: QName, withheld: frozenset[QName]) -> None:
        rooted = rooted_atom(_path(exposed))
        existing = result.get(rooted)
        if existing is not None and existing != origin:
            _raise_reexport_conflict(rooted, existing, origin, decl)
        result[rooted] = origin
        withheld |= removed
        withheld_result[rooted] = withheld_result.get(rooted, withheld) & withheld

    def add_selected_scope_prefixes(
        item: ExportItem,
        source: NameAtom,
        exposed: NameAtom,
        *,
        include_leaf: bool,
    ) -> None:
        """Publish actual source scopes needed to reach one selected path."""
        source_path = _path(source)
        exposed_path = _path(exposed)
        prefix = _item_path(item)
        limit = len(exposed_path) if include_leaf else len(exposed_path) - 1
        for length in range(1, limit + 1):
            source_prefix = (
                source_path[:length] if item.rename is None else (*prefix, *exposed_path[1:length])
            )
            origins = target_scopes[_atom(source_prefix)]
            rooted = rooted_atom(exposed_path[:length])
            scope_result[rooted] = scope_result.get(rooted, frozenset()) | origins

    def add_scopes_through(path: PathAtom, named: QName) -> None:
        """Publish the scopes reaching *path*, which names *named* through an alias.

        Those down to the alias are the target module's own; one beneath it
        is the target's path there.
        """
        module, atom = named
        target = _path(atom)
        for length in range(1, len(path)):
            origins = target_scopes.get(_atom(path[:length]))
            if origins is None:
                origins = frozenset({(module, _atom(target[: len(target) - len(path) + length]))})
            rooted = rooted_atom(path[:length])
            scope_result[rooted] = scope_result.get(rooted, frozenset()) | origins

    if not decl.items:
        for source, origin in target_exports.items():
            if not beneath_any(origin, removed):
                add(source, origin, target_withheld.get(source, frozenset()))
        for source, origins in target_scopes.items():
            kept = frozenset(origin for origin in origins if not beneath_any(origin, removed))
            if kept and source not in hidden_scopes:
                rooted = rooted_atom(_path(source))
                scope_result[rooted] = scope_result.get(rooted, frozenset()) | kept
        return _Additions(result, withheld_result, scope_result)

    for item, declarations, scopes, reached in selected_items:
        for source in declarations:
            exposed = exposed_for(item, source)
            add(exposed, target_exports[source], target_withheld.get(source, frozenset()))
            add_selected_scope_prefixes(item, source, exposed, include_leaf=False)
        for source in scopes:
            add_selected_scope_prefixes(item, source, exposed_for(item, source), include_leaf=True)
        withheld = withheld_through(_item_path(item))
        for relative, origin in reached.items():
            path = (*_item_path(item), *relative)
            add(exposed_for(item, _atom(path)), origin, withheld)
            if item.rename is None:
                add_scopes_through(path, origin)
    return _Additions(result, withheld_result, scope_result)


def _decl_to_import_target(decl: ImportDecl | ExportDecl, graph: ModuleGraph) -> ImportTarget:
    """Map an import/export declaration to an ImportTarget using the loaded graph.

    For single imports, returns a ``SingleTarget`` with the resolved
    ``ModuleId``.  For wildcard imports, returns a ``WildcardTarget`` over the
    modules the wildcard reaches.
    """
    if not decl.wildcard:
        mid = ModuleId(segments=tuple(decl.module_path))
        return SingleTarget(module=mid)
    return WildcardTarget(
        modules=frozenset(expand_module_wildcard(tuple(decl.module_path), graph.modules))
    )


def _declaration_atoms(
    program: Program,
) -> Iterator[
    tuple[
        FuncDef
        | RecordDef
        | EnumDef
        | ExceptionDef
        | TypeAlias
        | BuiltinVarDecl
        | LetDecl
        | VarDecl,
        NameAtom,
    ]
]:
    """Yield every declaration of *program* that claims a path, with the path it is written at."""
    for item in static_items(program.body.items):
        if isinstance(item, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias, BuiltinVarDecl)):
            yield item, _item_atom(item)
        elif isinstance(item, (LetDecl, VarDecl)):
            atom = _static_binding_atom(item)
            if atom is not None:
                yield item, atom


def _declared_keys(
    module_id: ModuleId, program: Program, declared: Mapping[NameAtom, QName]
) -> tuple[dict[int, ScopePath], dict[DeclarationKey, QName]]:
    """Key *module_id*'s declarations by the paths they are declared at.

    *declared* maps each declaration written otherwise than declared to the
    full path it is declared at (``def Geo::m`` with ``type Geo = Base`` at
    ``Base::m``, in ``Base``'s module). Returns the scope path, by node id,
    each declaration of *program* written otherwise is keyed at, and where
    each one keyed beneath another module's path is placed.
    """

    def keyed(atom: NameAtom) -> ScopePath:
        placed_module, placed = declared[atom]
        path = _path(placed)
        if placed_module != module_id:
            return path
        # An own path is keyed where the longest declaration above it is.
        for end in range(len(path) - 1, 0, -1):
            prefix = _atom(path[:end])
            if prefix in declared:
                return (*keyed(prefix), *path[end:])
        return path

    scopes: dict[int, ScopePath] = {}
    placements: dict[DeclarationKey, QName] = {}
    for item, atom in _declaration_atoms(program):
        if atom not in declared:
            continue
        path = keyed(atom)
        if path != _path(atom):
            scopes[item.node_id] = path[:-1]
        if declared[atom][0] != module_id:
            placements[module_id, path[:-1], path[-1]] = declared[atom]
    return scopes, placements


def _key_declarations(program: Program, scopes: Mapping[int, ScopePath]) -> Program:
    """Return *program* with each declaration in *scopes* written at the scope path it maps to.

    Regions keep their spelling: a region is where names are written, not a
    declaration.
    """

    def keyed[I: Item](item: I) -> I:
        if isinstance(item, ScopeRegion):
            return replace(item, items=tuple(keyed(child) for child in item.items))
        if (
            not isinstance(
                item,
                (
                    FuncDef,
                    RecordDef,
                    EnumDef,
                    ExceptionDef,
                    TypeAlias,
                    BuiltinVarDecl,
                    LetDecl,
                    VarDecl,
                ),
            )
            or item.node_id not in scopes
        ):
            return item
        path = scopes[item.node_id]
        written = item.scope_path
        segments = tuple(
            ScopeSegment(
                name,
                span=written[min(index, len(written) - 1)].span,
                node_id=written[min(index, len(written) - 1)].node_id,
            )
            for index, name in enumerate(path)
        )
        return replace(item, scope_path=segments)

    return replace(
        program, body=replace(program.body, items=tuple(keyed(item) for item in program.body.items))
    )


# ---------------------------------------------------------------------------
# Cross-module decl info type aliases
# ---------------------------------------------------------------------------

# Maps (module_id, name) → DeclInfo, for building BindingRef values for
# cross-module references.
_DeclInfo = dict[QName, DeclInfo]


@dataclass(frozen=True, slots=True)
class _ModuleTables:
    """One module's share of the whole-program tables, collected from its declarations."""

    exports: dict[NameAtom, QName]
    scope_exports: dict[NameAtom, ScopeOrigins]
    type_origins: frozenset[QName]
    alias_origins: frozenset[QName]
    funcs: dict[QName, FuncDef]
    types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias]
    decl_info: _DeclInfo
    type_owners: dict[QName, ReceiverOwner]
    constructor_refs: dict[QName, ConstructorRef]


def _module_tables(mid: ModuleId, program: Program) -> _ModuleTables:
    """Collect *mid*'s exports, declarations and declaration metadata from *program*."""
    funcs: dict[QName, FuncDef] = {}
    types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias] = {}
    decl_info: _DeclInfo = {}
    for item in static_items(program.body.items):
        if isinstance(item, FuncDef):
            if item.is_synthetic:
                continue
            key = (mid, _item_atom(item))
            funcs[key] = item
            decl_info[key] = DeclInfo(
                decl_node_id=item.node_id,
                decl_span=item.span,
                kind=BinderKind.function_binding,
                is_builtin=item.is_builtin,
                is_method=item.is_method,
            )
        elif isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            key = (mid, _item_atom(item))
            types[key] = item
            decl_info[key] = DeclInfo(
                decl_node_id=item.node_id,
                decl_span=item.span,
                kind=BinderKind.constructor_binding,
            )
            if isinstance(item, EnumDef):
                for member, member_atom in _inline_members(item, _path(key[1])):
                    decl_info[(mid, member_atom)] = DeclInfo(
                        decl_node_id=member.node_id,
                        decl_span=member.span,
                        kind=BinderKind.constructor_binding,
                    )
        elif isinstance(item, BuiltinVarDecl):
            key = (mid, _item_atom(item))
            decl_info[key] = DeclInfo(
                decl_node_id=item.node_id,
                decl_span=item.span,
                kind=BinderKind.builtin_var_binding,
            )
        elif isinstance(item, (LetDecl, VarDecl)):
            binding_atom = _static_binding_atom(item)
            if binding_atom is not None:
                decl_info[(mid, binding_atom)] = DeclInfo(
                    decl_node_id=static_binding_node_id(item),
                    decl_span=item.span,
                    kind=(
                        BinderKind.let_binding
                        if isinstance(item, LetDecl)
                        else BinderKind.var_binding
                    ),
                    is_param=is_param_declaration(item.attributes),
                )
    return _ModuleTables(
        exports=_compute_local_exports(mid, program),
        scope_exports=_compute_local_scope_exports(mid, program),
        type_origins=frozenset(types),
        alias_origins=frozenset(
            qname for qname, item in types.items() if isinstance(item, TypeAlias)
        ),
        funcs=funcs,
        types=types,
        decl_info=decl_info,
        type_owners=_public_type_owners(types),
        constructor_refs=_member_record_constructor_refs(types),
    )


def _declarations_beneath(decl_info: _DeclInfo) -> Callable[[QName], tuple[ScopePath, ...]]:
    """Return, for a path, the paths relative to it of every declaration beneath it."""
    beneath: dict[QName, list[ScopePath]] = {}
    for module_id, atom in decl_info:
        path = _path(atom)
        for end in range(1, len(path)):
            beneath.setdefault((module_id, _atom(path[:end])), []).append(path[end:])
    return lambda qname: tuple(beneath.get(qname, ()))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _reusable(
    cached_modules: Mapping[ModuleId, ResolvedModule] | None,
    module_id: ModuleId,
    loaded: LoadedModule,
) -> ResolvedModule | None:
    """Return an earlier resolution of *loaded*, when it is the very same module.

    Identity of the ``Program`` node is the whole condition: a reparse, a REPL
    splice or a redeclaration produces a different node and misses.
    """
    cached = cached_modules.get(module_id) if cached_modules is not None else None
    if cached is not None and cached.loaded_program is loaded.program:
        return cached
    return None


def resolve_program(
    graph: ModuleGraph,
    *,
    entry_repl_session_scope: ScopeNode | None = None,
    entry_repl_session_scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
    entry_repl_session_type_paths: Mapping[ScopePath, TypeOwner] | None = None,
    entry_repl_session_placements: Mapping[ScopePath, QName] | None = None,
    cached_modules: Mapping[ModuleId, ResolvedModule] | None = None,
) -> ResolvedProgram:
    """Run the full scope-resolution pass over a :class:`~agm.agl.modules.loader.ModuleGraph`.

    Parameters
    ----------
    graph:
        A loaded module graph from :func:`~agm.agl.modules.loader.build_repl_graph`.
    entry_repl_session_scope:
        When given, the entry module's root scope is parented to this scope
        so name lookups fall through to session bindings, and ``::name``
        self-references fall back to it too (REPL incremental mode).
    entry_repl_session_scope_nodes:
        Named scope layers promoted by prior REPL entries. They are copied into
        the entry's resolver so qualified members remain available.
    entry_repl_session_type_paths:
        Type-owned scope paths among the retained layers, each mapped to the
        :class:`~agm.agl.scope.symbols.TypeOwner` resolved when it was declared,
        so an alias keeps the target it resolved to then. The entry derives
        its retained constructors from these owners.
    entry_repl_session_placements:
        Each retained declaration keyed beneath another module's path, by
        its full path, mapped to the full path it is placed at there, as
        decided when it was declared.
    cached_modules:
        Resolutions from an earlier compilation of the same modules -- a REPL
        session's own image. A cached entry is reused only while it holds the
        very ``Program`` node this graph carries, so any reparse, splice or
        redeclaration misses it. Whatever this leaves uncovered is looked up in
        the process-global artifact cache, and this pass's own results are
        retained there for the next compilation.

    Returns
    -------
    ResolvedProgram
        The resolved graph with per-module resolution tables and whole-program
        pre-pass tables.

    Raises
    ------
    AglScopeError
        On the first static scope violation (first-error abort).
    """
    # ------------------------------------------------------------------
    # Step 1: Build local export maps (own declarations only).
    # ------------------------------------------------------------------
    retainable = retained_module_sources(graph)
    reusable: dict[ModuleId, ResolvedModule] = dict(retained_resolved_modules(retainable))
    if cached_modules is not None:
        reusable.update(cached_modules)
    cached_modules = reusable

    # Each module's declarations, keyed at their written paths until step 5
    # keys a declaration written otherwise at its declared path.
    # A reusable resolution brings its keyed declarations.
    programs: dict[ModuleId, Program] = {}
    for mid, loaded in graph.modules.items():
        cached = _reusable(cached_modules, mid, loaded)
        programs[mid] = loaded.program if cached is None else cached.resolved.program
    tables = {mid: _module_tables(mid, program) for mid, program in programs.items()}
    export_maps = {mid: dict(table.exports) for mid, table in tables.items()}
    scope_export_maps = {mid: dict(table.scope_exports) for mid, table in tables.items()}
    withheld: Withheld = {mid: {} for mid in tables}
    # Every re-export resolution starts over from the modules' own exports.
    local_exports = {mid: table.exports for mid, table in tables.items()}
    local_scope_exports = {mid: table.scope_exports for mid, table in tables.items()}
    local_atoms = {mid: frozenset(table.exports) for mid, table in tables.items()}

    # ------------------------------------------------------------------
    # Step 2: Map ImportDecl and ExportDecl → ImportTarget for every module.
    # ------------------------------------------------------------------
    all_targets: dict[int, ImportTarget] = {}
    for _mid, loaded in graph.modules.items():
        for decl in loaded.imports:
            target = _decl_to_import_target(decl, graph)
            all_targets[decl.node_id] = target
        for export_decl in loaded.export_decls:
            target = _decl_to_import_target(export_decl, graph)
            all_targets[export_decl.node_id] = target

    # ------------------------------------------------------------------
    # Step 3: Whole-program pre-pass — collect all funcs/types and
    # build decl_info for cross-module BindingRef construction.
    # ------------------------------------------------------------------
    all_public_funcs: dict[QName, FuncDef] = {}
    all_public_types: dict[QName, RecordDef | EnumDef | ExceptionDef | TypeAlias] = {}
    # decl_info: declaration metadata for building cross-module BindingRefs
    decl_info: _DeclInfo = {}
    type_origins: set[QName] = set()
    alias_origins: set[QName] = set()
    cross_module_type_owners: dict[QName, ReceiverOwner] = {}
    cross_module_constructor_refs: dict[QName, ConstructorRef] = {}

    def add_tables(table: _ModuleTables) -> None:
        all_public_funcs.update(table.funcs)
        all_public_types.update(table.types)
        decl_info.update(table.decl_info)
        type_origins.update(table.type_origins)
        alias_origins.update(table.alias_origins)
        cross_module_type_owners.update(table.type_owners)
        cross_module_constructor_refs.update(table.constructor_refs)

    def remove_tables(table: _ModuleTables) -> None:
        for funcs_key in table.funcs:
            del all_public_funcs[funcs_key]
        for types_key in table.types:
            del all_public_types[types_key]
        for info_key in table.decl_info:
            del decl_info[info_key]
        type_origins.difference_update(table.type_origins)
        alias_origins.difference_update(table.alias_origins)
        for owner_key in table.type_owners:
            del cross_module_type_owners[owner_key]
        for constructor_key in table.constructor_refs:
            del cross_module_constructor_refs[constructor_key]

    for table in tables.values():
        add_tables(table)

    prelude_static_decl_node_ids = _builtin_static_decl_node_ids(all_public_funcs, all_public_types)
    # ------------------------------------------------------------------
    # Step 4: The type-owner index over every module's prepared headers.
    # ------------------------------------------------------------------
    resolved_modules: dict[ModuleId, ResolvedModule] = {}
    resolvers: dict[ModuleId, _Resolver] = {}
    # Modules on a cycle whose exports are being settled (step 5): what their
    # aliases select is read against exports that may still change.
    settling: set[ModuleId] = set()

    declared_in_program = _declarations_beneath(decl_info)
    # Each module's declarations written beneath another path, with the full
    # path each is declared at; recorded as its exports are (``reexport``).
    declared_at: dict[ModuleId, Mapping[NameAtom, QName]] = {}

    def declared_beneath(qname: QName) -> Collection[ScopePath]:
        resolver = resolvers.get(qname[0])
        if resolver is None:
            return declared_in_program(qname)
        return resolver.retained_paths_beneath(_path(qname[1])).union(declared_in_program(qname))

    def reached_paths(
        qname: QName,
        spelling: NameT | AppliedT,
        paths: Collection[ScopePath],
        *,
        every_use: bool,
    ) -> frozenset[ScopePath]:
        module_id, atom = qname
        path = _path(atom)
        resolver = resolvers.get(module_id)
        if resolver is not None:
            return resolver.paths_reached_at(path[:-1], spelling, paths, every_use=every_use)
        hidden = resolved_modules[module_id].resolved.type_owners[path].hidden
        return frozenset(paths).difference(hidden)

    def builtin_scopes(qname: QName, name: str, *, every_use: bool) -> frozenset[QName]:
        module_id, atom = qname
        path = _path(atom)
        resolver = resolvers.get(module_id)
        if resolver is not None:
            return resolver.scopes_named_at(path[:-1], name, every_use=every_use)
        return resolved_modules[module_id].resolved.type_owners[path].scopes

    def alias_target(
        qname: QName, alias: TypeAlias, spelling: NameT | AppliedT
    ) -> TypeSelection | None:
        resolver = resolvers.get(qname[0])
        if resolver is not None:
            return resolver.alias_target(qname, alias, spelling, validate=qname[0] not in settling)
        return resolved_modules[qname[0]].resolved.owner_declarations.get(
            selection_node_id(spelling)
        )

    def current_selection(
        module_id: ModuleId,
        scope_path: ScopePath,
        spelling: NameT | AppliedT | VariantRef,
        *,
        every_use: bool,
    ) -> TypeSelection | None:
        resolver = resolvers.get(module_id)
        if resolver is not None:
            return resolver.type_name_selection_at(scope_path, spelling, every_use=every_use)
        return resolved_modules[module_id].resolved.owner_declarations.get(
            selection_node_id(spelling)
        )

    # The index answers from prepared headers, so it is only asked once every
    # module below is constructed.
    type_owners = TypeOwnerIndex(
        all_public_types=all_public_types,
        constructor_refs=cross_module_constructor_refs,
        alias_targets=alias_target,
        declared_beneath=declared_beneath,
        reached_paths=reached_paths,
        builtin_scopes=builtin_scopes,
        current_selection=current_selection,
        declared_paths=lambda module_id: declared_at.get(module_id, {}),
    )

    # What earlier REPL entries retain stays current unless the entry
    # redeclares its path or retires the member scope it lies in.
    declared = declared_member_scopes(graph.modules[graph.entry_id].program.body.items)
    retired = retired_member_scopes(entry_repl_session_type_paths or {}, declared)
    retained_type_owners = (
        None
        if entry_repl_session_type_paths is None
        else {
            path: owner
            for path, owner in entry_repl_session_type_paths.items()
            if path not in declared and not beneath(path, retired)
        }
    )
    retained_scope_nodes = (
        None
        if entry_repl_session_scope_nodes is None
        else {
            path: node
            for path, node in entry_repl_session_scope_nodes.items()
            if not beneath(path, retired)
        }
    )

    def through(module_id: ModuleId, path: PathAtom) -> dict[PathAtom, QName]:
        found = alias_prefix(path, export_maps[module_id], alias_origins)
        if found is None:
            return {}
        (alias_module, alias_atom), rest = found
        named = type_owners.path_target((alias_module, _atom((*_path(alias_atom), *rest))))
        target_module, target_atom = named
        reached: dict[PathAtom, QName] = {(): named} if named in decl_info else {}
        for relative in declared_in_program(named):
            reached[relative] = (target_module, _atom((*_path(target_atom), *relative)))
        return reached

    import_envs: dict[ModuleId, ImportEnv] = {}
    # Step 5 keys a declaration written otherwise at the scope path it is
    # declared at (by node id), and records where each one keyed beneath
    # another module's path is placed.
    declared_scopes: dict[ModuleId, Mapping[int, ScopePath]] = {}
    placements: dict[ModuleId, Mapping[DeclarationKey, QName]] = {}

    def retained_placements(program: Program) -> dict[DeclarationKey, QName]:
        """Return what earlier REPL entries placed, but at paths *program* does not declare."""
        declared = {_path(atom) for _item, atom in _declaration_atoms(program)}
        return {
            (graph.entry_id, path[:-1], path[-1]): placement
            for path, placement in (entry_repl_session_placements or {}).items()
            if path not in declared
        }

    placements[graph.entry_id] = retained_placements(programs[graph.entry_id])

    def prepare(mid: ModuleId) -> None:
        """Build *mid*'s import environment and resolver over the current exports."""
        loaded = graph.modules[mid]
        cached = _reusable(cached_modules, mid, loaded)
        if cached is not None:
            # An import environment is a function of the module's own
            # import declarations and the exports of what they name --
            # exactly what a reusable resolution was built against -- so
            # it is reused with it.
            import_envs[mid] = cached.import_env
            resolved_modules[mid] = cached
            return
        module_targets: dict[int, ImportTarget] = {
            decl.node_id: all_targets[decl.node_id] for decl in loaded.imports
        }
        import_envs[mid] = build_import_env(
            loaded.imports,
            module_targets,
            export_maps,
            scope_export_maps,
            alias_origins,
            withheld,
        )
        if mid not in settling:
            validate_imports(mid)
        is_entry = mid == graph.entry_id
        resolvers[mid] = _Resolver(
            loaded.program,
            module_id=mid,
            import_env=import_envs[mid],
            decl_info=decl_info,
            cross_module_constructor_refs=cross_module_constructor_refs,
            builtin_static_decl_node_ids=prelude_static_decl_node_ids,
            cross_module_type_owners=cross_module_type_owners,
            all_public_types=all_public_types,
            type_owners=(
                type_owners.with_retained(mid, retained_type_owners)
                if is_entry and retained_type_owners is not None
                else type_owners
            ),
            allow_root_statements=is_entry and entry_repl_session_scope is not None,
            is_standard_library_module=mid.is_standard_library,
            repl_session_scope=entry_repl_session_scope if is_entry else None,
            repl_session_scope_nodes=retained_scope_nodes if is_entry else None,
            repl_session_type_paths=retained_type_owners if is_entry else None,
            origin_path=loaded.path,
            spaced_qualifiers=loaded.spaced_qualifiers,
            ambient_type_names=_import_tail_type_names(import_envs[mid], all_public_types),
            placements=placements.get(mid),
            declared_scopes=declared_scopes.get(mid),
        )

    def validate_imports(mid: ModuleId) -> None:
        """Reject an import item of *mid* naming nothing, over the exports as they now stand."""
        validate_import_items(
            graph.modules[mid].imports, all_targets, export_maps, scope_export_maps, alias_origins
        )

    def declared_paths(mid: ModuleId) -> Mapping[NameAtom, QName]:
        resolver = resolvers.get(mid)
        if resolver is None:
            return resolved_modules[mid].resolved.declared_paths
        return resolver.declared_paths()

    def reexport(members: tuple[ModuleId, ...], *, validate: bool) -> None:
        """Resolve *members*' re-exports afresh through their prepared aliases.

        A module exports each own declaration at the path it is declared at
        (``def Geo::m`` with ``type Geo = Base`` as ``Base::m``).
        """
        for mid in members:
            declared = declared_paths(mid)
            declared_at[mid] = declared
            export_maps[mid] = {
                (declared[atom][1] if atom in declared else atom): qname
                for atom, qname in local_exports[mid].items()
            }
            local_atoms[mid] = frozenset(export_maps[mid])
            scope_export_maps[mid] = dict(local_scope_exports[mid])
            withheld[mid] = {}
        _resolve_reexports(
            export_maps,
            scope_export_maps,
            withheld,
            local_atoms,
            frozenset(type_origins),
            all_targets,
            graph,
            members,
            through,
            type_owners,
            validate=validate,
        )

    def settle(members: tuple[ModuleId, ...]) -> SourceSpan | None:
        """Prepare *members* over their exports and re-resolve them.

        Returns where the first member whose exports changed links into its
        cycle -- its first export or import -- or ``None`` when none changed.
        """
        read = [(export_maps[mid], scope_export_maps[mid], withheld[mid]) for mid in members]
        for mid in members:
            prepare(mid)
        reexport(members, validate=False)
        changed = [
            mid
            for mid, before in zip(members, read, strict=True)
            if (export_maps[mid], scope_export_maps[mid], withheld[mid]) != before
        ]
        if not changed:
            return None
        type_owners.forget(members)
        loaded = graph.modules[changed[0]]
        # A cycle's member links into it through an export or an import.
        return loaded.export_decls[0].span if loaded.export_decls else loaded.imports[0].span

    # ------------------------------------------------------------------
    # Step 5: Per strongly-connected component, dependencies first: prepare
    # its modules' import environments and headers, then resolve its
    # re-exports, which read its aliases. A component on a cycle imports its
    # own exports: it is first prepared over exports resolved without alias
    # reach, then prepared again over each re-resolution until they settle,
    # so every member reads its importees' final exports.
    # ------------------------------------------------------------------
    for component in graph.sccs:
        members = tuple(sorted(component, key=_mid_sort_key))
        if len(members) > 1 or members[0] in graph.adjacency.get(members[0], ()):
            _resolve_reexports(
                export_maps,
                scope_export_maps,
                withheld,
                local_atoms,
                frozenset(type_origins),
                all_targets,
                graph,
                members,
                _reaches_nothing,
                _Unresolved(),
                validate=False,
            )
            settling.update(members)
            _converge(partial(settle, members), _export_count(graph, members))
            settling.difference_update(members)
            # Its items name what its modules export once those settle.
            for mid in members:
                validate_imports(mid)
        else:
            prepare(members[0])
        reexport(members, validate=True)
        # A declaration written otherwise is keyed at the path it is declared
        # at, so every later reading -- exports, typecheck, display, REPL
        # retention -- sees the one key. Each member keyed afresh is prepared
        # again over its keyed declarations, as are its component's others,
        # which read them.
        for mid in members:
            declared_scopes[mid], placed = _declared_keys(mid, programs[mid], declared_paths(mid))
            if declared_scopes[mid]:
                programs[mid] = _key_declarations(programs[mid], declared_scopes[mid])
            placements[mid] = (
                {**retained_placements(programs[mid]), **placed}
                if mid == graph.entry_id
                else placed
            )
        keyed = [mid for mid in members if declared_scopes[mid]]
        if keyed:
            for mid in keyed:
                remove_tables(tables[mid])
                tables[mid] = _module_tables(mid, programs[mid])
                add_tables(tables[mid])
                local_exports[mid] = tables[mid].exports
                local_scope_exports[mid] = tables[mid].scope_exports
            prelude_static_decl_node_ids = _builtin_static_decl_node_ids(
                all_public_funcs, all_public_types
            )
            declared_in_program = _declarations_beneath(decl_info)
            type_owners.forget(members)
            for mid in members:
                prepare(mid)
            reexport(members, validate=True)

    # ------------------------------------------------------------------
    # Step 6: Resolve each prepared module's bodies against the type owners
    # the prepared headers make selectable.
    # ------------------------------------------------------------------
    for mid in graph.modules:
        resolver = resolvers.get(mid)
        if resolver is None:
            continue
        # Build cross-module constructor candidates from unqualified import tails.
        cross_module_candidates = _build_cross_module_constructor_candidates(
            import_envs[mid],
            all_public_types,
            cross_module_constructor_refs,
            type_owners,
            resolver.tail_removes,
        )
        resolved = resolver.resolve(ambient_constructor_candidates=cross_module_candidates or None)
        if programs[mid] is not resolved.program:
            resolved = replace(resolved, program=programs[mid])
        resolved_modules[mid] = ResolvedModule(
            module_id=mid,
            resolved=resolved,
            import_env=import_envs[mid],
            exports=export_maps[mid],
            scope_exports=scope_export_maps[mid],
            source_text=graph.modules[mid].source_text,
            loaded_program=graph.modules[mid].program,
        )

    resolved_modules = {mid: resolved_modules[mid] for mid in graph.modules}
    retain_resolved_modules(retainable, resolved_modules)
    return ResolvedProgram(
        modules=resolved_modules,
        all_public_funcs=all_public_funcs,
        all_public_types=all_public_types,
        graph=graph,
        retired_member_scopes=retired,
    )
