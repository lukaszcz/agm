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
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

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
)
from agm.agl.scope.resolver import _Resolver
from agm.agl.scope.symbols import (
    AglScopeError,
    BinderKind,
    ConstructorRef,
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
    LetDecl,
    Program,
    RecordDef,
    ScopeRegion,
    TypeAlias,
    VarDecl,
    VariantDef,
    VariantRef,
    exported_binding_name,
    static_binding_node_id,
    static_items,
)
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
    """

    module_id: ModuleId
    resolved: ModuleResolution
    import_env: ImportEnv
    exports: dict[NameAtom, QName]
    scope_exports: dict[NameAtom, ScopeOrigins]
    source_text: str


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
            src_path = _path(src_name)
            owner_path = src_path[:-1]
            if isinstance(decl, (RecordDef, ExceptionDef)):
                cref = cross_module_constructor_refs[key]
                add_candidate(exposed_name, cref)
            elif (
                isinstance(decl, TypeAlias)
                and (alias_ref := type_owners.alias_constructor(decl, key)) is not None
            ):
                add_candidate(exposed_name, alias_ref)
            elif isinstance(decl, EnumDef):
                for member in decl.members:
                    if isinstance(member, VariantRef):
                        for referenced_cref in type_owners.referenced_member_refs(key, member):
                            if not hidden_here(referenced_cref) and not tail_removes(
                                exposed_name, key, referenced_cref.qname
                            ):
                                add_candidate(referenced_cref.owner_name, referenced_cref)
                        continue
                    if declares_bare_constructor(
                        import_env.unqualified.get(member.name, ()), all_public_types
                    ):
                        continue
                    member_atom = _atom((*owner_path, decl.name, member.name))
                    member_qname = (mid, member_atom)
                    if member_qname in exposed_qnames and not tail_removes(
                        exposed_name, key, member_qname
                    ):
                        add_candidate(member.name, cross_module_constructor_refs[member_qname])
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

Declaration = Callable[[QName], QName]
"""The declaration a full path names (:meth:`TypeOwnerIndex.declaration`)."""

Withheld = dict[ModuleId, dict[NameAtom, frozenset[QName]]]
"""Per module, the declarations each re-exported atom's export ``hiding`` removes beneath it.

An atom absent withholds nothing.
"""


def _reaches_nothing(_module: ModuleId, _path: PathAtom) -> Mapping[PathAtom, QName]:
    """A :data:`Through` before any alias is resolvable."""
    return {}


def _names_itself(qname: QName) -> QName:
    """A :data:`Declaration` before any alias is resolvable."""
    return qname


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
    declaration: Declaration,
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
    each re-exported atom's export ``hiding`` removes, by *declaration*, is
    recorded in *withheld*: an atom several export declarations forward
    withholds only what each does, and a module's own (*local_atoms*) nothing.

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
            declaration,
            allow_missing=allow_missing,
        )

    def propagate() -> tuple[bool, ExportDecl | None]:
        changed = False
        changed_decl: ExportDecl | None = None
        for mid in component:
            removed: dict[NameAtom, frozenset[QName]] = dict.fromkeys(local_atoms[mid], frozenset())
            for decl in graph.modules[mid].export_decls:
                for target_mid in targets(decl):
                    found = additions(decl, target_mid, allow_missing=True)
                    for exposed, qname in found.declarations.items():
                        if exposed in scope_export_maps[mid] and qname not in type_origins:
                            _raise_reexport_scope_conflict(exposed, decl)
                        existing = export_maps[mid].get(exposed)
                        if existing is None:
                            export_maps[mid][exposed] = qname
                            changed = True
                            changed_decl = decl
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
                            changed = True
                            changed_decl = decl
            recorded = {exposed: hidden for exposed, hidden in removed.items() if hidden}
            if recorded != withheld[mid]:
                withheld[mid] = recorded
                changed = True
        return changed, changed_decl

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


def _converge(step: Callable[[], tuple[bool, ExportDecl | None]], declarations: int) -> None:
    """Repeat *step* until it changes nothing.

    *step* reports whether it changed anything, and the export declaration
    it last changed through, if any. A cycle's *declarations* bound how
    often a change can propagate; one further step observes convergence.
    """
    last_changed: ExportDecl | None = None
    for _ in range(declarations + 1):
        changed, changed_decl = step()
        if changed_decl is not None:
            last_changed = changed_decl
        if not changed:
            return
    raise AglScopeError(
        "cyclic re-export expansion does not converge",
        span=last_changed.span if last_changed else None,
    )


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
    declaration: Declaration,
    *,
    allow_missing: bool = False,
) -> _Additions:
    """Compute declaration and scope identities forwarded by one export.

    An item matching nothing the target exports names what *through* reaches
    beneath an alias the target exports, less what the target withholds
    beneath it, and forwards it. A ``hiding`` item removes the declarations
    it names (by *declaration*) and every one beneath them, whatever atom
    spells them, and each forwarded atom withholds them beneath it too. With
    *allow_missing*, a selected item matching nothing forwards nothing, and a
    ``hiding`` item matching nothing withholds the whole export.
    """
    result: dict[NameAtom, QName] = {}
    withheld_result: dict[NameAtom, frozenset[QName]] = {}
    scope_result: dict[NameAtom, ScopeOrigins] = {}
    region_prefix = tuple(segment.name for segment in decl.scope_path)

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
                if not _is_beneath_any(declaration(origin), withheld)
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
        declaration(origin)
        for declarations, _scopes, reached in hidden_items
        for origin in (*(target_exports[source] for source in declarations), *reached.values())
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
            if not _is_beneath_any(declaration(origin), removed):
                add(source, origin, target_withheld.get(source, frozenset()))
        for source, origins in target_scopes.items():
            kept = frozenset(
                origin for origin in origins if not _is_beneath_any(declaration(origin), removed)
            )
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


# ---------------------------------------------------------------------------
# Cross-module decl info type aliases
# ---------------------------------------------------------------------------

# Maps (module_id, name) → DeclInfo, for building BindingRef values for
# cross-module references.
_DeclInfo = dict[QName, DeclInfo]


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
    if cached is not None and cached.resolved.program is loaded.program:
        return cached
    return None


def resolve_program(
    graph: ModuleGraph,
    *,
    entry_repl_session_scope: ScopeNode | None = None,
    entry_repl_session_scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
    entry_repl_session_type_paths: Mapping[ScopePath, TypeOwner] | None = None,
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

    export_maps: dict[ModuleId, dict[NameAtom, QName]] = {}
    scope_export_maps: dict[ModuleId, dict[NameAtom, ScopeOrigins]] = {}
    type_origins: set[QName] = set()
    alias_origins: set[QName] = set()
    withheld: Withheld = {}
    for mid, loaded in graph.modules.items():
        export_maps[mid] = _compute_local_exports(mid, loaded.program)
        scope_export_maps[mid] = _compute_local_scope_exports(mid, loaded.program)
        withheld[mid] = {}
        for item in static_items(loaded.program.body.items):
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                type_origins.add((mid, _item_atom(item)))
            if isinstance(item, TypeAlias):
                alias_origins.add((mid, _item_atom(item)))

    # Every re-export resolution starts over from the modules' own exports.
    local_exports = {mid: dict(exports) for mid, exports in export_maps.items()}
    local_scope_exports = {mid: dict(scopes) for mid, scopes in scope_export_maps.items()}
    local_atoms = {mid: frozenset(exports) for mid, exports in export_maps.items()}

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

    for mid, loaded in graph.modules.items():
        for item in static_items(loaded.program.body.items):
            if isinstance(item, FuncDef):
                if item.is_synthetic:
                    continue
                key = (mid, _item_atom(item))
                all_public_funcs[key] = item
                decl_info[key] = DeclInfo(
                    decl_node_id=item.node_id,
                    decl_span=item.span,
                    kind=BinderKind.function_binding,
                    is_builtin=item.is_builtin,
                    is_method=item.is_method,
                )
            elif isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                key = (mid, _item_atom(item))
                all_public_types[key] = item
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
                    key = (mid, binding_atom)
                    decl_info[key] = DeclInfo(
                        decl_node_id=static_binding_node_id(item),
                        decl_span=item.span,
                        kind=(
                            BinderKind.let_binding
                            if isinstance(item, LetDecl)
                            else BinderKind.var_binding
                        ),
                        is_param=is_param_declaration(item.attributes),
                    )

    prelude_static_decl_node_ids = _builtin_static_decl_node_ids(all_public_funcs, all_public_types)
    cross_module_type_owners = _public_type_owners(all_public_types)
    cross_module_constructor_refs = _member_record_constructor_refs(all_public_types)
    # ------------------------------------------------------------------
    # Step 4: The type-owner index over every module's prepared headers.
    # ------------------------------------------------------------------
    resolved_modules: dict[ModuleId, ResolvedModule] = {}
    resolvers: dict[ModuleId, _Resolver] = {}
    # Modules on a cycle whose exports are being settled (step 5): what their
    # aliases select is read against exports that may still change.
    settling: set[ModuleId] = set()

    declared_in_program = _declarations_beneath(decl_info)

    def declared_beneath(qname: QName) -> Collection[ScopePath]:
        resolver = resolvers.get(qname[0])
        if resolver is None:
            return declared_in_program(qname)
        return resolver.retained_paths_beneath(_path(qname[1])).union(declared_in_program(qname))

    def reached_paths(
        qname: QName, spelling: NameT | AppliedT, paths: Collection[ScopePath]
    ) -> frozenset[ScopePath]:
        module_id, atom = qname
        path = _path(atom)
        resolver = resolvers.get(module_id)
        if resolver is not None:
            return resolver.paths_reached_at(path[:-1], spelling, paths)
        hidden = resolved_modules[module_id].resolved.type_owners[path].hidden
        return frozenset(paths).difference(hidden)

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
        module_id: ModuleId, scope_path: ScopePath, spelling: NameT | AppliedT | VariantRef
    ) -> TypeSelection | None:
        resolver = resolvers.get(module_id)
        if resolver is not None:
            return resolver.type_name_selection_at(scope_path, spelling)
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
        current_selection=current_selection,
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
        )

    def reexport(members: tuple[ModuleId, ...], *, validate: bool) -> None:
        """Resolve *members*' re-exports afresh through their prepared aliases."""
        for mid in members:
            export_maps[mid] = dict(local_exports[mid])
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
            type_owners.declaration,
            validate=validate,
        )

    def settle(members: tuple[ModuleId, ...]) -> tuple[bool, None]:
        """Prepare *members* over their exports and re-resolve them; whether those changed."""
        read = [(export_maps[mid], scope_export_maps[mid], withheld[mid]) for mid in members]
        for mid in members:
            prepare(mid)
        reexport(members, validate=False)
        if [(export_maps[mid], scope_export_maps[mid], withheld[mid]) for mid in members] == read:
            return False, None
        type_owners.forget(members)
        return True, None

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
                _names_itself,
                validate=False,
            )
            settling.update(members)
            _converge(partial(settle, members), _export_count(graph, members))
            settling.difference_update(members)
        else:
            prepare(members[0])
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
        resolved_modules[mid] = ResolvedModule(
            module_id=mid,
            resolved=resolved,
            import_env=import_envs[mid],
            exports=export_maps[mid],
            scope_exports=scope_export_maps[mid],
            source_text=graph.modules[mid].source_text,
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
