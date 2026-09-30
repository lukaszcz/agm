"""Static name-resolution pass for the AgL pipeline.

Each module's ``_Resolver`` performs a full single-pass walk over its AST,
building the lexical scope chain and populating side tables:

- ``resolution``:     ``VarRef.node_id`` / bare-name ``AssignStmt.node_id`` → ``BindingRef``
- ``builtin_calls``:  ``Call.node_id`` → ``BuiltinKind``  (for contextual built-ins)
- ``pattern_slots``: metadata for shared bindings owned by a pattern match site
- ``match_site_pattern_slots``: match-site node id → the slot ids it created

Scope rules
-----------
1. ``let``/``var``/``def`` bind in the current scope; redeclaration in the
   *same* scope is an error. ``let _`` and ``var _`` resolve their right-hand
   sides but bind no name.
2. A bare-name ``:=`` resolves to the nearest visible binding; ``:=`` on an
   undeclared name → error.  Whether that binding is assignable is decided by
   type checking, which alone knows a pattern slot's selected meaning.  A
   *qualified* target is settled here: only an exported ``var`` or a
   ``builtin var`` is assignable across a module boundary, and no qualified
   name is ever a pattern slot.
   Indexed and field targets (``target[index] := value`` and
   ``target.field := value``) create no assignment binding: their receivers
   (and an index target's index) are resolved as ordinary expressions, and no
   ``resolution`` entry is recorded for the ``AssignStmt``.
3. Reading (``VarRef``) a name not visible in the current scope chain → error.
   ``_`` is always a discard wildcard and never resolves as a readable name.
4. Pattern variables and catch binders are immutable and branch-local.
5. ``loop`` body bindings are visible to the ``until`` condition but not after.
6. ``def`` declarations, including ``program def``, are valid at the module
   root and in named scope regions; a pre-pass collects them by path so root
   and same-scope members support mutual recursion.

Built-in call classification
-----------------------------
Contextual built-ins are reached through ordinary name resolution. In call
position, the resolver records ``Call.node_id → BuiltinKind`` only when the
resolved function declaration has ``builtin`` provenance. Runtime built-ins
may also resolve as values; lowering turns each typed occurrence into an
ordinary closure. The link-time ``resource`` operation remains call-only
because its argument must be a source literal.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, TypeVar, assert_never

from agm.agl.attributes import CONFIG_ATTRIBUTE, is_param_declaration
from agm.agl.constraints import ConstraintKind, close_constraints
from agm.agl.diagnostics import (
    AglError,
    HiddenMemberError,
    ReferencedMemberError,
    static_root_message,
    type_name_not_a_value,
)
from agm.agl.modules.ids import RESERVED_ID, ModuleId, render_route_member, spell_declaration
from agm.agl.scope.attributes import recognize_attributes
from agm.agl.scope.imports import (
    BareRoute,
    ImportEnv,
    NameAtom,
    QName,
    QualResolutionFound,
    ScopeOrigins,
    declares_bare_constructor,
    qualifier_candidates,
    qualifier_hides,
    qualifier_members,
    qualifier_scope_paths,
    resolve_qualified,
    route_spelling,
)
from agm.agl.scope.lookup import (
    Candidate,
    LookupKind,
    Misfit,
    QualifiedTarget,
    Reading,
    lookup_bare,
    lookup_qualified,
    lookup_steps,
)
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    BUILTIN_METHOD_RECEIVER_NAMES,
    BUILTIN_TYPE_STATIC_OWNER_PATHS,
    AglScopeError,
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    BinderKind,
    BindingRef,
    BuiltinKind,
    BuiltinStaticKind,
    ConstructorRef,
    ContributionLayer,
    DeclarationKey,
    DeclInfo,
    DuplicateDeclarationError,
    ImmutableAssignmentError,
    ImportedModuleOrigin,
    ImportedUseContribution,
    Layers,
    LocalUseContribution,
    MissRepair,
    ModuleResolution,
    NoVisibleConstructorError,
    PatternSlot,
    QualificationOrigin,
    ReceiverOwner,
    ResolvedUseTarget,
    ScopeNode,
    ScopePath,
    SlotCandidate,
    TypeOwner,
    TypeSelection,
    UnknownMemberError,
    UnknownQualifierError,
    add_layers,
    anchored_layers,
    builtin_call_kind,
    builtin_type_static_kind,
    contribution_origin,
    contribution_origins,
    drop_layer,
    duplicate_binder_message,
    is_builtin_type_static_owner,
    is_qualified_function_member,
    layered,
    undefined_name_message,
)
from agm.agl.scope.symbols import binding_qname as _ref_qname
from agm.agl.scope.symbols import import_item_path as _item_path
from agm.agl.scope.symbols import to_bare_atom as _bare_atom
from agm.agl.scope.symbols import to_bare_path as _bare_path
from agm.agl.scope.type_names import (
    MemberHidden,
    MemberReferenced,
    member_chain,
    owner_member_selection,
    selection_node_id,
)
from agm.agl.scope.type_owners import TypeOwnerIndex, owned_constructors, root_type_names
from agm.agl.semantics.type_table import (
    BUILTIN_PRELUDE_MEMBER_TYPE_DEFS,
    BUILTIN_PRELUDE_TYPE_DEFS,
)
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPES,
    COMPATIBILITY_PRELUDE_TYPE_NAMES,
    EnumType,
)
from agm.agl.syntax.advisories import SpacedQualifier

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.scope.imports import ImportEnv
    from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.nodes import (
    ArrayLit,
    AssignStmt,
    BinaryOp,
    Block,
    BoolLit,
    Break,
    BuiltinVarDecl,
    Call,
    Case,
    Cast,
    CatchClause,
    ConstructorPattern,
    Continue,
    DecimalLit,
    DictLit,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    Expr,
    FieldAccess,
    FieldTarget,
    FuncDef,
    If,
    ImportDecl,
    ImportItem,
    IndexAccess,
    IndexTarget,
    InfixDecl,
    InterpSegment,
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    Loop,
    NameTarget,
    NullLit,
    OperatorRef,
    Param,
    Pattern,
    Placeholder,
    Program,
    QualifierAnchor,
    QualifierChain,
    QualifierSegment,
    Raise,
    RecordDef,
    RecordUpdate,
    Return,
    ScopeRegion,
    ScopeSegment,
    StringLit,
    Template,
    Try,
    TypeAlias,
    TypeApply,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    UseDecl,
    VarDecl,
    VariantDef,
    VariantRef,
    VarRef,
    declares_source_entry,
    declares_synthetic_entry,
    pattern_binder_candidates,
    static_binding_name,
    static_binding_node_id,
)
from agm.agl.syntax.spans import SourceSpan, span_covering
from agm.agl.syntax.types import (
    TYPE_PARAMETER_WILDCARD,
    AppliedT,
    NameT,
    render_qualified_name,
    render_qualifier_path,
    render_type_expr,
)
from agm.agl.syntax.visitor import SyntaxNode, walk

_T = TypeVar("_T")
_RootDeclItem = TypeVar(
    "_RootDeclItem",
    FuncDef,
    RecordDef,
    EnumDef,
    ExceptionDef,
    TypeAlias,
    LetDecl,
    VarDecl,
    BuiltinVarDecl,
)


def _relative_under(atom: NameAtom, target: ScopePath) -> ScopePath | None:
    """Return *atom*'s path relative to *target*, or ``None`` if it is not under it.

    The one definition of the prefix test every use-target expansion performs
    when it re-spells a module or scope surface relative to the target named
    by a ``use``.
    """
    path = _bare_path(atom)
    return path[len(target) :] if path[: len(target)] == target else None


def _atom_under_prefix(atom: NameAtom, prefix: ScopePath) -> bool:
    """Whether *atom* falls under a selection *prefix* (a use tail or hiding item).

    The one definition of the prefix-match test, shared by
    ``_select_use_members`` (tail selection and its ``hiding`` loop) and the
    wildcard-facade refresh in ``_facade_refresh`` -- so a name a ``use``
    declaration hid cannot be reinstated by either path.
    """
    return _relative_under(atom, prefix) is not None


def _route_root(source: ScopePath, relative: ScopePath) -> ScopePath:
    """Return *source* with a target-relative suffix trimmed back off its end."""
    return source[: len(source) - len(relative)] if relative else source


def _bare_route_sort_key(route: BareRoute) -> tuple[str, ScopePath]:
    return route[0].path_str(), route[1]


def _keyed_bare_route(item: tuple[BareRoute, object]) -> tuple[str, ScopePath]:
    """Sort a route-keyed pair by its route, whatever the pair carries."""
    return _bare_route_sort_key(item[0])


# ---------------------------------------------------------------------------
# Built-in names and reserved-name enforcement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LocalScopeRoute:
    """Identity of a local scope exposed through ``use``."""

    path: ScopePath


@dataclass(frozen=True, slots=True)
class _UseTargetResolution:
    """All local and imported routes reachable through one ``use`` target."""

    declaration: UseDecl
    target: ScopePath
    local: tuple[tuple[ScopePath, ...], ...]
    route: tuple[str, ...]
    direct_candidates: tuple[tuple[ModuleId, Mapping[NameAtom, QName]], ...]
    direct_import_scope_routes: Mapping[BareRoute, Mapping[NameAtom, frozenset[BareRoute]]]
    imported: tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...]


# Built-in call names, sourced from ``symbols.BUILTIN_CALL_NAMES`` (the single
# source of truth). Runtime operations are first-class; ``resource`` remains a
# link-time form whose argument must be visible in its direct call syntax.
_BUILTIN_CALL_NAMES = BUILTIN_CALL_NAMES

# The set of names that may NOT be used as any kind of binding.
_RESERVED_NAMES: frozenset[str] = frozenset(_BUILTIN_CALL_NAMES)

# The items a static module root or scope region may hold besides its headers.
_ROOT_DECLARATION_ITEMS = (
    ScopeRegion,
    FuncDef,
    BuiltinVarDecl,
    RecordDef,
    EnumDef,
    ExceptionDef,
    TypeAlias,
    LetDecl,
    VarDecl,
    InfixDecl,
)


def _start_offset(item: Item) -> int:
    """Where *item* starts in its source text."""
    return item.span.start_offset


def _nested_misplacement(item: Item) -> AglScopeError | None:
    """Why a nested block may not hold *item*, or ``None``: these declare at a module root."""
    if isinstance(item, (ImportDecl, ExportDecl)):
        kind = "import" if isinstance(item, ImportDecl) else "export"
        message = (
            f"'{kind}' declarations are only allowed at the program root, "
            "not inside a nested block."
        )
    elif isinstance(item, InfixDecl):
        message = "infix declarations are only allowed at the program root."
    elif isinstance(item, FuncDef) and not item.scope_path:
        message = (
            "'def' declarations are only allowed at the program root, "
            f"not inside a nested block (found 'def {item.name}' here)."
        )
    elif isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
        kind_word = (
            "record"
            if isinstance(item, RecordDef)
            else "enum"
            if isinstance(item, EnumDef)
            else "exception"
            if isinstance(item, ExceptionDef)
            else "type"
        )
        message = (
            "Type declarations are only allowed at the top level of the program, "
            f"not inside a nested block (found '{kind_word}' here)."
        )
    elif isinstance(item, (LetDecl, VarDecl)) and item.scope_path:
        keyword = "let" if isinstance(item, LetDecl) else "var"
        message = (
            f"A scoped '{keyword}' binder path is only allowed at the program root, "
            "not inside a nested block."
        )
    elif isinstance(item, BuiltinVarDecl):
        message = (
            "'builtin var' declarations are only allowed at the module root, "
            f"not inside a nested block (found 'builtin var {item.name}' here)."
        )
    else:
        return None
    return AglScopeError(message, span=item.span)


def _scope_path_sort_key(path: ScopePath) -> tuple[int, ScopePath]:
    """Order scope paths by depth, then lexical spelling."""
    return (len(path), path)


def _binding_sort_key(ref: BindingRef) -> tuple[tuple[str, ...], ScopePath, str, int]:
    """Order bindings by their declaration identity, never by set order."""
    return (ref.module_id.segments, ref.scope_path, ref.name, ref.decl_node_id)


def _constructor_candidate_sort_key(
    candidate: ConstructorRef,
) -> tuple[tuple[str, ...], ScopePath, str, int]:
    """Order constructor candidates by their declaration identity, never by set order."""
    return (
        candidate.owner_module_id.segments,
        candidate.owner_path,
        candidate.owner_name,
        candidate.owner_decl_node_id,
    )


def _constraints_related(a: ConstraintKind, b: ConstraintKind) -> bool:
    """True if *a* and *b* are the same kind or one implies the other."""
    return a in close_constraints(frozenset({b})) or b in close_constraints(frozenset({a}))


def _is_root_inline_member(constructor: ConstructorRef) -> bool:
    """Whether *constructor* is an inline member of an enum declared at its module root."""
    return (
        constructor.inline_enum_owner_decl_node_id is not None and len(constructor.owner_path) == 1
    )


def _unknown_member(chain: QualifierChain, member: str) -> UnknownMemberError:
    """Return the one verdict for ``chain::member``, as written, selecting no member."""
    return UnknownMemberError(render_qualified_name(chain, member), span=chain.span)


def _use_target_spelling(decl: UseDecl, member_path: ScopePath = ()) -> str:
    """Spell *decl*'s target, then *member_path* beneath it, as written."""
    names = (*(segment.name for segment in decl.target), *member_path)
    if decl.current_module:
        return render_route_member((), names)
    return render_route_member(tuple(names[0].split("/")), names[1:], anchored=decl.anchored)


def _unknown_qualifier(chain: QualifierChain) -> UnknownQualifierError:
    """Return the one verdict for *chain*, as written, naming nothing that qualifies."""
    return UnknownQualifierError(render_qualifier_path(chain), span=chain.span)


def _key_qname(key: DeclarationKey) -> QName:
    """Return the full path declaration *key* names."""
    module_id, scope_path, name = key
    return module_id, _bare_atom((*scope_path, name))


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------


class _Resolver:
    """Stateful resolver that builds the scope tree and resolution tables.

    Implements explicit ``isinstance`` dispatch for each node kind.
    Use ``resolve_program`` — the public whole-program entry point — rather
    than instantiating this class directly.

    Resolution has two phases. Construction collects *program*'s
    declarations and resolves its header contributions: what ``use`` and
    region-import headers contribute is what type names written in any module
    see, so every module of a program is constructed before any is resolved,
    and this phase never asks the type-owner index. A header's contributed
    constructors depend on type owners, so they are completed by
    :meth:`resolve`. Header diagnostics of every module therefore precede
    body diagnostics of any module.

    When *repl_session_scope* is given, the entry's root scope is parented to
    it so name lookups fall through to session bindings (incremental REPL
    sessions), and ``::name`` self-references fall back to it too. New
    declarations live in the entry's own root scope and shadow parent
    bindings without a duplicate-declaration error.

    *ambient_type_names* carries type names from prior entries so that
    qualified constructor access (``Owner::variant``) resolves for types
    declared in earlier REPL entries.
    """

    def __init__(
        self,
        program: Program,
        module_id: ModuleId,
        import_env: ImportEnv,
        all_public_types: dict[
            tuple[ModuleId, NameAtom], RecordDef | EnumDef | ExceptionDef | TypeAlias
        ],
        type_owners: TypeOwnerIndex,
        decl_info: dict[tuple[ModuleId, NameAtom], DeclInfo],
        cross_module_constructor_refs: Mapping[tuple[ModuleId, NameAtom], ConstructorRef],
        cross_module_type_owners: Mapping[QName, ReceiverOwner],
        *,
        ambient_type_names: frozenset[str] = frozenset(),
        builtin_static_decl_node_ids: frozenset[int] = frozenset(),
        allow_root_statements: bool = False,
        is_standard_library_module: bool = False,
        repl_session_scope: ScopeNode | None = None,
        repl_session_scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
        repl_session_type_paths: Mapping[ScopePath, TypeOwner] | None = None,
        origin_path: Path | None = None,
        spaced_qualifiers: tuple[SpacedQualifier, ...] = (),
    ) -> None:
        # The REPL session layer is copied into this entry's own image, exactly
        # as a retained named layer already is (see ``_build_scope_nodes``), so
        # resolving this entry never mutates session state.
        if repl_session_scope is not None:
            repl_session_scope = repl_session_scope.entry_copy()
        # Program parameters. One _Resolver is built per module of a whole
        # program (see scope/program.py::resolve_program); the import
        # environment and whole-program public-type table are always real,
        # never absent, since a program always builds them (an empty
        # ImportEnv/dict for a module with no imports or no public types).
        self._module_id: ModuleId = module_id
        self._import_env: ImportEnv = import_env
        # Declaration metadata used to build cross-module references.
        self._decl_info: dict[tuple[ModuleId, NameAtom], DeclInfo] = decl_info
        self._cross_module_constructor_refs: Mapping[tuple[ModuleId, NameAtom], ConstructorRef] = (
            cross_module_constructor_refs
        )
        self._builtin_static_decl_node_ids = builtin_static_decl_node_ids
        # One canonical metadata object represents each member-record
        # declaration. Import routes, local candidates, and retained REPL
        # candidates all reuse it by declaration id.
        self._constructor_metadata_by_decl_id: dict[tuple[ModuleId, int], ConstructorRef] = {
            (ref.owner_module_id, ref.owner_decl_node_id): ref
            for ref in self._cross_module_constructor_refs.values()
        }
        for (declaration_module_id, atom), declaration in all_public_types.items():
            if isinstance(declaration, (RecordDef, ExceptionDef)):
                constructor = self._cross_module_constructor_refs[(declaration_module_id, atom)]
                self._constructor_metadata_by_decl_id[
                    (declaration_module_id, declaration.node_id)
                ] = constructor
            elif isinstance(declaration, EnumDef):
                path = (atom,) if isinstance(atom, str) else atom
                for member in declaration.members:
                    if not isinstance(member, VariantDef):
                        continue
                    constructor = self._cross_module_constructor_refs[
                        (declaration_module_id, _bare_atom((*path, member.name)))
                    ]
                    self._constructor_metadata_by_decl_id[
                        (declaration_module_id, member.node_id)
                    ] = constructor
        # Public nominal declarations and inline enum members establish scope
        # paths even when they have no separately public child members.
        self._cross_module_type_owners: Mapping[QName, ReceiverOwner] = cross_module_type_owners
        # Whole-program public-type table, used to follow a type alias's
        # target across an import when deciding whether the alias has a
        # variant-less constructor.
        self._all_public_types: dict[
            tuple[ModuleId, NameAtom], RecordDef | EnumDef | ExceptionDef | TypeAlias
        ] = all_public_types
        # What each type path selects as a constructor owner, by identity.
        # It reads every module's prepared headers, so only ``resolve`` asks it.
        self._type_owners = type_owners
        # Header contributions whose constructors depend on type owners,
        # completed in order when ``resolve`` runs.
        self._deferred_constructors: list[Callable[[], object]] = []
        # The REPL is an incremental host and intentionally retains root
        # statements. File and inline exec entries use static roots.
        self._allow_root_statements = allow_root_statements
        self._is_standard_library_module = is_standard_library_module
        # Optional REPL session scope: the root's parent, so the module root
        # reads prior session bindings.
        self._repl_session_scope: ScopeNode | None = repl_session_scope
        # Named scope layers retained by prior REPL entries. They are copied
        # into this entry's fresh resolver image and committed only after a
        # successful evaluation, so a failed entry cannot mutate session state.
        self._repl_session_scope_nodes = dict(repl_session_scope_nodes or {})
        # A retained ordinary member may be replaced by a later member at the
        # same path, but it cannot simultaneously become a namespace prefix.
        # Type and named-scope members already own retained scope nodes and are
        # therefore excluded from this ordinary-member set.
        self._repl_session_ordinary_member_paths = {
            (*path, name)
            for path, node in self._repl_session_scope_nodes.items()
            for name in node.members
            if (*path, name) not in self._repl_session_scope_nodes
        }
        # Each retained type-owned path maps to the owner it resolved to when
        # declared, so a retained alias keeps its target after a redeclaration.
        self._repl_session_type_paths = dict(repl_session_type_paths or {})
        # Root-level (unscoped) names among the retained type-owned paths
        # above, the single derivation both an entry's ambient type-name
        # visibility and its bare root-enum member injection read.
        self._repl_session_root_type_names: frozenset[str] = root_type_names(
            self._repl_session_type_paths
        )
        # This module's canonical source file, or None for a module with no
        # backing file (inline `-c` sources, direct REPL entries). Drives the
        # `extern def` placement check — externs require a file-backed module.
        self._origin_path: Path | None = origin_path
        # Spaced-qualifier advisories from this module's lex pass, keyed by the
        # offset of the ``::`` that whitespace kept out of a qualifier.  A
        # self-qualified reference that fails to resolve consults them to
        # explain the mis-parse (see ``_spaced_qualifier_repair``).
        self._spaced_qualifiers: dict[int, SpacedQualifier] = {
            advisory.dcolon_offset: advisory for advisory in spaced_qualifiers
        }

        self._resolution: dict[int, BindingRef] = {}
        self._builtin_calls: dict[int, BuiltinKind] = {}
        self._builtin_static_calls: dict[int, BuiltinStaticKind] = {}
        self._use_targets: dict[int, ResolvedUseTarget] = {}
        self._superseded_use_targets: set[ResolvedUseTarget] = set()
        # The imported paths a use combining an own scope with imported ones reaches.
        self._imported_use_surfaces: dict[int, dict[NameAtom, None]] = {}
        self._current_use_declaration_ids: set[int] = set()
        self._program = program
        # The module's root ScopeNode: the first layer of the root step.
        self._root_scope = ScopeNode(
            node_id=program.node_id, parent=repl_session_scope, scope_path=()
        )
        # The current lexical scope.
        self._scope = self._root_scope
        # Whether we are at the program root (for root-only checks).
        self._at_root: bool = False
        # Top-level function defs.
        self._declared_functions: dict[str, FuncDef] = {}
        # Names of all root-level type declarations (RecordDef/EnumDef/TypeAlias).
        self._declared_type_names: set[str] = set()
        # Named declarations are keyed uniformly by module and scope path; the
        # root is simply the empty path.
        self._declarations: dict[DeclarationKey, BindingRef] = {}
        # Every ImportDecl's own scope path, keyed by node_id -- lets a
        # region-scoped import's decl_bare contribution be found from any
        # position that can lexically reach it (its own region and every
        # descendant region), independent of the decl's textual position
        # relative to whatever consults it.
        self._import_decl_scope_paths: dict[int, ScopePath] = {}
        # Every registered declaration's item, keyed like its binding, so
        # legacy root-only tables can be derived from the same collection
        # without admitting scoped members.
        self._declaration_items: dict[
            DeclarationKey,
            FuncDef
            | RecordDef
            | EnumDef
            | ExceptionDef
            | TypeAlias
            | LetDecl
            | VarDecl
            | BuiltinVarDecl,
        ] = {}
        # Seeded with every retained scope path so a fresh entry's collision
        # check (``_ensure_scope_path``, ``_register_declaration``, ``_define``)
        # sees a session-retained named scope the same way it sees one
        # declared earlier in this same entry: a member or declaration this
        # entry tries to register at a retained scope's own path collides,
        # cross-entry, exactly as it would within one entry.
        self._scope_entity_kinds: dict[DeclarationKey, str] = {
            (self._module_id, path[:-1], path[-1]): "scope"
            for path in self._repl_session_scope_nodes
            if path
        }
        self._scope_paths: set[ScopePath] = {(), *self._repl_session_scope_nodes}
        self._ordered_binding_paths: set[ScopePath] = set()
        self._scope_node_ids: dict[ScopePath, int] = {
            path: node.node_id for path, node in self._repl_session_scope_nodes.items() if path
        }
        # Retained type-owned paths cover only a redeclarable type's own
        # path; its inline members (enum variants) are never separately
        # promoted (see session.py's ``promoted_type_name_paths``), so they
        # are reconstructed here from each retained owner's own ``members``,
        # exactly as a fresh declaration registers them within one entry
        # (below): a variant scope path is this module's own type path whether
        # its owner was declared this entry or retained from an earlier one.
        self._type_paths: set[ScopePath] = {
            *self._repl_session_type_paths,
            *(
                (*path, member_name)
                for path, owner in self._repl_session_type_paths.items()
                for member_name in owner.members
            ),
        }
        self._type_declarations: list[
            tuple[RecordDef | EnumDef | ExceptionDef | TypeAlias, ScopePath]
        ] = []
        # The same declarations indexed by their declaring scope path, in
        # declaration order, so a bare constructor lookup inside a region
        # costs one dict hit instead of a scan of every type in the module.
        self._type_declarations_by_path: dict[
            ScopePath, list[RecordDef | EnumDef | ExceptionDef | TypeAlias]
        ] = {}
        # Structured method identity -> nominal receiver owner. This is
        # scope's single receiver classification artifact for later passes.
        self._method_declarations: dict[DeclarationKey, ReceiverOwner] = {}
        # Alias scope paths a receiver may not claim, built on first use from
        # the completed declaration pre-pass and shared by every method.
        self._alias_receivers: dict[ScopePath, TypeAlias] | None = None
        self._scoped_constructor_candidates: dict[tuple[ScopePath, str], list[ConstructorRef]] = {}
        # Constructor candidates: name -> ordered list of ConstructorRef.
        self._constructor_candidates: dict[str, list[ConstructorRef]] = {}
        # Resolved single-candidate constructor refs: VarRef.node_id -> ConstructorRef.
        self._constructor_refs: dict[int, ConstructorRef] = {}
        # Qualified type-owner chains: QualifierChain.node_id -> the full
        # owner::member path's declaration identity, by suffix resolution.
        self._owner_declarations: dict[int, TypeSelection] = {}
        # Type aliases whose declaration is validated -- its target selected --
        # by node id: on the first type-owner query or at the ordered walk.
        self._validated_aliases: set[int] = set()
        # Scope records constructor candidates for bare pattern names. The
        # checker classifies them after constructor fields have been mapped;
        # candidates do not depend on ordinary lexical value bindings.
        self._pattern_constructor_candidates: dict[int, tuple[ConstructorRef, ...]] = {}
        # Bare ``is`` spellings remain candidate sets until typecheck knows the
        # nominal type of the left operand.
        self._is_test_constructor_candidates: dict[int, tuple[ConstructorRef, ...]] = {}
        # Qualified pattern and ``is`` spellings whose qualifier names a local plain scope.
        self._scope_qualified_spellings: set[int] = set()
        # Each case branch creates one shared slot per pattern binding name.
        # The checker selects its final target after the branch is classified.
        self._pattern_slots: dict[int, PatternSlot] = {}
        self._match_site_pattern_slots_by_node: dict[int, tuple[int, ...]] = {}
        self._next_pattern_slot_id: int = 0
        # Loop-context flag: True when resolving inside a loop body (while_cond,
        # body, or until_cond). Reset to False across fn/def boundaries so that
        # `break`/`continue` cannot cross a function boundary into an outer loop.
        self._in_loop: bool = False
        # Function-body flag: True only while resolving a def/fn body (not parameter
        # defaults). Used to reject `return` outside the nearest function boundary.
        self._in_function: bool = False
        # Whether this module declares a ``program def`` of its own, which
        # decides how a static-root rejection is explained.
        self._declares_program_entry = declares_source_entry(program.body.items)
        self._prepare(program, ambient_type_names)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _prepare(self, program: Program, ambient_type_names: frozenset[str]) -> None:
        """Collect *program*'s declarations and resolve its header contributions."""
        combined_ambient_type_names = ambient_type_names | self._repl_session_root_type_names
        if combined_ambient_type_names:
            self._declared_type_names.update(combined_ambient_type_names)
            self._type_paths.update((name,) for name in combined_ambient_type_names)

        # Published on the returned ModuleResolution as ``static_root``: true
        # for every importable module and for a loose file with its own
        # program def, false for the REPL and for a synthetic
        # (``-c``/entry-less) root, whose statements execute in textual order.
        self._is_static_root_module = (
            not self._allow_root_statements and not declares_synthetic_entry(program.body.items)
        )

        # Placement is legality, so it is reported before any name resolves.
        self._validate_item_placement(program.body.items)

        # Pre-pass 1: collect every named declaration and scope mention. This
        # establishes path-keyed membership before qualifier validation and the
        # legacy root worker build their compatibility tables.
        self._collect_declarations(program)

        # Pre-pass 2: collect top-level def names for mutual recursion.
        self._collect_func_decls()
        # Pre-pass 3: collect type-declaration names and validate type_params.
        self._collect_type_decl_names()

        self._scope_nodes = self._build_scope_nodes(self._root_scope)
        self._at_root = True
        self._resolve_headers(program.body.items)

    def resolve(
        self,
        *,
        ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
    ) -> ModuleResolution:
        """Resolve the prepared program's constructors and bodies; the second phase.

        Runs once every module of the program is constructed, since the
        type-owner index answers from all of their headers. It first completes
        the prepared headers' constructors, then collects this module's own,
        then walks the bodies.

        *ambient_constructor_candidates* carries the other modules'
        constructors that import tails make bare here.
        """
        program = self._program
        root = self._root_scope
        type_owners = self._declared_type_owners()
        # A retained path's owner is re-derived through the index rather than
        # read off its stored, declaration-time value: an alias's
        # reachable members/hidden set can go stale as later entries change
        # what is imported (``TypeOwnerIndex.owner``). Constructor candidates
        # and alias member paths read this one derivation.
        current_type_owners = {
            path: owner
            for path in {**self._repl_session_type_paths, **type_owners}
            if (owner := self._type_owners.owner((self._module_id, _bare_atom(path)))) is not None
        }
        self._validate_alias_member_paths(current_type_owners)
        for complete in self._deferred_constructors:
            complete()
        if ambient_constructor_candidates:
            for cname, crefs in ambient_constructor_candidates.items():
                for cref in crefs:
                    self._add_constructor_candidate(cname, cref)
        # Pre-pass 4: collect constructor candidates from the module's current
        # types: the earlier REPL entries' this entry leaves current, then its own.
        self._collect_constructor_candidates(current_type_owners)

        # Define root functions as value bindings; scoped members are already
        # present in their named-scope layers.
        self._define_function_bindings()
        # A static-root module's root let/var bindings are visible the same
        # way; scoped ones are already present in their named-scope layers
        # (``_build_scope_nodes``, seeded from ``_declarations`` above).
        self._define_static_binding_bindings()
        # A standard-library module's root ``builtin var`` follows the same
        # relaxed order (always static-root, since it is always file-backed).
        self._define_static_builtin_var_bindings()
        # Define constructor bindings in root scope.
        self._define_constructor_bindings()
        self._resolve_root_items(program.body.items)
        self._validate_function_names()
        self._validate_non_method_type_params()
        # Receiver classification follows the ordered lexical walk, so attribute
        # recognition runs only after every method declaration is known.
        attribute_facts = recognize_attributes(program, declares_receiver=self._declares_receiver)
        self._validate_extern_backing()
        self._refresh_local_use_contributions()
        self._validate_retained_imported_use_routes()

        return ModuleResolution(
            program=program,
            resolution=self._resolution,
            builtin_calls=self._builtin_calls,
            builtin_static_calls=self._builtin_static_calls,
            root_scope=root,
            declarations=dict(self._declarations),
            scope_nodes=dict(self._scope_nodes),
            static_root=self._is_static_root_module,
            origin_path=self._origin_path,
            declared_type_paths=frozenset(self._type_paths),
            constructor_candidates={
                name: tuple(refs) for name, refs in self._constructor_candidates.items()
            },
            constructor_refs=dict(self._constructor_refs),
            pattern_constructor_candidates=dict(self._pattern_constructor_candidates),
            is_test_constructor_candidates=dict(self._is_test_constructor_candidates),
            scope_qualified_spellings=frozenset(self._scope_qualified_spellings),
            pattern_slots=dict(self._pattern_slots),
            match_site_pattern_slots=dict(self._match_site_pattern_slots_by_node),
            method_declarations=dict(self._method_declarations),
            reachable_declarations=self._reachable_declarations(),
            attributes=attribute_facts,
            type_owners=type_owners,
            owner_declarations=dict(self._owner_declarations),
        )

    # ------------------------------------------------------------------
    # Pre-passes
    # ------------------------------------------------------------------

    def _declared_type_owners(self) -> dict[ScopePath, TypeOwner]:
        """Return the owner each type this module declares resolves to."""
        return {
            (*path, declaration.name): self._type_owners.declared_owner(
                (self._module_id, _bare_atom((*path, declaration.name))), declaration
            )
            for declaration, path in self._type_declarations
        }

    def _validate_alias_member_paths(self, owners: Mapping[ScopePath, TypeOwner]) -> None:
        """Reject a declaration at a path an own alias already declares.

        An alias's members are its target's (``type C = E`` declares
        ``C::Red``), so an ordinary or type declaration of this entry beneath
        the alias's path declares that path twice, as one beneath the enum
        itself would. The later of the two is reported; a retained alias is
        the earlier.
        """
        declared_here = {id(declaration) for declaration, _path in self._type_declarations}
        for path, owner in owners.items():
            alias = owner.alias
            if alias is None:
                continue
            for member in owner.members:
                key = (self._module_id, path, member)
                if self._scope_entity_kinds.get(key) not in {"ordinary", "type"}:
                    continue
                duplicate = self._declaration_items[key]
                if (
                    id(alias) in declared_here
                    and alias.span.start_offset > duplicate.span.start_offset
                ):
                    duplicate = alias
                raise DuplicateDeclarationError(member, span=duplicate.span)

    def _reachable_declarations(self) -> frozenset[DeclarationKey]:
        """Return local, imported, and retained declaration identities."""
        reachable = set(self._declarations)
        for contribution in self._import_env.contributions.values():
            for module_id, atom in contribution.members.values():
                path = _bare_path(atom)
                reachable.add((module_id, path[:-1], path[-1]))
        retained_nodes = (*self._repl_session_scope_nodes.values(), self._repl_session_scope)
        for node in retained_nodes:
            if node is None:
                continue
            for ref in (*node.bindings.values(), *node.members.values()):
                if ref.module_id != RESERVED_ID and ref.kind in {
                    BinderKind.function_binding,
                    BinderKind.constructor_binding,
                }:
                    reachable.add((ref.module_id, ref.scope_path, ref.name))
        return frozenset(reachable)

    def _collect_declarations(self, program: Program) -> None:
        """Collect static declarations into one path-keyed namespace.

        Region and shorthand syntax both arrive as declarations with a scope
        path. Regions additionally contribute their own path, so an empty
        region and a shorthand-only path have identical namespace identity.
        """
        for item in program.body.items:
            self._collect_item_declaration(item, ())

    def _collect_item_declaration(self, item: Item, enclosing_path: ScopePath) -> None:
        if isinstance(item, ScopeRegion):
            path = enclosing_path + (item.segment.name,)
            self._ensure_scope_path(path, item.segment.node_id, item.span)
            for child in item.items:
                self._collect_item_declaration(child, path)
            return
        if isinstance(item, ImportDecl):
            self._import_decl_scope_paths[item.node_id] = tuple(
                segment.name for segment in item.scope_path
            )
            return
        if isinstance(item, FuncDef) and item.is_synthetic:
            # The inline host entry is executable AST, not a source declaration.
            # Its body is still resolved by the main walk, but its implementation
            # name must never claim a source namespace slot.
            return
        if isinstance(item, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            path = tuple(segment.name for segment in item.scope_path)
            self._ensure_scope_path(path, item.node_id, item.span)
            self._register_declaration(item, path)
            return
        if isinstance(item, BuiltinVarDecl):
            path = tuple(segment.name for segment in item.scope_path) or enclosing_path
            if path:
                self._ensure_scope_path(path, item.node_id, item.span)
            self._register_builtin_var_declaration(item, path)
            return
        if isinstance(item, (LetDecl, VarDecl)):
            path = tuple(segment.name for segment in item.scope_path) or enclosing_path
            if path:
                # A binder's scope layer is order-independent even though its
                # membership is not: create the path here so a binder with no
                # sibling declaration still gets a scope node.
                self._ensure_scope_path(path, item.node_id, item.span)
            name = static_binding_name(item)
            if name == "_":
                return
            if path:
                self._ordered_binding_paths.add((*path, name))
            self._register_static_binding_declaration(item, path, name)

    def _ensure_scope_path(self, path: ScopePath, node_id: int, span: SourceSpan) -> None:
        """Create every scope layer in *path*, rejecting ordinary-name clashes."""
        for index, name in enumerate(path):
            parent_path = path[:index]
            scope_path = path[: index + 1]
            key = (self._module_id, parent_path, name)
            existing = self._scope_entity_kinds.get(key)
            if existing == "ordinary" or scope_path in self._repl_session_ordinary_member_paths:
                raise DuplicateDeclarationError(name, span=span)
            self._scope_entity_kinds.setdefault(key, "scope")
            self._scope_paths.add(scope_path)
            self._scope_node_ids.setdefault(scope_path, node_id)

    def _register_declaration(
        self,
        item: FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias,
        path: ScopePath,
    ) -> None:
        """Register one declaration and its type members at *path*."""
        key = (self._module_id, path, item.name)
        is_type = isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias))
        existing_entity = self._scope_entity_kinds.get(key)
        if is_type:
            if existing_entity == "ordinary" or existing_entity == "type":
                raise DuplicateDeclarationError(item.name, span=item.span)
            self._scope_entity_kinds[key] = "type"
        else:
            if existing_entity is not None:
                raise DuplicateDeclarationError(item.name, span=item.span)
            self._scope_entity_kinds[key] = "ordinary"

        kind = BinderKind.constructor_binding if is_type else BinderKind.function_binding
        self._declarations[key] = BindingRef(
            name=item.name,
            mutable=False,
            decl_span=item.span,
            decl_node_id=item.node_id,
            kind=kind,
            module_id=self._module_id,
            scope_path=path,
            is_builtin=isinstance(item, FuncDef) and item.is_builtin,
            is_method=isinstance(item, FuncDef) and item.is_method,
        )
        self._declaration_items[key] = item
        self._validate_type_params(item)
        if isinstance(item, FuncDef):
            self._validate_constraints(item)
        if isinstance(item, FuncDef):
            return

        self._type_declarations.append((item, path))
        self._type_declarations_by_path.setdefault(path, []).append(item)
        type_scope = path + (item.name,)
        self._scope_paths.add(type_scope)
        self._scope_node_ids.setdefault(type_scope, item.node_id)
        self._type_paths.add(type_scope)
        if isinstance(item, EnumDef):
            for member in item.members:
                if not isinstance(member, VariantDef):
                    continue
                variant_key = (self._module_id, type_scope, member.name)
                existing_variant = self._scope_entity_kinds.get(variant_key)
                if existing_variant in {"ordinary", "type"}:
                    raise DuplicateDeclarationError(member.name, span=member.span)
                self._scope_entity_kinds[variant_key] = "type"
                self._declarations[variant_key] = BindingRef(
                    name=member.name,
                    mutable=False,
                    decl_span=member.span,
                    decl_node_id=member.node_id,
                    kind=BinderKind.constructor_binding,
                    module_id=self._module_id,
                    scope_path=type_scope,
                )
                member_scope = type_scope + (member.name,)
                self._scope_paths.add(member_scope)
                self._scope_node_ids.setdefault(member_scope, member.node_id)
                self._type_paths.add(member_scope)

    def _register_static_binding_declaration(
        self, item: LetDecl | VarDecl, path: ScopePath, name: str
    ) -> None:
        """Register one module-level simple let/var like a def or a type.

        Claims *name* at *path* in ``_scope_entity_kinds``, the registry
        ``_ensure_scope_path``/``_register_declaration`` also consult, so a
        let/var collides with a same-path scope, def, or type regardless of
        textual order, in every module; ``_declaration_items`` keeps its item.

        A static-root module additionally installs the binding itself, like
        ``_declared_functions`` does for a def: ``_declarations`` makes it a
        scope member before the ordered walk
        reaches it (root through ``_define_static_binding_bindings``, a scope
        region through ``_build_scope_nodes``), so a def declared above it may
        read it. A non-static module's scoped bindings install only when the
        walk reaches them, keeping a script's line-by-line visibility.

        A root-position binding is always recorded in ``_declarations`` even
        so, since ``ScopeNode.lookup`` never consults the true root's
        ``.members`` (only a named scope region's) and so grants no early
        visibility there -- it only carries the binding into ``root.members``
        via ``_build_scope_nodes``, as a def's always does, for a later REPL
        entry to detect a same-path scope-region clash against it.
        """
        key = (self._module_id, path, name)
        existing_entity = self._scope_entity_kinds.get(key)
        if existing_entity is not None:
            raise DuplicateDeclarationError(name, span=item.span)
        self._scope_entity_kinds[key] = "ordinary"
        self._declaration_items[key] = item
        if path and not self._is_static_root_module:
            return
        self._declarations[key] = self._binder_ref(
            item, decl_node_id=static_binding_node_id(item), name=name, scope_path=path
        )

    def _register_builtin_var_declaration(self, item: BuiltinVarDecl, path: ScopePath) -> None:
        """Claim a ``builtin var``'s name early, like a static let/var.

        A ``builtin var`` is declared only by a standard-library module, which
        is always file-backed and so always static-root: unlike a scoped
        simple let/var, there is no non-static-root case, so this always
        installs the binding alongside claiming the name (mirrors
        :meth:`_register_static_binding_declaration`). Placement is checked
        first (:meth:`_validate_item_placement`) and the module kind by
        :meth:`_resolve_builtin_var`, at the walk's ordered position.
        """
        key = (self._module_id, path, item.name)
        existing_entity = self._scope_entity_kinds.get(key)
        if existing_entity is not None:
            raise DuplicateDeclarationError(item.name, span=item.span)
        self._scope_entity_kinds[key] = "ordinary"
        self._declarations[key] = self._binder_ref(
            item, decl_node_id=item.node_id, name=item.name, scope_path=path
        )
        self._declaration_items[key] = item

    def _classify_method_declaration(self, declaration: FuncDef, written_in: ScopePath) -> None:
        """Classify one receiver, written in named scope *written_in*, after preceding
        lexical contributions are visible."""
        if not declaration.is_method:
            return
        receiver = declaration.params[0]
        owner_path = tuple(segment.name for segment in declaration.scope_path)
        owner = (
            ReceiverOwner(self._module_id, owner_path)
            if declaration.receiver_type is not None
            else self._receiver_owner(declaration.scope_path, receiver, written_in)
        )
        if owner is None:
            return
        if receiver.default is not None:
            raise AglScopeError(
                f"Receiver 'self' for method '{declaration.name}' cannot have a default value.",
                span=receiver.span,
            )
        key = (self._module_id, owner_path, declaration.name)
        self._method_declarations[key] = owner

    def _receiver_owner(
        self, segments: tuple[ScopeSegment, ...], receiver: Param, written_in: ScopePath
    ) -> ReceiverOwner | None:
        """Return the type method path *segments*' receiver attaches to, if any.

        An empty path has none. The type's spelling is the ``def``'s own
        qualifier, decided inside the enclosing regions *written_in*, or else
        those regions' whole path, decided at the module root -- both by the
        one lookup (:meth:`_receiver_type_owner`). An unannotated
        receiver's rejection is raised; an annotated one only makes the
        ``def`` an ordinary function.
        """
        owner: ReceiverOwner | AglError | None = None
        if segments:
            start = len(written_in) if len(segments) > len(written_in) else 0
            with self._named_scope(written_in[:start]):
                owner = self._receiver_type_owner(segments[start:], receiver.span)
        if receiver.type_expr is not None:
            return owner if isinstance(owner, ReceiverOwner) else None
        if isinstance(owner, AglError):
            raise owner
        if owner is None:
            raise AglScopeError("'self' requires an enclosing type scope.", span=receiver.span)
        return owner

    def _receiver_type_owner(
        self, segments: tuple[ScopeSegment, ...], span: SourceSpan
    ) -> ReceiverOwner | AglError | None:
        """Return the type receiver spelling *segments* selects, or why none.

        A one-segment spelling selecting no type may name a built-in
        receiver type, or -- bare -- a constructor, whose owner the one bare
        constructor decision (:meth:`_value_constructors`) selects.
        """
        name = segments[-1].name
        chain = (
            QualifierChain(
                None,
                tuple(
                    QualifierSegment(segment.name, None, segment.span, segment.node_id)
                    for segment in segments[:-1]
                ),
                name,
                span_covering(segments[0].span, segments[-1].span),
                segments[-1].node_id,
            )
            if len(segments) > 1
            else None
        )
        found = self._type_target(chain, name, span)
        if isinstance(found, AglError):
            return found
        key = None if found is None else found.key
        owner = None if key is None else self._receiver_key_owner(key, span)
        if owner is not None or chain is not None:
            return owner
        if name in BUILTIN_METHOD_RECEIVER_NAMES:
            return ReceiverOwner(self._module_id, (name,))
        owners: dict[ReceiverOwner, list[QualificationOrigin]] = {}
        for candidate, layers in self._value_constructors(name).items():
            owners.setdefault(
                ReceiverOwner(
                    candidate.owner_module_id, (*candidate.owner_path, candidate.owner_name)
                ),
                [],
            ).extend(contribution_origins(candidate.qname, layers))
        if len(owners) > 1:
            return AmbiguousQualificationError.for_origins(
                (),
                (name,),
                itertools.chain.from_iterable(owners.values()),
                span=span,
                local_to=self._module_id,
            )
        return next(iter(owners), None)

    def _receiver_key_owner(self, key: DeclarationKey, span: SourceSpan) -> ReceiverOwner | None:
        """Return the receiver owner declaration *key* names when it is a type, rejecting an alias.

        This module's aliases are read from its REPL-retention-aware alias
        table (:meth:`_alias_receiver_paths`).
        """
        module_id, scope_path, name = key
        path = (*scope_path, name)
        qname = _key_qname(key)
        if module_id == self._module_id:
            alias = self._alias_receiver_paths().get(path)
            if alias is not None:
                self._raise_alias_receiver(name, alias, span)
            return ReceiverOwner(module_id, path) if self._type_owners.is_declared(qname) else None
        declaration = self._all_public_types.get(qname)
        if isinstance(declaration, TypeAlias):
            self._raise_alias_receiver(name, declaration, span)
        return self._cross_module_type_owners.get(qname)

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

    @staticmethod
    def _qname_decl_key(qname: QName) -> DeclarationKey:
        """Return the declaration identity a full path *qname* names directly."""
        module_id, atom = qname
        path = _bare_path(atom)
        return (module_id, path[:-1], path[-1])

    def _alias_receiver_paths(self) -> Mapping[ScopePath, TypeAlias]:
        """Return the module's alias scope paths with their declarations, computed once.

        The declaration pre-pass fills ``_type_declarations`` before any
        receiver is classified, so the table is the same for every method.
        """
        if self._alias_receivers is None:
            aliases: dict[ScopePath, TypeAlias] = {
                path: owner.alias
                for path, owner in self._repl_session_type_paths.items()
                if owner.alias is not None
            }
            for type_decl, path in self._type_declarations:
                type_scope = path + (type_decl.name,)
                if isinstance(type_decl, TypeAlias):
                    aliases[type_scope] = type_decl
                else:
                    aliases.pop(type_scope, None)
            self._alias_receivers = aliases
        return self._alias_receivers

    def _raise_alias_receiver(self, name: str, alias: TypeAlias, span: SourceSpan) -> None:
        """Reject a method receiver that names *alias*."""
        raise AglScopeError(
            f"'self' cannot declare a method in alias scope '{name}', "
            f"which targets '{render_type_expr(alias.type_expr)}'.",
            span=span,
        )

    def _reachable_decl_contributions(
        self, table: Mapping[int, Mapping[NameAtom, frozenset[_T]]], path: ScopePath
    ) -> Iterator[tuple[int, Mapping[NameAtom, frozenset[_T]]]]:
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

    def _build_scope_nodes(self, root: ScopeNode) -> dict[ScopePath, ScopeNode]:
        """Build named-scope layers and populate their declaration memberships."""
        nodes: dict[ScopePath, ScopeNode] = {(): root}
        for path in sorted(self._scope_paths, key=_scope_path_sort_key):
            if not path:
                continue
            retained = self._repl_session_scope_nodes.get(path)
            if retained is None:
                node = ScopeNode(
                    node_id=self._scope_node_ids[path],
                    parent=nodes[path[:-1]],
                    scope_path=path,
                )
            else:
                # A replayed REPL scope layer's declarations start from a
                # fresh members dict -- a retained member is registered like
                # any freshly declared one, so this entry's own registrations
                # (below) never mutate the session's persisted layer.
                node = retained.entry_copy()
                node.parent = nodes[path[:-1]]
                node.members = {}
                for name, ref in retained.members.items():
                    node.register_member(name, ref)
                # Its use contributions are re-snapshotted later, once for
                # every layer, by ``_refresh_local_use_contributions``.
            nodes[path] = node
        # A replacement type declaration owns fresh constructors: stale enum
        # variants must not survive, while unrelated retained members remain.
        for item, path in self._type_declarations:
            type_path = path + (item.name,)
            nested_scope_names = frozenset(
                nested_path[-1] for nested_path in nodes if nested_path[:-1] == type_path
            )
            nodes[type_path].clear_owned_constructor_members(nested_scope_names)
        for (_module_id, path, name), declaration in self._declarations.items():
            nodes[path].register_member(name, declaration)
        return nodes

    def _root_declaration_items(
        self, declaration_type: type[_RootDeclItem]
    ) -> Iterator[_RootDeclItem]:
        """Yield collected declarations of *declaration_type* at the root path."""
        for (module_id, path, _name), item in self._declaration_items.items():
            if module_id == self._module_id and not path and isinstance(item, declaration_type):
                yield item

    def _collect_func_decls(self) -> None:
        """Populate the legacy function table from root-path declarations only."""
        for item in self._root_declaration_items(FuncDef):
            self._declared_functions[item.name] = item

    def _non_method_functions(self) -> Iterator[FuncDef]:
        """Yield the function declarations that are not methods, in declaration order.

        Runs after the ordered resolution walk, so a method (its key
        is in ``self._method_declarations``) is left out: methods live in their
        receiver type's own member namespace and are validated against it.
        """
        for key, declaration in self._declaration_items.items():
            if isinstance(declaration, FuncDef) and key not in self._method_declarations:
                yield declaration

    def _validate_function_names(self) -> None:
        """Reject a built-in call name reused as an ordinary function name.

        Methods are exempt: their names live in their receiver type's own
        member namespace, distinct from the namespace this rule protects.
        """
        for declaration in self._non_method_functions():
            self._validate_function_decl(declaration)

    def _validate_function_decl(self, decl: FuncDef) -> None:
        """Apply function declaration validation independently of its scope path."""
        if (
            decl.name in _RESERVED_NAMES
            and not decl.is_builtin
            and not is_qualified_function_member(
                not self._module_id.is_entry,
                tuple(segment.name for segment in decl.scope_path),
            )
        ):
            raise AglScopeError(
                f"'{decl.name}' is a built-in name and cannot be used as a function name.",
                span=decl.span,
            )

    def _validate_non_method_type_params(self) -> None:
        """Reject a '_' type-parameter slot on a function that is not a method.

        A method's receiver-prefix '_' slots are validated later, against the
        receiver's arity, during type checking. Every other function
        declaration — including a plain ``def`` inside a type scope that lacks
        a ``self`` receiver — may not use '_' at all.
        """
        for declaration in self._non_method_functions():
            self._reject_type_param_wildcard(declaration)

    def _reject_type_param_wildcard(
        self, decl: FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias
    ) -> None:
        """Raise AglScopeError if *decl* uses '_' as a type-parameter slot."""
        for index, tp in enumerate(decl.type_param_slots):
            if tp == TYPE_PARAMETER_WILDCARD:
                raise AglScopeError(
                    "'_' is only allowed in a method's receiver type parameters; "
                    f"type parameter {index} of '{decl.name}' needs a name.",
                    span=decl.span,
                )

    def _declares_receiver(self, node: FuncDef) -> bool:
        """Return whether *node* was classified as a method of a nominal owner."""
        return (
            self._module_id,
            tuple(segment.name for segment in node.scope_path),
            node.name,
        ) in self._method_declarations

    def _validate_extern_backing(self) -> None:
        """Reject extern declarations only after their module-wide collection."""
        if self._origin_path is not None:
            return
        extern = next(
            (
                item
                for item in self._declaration_items.values()
                if isinstance(item, FuncDef) and item.is_extern
            ),
            None,
        )
        if extern is not None:
            raise AglScopeError(
                f"'extern def {extern.name}' requires a file-backed module; "
                "externs are not allowed in inline sources or REPL entries.",
                span=extern.span,
            )

    def _collect_type_decl_names(self) -> None:
        """Collect root type names for the existing constructor resolver.

        Duplicate declarations and type parameters were already validated by
        path-keyed collection. Builtin names are seeded so root-qualified
        prelude constructors remain available.
        """
        self._declared_type_names.update(BUILTIN_PRELUDE_TYPES)
        self._declared_type_names.update(
            item.name
            for item in self._declaration_items.values()
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias))
            and not item.scope_path
        )

    def _validate_type_params(
        self, decl: FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias
    ) -> None:
        """Raise AglScopeError if *decl* repeats a binding type-parameter name.

        A type declaration (record/enum/exception/alias) can never carry a
        receiver, so a '_' slot is rejected outright here. A function
        declaration's '_' slot is validated later, by
        :meth:`_validate_non_method_type_params`, once method classification
        tells whether it is a method's receiver prefix — until then, '_' must
        keep being skipped below so a method's repeated ``[_, _]`` receiver
        prefix does not read as a duplicate type-parameter name.
        """
        if not isinstance(decl, FuncDef):
            self._reject_type_param_wildcard(decl)
        seen: set[str] = set()
        for tp in decl.type_param_slots:
            if tp == TYPE_PARAMETER_WILDCARD:
                continue
            if tp in seen:
                raise AglScopeError(
                    f"Duplicate type parameter '{tp}' in '{decl.name}'.",
                    span=decl.span,
                )
            seen.add(tp)

    def _validate_constraints(self, decl: FuncDef) -> None:
        """Raise AglScopeError for a malformed ``{…}`` constraint block on *decl*.

        A constrained name must be one of *decl*'s type parameters
        (``type_params``, which for a method already includes its receiver's,
        see ``_receiver_type_params`` in the parser). Two constraints on the
        same parameter are rejected whenever they are the same kind or one
        implies the other (``close_constraints``), which also covers an exact
        duplicate.
        """
        if not decl.constraints:
            return
        type_params = frozenset(decl.type_params)
        if not type_params:
            raise AglScopeError(
                f"'{decl.name}' has no type parameters to constrain.",
                span=decl.constraints[0].span,
            )
        seen: dict[str, set[ConstraintKind]] = {}
        for constraint in decl.constraints:
            if constraint.param not in type_params:
                raise AglScopeError(
                    f"'{constraint.param}' is not a type parameter of '{decl.name}'.",
                    span=constraint.span,
                )
            prior_kinds = seen.setdefault(constraint.param, set())
            if any(_constraints_related(constraint.kind, kind) for kind in prior_kinds):
                raise AglScopeError(
                    f"Constraint '{constraint.kind.value} {constraint.param}' on "
                    f"'{decl.name}' is redundant with an existing constraint on "
                    f"'{constraint.param}'.",
                    span=constraint.span,
                )
            prior_kinds.add(constraint.kind)

    def _canonical_constructor_ref(self, ref: ConstructorRef) -> ConstructorRef:
        """Intern *ref* as its member declaration's canonical metadata."""
        key = (ref.owner_module_id, ref.owner_decl_node_id)
        return self._constructor_metadata_by_decl_id.setdefault(key, ref)

    def _seed_builtin_constructor_candidates(self) -> None:
        """Seed constructor candidates for built-in types (exceptions and prelude types).

        Built-in exception types (Abort, AgentParseError, …) and prelude record
        types (ExecResult, AgentRequest) are available without a source-level
        declaration.  We register each as a constructor candidate so that
        VarRef nodes that refer to them (e.g. ``Abort(message: …)``) resolve
        correctly and are placed in ``constructor_refs``.

        For builtin prelude ENUM types (e.g. ParsePolicy), we register each
        variant whose name does NOT conflict with any builtin exception type name.
        Conflicting variants (like ``ParsePolicy::Abort``) must be accessed via
        qualified syntax (e.g. ``ParsePolicy::Abort``).

        Seeded handles and member ``TypeDef`` values provide the same stable
        declaration identities as source member records.
        """
        exception_names: frozenset[str] = frozenset(BUILTIN_EXCEPTIONS)

        for exc_name, exc_type in BUILTIN_EXCEPTIONS.items():
            self._add_constructor_candidate(
                exc_name,
                self._canonical_constructor_ref(
                    ConstructorRef(
                        owner_name=exc_name,
                        owner_decl_node_id=exc_type.decl_id,
                        type_params=(),
                        owner_module_id=RESERVED_ID,
                        is_builtin=True,
                    )
                ),
            )

        for type_name, type_val in BUILTIN_PRELUDE_TYPES.items():
            if type_name in COMPATIBILITY_PRELUDE_TYPE_NAMES:
                continue
            if isinstance(type_val, EnumType):
                # Register variants that don't conflict with exception names.
                # Conflicting variants (e.g. ParsePolicy::Abort ↔ Abort exception)
                # must be used in qualified form.  Variant names come from the
                # shared prelude TypeDef literal — the handle itself carries
                # no shape data.
                typedef = BUILTIN_PRELUDE_TYPE_DEFS[type_name]
                for member in typedef.members:
                    member_def = BUILTIN_PRELUDE_MEMBER_TYPE_DEFS[member.decl_id]
                    variant_name = member.name
                    if variant_name not in exception_names:
                        self._add_constructor_candidate(
                            variant_name,
                            self._canonical_constructor_ref(
                                ConstructorRef(
                                    owner_name=variant_name,
                                    owner_decl_node_id=member.decl_id,
                                    type_params=member_def.type_params,
                                    owner_module_id=RESERVED_ID,
                                    can_match_bare_pattern=not member_def.fields,
                                    owner_path=(type_name,),
                                    is_builtin=True,
                                )
                            ),
                        )
            else:
                self._add_constructor_candidate(
                    type_name,
                    self._canonical_constructor_ref(
                        ConstructorRef(
                            owner_name=type_name,
                            owner_decl_node_id=type_val.decl_id,
                            type_params=(),
                            owner_module_id=RESERVED_ID,
                            is_builtin=True,
                        )
                    ),
                )

    def _remove_seeded_builtin_candidate(self, name: str) -> None:
        """Let a source root builtin declaration replace the seeded host constructor."""
        self._constructor_candidates[name] = [
            ref
            for ref in self._constructor_candidates.get(name, [])
            if not ref.owner_module_id.is_reserved
        ]
        key = ((), name)
        self._scoped_constructor_candidates[key] = [
            ref
            for ref in self._scoped_constructor_candidates.get(key, [])
            if not ref.owner_module_id.is_reserved
        ]

    def _add_constructor_candidate(
        self,
        ctor_key: str,
        cref: ConstructorRef,
        *,
        scope_path: ScopePath = (),
        inject_bare: bool = True,
    ) -> None:
        """Add *cref* to *ctor_key*'s candidates in *scope_path*, and bare if *inject_bare*."""
        cref = self._canonical_constructor_ref(cref)
        scoped_key = (scope_path, ctor_key)
        self._scoped_constructor_candidates[scoped_key] = self._place_candidate(
            self._scoped_constructor_candidates.get(scoped_key, []), cref
        )
        if not inject_bare:
            return
        self._constructor_candidates[ctor_key] = self._place_candidate(
            self._constructor_candidates.get(ctor_key, []), cref
        )

    @staticmethod
    def _place_candidate(
        existing: list[ConstructorRef], cref: ConstructorRef
    ) -> list[ConstructorRef]:
        """Return *existing* with *cref* joining it, or replacing a built-in in place.

        The same declaration contributed twice (over overlapping import
        routes) joins once.
        """
        if any(
            candidate.owner_module_id == cref.owner_module_id
            and candidate.owner_decl_node_id == cref.owner_decl_node_id
            for candidate in existing
        ):
            return existing
        if cref.is_builtin:
            for index, candidate in enumerate(existing):
                if candidate.is_builtin and candidate.owner_path == cref.owner_path:
                    # A standard-library declaration never displaces an
                    # override; an override displaces a standard-library or
                    # reserved one.
                    if cref.owner_module_id.owns_standard_builtins:
                        return existing
                    if candidate.owner_module_id.owns_standard_builtins:
                        return [*existing[:index], cref, *existing[index + 1 :]]
        return [*existing, cref]

    def _collect_constructor_candidates(self, owners: Mapping[ScopePath, TypeOwner]) -> None:
        """Build constructor candidates from the module's current type *owners*."""
        self._seed_builtin_constructor_candidates()
        for item, path in self._type_declarations:
            if isinstance(item, RecordDef) and item.is_builtin and not path:
                self._remove_seeded_builtin_candidate(item.name)
        for name, constructor, scope_path, bare in owned_constructors(self._module_id, owners):
            self._add_constructor_candidate(
                name, constructor, scope_path=scope_path, inject_bare=bare
            )

    def _root_declaring_candidates(self, name: str) -> tuple[ConstructorRef, ...]:
        """Candidates that declare *name* itself in this module's root.

        An enum variant's bare spelling is a convenience injection of the member
        ``Owner::variant``, which stays reachable however the bare name is
        claimed, so a variant never declares the bare name.  Record, exception,
        and alias constructors do: their constructor name *is* the root
        declaration, and nothing else reaches them.
        """
        return tuple(
            cref
            for cref in self._constructor_candidates.get(name, ())
            if not cref.owner_path and cref.owner_module_id == self._module_id
        )

    def _root_declaring_span(self, cref: ConstructorRef) -> SourceSpan:
        """The source span of the declaration *cref* was collected from.

        Every same-module constructor candidate is built from a declaration
        this pass registered, so its binding -- and with it the span to blame
        for a collision -- is always on hand.
        """
        return self._declarations[(self._module_id, cref.owner_path, cref.owner_name)].decl_span

    def _define_constructor_bindings(self) -> None:
        """Define each constructor name as a value binding in the current (root) scope.

        Collision rules:
        - Constructor-vs-constructor at the same scope: allowed (overload set).
        - A constructor declared in *another* module never collides: the two
          spellings stay separable by qualification, so declaring over a
          prelude name such as ``Retry`` is always legal.
        - A same-module constructor that declares the bare name (record,
          exception, alias) colliding with an ordinary value binding is a
          duplicate declaration, because no qualified spelling could tell the
          two apart afterwards.
        """
        scope = self._scope
        for name, crefs in self._constructor_candidates.items():
            if name in scope.bindings:
                # Path-keyed collection has already rejected same-module
                # declaration collisions. An enum variant's bare spelling yields
                # to a value binding that already claimed it; `Owner::variant`
                # still reaches the variant and pattern position still classifies it.
                continue
            parent_ref = scope.parent.lookup(name) if scope.parent is not None else None
            if parent_ref is not None and parent_ref.kind is not BinderKind.constructor_binding:
                declaring = self._root_declaring_candidates(name)
                if declaring:
                    raise DuplicateDeclarationError(
                        name, span=self._root_declaring_span(declaring[0])
                    )
                # A REPL entry's new variants remain available to pattern
                # classification, but an ordinary session binding retains its
                # expression-position meaning.
                continue
            # A local bare constructor shadows imported candidates with the
            # same spelling. Enum members are injected at the root from their
            # owner scope, so they count as local even though their canonical
            # record path is not root-level.
            rep = next(
                (candidate for candidate in crefs if candidate.owner_module_id == self._module_id),
                crefs[0],
            )
            ref = BindingRef(
                name=name,
                mutable=False,
                decl_span=SourceSpan(
                    start_line=0,
                    start_col=0,
                    end_line=0,
                    end_col=0,
                    start_offset=0,
                    end_offset=0,
                ),
                decl_node_id=rep.owner_decl_node_id,
                kind=BinderKind.constructor_binding,
                module_id=rep.owner_module_id,
            )
            scope.define(name, ref)

    def _define_function_bindings(self) -> None:
        """Define each collected function as a value binding in the current scope."""
        for name, decl in self._declared_functions.items():
            ref = BindingRef(
                name=name,
                mutable=False,
                decl_span=decl.span,
                decl_node_id=decl.node_id,
                kind=BinderKind.function_binding,
                module_id=self._module_id,
                is_builtin=decl.is_builtin,
                is_method=decl.is_method,
            )
            self._scope.define(name, ref)

    def _define_static_binding_bindings(self) -> None:
        """Define each collected root static let/var as a value binding.

        Mirrors :meth:`_collect_func_decls`/``_define_function_bindings``: a
        no-op except in a static-root module, whose root simple let/var
        declarations (:meth:`_register_static_binding_declaration`) each
        already own a ref in ``_declarations``, reused here instead of rebuilt.
        """
        if not self._is_static_root_module:
            return
        for let_item in self._root_declaration_items(LetDecl):
            name = static_binding_name(let_item)
            self._scope.define(name, self._declarations[(self._module_id, (), name)])
        for var_item in self._root_declaration_items(VarDecl):
            self._scope.define(
                var_item.name, self._declarations[(self._module_id, (), var_item.name)]
            )

    def _define_static_builtin_var_bindings(self) -> None:
        """Define each collected root ``builtin var`` as a value binding.

        Mirrors :meth:`_define_static_binding_bindings`: reuses the ref
        :meth:`_register_builtin_var_declaration` already built into
        ``_declarations`` instead of rebuilding it.
        """
        for item in self._root_declaration_items(BuiltinVarDecl):
            self._scope.define(item.name, self._declarations[(self._module_id, (), item.name)])

    def _resolve_builtin_var(self, node: BuiltinVarDecl) -> None:
        """Resolve a standard-library host-backed mutable binding.

        A ``builtin var`` may be declared by any standard-library module and
        is otherwise a member like any other, at the module root or inside a
        named scope region (placement is checked first). The typecheck pass
        reserves engine-setting names and types to ``std/config``.
        """
        if not self._is_standard_library_module:
            raise AglScopeError(
                "'builtin var' declarations are only allowed in standard-library modules "
                "(including std/config).",
                span=node.span,
            )
        ref = BindingRef(
            name=node.name,
            mutable=True,
            decl_span=node.span,
            decl_node_id=node.node_id,
            kind=BinderKind.builtin_var_binding,
            module_id=self._module_id,
        )
        self._define(node.name, ref)
        if node.default is not None:
            self._resolve_expr(node.default)

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _named_scope(self, path: ScopePath) -> Iterator[ScopeNode]:
        """Resolve a region or shorthand body in its named lexical layer."""
        named = self._scope_nodes[path]
        previous = self._scope
        self._scope = named
        try:
            yield named
        finally:
            self._scope = previous

    @contextmanager
    def _child_scope(self, node_id: int) -> Iterator[ScopeNode]:
        """Open a fresh child scope and yield it.

        Clears the root flag for its lifetime (only the program root is
        ``_at_root``) and restores on exit.
        """
        previous = self._scope
        child = ScopeNode(node_id=node_id, parent=previous)
        self._scope = child
        was_root = self._at_root
        self._at_root = False
        try:
            yield child
        finally:
            self._at_root = was_root
            self._scope = previous

    @contextmanager
    def _resolution_flags_ctx(
        self, *, in_loop: bool | None = None, in_function: bool | None = None
    ) -> Iterator[None]:
        """Resolve a region with the loop and function flags it dictates.

        A flag passed as ``None`` keeps its enclosing value.  Both are restored
        on the way out, so nested regions and the code after one see the flags
        their own position dictates.
        """
        previous = (self._in_loop, self._in_function)
        if in_loop is not None:
            self._in_loop = in_loop
        if in_function is not None:
            self._in_function = in_function
        try:
            yield
        finally:
            self._in_loop, self._in_function = previous

    def _define(self, name: str, ref: BindingRef) -> None:
        """Define *name* in the current scope; error on redeclaration.

        A named scope's own body — reached while resolving a scope region or
        the pushed layer of a root-position binder-path shorthand — is a
        members layer, not a lexical bindings layer: a binder resolved there
        is registered into ``members`` instead. Only a let, var, builtin
        var, or builtin var ever reaches this branch (a
        parameter, catch binder, or loop variable always resolves inside a
        fresh child scope with an empty path instead), and each one's own
        pre-pass (``_register_static_binding_declaration``/
        ``_register_builtin_var_declaration``) already claimed *name*
        in ``_scope_entity_kinds`` and raised there on a same-path collision
        with a ``def``, a type, or a nested scope, regardless of textual
        order -- so this call always installs its own pre-claimed member,
        never a fresh or colliding one.

        At the true root, a declaration may claim the bare spelling of a
        constructor declared in another module (e.g. the prelude
        ``Retry``/``ExecResult`` names) or of a same-module enum variant,
        since ``Owner::variant`` and a qualified module path still reach
        those.  A same-module record, exception, or alias constructor
        declares the bare name itself, so it collides.
        """
        scope = self._scope
        if scope.scope_path:
            # The pre-pass already claimed this key for this exact binder
            # (see the docstring); install the member it pre-claimed.
            scope.register_member(name, replace(ref, scope_path=scope.scope_path))
            return
        existing = scope.bindings.get(name)
        if existing is not None and existing.decl_node_id == ref.decl_node_id:
            # Same self-registration case as above, for a root-position binder
            # a static-root module already defined via
            # ``_define_static_binding_bindings``.
            scope.define(name, ref)
            return
        if existing is not None and (
            existing.kind is not BinderKind.constructor_binding
            or self._root_declaring_candidates(name)
        ):
            raise DuplicateDeclarationError(name, span=ref.decl_span)
        scope.define(name, ref)

    def _check_not_reserved(self, name: str, span: SourceSpan) -> None:
        """Raise if *name* is a built-in contextual name."""
        if name in _RESERVED_NAMES:
            raise AglScopeError(
                f"'{name}' is a reserved contextual keyword and cannot be "
                "used as a binding or parameter name.",
                span=span,
            )

    # ------------------------------------------------------------------
    # Block item resolution
    # ------------------------------------------------------------------

    def _validate_item_placement(
        self, items: tuple[Item, ...], *, entry_body: bool = False
    ) -> None:
        """Reject misplaced headers and root statements in a module's own item sequences.

        The module root and every scope region: ``import`` and ``export``
        precede every other item of the same sequence (a region is one item of
        its enclosing sequence and has its own header), and a static root holds
        no assignment or bare expression. The REPL is the sole incremental host
        that admits root statements. A synthetic entry's body holds what the
        source wrote at its root after an executable item, so a header there
        breaks the ordering rule. Each item's nested blocks follow, in source
        order (:func:`_nested_misplacement`).
        """
        seen_non_header = False
        for item in items:
            if isinstance(item, UseDecl):
                continue
            if isinstance(item, (ImportDecl, ExportDecl)):
                if seen_non_header or entry_body:
                    raise AglScopeError(
                        "Import and export declarations must appear before any other "
                        "declarations in a module or scope region.",
                        span=item.span,
                    )
                continue
            seen_non_header = True
            if entry_body:
                self._raise_nested_misplacement(item)
                continue
            if isinstance(item, ScopeRegion):
                self._validate_item_placement(item.items)
            elif isinstance(item, FuncDef) and item.is_synthetic and isinstance(item.body, Block):
                self._validate_item_placement(item.body.items, entry_body=True)
            elif not self._allow_root_statements and not isinstance(item, _ROOT_DECLARATION_ITEMS):
                statement = (
                    "Assignment statements are not allowed at a static module root."
                    if isinstance(item, AssignStmt)
                    else "Bare expressions are not allowed at a static module root."
                )
                raise AglScopeError(
                    static_root_message(
                        statement,
                        subject="statements",
                        file_backed=self._origin_path is not None,
                        declares_program_entry=self._declares_program_entry,
                    ),
                    span=item.span,
                )
            else:
                self._raise_nested_misplacement(item)

    @staticmethod
    def _raise_nested_misplacement(item: Item) -> None:
        """Raise for the first item, in source order, that a nested block of *item* may not hold."""
        errors: dict[int, AglScopeError] = {}

        def check(node: object) -> None:
            if isinstance(node, Block):
                errors.update(
                    (nested.span.start_offset, error)
                    for nested in node.items
                    if (error := _nested_misplacement(nested)) is not None
                )

        walk(item, check)
        if errors:
            raise errors[min(errors)]

    def _resolve_block_items(self, items: tuple[Item, ...]) -> None:
        """Resolve items in order; each binder adds to the current scope.

        This is the core sequencing logic.  Binders (``LetDecl``, ``VarDecl``,
        ``AssignStmt``) and declarations (``FuncDef``, etc.) that
        are not pure expressions are handled first; everything else is treated
        as an expression item.

        Every item sequence, nested blocks included, was checked for
        placement before anything resolved (:meth:`_validate_item_placement`).
        """
        for item in items:
            if isinstance(item, UseDecl):
                # Resolved with the headers (``_resolve_headers``); what its
                # tail or hiding names under an own target is checked here.
                self._validate_own_use_selection(item)
                continue
            if isinstance(item, (ImportDecl, ExportDecl, InfixDecl)):
                # The program module-system pass processes imports/exports; a
                # region-scoped import's contribution is resolved with the
                # headers, and infix declarations with the parse.
                continue
            # Named declarations (def/record/enum/exception/type, and a
            # scoped let/var binder) validate their own whole subtree,
            # including any nested blocks, in their own handlers after
            # entering the right lexical scope. A scope region keeps
            # `_at_root` set and validates its own items one by one through
            # this same walk: the parser gives a region's `let`/`var` items the
            # region's scope path, so they reach the scoped-binder handler
            # like a root `let S::x`. Every other root item is validated here; a
            # nested block's own item is already covered by the walk of its
            # enclosing item or declaration.
            skip_item_validation = isinstance(
                item, (ScopeRegion, FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias)
            ) or (isinstance(item, (LetDecl, VarDecl)) and item.scope_path)
            if self._at_root and not skip_item_validation:
                self._validate_qualifier_chains(item)
            if isinstance(item, ScopeRegion):
                self._resolve_scope_region(item)
            elif isinstance(item, FuncDef):
                self._resolve_funcdef(item)
            elif isinstance(item, BuiltinVarDecl):
                # Placement in the canonical standard-library module is
                # enforced by the declaration handler.
                self._resolve_builtin_var(item)
            elif isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                self._resolve_type_decl(item)
            elif isinstance(item, LetDecl):
                self._resolve_let(item)
            elif isinstance(item, VarDecl):
                self._resolve_var(item)
            elif isinstance(item, AssignStmt):
                self._resolve_assign(item)
            else:
                # Pure expression item (Expr union).
                self._resolve_expr(item)

    def _resolve_root_items(self, items: tuple[Item, ...]) -> None:
        """Resolve the module root's *items*, an inline entry's statements among them.

        The inline wrap (``parser/wrap.py``) moves the root statements of a
        source without a ``program def`` into a synthetic entry after the root
        declarations. They still resolve at their own source positions among
        those declarations, in the root scope, with a function body's flags:
        the text reads -- and reports its first error -- exactly as written,
        and a statement's ``let``/``var`` is a root binding ``::name`` reaches.
        """
        statements: tuple[Item, ...] = ()
        declarations: list[Item] = []
        for item in items:
            if isinstance(item, FuncDef) and item.is_synthetic and isinstance(item.body, Block):
                statements = item.body.items
            else:
                declarations.append(item)
        statement_ids = {id(statement) for statement in statements}
        for item in sorted((*declarations, *statements), key=_start_offset):
            if id(item) not in statement_ids:
                self._resolve_block_items((item,))
                continue
            with self._resolution_flags_ctx(in_loop=False, in_function=True):
                self._resolve_block_items((item,))

    def _resolve_headers(self, items: tuple[Item, ...]) -> None:
        """Contribute the ``use`` and region-scoped ``import`` headers of *items* and its regions.

        A header contributes bindings at once; a constructor that depends on
        type owners is deferred to ``resolve``. Every import here is in its
        sequence's header: placement was checked first.
        """
        regional_exposures = self._regional_import_exposures(items)
        for item in items:
            if isinstance(item, UseDecl):
                self._resolve_use_decl(item)
            elif isinstance(item, ImportDecl) and item.scope_path:
                self._contribute_regional_import_bare(item, exposures=regional_exposures)
            elif isinstance(item, ScopeRegion):
                path = self._scope.scope_path + (item.segment.name,)
                with self._named_scope(path):
                    self._resolve_headers(item.items)

    # ------------------------------------------------------------------
    # Declaration handlers
    # ------------------------------------------------------------------

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

    def _regional_import_exposures(
        self, items: tuple[Item, ...]
    ) -> dict[NameAtom, frozenset[QName]]:
        """Union the bare atoms every region-scoped import in *items* exposes.

        A region's imports form one surface, however many declarations spell
        it: whether an enum member's bare spelling yields to a same-named
        record or exception must not depend on which import came first.
        """
        exposures: dict[NameAtom, frozenset[QName]] = {}
        for item in items:
            if not isinstance(item, ImportDecl) or not item.scope_path:
                continue
            for atom, qnames in self._import_env.decl_bare.get(item.node_id, {}).items():
                exposures[atom] = exposures.get(atom, frozenset()) | frozenset(qnames)
        return exposures

    def _contribute_regional_import_bare(
        self, decl: ImportDecl, *, exposures: Mapping[NameAtom, Collection[QName]]
    ) -> None:
        """Contribute a region-scoped import tail to its own region only.

        ``build_import_env`` keeps a tail's bare atoms per declaration in
        ``ImportEnv.decl_bare`` rather than merging them into the root
        ``unqualified`` table. This snapshots them onto the current
        ``ScopeNode`` while the declaration's qualified routes remain
        module-wide through ``self._import_env.contributions``.

        A wildcard candidate may draw one atom from several modules (e.g.
        two sibling modules both exporting ``alpha``); every origin is
        contributed, deferring an eventual clash to the atom's first use --
        the same policy ``unqualified`` already applies at the module root --
        rather than raising here.
        """
        bare = self._import_env.decl_bare.get(decl.node_id, {})
        scope = self._scope
        for atom, qnames in bare.items():
            for qname in qnames:
                scope.contribute_bare(
                    atom, self._cross_module_binding_ref(qname), ContributionLayer.IMPORTED
                )
                self._deferred_constructors.append(
                    partial(
                        self._contribute_bare_constructor,
                        scope,
                        atom,
                        qname,
                        ContributionLayer.IMPORTED,
                    )
                )
                if isinstance(atom, str):
                    self._deferred_constructors.append(
                        partial(
                            self._contribute_regional_enum_variants,
                            scope,
                            qname,
                            decl.span,
                            exposures,
                        )
                    )

    def _contribute_regional_enum_variants(
        self,
        scope: ScopeNode,
        qname: QName,
        span: SourceSpan,
        exposures: Mapping[NameAtom, Collection[QName]],
    ) -> None:
        """Expand a bare-exposed enum type into its own bare variants, region-scoped.

        A bare enum *type* name alone does not make its variants callable or
        matchable -- ``_build_cross_module_constructor_candidates`` performs
        the same expansion module-wide, from a root-position bare exposure.
        Mirroring it here covers the scoped case, whose bare exposure never
        reaches that module-wide table. Deferred, since a referenced member
        reads the type owners.
        """
        selected_qnames = frozenset(qname for qnames in exposures.values() for qname in qnames)
        for atom, constructor, path in self._enum_variant_members(qname):
            # An inline member the tail hides, or whose spelling a same-named
            # record or exception exposed bare already owns, stays unexposed.
            if constructor.inline_enum_owner_decl_node_id is not None and (
                (qname[0], _bare_atom(path)) not in selected_qnames
                or declares_bare_constructor(exposures.get(atom, ()), self._all_public_types)
            ):
                continue
            scope.contribute_bare(
                atom, self._variant_binding_ref(constructor, span), ContributionLayer.IMPORTED
            )
            scope.contribute_bare_constructor(atom, constructor, ContributionLayer.IMPORTED)

    def _contribute_bare_constructor(
        self, scope: ScopeNode, atom: NameAtom, qname: QName, layer: ContributionLayer
    ) -> ConstructorRef | None:
        """Contribute the constructor imported *qname* names bare to *scope* through *layer*."""
        constructor = self._cross_module_constructor(qname)
        if constructor is not None:
            scope.contribute_bare_constructor(atom, constructor, layer)
        return constructor

    def _validate_retained_imported_use_routes(self) -> None:
        """Reject a retained imported use whose current import replacement hides its route."""
        scope = self._root_scope.parent
        while scope is not None:
            for contribution in scope.imported_use_contributions:
                if contribution.target.wildcard_facade_origin_node_id is not None:
                    continue
                for module, path in contribution.target.imported_routes:
                    if not path:
                        continue
                    if module not in self._import_env.contributions:
                        continue
                    if (module, path) not in self._import_env.scope_origins_by_route:
                        raise UnknownQualifierError(
                            spell_declaration(module, path),
                            span=self._import_env.decl_spans[module],
                            repair=MissRepair.IMPORT_MODULE,
                        )
            scope = scope.parent

    def _resolve_use_target(self, decl: UseDecl) -> _UseTargetResolution:
        """Gather every local and imported route reachable through a use target."""
        if decl.alias is not None and len(decl.target) >= 2:
            parent_decl = replace(decl, target=decl.target[:-1], alias=None)
            parent = self._resolve_use_target(parent_decl)
            member = decl.target[-1].name
            local_member = any(
                (*local, member) not in self._scope_nodes
                and (
                    member in self._scope_nodes[local].members
                    or (*local, member) in self._ordered_binding_paths
                )
                for tier in parent.local
                for local in tier
            )
            imported_member = any(
                (qname := members.get(member)) is not None
                and qname not in self._cross_module_type_owners
                for _route, members in parent.imported
            )
            if local_member or imported_member:
                segment = decl.target[-1]
                selected = ImportItem(
                    name=segment.name,
                    rename=decl.alias,
                    span=segment.span,
                    node_id=segment.node_id,
                )
                return self._resolve_use_target(
                    replace(decl, target=decl.target[:-1], tail=(selected,), alias=None)
                )
        target = tuple(segment.name for segment in decl.target)
        route = () if decl.current_module else tuple(target[0].split("/"))
        route_target = () if decl.current_module else target[1:]
        direct_candidates = (
            ()
            if decl.current_module
            else qualifier_members(self._import_env, route, anchored=decl.anchored)
        )
        direct_scope_paths = dict(
            ()
            if decl.current_module
            else qualifier_scope_paths(self._import_env, route, anchored=decl.anchored)
        )
        direct_imports: tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...] = tuple(
            ((module, route_target), relative_members)
            for module, members in direct_candidates
            if (
                relative_members := self._relative_use_import_members(
                    members,
                    route_target,
                    target_exists=bool(route_target)
                    and _bare_atom(route_target) in direct_scope_paths.get(module, frozenset()),
                )
            )
            is not None
        )
        direct_import_scope_routes = {
            imported_route: self._relative_use_import_scope_routes(
                direct_scope_paths[imported_route[0]], imported_route
            )
            for imported_route, _members in direct_imports
        }
        bare_imports = () if decl.anchored else self._bare_use_import_targets(target)
        used_imports = () if decl.anchored else self._used_import_targets(target)
        return _UseTargetResolution(
            declaration=decl,
            target=target,
            local=self._use_local_tiers(decl, target),
            route=route,
            direct_candidates=direct_candidates,
            direct_import_scope_routes=direct_import_scope_routes,
            imported=self._merge_use_import_targets(direct_imports, bare_imports, used_imports),
        )

    def _alias_target(self, qname: QName) -> tuple[TypeOwner, QName] | None:
        """Return alias *qname*'s owner and the nominal declaration its alias chain ends at.

        An alias segment stands for its target's path, so a ``use`` of an
        alias reaches what its target declares.
        """
        owner = self._type_owners.owner(qname)
        if owner is None or owner.target is None:
            return None
        target = owner.target.qname
        reached = self._type_owners.owner(target)
        while reached is not None and reached.target is not None:
            target = reached.target.qname
            reached = self._type_owners.owner(target)
        return owner, target

    def _use_target_aliases(
        self,
        decl: UseDecl,
        target: ScopePath,
        imported: tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...],
    ) -> list[QName]:
        """Return the type aliases *decl*'s target names: an own one, else every imported one.

        An own declaration wins its path, found at the lexical bases a local
        target is; imported routes that end at an alias each name theirs.
        """
        own = next(
            (
                base + target
                for base in self._use_local_bases(decl)
                if base + target in self._alias_receiver_paths()
            ),
            None,
        )
        if own is not None:
            return [(self._module_id, _bare_atom(own))]
        return [
            qname
            for (module, path), _members in imported
            if path
            and isinstance(
                self._all_public_types.get(qname := (module, _bare_atom(path))), TypeAlias
            )
        ]

    def _resolve_use_decl(self, decl: UseDecl) -> None:
        """Inject the members every scope a use target reaches exposes bare.

        An own scope and imported scopes of the target's path combine: each
        contributes what it reaches, so a path the own module declares selects
        the own declaration and a path two imports declare is ambiguous where
        it is used. A tail or ``hiding`` names paths any of them reach.
        """
        resolved_target = self._resolve_use_target(decl)
        decl = resolved_target.declaration
        target = resolved_target.target
        local = resolved_target.local
        route = resolved_target.route
        direct_candidates = resolved_target.direct_candidates
        direct_import_scope_routes = resolved_target.direct_import_scope_routes
        imported = resolved_target.imported
        aliases = self._use_target_aliases(decl, target, imported)
        if not local and not imported and not aliases:
            raise UnknownQualifierError(
                _use_target_spelling(decl), span=decl.span, repair=MissRepair.IMPORT_MODULE
            )
        facade_declarations = (
            self._import_env.facade_aliases.get(route[0], {}) if len(route) == 1 else {}
        )
        facade_origin_node_id: int | None = None
        if not decl.anchored and len(route) == 1:
            direct_modules = frozenset(module for module, _members in direct_candidates)
            matching_origins = tuple(
                origin
                for origin, modules in facade_declarations.items()
                if modules == direct_modules
            )
            if len(matching_origins) == 1:
                facade_origin_node_id = matching_origins[0]
        resolved_identity = ResolvedUseTarget(
            local_paths=tuple(path for tier in local for path in tier),
            imported_routes=tuple(route for route, _members in imported),
            aliases=tuple(aliases),
            wildcard_facade_origin_node_id=facade_origin_node_id,
        )
        self._superseded_use_targets.add(resolved_identity)
        self._current_use_declaration_ids.add(decl.node_id)
        self._use_targets[decl.node_id] = resolved_identity

        def scope_routes_for(imported_route: BareRoute) -> Mapping[NameAtom, frozenset[BareRoute]]:
            """Keep selected provenance unless replay requires the target's full route."""
            direct = direct_import_scope_routes.get(imported_route)
            if direct is not None:
                return direct
            selected = {
                **self._bare_use_import_scope_routes(target, imported_route),
                **self._used_import_scope_routes(target, imported_route),
            }
            return selected

        surfaces = tuple(
            (members, scope_routes_for(imported_route)) for imported_route, members in imported
        )
        imported_surface: dict[NameAtom, None] = {
            atom: None for members, scope_routes in surfaces for atom in (*scope_routes, *members)
        }
        # An alias target's members are its target's, known once every
        # module's headers are prepared; the tail and hiding wait for them.
        if aliases:
            self._deferred_constructors.append(
                partial(
                    self._contribute_aliased_use_members,
                    decl,
                    aliases,
                    imported_surface,
                    self._scope,
                    local=bool(local),
                )
            )
        elif not local:
            self._select_use_members(decl, imported_surface)
        if local:
            # The own scopes' members are known once the whole entry is
            # declared, so the tail and hiding are checked where the walk
            # reaches the use (``_validate_own_use_selection``).
            self._imported_use_surfaces[decl.node_id] = imported_surface
            for index, tier in enumerate(local):
                outranked_by = tuple(path for earlier in local[:index] for path in earlier)
                for path in tier:
                    self._scope.contribute_local_use(
                        LocalUseContribution(
                            declaration=decl,
                            source=self._scope_nodes[path],
                            target=resolved_identity,
                            outranked_by=outranked_by,
                        )
                    )
        for members, scope_routes in surfaces:
            self._contribute_use_members(decl, members, scope_routes, self._scope)

    def _contribute_aliased_use_members(
        self,
        decl: UseDecl,
        aliases: list[QName],
        imported_surface: dict[NameAtom, None],
        scope: ScopeNode,
        *,
        local: bool,
    ) -> None:
        """Expose in *scope* the members *decl*'s type-alias targets declare.

        An alias segment stands for its target's path, so a ``use`` of an
        alias reaches what its target declares: an own target as a local
        scope, an imported one as imported members. Read once every module's
        headers are prepared, as the type-owner index requires. With local
        scopes reached, the tail and hiding are checked where the walk reaches it.
        """
        surface = dict(imported_surface)
        members: dict[NameAtom, QName] = {}
        own_paths: list[ScopePath] = []
        for alias in aliases:
            found = self._alias_target(alias)
            if found is None:
                continue
            owner, (target_module, target_atom) = found
            target_path = _bare_path(target_atom)
            surface.update((name, None) for name in owner.members)
            if target_module == self._module_id:
                own_paths.append(target_path)
            else:
                members.update(
                    (name, (target_module, _bare_atom((*target_path, name))))
                    for name in owner.members
                )
        if local or own_paths:
            self._imported_use_surfaces[decl.node_id] = surface
        else:
            self._select_use_members(decl, surface)
        for path in own_paths:
            scope.contribute_local_use(
                LocalUseContribution(
                    declaration=decl,
                    source=self._scope_nodes[path],
                    target=self._use_targets[decl.node_id],
                )
            )
        self._contribute_use_members(decl, members, {}, scope)

    def _relative_use_import_members(
        self,
        members: Mapping[NameAtom, QName],
        target: ScopePath,
        *,
        target_exists: bool = False,
    ) -> dict[NameAtom, QName] | None:
        """Return an existing target's public subtree under target-relative paths."""
        relative_members: dict[NameAtom, QName] = {}
        exists = not target or target_exists
        for atom, qname in members.items():
            path = _bare_path(atom)
            if path == target:
                exists = exists or qname in self._cross_module_type_owners
            elif path[: len(target)] == target:
                exists = True
                relative_members[_bare_atom(path[len(target) :])] = qname
        return relative_members if exists else None

    @staticmethod
    def _relative_use_import_scope_routes(
        scope_paths: Collection[NameAtom], imported_route: BareRoute
    ) -> dict[NameAtom, frozenset[BareRoute]]:
        """Return a target's scope identities under target-relative spellings."""
        module, target = imported_route
        relative_routes: dict[NameAtom, frozenset[BareRoute]] = {(): frozenset({imported_route})}
        for atom in scope_paths:
            relative = _relative_under(atom, target)
            if relative is None:
                continue
            relative_routes[_bare_atom(relative)] = frozenset({(module, _bare_path(atom))})
        return relative_routes

    def _bare_scope_route_contributions(
        self,
    ) -> tuple[Mapping[NameAtom, frozenset[BareRoute]], ...]:
        """Return the namespace-only bare contributions reachable from here."""
        return (
            self._import_env.unqualified_scope_routes,
            *(
                routes
                for _node_id, routes in self._reachable_decl_contributions(
                    self._import_env.decl_bare_scope_routes, self._scope.scope_path
                )
            ),
        )

    def _bare_use_import_targets(
        self, target: ScopePath
    ) -> tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...]:
        """Find scope subtrees exposed by an already-bare import tail."""
        members_by_route: dict[BareRoute, dict[NameAtom, QName]] = {}
        bare_contributions = (
            (self._import_env.unqualified, self._import_env.unqualified_routes),
            *(
                (bare, self._import_env.decl_bare_routes.get(node_id, {}))
                for node_id, bare in self._reachable_decl_contributions(
                    self._import_env.decl_bare, self._scope.scope_path
                )
            ),
        )
        for contributions, provenances in bare_contributions:
            for atom, routes in provenances.items():
                relative = _relative_under(atom, target)
                if relative is None:
                    continue
                for module, source in routes:
                    qnames = contributions.get(atom, frozenset())
                    origin = self._import_env.contributions[module].members.get(_bare_atom(source))
                    selected = (origin,) if origin in qnames else ()
                    if not selected or (
                        not relative
                        and not any(qname in self._cross_module_type_owners for qname in selected)
                    ):
                        continue
                    members = members_by_route.setdefault(
                        (module, _route_root(source, relative)), {}
                    )
                    if not relative:
                        continue
                    exposed = _bare_atom(relative)
                    for qname in selected:
                        members.setdefault(exposed, qname)

        scope_routes = self._bare_scope_route_contributions()
        for provenances in scope_routes:
            for atom, routes in provenances.items():
                relative = _relative_under(atom, target)
                if relative is None:
                    continue
                for module, source in routes:
                    members_by_route.setdefault((module, _route_root(source, relative)), {})
        return tuple(
            (imported_route, members_by_route[imported_route])
            for imported_route in sorted(members_by_route, key=_bare_route_sort_key)
        )

    def _bare_use_import_scope_routes(
        self, target: ScopePath, imported_route: BareRoute
    ) -> dict[NameAtom, frozenset[BareRoute]]:
        """Return scope identities supplied by raw bare import contributions."""
        result: dict[NameAtom, set[BareRoute]] = {}
        provenances = self._bare_scope_route_contributions()
        for routes_by_atom in provenances:
            for atom, routes in routes_by_atom.items():
                relative = _relative_under(atom, target)
                if relative is None:
                    continue
                for module, source in routes:
                    if (module, _route_root(source, relative)) == imported_route:
                        result.setdefault(_bare_atom(relative), set()).add((module, source))
        return {atom: frozenset(routes) for atom, routes in result.items()}

    def _used_import_targets(
        self, target: ScopePath
    ) -> tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...]:
        """Find imported scopes exposed by an earlier ``use`` in the nearest region."""
        layer: ScopeNode | None = self._scope
        exposed_target = _bare_atom(target)
        while layer is not None:
            candidates: list[tuple[BareRoute, Mapping[NameAtom, QName]]] = []
            for contribution in layer.imported_use_contributions:
                imported_routes = contribution.scope_routes.get(exposed_target)
                if imported_routes is None:
                    continue
                members = {
                    _bare_atom(path[len(target) :]): qname
                    for atom, qname in contribution.members.items()
                    if (path := _bare_path(atom))[: len(target)] == target
                    and len(path) > len(target)
                }
                candidates.extend((imported_route, members) for imported_route in imported_routes)
            if candidates:
                return self._merge_use_import_targets(tuple(candidates))
            layer = layer.parent
        return ()

    def _used_import_scope_routes(
        self, target: ScopePath, imported_route: BareRoute
    ) -> dict[NameAtom, frozenset[BareRoute]]:
        """Return relative identities from the earlier use that exposed *target*."""
        layer: ScopeNode | None = self._scope
        exposed_target = _bare_atom(target)
        while layer is not None:
            contributions = layer.imported_use_contributions
            matching = [
                contribution
                for contribution in contributions
                if imported_route in contribution.scope_routes.get(exposed_target, frozenset())
            ]
            if matching:
                routes: dict[NameAtom, set[BareRoute]] = {}
                for contribution in matching:
                    for atom, sources in contribution.scope_routes.items():
                        relative = _relative_under(atom, target)
                        if relative is not None:
                            routes.setdefault(_bare_atom(relative), set()).update(sources)
                return {atom: frozenset(sources) for atom, sources in routes.items()}
            layer = layer.parent
        return {}

    def _merge_use_import_targets(
        self, *targets: tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...]
    ) -> tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...]:
        """Merge scope routes that retain the same defining origins."""
        grouped: dict[frozenset[QName], tuple[BareRoute, dict[NameAtom, QName]]] = {}
        candidates = sorted(
            (candidate for routes in targets for candidate in routes), key=_keyed_bare_route
        )
        for imported_route, members in candidates:
            origins = self._scope_route_origins(imported_route)
            _representative, merged = grouped.setdefault(origins, (imported_route, {}))
            for atom, qname in members.items():
                merged.setdefault(atom, qname)

        return tuple(sorted(grouped.values(), key=_keyed_bare_route))

    def _scope_route_origins(self, route: BareRoute) -> ScopeOrigins:
        """Return the declarations scope route *route* reaches, through any number of re-exports."""
        module, path = route
        return self._import_env.scope_origins_by_route.get(
            route, frozenset({(module, _bare_atom(path))})
        )

    def _use_local_tiers(
        self, decl: UseDecl, target: ScopePath
    ) -> tuple[tuple[ScopePath, ...], ...]:
        """Return the own scopes *decl*'s target reaches, in tiers ordered as their paths win.

        Each lexical step, innermost first, offers the own scope declared at
        its path, then the own scopes an earlier local ``use`` anchored there
        exposes as *target*. A path beneath the target is read from the first
        tier declaring it; the scopes of one tier compete only where one path
        is used. A ``::`` target names the module root's own scope; any other
        anchored one names a module route, never an own path.
        """
        if decl.current_module:
            return ((target,),) if target in self._scope_nodes else ()
        if decl.anchored:
            return ()
        tiers: list[tuple[ScopePath, ...]] = []
        for layer in self._layer_chain(self._scope):
            reached = {path for tier in tiers for path in tier}
            own = layer.scope_path + target
            if own in self._scope_nodes and own not in reached:
                tiers.append((own,))
                reached.add(own)
            exposed = tuple(
                path
                for path in self._use_contributed_local_targets(layer, target)
                if path not in reached
            )
            if exposed:
                tiers.append(exposed)
        return tuple(tiers)

    def _use_local_bases(self, decl: UseDecl) -> list[ScopePath]:
        """Return the own scope paths *decl*'s target is tried under, innermost first.

        A ``::`` target names the module root; any other anchored one names
        a module route, never an own path.
        """
        if decl.current_module:
            return [()]
        return [] if decl.anchored else self._lexical_scope_bases()

    def _lexical_scope_bases(self) -> list[ScopePath]:
        """Return the scope paths enclosing the current one, innermost first."""
        bases: list[ScopePath] = []
        scope: ScopeNode | None = self._scope
        while scope is not None:
            if scope.scope_path not in bases:
                bases.append(scope.scope_path)
            scope = scope.parent
        return bases

    def _local_contribution_superseded(self, contribution: LocalUseContribution) -> bool:
        """Whether *contribution*'s target was re-targeted earlier in this entry.

        A retained contribution's target can be re-targeted by a fresh ``use``
        declaration resolved anywhere in the current entry (``_resolve_use_decl``
        adds every target it resolves to ``_superseded_use_targets``, keyed by
        target rather than declaration). ``_current_use_declaration_ids``
        exempts the declarations resolved so far in *this* entry, so a
        contribution is never treated as stale by its own declaration's
        target -- only a target some OTHER (necessarily earlier) declaration
        introduced counts as superseded. In a batch (non-REPL) run every use
        decl belongs to the single resolution pass, so every contribution's
        declaration is always in ``_current_use_declaration_ids`` and this is
        always ``False``; only a REPL entry can see a retained contribution
        from a prior entry whose declaration id was never added here.

        The one definition of this predicate, shared by the two live
        consumers below and by :meth:`_refresh_local_use_contributions`'s
        bare-contribution snapshot, so a superseded contribution's members
        never leak into a static ``bare_contributions`` read that (unlike the
        live consumers) applies no filter of its own.
        """
        return (
            contribution.target in self._superseded_use_targets
            and contribution.declaration.node_id not in self._current_use_declaration_ids
        )

    def _use_contributed_local_targets(
        self, layer: ScopeNode, target: ScopePath
    ) -> tuple[ScopePath, ...]:
        """Return every own scope an earlier local ``use`` of *layer* exposes as *target*."""
        candidates: set[ScopePath] = set()
        for local_contribution in layer.local_use_contributions:
            if self._local_contribution_superseded(local_contribution):
                continue
            for exposed, source in self._local_use_exposures(local_contribution):
                exposed_path = _bare_path(exposed)
                if exposed_path[: len(target)] != target:
                    continue
                source_path = (
                    source.path
                    if isinstance(source, _LocalScopeRoute)
                    else (*source.scope_path, source.name)
                )
                trailing_length = len(exposed_path) - len(target)
                candidate = source_path if trailing_length == 0 else source_path[:-trailing_length]
                if candidate in self._scope_nodes:
                    candidates.add(candidate)
        return tuple(sorted(candidates))

    def _local_use_exposures(
        self, contribution: LocalUseContribution
    ) -> list[tuple[NameAtom, BindingRef | _LocalScopeRoute]]:
        """Expand one local ``use`` into every bare spelling it exposes.

        The single definition of the select-then-rename protocol: a use's tail
        selection and its additive renames both draw on the same snapshot of
        the target subtree, so every consumer sees one consistent surface. A
        declaration path an outranking target declares is read from there.
        """
        declaration = contribution.declaration
        outranking = {
            atom
            for path in contribution.outranked_by
            for atom, member in self._local_use_members(path).items()
            if isinstance(member, BindingRef)
        }
        source_members = {
            atom: member
            for atom, member in self._local_use_members(contribution.source.scope_path).items()
            if not isinstance(member, BindingRef) or atom not in outranking
        }
        selected = self._select_use_members(declaration, source_members, validate=False)
        return [*selected.items(), *self._use_renamed_members(declaration, source_members)]

    def _validate_own_use_selection(self, decl: UseDecl) -> None:
        """Check that *decl*'s tail and hiding name paths its targets reach, own ones included.

        Every own member is declared before the walk, but a non-static
        module's scoped let/var is installed only when the walk reaches it,
        so the declaration registry supplies its path.
        """
        surface = self._imported_use_surfaces.get(decl.node_id)
        if surface is None:
            return
        members = dict(surface)
        for target in self._use_targets[decl.node_id].local_paths:
            members.update((atom, None) for atom in self._local_use_members(target))
            members.update(
                (_bare_atom(relative), None)
                for module, path, name in self._scope_entity_kinds
                if module == self._module_id
                and (relative := _relative_under((*path, name), target))
            )
        self._select_use_members(decl, members)

    def _local_use_members(
        self, target: ScopePath
    ) -> dict[NameAtom, BindingRef | _LocalScopeRoute]:
        """Expose one local scope subtree and its nested scope identities."""
        members: dict[NameAtom, BindingRef | _LocalScopeRoute] = {}
        for path, scope in self._scope_nodes.items():
            relative = _relative_under(path, target)
            if relative is None:
                continue
            members.setdefault(_bare_atom(relative), _LocalScopeRoute(path))
            for name, ref in scope.members.items():
                members[_bare_atom((*relative, name))] = ref
        return members

    def _refresh_local_use_contributions(self) -> None:
        """Re-snapshot the uses of every layer this entry built.

        A REPL session parent layer holds only earlier entries' uses, and REPL
        promotion reads only the layers this entry built.
        """
        for layer in self._scope_nodes.values():
            self._refresh_layer_contributions(layer)

    def _refresh_layer_contributions(self, layer: ScopeNode) -> None:
        """Re-snapshot *layer*'s uses against their live derivation.

        Re-snapshotting is not a validity check -- a use's tail and hiding
        are checked where it is written (:meth:`_validate_own_use_selection`)
        -- so it runs for every use regardless of which entry declared it.
        """
        layer.imported_use_contributions = [
            self._refresh_imported_use(imported_contribution)
            for imported_contribution in layer.imported_use_contributions
        ]
        rebuilt: list[LocalUseContribution] = []
        for local_contribution in layer.local_use_contributions:
            exposures = self._local_use_exposures(local_contribution)
            if self._local_contribution_superseded(local_contribution):
                # Dropped from the rebuilt list rather than kept with an
                # empty snapshot: a named (non-root) scope re-seeds its node
                # from the retained one on every later entry
                # (``_build_scope_nodes``), so a superseded contribution kept
                # around would keep being copied forward and, once some later
                # entry declares no competing ``use`` of its own, would read
                # as live again -- dropping it here is what makes
                # supersession permanent rather than only good for the one
                # entry that introduced it.
                continue
            bindings, constructors = self._exposure_snapshot(exposures)
            rebuilt.append(
                replace(local_contribution, bindings=bindings, constructors=constructors)
            )
        layer.local_use_contributions = rebuilt

    def _exposure_snapshot(
        self, exposures: list[tuple[NameAtom, BindingRef | _LocalScopeRoute]]
    ) -> tuple[
        Mapping[NameAtom, frozenset[BindingRef]], Mapping[NameAtom, frozenset[ConstructorRef]]
    ]:
        """Return one local use's snapshot of its live exposures and their constructors."""
        bindings: dict[NameAtom, set[BindingRef]] = {}
        constructors: dict[NameAtom, set[ConstructorRef]] = {}
        for exposed, source in exposures:
            if not isinstance(source, BindingRef):
                continue
            bindings.setdefault(exposed, set()).add(source)
            for constructor in self._declaring_constructor_candidates(source.name, source):
                constructors.setdefault(exposed, set()).add(constructor)
        return (
            {atom: frozenset(refs) for atom, refs in bindings.items()},
            {atom: frozenset(refs) for atom, refs in constructors.items()},
        )

    def _contribute_use_members(
        self,
        decl: UseDecl,
        members: Mapping[NameAtom, QName],
        scope_routes: Mapping[NameAtom, frozenset[BareRoute]],
        scope: ScopeNode,
    ) -> None:
        """Select, rename, and add one imported surface of a use declaration bare in *scope*."""
        selected = self._select_use_members(decl, members, validate=False)
        selected_scope_routes = self._select_use_members(
            decl,
            scope_routes,
            validate=False,
            merge=lambda left, right: left | right,
        )
        exposed_scope_routes = {
            atom: route for atom, route in selected_scope_routes.items() if _bare_path(atom)
        }
        contributed_bindings: dict[NameAtom, set[BindingRef]] = {}
        contributed_sources: list[tuple[NameAtom, QName]] = []

        def contribute(exposed: NameAtom, source: QName) -> None:
            ref = self._cross_module_binding_ref(source)
            scope.contribute_bare(exposed, ref, ContributionLayer.USE)
            contributed_bindings.setdefault(exposed, set()).add(ref)
            contributed_sources.append((exposed, source))

        for exposed, qname in selected.items():
            contribute(exposed, qname)
        for exposed, source in self._use_renamed_members(decl, members):
            contribute(exposed, source)
        index = len(scope.imported_use_contributions)
        scope.imported_use_contributions.append(
            ImportedUseContribution(
                declaration=decl,
                target=self._use_targets[decl.node_id],
                refreshes_all_members=decl.tail == (),
                members=dict(selected),
                scope_routes=exposed_scope_routes,
                bindings={atom: frozenset(refs) for atom, refs in contributed_bindings.items()},
                hidden_prefixes=frozenset(_item_path(item) for item in decl.hidden),
            )
        )

        def complete_constructors() -> None:
            constructors: dict[NameAtom, set[ConstructorRef]] = {}
            for exposed, source in contributed_sources:
                constructor = self._contribute_bare_constructor(
                    scope, exposed, source, ContributionLayer.USE
                )
                if constructor is not None:
                    constructors.setdefault(exposed, set()).add(constructor)
            scope.imported_use_contributions[index] = replace(
                scope.imported_use_contributions[index],
                constructors={atom: frozenset(refs) for atom, refs in constructors.items()},
            )

        self._deferred_constructors.append(complete_constructors)

    @staticmethod
    def _use_renamed_members(
        decl: UseDecl, members: Mapping[NameAtom, _T]
    ) -> Iterator[tuple[NameAtom, _T]]:
        """Yield every source reached by additive use-tail renames."""
        for item in decl.tail or ():
            if item.rename is None:
                continue
            prefix = _item_path(item)
            for atom, source in members.items():
                path = _bare_path(atom)
                if path[: len(prefix)] == prefix:
                    yield _bare_atom((item.rename, *path[len(prefix) :])), source

    def _select_use_members(
        self,
        decl: UseDecl,
        members: Mapping[NameAtom, _T],
        *,
        validate: bool = True,
        merge: Callable[[_T, _T], _T] | None = None,
    ) -> dict[NameAtom, _T]:
        """Apply a use tail, hiding clause, or additive route alias to members."""

        def matching(item: ImportItem) -> tuple[NameAtom, ...]:
            prefix = _item_path(item)
            matches = tuple(atom for atom in members if _atom_under_prefix(atom, prefix))
            if not matches and validate:
                raise UnknownMemberError(_use_target_spelling(decl, prefix), span=decl.span)
            return matches

        if decl.alias is not None:
            return {
                _bare_atom((decl.alias, *_bare_path(atom))): source
                for atom, source in members.items()
            }
        selected: dict[NameAtom, _T] = {}

        def add(atom: NameAtom, source: _T) -> None:
            if merge is not None and atom in selected:
                selected[atom] = merge(selected[atom], source)
            else:
                selected[atom] = source

        if not decl.tail:
            selected.update(members)
        else:
            for item in decl.tail:
                for atom in matching(item):
                    add(atom, members[atom])
                    if item.rename is not None:
                        suffix = _bare_path(atom)[len(_item_path(item)) :]
                        add(_bare_atom((item.rename, *suffix)), members[atom])
        for hidden in decl.hidden:
            for atom in matching(hidden):
                selected.pop(atom, None)
        return selected

    def _resolve_scope_region(self, region: ScopeRegion) -> None:
        """Resolve a named region in its member layer."""
        path = self._scope.scope_path + (region.segment.name,)
        with self._named_scope(path):
            self._resolve_block_items(region.items)

    def _resolve_funcdef(self, node: FuncDef) -> None:
        """Resolve a ``def`` declaration (body + params).

        At the root, the pre-pass already defined the function binding, so we
        just resolve the body with a fresh param scope. Named-scope members
        use their collected member layer; a nested block holds no ``def``
        (placement is checked first).
        """
        if node.scope_path:
            written_in = self._scope.scope_path
            with self._named_scope(tuple(segment.name for segment in node.scope_path)):
                self._classify_method_declaration(node, written_in)
                self._validate_qualifier_chains(node, node.type_params)
                self._resolve_program_config(node)
                self._resolve_params_and_body(node)
            return
        # Defaults are resolved in the enclosing (root) scope — they are
        # evaluated in the function's definition scope.
        self._classify_method_declaration(node, ())
        self._validate_qualifier_chains(node, node.type_params)
        self._resolve_program_config(node)
        self._resolve_params_and_body(node)

    def _resolve_program_config(self, node: FuncDef) -> None:
        """Resolve a ``program def``'s ``@config`` keys and values, if it carries one.

        Both resolve in the scope that declares the program, before its
        parameter scope opens.
        """
        if not node.is_program:
            return
        for attribute in node.attributes:
            if attribute.name != CONFIG_ATTRIBUTE:
                continue
            for entry in attribute.keyed_args:
                self._resolve_varref(entry.key)
                self._resolve_expr(entry.value)

    def _resolve_type_decl(self, node: RecordDef | EnumDef | ExceptionDef | TypeAlias) -> None:
        """Validate a type declaration's type names in its lexical owner layer."""
        # Type declarations do not otherwise participate in value resolution,
        # but their annotations still need qualifier validation in the actual
        # lexical owner layer.
        path = (
            tuple(segment.name for segment in node.scope_path)
            if node.scope_path
            else self._named_scope_path()
        )
        if isinstance(node, TypeAlias):
            self._validate_alias(path, node)
        else:
            with self._named_scope(path):
                self._validate_type_decl(node)

    def alias_target(
        self, qname: QName, alias: TypeAlias, spelling: NameT | AppliedT
    ) -> TypeSelection | None:
        """Return what alias *qname*'s nominal target *spelling* selects.

        Exactly what the target's own type position records where the alias is
        declared, validating the alias first when the type-owner index asks
        before the ordered walk reaches it.
        """
        self._validate_alias(_bare_path(qname[1])[:-1], alias)
        return self._owner_declarations.get(selection_node_id(spelling))

    def _validate_alias(self, path: ScopePath, alias: TypeAlias) -> None:
        """Validate *alias*, declared at scope *path*, once."""
        if alias.node_id not in self._validated_aliases:
            self._validated_aliases.add(alias.node_id)
            with self._named_scope(path):
                self._validate_type_decl(alias)

    def _validate_type_decl(self, node: RecordDef | EnumDef | ExceptionDef | TypeAlias) -> None:
        """Validate *node*'s type names in the current layer, an exception's base included."""
        self._validate_qualifier_chains(node, node.type_params)
        if isinstance(node, ExceptionDef) and node.base is not None:
            self._select_type_name(node.base, node.span, node.node_id)

    # ------------------------------------------------------------------
    # Binder handlers
    # ------------------------------------------------------------------

    @contextmanager
    def _binder_scope(self, node: LetDecl | VarDecl) -> Iterator[None]:
        """Enter the named layer a scoped ``let``/``var`` binder path selects.

        A path prefix stands only at the program root or in a scope region's
        own body (placement is checked first). An unprefixed binder resolves
        ambiently, so a region's bare contents and a root shorthand for the
        same path compose identically.
        """
        if not node.scope_path:
            yield
            return
        with self._named_scope(tuple(segment.name for segment in node.scope_path)):
            self._validate_qualifier_chains(node)
            yield

    def _resolve_let(self, node: LetDecl) -> None:
        with self._binder_scope(node):
            # Resolve the initializer before the ordered definition. Static-root
            # bindings may already expose this identity through the pre-pass.
            self._resolve_expr(node.value)
            self._check_not_reserved(node.name, node.span)
            self._define_binder(
                node,
                decl_node_id=node.node_id,
                name=node.name,
            )

    def _resolve_var(self, node: VarDecl) -> None:
        with self._binder_scope(node):
            self._check_not_reserved(node.name, node.span)
            # Resolve RHS before defining the name (lambda non-recursion).
            self._resolve_expr(node.value)
            self._define_binder(node, decl_node_id=node.node_id, name=node.name)

    def _binder_ref(
        self,
        node: LetDecl | VarDecl | BuiltinVarDecl,
        *,
        decl_node_id: int,
        name: str,
        scope_path: ScopePath = (),
    ) -> BindingRef:
        """Build the ``BindingRef`` one ``let``/``var``/``builtin var`` binder resolves to.

        Mutability and kind follow *node*'s own type, the same for a binder
        looked up before its own textual position (the static-root pre-passes
        ``_register_static_binding_declaration``/``_register_builtin_var_declaration``)
        as for one defined at it (``_define_binder``).
        """
        if isinstance(node, VarDecl):
            mutable, kind = True, BinderKind.var_binding
        elif isinstance(node, BuiltinVarDecl):
            mutable, kind = True, BinderKind.builtin_var_binding
        else:
            mutable, kind = False, BinderKind.let_binding
        return BindingRef(
            name=name,
            mutable=mutable,
            decl_span=node.span,
            decl_node_id=decl_node_id,
            kind=kind,
            module_id=self._module_id,
            scope_path=scope_path,
            is_param=is_param_declaration(node.attributes),
        )

    def _define_binder(self, node: LetDecl | VarDecl, *, decl_node_id: int, name: str) -> None:
        """Define a resolved simple-name binder using its declaration identity."""
        if name == "_":
            return
        self._define(name, self._binder_ref(node, decl_node_id=decl_node_id, name=name))

    def _resolve_assign(self, node: AssignStmt) -> None:
        target = node.target
        if isinstance(target, IndexTarget):
            # An indexed target has no binding of its own: it mutates whatever
            # container its object expression evaluates to, so scope resolves
            # ``obj`` and ``index`` as ordinary expressions rather than looking
            # up a mutable binding.
            self._resolve_expr(target.obj)
            self._resolve_expr(target.index)
            self._resolve_expr(node.value)
            return
        if isinstance(target, FieldTarget):
            # Like an index target, a field target creates no assignment
            # binding: it mutates its receiver value. Field validity and
            # mutability are type rules.
            self._resolve_expr(target.obj)
            self._resolve_expr(node.value)
            return
        if target.qualifier is not None:
            self._resolve_qualified_assign(node, target.name, target.qualifier)
            return
        name = target.name
        # A bare target reaches an imported mutable binding just like a read.
        ref = self._bare_value(name, node.span)
        if ref is None:
            raise AglScopeError(
                f"'{name}' is not declared; assignment requires an existing mutable binding.",
                span=node.span,
            )
        # Mutability is not decided here: a field-directed pattern slot's final
        # binding is only known once checking selects it, so type checking owns
        # the ``:=``-on-immutable rejection for every unqualified target.
        self._resolution[node.node_id] = ref
        self._resolve_expr(node.value)

    def _resolve_qualified_assign(
        self, node: AssignStmt, name: str, qualifier: QualifierChain
    ) -> None:
        """Resolve a qualified assignment target (``PATH::name := expr``).

        The target is selected exactly as the same qualified spelling read
        as a value is (:meth:`_select_qualified`), so a scoped ``var`` is
        assignable through its path while a scoped ``let`` -- or a ``def``,
        a type, or an agent sharing its path -- reuses the immutable-binder
        diagnostic below; only an exported ``var`` -- ordinary or ``builtin
        var`` -- is assignable across a module boundary (a cross-module
        ``let`` reuses the immutable-binder diagnostic too). A constructor
        only a type owner's own table selects is never assignable either.
        """
        target = self._select_qualified(qualifier, name, LookupKind.VALUE, node.span)
        ref = None if isinstance(target, Misfit) else target.ref
        if ref is None:
            raise ImmutableAssignmentError(
                name, BinderKind.constructor_binding, cross_module=False, span=node.span
            )
        if not ref.mutable:
            raise ImmutableAssignmentError(
                name, ref.kind, cross_module=ref.module_id != self._module_id, span=node.span
            )
        self._resolution[node.node_id] = ref
        self._resolve_expr(node.value)

    # ------------------------------------------------------------------
    # Expression resolution
    # ------------------------------------------------------------------

    def _resolve_expr_or_block(self, expr: Expr) -> None:
        """Resolve *expr*, opening a child scope if it is a ``Block``.

        This is used for branch/function/lambda/try bodies: if the body IS a
        block, open a fresh child scope and resolve its items there; otherwise
        resolve the expression directly.
        """
        if isinstance(expr, Block):
            with self._child_scope(expr.node_id):
                self._resolve_block_items(expr.items)
        else:
            self._resolve_expr(expr)

    def _resolve_expr(self, expr: Expr) -> None:
        """Recursively resolve all names in *expr*."""
        match expr:
            case VarRef():
                self._resolve_varref(expr)
            case Call():
                self._resolve_call(expr)
            case Template():
                self._resolve_template(expr)
            case Block():
                with self._child_scope(expr.node_id):
                    self._resolve_block_items(expr.items)
            case If():
                self._resolve_if(expr)
            case Case():
                self._resolve_case(expr)
            case Loop():
                self._resolve_loop(expr)
            case Try():
                self._resolve_try(expr)
            case Lambda():
                self._resolve_lambda(expr)
            case Raise():
                self._resolve_expr(expr.exc)
            case Return():
                if not self._in_function:
                    raise AglScopeError(
                        "'return' used outside a function.",
                        span=expr.span,
                    )
                if expr.value is not None:
                    self._resolve_expr(expr.value)
            case Break():
                if not self._in_loop:
                    raise AglScopeError(
                        "'break' used outside a loop.",
                        span=expr.span,
                    )
            case Continue():
                if not self._in_loop:
                    raise AglScopeError(
                        "'continue' used outside a loop.",
                        span=expr.span,
                    )
            case FieldAccess():
                self._resolve_field_access(expr)
            case RecordUpdate():
                self._resolve_expr(expr.target)
                for update in expr.updates:
                    self._resolve_expr(update.value)
            case IndexAccess():
                self._resolve_expr(expr.obj)
                self._resolve_expr(expr.index)
            case BinaryOp():
                self._resolve_expr(expr.left)
                self._resolve_expr(expr.right)
            case UnaryNot():
                self._resolve_expr(expr.operand)
            case UnaryNeg():
                self._resolve_expr(expr.operand)
            case IsTest():
                candidates = (
                    self._visible_bare_constructor_candidates(expr.variant, expr.span)
                    if expr.qualifier is None
                    else self._qualified_constructor_candidates(
                        expr.node_id, expr.qualifier, expr.variant
                    )
                )
                self._is_test_constructor_candidates[expr.node_id] = candidates
                if len(candidates) == 1:
                    self._constructor_refs[expr.node_id] = candidates[0]
                self._resolve_expr(expr.expr)
            case Cast():
                self._resolve_expr(expr.expr)
            case TypeApply():
                self._resolve_expr(expr.expr)
            case ArrayLit():
                for elem in expr.elements:
                    self._resolve_expr(elem)
            case DictLit():
                for entry in expr.entries:
                    self._resolve_expr(entry.key)
                    self._resolve_expr(entry.value)
            case (
                UnitLit()
                | IntLit()
                | DecimalLit()
                | BoolLit()
                | NullLit()
                | StringLit()
                | OperatorRef()
            ):
                pass  # Literals and operator references name nothing.
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _resolve_varref(self, node: VarRef, *, is_call_target: bool = False) -> None:
        """Resolve a name reference.

        When a VarRef resolves to a constructor_binding, look up the candidate set:
        - Exactly 1 candidate → record in constructor_refs.
        - ≥ 2 candidates → ambiguity error.

        A bare reference uses lexical scope then tailed imports; a
        current-module chain uses the own root scope; and a module-route
        chain uses qualified cross-module access.

        *is_call_target* is set only by :meth:`_resolve_call`, resolving its
        own callee directly rather than through :meth:`_resolve_expr`: a
        resolved binding that turns out to be a built-in function has a host
        dispatch when it is the callee of a call (``_resolve_call`` classifies
        it). Outside call position, runtime built-ins remain ordinary resolved
        references and are materialized as typed closures during lowering.
        """
        if node.name == "_":
            raise AglScopeError(undefined_name_message("_"), span=node.span)
        qualifier = node.qualifier
        if qualifier is not None:
            found = self._qualified_lookup(qualifier, node.name, LookupKind.VALUE, node.span)
            if (
                is_call_target
                and isinstance(found, (UnknownQualifierError, UnknownMemberError))
                and self._qualifier_denotes_builtin_static_owner(qualifier)
            ):
                found = self._unknown_static_error(node, qualifier)
            target = self._recorded_selection(qualifier, found)
            if isinstance(target, Misfit):
                raise type_name_not_a_value(render_qualified_name(qualifier, node.name), node.span)
            ref = target.ref
            if ref is not None:
                # A module root's own constructor keeps its binding; any
                # other selected constructor is read through its candidate.
                if ref.kind is not BinderKind.constructor_binding or not qualifier.segments:
                    self._resolution[node.node_id] = ref
            if target.constructor is not None:
                self._constructor_refs[node.node_id] = target.constructor
            self._raise_unrecognized_builtin_static(node, qualifier)
            self._reject_builtin_value_ref(
                node, self._resolution.get(node.node_id), is_call_target=is_call_target
            )
            return
        # A builtin spelling is reserved in bare namespace lookups. Qualified
        # module and scope members may share it, but cannot intercept the
        # builtin through a ``use`` or other bare contribution, nor through
        # an own function.
        reserved = node.name in _BUILTIN_CALL_NAMES
        ref = self._bare_value(node.name, node.span, contributions=not reserved)
        if reserved and (
            ref is None
            or (ref.kind is BinderKind.function_binding and not self._is_builtin_function_ref(ref))
        ):
            builtin_ref = self._bare_builtin_ref(node.name)
            if builtin_ref is not None:
                self._reject_builtin_value_ref(node, builtin_ref, is_call_target=is_call_target)
                self._record_varref_binding(node, builtin_ref)
                return
            raise AglScopeError(undefined_name_message(node.name), span=node.span)
        if ref is None:
            raise (
                self._bare_type_misfit(node.name, node.span)
                or self._spaced_qualifier_repair(
                    self._spaced_qualifier_around(node.span), node.span
                )
                or AglScopeError(undefined_name_message(node.name), span=node.span)
            )
        self._reject_builtin_value_ref(node, ref, is_call_target=is_call_target)
        self._record_varref_binding(node, ref, candidates=self._value_constructors(node.name))

    def _qualifier_denotes_builtin_static_owner(self, chain: QualifierChain) -> bool:
        """Return whether *chain* spells a host static's nominal owner no declaration claims."""
        relative_path = tuple(segment.name for segment in chain.segments)
        return (
            self._denotes_builtin_static_owner(relative_path)
            and not self.names_own(relative_path)
            and not qualifier_candidates(self._import_env, relative_path, anchored=chain.anchored)
        )

    def _denotes_builtin_static_owner(self, relative_path: ScopePath) -> bool:
        """Return whether *relative_path* names a live prelude built-in static owner."""
        return relative_path in BUILTIN_TYPE_STATIC_OWNER_PATHS and bool(
            self._builtin_static_decl_node_ids
        )

    def _builtin_static_kind(self, ref: BindingRef | None) -> BuiltinStaticKind | None:
        """Return the static kind attached to its resolved prelude owner."""
        if ref is None or not ref.is_builtin:
            return None
        kind = builtin_type_static_kind(ref.module_id, ref.scope_path, ref.name)
        if kind is None or ref.decl_node_id not in self._builtin_static_decl_node_ids:
            return None
        return kind

    def _is_unrecognized_builtin_static(self, node: VarRef) -> bool:
        """Return whether a qualified prelude owner rejected an unknown static."""
        constructor = self._constructor_refs.get(node.node_id)
        if constructor is None:
            return False
        owner_path = (*constructor.owner_path, constructor.owner_name)
        return is_builtin_type_static_owner(constructor.owner_module_id, owner_path)

    def _raise_unrecognized_builtin_static(self, node: VarRef, qualifier: QualifierChain) -> None:
        """Raise when a resolved prelude owner does not declare the requested static."""
        if self._is_unrecognized_builtin_static(node):
            raise self._unknown_static_error(node, qualifier)

    @staticmethod
    def _unknown_static_error(node: VarRef, qualifier: QualifierChain) -> UnknownMemberError:
        """Build the diagnostic for a prelude owner that lacks the requested static."""
        return UnknownMemberError(render_qualified_name(qualifier, node.name), span=node.span)

    #: Built-in names that genuinely require direct call syntax and may never
    #: be referenced as a first-class value, each mapped to why: ``resource``
    #: needs a literal path argument visible at its own call site;
    #: ``parse``/``try-parse`` need their target type resolved at the call
    #: site (explicit ``::[T]`` or the contextual expected type), which no
    #: eta-expanded closure value carries.
    _FIRST_CLASS_REJECTED_BUILTIN_REASONS: Mapping[str, str] = {
        "resource": "it requires a literal path argument in direct call position",
        "parse": "its target type is resolved at the call site",
        "try-parse": "its target type is resolved at the call site",
    }

    def _reject_builtin_value_ref(
        self, node: VarRef, ref: BindingRef | None, *, is_call_target: bool
    ) -> None:
        """Reject the built-in forms that genuinely require direct call syntax."""
        if is_call_target or ref is None or not self._is_builtin_function_ref(ref):
            return
        reason = self._FIRST_CLASS_REJECTED_BUILTIN_REASONS.get(ref.name)
        if reason is None:
            return
        raise AglScopeError(
            f"Built-in function '{ref.name}' cannot be referenced as a value; "
            f"call it directly ({reason}).",
            span=node.span,
        )

    def _lookup_value(self, name: str, scope: ScopeNode | None = None) -> BindingRef | None:
        """Look up *name* in the value namespace from *scope* or the current scope.

        A named-scope type with no constructor of its own binds no value, so it
        never hides an outer, imported, or builtin value of the same spelling.
        """
        start = self._scope if scope is None else scope
        return start.lookup(name, member_predicate=self._is_value_member)

    def _is_value_member(self, ref: BindingRef) -> bool:
        """Whether a named-scope member denotes a value rather than only a type."""
        return ref.kind is not BinderKind.constructor_binding or bool(
            self._scoped_constructor_candidates.get((ref.scope_path, ref.name))
        )

    def _record_varref_binding(
        self,
        node: VarRef,
        ref: BindingRef,
        *,
        candidates: Mapping[ConstructorRef, Layers] | None = None,
    ) -> None:
        """Record an ordinary value binding and, for a constructor, its one selected candidate.

        *candidates* is the constructor decision's, each with its contributing
        layers; several are ambiguous.
        """
        self._resolution[node.node_id] = ref
        if ref.kind != BinderKind.constructor_binding or not candidates:
            return
        if len(candidates) >= 2:
            ordered = sorted(candidates, key=_constructor_candidate_sort_key)
            raise self._ambiguous_constructor(
                node.name,
                {candidate: candidates[candidate] for candidate in ordered},
                self._bare_constructor_repair(ordered[0], node),
                node.span,
            )
        (self._constructor_refs[node.node_id],) = candidates

    def _bare_constructor_repair(self, candidate: ConstructorRef, node: VarRef) -> str:
        """Spell *candidate*, which bare *node* selects among others, as it resolves at *node*.

        An imported member is qualified by the owner name a root import tail
        makes bare, renamed as that import exposes it, while that spelling
        selects *candidate* where *node* is written; else by its shortest
        unique import route. Anything else is spelled by its declaration path.
        """
        path = (*candidate.owner_path, node.name)
        if candidate.owner_module_id == self._module_id:
            return spell_declaration(self._module_id, path, local_to=self._module_id)
        origin = (
            candidate.owner_module_id,
            _bare_atom((*candidate.owner_path, candidate.owner_name)),
        )
        unqualified = self._import_env.unqualified
        owner = next(
            (
                atom[0]
                for atom, qnames in unqualified.items()
                if isinstance(atom, tuple)
                and atom[1:] == (node.name,)
                and origin in qnames
                and len(unqualified.get(atom[0], ())) == 1
            ),
            None,
        )
        if owner is not None and self._spelling_selects(owner, candidate, node):
            return f"{owner}::{node.name}"
        return self._routed_spelling(origin, candidate.owner_module_id, path)

    def _spelling_selects(self, owner: str, candidate: ConstructorRef, node: VarRef) -> bool:
        """Whether ``owner::name``, written where bare *node* is, selects *candidate*."""
        segment = QualifierSegment(owner, None, node.span, node.node_id)
        chain = QualifierChain(None, (segment,), node.name, node.span, node.node_id)
        found = self._qualified_lookup(chain, node.name, LookupKind.VALUE, node.span)
        return isinstance(found, QualifiedTarget) and found.constructor == candidate

    def _routed_spelling(self, origin: QName, module: ModuleId, path: ScopePath) -> str:
        """Spell imported *origin* by its shortest unique route, else as *path* in *module*."""
        return route_spelling(self._import_env, origin) or spell_declaration(
            module, path, local_to=self._module_id
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
            local_to=self._module_id,
        )

    def _declaring_constructor_candidates(
        self, name: str, ref: BindingRef
    ) -> tuple[ConstructorRef, ...]:
        """Return the candidates declared where *ref*, a ``use``-contributed binding, was found.

        This module's own contributed member is selected by its named scope's
        structured identity: a member without a constructor (an enum, say)
        has none there.
        """
        if ref.module_id == self._module_id:
            return tuple(self._scoped_constructor_candidates.get((ref.scope_path, name), ()))
        return tuple(self._constructor_candidates.get(name, ()))

    def _validate_qualifier_chains(self, root: SyntaxNode, type_params: Iterable[str] = ()) -> None:
        """Validate qualifier syntax in the current lexical scope layer.

        *type_params* are the enclosing declaration's type parameters, which
        shadow every qualifier's leading segment, exactly as they shadow a
        bare name: ``def f[E](x: E::A)`` and ``def f[types](x: types::Color)``
        both name the type parameter, which qualifies nothing, so both are
        rejected here rather than left to fall through to a module route.
        """
        type_param_set = frozenset(type_params)

        def validate(node: object) -> None:
            chain = (
                node.chain
                if isinstance(node, VariantRef)
                else node.qualifier
                if isinstance(
                    node, (VarRef, NameTarget, ConstructorPattern, IsTest, NameT, AppliedT)
                )
                else None
            )
            if chain is None:
                if isinstance(node, (NameT, AppliedT)) and node.name not in type_param_set:
                    self._select_type_name(node.name, node.span, node.node_id)
                return
            nonleading_segments = chain.segments[1:]
            if any(segment.anchored or "/" in segment.name for segment in nonleading_segments) or (
                chain.anchor is QualifierAnchor.CURRENT_MODULE
                and chain.segments
                and (chain.segments[0].anchored or "/" in chain.segments[0].name)
            ):
                raise AglScopeError(
                    "Only the leading qualifier segment may name a module route.", span=chain.span
                )
            if chain.anchor is None and chain.segments and chain.segments[0].name in type_param_set:
                raise _unknown_qualifier(chain)
            if isinstance(node, (NameT, AppliedT, VariantRef)):
                self._record_type_selection(chain.node_id, self._type_name_target(node))

        walk(root, validate)

    def _select_type_name(self, name: str, span: SourceSpan, node_id: int) -> None:
        """Decide what bare type name *name*, spelled by node *node_id*, selects, and record it.

        Behind every bare type name -- an annotation, alias target, type
        argument, applied type, caught exception type and ``extends`` base.
        Typecheck reads the recorded identity back; a name selecting nothing
        is left to its built-in fallback names.
        """
        self._record_type_selection(node_id, self._type_target(None, name, span))

    def _record_type_selection(
        self, node_id: int, found: QualifiedTarget | AglError | None
    ) -> None:
        """Record what a type spelling selects (*found*) under *node_id*, or raise why nothing."""
        if isinstance(found, AglError):
            raise found
        selection = None if found is None else found.selection
        if selection is not None:
            self._owner_declarations[node_id] = selection

    def _type_name_target(
        self, spelling: NameT | AppliedT | VariantRef
    ) -> QualifiedTarget | AglError | None:
        """Return what type name or member reference *spelling* selects here, or why nothing."""
        if isinstance(spelling, VariantRef):
            chain = spelling.chain
        elif spelling.qualifier is None:
            return self._type_target(None, spelling.name, spelling.span)
        else:
            chain = spelling.qualifier
        return self._type_target(chain, chain.member, chain.span)

    def _type_target(
        self, chain: QualifierChain | None, name: str, span: SourceSpan
    ) -> QualifiedTarget | AglError | None:
        """Return what type spelling *chain*``::``*name* selects, or why nothing.

        A qualified spelling selecting only a value yields that value, which
        typecheck reports as no type; a bare one selecting nothing yields
        ``None``.
        """
        if chain is None:
            return self._bare_lookup(name, LookupKind.TYPE, span)
        qualified = self._qualified_lookup(chain, name, LookupKind.TYPE, span)
        return qualified.target if isinstance(qualified, Misfit) else qualified

    def _bare_lookup(
        self, name: str, kind: LookupKind, span: SourceSpan
    ) -> QualifiedTarget | AglError | None:
        """Return what bare *name* of *kind* selects in the current named scope (:mod:`lookup`)."""
        return lookup_bare(
            self, name, self._named_scope_path(), kind, span=span, local_to=self._module_id
        )

    def _qualified_lookup(
        self, chain: QualifierChain, member: str, kind: LookupKind, span: SourceSpan
    ) -> QualifiedTarget | Misfit | AglError:
        """Return what *chain*``::``*member* selects in the current named scope (:mod:`lookup`).

        A ``::`` spelling naming nothing is first offered as a module
        qualifier whitespace split off.
        """
        found = lookup_qualified(
            self,
            chain,
            member,
            self._named_scope_path(),
            kind,
            span=span,
            local_to=self._module_id,
        )
        if chain.anchor is QualifierAnchor.CURRENT_MODULE and isinstance(
            found, (UnknownMemberError, UnknownQualifierError)
        ):
            return (
                self._spaced_qualifier_repair(
                    self._spaced_qualifier_at(chain.span) or self._spaced_qualifier_around(span),
                    span,
                )
                or found
            )
        return found

    def _select_qualified(
        self, chain: QualifierChain, member: str, kind: LookupKind, span: SourceSpan
    ) -> QualifiedTarget | Misfit:
        """Decide what ``chain::member`` selects in a position taking *kind*, and record it.

        The one decision behind a qualified value, assignment target,
        pattern and ``is`` test. The selected declaration's identity is
        recorded in ``owner_declarations`` under the chain's node id, so
        typecheck reads it back instead of re-resolving the qualifier.
        """
        return self._recorded_selection(chain, self._qualified_lookup(chain, member, kind, span))

    def _recorded_selection(
        self, chain: QualifierChain, found: QualifiedTarget | Misfit | AglError
    ) -> QualifiedTarget | Misfit:
        """Record the declaration *chain* selects (*found*) under its node id, or raise why none."""
        if isinstance(found, AglError):
            raise found
        self._record_type_selection(
            chain.node_id, found.target if isinstance(found, Misfit) else found
        )
        return found

    def _bare_type_misfit(self, name: str, span: SourceSpan) -> AglError | None:
        """Return why bare *name*, naming no value, spells a visible type -- or several."""
        found = self._bare_lookup(name, LookupKind.TYPE, span)
        return type_name_not_a_value(name, span) if isinstance(found, QualifiedTarget) else found

    def _named_scope_path(self) -> ScopePath:
        """Return the path of the nearest named scope enclosing the current layer."""
        layer: ScopeNode | None = self._scope
        while layer is not None and not layer.scope_path:
            layer = layer.parent
        return () if layer is None else layer.scope_path

    # -- What the one lookup reads (``lookup.PathSources``) --

    def own_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """This module's own declaration of *kind* at full *path*."""
        if kind is LookupKind.TYPE:
            qname = (self._module_id, _bare_atom(path))
            target = (
                QualifiedTarget(self._qname_decl_key(qname), None, None)
                if self._type_owners.is_declared(qname)
                else None
            )
        elif len(path) == 1:
            target = self._own_root_value(path[0])
        else:
            target = self._own_scoped_value(path[:-1], path[-1])
        if target is None or not self._fits(target, kind):
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
        """What the contributions anchored at or above *step* reach at full *path*."""
        constructors = self._contributed_constructors(step, path, kind)
        candidates = (
            Candidate(
                self._contributed_target(ref, constructors),
                layer,
                contribution_origin(_ref_qname(ref), layer),
            )
            for ref, layers in self._contributed_bindings(step, path, kind).items()
            for layer in layered(layers)
        )
        return Reading(tuple(c for c in candidates if self._fits(c.target, kind)))

    def routed_at(self, chain: QualifierChain, path: ScopePath, kind: LookupKind) -> Reading:
        """What *chain*'s leading module route alone reaches at *path* beneath it."""
        candidates = (
            Candidate(
                self._contributed_target(self._cross_module_binding_ref(qname), ()),
                ContributionLayer.IMPORTED,
                ImportedModuleOrigin(qname),
            )
            for qname in self._routed_qnames(chain.leading_route, path, anchored=chain.anchored)
        )
        return Reading(tuple(c for c in candidates if self._fits(c.target, kind)))

    def aliased_at(self, step: ScopePath, path: ScopePath) -> Reading:
        """The types a ``use`` alias anchored at or above *step* spelled *path* stands for."""
        targets: list[QName] = []
        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            for contribution in layer.imported_use_contributions:
                targets.extend(
                    (module_id, _bare_atom(route))
                    for module_id, route in contribution.scope_routes.get(atom, ())
                )
            targets.extend(
                (self._module_id, _bare_atom(local.source.scope_path))
                for local in layer.local_use_contributions
                if local.declaration.alias == atom
            )
        layer_tag = ContributionLayer.USE
        return Reading(
            tuple(
                Candidate(
                    QualifiedTarget(self._qname_decl_key(qname), None, None),
                    layer_tag,
                    contribution_origin(qname, layer_tag),
                )
                for qname in dict.fromkeys(targets, True)
                if self._type_owners.is_declared(qname)
            )
        )

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
                if candidate.owner_module_id == self._module_id
                and _is_root_inline_member(candidate)
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
                injected[min(injected, key=_constructor_candidate_sort_key)],
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
    ) -> Reading:
        """What type *owner*, made visible by *layer*, selects for *rest* by its own member table.

        Each name of *rest* but the last must name a type declared beneath
        the one before; one the owner so far only references or hides is
        refused. The owner reached last decides the member: a referenced or
        hidden member is refused, and so is a type it declares (an inline enum
        member or a nested type), which only a contribution reaching its full
        path selects -- so a ``hiding`` removes exactly that path. An
        alias's projection, or a record's own spelling, selects.
        """
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
            current = (current[0], _bare_atom((*_bare_path(current[1]), name)))
        if (table.alias is None and name in table.members) or self._type_owners.is_declared(
            current
        ):
            return Reading(refusals=(HiddenMemberError(spelling, name, span=chain.span),))
        constructor = table.select(name, segments[-1].name)
        if constructor is None:
            return Reading()
        key = self._qname_decl_key(current)
        origin = contribution_origin(current, layer)
        return Reading((Candidate(QualifiedTarget(key, None, constructor), layer, origin),))

    def inline_arity(self, owner: DeclarationKey, member: str) -> int | None:
        """The type parameters type *owner* takes when *member* is one of its inline members."""
        reached = self._type_owners.owner(_key_qname(owner))
        return None if reached is None or member not in reached.members else reached.arity

    def hidden_at(self, step: ScopePath, path: ScopePath) -> bool:
        """Whether a ``hiding`` of a contribution anchored at or above *step* removed *path*."""
        for layer in self._layer_chain(self._scope_nodes[step]):
            atom = _bare_atom(path[len(layer.scope_path) :])
            prefixes = (
                *(
                    prefix
                    for contribution in layer.imported_use_contributions
                    for prefix in contribution.hidden_prefixes
                ),
                *(
                    _item_path(item)
                    for contribution in layer.local_use_contributions
                    for item in contribution.declaration.hidden
                ),
            )
            if any(_atom_under_prefix(atom, prefix) for prefix in prefixes):
                return True
        if any(
            _bare_atom(path[len(anchor) :]) in hidden
            for node_id, hidden in self._import_env.decl_hidden.items()
            if step[: len(anchor := self._import_decl_scope_paths.get(node_id, ()))] == anchor
        ):
            return True
        return len(path) > 1 and qualifier_hides(self._import_env, (path[0],), _bare_atom(path[1:]))

    def routed_hidden(self, chain: QualifierChain, path: ScopePath) -> bool:
        """Whether a ``hiding`` removed *path* from *chain*'s leading module route."""
        return qualifier_hides(
            self._import_env, chain.leading_route, _bare_atom(path), anchored=chain.anchored
        )

    def names_own(self, path: ScopePath) -> bool:
        """Whether full *path* is one of this module's own scope paths or types."""
        return path in self._scope_nodes or self._type_owners.is_declared(
            (self._module_id, _bare_atom(path))
        )

    def names_contributed(self, step: ScopePath, path: ScopePath) -> bool:
        """Whether a contribution anchored at or above *step* reaches *path* or beneath it.

        An injected enum member is a constructor value only, never a name a
        qualifier reaches.
        """
        for layer in self._layer_chain(self._scope_nodes[step]):
            relative = path[len(layer.scope_path) :]
            atoms = (
                *(
                    atom
                    for atom, refs in layer.bare_contributions.items()
                    if any(ref.contributes_a_type for ref in refs)
                ),
                *(
                    atom
                    for contribution in layer.imported_use_contributions
                    for atom in contribution.scope_routes
                ),
                *(
                    exposed
                    for contribution in layer.local_use_contributions
                    for exposed, _source in self._local_use_exposures(contribution)
                ),
            )
            if any(_atom_under_prefix(atom, relative) for atom in atoms):
                return True
        env = self._import_env
        return any(
            _atom_under_prefix(atom, path)
            for atom in (*env.unqualified, *env.unqualified_scope_routes)
        ) or self._routed_names((path[0],), path[1:], anchored=False)

    def names_routed(self, chain: QualifierChain, path: ScopePath) -> bool:
        """Whether *chain*'s leading module route reaches *path* or beneath it."""
        return self._routed_names(chain.leading_route, path, anchored=chain.anchored)

    def _routed_names(self, route: tuple[str, ...], path: ScopePath, *, anchored: bool) -> bool:
        """Whether module *route* is imported and reaches *path* or beneath it."""
        env = self._import_env
        if not qualifier_candidates(env, route, anchored=anchored):
            return False
        atoms = (
            *(
                atom
                for _module, members in qualifier_members(env, route, anchored=anchored)
                for atom in members
            ),
            *(
                atom
                for _module, routes in qualifier_scope_paths(env, route, anchored=anchored)
                for atom in routes
            ),
        )
        return not path or any(_atom_under_prefix(atom, path) for atom in atoms)

    def _routed_qnames(
        self, route: tuple[str, ...], path: ScopePath, *, anchored: bool
    ) -> Iterator[QName]:
        """Yield what module *route* reaches at *path* beneath it."""
        atom = _bare_atom(path)
        for _module, members in qualifier_members(self._import_env, route, anchored=anchored):
            qname = members.get(atom)
            if qname is not None:
                yield qname

    def _contributed_bindings(
        self, step: ScopePath, path: ScopePath, kind: LookupKind
    ) -> dict[BindingRef, Layers]:
        """Return what contributions anchored at or above *step* bind at full *path*, with layers.

        Every layer from *step* outward contributes the path relative to its
        own; the module root's import tails and the module route spelled by
        its leading name contribute it whole. A binding several contribute
        keeps every one's tag. *kind* is the position's, which decides what a
        ``use``'s own scope wins (:meth:`_own_use_winners`).
        """
        bindings: dict[BindingRef, Layers] = {}
        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            for ref, layers in self._layer_bare_bindings(layer, atom, kind).items():
                add_layers(bindings, ref, layers)
        imported: Iterable[QName] = self._import_env.unqualified.get(_bare_atom(path), ())
        if path[1:]:
            imported = itertools.chain(
                imported, self._routed_qnames((path[0],), path[1:], anchored=False)
            )
        for qname in imported:
            add_layers(
                bindings, self._cross_module_binding_ref(qname), (ContributionLayer.IMPORTED,)
            )
        return bindings

    def _contributed_constructors(
        self, step: ScopePath, path: ScopePath, kind: LookupKind
    ) -> dict[ConstructorRef, Layers]:
        """Return the constructor candidates layers anchored at or above *step* give full *path*.

        *kind* is the position's, as for :meth:`_contributed_bindings`.
        """
        constructors: dict[ConstructorRef, Layers] = {}
        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            for constructor, layers in self._layer_bare_constructors(layer, atom, kind).items():
                add_layers(constructors, constructor, layers)
        return constructors

    def _fits(self, target: QualifiedTarget, kind: LookupKind) -> bool:
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

    def _own_use_winners(
        self, layer: ScopeNode, name: NameAtom, kind: LookupKind
    ) -> frozenset[int]:
        """Return the ``use`` declarations of *layer* whose own target wins *name* for *kind*.

        A use combining an own scope with imported ones exposes ``X::p`` as
        its whole-path lookup selects it: the own declaration of the
        position's kind at ``X::p`` wins that path from its imported surfaces.
        """
        return frozenset(
            contribution.declaration.node_id
            for contribution in layer.local_use_contributions
            if contribution.target.imported_routes
            and not self._local_contribution_superseded(contribution)
            and any(
                self._fits(
                    self._contributed_target(
                        source, self._declaring_constructor_candidates(source.name, source)
                    ),
                    kind,
                )
                for source in self._local_use_sources(contribution, name)
            )
        )

    def _layer_bare_bindings(
        self, layer: ScopeNode, name: NameAtom, kind: LookupKind
    ) -> dict[BindingRef, Layers]:
        """Return *layer*'s static bindings for *name*, refreshed by its live uses, with layers.

        A refresh retracts only a ``use``'s own contribution: another layer
        contributing the same binding keeps contributing it.
        """
        use = ContributionLayer.USE
        bindings = dict(layer.bare_contributions.get(name, {}))
        winners = self._own_use_winners(layer, name, kind)
        for contribution in layer.imported_use_contributions:
            refresh = self._facade_refresh(contribution, name)
            if contribution.declaration.node_id in winners:
                current = set() if refresh is None else refresh[1]
                for ref in current.union(contribution.bindings.get(name, frozenset())):
                    drop_layer(bindings, ref, use)
            elif refresh is not None:
                stale, refs, _ = refresh
                for ref in stale:
                    drop_layer(bindings, ref, use)
                for ref in refs:
                    add_layers(bindings, ref, (use,))
        for local_contribution in layer.local_use_contributions:
            # The static read above carries this contribution's snapshot,
            # taken before a later REPL entry may have redeclared one of its
            # source members; the live exposure below replaces it, so a
            # redeclared member is never a second candidate.
            for ref in local_contribution.bindings.get(name, ()):
                drop_layer(bindings, ref, use)
            sources = self._local_use_sources(local_contribution, name)
            if self._local_contribution_superseded(local_contribution):
                # The static read above can already carry a superseded
                # contribution's binding, so retract it rather than skip it.
                for source in sources:
                    drop_layer(bindings, source, use)
            else:
                for source in sources:
                    add_layers(bindings, source, (use,))
        return bindings

    def _layer_bare_constructors(
        self, layer: ScopeNode, name: NameAtom, kind: LookupKind
    ) -> dict[ConstructorRef, Layers]:
        """Return *layer*'s static constructor candidates for *name*, refreshed, with layers.

        Retracts only a ``use``'s own contribution, as :meth:`_layer_bare_bindings` does.
        """
        use = ContributionLayer.USE
        constructors = dict(layer.bare_constructor_contributions.get(name, {}))
        winners = self._own_use_winners(layer, name, kind)
        for contribution in layer.imported_use_contributions:
            refresh = self._facade_refresh(contribution, name)
            if contribution.declaration.node_id in winners:
                current_constructors = set() if refresh is None else refresh[2]
                for constructor in current_constructors.union(
                    contribution.constructors.get(name, frozenset())
                ):
                    drop_layer(constructors, constructor, use)
            elif refresh is not None:
                for constructor in refresh[2]:
                    add_layers(constructors, constructor, (use,))
        for local_contribution in layer.local_use_contributions:
            for constructor in local_contribution.constructors.get(name, ()):
                drop_layer(constructors, constructor, use)
            candidates = [
                candidate
                for source in self._local_use_sources(local_contribution, name)
                for candidate in self._declaring_constructor_candidates(source.name, source)
            ]
            if self._local_contribution_superseded(local_contribution):
                for candidate in candidates:
                    drop_layer(constructors, candidate, use)
            else:
                for candidate in candidates:
                    add_layers(constructors, candidate, (use,))
        return constructors

    def _facade_modules(
        self, contribution: ImportedUseContribution, origin: int
    ) -> frozenset[ModuleId]:
        """Return *contribution*'s current wildcard-facade modules, live at this entry's site.

        A later declaration importing one of the wildcard's own modules by
        its exact path -- under any alias -- wins that module's generation
        slot and so drops it from *origin*'s own tracked set, even though the
        module itself is still perfectly importable. Such a module is kept
        anyway: it stays live through the contribution's own prior bindings,
        each checked against the current import environment rather than
        against *origin* specifically.
        """
        modules = {
            module
            for declarations in self._import_env.facade_aliases.values()
            for node_id, candidates in declarations.items()
            if node_id == origin
            for module in candidates
        }
        modules.update(
            ref.module_id
            for refs in contribution.bindings.values()
            for ref in refs
            if ref.module_id in self._import_env.contributions
        )
        return frozenset(modules)

    def _use_modules(
        self, contribution: ImportedUseContribution, origin: int | None
    ) -> frozenset[ModuleId]:
        """Return *contribution*'s currently live module set.

        Fixed to *contribution*'s own snapshot unless it is a wildcard-facade
        use, whose module set is re-derived live (:meth:`_facade_modules`).
        """
        if origin is None:
            return frozenset(qname[0] for qname in contribution.members.values())
        return self._facade_modules(contribution, origin)

    def _facade_refresh(
        self, contribution: ImportedUseContribution, name: NameAtom
    ) -> tuple[set[BindingRef], set[BindingRef], set[ConstructorRef]] | None:
        """Return a refreshing use's stale and current bindings/constructors for bare *name*.

        ``None`` when *contribution* does not refresh *name*: a selective use
        naming an explicit tail never refreshes, and a hidden name never
        does either. Only a wildcard facade's module set can change between
        entries (:meth:`_facade_modules`), so its bindings are re-derived over
        the live modules; any other use keeps its own snapshot. Either adds
        the variant expansion of the enums it exposes bare.
        """
        if not contribution.refreshes_all_members:
            return None
        if any(_atom_under_prefix(name, prefix) for prefix in contribution.hidden_prefixes):
            return None
        origin = contribution.target.wildcard_facade_origin_node_id
        modules = self._use_modules(contribution, origin)
        if origin is None:
            stale: set[BindingRef] = set()
            bindings = set(contribution.bindings.get(name, frozenset()))
            constructors = set(contribution.constructors.get(name, frozenset()))
        else:
            existing_refs = tuple(ref for refs in contribution.bindings.values() for ref in refs)
            if not existing_refs:
                return None
            stale = {
                ref
                for ref in contribution.bindings.get(name, frozenset())
                if ref.module_id not in modules
            }
            qnames = [
                qname
                for module in modules
                if (qname := self._import_env.contributions[module].members.get(name)) is not None
            ]
            bindings = {self._cross_module_binding_ref(qname) for qname in qnames}
            constructors = {
                constructor
                for qname in qnames
                if (constructor := self._cross_module_constructor(qname)) is not None
            }
        variant_bindings, variant_constructors = self._facade_variant_refs(
            contribution, modules, name
        )
        bindings |= variant_bindings
        constructors |= variant_constructors
        return stale, bindings, constructors

    def _variant_binding_ref(self, constructor: ConstructorRef, span: SourceSpan) -> BindingRef:
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

    def _enum_variant_members(
        self, qname: QName
    ) -> Iterator[tuple[NameAtom, ConstructorRef, ScopePath]]:
        """Yield each bare atom, constructor, and hidden-check path an enum expands into.

        Shared by the live overlay (:meth:`_facade_variant_refs`), its
        static re-snapshot (:meth:`_refresh_imported_use`) and a region-scoped
        import tail (:meth:`_contribute_regional_enum_variants`): a bare-exposed
        enum type makes its own variants bare-matchable too, referenced
        members included. A referenced member's hidden-check path is its own
        declaration route; an inline member's is its path under the enum's
        own scope, since hiding always spells the enum-qualified path.
        """
        declaration = self._all_public_types.get(qname)
        if not isinstance(declaration, EnumDef):
            return
        module, source = qname
        owner_path = _bare_path(source)
        for member in declaration.members:
            if isinstance(member, VariantRef):
                for constructor in self._type_owners.referenced_member_refs(qname, member):
                    yield constructor.owner_name, constructor, constructor.owner_path
                continue
            variant_qname = (module, _bare_atom((*owner_path, member.name)))
            variant_constructor = self._cross_module_constructor_refs[variant_qname]
            yield member.name, variant_constructor, (*owner_path, member.name)

    def _facade_variant_refs(
        self, contribution: ImportedUseContribution, modules: frozenset[ModuleId], name: NameAtom
    ) -> tuple[set[BindingRef], set[ConstructorRef]]:
        """Return the bindings/constructors *contribution*'s bare-exposed enums give bare *name*.

        A facade's bare-exposed enum type makes its own variants
        bare-matchable too, the same expansion
        :meth:`_contribute_regional_enum_variants` performs for a
        region-scoped import, but derived live here instead of as a
        declaration-time side effect, so a later entry re-derives it exactly
        like every other candidate rather than losing it.
        """
        bindings: set[BindingRef] = set()
        constructors: set[ConstructorRef] = set()
        if not isinstance(name, str):
            return bindings, constructors

        def hidden(path: ScopePath) -> bool:
            return any(_atom_under_prefix(path, prefix) for prefix in contribution.hidden_prefixes)

        exposures = {exposed: (qname,) for exposed, qname in contribution.members.items()}
        for exposed, qname in contribution.members.items():
            if not isinstance(exposed, str) or qname[0] not in modules:
                continue
            for atom, constructor, hidden_path in self._enum_variant_members(qname):
                if atom != name or hidden(hidden_path):
                    continue
                # A same-named record or exception exposed bare already owns
                # the spelling; its own bare contribution stands alone. Only
                # an inline member can clash this way -- a referenced member
                # is always a standalone declaration reached under its own
                # route already.
                if constructor.inline_enum_owner_decl_node_id is not None and (
                    declares_bare_constructor(exposures.get(atom, ()), self._all_public_types)
                ):
                    continue
                bindings.add(self._variant_binding_ref(constructor, contribution.declaration.span))
                constructors.add(constructor)
        return bindings, constructors

    def _refresh_imported_use(
        self, contribution: ImportedUseContribution
    ) -> ImportedUseContribution:
        """Return *contribution* re-snapshotted against its live derivation.

        The static counterpart of the per-name reads in
        :meth:`_layer_bare_bindings`/:meth:`_layer_bare_constructors`: the same
        :meth:`_facade_refresh` derivation, applied once across every atom the
        contribution's own snapshot or its current modules could expose, for
        every refreshing use -- so the stored tables a REPL session promotes
        agree with a live lookup. A contribution that does not refresh every
        member -- a selective use, even of a wildcard-facade alias -- keeps
        its snapshot as declared.
        """
        if not contribution.refreshes_all_members:
            return contribution
        origin = contribution.target.wildcard_facade_origin_node_id
        modules = self._use_modules(contribution, origin)
        atoms = set(contribution.bindings) | set(contribution.constructors)
        if origin is not None:
            # A wildcard facade's live modules may expose atoms its snapshot lacks.
            for module in modules:
                atoms.update(self._import_env.contributions[module].members)
        for qname in contribution.members.values():
            atoms.update(atom for atom, _, _ in self._enum_variant_members(qname))
        bindings: dict[NameAtom, frozenset[BindingRef]] = {}
        constructors: dict[NameAtom, frozenset[ConstructorRef]] = {}
        for atom in atoms:
            refresh = self._facade_refresh(contribution, atom)
            if refresh is None:
                continue
            _, refs, constructor_refs = refresh
            if refs:
                bindings[atom] = frozenset(refs)
            if constructor_refs:
                constructors[atom] = frozenset(constructor_refs)
        return replace(contribution, bindings=bindings, constructors=constructors)

    def _local_use_sources(
        self, contribution: LocalUseContribution, name: NameAtom
    ) -> list[BindingRef]:
        """Return the live declarations local use *contribution* exposes as *name*."""
        return [
            source
            for exposed, source in self._local_use_exposures(contribution)
            if exposed == name and isinstance(source, BindingRef)
        ]

    @staticmethod
    def _layer_chain(layer: ScopeNode | None) -> tuple[ScopeNode, ...]:
        """Return *layer* and every layer enclosing it, innermost first."""
        chain: list[ScopeNode] = []
        while layer is not None:
            chain.append(layer)
            layer = layer.parent
        return tuple(chain)

    def _is_value_contribution(self, ref: BindingRef) -> bool:
        """Whether a shared contribution denotes a value in addition to any type."""
        return (
            ref.kind is not BinderKind.constructor_binding
            or self._cross_module_constructor(_ref_qname(ref)) is not None
            or bool(self._declaring_constructor_candidates(ref.name, ref))
        )

    def _bare_value(
        self, name: str, span: SourceSpan, *, contributions: bool = True
    ) -> BindingRef | None:
        """Return the value binding bare *name* reads at the current scope.

        A binding of an enclosing block or function is nearest. Then each
        step of the enclosing named scope's path (:func:`lookup_steps`) reads
        this module's own binding there, else -- unless *contributions* is
        off -- what the contributions anchored at or above it make it
        (:meth:`_contributed_value`).
        """
        for layer in self._lexical_layers():
            ref = self._own_level_value((layer,), name)
            if ref is not None:
                return ref
        for step in lookup_steps(self._named_scope_path()):
            ref = self._own_level_value(self._step_layers(step), name)
            if ref is None and contributions:
                ref = self._contributed_value(step, name, span)
            if ref is not None:
                return ref
        return None

    def _lexical_layers(self) -> Iterator[ScopeNode]:
        """Yield the block and function layers enclosing the current one, innermost first."""
        layer: ScopeNode | None = self._scope
        while layer is not None and layer is not self._root_scope and not layer.scope_path:
            yield layer
            layer = layer.parent

    def _step_layers(self, step: ScopePath) -> tuple[ScopeNode, ...]:
        """Return the layers declaring this module's own names at *step*.

        The module root is one step however many layers form it -- a REPL
        session chains one root layer per retained entry -- so an entry's
        grouping never changes what a name reads.
        """
        return (self._scope_nodes[step],) if step else self._layer_chain(self._root_scope)

    def _contributed_value(self, step: ScopePath, name: str, span: SourceSpan) -> BindingRef | None:
        """Return the value binding contributions anchored at or above *step* make bare *name*.

        At the module root, the root's binding for constructors only other
        modules declare stands alone for them; beside other contributions,
        each of them is an import tail's reading. Several distinct constructors
        leave the choice to the constructor decision
        (:meth:`_value_constructors`), which reports their ambiguity; any
        other clash is an ambiguous qualification, each origin tagged with
        the layer contributing it.
        """
        contributed = {
            ref: layers
            for ref, layers in self._contributed_bindings(
                step, (*step, name), LookupKind.VALUE
            ).items()
            if self._is_value_contribution(ref)
        }
        if not contributed:
            ref = None if step else self._level_value(self._step_layers(step), name)
            return ref if ref is not None and self._is_imported_constructor_binding(ref) else None
        if not step:
            for candidate in self._tail_constructors(name):
                add_layers(
                    contributed,
                    self._cross_module_binding_ref(candidate.qname),
                    (ContributionLayer.IMPORTED,),
                )
        distinct: dict[tuple[object, ...], list[tuple[BindingRef, Layers]]] = {}
        for ref, layers in contributed.items():
            distinct.setdefault(
                (ref.module_id, ref.scope_path, ref.decl_node_id, ref.kind), []
            ).append((ref, layers))
        chosen = [readings[0][0] for readings in distinct.values()]
        if all(ref.kind is BinderKind.constructor_binding for ref in chosen):
            return min(chosen, key=_binding_sort_key)
        if len(chosen) == 1:
            return chosen[0]
        raise AmbiguousQualificationError.for_origins(
            (),
            _bare_path(name),
            (
                origin
                for readings in distinct.values()
                for ref, layers in readings
                for origin in contribution_origins(_ref_qname(ref), layers)
            ),
            span=span,
            local_to=self._module_id,
        )

    def _own_level_value(self, level: tuple[ScopeNode, ...], name: str) -> BindingRef | None:
        """Return this module's own value binding *name* at *level*.

        The module root's binding for a constructor only other modules
        declare stands for their candidates, which are contributions.
        """
        ref = self._level_value(level, name)
        return None if ref is None or self._is_imported_constructor_binding(ref) else ref

    def _level_value(self, level: tuple[ScopeNode, ...], name: str) -> BindingRef | None:
        """Return the first of *level*'s layers' own value bindings *name*."""
        return next(
            (
                ref
                for layer in level
                if (ref := layer.own_binding(name, member_predicate=self._is_value_member))
                is not None
            ),
            None,
        )

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

    def _spaced_qualifier_repair(
        self, advisory: SpacedQualifier | None, failure_span: SourceSpan
    ) -> AglScopeError | None:
        """Explain a reference that whitespace split from its module qualifier.

        A run such as ``app/config ::x`` never becomes a qualifier — the lexer
        requires byte adjacency — so it parses as an unrelated expression whose
        parts then fail to resolve on their own.  The lexer recorded the run it
        saw; the repair is offered only when that run actually contributes the
        intended member (and, for ``Type::Ctor``, only when the member names a
        constructible type).
        """
        if advisory is None:
            return None
        result = resolve_qualified(
            self._import_env, advisory.segments, advisory.member, anchored=advisory.anchored
        )
        if not isinstance(result, QualResolutionFound):
            return None
        if advisory.type_qualified and not self._is_constructible_type_ref(result.qname):
            return None
        rendered = render_route_member(
            advisory.segments, (advisory.member_text,), anchored=advisory.anchored
        )
        return AglScopeError(
            f"Whitespace before '::{advisory.member_text}' makes this a call with a "
            f"self-reference, not a module qualifier. "
            f"Write '{rendered}' without whitespace.",
            # The lexer knows the offsets but not which module it scanned; the
            # failing reference supplies the source identity.
            span=replace(advisory.dcolon_span, source=failure_span.source),
        )

    def _is_constructible_type_ref(self, qname: QName) -> bool:
        """Whether *qname* names a type path that qualifies constructors."""
        return self._constructible_owner(qname) is not None

    def _constructible_owner(self, qname: QName) -> TypeOwner | None:
        """Return *qname*'s owner when it qualifies constructors, else ``None``."""
        owner = self._type_owners.owner(qname)
        return owner if owner is not None and owner.constructs else None

    def members_reached_at(
        self, scope_path: ScopePath, spelling: NameT | AppliedT, members: Collection[str]
    ) -> frozenset[str]:
        """Return the names among *members* ``<spelling>::name``, at *scope_path*, reaches.

        A name is left out when its whole-path type lookup (:mod:`lookup`)
        finds it hidden: a ``hiding`` removed the path and nothing else
        reaches it. Another declaration at that path hides nothing.
        """
        with self._named_scope(scope_path):
            return frozenset(
                name
                for name in members
                if not isinstance(
                    self._type_target(member_chain(spelling, name), name, spelling.span),
                    HiddenMemberError,
                )
            )

    def type_name_selection_at(
        self, scope_path: ScopePath, spelling: NameT | AppliedT | VariantRef
    ) -> TypeSelection | None:
        """Return what type name or member reference *spelling*, at *scope_path*, selects now.

        Scope's own type-position decision (:meth:`_type_name_target`), read
        without recording it; ``None`` when the spelling selects no
        declaration there, the decision's own rejections included.
        """
        with self._named_scope(scope_path):
            found = self._type_name_target(spelling)
        return found.selection if isinstance(found, QualifiedTarget) else None

    def _spaced_qualifier_at(self, span: SourceSpan) -> SpacedQualifier | None:
        """Return the advisory for a self-qualified reference whose ``::`` is at *span*."""
        return self._spaced_qualifiers.get(span.start_offset)

    def _spaced_qualifier_around(self, span: SourceSpan) -> SpacedQualifier | None:
        """Return the advisory whose broken qualifier run contains *span*."""
        for advisory in self._spaced_qualifiers.values():
            if advisory.covers(span.start_offset):
                return advisory
        return None

    def _is_builtin_function_ref(self, ref: BindingRef | None) -> bool:
        """Return whether *ref* names an actual ``builtin def`` declaration."""
        return ref is not None and ref.kind is BinderKind.function_binding and ref.is_builtin

    def _bare_builtin_ref(self, name: str) -> BindingRef | None:
        """Find the host builtin made visible by the standard-library prelude."""
        qnames = self._import_env.unqualified.get(name, frozenset())
        builtin_refs = [
            ref
            for qname in qnames
            if self._is_builtin_function_ref(ref := self._cross_module_binding_ref(qname))
        ]
        if len(builtin_refs) == 1:
            return builtin_refs[0]
        return None

    def _resolve_call(self, node: Call) -> None:
        """Resolve a ``Call`` node.

        A callee under a built-in name is classified in ``builtin_calls``
        only once it resolves to a ``builtin def`` declaration, through the
        same resolution every other reference uses — lexical lookup, region
        contributions, and the import environment alike. A bare ``print(x)``
        is therefore reached through the standard library's own declaration,
        exactly as ``A::print(x)`` is reached through that region's; a name
        no declaration provides is an ordinary scope error. Which scope path
        reached the declaration never changes the host implementation the
        name denotes.
        """
        callee = node.callee
        if isinstance(callee, VarRef):
            self._resolve_varref(callee, is_call_target=True)
            ref = self._resolution.get(callee.node_id)
            static_kind = self._builtin_static_kind(ref)
            if static_kind is not None:
                self._builtin_static_calls[node.node_id] = static_kind
            elif ref is not None and self._is_builtin_function_ref(ref) and not ref.is_method:
                kind = builtin_call_kind(ref.name)
                if kind is not None:
                    self._builtin_calls[node.node_id] = kind
        elif isinstance(callee, FieldAccess):
            self._resolve_field_access(callee)
            # Member selection is type-directed, so scope cannot yet know
            # whether this spelling names a builtin method or an ordinary
            # method with the same name. Record the possible host route; the
            # checker confirms it only after selecting the method declaration.
            if (kind := builtin_call_kind(callee.field)) is not None:
                self._builtin_calls[node.node_id] = kind
        else:
            self._resolve_expr(callee)
        # Resolve argument expressions; a placeholder refers to nothing.
        for arg in (*node.args, *(named.value for named in node.named_args)):
            if not isinstance(arg, Placeholder):
                self._resolve_expr(arg)

    def _resolve_field_access(self, expr: FieldAccess) -> None:
        """Resolve a field-access expression by resolving its object as a value."""
        if isinstance(expr.obj, VarRef) and expr.obj.qualifier is None:
            existing = self._lookup_value(expr.obj.name)
            if expr.obj.name in self._declared_type_names and (
                existing is None or existing.kind is BinderKind.constructor_binding
            ):
                raise AglScopeError(
                    f"'{expr.obj.name}' is a type name, not a value; use '::' for "
                    f"constructor qualification (for example, '{expr.obj.name}::{expr.field}').",
                    span=expr.obj.span,
                )
        self._resolve_expr(expr.obj)

    def _resolve_template(self, node: Template) -> None:
        for seg in node.segments:
            if isinstance(seg, InterpSegment):
                self._resolve_expr(seg.expr)

    # ------------------------------------------------------------------
    # Control-flow expression resolution
    # ------------------------------------------------------------------

    def _resolve_if(self, node: If) -> None:
        from agm.agl.syntax.nodes import ElseSentinel

        for branch in node.branches:
            if not isinstance(branch.cond, ElseSentinel):
                self._resolve_expr(branch.cond)
            # Branch body: open a child scope if the body is a Block.
            self._resolve_expr_or_block(branch.body)

    def _resolve_case(self, node: Case) -> None:
        self._resolve_expr(node.subject)
        for branch in node.branches:
            with self._child_scope(branch.node_id) as branch_scope:
                self._bind_pattern_vars(branch.pattern, branch_scope, branch.node_id)
                self._resolve_expr_or_block(branch.body)

    def _resolve_loop(self, node: Loop) -> None:
        """Resolve a unified loop expression.

        Resolution order (all in the ENCLOSING scope, before the loop variable
        is bound, so none of these can reference the loop variable):
        - ``bound`` (if any)
        - ``for_iter`` (if any) — the range start value for a range ``for``
        - ``for_range_to`` (if any) — the range upper/lower bound
        - ``for_range_step`` (if any) — the range step

        Then a single child scope is opened and ``for_var`` (if any) is bound
        immutably into it.  The loop interior (``while_cond``, body,
        ``until_cond``) is resolved in that child scope with ``_in_loop``
        set to ``True``.  If the body is a ``Block``, its items are resolved
        directly in the child scope so body bindings are visible to
        ``until_cond``.

        ``_in_loop`` is left at its enclosing value when resolving ``bound``
        and the range-clause expressions (all evaluated before loop entry, in
        the enclosing frame — ``break`` there is valid only if an outer loop
        already has ``_in_loop`` set).
        """
        # Resolve bound, for_iter, and the range-clause expressions in the enclosing
        # scope (before the loop variable is bound), so none of them can see the
        # loop variable.  Range expressions are resolved in source order: start (a),
        # then to/downto bound (b), then by step (k).
        if node.bound is not None:
            self._resolve_expr(node.bound)
        if node.for_iter is not None:
            self._resolve_expr(node.for_iter)
        if node.for_range_to is not None:
            self._resolve_expr(node.for_range_to)
        if node.for_range_step is not None:
            self._resolve_expr(node.for_range_step)
        with self._child_scope(node.node_id) as loop_scope:
            with self._resolution_flags_ctx(in_loop=True):
                # Bind for_var (immutable) before resolving while_cond/body/until_cond.
                if node.for_var is not None:
                    self._check_not_reserved(node.for_var, node.span)
                    ref = BindingRef(
                        name=node.for_var,
                        mutable=False,
                        decl_span=node.span,
                        decl_node_id=node.node_id,
                        kind=BinderKind.loop_var_binding,
                        module_id=self._module_id,
                    )
                    loop_scope.define(node.for_var, ref)
                if node.while_cond is not None:
                    self._resolve_expr(node.while_cond)
                if isinstance(node.body, Block):
                    # Inline block items directly — no extra block scope.
                    self._resolve_block_items(node.body.items)
                else:
                    self._resolve_expr(node.body)
                # until_cond sees all body bindings.
                if node.until_cond is not None:
                    self._resolve_expr(node.until_cond)

    def _resolve_try(self, node: Try) -> None:
        # Try body — its own scope.
        self._resolve_expr_or_block(node.body)
        # Each catch clause gets its own scope.
        for clause in node.handlers:
            self._resolve_catch_clause(clause)

    def _resolve_catch_clause(self, clause: CatchClause) -> None:
        if clause.exc_type is not None:
            self._select_type_name(clause.exc_type, clause.span, clause.node_id)
        with self._child_scope(clause.node_id) as catch_scope:
            if clause.binding is not None:
                self._check_not_reserved(clause.binding, clause.span)
                ref = BindingRef(
                    name=clause.binding,
                    mutable=False,
                    decl_span=clause.span,
                    decl_node_id=clause.node_id,
                    kind=BinderKind.catch_binder,
                    module_id=self._module_id,
                )
                catch_scope.define(clause.binding, ref)
            self._resolve_expr_or_block(clause.body)

    def _resolve_lambda(self, node: Lambda) -> None:
        """Resolve a ``fn(params) => body`` lambda.

        Defaults are resolved in the ENCLOSING scope (lambda is not in scope
        inside its own body — non-self-recursive).  Then a child scope is
        opened for the params + body.
        """
        # Defaults are resolved in the current (enclosing) scope.
        self._resolve_params_and_body(node)

    def _resolve_params_and_body(self, node: FuncDef | Lambda) -> None:
        """Resolve param defaults (enclosing scope), then params + body in a child scope.

        Shared by ``def`` and ``fn`` — both evaluate defaults in their definition
        scope and bind params into a fresh child scope for the body.

        ``_in_loop`` is reset to ``False`` for the entire method so that neither
        a ``break``/``continue`` in a parameter default nor one in the body can
        cross the function boundary into an outer loop.  Defaults are still
        resolved in the enclosing scope (only ``_in_loop`` changes, not the
        scope stack), so they can reference outer bindings but not the params.
        """
        with self._resolution_flags_ctx(in_loop=False, in_function=False):
            for param in node.params:
                if param.default is not None:
                    self._resolve_expr(param.default)
            with self._child_scope(node.node_id) as param_scope:
                for param in node.params:
                    self._check_not_reserved(param.name, param.span)
                    ref = BindingRef(
                        name=param.name,
                        mutable=False,
                        decl_span=param.span,
                        decl_node_id=param.node_id,
                        kind=BinderKind.param_binding,
                        module_id=self._module_id,
                    )
                    if param.name in param_scope.bindings:
                        raise DuplicateDeclarationError(param.name, span=param.span)
                    param_scope.define(param.name, ref)
                if node.body is not None:
                    with self._resolution_flags_ctx(in_function=True):
                        self._resolve_expr_or_block(node.body)

    # ------------------------------------------------------------------
    # Pattern variable binding
    # ------------------------------------------------------------------

    def _qualified_constructor_candidates(
        self, node_id: int, chain: QualifierChain, name: str
    ) -> tuple[ConstructorRef, ...]:
        """Select the constructors pattern or ``is`` spelling ``chain::name`` can match.

        Decided by :meth:`_select_qualified`, exactly as the same value
        spelling is. A selected ``def`` or ``let`` is no constructor: a
        scope's is an unknown member of that scope, a module's names no
        constructor. A module's root type without one is a type
        name, as its value is; any other type is left for the checker to
        report against the matched type. A selection made without a type
        owner -- a scope's member, or a module's -- is complete, so typecheck
        never re-reads its spelling as an owner's.
        """
        found = self._select_qualified(chain, name, LookupKind.CONSTRUCTOR, chain.span)
        target = found.target if isinstance(found, Misfit) else found
        if target.key is None or self._owner_less(target.key):
            self._scope_qualified_spellings.add(node_id)
        if target.constructor is not None:
            return (target.constructor,)
        ref = target.ref
        if ref is None or ref.kind is BinderKind.constructor_binding:
            if target.key is None or target.key[1]:
                return ()
            raise type_name_not_a_value(render_qualified_name(chain, name), chain.span)
        if ref.scope_path:
            raise _unknown_member(chain, name)
        raise AglScopeError(
            f"'{render_qualified_name(chain, name)}' names no constructor.", span=chain.span
        )

    def _route_injected_members(
        self, chain: QualifierChain, name: str
    ) -> dict[ConstructorRef, str]:
        """Map each root enum inline member one-segment route *chain* injects as *name*.

        A re-exported enum's members are injected too. Each maps to its
        owner-qualified spelling where *chain* is written: through *chain*
        when it matches one module, else through a route selecting only its
        exposing module.
        """
        surfaces = qualifier_members(self._import_env, chain.leading_route, anchored=chain.anchored)
        injected: dict[ConstructorRef, str] = {}
        for module, members in surfaces:
            for atom, origin in members.items():
                path = _bare_path(atom)
                constructor = self._cross_module_constructor_refs.get(origin)
                if (
                    path[1:] == (name,)
                    and constructor is not None
                    and _is_root_inline_member(constructor)
                ):
                    injected.setdefault(
                        constructor,
                        render_qualified_name(chain, "::".join(path))
                        if len(surfaces) == 1
                        else self._routed_spelling(origin, module, path),
                    )
        return injected

    def _pattern_constructors(self, name: str) -> tuple[ConstructorRef, ...]:
        """Return the constructor candidates a bare pattern or ``is`` spelling *name* reaches.

        Its scrutinee selects among them, so the nearest step declaring any
        of this module's own (:meth:`_own_step_constructors`, types' members
        included) and the nearest step any contribution reaches
        (:meth:`_contributed_step_constructors`) each supply theirs.
        """
        steps = lookup_steps(self._named_scope_path())
        own = next(
            (found for step in steps if (found := self._own_step_constructors(step, name))), {}
        )
        contributed = next(
            (
                found
                for step in steps
                if (
                    found := self._contributed_step_constructors(step, name, LookupKind.CONSTRUCTOR)
                )
            ),
            {},
        )
        return (*own, *(candidate for candidate in contributed if candidate not in own))

    def _own_step_constructors(
        self, step: ScopePath, name: str, *, nested: bool = True
    ) -> dict[ConstructorRef, Layers]:
        """Return the constructor candidates this module declares as bare *name* at *step*.

        At the module root, its own in the module-wide table, an enum's
        injected members included; in a named scope, the one it declares
        there and -- when *nested*, as a pattern or an ``is`` test reads it --
        one its types declare directly beneath them
        (:meth:`_owned_scope_constructor_candidates`).
        """
        candidates: Iterable[ConstructorRef]
        if not step:
            candidates = (
                candidate
                for candidate in self._constructor_candidates.get(name, ())
                if candidate.owner_module_id == self._module_id
            )
        elif nested:
            candidates = self._owned_scope_constructor_candidates(step, name)
        else:
            candidates = self._scoped_constructor_candidates.get((step, name), ())
        return dict.fromkeys(candidates, frozenset({ContributionLayer.DECLARED}))

    def _contributed_step_constructors(
        self, step: ScopePath, name: str, kind: LookupKind
    ) -> dict[ConstructorRef, Layers]:
        """Return the constructor candidates contributions anchored at or above *step* make *name*.

        The contributing layers' own candidates, each contributed import's
        constructor, and -- at the module root -- the ones import tails
        expose. *kind* is the position's.
        """
        path = (*step, name)
        found = self._contributed_constructors(step, path, kind)
        for ref, layers in self._contributed_bindings(step, path, kind).items():
            constructor = self._contributed_target(ref, ()).constructor
            if constructor is not None:
                add_layers(found, constructor, layers)
        if not step:
            for candidate in self._tail_constructors(name):
                add_layers(found, candidate, (ContributionLayer.IMPORTED,))
        return found

    def _tail_constructors(self, name: str) -> Iterator[ConstructorRef]:
        """Yield the other modules' constructors the module root's import tails make bare *name*."""
        for candidate in self._constructor_candidates.get(name, ()):
            if candidate.owner_module_id != self._module_id:
                yield candidate

    def _value_constructors(self, name: str) -> dict[ConstructorRef, Layers]:
        """Decide the constructor candidates a bare value or receiver *name* selects.

        Read in the bare value's order (:meth:`_bare_value`): at each step,
        this module's own candidates -- a named scope's own declarations only
        -- else the contributed ones. A root declaration shadows the enum
        members the module root injects under its name; several remaining
        candidates are ambiguous.
        """
        candidates: dict[ConstructorRef, Layers] = {}
        for step in lookup_steps(self._named_scope_path()):
            candidates = self._own_step_constructors(
                step, name, nested=False
            ) or self._contributed_step_constructors(step, name, LookupKind.VALUE)
            if candidates:
                break
        own = [
            candidate for candidate in candidates if candidate.owner_module_id == self._module_id
        ]
        declared = [candidate for candidate in own if not candidate.owner_path]
        return {candidate: candidates[candidate] for candidate in declared or own or candidates}

    def _visible_bare_constructor_candidates(
        self, name: str, span: SourceSpan
    ) -> tuple[ConstructorRef, ...]:
        """Return a bare pattern or ``is`` spelling's candidates, rejecting a spelling with none."""
        candidates = self._pattern_constructors(name)
        if not candidates:
            raise NoVisibleConstructorError(f"'{name}' is not a visible constructor.", span=span)
        return candidates

    def _owned_scope_constructor_candidates(
        self, scope_path: ScopePath, name: str
    ) -> tuple[ConstructorRef, ...]:
        """Return the union of *name*'s candidates declared directly in *scope_path*.

        Covers a record or exception constructor, registered at *scope_path*
        itself, and an enum variant, registered one level deeper at the type
        scope of an enum declared directly in *scope_path*. Every same-named
        candidate is returned, in declaration order, rather than the first
        non-empty bucket found: the checker's scrutinee-type selection
        disambiguates candidates, exactly as it does for the module-root
        candidate table, so stopping early here would hide a same-scope
        candidate behind an unrelated same-named one, non-deterministically.
        The types are earlier REPL entries' retained ones, then this entry's,
        each in declaration order.
        """
        type_names: dict[str, None] = dict.fromkeys(
            (
                *(path[-1] for path in self._repl_session_type_paths if path[:-1] == scope_path),
                *(item.name for item in self._type_declarations_by_path.get(scope_path, ())),
            )
        )
        candidates = list(self._scoped_constructor_candidates.get((scope_path, name), ()))
        for type_name in type_names:
            candidates.extend(
                self._scoped_constructor_candidates.get((scope_path + (type_name,), name), ())
            )
        return tuple(candidates)

    def _bind_pattern_vars(
        self, pattern: Pattern, scope: ScopeNode, match_site_node_id: int
    ) -> None:
        """Bind one case branch's pattern names into slots owned by *match_site_node_id*.

        The syntax helper is the sole pattern walker. Root bare names remain
        constructor-only, nested bare names remain field-directed, and ``as``
        names always bind.
        """
        match_site_slots: dict[str, int] = {}

        def record_constructor_candidates(node: object) -> None:
            if not isinstance(node, ConstructorPattern):
                return
            if node.qualifier is None:
                self._pattern_constructor_candidates[node.node_id] = (
                    self._visible_bare_constructor_candidates(node.name, node.span)
                )
                return
            # A qualified spelling never falls back to the bare one: every
            # failure yields an empty set so the checker can judge the
            # spelling against the type actually being matched.
            candidates = self._qualified_constructor_candidates(
                node.node_id, node.qualifier, node.name
            )
            self._pattern_constructor_candidates[node.node_id] = candidates
            if len(candidates) == 1:
                self._constructor_refs[node.node_id] = candidates[0]

        walk(pattern, record_constructor_candidates)
        for candidate in pattern_binder_candidates(pattern):
            constructor_candidates = (
                self._pattern_constructors(candidate.name) if not candidate.is_as_pattern else ()
            )
            if constructor_candidates:
                self._pattern_constructor_candidates[candidate.node_id] = constructor_candidates
            binds = candidate.is_as_pattern or candidate.nested
            if not binds:
                if not constructor_candidates:
                    raise NoVisibleConstructorError(
                        f"Bare case pattern '{candidate.name}' is not a visible constructor. "
                        "Use '_' or '_ as name' for a catch-all binder.",
                        span=candidate.span,
                    )
                continue
            self._add_pattern_slot_candidate(
                candidate.name,
                candidate.span,
                candidate.node_id,
                scope,
                match_site_slots,
                match_site_node_id,
                can_match_bare_pattern=any(
                    constructor.can_match_bare_pattern for constructor in constructor_candidates
                ),
            )
        self._match_site_pattern_slots_by_node[match_site_node_id] = tuple(
            sorted(match_site_slots.values())
        )

    def _add_pattern_slot_candidate(
        self,
        name: str,
        span: SourceSpan,
        pattern_node_id: int,
        scope: ScopeNode,
        match_site_slots: dict[str, int],
        match_site_node_id: int,
        *,
        can_match_bare_pattern: bool,
    ) -> None:
        """Join a candidate to its match-site-local shared binding.

        Reject duplicate binders immediately only when neither the arriving
        candidate nor any prior slot candidate can match a bare pattern.
        """
        self._check_not_reserved(name, span)
        slot_id = match_site_slots.get(name)
        if slot_id is None:
            slot_id = self._next_pattern_slot_id
            self._next_pattern_slot_id += 1
            match_site_slots[name] = slot_id
            alternative = (
                self._lookup_value(name, scope.parent) if scope.parent is not None else None
            )
            self._pattern_slots[slot_id] = PatternSlot(
                slot_id=slot_id,
                name=name,
                candidates=(),
                alternative=alternative,
                match_site_node_id=match_site_node_id,
            )
            self._define(
                name,
                BindingRef(
                    name=name,
                    mutable=False,
                    decl_span=span,
                    decl_node_id=pattern_node_id,
                    kind=BinderKind.pattern_slot,
                    module_id=self._module_id,
                    slot_id=slot_id,
                ),
            )
        slot = self._pattern_slots[slot_id]
        if (
            slot.candidates
            and not can_match_bare_pattern
            and not any(candidate.can_match_bare_pattern for candidate in slot.candidates)
        ):
            raise AglScopeError(duplicate_binder_message(name), span=span)
        self._pattern_slots[slot_id] = replace(
            slot,
            candidates=(
                *slot.candidates,
                SlotCandidate(
                    pattern_node_id=pattern_node_id,
                    span=span,
                    can_match_bare_pattern=can_match_bare_pattern,
                ),
            ),
        )
