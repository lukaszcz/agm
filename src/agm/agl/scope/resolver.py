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

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
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
from agm.agl.modules.ids import RESERVED_ID, ModuleId, spell_declaration
from agm.agl.scope.attributes import recognize_attributes
from agm.agl.scope.imports import (
    BareRoute,
    ImportEnv,
    NameAtom,
    QName,
    QualResolution,
    QualResolutionAmbiguous,
    QualResolutionFound,
    declares_bare_constructor,
    qualifier_candidates,
    qualifier_contributes,
    qualifier_members,
    qualifier_scope_paths,
    render_qualifier,
    resolve_qualified,
    resolve_qualified_member,
    route_spelling,
)
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    BUILTIN_METHOD_RECEIVER_NAMES,
    BUILTIN_TYPE_STATIC_OWNER_PATHS,
    AglScopeError,
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    BareContributionSource,
    BinderKind,
    BindingRef,
    BuiltinKind,
    BuiltinStaticKind,
    ConstructorRef,
    DeclarationKey,
    DeclaredOrigin,
    DeclInfo,
    ImmutableAssignmentError,
    ImportedModuleOrigin,
    ImportedUseContribution,
    LocalUseContribution,
    ModuleResolution,
    NoVisibleConstructorError,
    PatternSlot,
    QualificationOrigin,
    ReceiverOwner,
    ResolvedUseTarget,
    RouteClashError,
    ScopeNode,
    ScopePath,
    SlotCandidate,
    TypeOwner,
    UnknownMemberError,
    UnknownQualifierError,
    UseDeclarationOrigin,
    builtin_call_kind,
    builtin_type_static_kind,
    contributed_declarations,
    duplicate_binder_message,
    is_builtin_type_static_owner,
    is_qualified_function_member,
    qualification_repair_guidance,
    undefined_name_message,
    unnameable_use_target_error,
)
from agm.agl.scope.symbols import binding_qname as _ref_qname
from agm.agl.scope.symbols import import_item_path as _item_path
from agm.agl.scope.symbols import to_bare_atom as _bare_atom
from agm.agl.scope.symbols import to_bare_path as _bare_path
from agm.agl.scope.type_names import (
    MemberHidden,
    MemberReferenced,
    OwnerRoute,
    TypeNameSite,
    owner_member_selection,
    owner_type_expr,
    routed_qualifier_and_member,
    type_name_selection,
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
    Pattern,
    Placeholder,
    Program,
    QualifierAnchor,
    QualifierChain,
    Raise,
    RecordDef,
    RecordUpdate,
    Return,
    ScopeRegion,
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
from agm.agl.syntax.qualifiers import enclosing_scope_bases
from agm.agl.syntax.spans import SourceSpan
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
    local: ScopePath | None
    route: tuple[str, ...]
    route_target: ScopePath
    direct_candidates: tuple[tuple[ModuleId, Mapping[NameAtom, QName]], ...]
    direct_imports: tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...]
    direct_import_scope_routes: Mapping[BareRoute, Mapping[NameAtom, frozenset[BareRoute]]]
    imported: tuple[tuple[BareRoute, Mapping[NameAtom, QName]], ...]


# Built-in call names, sourced from ``symbols.BUILTIN_CALL_NAMES`` (the single
# source of truth). Runtime operations are first-class; ``resource`` remains a
# link-time form whose argument must be visible in its direct call syntax.
_BUILTIN_CALL_NAMES = BUILTIN_CALL_NAMES

# The set of names that may NOT be used as any kind of binding.
_RESERVED_NAMES: frozenset[str] = frozenset(_BUILTIN_CALL_NAMES)

_TEXTUALLY_ORDERED_BINDER_KINDS: frozenset[BinderKind] = frozenset(
    {
        BinderKind.let_binding,
        BinderKind.var_binding,
        BinderKind.param_binding,
        BinderKind.pattern_slot,
    }
)


def _scope_path_sort_key(path: ScopePath) -> tuple[int, ScopePath]:
    """Order scope paths by depth, then lexical spelling."""
    return (len(path), path)


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


def _unknown_scope_member(name: str, path: ScopePath, span: SourceSpan) -> UnknownMemberError:
    """Return the error for local scope *path* declaring no member *name*."""
    return UnknownMemberError(
        f"Unknown member '{name}' in scope path '{'::'.join(path)}'.", span=span
    )


def _use_route_unknown_member(rendered: str, span: SourceSpan) -> UnknownMemberError:
    """Return the error for a use route that resolves but selects no constructor *rendered*."""
    return UnknownMemberError(
        f"Constructor '{rendered}' is not visible through this use route.", span=span
    )


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
        # this entry's contribution re-derivation never mutates session state.
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
        # Optional REPL session scope for ``::name`` self-references.
        # When set, ``_lookup_own_root`` consults this scope for names not
        # in the entry's own root scope, allowing ``::name`` to resolve to a
        # prior session binding.
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
        self._current_use_declaration_ids: set[int] = set()
        self._program = program
        # The module's root ScopeNode; used by _lookup_own_root to bypass
        # lexical shadows introduced by nested scopes for ::name.
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
        # The declaration item accompanies its path-keyed binding so legacy
        # root-only tables can be derived from the same collection without
        # admitting scoped members.
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
        self._scope_region_paths: set[ScopePath] = {
            path for path, node in self._repl_session_scope_nodes.items() if node.is_scope_region
        }
        self._ordered_binding_paths: set[ScopePath] = set()
        self._scope_node_ids: dict[ScopePath, int] = {
            path: node.node_id for path, node in self._repl_session_scope_nodes.items() if path
        }
        self._type_paths: set[ScopePath] = set(self._repl_session_type_paths)
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
        self._owner_declarations: dict[int, DeclarationKey] = {}
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
        # The inline wrapper moves static bindings ahead of its synthetic
        # entry. While resolving that entry, source offsets preserve the
        # bindings' original textual visibility despite the AST partition.
        self._in_synthetic_entry: bool = False
        # The synthetic entry's own body items, which the wrapper filled with
        # what the user wrote at the root. A rejection there is phrased in the
        # terms the source is written in, not in terms of the generated block.
        self._synthetic_entry_items: tuple[Item, ...] | None = None
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
        for complete in self._deferred_constructors:
            complete()
        if ambient_constructor_candidates:
            for cname, crefs in ambient_constructor_candidates.items():
                for cref in crefs:
                    self._add_constructor_candidate(cname, cref)
        type_owners = self._declared_type_owners()
        # A retained path's owner is re-derived through the index rather than
        # read off its stored, declaration-time value: an indirect alias's
        # reachable members/hidden set can go stale as later entries change
        # what is imported (``TypeOwnerIndex.owner``). Constructor candidates
        # read this one derivation.
        current_type_owners = {
            path: owner
            for path in {**self._repl_session_type_paths, **type_owners}
            if (owner := self._type_owners.owner((self._module_id, _bare_atom(path)))) is not None
        }
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
        self._resolve_block_items(program.body.items)
        self._validate_function_names()
        self._validate_non_method_type_params()
        # Receiver classification follows the ordered lexical walk, so attribute
        # recognition runs only after every method declaration is known.
        attribute_facts = recognize_attributes(program, declares_receiver=self._declares_receiver)
        self._validate_extern_backing()
        self._validate_local_use_contributions()
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
            constructor_candidates_by_path={
                key: tuple(refs) for key, refs in self._scoped_constructor_candidates.items()
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
            type_owners=type_owners if self._allow_root_statements else {},
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
            self._scope_region_paths.add(path)
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
                raise AglScopeError(f"Name '{name}' is already declared in this scope.", span=span)
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
                raise AglScopeError(
                    f"Name '{item.name}' is already declared in this scope.", span=item.span
                )
            self._scope_entity_kinds[key] = "type"
        else:
            if existing_entity is not None:
                raise AglScopeError(
                    f"Name '{item.name}' is already declared in this scope.", span=item.span
                )
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
                    raise AglScopeError(
                        f"Name '{member.name}' is already declared in this scope.",
                        span=member.span,
                    )
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
        textual order, in every module.

        A static-root module additionally installs the binding itself, like
        ``_declared_functions`` does for a def: ``_declarations``/
        ``_declaration_items`` make it a scope member before the ordered walk
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
            raise AglScopeError(f"Name '{name}' is already declared in this scope.", span=item.span)
        self._scope_entity_kinds[key] = "ordinary"
        decl_node_id = static_binding_node_id(item)
        if path and not self._is_static_root_module:
            return
        self._declarations[key] = self._binder_ref(
            item, decl_node_id=decl_node_id, name=name, scope_path=path
        )
        self._declaration_items[key] = item

    def _register_builtin_var_declaration(self, item: BuiltinVarDecl, path: ScopePath) -> None:
        """Claim a ``builtin var``'s name early, like a static let/var.

        A ``builtin var`` is declared only by a standard-library module, which
        is always file-backed and so always static-root: unlike a scoped
        simple let/var, there is no non-static-root case, so this always
        installs the binding alongside claiming the name (mirrors
        :meth:`_register_static_binding_declaration`). The module-kind and
        root-only checks stay in :meth:`_resolve_builtin_var`, at the walk's
        ordered position, since a misplaced declaration must still fail
        regardless of pre-pass timing.
        """
        key = (self._module_id, path, item.name)
        existing_entity = self._scope_entity_kinds.get(key)
        if existing_entity is not None:
            raise AglScopeError(
                f"Name '{item.name}' is already declared in this scope.", span=item.span
            )
        self._scope_entity_kinds[key] = "ordinary"
        self._declarations[key] = self._binder_ref(
            item, decl_node_id=item.node_id, name=item.name, scope_path=path
        )
        self._declaration_items[key] = item

    def _classify_method_declaration(self, declaration: FuncDef) -> None:
        """Classify one receiver after preceding lexical contributions are visible."""
        if not declaration.is_method:
            return
        aliases = self._alias_receiver_paths()
        receiver = declaration.params[0]
        owner_path = tuple(segment.name for segment in declaration.scope_path)
        region_path, type_path = self._receiver_region_and_type_path(owner_path)
        owner = (
            self._local_receiver_owner(owner_path, declaration)
            if declaration.receiver_type is not None
            else self._lexical_receiver_owner(region_path, type_path, aliases, receiver.span)
        )
        if owner is None:
            owner = self._local_receiver_owner(owner_path, declaration)
        if owner is None:
            owner = self._foreign_receiver_owner(owner_path, receiver.span)
        if owner is None:
            if receiver.type_expr is None:
                self._reject_referenced_receiver(region_path, type_path, receiver.span)
                raise AglScopeError("'self' requires an enclosing type scope.", span=receiver.span)
            return
        if receiver.default is not None:
            raise AglScopeError(
                f"Receiver 'self' for method '{declaration.name}' cannot have a default value.",
                span=receiver.span,
            )
        key = (self._module_id, owner_path, declaration.name)
        self._method_declarations[key] = owner

    def _reject_referenced_receiver(
        self, region_path: ScopePath, type_path: ScopePath, span: SourceSpan
    ) -> None:
        """Raise when a receiver path spells a member its enum owner only references."""
        if len(type_path) < 2:
            return
        owner_path = type_path[:-1]
        owner = self._lexical_receiver_owner(region_path, owner_path, {}, span)
        if owner is None:
            owner = self._foreign_receiver_owner((*region_path, *owner_path), span)
        if owner is None:
            return
        self._select_owner_member(
            self._type_owners.owner((owner.module_id, _bare_atom(owner.scope_path))),
            "::".join(owner_path),
            type_path[-1],
            span,
            None,
        )

    def _select_owner_member(
        self,
        owner: TypeOwner | None,
        spelling: str,
        member: str,
        span: SourceSpan | None,
        route: OwnerRoute | None,
    ) -> None:
        """Raise if ``spelling::member`` is currently unreachable through *owner*, by identity.

        Raises :class:`ReferencedMemberError` or the hidden-member error for
        a referenced or hidden member; does nothing when *owner* itself is
        unresolved (an unknown or ambiguous owner, left to typecheck's
        existing diagnostics) or when the member selects. *route*, when
        given, live-filters a nominal (non-alias) owner reached indirectly
        (see :func:`~agm.agl.scope.type_names.owner_member_selection`).
        """
        if owner is None:
            return
        selection = owner_member_selection(owner, member, route)
        if isinstance(selection, MemberReferenced):
            raise ReferencedMemberError(spelling, member, span=span)
        if isinstance(selection, MemberHidden):
            raise HiddenMemberError(f"{spelling}::{member}", span=span)

    @staticmethod
    def _unknown_owner_member(
        spelling: str, member: str, span: SourceSpan | None
    ) -> UnknownMemberError:
        """The one verdict for a resolved owner declaring no ``member`` at all, in any position."""
        return UnknownMemberError(f"'{member}' is not a member of '{spelling}'.", span=span)

    def _select_owner_member_or_raise(
        self,
        owner: TypeOwner,
        spelling: str,
        member: str,
        written: str,
        span: SourceSpan | None,
        route: OwnerRoute | None,
    ) -> ConstructorRef:
        """Return the constructor ``spelling::member`` selects from *owner*, already resolved.

        Runs the hiding/referenced check, then raises the one unknown-member
        verdict when *owner* declares no such *member* at all -- in every
        position that resolves an owner this way (value, pattern, ``is``,
        annotation, alias target, applied type).
        """
        self._select_owner_member(owner, spelling, member, span, route)
        constructor = owner.select(member, written)
        if constructor is None:
            raise self._unknown_owner_member(spelling, member, span)
        return constructor

    @staticmethod
    def _owner_member_key(owner_qname: QName, member: str) -> DeclarationKey:
        """Return the declaration identity ``owner_qname::member`` names, by nested path."""
        module_id, atom = owner_qname
        return (module_id, _bare_path(atom), member)

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

    def _local_receiver_owner(
        self, owner_path: ScopePath, declaration: FuncDef
    ) -> ReceiverOwner | None:
        """Return a local nominal or builtin receiver owner, if one is declared."""
        if declaration.receiver_type is not None:
            return ReceiverOwner(self._module_id, owner_path)
        _region_path, type_path = self._receiver_region_and_type_path(owner_path)
        if len(type_path) == 1 and type_path[0] in BUILTIN_METHOD_RECEIVER_NAMES:
            return ReceiverOwner(self._module_id, type_path)
        return None

    def _local_nominal_receiver_owner(self, path: ScopePath) -> ReceiverOwner | None:
        """Return the local nominal type declaration at *path*, if present."""
        if not path:
            return None
        key = (self._module_id, path[:-1], path[-1])
        if path in self._type_paths and (
            key in self._declaration_items
            or self._scope_entity_kinds.get(key) == "type"
            or path in self._repl_session_type_paths
        ):
            return ReceiverOwner(self._module_id, path)
        return None

    def _lexical_receiver_owner(
        self,
        region_path: ScopePath,
        type_path: ScopePath,
        aliases: Mapping[ScopePath, TypeAlias],
        span: SourceSpan,
    ) -> ReceiverOwner | None:
        """Resolve a local receiver from the nearest enclosing lexical region."""
        for base in enclosing_scope_bases(region_path):
            candidate = (*base, *type_path)
            alias = aliases.get(candidate)
            if alias is not None:
                self._raise_alias_receiver(candidate[-1], alias, span)
            owner = self._local_nominal_receiver_owner(candidate)
            if owner is not None:
                return owner
        return None

    def _foreign_receiver_owner(
        self, owner_path: ScopePath, span: SourceSpan
    ) -> ReceiverOwner | None:
        """Resolve a receiver through the ordinary nearest bare contribution layer."""
        _region_path, type_path = self._receiver_region_and_type_path(owner_path)
        if not type_path:
            return None
        atom = _bare_atom(type_path)

        def is_type(ref: BindingRef) -> bool:
            source = _bare_atom((*ref.scope_path, ref.name))
            key = (ref.module_id, source)
            return key in self._cross_module_type_owners or isinstance(
                self._all_public_types.get(key), TypeAlias
            )

        nearest = self._nearest_bare_contribution_layer(atom, binding_predicate=is_type)
        refs: set[BindingRef] = set()
        selected_layer: ScopeNode | None = None
        if nearest is not None:
            selected_layer, refs = nearest
        imported_refs = {
            ref
            for qname in self._import_env.unqualified.get(atom, frozenset())
            if is_type(ref := self._cross_module_binding_ref(qname))
        }
        if selected_layer is None:
            refs = imported_refs
        elif selected_layer is self._root_scope:
            refs.update(imported_refs)

        owners: set[ReceiverOwner] = set()
        if not refs and len(type_path) == 1:
            for exposed, qnames in self._import_env.unqualified.items():
                if _bare_path(exposed)[-1:] != type_path:
                    continue
                for module_id, source in qnames:
                    source_path = _bare_path(source)
                    parent = _bare_atom(source_path[:-1])
                    if isinstance(self._all_public_types.get((module_id, parent)), EnumDef):
                        owner = self._cross_module_type_owners.get((module_id, source))
                        if owner is not None:
                            owners.add(owner)
            owners.update(
                ReceiverOwner(
                    candidate.owner_module_id, (*candidate.owner_path, candidate.owner_name)
                )
                for candidate in self._bare_constructor_candidates(type_path[0])
            )
        for ref in refs:
            source = _bare_atom((*ref.scope_path, ref.name))
            declaration = self._all_public_types.get((ref.module_id, source))
            if isinstance(declaration, TypeAlias):
                self._raise_alias_receiver(type_path[0], declaration, span)
            owners.add(self._cross_module_type_owners[(ref.module_id, source)])
        if len(owners) == 1:
            return next(iter(owners))
        if len(owners) > 1:
            raise AmbiguousQualificationError.for_origins(
                (),
                type_path,
                (
                    self._declaration_origin((owner.module_id, _bare_atom(owner.scope_path)))
                    for owner in owners
                ),
                span=span,
                local_to=self._module_id,
            )
        return None

    def _receiver_region_and_type_path(self, owner_path: ScopePath) -> tuple[ScopePath, ScopePath]:
        """Split a method path at its longest prefix of plain scope regions."""
        region_length = 0
        for length in range(1, len(owner_path)):
            prefix = owner_path[:length]
            key = (self._module_id, prefix[:-1], prefix[-1])
            if (
                prefix in self._scope_region_paths
                and self._scope_entity_kinds.get(key) != "type"
                and prefix not in self._repl_session_type_paths
            ):
                region_length = length
            else:
                break
        return owner_path[:region_length], owner_path[region_length:]

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
                    is_scope_region=path in self._scope_region_paths,
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
                # every layer, by ``_validate_local_use_contributions``.
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
                    raise AglScopeError(
                        f"Name '{name}' is already declared in this scope.",
                        span=self._root_declaring_span(declaring[0]),
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
        is otherwise a member like any other: legal at the module root and
        inside a named scope region (both keep ``_at_root`` set), rejected
        only inside a nested block. The typecheck pass reserves engine-setting
        names and types to ``std/config``.
        """
        if not self._at_root:
            raise AglScopeError(
                f"'builtin var' declarations are only allowed at the module root, "
                f"not inside a nested block (found 'builtin var {node.name}' here).",
                span=node.span,
            )
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
            raise AglScopeError(
                f"Name '{name}' is already declared in this scope.",
                span=ref.decl_span,
            )
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

    def _resolve_block_items(self, items: tuple[Item, ...]) -> None:
        """Resolve items in order; each binder adds to the current scope.

        This is the core sequencing logic.  Binders (``LetDecl``, ``VarDecl``,
        ``AssignStmt``) and declarations (``FuncDef``, etc.) that
        are not pure expressions are handled first; everything else is treated
        as an expression item.

        Additional enforcement:

        - Every static module root rejects assignments and bare expressions.
          The REPL is the sole incremental-host exception.
        - Every module root and every region's own item sequence: ``ImportDecl`` and
          ``ExportDecl`` must precede all other items *in that same items
          sequence* (header-only; ``seen_non_import_item`` tracks this locally
          to this call, regardless of module kind). A region is one item of its
          enclosing sequence for this purpose, so the enclosing sequence's
          header rule governs a region as a whole, while its own header rule --
          tracked by the recursive call this function makes for the region's
          body -- governs the region's own items independently. A synthetic
          entry's own body holds the items the source wrote at its root, so a
          header there is reported as the header-ordering violation it is.
        """
        seen_non_import_item = False

        for item in items:
            if isinstance(item, UseDecl):
                # Resolved with the headers (``_resolve_headers``).
                continue
            if isinstance(item, (ImportDecl, ExportDecl)):
                # A header the entry transform moved into the synthetic body is
                # one the source wrote after an executable item: the rule it
                # broke is the ordering one, stated in the source's own terms
                # rather than in terms of the generated block.
                in_entry_body = items is self._synthetic_entry_items
                if not self._at_root and not in_entry_body:
                    kind = "import" if isinstance(item, ImportDecl) else "export"
                    raise AglScopeError(
                        f"'{kind}' declarations are only allowed at the program root, "
                        "not inside a nested block.",
                        span=item.span,
                    )
                if seen_non_import_item or in_entry_body:
                    raise AglScopeError(
                        "Import and export declarations must appear before any other "
                        "declarations in a module or scope region.",
                        span=item.span,
                    )
                # The program module-system pass processes imports/exports; a
                # region-scoped import's contribution is resolved with the headers.
                continue
            if isinstance(item, InfixDecl):
                if not self._at_root:
                    raise AglScopeError(
                        "infix declarations are only allowed at the program root.",
                        span=item.span,
                    )
                seen_non_import_item = True
                continue
            # Header enforcement: track that a non-import item has been seen.
            seen_non_import_item = True
            # Named declarations (def/record/enum/exception/type, and a
            # scoped let/var binder) validate their own whole subtree,
            # including any nested blocks, in their own handlers after
            # entering the right lexical scope. A scope region keeps
            # `_at_root` set and revalidates its own items one by one through
            # this same walk. Every other root item is validated here; a
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
                if self._at_root and not self._allow_root_statements:
                    raise AglScopeError(
                        static_root_message(
                            "Assignment statements are not allowed at a static module root.",
                            subject="statements",
                            file_backed=self._origin_path is not None,
                            declares_program_entry=self._declares_program_entry,
                        ),
                        span=item.span,
                    )
                self._resolve_assign(item)
            else:
                # Pure expression item (Expr union).
                if self._at_root and not self._allow_root_statements:
                    raise AglScopeError(
                        static_root_message(
                            "Bare expressions are not allowed at a static module root.",
                            subject="statements",
                            file_backed=self._origin_path is not None,
                            declares_program_entry=self._declares_program_entry,
                        ),
                        span=item.span,
                    )
                self._resolve_expr(item)

    def _resolve_headers(self, items: tuple[Item, ...]) -> None:
        """Contribute the ``use`` and region-scoped ``import`` headers of *items* and its regions.

        A header contributes bindings at once; a constructor that depends on
        type owners is deferred to ``resolve``. An import after the header
        prefix contributes nothing: the ordered walk rejects it.
        """
        regional_exposures = self._regional_import_exposures(items)
        in_header = True
        for item in items:
            if isinstance(item, UseDecl):
                self._resolve_use_decl(item)
            elif isinstance(item, ImportDecl):
                if in_header and item.scope_path:
                    self._contribute_regional_import_bare(item, exposures=regional_exposures)
            elif not isinstance(item, ExportDecl):
                in_header = False
                if isinstance(item, ScopeRegion):
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
                    atom, self._cross_module_binding_ref(qname), BareContributionSource.IMPORT_TAIL
                )
                self._deferred_constructors.append(
                    partial(self._contribute_bare_constructor, scope, atom, qname)
                )
                if isinstance(atom, str):
                    self._contribute_regional_enum_variants(qname, decl.span, exposures=exposures)

    def _contribute_regional_enum_variants(
        self, qname: QName, span: SourceSpan, *, exposures: Mapping[NameAtom, Collection[QName]]
    ) -> None:
        """Expand a bare-exposed enum type into its own bare variants, region-scoped.

        A bare enum *type* name alone does not make its variants callable or
        matchable -- ``_build_cross_module_constructor_candidates`` performs
        the same expansion module-wide, from a root-position bare exposure.
        Mirroring it here covers the scoped case, whose bare exposure never
        reaches that module-wide table.
        """
        declaration = self._all_public_types.get(qname)
        if not isinstance(declaration, EnumDef):
            return
        module, source = qname
        owner_path = _bare_path(source)
        scope = self._scope
        selected_qnames = frozenset(qname for qnames in exposures.values() for qname in qnames)
        for member in declaration.members:
            if isinstance(member, VariantRef):
                self._deferred_constructors.append(
                    partial(self._contribute_referenced_member, scope, module, member, span)
                )
                continue
            # A same-named record or exception exposed bare already owns the
            # spelling; its own bare contribution stands alone.
            if declares_bare_constructor(exposures.get(member.name, ()), self._all_public_types):
                continue
            variant_qname = (module, _bare_atom((*owner_path, member.name)))
            if variant_qname not in selected_qnames:
                continue
            scope.contribute_bare(
                member.name,
                replace(self._cross_module_binding_ref(variant_qname), is_variant_member=True),
                BareContributionSource.IMPORT_TAIL,
            )
            scope.contribute_bare_constructor(
                member.name, self._cross_module_constructor_refs[variant_qname]
            )

    def _contribute_referenced_member(
        self, scope: ScopeNode, module: ModuleId, member: VariantRef, span: SourceSpan
    ) -> None:
        """Contribute the record constructors a bare-exposed enum's member reference denotes."""
        for constructor in self._type_owners.referenced_member_refs(module, member):
            scope.contribute_bare(
                constructor.owner_name,
                self._variant_binding_ref(constructor, span),
                BareContributionSource.IMPORT_TAIL,
            )
            scope.contribute_bare_constructor(constructor.owner_name, constructor)

    def _contribute_bare_constructor(
        self, scope: ScopeNode, atom: NameAtom, qname: QName
    ) -> ConstructorRef | None:
        """Contribute the constructor an imported *qname* names bare to *scope*, if any."""
        constructor = self._cross_module_constructor(qname)
        if constructor is not None:
            scope.contribute_bare_constructor(atom, constructor)
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
                        rendered = "::".join(path)
                        raise unnameable_use_target_error(
                            f"{module.display()}::{rendered}",
                            span=self._import_env.decl_spans[module],
                        )
            scope = scope.parent

    def _resolve_use_target(self, decl: UseDecl) -> _UseTargetResolution:
        """Gather every local and imported route reachable through a use target."""
        if decl.alias is not None and len(decl.target) >= 2:
            parent_decl = replace(decl, target=decl.target[:-1], alias=None)
            parent = self._resolve_use_target(parent_decl)
            member = decl.target[-1].name
            local_member = (
                parent.local is not None
                and (*parent.local, member) not in self._scope_nodes
                and (
                    member in self._scope_nodes[parent.local].members
                    or (*parent.local, member) in self._ordered_binding_paths
                )
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
        local = self._use_local_target(decl, target)
        if local is None and not decl.anchored:
            local = self._use_contributed_local_target(target, decl.span)
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
            local=local,
            route=route,
            route_target=route_target,
            direct_candidates=direct_candidates,
            direct_imports=direct_imports,
            direct_import_scope_routes=direct_import_scope_routes,
            imported=self._merge_use_import_targets(direct_imports, bare_imports, used_imports),
        )

    def _resolve_use_decl(self, decl: UseDecl) -> None:
        """Inject the selected members of one already-nameable route bare."""
        resolved_target = self._resolve_use_target(decl)
        decl = resolved_target.declaration
        resolved_identity = ResolvedUseTarget(
            local_path=resolved_target.local,
            imported_routes=tuple(route for route, _members in resolved_target.imported),
        )
        self._superseded_use_targets.add(resolved_identity)
        self._current_use_declaration_ids.add(decl.node_id)
        target = resolved_target.target
        local = resolved_target.local
        route = resolved_target.route
        route_target = resolved_target.route_target
        direct_candidates = resolved_target.direct_candidates
        direct_imports = resolved_target.direct_imports
        direct_import_scope_routes = resolved_target.direct_import_scope_routes
        imported = resolved_target.imported
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
        direct_routes = {imported_route for imported_route, _members in direct_imports}
        shared_alias_facade = (
            not decl.anchored
            and len(direct_candidates) > 1
            and tuple(facade_declarations.values())
            == (frozenset(module for module, _members in direct_candidates),)
            and {imported_route for imported_route, _members in imported} == direct_routes
        )
        if local is not None and imported:
            candidates = ", ".join(module.display() for (module, _root), _members in imported)
            module_targets = ", ".join(
                self._render_use_module_target(
                    imported_route[0],
                    route_target if imported_route in direct_routes else target,
                )
                for imported_route, _members in imported
            )
            rendered = "::".join(target)
            raise RouteClashError(
                f"Use target '{rendered}' is both local scope '{'::'.join(local)}' and imported "
                f"module route(s): {candidates}. Use {module_targets} to select the module route "
                f"or ::{rendered} to select the local scope.",
                span=decl.span,
            )
        if len(imported) > 1 and not shared_alias_facade:
            raise AmbiguousQualificationError.for_origins(
                route[:-1],
                (route[-1],),
                (ImportedModuleOrigin((module, root)) for (module, root), _members in imported),
                anchored=decl.anchored,
                span=decl.span,
                local_to=self._module_id,
            )
        if local is None and not imported:
            rendered = "/".join(route) if route else "::".join(target)
            raise unnameable_use_target_error(rendered, span=decl.span)
        if local is not None:
            self._use_targets[decl.node_id] = ResolvedUseTarget(local_path=local)
            self._scope.contribute_local_use(
                LocalUseContribution(
                    declaration=decl,
                    source=self._scope_nodes[local],
                    target=self._use_targets[decl.node_id],
                )
            )
            return

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

        if shared_alias_facade:
            self._use_targets[decl.node_id] = ResolvedUseTarget(
                imported_routes=tuple(route for route, _members in imported),
                wildcard_facade_origin_node_id=facade_origin_node_id,
            )
            self._contribute_use_facade_members(
                decl,
                tuple(members for _route, members in imported),
                tuple(scope_routes_for(route) for route, _members in imported),
            )
            return
        imported_route, imported_members = imported[0]
        self._use_targets[decl.node_id] = ResolvedUseTarget(
            imported_routes=(imported_route,),
            wildcard_facade_origin_node_id=facade_origin_node_id,
        )
        self._contribute_use_members(decl, imported_members, scope_routes_for(imported_route))

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

    @staticmethod
    def _render_use_module_target(module: ModuleId, target: ScopePath) -> str:
        """Render an anchored, reachable module reading of a use target."""
        suffix = "" if not target else f"::{'::'.join(target)}"
        return f"/{module.path_str()}{suffix}"

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
            module, path = imported_route
            origins = self._import_env.scope_origins_by_route.get(
                imported_route, frozenset({(module, _bare_atom(path))})
            )
            _representative, merged = grouped.setdefault(origins, (imported_route, {}))
            for atom, qname in members.items():
                merged.setdefault(atom, qname)

        return tuple(sorted(grouped.values(), key=_keyed_bare_route))

    def _use_local_target(self, decl: UseDecl, target: ScopePath) -> ScopePath | None:
        """Resolve a use target through exact lexical scope paths."""
        if decl.anchored and not decl.current_module:
            return None
        bases = [()] if decl.current_module else self._lexical_scope_bases()
        return next((base + target for base in bases if base + target in self._scope_nodes), None)

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
        consumers below and by :meth:`_validate_local_use_contributions`'s
        bare-contribution snapshot, so a superseded contribution's members
        never leak into a static ``bare_contributions`` read that (unlike the
        live consumers) applies no filter of its own.
        """
        return (
            contribution.target in self._superseded_use_targets
            and contribution.declaration.node_id not in self._current_use_declaration_ids
        )

    def _use_contributed_local_target(
        self, target: ScopePath, span: SourceSpan
    ) -> ScopePath | None:
        """Resolve a scope route exposed by an earlier local ``use``."""
        layer: ScopeNode | None = self._scope
        while layer is not None:
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
                    candidate = (
                        source_path if trailing_length == 0 else source_path[:-trailing_length]
                    )
                    if candidate in self._scope_nodes:
                        candidates.add(candidate)
            if len(candidates) > 1:
                raise AmbiguousQualificationError.for_origins(
                    (),
                    target,
                    (
                        UseDeclarationOrigin((self._module_id, candidate))
                        for candidate in candidates
                    ),
                    span=span,
                    local_to=self._module_id,
                )
            if candidates:
                return next(iter(candidates))
            layer = layer.parent
        return None

    def _local_use_exposures(
        self, contribution: LocalUseContribution, *, validate: bool = False
    ) -> list[tuple[NameAtom, BindingRef | _LocalScopeRoute]]:
        """Expand one local ``use`` into every bare spelling it exposes.

        The single definition of the select-then-rename protocol: a use's tail
        selection and its additive renames both draw on the same snapshot of
        the target subtree, so every consumer sees one consistent surface.
        """
        source_members = self._local_use_members(contribution.source.scope_path)
        selected = self._select_use_members(
            contribution.declaration, source_members, validate=validate
        )
        return [
            *selected.items(),
            *self._use_renamed_members(contribution.declaration, source_members),
        ]

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

    def _validate_local_use_contributions(self) -> None:
        """Re-derive every visible layer's stored bare tables from its current uses.

        Runs for every named layer this entry built plus, when this entry has
        a REPL session parent, that parent layer too -- the one layer
        :meth:`_build_scope_nodes` never visits, whose stored tables would
        otherwise go stale relative to its live derivation
        (:meth:`_layer_bare_bindings`).
        """
        layers: Iterable[ScopeNode] = self._scope_nodes.values()
        if self._repl_session_scope is not None:
            layers = (*layers, self._repl_session_scope)
        for layer in layers:
            self._refresh_layer_contributions(layer)

    def _refresh_layer_contributions(self, layer: ScopeNode) -> None:
        """Make *layer*'s stored bare tables equal its live derivation, re-snapshotting its uses.

        Only uses declared in the current entry are re-validated below: an
        earlier entry's use was already validated when it was declared, and
        re-validating it here would turn a later, unrelated redeclaration
        into a static error on that old entry instead of on the entry that
        actually changed. Re-snapshotting is unconditional -- it is not a
        validity check -- so it runs for every use regardless of which entry
        declared it.
        """
        layer.imported_use_contributions = [
            self._refresh_imported_use(layer, imported_contribution)
            for imported_contribution in layer.imported_use_contributions
        ]
        rebuilt: list[LocalUseContribution] = []
        for local_contribution in layer.local_use_contributions:
            layer.retract_bare(local_contribution.bindings, local_contribution.constructors)
            # A local use's tail/hiding names are checked against the members
            # visible when it is first declared -- this entry's own new
            # declarations, identified the same way
            # ``_local_contribution_superseded`` tells them apart from a
            # retained one. A retained contribution from an earlier entry is
            # never re-validated here: a later entry may retire or reshape
            # its target so that a hidden name no longer matches, which must
            # not turn an already-valid ``use`` into a static error on
            # re-derivation, and it must agree with the live overlay
            # (``_local_use_sources``), which never validates either.
            validate = local_contribution.declaration.node_id in self._current_use_declaration_ids
            exposures = self._local_use_exposures(local_contribution, validate=validate)
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
            bindings, constructors = self._contribute_exposures(layer, exposures)
            rebuilt.append(
                replace(local_contribution, bindings=bindings, constructors=constructors)
            )
        layer.local_use_contributions = rebuilt

    def _contribute_exposures(
        self, layer: ScopeNode, exposures: list[tuple[NameAtom, BindingRef | _LocalScopeRoute]]
    ) -> tuple[
        Mapping[NameAtom, frozenset[BindingRef]], Mapping[NameAtom, frozenset[ConstructorRef]]
    ]:
        """Add one local use's live exposures to *layer* and return its fresh snapshot."""
        bindings: dict[NameAtom, set[BindingRef]] = {}
        constructors: dict[NameAtom, set[ConstructorRef]] = {}
        for exposed, source in exposures:
            if not isinstance(source, BindingRef):
                continue
            layer.contribute_bare(exposed, source, BareContributionSource.USE_DECLARATION)
            bindings.setdefault(exposed, set()).add(source)
            for constructor in self._declaring_constructor_candidates(source.name, source):
                layer.contribute_bare_constructor(exposed, constructor)
                constructors.setdefault(exposed, set()).add(constructor)
        return (
            {atom: frozenset(refs) for atom, refs in bindings.items()},
            {atom: frozenset(refs) for atom, refs in constructors.items()},
        )

    def _contribute_use_facade_members(
        self,
        decl: UseDecl,
        member_maps: tuple[Mapping[NameAtom, QName], ...],
        scope_route_maps: tuple[Mapping[NameAtom, frozenset[BareRoute]], ...],
    ) -> None:
        """Contribute one shared alias facade while retaining cross-module clashes."""
        combined = {atom: qname for members in member_maps for atom, qname in members.items()}
        combined_scope_routes: dict[NameAtom, frozenset[BareRoute]] = {}
        for routes in scope_route_maps:
            for atom, candidates in routes.items():
                combined_scope_routes[atom] = combined_scope_routes.get(atom, frozenset()).union(
                    candidates
                )
        self._select_use_members(decl, {**combined_scope_routes, **combined})
        for members, scope_routes in zip(member_maps, scope_route_maps, strict=True):
            self._contribute_use_members(decl, members, scope_routes, validate=False)

    def _contribute_use_members(
        self,
        decl: UseDecl,
        members: Mapping[NameAtom, QName],
        scope_routes: Mapping[NameAtom, frozenset[BareRoute]],
        *,
        validate: bool = True,
    ) -> None:
        """Select, rename, and add one use declaration's bare contribution."""
        if validate:
            self._select_use_members(decl, {**scope_routes, **members})
        selected = self._select_use_members(decl, members, validate=False)
        selected_scope_routes = self._select_use_members(
            decl,
            scope_routes,
            validate=False,
            merge=lambda left, right: left | right,
        )
        scope = self._scope
        exposed_scope_routes = {
            atom: route for atom, route in selected_scope_routes.items() if _bare_path(atom)
        }
        contributed_bindings: dict[NameAtom, set[BindingRef]] = {}
        contributed_sources: list[tuple[NameAtom, QName]] = []

        def contribute(exposed: NameAtom, source: QName) -> None:
            ref = self._cross_module_binding_ref(source)
            scope.contribute_bare(exposed, ref, BareContributionSource.USE_DECLARATION)
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
                constructor = self._contribute_bare_constructor(scope, exposed, source)
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
                raise UnknownMemberError(
                    f"name {'::'.join(prefix)!r} is not declared by this use target.",
                    span=decl.span,
                )
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
        use their collected member layer; ordinary blocks reject ``def``.
        """
        if node.scope_path:
            with self._named_scope(tuple(segment.name for segment in node.scope_path)):
                self._classify_method_declaration(node)
                self._validate_qualifier_chains(node, node.type_params)
                self._resolve_program_config(node)
                self._resolve_params_and_body(node)
            return
        if not self._at_root:
            raise AglScopeError(
                f"'def' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'def {node.name}' here).",
                span=node.span,
            )
        # Defaults are resolved in the enclosing (root) scope — they are
        # evaluated in the function's definition scope.
        self._classify_method_declaration(node)
        self._validate_qualifier_chains(node, node.type_params)
        self._resolve_program_config(node)
        previous_synthetic_entry = self._in_synthetic_entry
        previous_entry_items = self._synthetic_entry_items
        self._in_synthetic_entry = node.is_synthetic
        if node.is_synthetic and isinstance(node.body, Block):
            self._synthetic_entry_items = node.body.items
        try:
            self._resolve_params_and_body(node)
        finally:
            self._in_synthetic_entry = previous_synthetic_entry
            self._synthetic_entry_items = previous_entry_items

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
        """Reject type declarations outside the program root."""
        if not self._at_root:
            kind_word = (
                "record"
                if isinstance(node, RecordDef)
                else "enum"
                if isinstance(node, EnumDef)
                else "exception"
                if isinstance(node, ExceptionDef)
                else "type"
            )
            raise AglScopeError(
                f"Type declarations are only allowed at the top level of the "
                f"program, not inside a nested block (found '{kind_word}' here).",
                span=node.span,
            )
        # Type declarations do not otherwise participate in value resolution,
        # but their annotations still need qualifier validation in the actual
        # lexical owner layer.
        if node.scope_path:
            with self._named_scope(tuple(segment.name for segment in node.scope_path)):
                self._validate_qualifier_chains(node, node.type_params)
        else:
            self._validate_qualifier_chains(node, node.type_params)

    # ------------------------------------------------------------------
    # Binder handlers
    # ------------------------------------------------------------------

    @contextmanager
    def _binder_scope(self, node: LetDecl | VarDecl, keyword: str) -> Iterator[None]:
        """Enter the named layer a scoped ``let``/``var`` binder path selects.

        A path prefix is legal at the program root and inside a scope
        region's own body — both keep ``_at_root`` set, since a region does
        not open a fresh lexical block — and rejected inside a nested block,
        root-only like ``def``. An unprefixed binder resolves ambiently, so
        a region's bare contents and a root shorthand for the same path
        compose identically.
        """
        if not node.scope_path:
            yield
            return
        if not self._at_root:
            raise AglScopeError(
                f"A scoped '{keyword}' binder path is only allowed at the program root, "
                f"not inside a nested block.",
                span=node.span,
            )
        with self._named_scope(tuple(segment.name for segment in node.scope_path)):
            self._validate_qualifier_chains(node)
            yield

    def _resolve_let(self, node: LetDecl) -> None:
        with self._binder_scope(node, "let"):
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
        with self._binder_scope(node, "var"):
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
        ref = self._lookup_value(name)
        if ref is None:
            ref = self._lookup_bare_contribution(name, node.span, self._is_value_contribution)
            # Structured bare import contributions come next, so a bare target
            # reaches an imported mutable binding just like a read.
            if ref is None:
                ref = self._lookup_import_env_unqualified(
                    name, node.span, self._is_value_contribution
                )
            if ref is None:
                raise AglScopeError(
                    f"'{name}' is not declared; assignment requires an existing mutable binding.",
                    span=node.span,
                )
        # Mutability is not decided here: a field-directed pattern slot's final
        # binding is only known once checking selects it, so type checking owns
        # the ``:=``-on-immutable rejection for every unqualified target.
        self._require_textually_visible(ref, node.span)
        self._resolution[node.node_id] = ref
        self._resolve_expr(node.value)

    def _resolve_qualified_assign(
        self, node: AssignStmt, name: str, qualifier: QualifierChain
    ) -> None:
        """Resolve a qualified assignment target (``PATH::name := expr``).

        A local scope path is consulted first, through the same per-path
        member namespace ``_resolve_local_scope_member`` consults for a
        qualified read -- including the same ambiguity guard
        (``_check_local_scope_route_ambiguity``) when the leading qualifier
        segment is also a module route -- so a scoped ``var`` is assignable
        through its path while a scoped ``let`` -- or a ``def``, a type, or
        an agent sharing its path -- reuses the immutable-binder diagnostic
        below. Only when the qualifier does not name a local scope path is a
        cross-module target attempted; only an exported ``var`` -- ordinary or
        ``builtin var`` -- is assignable across a module boundary (a
        cross-module ``let`` reuses the immutable-binder diagnostic too).
        """
        local_path = self._validate_local_scope_chain(qualifier)
        if local_path is not None:
            self._check_local_scope_route_ambiguity(qualifier, name, local_path)
            ref = self._scope_nodes[local_path].members.get(name)
            if ref is None:
                raise _unknown_scope_member(name, local_path, node.span)
        else:
            ref = self._lookup_qualified_use_contribution(qualifier, name, node.span)
            if ref is None:
                if not qualifier.route_segments:
                    raise AglScopeError(
                        f"'{name}' is not declared; assignment requires an existing "
                        f"mutable binding.",
                        span=node.span,
                    )
                ref = self._lookup_qualified_binding(qualifier, name, node.span)
        self._require_textually_visible(ref, node.span)
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
                        expr.node_id, expr.qualifier, expr.variant, expr.span
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
            if qualifier.anchor is QualifierAnchor.CURRENT_MODULE and not qualifier.segments:
                ref = self._lookup_own_root(node.name)
                candidates: tuple[ConstructorRef, ...] | None = None
                if ref is None or ref.kind is BinderKind.constructor_binding:
                    # A root constructor binding also carries imported and
                    # referenced candidates; the module surface selects among
                    # its own, and reports a member only a root enum references.
                    constructor = self._module_surface_constructor(qualifier, node.name)
                    candidates = None if constructor is None else (constructor,)
                    ref = None if constructor is None else ref
                if ref is None:
                    raise self._own_root_miss(qualifier, node.name, node.span, node.node_id)
                self._reject_builtin_value_ref(node, ref, is_call_target=is_call_target)
                self._record_varref_binding(node, ref, candidates=candidates)
                return
            if not self._resolve_local_scope_member(node, qualifier):
                self._resolve_qualified_chain(node, qualifier)
            self._raise_unrecognized_builtin_static(node, qualifier)
            self._reject_builtin_value_ref(
                node, self._resolution.get(node.node_id), is_call_target=is_call_target
            )
            return
        # A builtin spelling is reserved in bare namespace lookups. Qualified
        # module and scope members may share it, but cannot intercept the
        # builtin after a ``use`` or other bare contribution.
        ref = self._lookup_value(node.name)
        if node.name in _BUILTIN_CALL_NAMES and (
            ref is None
            or (ref.kind is BinderKind.function_binding and not self._is_builtin_function_ref(ref))
        ):
            builtin_ref = self._bare_builtin_ref(node.name)
            if builtin_ref is not None:
                self._reject_builtin_value_ref(node, builtin_ref, is_call_target=is_call_target)
                self._record_varref_binding(node, builtin_ref)
                return
            raise AglScopeError(undefined_name_message(node.name), span=node.span)
        # Only a missing or constructor binding can consume regional candidates:
        # any other kind leaves *ref* untouched below and returns from
        # ``_record_varref_binding`` before the candidates are read.
        regional_candidates = (
            self._regional_constructor_candidates(node.name)
            if ref is None or ref.kind is BinderKind.constructor_binding
            else None
        )
        if ref is None or (
            ref.kind is BinderKind.constructor_binding
            and ref.module_id != self._module_id
            and regional_candidates is not None
        ):
            contributed = self._lookup_bare_contribution(
                node.name, node.span, self._is_value_contribution
            )
            if contributed is not None:
                ref = contributed
        if ref is None:
            # Structured bare import contributions come next.
            ref = self._lookup_import_env_unqualified(
                node.name, node.span, self._is_value_contribution
            )
            if ref is None:
                self._reject_type_name_value(node.qualifier, node.name, node.span, node.node_id)
                raise self._spaced_qualifier_repair(
                    self._spaced_qualifier_around(node.span), node.span
                ) or AglScopeError(
                    undefined_name_message(node.name),
                    span=node.span,
                )
        self._reject_builtin_value_ref(node, ref, is_call_target=is_call_target)
        self._record_varref_binding(
            node,
            ref,
            candidates=self._regional_constructor_candidates(node.name),
        )

    def _qualifier_denotes_builtin_static_owner(self, chain: QualifierChain) -> bool:
        """Return whether the resolved qualifier *chain* is a host static's nominal owner."""
        relative_path = tuple(segment.name for segment in chain.segments)
        if self._validate_local_scope_chain(chain) is not None:
            return False
        if qualifier_candidates(self._import_env, relative_path, anchored=chain.anchored):
            return False
        return self._denotes_builtin_static_owner(relative_path)

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
        owner = qualifier.render()
        return UnknownMemberError(
            f"Unknown static '{owner}::{node.name}' on prelude type '{owner}'.",
            span=node.span,
        )

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
        candidates: Collection[ConstructorRef] | None = None,
    ) -> None:
        """Record an ordinary value binding and its constructor metadata."""
        self._require_textually_visible(ref, node.span)
        self._resolution[node.node_id] = ref
        if ref.kind != BinderKind.constructor_binding:
            return
        resolved_candidates = tuple(
            self._declaring_constructor_candidates(node.name, ref)
            if candidates is None
            else candidates
        )
        if len(resolved_candidates) >= 2:
            ordered = sorted(resolved_candidates, key=_constructor_candidate_sort_key)
            raise self._ambiguous_constructor(
                node.name,
                ordered,
                self._bare_constructor_repair(ordered[0], node),
                node.span,
            )
        if len(resolved_candidates) == 1:
            self._constructor_refs[node.node_id] = resolved_candidates[0]

    def _bare_constructor_repair(self, candidate: ConstructorRef, node: VarRef) -> str:
        """Spell *candidate*, which bare *node* selects among others, as it resolves at *node*.

        An imported member is qualified by the owner name a root import tail
        makes bare, renamed as that import exposes it, while that name selects
        the imported owner where *node* is written; else by its shortest unique
        import route. Anything else is spelled by its declaration path.
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
        if owner is not None and self._selects_imported_type(
            NameT(owner, node.span, node.node_id), unqualified[owner]
        ):
            return f"{owner}::{node.name}"
        return self._routed_spelling(origin, candidate.owner_module_id, path)

    def _selects_imported_type(self, spelling: NameT, imported: frozenset[QName]) -> bool:
        """Whether bare qualifier *spelling* names no local scope here and selects *imported*."""
        return (
            not any(
                (*base, spelling.name) in self._scope_paths for base in self._lexical_scope_bases()
            )
            and self._visible_type_selection(spelling) == imported
        )

    def _routed_spelling(self, origin: QName, module: ModuleId, path: ScopePath) -> str:
        """Spell imported *origin* by its shortest unique route, else as *path* in *module*."""
        return route_spelling(self._import_env, origin) or spell_declaration(
            module, path, local_to=self._module_id
        )

    def _declaration_origin(self, qname: QName) -> QualificationOrigin:
        """Classify how *qname* is visible here: declared, imported, or use-contributed.

        *qname* is already resolved, so this asks only how it reached
        visibility: this module's own declaration, a spellable import route,
        or (the remaining case) a ``use`` declaration's contribution.
        """
        if qname[0] == self._module_id:
            return DeclaredOrigin(qname)
        if route_spelling(self._import_env, qname) is not None:
            return ImportedModuleOrigin(qname)
        return UseDeclarationOrigin(qname)

    def _ambiguous_constructor(
        self,
        spelling: str,
        candidates: Sequence[ConstructorRef],
        repair: str,
        span: SourceSpan,
    ) -> AmbiguousConstructorError:
        """Report *spelling* as ambiguous among *candidates*; *repair* selects the first."""
        return AmbiguousConstructorError.for_constructor_origins(
            spelling,
            (self._declaration_origin(candidate.qname) for candidate in candidates),
            repair=repair,
            span=span,
            local_to=self._module_id,
        )

    def _require_textually_visible(self, ref: BindingRef, span: SourceSpan) -> None:
        """Reject a binding that inline wrapping moved before an earlier use."""
        if (
            self._in_synthetic_entry
            and ref.module_id == self._module_id
            and ref.kind in _TEXTUALLY_ORDERED_BINDER_KINDS
            and ref.decl_span.start_offset > span.start_offset
        ):
            raise AglScopeError(f"'{ref.name}' is not defined.", span=span)

    def _declaring_constructor_candidates(
        self, name: str, ref: BindingRef
    ) -> tuple[ConstructorRef, ...]:
        """Return the candidates declared where *ref* was found.

        A member of a named scope is selected by that scope's structured
        identity; only a module-root binding uses the root-only candidate map.
        """
        if ref.module_id == self._module_id:
            if ref.scope_path:
                scoped = tuple(self._scoped_constructor_candidates.get((ref.scope_path, name), ()))
                if scoped:
                    return scoped
            candidates = self._constructor_candidates.get(name, ())
            return self._prefer_local_constructor_candidates(candidates)
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
                raise UnknownQualifierError(
                    f"'{chain.segments[0].name}' is a type parameter here and cannot be used "
                    "as a qualifier route.",
                    span=chain.segments[0].span,
                )
            local_path = self._validate_local_scope_chain(chain)
            if (
                isinstance(node, (NameT, AppliedT, ConstructorPattern, IsTest, VariantRef))
                and chain.segments
            ):
                if local_path is not None and local_path in self._scope_region_paths:
                    # An exact local scope-region match takes precedence
                    # over any import route or use contribution sharing its
                    # spelling, exactly as it does for a value read
                    # (``_resolve_local_scope_member``): a scoped
                    # constructor member is decided here, by the scope's own
                    # member set, whether or not it declares *chain.member*.
                    # A plain type declared at the scope's own path -- no
                    # constructor of its own -- is left to the ordinary
                    # owner/full-path resolution below, which finds it
                    # directly through its exact nested identity. A
                    # ``local_path`` matching neither a scope region nor a
                    # type is a method-owner namespace instead (a
                    # ``def Owner::Name::method`` receiver path extending an
                    # imported type): it has no member set of its own to
                    # decide against, so it always falls through too.
                    self._check_local_scope_route_ambiguity(chain, chain.member, local_path)
                    candidates = self._scoped_constructor_candidates.get(
                        (local_path, chain.member), ()
                    )
                    if candidates:
                        scoped_owner_qname = (self._module_id, _bare_atom(local_path))
                        self._owner_declarations[chain.node_id] = self._owner_member_key(
                            scoped_owner_qname, chain.member
                        )
                        return
                    if (*local_path, chain.member) not in self._type_paths:
                        raise _unknown_scope_member(chain.member, local_path, chain.span)
                site = self._type_owners.site(self._module_id, self._scope.scope_path, type_params)
                owner_expr = owner_type_expr(chain)
                # The full ``owner::member`` path is tried first: it can only
                # match a real declaration at that exact nested identity, so
                # it never mistakes a same-named, unrelated declaration (a
                # plain alias coincidentally sharing its module's own route
                # name, say) for the member's true owner. The owner's own
                # spelling is a fallback for what the full path's import
                # machinery cannot walk on its own -- a local alias or enum
                # never reachable through a routed selection.
                owner_selection = type_name_selection(site, owner_expr)
                # The full path also settles whether *chain.member* exists at
                # all: an inline member the owner alone would not find (a
                # scope-region member, or a referenced member declared at the
                # owner's own nested path) still names a real declaration
                # there, exactly as a module route's full path does.
                full_selection = type_name_selection(
                    site, NameT(chain.member, chain.span, chain.node_id, qualifier=chain)
                )
                if (
                    isinstance(node, (NameT, AppliedT))
                    and chain.anchor is None
                    and chain.segments
                    and len(full_selection) == 1
                ):
                    # A contribution's own layer stops short of trying a
                    # module route sharing the same spelling (a nested
                    # region's own contribution is final, but a root one
                    # is not: a route only ever reaches the root). Trying
                    # the route here, always, catches a root contribution
                    # -- a ``use`` alias, most often -- clashing with an
                    # identically spelled import, exactly as a value read
                    # does; a route reaching the same declaration a second
                    # time changes nothing.
                    imported = self._qualified_import_resolution(chain, chain.member)
                    if isinstance(imported, QualResolutionFound):
                        full_selection = full_selection | {imported.qname}
                    elif isinstance(imported, QualResolutionAmbiguous):
                        full_selection = full_selection | {
                            (module, imported.member) for module in imported.candidates
                        }
                if len(full_selection) > 1:
                    route, member = routed_qualifier_and_member(chain, (chain.member,))
                    raise AmbiguousQualificationError.for_origins(
                        route,
                        _bare_path(member),
                        (
                            DeclaredOrigin(qname)
                            if qname[0] == self._module_id
                            else ImportedModuleOrigin(qname)
                            for qname in full_selection
                        ),
                        anchored=chain.anchored,
                        span=chain.span,
                        local_to=self._module_id,
                    )
                if len(full_selection) == 1:
                    # The full path independently names a real declaration --
                    # more reliable than an owner-selected spelling, which can
                    # coincidentally match an unrelated same-named plain type
                    # (e.g. a scalar alias sharing its module's own route
                    # name) incapable of owning any member at all.
                    (only,) = full_selection
                    owner_qname = self._owner_of_member_qname(only)
                    if owner_qname is None or self._type_owners.owner(owner_qname) is None:
                        # No type owner: either the full path names a plain
                        # declaration directly (a module route to a top-level
                        # or nested type), or its prefix is a scope region --
                        # local or imported, not itself a type -- either way
                        # the full path already names a real declaration,
                        # which typecheck resolves by this identity rather
                        # than re-resolving the qualifier's spelling.
                        self._owner_declarations[chain.node_id] = self._qname_decl_key(only)
                    else:
                        self._select_chain_owner_member(
                            owner_qname, chain, owner_expr, site, exists=True
                        )
                    return
                if len(owner_selection) == 1:
                    self._select_chain_owner_member(
                        next(iter(owner_selection)), chain, owner_expr, site, exists=False
                    )
                    return
                if len(owner_selection) > 1:
                    # An owner ambiguous in isolation, and a full path that
                    # names no declaration at all through any candidate: no
                    # single candidate could ever supply *chain.member*, so
                    # this is the one unknown-member verdict, not a further
                    # ambiguity -- the same builder every other owner-member
                    # miss above uses.
                    raise self._unknown_owner_member(
                        render_qualifier_path(chain), chain.member, chain.span
                    )
                if isinstance(node, (NameT, AppliedT)) or len(chain.segments) > 1:
                    # Neither the full path nor the owner alone selects
                    # anything at all: this qualifier's route is unknown, or
                    # a known route is simply missing the member -- decided
                    # here through the very same route classification and
                    # error builders the value path uses for the identical
                    # qualifier shape. A one-segment constructor position
                    # (pattern, is, variant) instead defers to its own
                    # module-surface injection lookup (``_module_surface_
                    # constructor``), which a plain type-name search cannot
                    # see -- a route-injected root enum member has no
                    # separate type identity at this qualifier's spelling --
                    # and which itself raises for a route that also misses
                    # there; a multi-segment route names no such injected
                    # surface, so it is decided here alone. A
                    # ``CURRENT_MODULE``-anchored chain (``::A::Name``) is
                    # never a route -- the same split the value path draws
                    # before ever reaching the shared route resolver --
                    # so it raises the current-module-scope-miss verdict
                    # directly instead.
                    if chain.anchor is QualifierAnchor.CURRENT_MODULE:
                        raise self._own_scope_miss(chain, chain.span)
                    self._resolve_qualified_qname(chain, chain.member, chain.span)

        walk(root, validate)

    def _select_chain_owner_member(
        self,
        owner_qname: QName,
        chain: QualifierChain,
        owner_expr: NameT | AppliedT,
        site: TypeNameSite,
        *,
        exists: bool,
    ) -> None:
        """Validate ``chain``'s owner, resolved to *owner_qname*, and record its identity.

        *exists* reports whether the full path independently names a
        declaration (a scope-region member, or a referenced member declared
        at the owner's own nested path): true there needs no further check.
        Otherwise *owner*'s own alias-aware projection (:meth:`TypeOwner.select`)
        decides -- an alias's reachable members never show up as a separately
        declared path. Raises the one unknown-member verdict when neither
        finds *chain.member*. Records the full ``owner::member`` path's
        declaration identity in ``owner_declarations`` (keyed by the chain's
        node id) so typecheck reads the resolved owner directly instead of
        re-resolving the qualifier.
        """
        owner = self._type_owners.owner(owner_qname)
        if owner is None:
            return
        rendered = render_qualifier_path(chain)
        self._select_owner_member(
            owner, rendered, chain.member, chain.span, OwnerRoute(site, owner_expr)
        )
        if not exists and owner.select(chain.member, chain.segments[-1].name) is None:
            raise self._unknown_owner_member(rendered, chain.member, chain.span)
        self._owner_declarations[chain.node_id] = self._owner_member_key(owner_qname, chain.member)

    @staticmethod
    def _owner_of_member_qname(qname: QName) -> QName | None:
        """Return the owner *qname*'s trailing member atom selects beneath, if it has one."""
        module_id, atom = qname
        path = _bare_path(atom)
        return None if len(path) < 2 else (module_id, _bare_atom(path[:-1]))

    def _scope_bases(self, chain: QualifierChain) -> list[ScopePath]:
        """Return lexical bases from which an unanchored scope path may start."""
        if chain.anchor is QualifierAnchor.CURRENT_MODULE:
            return [()]
        return self._lexical_scope_bases()

    def _validate_local_scope_chain(self, chain: QualifierChain) -> ScopePath | None:
        """Find a local scope path and reject applications to plain scopes.

        The chain starts from the active lexical layers. A non-local chain is
        deliberately left for module-route and constructor compatibility
        resolution.
        """
        if chain.anchor is QualifierAnchor.MODULE or not chain.segments:
            return None
        relative_path = tuple(segment.name for segment in chain.segments)
        paths = [base + relative_path for base in self._scope_bases(chain)]
        path = next((candidate for candidate in paths if candidate in self._scope_paths), None)
        if path is None:
            return None
        prefix_length = len(path) - len(relative_path)
        for index, segment in enumerate(chain.segments, start=1):
            if (
                segment.type_args is not None
                and path[: prefix_length + index] not in self._type_paths
            ):
                raise AglScopeError(
                    f"Type arguments cannot be applied to scope segment '{segment.name}'.",
                    span=segment.span,
                )
        return path

    def _check_local_scope_route_ambiguity(
        self, chain: QualifierChain, name: str, path: ScopePath
    ) -> None:
        """Reject *chain* naming both local scope or type *path* and a module route for *name*.

        Shared by values, assignments, patterns, and ``is`` tests, so every
        position reports one spelling alike: an unanchored leading segment is
        ambiguous when it is also a module route contributing *name* --
        declared on the route's surface or injected as a root enum's inline
        member -- or, for a plain scope not declaring *name*, any module route.
        """
        if chain.anchor is not None:
            return
        relative_path = tuple(segment.name for segment in chain.segments)
        if (
            qualifier_contributes(self._import_env, relative_path, name)
            or (len(relative_path) == 1 and self._route_injected_members(chain, name))
            or (
                path not in self._type_paths
                and name not in self._scope_nodes[path].members
                and qualifier_candidates(self._import_env, relative_path[:1], anchored=False)
            )
        ):
            local_kind = "a type name" if path in self._type_paths else "a local scope"
            raise RouteClashError(
                f"Qualifier '{relative_path[0]}' is both {local_kind} and a module route for "
                f"'{name}'. {qualification_repair_guidance()}",
                span=chain.span,
            )

    def _resolve_local_scope_member(self, node: VarRef, chain: QualifierChain) -> bool:
        """Resolve a scoped member through ordered lexical scope layers.

        Local paths are considered before import routes. A member absent from
        a local type path, or qualified by a type owner a ``use`` contributes,
        is left to the qualified chain resolver, which owns constructor
        diagnostics. A type member with no constructor of its own is a type
        name, not a value.
        """
        if chain.anchor is QualifierAnchor.MODULE or not chain.segments:
            return False

        relative_path = tuple(segment.name for segment in chain.segments)
        path = self._validate_local_scope_chain(chain)
        if path is None:
            # Scopes only match exact lexical paths. A scope nested under a
            # different owner must not mask a module route sharing its suffix.
            # Keep a genuine route intact so its ordinary member diagnostics
            # remain available.
            has_module_route = bool(
                qualifier_candidates(self._import_env, relative_path, anchored=chain.anchored)
            )
            # A route needs only its leading segment to be real: `lib::A::f()`
            # is a genuine import route through `lib` even though `A::f` (or a
            # longer suffix) does not itself name a complete module route.
            has_leading_route = bool(
                qualifier_candidates(self._import_env, relative_path[:1], anchored=chain.anchored)
            )
            has_opened_member = (
                self._bare_contribution_candidates(_bare_atom((*relative_path, node.name)))
                is not None
            )
            has_opened_type_owner = (
                self._nearest_bare_contribution_layer(
                    _bare_atom(relative_path), binding_predicate=self._is_type_contribution
                )
                is not None
            )
            known_segment = any(relative_path[0] in scope_path for scope_path in self._scope_nodes)
            is_prelude_static_owner = self._denotes_builtin_static_owner(relative_path)
            if (chain.anchor is QualifierAnchor.CURRENT_MODULE and len(chain.segments) > 1) or (
                known_segment
                and not has_module_route
                and not has_leading_route
                and not has_opened_member
                and not has_opened_type_owner
                and not is_prelude_static_owner
            ):
                raise self._unknown_scope_path(chain)
            return False

        ref = self._scope_nodes[path].members.get(node.name)
        if ref is None and path in self._type_paths:
            return False
        self._check_local_scope_route_ambiguity(chain, node.name, path)
        if ref is None:
            raise _unknown_scope_member(node.name, path, node.span)
        self._require_textually_visible(ref, node.span)
        if ref.kind is BinderKind.constructor_binding:
            # One declaration owns each scoped spelling, so it has at most one constructor.
            candidates = self._scoped_constructor_candidates.get((path, node.name), ())
            if not candidates:
                raise type_name_not_a_value(render_qualified_name(chain, node.name), node.span)
            self._constructor_refs[node.node_id] = candidates[0]
            owner_qname = (self._module_id, _bare_atom(path))
            self._owner_declarations[chain.node_id] = self._owner_member_key(owner_qname, node.name)
            return True
        self._resolution[node.node_id] = ref
        return True

    def _resolve_qualified_chain(self, node: VarRef, chain: QualifierChain) -> None:
        """Resolve a qualified value through one ordered scope-or-route chain.

        An imported route owns the complete requested path atom. Resolving it
        first keeps ordinary members of a type-owned scope distinct from enum
        variants and record construction, while also ensuring selection filters
        apply to the final atom rather than only its type owner.
        """
        atom = _bare_atom((*tuple(segment.name for segment in chain.segments), node.name))
        ref = self._lookup_qualified_use_contribution(chain, node.name, node.span)
        if ref is not None:
            self._record_varref_binding(
                node,
                ref,
                candidates=self._regional_constructor_candidates(atom),
            )
            return
        direct_error: AglScopeError | None = None
        local_type_path = self._validate_local_scope_chain(chain)
        if (
            chain.anchor is not QualifierAnchor.CURRENT_MODULE
            and chain.segments
            and local_type_path not in self._type_paths
        ):
            try:
                self._resolve_varref_qualified(node, chain)
                return
            except AglScopeError as error:
                direct_error = error
        if self._resolve_constructor_chain(node.node_id, chain, node.name):
            return
        if direct_error is not None:
            injected = self._module_surface_constructor(chain, node.name)
            if injected is None:
                raise direct_error
            self._constructor_refs[node.node_id] = injected
            return
        # Only a ``CURRENT_MODULE``-anchored chain reaches here: every other
        # anchor either resolves above or leaves ``direct_error`` set. Its
        # empty-segment self-reference form is handled in ``_resolve_varref``,
        # so the chain names one segment this module does not declare.
        raise self._own_scope_miss(chain, node.span)

    def _own_root_miss(
        self, chain: QualifierChain, name: str, span: SourceSpan, node_id: int
    ) -> AglScopeError:
        """Return why ``::name``, spelled by node *node_id*, names nothing at this module's root."""
        self._reject_type_name_value(chain, name, span, node_id)
        return self._spaced_qualifier_repair(
            self._spaced_qualifier_at(chain.span), chain.span
        ) or UnknownMemberError(undefined_name_message(name, in_module=True), span=span)

    def _own_scope_miss(self, chain: QualifierChain, span: SourceSpan) -> AglScopeError:
        """Return why current-module *chain*, used at *span*, names no scope of this module."""
        if len(chain.segments) > 1:
            return self._unknown_scope_path(chain)
        segment = chain.segments[0]
        return self._spaced_qualifier_repair(
            self._spaced_qualifier_at(chain.span) or self._spaced_qualifier_around(span), span
        ) or UnknownQualifierError(
            undefined_name_message(segment.name, in_module=True), span=segment.span
        )

    @staticmethod
    def _unknown_scope_path(chain: QualifierChain) -> UnknownQualifierError:
        """Return the error for *chain* spelling no scope path."""
        return UnknownQualifierError(
            f"Unknown scope path '{'::'.join(segment.name for segment in chain.segments)}'.",
            span=chain.span,
        )

    def _qualified_import_resolution(self, chain: QualifierChain, name: str) -> QualResolution:
        """Resolve a chain as one module route followed by an exact path atom."""
        route = tuple(part for part in chain.segments[0].name.split("/"))
        atom = _bare_atom((*(segment.name for segment in chain.segments[1:]), name))
        return resolve_qualified(self._import_env, route, atom, anchored=chain.anchored)

    def _lookup_qualified_use_contribution(
        self, chain: QualifierChain, name: str, span: SourceSpan
    ) -> BindingRef | None:
        """Resolve a complete unanchored qualified atom contributed by ``use``."""
        if chain.anchor is not None:
            return None
        atom = _bare_atom((*tuple(segment.name for segment in chain.segments), name))
        opened = self._lookup_bare_contribution(atom, span, self._is_value_contribution)
        if opened is None:
            return None
        imported = self._qualified_import_resolution(chain, name)
        if isinstance(imported, QualResolutionAmbiguous):
            raise AmbiguousQualificationError.for_origins(
                chain.route_segments,
                (name,),
                (
                    UseDeclarationOrigin(_ref_qname(opened)),
                    *(
                        ImportedModuleOrigin((module, imported.member))
                        for module in imported.candidates
                    ),
                ),
                anchored=chain.anchored,
                span=span,
                local_to=self._module_id,
            )
        if isinstance(imported, QualResolutionFound):
            imported_ref = self._cross_module_binding_ref(imported.qname)
            opened_identity = (
                opened.module_id,
                opened.scope_path,
                opened.decl_node_id,
                opened.kind,
            )
            imported_identity = (
                imported_ref.module_id,
                imported_ref.scope_path,
                imported_ref.decl_node_id,
                imported_ref.kind,
            )
            if opened_identity != imported_identity:
                raise AmbiguousQualificationError.for_origins(
                    chain.route_segments,
                    (name,),
                    (
                        UseDeclarationOrigin(_ref_qname(opened)),
                        ImportedModuleOrigin(imported.qname),
                    ),
                    anchored=chain.anchored,
                    span=span,
                    local_to=self._module_id,
                )
        return opened

    def _resolve_constructor_chain(self, node_id: int, chain: QualifierChain, variant: str) -> bool:
        """Record a constructor value result when *chain* ends at a type owner."""
        opened = self._use_constructor_candidates(chain, variant)
        if opened is not None:
            if len(opened) == 1:
                self._constructor_refs[node_id] = next(iter(opened))
                return True
            rendered = "::".join((*tuple(segment.name for segment in chain.segments), variant))
            raise _use_route_unknown_member(rendered, chain.span)
        local_path = self._validate_local_scope_chain(chain)
        if local_path in self._type_paths:
            self._check_local_scope_route_ambiguity(chain, variant, local_path)
            self._constructor_refs[node_id] = self._local_owner_constructor(
                chain, local_path, variant
            )
            return True
        constructor = self._routed_owner_constructor(chain, variant, defer_diagnostics=False)
        if constructor is None:
            return False
        self._constructor_refs[node_id] = constructor
        return True

    def _routed_owner_constructor(
        self, chain: QualifierChain, variant: str, *, defer_diagnostics: bool
    ) -> ConstructorRef | None:
        """Return the constructor *chain*, naming a type owner no local path does, selects.

        Patterns and ``is`` tests defer a bad imported owner route until the
        checker can assess it against the type being matched.
        """
        try:
            resolved = self._imported_chain_owner(chain, variant)
        except AglScopeError:
            if defer_diagnostics:
                return None
            raise
        if resolved is not None:
            decl_key, owner = resolved
            return self._owner_constructor(decl_key, owner, chain, variant)
        if (
            chain.anchor is not QualifierAnchor.MODULE
            and len(chain.segments) == 1
            and chain.segments[0].name in self._declared_type_names
        ):
            type_name = chain.segments[0].name
            candidate = next(
                (
                    candidate
                    for candidate in self._constructor_candidates.get(variant, ())
                    if candidate.owner_path == (type_name,)
                    and (
                        chain.anchor is not QualifierAnchor.CURRENT_MODULE
                        or candidate.owner_module_id == self._module_id
                    )
                ),
                None,
            )
            if candidate is not None:
                self._owner_declarations[chain.node_id] = (
                    candidate.owner_module_id,
                    candidate.owner_path,
                    candidate.owner_name,
                )
            return candidate
        return None

    def _use_constructor_candidates(
        self, chain: QualifierChain, variant: str
    ) -> set[ConstructorRef] | None:
        """Return a use route's exact constructor selection, including hidden verdicts."""
        if chain.anchor is not None:
            return None
        relative_path = tuple(segment.name for segment in chain.segments)
        # A route contributing more than one candidate here is already an
        # import collision, and every route that can produce one is rejected
        # earlier: a use declaration's own target ambiguity
        # (``_resolve_use_target``), a general bare-value-contribution clash
        # for any candidate a declared constructor makes visible
        # (``_lookup_bare_contribution``), or a whole-route import collision
        # (``_qualified_import_resolution`` below). So *opened*, once
        # non-empty, is always the single candidate its route selects.
        opened = self._regional_constructor_candidates(_bare_atom((*relative_path, variant)))
        if opened:
            imported = self._qualified_import_resolution(chain, variant)
            extra_origins: list[QualificationOrigin] | None = None
            if isinstance(imported, QualResolutionAmbiguous):
                extra_origins = [
                    ImportedModuleOrigin((module, imported.member))
                    for module in imported.candidates
                ]
            elif isinstance(imported, QualResolutionFound):
                imported_constructor = self._cross_module_constructor_refs.get(imported.qname)
                if opened != {imported_constructor}:
                    extra_origins = [ImportedModuleOrigin(imported.qname)]
            if extra_origins is not None:
                candidate = next(iter(opened))
                origins: list[QualificationOrigin] = [
                    UseDeclarationOrigin(
                        (
                            candidate.owner_module_id,
                            _bare_atom((*candidate.owner_path, candidate.owner_name)),
                        )
                    ),
                    *extra_origins,
                ]
                raise AmbiguousQualificationError.for_origins(
                    (),
                    (*relative_path, variant),
                    origins,
                    span=chain.span,
                    local_to=self._module_id,
                )
        if opened is not None:
            return opened
        owner_ref = self._lookup_bare_contribution(
            _bare_atom(relative_path), chain.span, self._is_type_contribution
        )
        if owner_ref is None:
            self._reject_use_alias_referenced_member(chain, variant)
            return None
        type_owner = self._type_owners.owner(_ref_qname(owner_ref))
        # Only the raise matters here: the returned reference, below, always
        # comes from `select`, which folds an alias's own type parameters
        # back in rather than the target's raw member (see `TypeOwner.select`).
        self._select_owner_member(
            type_owner, render_qualifier_path(chain), variant, chain.span, self._owner_route(chain)
        )
        declared_child = _bare_atom((*owner_ref.scope_path, owner_ref.name, variant))
        if (owner_ref.module_id, declared_child) in self._decl_info:
            return set()
        if type_owner is not None and type_owner.constructor is None:
            return set()
        constructor = None if type_owner is None else type_owner.select(variant, relative_path[-1])
        if constructor is None:
            raise self._unknown_owner_member("::".join(relative_path), variant, chain.span)
        return {constructor}

    def _owner_route(self, chain: QualifierChain) -> OwnerRoute:
        """Return the route *chain* spells its owner through, at the current lexical layer."""
        site = self._type_owners.site(self._module_id, self._scope.scope_path)
        return OwnerRoute(site, owner_type_expr(chain))

    def _reject_use_alias_referenced_member(self, chain: QualifierChain, variant: str) -> None:
        """Raise when a whole-target ``use`` alias spells a referenced or hidden member.

        ``use Status as S`` contributes only what ``Status``'s scope declares
        beneath ``S``, so ``S::member`` of a member ``Status`` merely
        references, or its import hides, reaches no contribution.
        """
        alias, *rest = (segment.name for segment in chain.segments)
        route = self._owner_route(chain)
        layer: ScopeNode | None = self._scope
        while layer is not None:
            contributions: list[LocalUseContribution | ImportedUseContribution] = [
                *layer.local_use_contributions,
                *layer.imported_use_contributions,
            ]
            for contribution in contributions:
                if contribution.declaration.alias != alias:
                    continue
                target = contribution.target
                targets = (
                    ((self._module_id, target.local_path),)
                    if target.local_path is not None
                    else target.imported_routes
                )
                for module_id, path in targets:
                    self._select_owner_member(
                        self._type_owners.owner((module_id, _bare_atom((*path, *rest)))),
                        render_qualifier_path(chain),
                        variant,
                        chain.span,
                        route,
                    )
            layer = layer.parent

    def _local_owner_constructor(
        self, chain: QualifierChain, path: ScopePath, variant: str
    ) -> ConstructorRef:
        """Return the constructor *chain*, naming this module's type *path*, selects as *variant*.

        Raises when the type qualifies no constructor at all, or none named *variant*.
        """
        owner_qname = (self._module_id, _bare_atom(path))
        owner = self._constructible_owner(owner_qname)
        if owner is None:
            raise type_name_not_a_value(render_qualifier_path(chain), chain.span)
        decl_key = self._owner_member_key(owner_qname, variant)
        scoped = self._scoped_constructor_candidates.get((path, variant), ())
        if scoped:
            self._owner_declarations[chain.node_id] = decl_key
            return scoped[0]
        return self._owner_constructor(decl_key, owner, chain, variant)

    def _owner_constructor(
        self, decl_key: DeclarationKey, owner: TypeOwner, chain: QualifierChain, variant: str
    ) -> ConstructorRef:
        """Return the constructor ``chain::variant`` selects from *owner*, already resolved.

        *owner* is resolved by declaration identity, each alias on its chain
        where that alias is declared; *chain* is the owner's spelling at the
        use. Every caller already confirmed *owner* qualifies constructors
        (see :meth:`_constructible_owner`). Raises the one unknown-member
        verdict when *owner* declares no such *variant*, in every position
        that reaches an owner this way (value, pattern, ``is``). Records
        *decl_key* -- the resolved declaration identity a caller already
        computed for ``chain``'s spelling -- in ``owner_declarations`` (keyed
        by the chain's node id), exactly as the type-position walk does, so
        typecheck reads it back directly instead of re-resolving the
        qualifier.
        """
        rendered = render_qualifier_path(chain)
        # An alias's own constructor identity -- not the target's raw member --
        # must reach the caller, so the returned reference always comes from
        # `select`, which folds the alias's own type parameters back in (see
        # `TypeOwner.select`).
        constructor = self._select_owner_member_or_raise(
            owner,
            rendered,
            variant,
            chain.segments[-1].name,
            chain.span,
            self._owner_route(chain),
        )
        self._owner_declarations[chain.node_id] = decl_key
        return constructor

    def _imported_chain_owner(
        self, chain: QualifierChain, variant: str
    ) -> tuple[DeclarationKey, TypeOwner] | None:
        """Resolve a constructible type owner reached through imports, with its declaration key.

        The final chain segment is selected as a normal imported member; any
        preceding segments are its route.  A one-segment chain may instead
        name a type exposed by an import tail. This is the same chain walk used for
        ordinary qualified values, with only the resulting member kind
        determining whether it owns a constructor. The returned key is
        already the full ``owner::variant`` declaration identity to record in
        ``owner_declarations`` -- the ambiguous-bare-owner fallback below
        resolves *variant* directly, so its own qname already names that
        identity, unlike the other branches' separately named owner.
        """
        if chain.anchor is QualifierAnchor.CURRENT_MODULE or not chain.segments:
            return None
        if len(chain.segments) == 1 and not chain.anchored:
            segment = chain.segments[0]
            try:
                owner_ref = self._lookup_import_env_unqualified(
                    segment.name, chain.span, self._is_type_contribution
                )
            except AmbiguousQualificationError:
                if segment.type_args is None:
                    raise
                # An owner ambiguous by its bare spelling alone, always an
                # import collision, still selects the one module declaring
                # *variant* through it -- exactly as the unapplied owner
                # does, but type arguments make the bare lookup above raise
                # before it can try that: resolve the full ``owner::variant``
                # path instead.
                result = resolve_qualified(
                    self._import_env, (), _bare_atom((segment.name, variant)), anchored=False
                )
                if isinstance(result, QualResolutionFound):
                    owner = self._constructible_owner(result.qname)
                    return None if owner is None else (self._qname_decl_key(result.qname), owner)
                if isinstance(result, QualResolutionAmbiguous):
                    raise AmbiguousQualificationError.for_origins(
                        (),
                        (segment.name, variant),
                        (
                            ImportedModuleOrigin((module, result.member))
                            for module in result.candidates
                        ),
                        span=chain.span,
                        local_to=self._module_id,
                    ) from None
                # No candidate declares *variant* either: the same unknown-member
                # verdict every other owner-member miss raises, not the bare
                # owner's own ambiguity (already caught above) nor a bogus
                # module-route diagnostic.
                raise self._unknown_owner_member(
                    render_qualifier_path(chain), variant, chain.span
                ) from None
            if owner_ref is None:
                return None
            owner_qname = _ref_qname(owner_ref)
            owner = self._constructible_owner(owner_qname)
            return None if owner is None else (self._owner_member_key(owner_qname, variant), owner)
        if len(chain.segments) > 1:
            route = QualifierChain(
                anchor=chain.anchor,
                segments=chain.segments[:-1],
                member="",
                span=chain.span,
                node_id=chain.node_id,
            )
            try:
                owner_ref = self._lookup_qualified_binding(
                    route, chain.segments[-1].name, chain.span
                )
            except (UnknownQualifierError, UnknownMemberError):
                # The final scope segment can be part of a selected path atom
                # rather than a separately selected type owner. Let the normal
                # module-path resolver consume the complete atom. Any other
                # verdict -- notably ambiguity -- is a real error and propagates.
                return None
            owner_qname = _ref_qname(owner_ref)
            owner = self._constructible_owner(owner_qname)
            if owner is None:
                rendered = render_qualifier_path(chain)
                if self._type_owners.is_declared(owner_qname):
                    raise type_name_not_a_value(rendered, chain.span)
                raise AglScopeError(f"'{rendered}' is not a constructible type.", span=chain.span)
            return self._owner_member_key(owner_qname, variant), owner
        return None

    def _nearest_layer[T](
        self, read: Callable[[ScopeNode], set[T]], start: ScopeNode | None = None
    ) -> tuple[ScopeNode, set[T]] | None:
        """Return the nearest layer, from *start* or the current scope, where *read* finds any."""
        layer: ScopeNode | None = self._scope if start is None else start
        while layer is not None:
            found = read(layer)
            if found:
                return layer, found
            layer = layer.parent
        return None

    def _nearest_bare_contribution_layer(
        self,
        name: NameAtom,
        *,
        binding_predicate: Callable[[BindingRef], bool] | None = None,
        start: ScopeNode | None = None,
    ) -> tuple[ScopeNode, set[BindingRef]] | None:
        """Return the nearest layer's static and live-use bindings for *name*.

        The search starts at *start*, or the current scope, and skips layers
        with no binding satisfying *binding_predicate*.
        """

        def read(layer: ScopeNode) -> set[BindingRef]:
            bindings = self._layer_bare_bindings(layer, name)
            if binding_predicate is None:
                return bindings
            return {ref for ref in bindings if binding_predicate(ref)}

        return self._nearest_layer(read, start)

    def _layer_bare_bindings(self, layer: ScopeNode, name: NameAtom) -> set[BindingRef]:
        """Return *layer*'s static bindings for *name*, refreshed by its live uses."""
        bindings = set(layer.bare_contributions.get(name, ()))
        for contribution in layer.imported_use_contributions:
            refresh = self._facade_refresh(contribution, name)
            if refresh is not None:
                stale, refs, _ = refresh
                bindings.difference_update(stale)
                bindings.update(refs)
        for local_contribution in layer.local_use_contributions:
            # The static read above carries this contribution's snapshot,
            # taken before a later REPL entry may have redeclared one of its
            # source members; the live exposure below replaces it, so a
            # redeclared member is never a second candidate.
            bindings.difference_update(local_contribution.bindings.get(name, ()))
            sources = self._local_use_sources(local_contribution, name)
            if self._local_contribution_superseded(local_contribution):
                # The static read above can already carry a superseded
                # contribution's binding, so retract it rather than skip it.
                bindings.difference_update(sources)
            else:
                bindings.update(sources)
        return bindings

    def _layer_bare_constructors(self, layer: ScopeNode, name: NameAtom) -> set[ConstructorRef]:
        """Return *layer*'s static constructor candidates for *name*, refreshed by its live uses."""
        constructors = set(layer.bare_constructor_contributions.get(name, ()))
        for contribution in layer.imported_use_contributions:
            refresh = self._facade_refresh(contribution, name)
            if refresh is not None:
                constructors.update(refresh[2])
        for local_contribution in layer.local_use_contributions:
            constructors.difference_update(local_contribution.constructors.get(name, ()))
            candidates = {
                candidate
                for source in self._local_use_sources(local_contribution, name)
                for candidate in self._declaring_constructor_candidates(source.name, source)
            }
            if self._local_contribution_superseded(local_contribution):
                constructors.difference_update(candidates)
            else:
                constructors.update(candidates)
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
        does either. A wildcard-facade use additionally re-derives which
        modules are currently live (:meth:`_facade_modules`); any other
        ``use ...::*`` names a fixed module set that cannot itself grow or
        shrink, so only the variant expansion below is live for it.
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

        Shared by the live overlay (:meth:`_facade_variant_refs`) and its
        static re-snapshot (:meth:`_refresh_imported_use`): a bare-exposed
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
                for constructor in self._type_owners.referenced_member_refs(module, member):
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
        :meth:`_contribute_regional_enum_variants`/:meth:`_contribute_referenced_member`
        perform for a region-scoped import, but derived live here instead of
        as a declaration-time side effect, so a later entry re-derives it
        exactly like every other candidate rather than losing it.
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
        self, layer: ScopeNode, contribution: ImportedUseContribution
    ) -> ImportedUseContribution:
        """Return *contribution* re-snapshotted against *layer*'s live derivation.

        The static counterpart of the per-name reads in
        :meth:`_layer_bare_bindings`/:meth:`_layer_bare_constructors`: the same
        :meth:`_facade_refresh` derivation, applied once across every atom the
        contribution's own snapshot or its current facade modules could
        expose -- so a stored table and a live lookup always agree, whether a
        module was gained or dropped or a member's own variant expansion
        changed. Applied to every refreshing use, not only a wildcard facade:
        even a fixed module set, which cannot itself grow or shrink (see
        :meth:`_facade_refresh`), still needs its variant expansion re-derived
        the same way. A contribution that does not refresh every member -- a
        selective use, even of a wildcard-facade alias -- needs no refresh
        either: its snapshot, already present in *layer*'s copied bare
        tables, stands as declared.
        """
        if not contribution.refreshes_all_members:
            return contribution
        origin = contribution.target.wildcard_facade_origin_node_id
        layer.retract_bare(contribution.bindings, contribution.constructors)
        modules = self._use_modules(contribution, origin)
        atoms = set(contribution.bindings) | set(contribution.constructors)
        if origin is not None:
            # A fixed module set (below) can only lose or keep its members;
            # only a wildcard facade's module set itself grows or shrinks.
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
            for ref in refs:
                layer.contribute_bare(atom, ref, BareContributionSource.USE_DECLARATION)
            if refs:
                bindings[atom] = frozenset(refs)
            for constructor in constructor_refs:
                layer.contribute_bare_constructor(atom, constructor)
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

    def type_contributions(
        self, scope_path: ScopePath, name: NameAtom, is_type: Callable[[QName], bool]
    ) -> tuple[ScopePath, frozenset[QName]] | None:
        """Return the nearest layer above *scope_path* contributing type *name*, and its types.

        Reads the prepared headers, so a type name's selection never depends
        on type owners.
        """
        nearest = self._nearest_bare_contribution_layer(
            name,
            binding_predicate=lambda ref: ref.contributes_a_type and is_type(_ref_qname(ref)),
            start=self._scope_nodes[scope_path],
        )
        return None if nearest is None else contributed_declarations(nearest[0], nearest[1])

    def _is_value_contribution(self, ref: BindingRef) -> bool:
        """Whether a shared contribution denotes a value in addition to any type."""
        return (
            ref.kind is not BinderKind.constructor_binding
            or self._cross_module_constructor(_ref_qname(ref)) is not None
            or bool(self._declaring_constructor_candidates(ref.name, ref))
        )

    def _bare_contribution_candidates(self, name: NameAtom) -> set[BindingRef] | None:
        """Return the nearest region's bare contributions for *name*."""
        nearest = self._nearest_bare_contribution_layer(name)
        return None if nearest is None else nearest[1]

    def _prefer_local_constructor_candidates(
        self, candidates: Collection[ConstructorRef]
    ) -> tuple[ConstructorRef, ...]:
        """Let a local bare constructor shadow imported candidates.

        Inline members are records below their enum owner, but their terminal
        constructor names are injected into the module root. They therefore
        have the same lexical precedence as root record constructors over the
        automatic ``std/prelude`` prelude.
        """
        local = tuple(
            candidate for candidate in candidates if candidate.owner_module_id == self._module_id
        )
        local_root = tuple(candidate for candidate in local if not candidate.owner_path)
        return local_root or local or tuple(candidates)

    def _regional_constructor_candidates(self, name: NameAtom) -> set[ConstructorRef] | None:
        """Return the nearest region's constructor candidates, including live local uses."""
        nearest = self._nearest_layer(lambda layer: self._layer_bare_constructors(layer, name))
        return (
            None if nearest is None else set(self._prefer_local_constructor_candidates(nearest[1]))
        )

    def _lookup_bare_contribution(
        self, name: NameAtom, span: SourceSpan, binding_predicate: Callable[[BindingRef], bool]
    ) -> BindingRef | None:
        """Resolve one region's bare contributions satisfying *binding_predicate*.

        Clashes are deferred to use sites. Each candidate's own recorded
        :class:`~agm.agl.scope.symbols.BareContributionSource` decides its
        origin kind: a root-position tailed import's contribution, or a
        region-scoped import tail's (see :class:`BareContributionSource`), is
        an :class:`ImportedModuleOrigin`; a ``use`` declaration's is a
        :class:`UseDeclarationOrigin`. A candidate live-refreshed past the
        static snapshot (no recorded source at *name*) can only have reached
        visibility through a ``use`` declaration -- an import tail never
        refreshes -- so it defaults to :class:`UseDeclarationOrigin` too.
        """
        nearest = self._nearest_bare_contribution_layer(name, binding_predicate=binding_predicate)
        if nearest is None:
            return None
        selected_layer, region_refs = nearest
        root_refs: set[BindingRef] = set()
        if selected_layer is self._root_scope:
            root_refs = {
                ref
                for qname in self._import_env.unqualified.get(name, frozenset())
                if binding_predicate(ref := self._cross_module_binding_ref(qname))
            }
        resolved = region_refs | root_refs
        distinct = {
            (ref.module_id, ref.scope_path, ref.decl_node_id, ref.kind): ref for ref in resolved
        }
        if len(distinct) == 1:
            return next(iter(distinct.values()))
        sources = selected_layer.bare_contributions.get(name, {})
        origins = tuple(
            ImportedModuleOrigin(_ref_qname(ref))
            if ref in root_refs or sources.get(ref) is BareContributionSource.IMPORT_TAIL
            else UseDeclarationOrigin(_ref_qname(ref))
            for ref in distinct.values()
        )
        raise AmbiguousQualificationError.for_origins(
            (), _bare_path(name), origins, span=span, local_to=self._module_id
        )

    def _lookup_import_env_unqualified(
        self, name: str, span: SourceSpan, binding_predicate: Callable[[BindingRef], bool]
    ) -> BindingRef | None:
        """Look up a bare name in the tailed-import environment.

        Only declarations satisfying *binding_predicate* are candidates.
        Returns a ``BindingRef`` if exactly one ``QName`` matches, or raises
        :class:`AmbiguousQualificationError` on ambiguity (clash-on-use).
        Returns ``None`` if no tailed import contributes a candidate. Shared
        by bare value references, bare assignment targets, and type owners.
        """
        qnames = {
            qname
            for qname in self._import_env.unqualified.get(name, frozenset())
            if binding_predicate(self._cross_module_binding_ref(qname))
        }
        if not qnames:
            return None
        if len(qnames) > 1:
            # Clash-on-use: more than one module exposes this name.
            raise AmbiguousQualificationError.for_origins(
                (),
                (name,),
                (ImportedModuleOrigin(qname) for qname in qnames),
                span=span,
                local_to=self._module_id,
            )
        # Exactly one QName.
        qname = next(iter(qnames))
        return self._make_cross_module_ref(qname)

    def _resolve_varref_qualified(self, node: VarRef, module_qualifier: QualifierChain) -> None:
        """Resolve an imported qualified VarRef."""
        qname = self._resolve_qualified_qname(module_qualifier, node.name, node.span)
        constructor = self._cross_module_constructor(qname)
        if constructor is not None:
            self._constructor_refs[node.node_id] = constructor
            # *qname* already names the constructor's own full declaration
            # path (route segments plus the constructed member), exactly as
            # the type-position walk's no-separate-owner case does.
            self._owner_declarations[module_qualifier.node_id] = self._qname_decl_key(qname)
            return
        if self._type_owners.is_declared(qname):
            # A type bearing no constructor leaves the name to the module surface.
            injected = self._module_surface_constructor(module_qualifier, node.name)
            if injected is None:
                raise type_name_not_a_value(
                    render_qualified_name(module_qualifier, node.name), node.span
                )
            self._constructor_refs[node.node_id] = injected
            return
        self._resolution[node.node_id] = self._make_cross_module_ref(qname)

    def _lookup_qualified_binding(
        self, qualifier: QualifierChain, name: str, span: SourceSpan
    ) -> BindingRef:
        """Resolve a qualified value or assignment target through the shared resolver."""
        qname = self._resolve_qualified_qname(qualifier, name, span)
        return self._make_cross_module_ref(qname)

    def _resolve_qualified_qname(
        self, qualifier: QualifierChain, name: str, span: SourceSpan
    ) -> tuple[ModuleId, NameAtom]:
        """Resolve a module route followed by one structured declaration path."""
        route_segment = qualifier.segments[0]
        if route_segment.type_args is not None:
            raise AglScopeError(
                f"Type arguments cannot be applied to module route '{route_segment.name}'.",
                span=route_segment.span,
            )
        route = tuple(part for part in route_segment.name.split("/"))
        atom_path = (*tuple(segment.name for segment in qualifier.segments[1:]), name)
        route_members = qualifier_members(self._import_env, route, anchored=qualifier.anchored)
        cumulative: ScopePath = ()
        for segment in qualifier.segments[1:]:
            cumulative = (*cumulative, segment.name)
            if segment.type_args is not None and not any(
                members.get(_bare_atom(cumulative)) in self._cross_module_type_owners
                and members.get(_bare_atom(atom_path)) is not None
                for _, members in route_members
            ):
                raise AglScopeError(
                    f"Type arguments cannot be applied to scope segment '{segment.name}'.",
                    span=segment.span,
                )
        atom: NameAtom = atom_path[0] if len(atom_path) == 1 else atom_path
        return resolve_qualified_member(
            self._import_env,
            route,
            atom,
            anchored=qualifier.anchored,
            unknown_qualifier=lambda rendered: UnknownQualifierError(
                f"No module imported under qualifier '{rendered}'.", span=span
            ),
            missing_member=lambda rendered: UnknownMemberError(
                f"'{name}' is not a public member of imported module '{rendered}' or is hidden.",
                span=span,
            ),
            ambiguous=lambda candidates: AmbiguousQualificationError.for_origins(
                route,
                _bare_path(atom),
                (ImportedModuleOrigin((module, atom)) for module in candidates),
                anchored=qualifier.anchored,
                span=span,
                local_to=self._module_id,
            ),
        )

    def _lookup_own_root(self, name: str) -> BindingRef | None:
        """Look up *name* in the module's own root scope bindings only.

        ``::name`` must resolve to the current module's OWN top-level declaration,
        bypassing any lexical shadows introduced by nested scopes (params, let, etc.).
        We look ONLY in the root frame's direct ``bindings`` dict — we do NOT call
        ``lookup()`` (which walks the parent chain and would fall through to a session
        parent scope or find nested shadows first).

        In the REPL program context, if *name* is not in the entry's own root scope,
        we fall back to the session scope (``_repl_session_scope``) so that
        ``::name`` can resolve to a prior session binding.
        """
        ref = self._root_scope.bindings.get(name)
        if ref is None and self._repl_session_scope is not None:
            ref = self._repl_session_scope.bindings.get(name)
        return ref

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
        rendered = render_qualifier(advisory.segments, anchored=advisory.anchored)
        return AglScopeError(
            f"Whitespace before '::{advisory.member_text}' makes this a call with a "
            f"self-reference, not a module qualifier. "
            f"Write '{rendered}::{advisory.member_text}' without whitespace.",
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

    def _reject_type_name_value(
        self, qualifier: QualifierChain | None, name: str, span: SourceSpan, node_id: int
    ) -> None:
        """Raise when ``qualifier::name``, naming no value, spells a type visible at its use."""
        if self._names_visible_type(NameT(name, span, node_id, qualifier=qualifier)):
            raise type_name_not_a_value(render_qualified_name(qualifier, name), span)

    def _names_visible_type(self, spelling: NameT) -> bool:
        """Whether type name *spelling* selects a type visible at its use."""
        return bool(self._visible_type_selection(spelling))

    def _visible_type_selection(self, spelling: NameT) -> frozenset[QName]:
        """Return the declarations type name *spelling* selects at its use."""
        scope: ScopeNode | None = self._scope
        while scope is not None and not scope.scope_path:
            scope = scope.parent
        site = self._type_owners.site(self._module_id, () if scope is None else scope.scope_path)
        return type_name_selection(site, spelling)

    def _is_type_contribution(self, ref: BindingRef) -> bool:
        """Whether a shared contribution denotes a type."""
        return self._type_owners.is_declared(_ref_qname(ref))

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
            try:
                self._resolve_varref(callee, is_call_target=True)
            except (UnknownQualifierError, UnknownMemberError):
                qualifier = callee.qualifier
                if qualifier is None or not self._qualifier_denotes_builtin_static_owner(qualifier):
                    raise
                raise self._unknown_static_error(callee, qualifier) from None
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
                        raise AglScopeError(
                            f"Name '{param.name}' is already declared in this scope.",
                            span=param.span,
                        )
                    param_scope.define(param.name, ref)
                if node.body is not None:
                    with self._resolution_flags_ctx(in_function=True):
                        self._resolve_expr_or_block(node.body)

    # ------------------------------------------------------------------
    # Pattern variable binding
    # ------------------------------------------------------------------

    def _qualified_constructor_candidates(
        self, node_id: int, chain: QualifierChain, name: str, span: SourceSpan
    ) -> tuple[ConstructorRef, ...]:
        """Select the constructors pattern or ``is`` spelling ``chain::name`` can match.

        Ordering mirrors qualified value resolution: a scope-use contribution or
        a local scope member is considered before an import route owning the
        complete atom, and only then is the chain read as a type owner with
        *name* as its variant, else as a module qualifier. A local scope's
        selection is complete; a local scope lacking *name*, a local or
        imported type owner declaring no such member, or a module qualifier
        selecting nothing, all fail with the one unknown-member verdict, as
        the same value spelling does.
        """
        opened = self._use_constructor_candidates(chain, name)
        if opened is not None:
            if not opened:
                rendered = "::".join((*(segment.name for segment in chain.segments), name))
                raise _use_route_unknown_member(rendered, chain.span)
            return tuple(opened)
        local_path = self._validate_local_scope_chain(chain)
        if local_path is not None:
            self._check_local_scope_route_ambiguity(chain, name, local_path)
        if local_path in self._type_paths:
            return (self._local_owner_constructor(chain, local_path, name),)
        if local_path is not None:
            # A scope-region member miss is already scope's own verdict,
            # raised by ``_validate_qualifier_chains`` for every position
            # before pattern/``is`` candidates are ever collected: reaching
            # here means *name* is a real member of the scope.
            # The scope's selection is complete; typecheck never reads it as a module route.
            self._scope_qualified_spellings.add(node_id)
            return tuple(self._scoped_constructor_candidates.get((local_path, name), ()))
        module_qualifier = self._is_module_qualifier(chain)
        if module_qualifier:
            # A module qualifier's selection is complete, as a local scope's is.
            self._scope_qualified_spellings.add(node_id)
        selected = (
            self._imported_atom_constructor(chain, name)
            or self._routed_owner_constructor(chain, name, defer_diagnostics=True)
            or self._module_surface_constructor(chain, name)
        )
        if selected is not None:
            return (selected,)
        if module_qualifier or chain.anchor is QualifierAnchor.CURRENT_MODULE:
            raise self._module_qualifier_miss(node_id, chain, name, span)
        return ()

    def _is_module_qualifier(self, chain: QualifierChain) -> bool:
        """Whether *chain*, naming no local scope, is ``::`` alone or one route, never a type."""
        if not chain.segments:
            return True
        if len(chain.segments) > 1 or chain.anchor is QualifierAnchor.CURRENT_MODULE:
            return False
        segment = chain.segments[0]
        return chain.anchor is QualifierAnchor.MODULE or not self._names_visible_type(
            NameT(segment.name, segment.span, segment.node_id)
        )

    def _module_qualifier_miss(
        self, node_id: int, chain: QualifierChain, name: str, span: SourceSpan
    ) -> AglError:
        """Return why module qualifier *chain* selects no constructor *name*.

        A spelling naming no value or a type fails as a value reference does;
        one naming a value names no constructor.
        """
        if chain.anchor is QualifierAnchor.CURRENT_MODULE:
            if not chain.segments:
                return self._own_root_miss(chain, name, span, node_id)
            return self._own_scope_miss(chain, span)
        qname = self._resolve_qualified_qname(chain, name, span)
        if self._type_owners.is_declared(qname):
            return type_name_not_a_value(render_qualified_name(chain, name), span)
        return AglScopeError(
            f"'{render_qualified_name(chain, name)}' names no constructor.", span=span
        )

    def _imported_atom_constructor(self, chain: QualifierChain, name: str) -> ConstructorRef | None:
        """Return the constructor an import route owning the complete ``chain::name`` atom names."""
        if chain.anchor is QualifierAnchor.CURRENT_MODULE or not chain.segments:
            return None
        qname = self._try_resolve_qualified_qname(chain, name)
        return None if qname is None else self._cross_module_constructor(qname)

    def _module_surface_constructor(
        self, chain: QualifierChain, name: str
    ) -> ConstructorRef | None:
        """Return the inline member module qualifier *chain* selects by *name*, if any.

        A module qualifier is ``::`` alone (this module's own root
        declarations) or one import route segment. Its surface names the
        constructor this module declares at its root as *name*, else injects
        the terminal name of its root enums' inline members; a referenced
        member keeps its own path and is never injected. A route's own
        declarations are the caller's to resolve first. Values, patterns, and
        ``is`` tests all select a module-qualified member here. Two injected
        members are ambiguous; a name only a root enum references is a
        :class:`ReferencedMemberError`. ``None`` when *chain* is no module
        qualifier or its surface has no constructor of that name.
        """
        if chain.anchor is QualifierAnchor.CURRENT_MODULE:
            if chain.segments:
                return None
            own = tuple(
                candidate
                for candidate in self._constructor_candidates.get(name, ())
                if candidate.owner_module_id == self._module_id
            )
            declared = next((candidate for candidate in own if not candidate.owner_path), None)
            if declared is not None:
                return declared
            injected = {
                candidate: render_qualified_name(chain, f"{candidate.owner_path[0]}::{name}")
                for candidate in own
                if _is_root_inline_member(candidate)
            }
            # Earlier REPL entries' root types, then this entry's.
            roots: Iterable[tuple[str, QName]] = (
                (root, (self._module_id, root))
                for root in (
                    *self._repl_session_root_type_names,
                    *(item.name for item, path in self._type_declarations if not path),
                )
            )
        else:
            if len(chain.segments) != 1:
                return None
            route = tuple(chain.segments[0].name.split("/"))
            surfaces = qualifier_members(self._import_env, route, anchored=chain.anchored)
            if not surfaces:
                return None
            resolution = self._qualified_import_resolution(chain, name)
            if isinstance(resolution, QualResolutionAmbiguous):
                raise AmbiguousQualificationError.for_origins(
                    route,
                    (name,),
                    (
                        ImportedModuleOrigin((module, resolution.member))
                        for module in resolution.candidates
                    ),
                    anchored=chain.anchored,
                    span=chain.span,
                    local_to=self._module_id,
                )
            injected = self._route_injected_members(chain, name)
            roots = (
                (atom, origin)
                for _module, members in surfaces
                for atom, origin in members.items()
                if isinstance(atom, str)
            )
        if len(injected) > 1:
            raise self._ambiguous_constructor(
                render_qualified_name(chain, name),
                tuple(injected),
                next(iter(injected.values())),
                chain.span,
            )
        if injected:
            return next(iter(injected))
        for root, qname in roots:
            owner = self._type_owners.owner(qname)
            if (
                owner is not None
                and owner.constructor is None
                and owner.alias is None
                and name in owner.referenced
            ):
                raise ReferencedMemberError(
                    render_qualified_name(chain, root), name, span=chain.span
                )
        return None

    def _route_injected_members(
        self, chain: QualifierChain, name: str
    ) -> dict[ConstructorRef, str]:
        """Map each root enum inline member one-segment route *chain* injects as *name*.

        A re-exported enum's members are injected too. Each maps to its
        owner-qualified spelling where *chain* is written: through *chain*
        when it matches one module, else through a route selecting only its
        exposing module.
        """
        route = tuple(chain.segments[0].name.split("/"))
        surfaces = qualifier_members(self._import_env, route, anchored=chain.anchored)
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

    def _try_resolve_qualified_qname(
        self, chain: QualifierChain, name: str
    ) -> tuple[ModuleId, NameAtom] | None:
        """Resolve ``chain::name`` as one imported atom, or ``None`` on any failure."""
        result = self._qualified_import_resolution(chain, name)
        return result.qname if isinstance(result, QualResolutionFound) else None

    def _bare_constructor_candidates(self, name: str) -> tuple[ConstructorRef, ...]:
        """Return the constructor candidates an unqualified *name* can select.

        A region's scope-use contributions take precedence over the module-wide
        candidate table, so a bare spelling inside a scope selects the same
        constructor a qualified reference would. Absent a scope-use
        contribution, the enclosing named scopes are searched outward -- a
        nominal declared in the same scope (or an ancestor scope) as the bare
        spelling is reachable exactly as a bare value reference already finds
        it via ``ScopeNode.lookup``, ahead of the module-wide table used only
        for a root-declared nominal.
        """
        regional = self._regional_constructor_candidates(name)
        if regional is not None:
            return tuple(regional)
        scope: ScopeNode | None = self._scope
        while scope is not None:
            if scope.scope_path:
                owned = self._owned_scope_constructor_candidates(scope.scope_path, name)
                if owned:
                    return owned
            scope = scope.parent
        return tuple(self._constructor_candidates.get(name, ()))

    def _visible_bare_constructor_candidates(
        self, name: str, span: SourceSpan
    ) -> tuple[ConstructorRef, ...]:
        """Return a bare pattern or ``is`` spelling's candidates, rejecting a spelling with none."""
        candidates = self._bare_constructor_candidates(name)
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
        candidate behind an unrelated same-named one, non-deterministically
        (``_type_declarations_by_path`` preserves declaration order).
        """
        candidates = list(self._scoped_constructor_candidates.get((scope_path, name), ()))
        for item in self._type_declarations_by_path.get(scope_path, ()):
            candidates.extend(
                self._scoped_constructor_candidates.get((scope_path + (item.name,), name), ())
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
                node.node_id, node.qualifier, node.name, node.span
            )
            self._pattern_constructor_candidates[node.node_id] = candidates
            if len(candidates) == 1:
                self._constructor_refs[node.node_id] = candidates[0]

        walk(pattern, record_constructor_candidates)
        for candidate in pattern_binder_candidates(pattern):
            constructor_candidates = (
                self._bare_constructor_candidates(candidate.name)
                if not candidate.is_as_pattern
                else ()
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
