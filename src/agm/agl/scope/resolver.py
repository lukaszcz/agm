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
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
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
    unknown_type,
)
from agm.agl.modules.ids import RESERVED_ID, ModuleId, render_route_member, spell_declaration
from agm.agl.scope.attributes import recognize_attributes
from agm.agl.scope.imports import (
    BareRoute,
    ImportEnv,
    ItemDeclaration,
    NameAtom,
    PathAtom,
    QName,
    QualResolutionFound,
    ScopeOrigins,
    declares_bare_constructor,
    qualifier_candidates,
    qualifier_decls,
    qualifier_hides,
    qualifier_member_decls,
    qualifier_members,
    qualifier_scope_paths,
    resolve_qualified,
    route_spelling,
)
from agm.agl.scope.lookup import (
    NOT_HIDDEN,
    Candidate,
    Hiding,
    LookupKind,
    Misfit,
    QualifiedTarget,
    Reading,
    is_removed,
    lookup_bare,
    lookup_declared,
    lookup_origins,
    lookup_qualified,
    lookup_reached,
    lookup_steps,
    removes,
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
    BuiltinMethodReceiver,
    BuiltinStaticKind,
    ConstructorRef,
    ContributionLayer,
    DeclarationKey,
    DeclInfo,
    DuplicateDeclarationError,
    ImportedModuleOrigin,
    Layers,
    MissRepair,
    ModuleResolution,
    NoVisibleConstructorError,
    PatternSlot,
    QualificationOrigin,
    ReceiverOwner,
    ScopeNode,
    ScopePath,
    SlotCandidate,
    TypeArgumentsError,
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
    is_nominal_type_expr,
    member_chain,
    owner_member_selection,
    renames_target,
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
    is_builtin_type_name,
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
    ArrayT,
    DictT,
    NameT,
    TextT,
    TypeExpr,
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

    The one definition of the prefix test every reading of a contributed
    surface performs when it re-spells it relative to a path.
    """
    path = _bare_path(atom)
    return path[len(target) :] if path[: len(target)] == target else None


def _atom_under_prefix(atom: NameAtom, prefix: ScopePath) -> bool:
    """Whether *atom* falls under a selection *prefix* (a use tail or hiding item)."""
    return _relative_under(atom, prefix) is not None


def _route_root(source: ScopePath, relative: ScopePath) -> ScopePath:
    """Return *source* with a target-relative suffix trimmed back off its end."""
    return source[: len(source) - len(relative)] if relative else source


# ---------------------------------------------------------------------------
# Built-in names and reserved-name enforcement
# ---------------------------------------------------------------------------


# What one outermost read has learned each ``use`` reaches, by declaration, path and kind.
type _UseReads = dict[tuple[int, ScopePath, LookupKind], tuple[Candidate, ...]]


def _use_target(decl: UseDecl) -> ScopePath:
    """The path *decl* names as its target, beneath its anchor."""
    return tuple(segment.name for segment in decl.target)


def _use_route(decl: UseDecl, names: ScopePath) -> tuple[str, ...] | None:
    """The module route *names*, beneath *decl*'s anchor, spell alone; ``None`` for a path."""
    if len(names) == 1 and not decl.current_module and (decl.anchored or "/" in names[0]):
        return tuple(names[0].split("/"))
    return None


def _use_chain(decl: UseDecl, names: ScopePath) -> QualifierChain:
    """Spell *names* beneath *decl*'s anchor as a chain written where *decl* is."""
    anchor = (
        QualifierAnchor.CURRENT_MODULE
        if decl.current_module
        else QualifierAnchor.MODULE
        if decl.anchored
        else None
    )
    return QualifierChain(
        anchor,
        tuple(QualifierSegment(name, None, decl.span, decl.node_id) for name in names[:-1]),
        names[-1],
        decl.span,
        decl.node_id,
    )


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


def _source_order_key(span: SourceSpan) -> int:
    """Order spans by where they start."""
    return span.start_offset


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


def _type_qnames(refs: Iterable[BindingRef]) -> Iterator[QName]:
    """The full paths of *refs* that may name a type; an injected enum member never does."""
    return (_ref_qname(ref) for ref in refs if ref.contributes_a_type)


def _item_order(named: ItemDeclaration) -> PathAtom:
    """Order import items naming declarations by the path they spell."""
    return named.item


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
        # The uses this entry writes; the others a REPL session retained.
        self._entry_use_ids: set[int] = set()
        # While a use's target is read, only the uses written before it are
        # visible (``_reading_use``); ``None`` when every use is.
        self._use_horizon: int | None = None
        # What the outermost read in progress has learned about each use.
        self._use_reads: _UseReads | None = None
        # Whether each use is a single-item rename, the target it names, and
        # the declarations its ``hiding`` removes.
        self._use_renamed: dict[int, bool] = {}
        self._use_identities: dict[int, frozenset[QName]] = {}
        self._use_hidden_by: dict[int, frozenset[DeclarationKey]] = {}
        # Every enum this module reads by the names of its members, built on
        # first use.
        self._enum_member_index: dict[str, dict[QName, ConstructorRef]] | None = None
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
        # Import declaration id -> the declarations its ``hiding`` removes, by identity.
        self._hidden_by: dict[int, frozenset[DeclarationKey]] = {}
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
        # The full path each own scope path spelled otherwise declares (an
        # alias segment stands for its target's path), and those declaring
        # each; recorded as lookups first read them, once every module's
        # headers are prepared, since reading an alias needs them
        # (``_declare_scope_paths``).
        self._declared_paths: dict[ScopePath, QName] = {}
        self._declaring_paths: dict[QName, tuple[ScopePath, ...]] = {}
        self._undeclared_scope_paths: list[ScopePath] = []
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
        self._undeclared_scope_paths = sorted(
            (path for path in self._scope_nodes if path), key=_scope_path_sort_key, reverse=True
        )
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
        # read this one derivation.
        current_type_owners = {
            path: owner
            for path in {**self._repl_session_type_paths, **type_owners}
            if (owner := self._type_owners.owner((self._module_id, _bare_atom(path)))) is not None
        }
        self._declare_scope_paths_once()
        # A tail or ``hiding`` item written through an alias must name a
        # declaration whether or not anything reads it.
        for node_id in self._import_env.decl_hiding:
            self._import_hidden(node_id)
        for exposures in self._import_env.decl_tail_beneath.values():
            for named in sorted(
                {named for items in exposures.values() for named in items}, key=_item_order
            ):
                self._named_by(named)
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
            use_targets=self._use_targets(),
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

    def _declare_scope_paths(self) -> None:
        """Record the full path each own scope path declares, shorter paths first.

        A scope path whose whole spelling selects a type declares that type's
        path: through an alias, its target's (``def Geo::m`` with ``type Geo
        = Base`` declares ``Base::m``). Any other declares its parent's path
        and its own name. Own scope paths declaring one path are one path: a
        spelling of any reaches what the others declare (:meth:`own_at`).

        The first lookup reading them records them. Recording one path may
        read another (an alias target beneath an alias), so a lookup made
        while recording records the rest first. A path being recorded
        declares nothing yet, and one beneath it waits for it; so does one
        selecting a type not :meth:`~TypeOwnerIndex.settled` yet.
        """
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
                local_to=self._module_id,
            )
            key = found.key if isinstance(found, QualifiedTarget) else None
            if key is not None and not self._type_owners.settled(_key_qname(key)):
                waiting.append(path)
                continue
            if key is not None:
                qname = _key_qname(self.identity(key))
            else:
                module_id, atom = self._declared_paths.get(parent, (self._module_id, ()))
                qname = (module_id, _bare_atom((*_bare_path(atom), path[-1])))
            self._declared_paths[path] = qname
            if qname != (self._module_id, _bare_atom(path)):
                self._declaring_paths[qname] = (*self._declaring_paths.get(qname, ()), path)
        self._undeclared_scope_paths.extend(reversed(waiting))

    def _declaring(self) -> Mapping[QName, tuple[ScopePath, ...]]:
        """Map each full path that own scope paths spelled otherwise declare to those paths."""
        self._declare_scope_paths()
        return self._declaring_paths

    def _declare_scope_paths_once(self) -> None:
        """Reject a name declared twice beneath own scope paths declaring one path."""
        for (module_id, atom), paths in self._declaring().items():
            spelled = _bare_path(atom)
            if module_id == self._module_id and spelled in self._scope_nodes:
                paths = (spelled, *paths)
            if len(paths) > 1:
                self._declare_once_beneath(paths)

    def _declare_once_beneath(self, paths: tuple[ScopePath, ...]) -> None:
        """Reject a name declared twice beneath *paths*, which declare one path.

        A retained REPL entry's declaration beneath one of them counts, unless
        this entry redeclares it at its own spelling. The later declaration of
        this entry, or an alias of this entry making *paths* one after both,
        is reported.
        """
        aliases = [
            item.span
            for path in paths
            for index in range(len(path))
            if isinstance(
                item := self._declaration_items.get((self._module_id, path[:index], path[index])),
                TypeAlias,
            )
        ]
        declared: dict[str, dict[ScopePath, str]] = {}
        for member_path in self._repl_session_ordinary_member_paths:
            if member_path[:-1] in paths:
                declared.setdefault(member_path[-1], {})[member_path[:-1]] = "ordinary"
        for (_module_id, path, name), entity in self._scope_entity_kinds.items():
            if path in paths:
                retained_type = entity == "scope" and (*path, name) in self._type_paths
                declared.setdefault(name, {})[path] = "type" if retained_type else entity
        for name, entities in declared.items():
            kinds = list(entities.values())
            if len(kinds) > 1 and ("ordinary" in kinds or kinds.count("type") > 1):
                spans = (
                    self._declaration_items[key].span
                    for path in entities
                    if (key := (self._module_id, path, name)) in self._declaration_items
                )
                later = max((*spans, *aliases), key=_source_order_key)
                raise DuplicateDeclarationError(name, span=later)

    def _declares(self, qname: QName) -> bool:
        """Whether a declaration of any kind stands at full path *qname*."""
        return any(
            self._declared_at(qname, ContributionLayer.DECLARED, kind).candidates
            for kind in LookupKind
        )

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

    def _classify_function_head(self, declaration: FuncDef, written_in: ScopePath) -> None:
        """Classify one ``def``'s head, written in named scope *written_in*, and its receiver,
        after preceding lexical contributions are visible."""
        head = declaration.receiver_type
        if head is None:
            self._reject_applied_head(written_in, declaration.scope_path[len(written_in) :])
        elif not declaration.is_method:
            raise TypeArgumentsError(declaration.scope_path[-1].name, None, span=head.span)
        if not declaration.is_method:
            return
        receiver = declaration.params[0]
        owner_path = tuple(segment.name for segment in declaration.scope_path)
        owner = (
            self._receiver_head(owner_path, head)
            if head is not None
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

    def _reject_applied_head(self, base: ScopePath, written: Sequence[ScopeSegment]) -> None:
        """Reject a declaration path, *written* beneath scope *base*, through a type application.

        A path beneath a segment selecting an applied alias reaches only the
        application's inline members (the segment type-argument rule), which
        no declaration names.
        """
        path = base
        for segment in written:
            path = (*path, segment.name)
            found = lookup_declared(
                self, path, None, LookupKind.TYPE, span=segment.span, local_to=self._module_id
            )
            if (
                isinstance(found, QualifiedTarget)
                and found.key is not None
                and self.applies(found.key)
            ):
                raise TypeArgumentsError(segment.name, None, span=segment.span)

    def _receiver_head(self, owner_path: ScopePath, head: TypeExpr) -> ReceiverOwner:
        """Classify method head *head*, written with type arguments, at *owner_path*.

        Only a builtin receiver in its bare generic form (``array[E]``,
        ``dict[K, V]``) or ``dict[text, V]`` takes them; any other head
        declares beneath a type application.
        """
        if isinstance(head, ArrayT) and isinstance(head.elem, NameT):
            builtin = BuiltinMethodReceiver("array", head.elem.name)
        elif (
            isinstance(head, DictT)
            and isinstance(head.value, NameT)
            and isinstance(head.key, (NameT, TextT))
        ):
            key = head.key.name if isinstance(head.key, NameT) else None
            builtin = BuiltinMethodReceiver("dict", head.value.name, key)
        elif isinstance(head, (ArrayT, DictT)):
            raise AglScopeError(
                "Builtin method receivers must use their bare generic form.", span=head.span
            )
        else:
            raise TypeArgumentsError(owner_path[-1], None, span=head.span)
        return ReceiverOwner(self._module_id, owner_path, builtin)

    def _receiver_owner(
        self, segments: tuple[ScopeSegment, ...], receiver: Param, written_in: ScopePath
    ) -> ReceiverOwner | None:
        """Return the type method path *segments*' receiver attaches to, if any.

        An empty path has none. The receiver takes the type declared at the
        whole path (:meth:`_receiver_type_owner`); the ``def``'s own
        qualifier -- the segments beyond the enclosing regions *written_in*
        -- is what a miss is reported on. An unannotated receiver's rejection
        is raised; an annotated one only makes the ``def`` an ordinary
        function.
        """
        owner: ReceiverOwner | AglError | None = None
        if segments:
            owner = self._receiver_type_owner(segments, segments[len(written_in) :], receiver.span)
        if receiver.type_expr is not None:
            return owner if isinstance(owner, ReceiverOwner) else None
        if isinstance(owner, AglError):
            raise owner
        if owner is None:
            raise AglScopeError("'self' requires an enclosing type scope.", span=receiver.span)
        return owner

    def _receiver_type_owner(
        self,
        segments: tuple[ScopeSegment, ...],
        written: tuple[ScopeSegment, ...],
        span: SourceSpan,
    ) -> ReceiverOwner | AglError | None:
        """Return the type declared at method path *segments*, or why none.

        The whole path is read at its parent step alone (:func:`lookup_declared`);
        a qualifier *written* in the ``def`` itself gets the verdict of a miss.
        A path declaring no type may end in a constructor its parent step
        reads bare, whose owner the bare constructor decision there
        (:meth:`_step_value_constructors`) selects; a one-segment head -- as
        written, else the whole path -- may name a built-in receiver type.
        """
        path = tuple(segment.name for segment in segments)
        chain = (
            QualifierChain(
                None,
                tuple(
                    QualifierSegment(segment.name, None, segment.span, segment.node_id)
                    for segment in written[:-1]
                ),
                written[-1].name,
                span_covering(written[0].span, written[-1].span),
                written[-1].node_id,
            )
            if len(written) > 1
            else None
        )
        found = lookup_declared(
            self, path, chain, LookupKind.TYPE, span=span, local_to=self._module_id
        )
        if isinstance(found, AglError):
            return found
        key = None if found is None else found.key
        owner = None if key is None else self._receiver_key_owner(key, span)
        if owner is not None:
            return owner
        if len(written or segments) == 1 and path[-1] in BUILTIN_METHOD_RECEIVER_NAMES:
            if path[-1] in ("array", "dict"):
                # A generic receiver binds its type parameters only in applied form.
                raise AglScopeError(
                    "Builtin method receivers must use their bare generic form "
                    "(array[T] or dict[K, V]).",
                    span=(written or segments)[-1].span,
                )
            return ReceiverOwner(self._module_id, path[-1:], BuiltinMethodReceiver(path[-1]))
        owners: dict[ReceiverOwner, list[QualificationOrigin]] = {}
        for candidate, layers in self._step_value_constructors(path[:-1], path[-1]).items():
            owners.setdefault(
                ReceiverOwner(
                    candidate.owner_module_id, (*candidate.owner_path, candidate.owner_name)
                ),
                [],
            ).extend(contribution_origins(candidate.qname, layers))
        if len(owners) > 1:
            return AmbiguousQualificationError.for_origins(
                (),
                (path[-1],),
                itertools.chain.from_iterable(owners.values()),
                span=span,
                local_to=self._module_id,
            )
        return next(iter(owners), None)

    def _receiver_key_owner(self, key: DeclarationKey, span: SourceSpan) -> ReceiverOwner | None:
        """Return the receiver owner type *key* names, rejecting an alias of its own.

        A renaming alias names its target (:meth:`TypeOwnerIndex.identity`),
        so its methods are the target's; any other alias is no receiver.
        """
        qname = self._type_owners.identity(_key_qname(key))
        owner = self._type_owners.owner(qname)
        if owner is not None and owner.alias is not None:
            self._raise_alias_receiver(key[2], owner.alias, span)
        module_id, path, name = self._qname_decl_key(qname)
        if module_id == self._module_id:
            return (
                ReceiverOwner(module_id, (*path, name))
                if self._type_owners.is_declared(qname)
                else None
            )
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
                # Recorded with the headers (``_resolve_headers``); what it
                # names is checked where it is written.
                self._validate_use(item)
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
        """Record the ``use`` and contribute the region-scoped ``import`` headers of *items*.

        Its regions' headers too. A use is read where it is written whenever
        it is used; an import contributes bindings at once, and a constructor
        that depends on type owners is deferred to ``resolve``. Every import
        here is in its sequence's header: placement was checked first.
        """
        regional_exposures = self._regional_import_exposures(items)
        for item in items:
            if isinstance(item, UseDecl):
                self._scope.uses.append(item)
                self._entry_use_ids.add(item.node_id)
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
                        self._contribute_regional_constructors, scope, decl, atom, qname, exposures
                    )
                )

    def _contribute_regional_constructors(
        self,
        scope: ScopeNode,
        decl: ImportDecl,
        atom: NameAtom,
        qname: QName,
        exposures: Mapping[NameAtom, Collection[QName]],
    ) -> None:
        """Contribute the constructors region-scoped *decl* makes bare *atom* through *qname*.

        *qname*'s own, and an enum's variants when *atom* is one name. What
        *decl*'s ``hiding`` removes is none. Deferred, since the constructors
        and what the ``hiding`` names read the type owners.
        """
        hiding = self._hiding((decl.node_id,), qname)
        if removes(hiding, self._qname_decl_key(qname), self.identity):
            return
        constructor = self._cross_module_constructor(qname)
        if constructor is not None:
            scope.contribute_bare_constructor(atom, constructor, ContributionLayer.IMPORTED)
        if isinstance(atom, str):
            self._contribute_regional_enum_variants(scope, qname, decl.span, exposures, hiding)

    def _contribute_regional_enum_variants(
        self,
        scope: ScopeNode,
        qname: QName,
        span: SourceSpan,
        exposures: Mapping[NameAtom, Collection[QName]],
        hiding: Hiding,
    ) -> None:
        """Expand a bare-exposed enum type into its own bare variants, region-scoped.

        A bare enum *type* name alone does not make its variants callable or
        matchable -- ``_build_cross_module_constructor_candidates`` performs
        the same expansion module-wide, from a root-position bare exposure.
        Mirroring it here covers the scoped case, whose bare exposure never
        reaches that module-wide table. A member *hiding* removes is none.
        """
        selected_qnames = frozenset(qname for qnames in exposures.values() for qname in qnames)
        for atom, constructor, path in self._enum_variant_members(qname):
            # An inline member the tail hides, or whose spelling a same-named
            # record or exception exposed bare already owns, stays unexposed.
            if (
                constructor.inline_enum_owner_decl_node_id is not None
                and (
                    (qname[0], _bare_atom(path)) not in selected_qnames
                    or declares_bare_constructor(exposures.get(atom, ()), self._all_public_types)
                )
            ) or removes(hiding, self._qname_decl_key(constructor.qname), self.identity):
                continue
            scope.contribute_bare(
                atom, self._variant_binding_ref(constructor, span), ContributionLayer.IMPORTED
            )
            scope.contribute_bare_constructor(atom, constructor, ContributionLayer.IMPORTED)

    # -- ``use`` declarations: read where written, whenever used --

    @contextmanager
    def _reading_use(self, decl: UseDecl) -> Iterator[_UseReads]:
        """Read what *decl* reaches: only the uses written before it are visible meanwhile.

        What one outermost read learns about every use is kept until it ends.
        """
        horizon, reads = self._use_horizon, self._use_reads
        current: _UseReads = {} if reads is None else reads
        self._use_horizon, self._use_reads = decl.node_id, current
        try:
            yield current
        finally:
            self._use_horizon, self._use_reads = horizon, reads

    @contextmanager
    def _full_view(self) -> Iterator[None]:
        """Read with every use visible, as the type-owner index, which keeps its answers, asks."""
        horizon = self._use_horizon
        self._use_horizon = None
        try:
            yield
        finally:
            self._use_horizon = horizon

    def _visible_uses(self, layer: ScopeNode) -> Iterator[UseDecl]:
        """Yield *layer*'s uses a read now sees.

        Within a use's read, only those written before it. A use an earlier
        REPL entry retained yields to one this entry writes in the same
        region naming the same target.
        """
        horizon = self._use_horizon
        for decl in layer.uses:
            if horizon is not None and decl.node_id >= horizon:
                continue
            if decl.node_id in self._entry_use_ids or not self._use_replaced(
                layer.scope_path, decl
            ):
                yield decl

    def _use_replaced(self, site: ScopePath, decl: UseDecl) -> bool:
        """Whether this entry writes a visible use in region *site* naming *decl*'s target."""
        horizon = self._use_horizon
        replacing = [
            use
            for use in self._scope_nodes[site].uses
            if use.node_id in self._entry_use_ids and (horizon is None or use.node_id < horizon)
        ]
        return bool(replacing) and self._use_identity(site, decl) in {
            self._use_identity(site, use) for use in replacing
        }

    def _use_identity(self, site: ScopePath, decl: UseDecl) -> frozenset[QName]:
        """The scopes and types *decl*, written in region *site*, names as its target.

        A single-item rename's target is the path holding the item.
        """
        found = self._use_identities.get(decl.node_id)
        if found is None:
            target = _use_target(decl)
            with self._reading_use(decl):
                named = target[:-1] if len(target) > 1 and self._use_renames(site, decl) else target
                found = self._use_path_origins(site, decl, named)
            self._use_identities[decl.node_id] = found
        return found

    def _use_renames(self, site: ScopePath, decl: UseDecl) -> bool:
        """Whether *decl*, written in region *site*, is a single-item rename.

        ``use P::m as A`` is ``use P::{m as A}`` when ``P::m`` is a
        declaration, and ``use T as A`` when ``T`` is a type; otherwise the
        alias names the whole target.
        """
        if decl.alias is None:
            return False
        found = self._use_renamed.get(decl.node_id)
        if found is None:
            target = _use_target(decl)
            kinds = (LookupKind.VALUE, LookupKind.TYPE) if len(target) > 1 else (LookupKind.TYPE,)
            with self._reading_use(decl):
                found = any(self._use_path_reached(site, decl, target, kind) for kind in kinds) or (
                    len(target) > 1 and self._pending_at(site, decl, target)
                )
            self._use_renamed[decl.node_id] = found
        return found

    def _pending_at(self, site: ScopePath, decl: UseDecl, names: ScopePath) -> bool:
        """Whether *names*, as *decl* in region *site* spells them, is an own member not yet bound.

        A non-static module installs a scoped let/var only where the walk
        reaches it.
        """
        if decl.current_module:
            bases: Iterable[ScopePath] = ((),)
        elif decl.anchored:
            return False
        else:
            bases = lookup_steps(site)
        return any(
            self._scope_entity_kinds.get((self._module_id, (*base, *names[:-1]), names[-1]))
            == "ordinary"
            for base in bases
        )

    def _use_rests(
        self, site: ScopePath, decl: UseDecl, relative: ScopePath
    ) -> tuple[ScopePath, ...]:
        """The paths beneath *decl*'s target it exposes as *relative* in region *site*.

        An alias stands for the target, a single-item rename adds the item
        under its own name, a glob exposes every path its ``hiding`` leaves,
        and a tail each path under an item, a renamed item under its rename
        too.
        """
        rests: dict[ScopePath, None] = {}
        if decl.tail is None:
            if relative[0] == decl.alias:
                rests[relative[1:]] = None
            target = _use_target(decl)
            if len(target) > 1 and relative[0] == target[-1] and self._use_renames(site, decl):
                rests[relative[1:]] = None
        elif not decl.tail:
            atom = _bare_atom(relative)
            if not any(_atom_under_prefix(atom, _item_path(item)) for item in decl.hidden):
                rests[relative] = None
        else:
            for item in decl.tail:
                path = _item_path(item)
                if _atom_under_prefix(_bare_atom(relative), path):
                    rests[relative] = None
                if relative[0] == item.rename:
                    rests[(*path, *relative[1:])] = None
        return tuple(rests)

    def _use_reached(
        self, site: ScopePath, decl: UseDecl, relative: ScopePath, kind: LookupKind
    ) -> tuple[Candidate, ...]:
        """What *decl*, written in region *site*, reaches of *kind* as *relative*.

        Each path beneath the target it exposes there is read as the use
        site reads the target spelled with it. Only a type the target's path
        reaches projects its member table there: a type the use exposes
        projects where the exposed path is read, a renamed one included.
        """
        with self._reading_use(decl) as reads:
            key = (decl.node_id, relative, kind)
            found = reads.get(key)
            if found is None:
                target = _use_target(decl)
                owners_within = len(target) - (1 if self._use_renames(site, decl) else 0)
                found = tuple(
                    candidate
                    for rest in self._use_rests(site, decl, relative)
                    for candidate in self._use_path_reached(
                        site, decl, (*target, *rest), kind, owners_within
                    )
                )
                reads[key] = found
            return found

    def _use_exposure(
        self, site: ScopePath, decl: UseDecl, relative: ScopePath, kind: LookupKind
    ) -> Iterator[Candidate]:
        """What *decl*, written in region *site*, contributes of *kind* as *relative*.

        What it reaches, and a bare enum member an enum it exposes injects.
        """
        reached: Iterable[Candidate] = self._use_reached(site, decl, relative, kind)
        if kind is not LookupKind.TYPE and len(relative) == 1:
            reached = itertools.chain(reached, self._use_injected(site, decl, relative[0]))
        for candidate in reached:
            used = self._as_used(site, decl, candidate, alone=len(relative) == 1)
            if self._fits(used.target, kind):
                yield used

    def _as_used(
        self, site: ScopePath, decl: UseDecl, candidate: Candidate, *, alone: bool
    ) -> Candidate:
        """*candidate* as *decl*, in region *site*, contributes it, exposed *alone* or owned.

        An alias segment stands for its target's path, so a member an alias
        selects that the use exposes *alone* -- no longer spelled beneath
        the alias -- is the target's own. Each way it was reached also
        removes what the use's ``hiding`` names.
        """
        target = candidate.target
        declaration = candidate.origin.declaration
        key = target.key
        constructor = target.constructor
        member = (
            None
            if not alone or key is None or constructor is None or constructor.member is None
            else self._type_owners.owner_member(
                self._qname_decl_key((key[0], _bare_atom(key[1]))), key[2]
            )
        )
        if member is not None:
            target = QualifiedTarget(
                (member.owner_module_id, member.owner_path, member.owner_name), None, member
            )
            declaration = member.qname
        layer = ContributionLayer.USE
        removed = self._use_hidden(site, decl)
        hiding = (
            frozenset(way | removed for way in candidate.hiding) if removed else candidate.hiding
        )
        return Candidate(target, layer, contribution_origin(declaration, layer), hiding)

    def _use_hidden(self, site: ScopePath, decl: UseDecl) -> frozenset[DeclarationKey]:
        """The declarations *decl*'s ``hiding``, read in region *site*, names, by identity."""
        found = self._use_hidden_by.get(decl.node_id)
        if found is None:
            target = _use_target(decl)
            with self._reading_use(decl):
                found = frozenset(
                    self.identity(key)
                    for item in decl.hidden
                    for kind in LookupKind
                    for candidate in self._use_path_reached(
                        site, decl, (*target, *_item_path(item)), kind, len(target)
                    )
                    if (key := candidate.target.key) is not None
                )
            self._use_hidden_by[decl.node_id] = found
        return found

    def _use_injected(self, site: ScopePath, decl: UseDecl, name: str) -> Iterator[Candidate]:
        """Yield the enum member *name* each enum *decl*, in region *site*, exposes injects.

        An inline member its ``hiding`` removed, or whose name a record or
        exception the use exposes owns, is not injected.
        """
        spellings = [] if decl.alias is None else [decl.alias]
        spellings.extend(
            item.rename
            for item in decl.tail or ()
            if item.rename is not None and not item.scope_path
        )
        layer = ContributionLayer.USE
        for qname, constructor in self._enum_members_named(name).items():
            inline = constructor.inline_enum_owner_decl_node_id is not None
            key = self._qname_decl_key(qname)
            exposed = any(
                any(
                    candidate.target.key == key
                    for candidate in self._use_reached(site, decl, (spelling,), LookupKind.TYPE)
                )
                and (not inline or self._use_rests(site, decl, (spelling, name)))
                for spelling in (key[2], *spellings)
            )
            if not exposed or (inline and self._exposes_standalone(site, decl, name)):
                continue
            yield Candidate(
                QualifiedTarget(
                    (constructor.owner_module_id, constructor.owner_path, constructor.owner_name),
                    self._variant_binding_ref(constructor, decl.span),
                    constructor,
                ),
                layer,
                contribution_origin(constructor.qname, layer),
            )

    def _exposes_standalone(self, site: ScopePath, decl: UseDecl, name: str) -> bool:
        """Whether *decl*, in region *site*, exposes a record or exception as *name*."""
        for candidate in self._use_reached(site, decl, (name,), LookupKind.VALUE):
            target = candidate.target
            if (
                target.key is None
                or target.constructor is None
                or target.constructor.inline_enum_owner_decl_node_id is not None
            ):
                continue
            owner = self._type_owners.owner(_key_qname(target.key))
            if owner is not None and owner.alias is None and owner.constructor is not None:
                return True
        return False

    def _enum_members_named(self, name: str) -> Mapping[QName, ConstructorRef]:
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

    def _use_origins(self, site: ScopePath, decl: UseDecl, relative: ScopePath) -> frozenset[QName]:
        """The scopes and types *decl*, written in region *site*, exposes as *relative*.

        A path holding a tail item names what the target's path there names.
        """
        with self._reading_use(decl):
            target = _use_target(decl)
            paths = [(*target, *rest) for rest in self._use_rests(site, decl, relative)]
            if any(
                len(path := _item_path(item)) > len(relative) and path[: len(relative)] == relative
                for item in decl.tail or ()
            ):
                paths.append((*target, *relative))
            return frozenset().union(*(self._use_path_origins(site, decl, path) for path in paths))

    def _use_path_reached(
        self,
        site: ScopePath,
        decl: UseDecl,
        names: ScopePath,
        kind: LookupKind,
        owners_within: int | None = None,
    ) -> tuple[Candidate, ...]:
        """What *names*, spelled beneath *decl*'s anchor in region *site*, reaches of *kind*.

        With *owners_within*, only a type its first that many names reach projects.
        """
        if _use_route(decl, names) is not None:
            return ()
        return lookup_reached(
            self,
            _use_chain(decl, names),
            site,
            kind,
            local_to=self._module_id,
            owners_within=owners_within,
        )

    def _use_path_origins(
        self, site: ScopePath, decl: UseDecl, names: ScopePath
    ) -> frozenset[QName]:
        """The scopes and types *names*, spelled beneath *decl*'s anchor in region *site*, name."""
        route = _use_route(decl, names)
        if route is not None:
            return self._routed_origins(route, (), anchored=decl.anchored)
        return lookup_origins(self, _use_chain(decl, names), site)

    def _names_qualifier(
        self, site: ScopePath, decl: UseDecl, names: ScopePath, owners_within: int | None = None
    ) -> bool:
        """Whether *names*, spelled beneath *decl*'s anchor in region *site*, names a qualifier.

        With *owners_within*, only a type its first that many names reach projects.
        """
        return bool(self._use_path_origins(site, decl, names)) or bool(
            self._use_path_reached(site, decl, names, LookupKind.TYPE, owners_within)
        )

    def _validate_use(self, decl: UseDecl) -> None:
        """Check that *decl*'s target names a qualifier and each tail or hiding item a path beneath.

        A single-item rename's target is the declaration it renames. An item
        is a path declared beneath the target: a type the target reaches
        projects its member table, and an alias inside the item stands for
        its target's path.
        """
        site = self._scope.scope_path
        target = _use_target(decl)
        with self._reading_use(decl):
            if not self._use_renames(site, decl) and not self._names_qualifier(site, decl, target):
                raise UnknownQualifierError(
                    _use_target_spelling(decl), span=decl.span, repair=MissRepair.IMPORT_MODULE
                )
            for item in (*(decl.tail or ()), *decl.hidden):
                item_path = _item_path(item)
                path = (*target, *item_path)
                if not (
                    self._names_qualifier(site, decl, path, len(target))
                    or self._use_path_reached(site, decl, path, LookupKind.VALUE, len(target))
                    or self._pending_at(site, decl, path)
                ):
                    raise UnknownMemberError(_use_target_spelling(decl, item_path), span=decl.span)

    def _use_targets(self) -> dict[int, frozenset[QName]]:
        """Every use this REPL entry reads, with the target it names now."""
        if not self._allow_root_statements:
            return {}
        return {
            decl.node_id: self._use_identity(layer.scope_path, decl)
            for layer in (*self._scope_nodes.values(), *self._layer_chain(self._root_scope))
            for decl in layer.uses
        }

    def _scope_route_origins(self, route: BareRoute) -> ScopeOrigins:
        """Return the declarations scope route *route* reaches, through any number of re-exports."""
        module, path = route
        return self._import_env.scope_origins_by_route.get(
            route, frozenset({(module, _bare_atom(path))})
        )

    def _resolve_scope_region(self, region: ScopeRegion) -> None:
        """Resolve a named region in its member layer."""
        self._reject_applied_head(self._scope.scope_path, (region.segment,))
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
                self._classify_function_head(node, written_in)
                self._validate_qualifier_chains(node, node.type_params)
                self._resolve_program_config(node)
                self._resolve_params_and_body(node)
            return
        # Defaults are resolved in the enclosing (root) scope — they are
        # evaluated in the function's definition scope.
        self._classify_function_head(node, ())
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
        self._reject_applied_head(
            self._scope.scope_path, node.scope_path[len(self._scope.scope_path) :]
        )
        if isinstance(node, TypeAlias):
            self._validate_alias(path, node)
        else:
            with self._named_scope(path):
                self._validate_type_decl(node)

    def alias_target(
        self, qname: QName, alias: TypeAlias, spelling: NameT | AppliedT, *, validate: bool
    ) -> TypeSelection | None:
        """Return what alias *qname*'s nominal target *spelling* selects.

        Exactly what the target's own type position records where the alias is
        declared, validating the alias first when the type-owner index asks
        before the ordered walk reaches it. Unless *validate*, the target is
        only read, selecting nothing when it names nothing; the ordered walk
        validates the alias.
        """
        path = _bare_path(qname[1])[:-1]
        if validate:
            self._validate_alias(path, alias)
            return self._owner_declarations.get(selection_node_id(spelling))
        with self._full_view(), self._named_scope(path):
            found = self._type_name_target(spelling)
        return found.selection if isinstance(found, QualifiedTarget) else None

    def _validate_alias(self, path: ScopePath, alias: TypeAlias) -> None:
        """Validate *alias*, declared at scope *path*, once.

        A bare nominal target selecting nothing that names no built-in type is
        unknown here, where the alias is declared, whoever reads it.
        """
        if alias.node_id not in self._validated_aliases:
            self._validated_aliases.add(alias.node_id)
            with self._full_view(), self._named_scope(path):
                self._validate_type_decl(alias)
            target = alias.type_expr
            if (
                is_nominal_type_expr(target, alias.type_params)
                and target.qualifier is None
                and target.node_id not in self._owner_declarations
                and not is_builtin_type_name(target.name)
            ):
                raise unknown_type(target.name, alias.span)

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
        name = target.name
        # A target is selected exactly as the same spelling read as a value
        # is; a bare one reaches an imported mutable binding just like a read.
        ref = (
            self._bare_value(name, target.span)
            if target.qualifier is None
            else self._qualified_assign_target(name, target.qualifier, target.span)
        )
        if ref is None:
            raise AglScopeError(
                f"'{name}' is not declared; assignment requires an existing mutable binding.",
                span=target.span,
            )
        # Mutability is not decided here: a field-directed pattern slot's final
        # binding is only known once checking selects it, so type checking owns
        # the ``:=``-on-immutable rejection for every target, qualified or not.
        self._resolution[node.node_id] = ref
        self._resolve_expr(node.value)

    def _qualified_assign_target(
        self, name: str, qualifier: QualifierChain, span: SourceSpan
    ) -> BindingRef | None:
        """Return the binding qualified target ``qualifier::name`` selects, if a value.

        Selected as the same spelling read as a value is
        (:meth:`_select_qualified`). A constructor only a type owner's own
        table selects is its constructor binding; a declaration naming no
        value (a type without a constructor) is none.
        """
        found = self._select_qualified(qualifier, name, LookupKind.VALUE, span)
        if isinstance(found, Misfit):
            return None
        if found.ref is None and found.constructor is not None:
            return self._variant_binding_ref(found.constructor, span)
        return found.ref

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
            and not self.own_origins(relative_path)
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
        """This module's own declarations of *kind* at full *path*.

        Those spelled *path*, and those beneath every own scope path spelled
        otherwise that declares the path its parent does.
        """
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

    def _own_spelled_at(self, path: ScopePath, kind: LookupKind) -> Reading:
        """This module's own declaration of *kind* spelled *path*."""
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
        """What the contributions anchored at or above *step* reach at full *path*.

        Import tails' snapshots, and what each visible ``use`` exposes there.
        """
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
            (*(candidate for candidate in imported if self._fits(candidate.target, kind)), *used)
        )

    def routed_at(self, chain: QualifierChain, path: ScopePath, kind: LookupKind) -> Reading:
        """What *chain*'s leading module route alone reaches at *path* beneath it."""
        candidates = (
            Candidate(
                self._contributed_target(self._cross_module_binding_ref(qname), ()),
                ContributionLayer.IMPORTED,
                ImportedModuleOrigin(qname),
                self._hiding(decls, qname),
            )
            for qname, decls in qualifier_member_decls(
                self._import_env, chain.leading_route, _bare_atom(path), anchored=chain.anchored
            ).items()
        )
        return Reading(tuple(c for c in candidates if self._fits(c.target, kind)))

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
        kind: LookupKind,
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
        (``def Geo::m`` with ``Geo`` an alias of an imported ``Base``); an own
        owner's spellings are own scope paths, which :meth:`own_at` reads.
        """
        declared = _key_qname(self.identity(owner))
        selected = self._selected_by_table(owner, layer, rest, chain, kind)
        if declared[0] == self._module_id:
            return selected
        return (
            sum(
                (
                    self.own_at((*spelling, *rest), kind)
                    for spelling in self._declaring().get(declared, ())
                ),
                Reading(),
            )
            + selected
        )

    def _selected_by_table(
        self,
        owner: DeclarationKey,
        layer: ContributionLayer,
        rest: ScopePath,
        chain: QualifierChain,
        kind: LookupKind,
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
            if table.target is not None and not (
                index == len(segments)
                and (name in table.members or table.select(name, segments[-1].name) is not None)
            ):
                return self._beneath_alias(
                    current, table, rest[index - start :], layer, chain, kind
                )
            current = (current[0], _bare_atom((*_bare_path(current[1]), name)))
        if (table.alias is None and name in table.members) or self._type_owners.is_declared(
            current
        ):
            return Reading(refusals=(HiddenMemberError(spelling, name, span=chain.span),))
        constructor = table.select(name, segments[-1].name)
        if constructor is None:
            return Reading()
        key = self._qname_decl_key(current)
        if layer is ContributionLayer.DECLARED and self.identity(key)[0] != self._module_id:
            # Another module's declaration is never this module's own, not
            # even reached through an own alias.
            layer = ContributionLayer.IMPORTED
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
    ) -> Reading:
        """What *path* beneath *alias* (whose owner is *table*) selects as a declaration of *kind*.

        An alias segment stands for its target's path; a path a ``hiding``
        at the alias's site removed is refused.
        """
        beneath = self._type_owners.beneath_alias(alias, table, path)
        if beneath is None:
            return Reading()
        qname, hidden = beneath
        if hidden:
            spelling = render_qualifier_path(chain)
            return Reading(refusals=(HiddenMemberError(spelling, path[-1], span=chain.span),))
        return self._declared_at(qname, layer, kind)

    def _declared_at(self, qname: QName, layer: ContributionLayer, kind: LookupKind) -> Reading:
        """The declaration at full path *qname*, as one of *kind*, made visible by *layer*."""
        module_id, atom = qname
        targets: Iterable[QualifiedTarget]
        if module_id == self._module_id:
            targets = (
                candidate.target for candidate in self.own_at(_bare_path(atom), kind).candidates
            )
        elif qname in self._decl_info:
            targets = (self._contributed_target(self._cross_module_binding_ref(qname), ()),)
            # Another module's declaration is never this module's own, not
            # even reached through an own alias.
            if layer is ContributionLayer.DECLARED:
                layer = ContributionLayer.IMPORTED
        else:
            targets = ()
        origin = contribution_origin(qname, layer)
        return Reading(
            tuple(
                Candidate(target, layer, origin) for target in targets if self._fits(target, kind)
            )
        )

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
        return (
            reached is not None
            and reached.alias is not None
            and isinstance(reached.alias.type_expr, AppliedT)
            and not renames_target(reached.alias)
        )

    def hidden_at(self, step: ScopePath, path: ScopePath) -> bool:
        """Whether a ``hiding`` of a contribution anchored at or above *step* removed *path*."""
        for layer in self._layer_chain(self._scope_nodes[step]):
            atom = _bare_atom(path[len(layer.scope_path) :])
            if any(
                _atom_under_prefix(atom, _item_path(item))
                for decl in self._visible_uses(layer)
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
                self.identity,
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
            and self.aliases(self._qname_decl_key(named.declaration))
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
            return self._qname_decl_key(owners.declaration(qname))
        return self._qname_decl_key(owners.identity(qname))

    def denotes(self, key: DeclarationKey) -> object:
        """What *key* names in an ambiguity: its identity, or the type an alias denotes."""
        identity = self.identity(key)
        denoted = self._type_owners.denotation(_key_qname(identity))
        return identity if denoted is None else denoted

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
            for decl in self._visible_uses(layer):
                found |= self._use_origins(layer.scope_path, decl, relative)
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
                rest = _relative_under(exposed, path[len(anchor) :])
                if rest is not None:
                    found.update(
                        origin
                        for module, source in sources
                        for origin in self._scope_route_origins((module, _route_root(source, rest)))
                    )
        return frozenset(found | self._routed_origins((path[0],), path[1:], anchored=False))

    def routed_origins(self, chain: QualifierChain, path: ScopePath) -> frozenset[QName]:
        """The scopes and types *chain*'s leading module route reaches as *path* beneath it."""
        return self._routed_origins(chain.leading_route, path, anchored=chain.anchored)

    def _routed_origins(
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
            if any(_relative_under(atom, path) is not None for atom in scope_paths):
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
        rest = _relative_under(exposed, path)
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
        atom = _bare_atom(path)
        imported = dict(env.unqualified_decls.get(atom, {}))
        if path[1:]:
            routed = qualifier_member_decls(env, (path[0],), _bare_atom(path[1:]))
            for qname, decls in routed.items():
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
                | {self._qname_decl_key(q) for q in withheld.get(node_id, {}).get(entry, ())}
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
        return self.identity(
            self._qname_decl_key((module_id, _bare_atom((*_bare_path(atom), *path))))
        )

    def tail_removes(self, exposed: NameAtom, qname: QName, declaration: QName) -> bool:
        """Whether every root import tail exposing *qname* as *exposed* removes *declaration*.

        *declaration* is *qname*'s own or a member's beneath it.
        """
        decls = self._import_env.unqualified_decls.get(exposed, {}).get(qname, ())
        return removes(self._hiding(decls, qname), self._qname_decl_key(declaration), self.identity)

    def _imported_bindings(self, step: ScopePath, path: ScopePath) -> dict[BindingRef, Layers]:
        """Return what import tails anchored at or above *step* bind at full *path*, with layers.

        A binding every declaration contributing it hides is none.
        """
        return {
            ref: layers
            for ref, (layers, hiding) in self._imported(step, path).items()
            if not removes(hiding, (ref.module_id, ref.scope_path, ref.name), self.identity)
        }

    def _use_exposures(
        self, step: ScopePath, path: ScopePath, kind: LookupKind
    ) -> Iterator[tuple[Candidate, UseDecl]]:
        """Yield what each use visible at or above *step* exposes of *kind* at full *path*."""
        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            for decl in self._visible_uses(layer):
                for candidate in self._use_exposure(layer.scope_path, decl, _bare_path(atom), kind):
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
            self._exposed_binding(candidate.target, decl.span)
            for candidate, decl in self._use_exposures(step, path, kind)
            if not is_removed(candidate, self.identity)
        )
        for ref in filter(None, exposed):
            add_layers(bindings, ref, (ContributionLayer.USE,))
        return bindings

    def _exposed_binding(self, target: QualifiedTarget, span: SourceSpan) -> BindingRef | None:
        """The binding a use exposing *target* at *span* contributes, if any.

        A member only its type's own table selects binds as a variant.
        """
        constructor = target.constructor
        if target.ref is not None or constructor is None:
            return target.ref
        return self._variant_binding_ref(constructor, span)

    def _contributed_constructors(
        self, step: ScopePath, path: ScopePath, kind: LookupKind
    ) -> dict[ConstructorRef, Layers]:
        """Return the constructor candidates layers anchored at or above *step* give full *path*.

        *kind* is the position's, as for :meth:`_contributed_bindings`.
        """
        constructors: dict[ConstructorRef, Layers] = {}
        for layer, atom in anchored_layers(self._scope_nodes, step, path):
            for constructor, layers in layer.bare_constructor_contributions.get(atom, {}).items():
                add_layers(constructors, constructor, layers)
        for candidate, _decl in self._use_exposures(step, path, kind):
            if candidate.target.constructor is not None and not is_removed(
                candidate, self.identity
            ):
                add_layers(constructors, candidate.target.constructor, (ContributionLayer.USE,))
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

        A region-scoped import tail's bare-exposed enum type makes its own
        variants bare-matchable too, referenced members included. A
        referenced member's hidden-check path is its own declaration route;
        an inline member's is its path under the enum's own scope, since
        hiding always spells the enum-qualified path.
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

    def retained_paths_beneath(self, path: ScopePath) -> frozenset[ScopePath]:
        """Return, relative to own *path*, the paths of the declarations earlier REPL entries
        retain beneath it: types, their inline members and ordinary members."""
        retained = {
            *self._repl_session_type_paths,
            *(
                (*owner_path, member)
                for owner_path, owner in self._repl_session_type_paths.items()
                if owner.alias is None
                for member in owner.members
            ),
            *self._repl_session_ordinary_member_paths,
        }
        return frozenset(
            declared[len(path) :]
            for declared in retained
            if len(declared) > len(path) and declared[: len(path)] == path
        )

    def paths_reached_at(
        self, scope_path: ScopePath, spelling: NameT | AppliedT, paths: Collection[ScopePath]
    ) -> frozenset[ScopePath]:
        """Return the paths among *paths* ``<spelling>::path``, at *scope_path*, reaches.

        A path is left out when its whole-path type lookup (:mod:`lookup`)
        finds it hidden: a ``hiding`` removed the path and nothing else
        reaches it. Another declaration at that path hides nothing.
        """
        with self._full_view(), self._named_scope(scope_path):
            return frozenset(
                path
                for path in paths
                if not isinstance(
                    self._type_target(member_chain(spelling, path), path[-1], spelling.span),
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
        with self._full_view(), self._named_scope(scope_path):
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
        found = dict(own)
        for candidate, layers in contributed.items():
            add_layers(found, candidate, layers)
        return tuple(self._one_per_declaration(found))

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
        """Decide the constructor candidates a bare value *name* selects.

        Read in the bare value's order (:meth:`_bare_value`): the first step
        whose decision (:meth:`_step_value_constructors`) finds any decides;
        several remaining candidates are ambiguous.
        """
        return next(
            (
                found
                for step in lookup_steps(self._named_scope_path())
                if (found := self._step_value_constructors(step, name))
            ),
            {},
        )

    def _step_value_constructors(self, step: ScopePath, name: str) -> dict[ConstructorRef, Layers]:
        """Decide the constructor candidates bare *name* selects at *step* alone.

        This module's own candidates -- a named scope's own declarations only
        -- else the contributed ones; a root declaration shadows the enum
        members the module root injects under its name.
        """
        candidates = self._own_step_constructors(
            step, name, nested=False
        ) or self._contributed_step_constructors(step, name, LookupKind.VALUE)
        own = [
            candidate for candidate in candidates if candidate.owner_module_id == self._module_id
        ]
        declared = [candidate for candidate in own if not candidate.owner_path]
        return self._one_per_declaration(
            {candidate: candidates[candidate] for candidate in declared or own or candidates}
        )

    def _one_per_declaration(
        self, candidates: Mapping[ConstructorRef, Layers]
    ) -> dict[ConstructorRef, Layers]:
        """*candidates*, one per declaration they construct, with every layer reaching it.

        A renaming alias's constructor is its target's
        (:meth:`TypeOwnerIndex.constructor_identity`), and aliases denoting one
        type construct one (:meth:`TypeOwnerIndex.denotation`): the candidate
        naming the declaration directly stands for it, else its first by path.
        """
        grouped: dict[object, dict[ConstructorRef, Layers]] = {}
        for candidate, layers in candidates.items():
            named = self._type_owners.constructor_identity(candidate)
            denoted = self._type_owners.denotation(named.qname)
            grouped.setdefault(named if denoted is None else denoted, {})[candidate] = layers
        return {
            (
                named if named in reached else min(reached, key=_constructor_candidate_sort_key)
            ): frozenset().union(*reached.values())
            for named, reached in grouped.items()
        }

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
