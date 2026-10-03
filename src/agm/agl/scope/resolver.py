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

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, TypeVar, assert_never, cast

from agm.agl.attributes import CONFIG_ATTRIBUTE, is_param_declaration
from agm.agl.constraints import ConstraintKind, close_constraints
from agm.agl.diagnostics import (
    AglError,
    CycleAlias,
    HiddenMemberError,
    alias_cycle_error,
    not_a_type,
    static_root_message,
    type_name_not_a_value,
    unknown_type,
)
from agm.agl.modules.ids import (
    RESERVED_ID,
    ModuleId,
    render_route_member,
    spell_declaration,
)
from agm.agl.scope.attributes import recognize_attributes
from agm.agl.scope.imports import (
    ImportEnv,
    ItemDeclaration,
    NameAtom,
    PathAtom,
    QName,
    contribution_routes,
    qualifier_members,
)
from agm.agl.scope.lookup import (
    LookupKind,
    Misfit,
    QualifiedTarget,
    is_removed,
    lookup_bare,
    lookup_declared,
    lookup_origins,
    lookup_qualified,
    lookup_steps,
)
from agm.agl.scope.sources import (
    ModuleSources,
    constructor_binding,
    constructor_candidate_sort_key,
    is_root_inline_member,
    scope_path_sort_key,
)
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    BUILTIN_METHOD_RECEIVER_NAMES,
    AglScopeError,
    Ambiguity,
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
    Layers,
    ModuleResolution,
    NoVisibleConstructorError,
    PatternSlot,
    ReceiverOwner,
    ScopeNode,
    ScopePath,
    SlotCandidate,
    SpacedQualifierError,
    TypeArgumentsError,
    TypeOwner,
    TypeSelection,
    UnknownMemberError,
    UnknownQualifierError,
    add_layers,
    builtin_call_kind,
    builtin_type_static_kind,
    duplicate_binder_message,
    is_builtin_type_static_owner,
    is_qualified_function_member,
    undefined_name_message,
)
from agm.agl.scope.symbols import declaration_qname as _key_qname
from agm.agl.scope.symbols import to_bare_atom as _bare_atom
from agm.agl.scope.symbols import to_bare_path as _bare_path
from agm.agl.scope.type_names import (
    member_chain,
    selection_node_id,
)
from agm.agl.scope.type_owners import (
    TypeOwnerIndex,
    injected_members,
    owned_constructors,
    root_type_names,
)
from agm.agl.scope.uses import UseReader
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


# ---------------------------------------------------------------------------
# Built-in names and reserved-name enforcement
# ---------------------------------------------------------------------------


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


def _root_items_as_written(items: tuple[Item, ...]) -> tuple[tuple[Item, ...], tuple[Item, ...]]:
    """The module root's *items* as its source wrote them, and an inline entry's statements.

    The inline wrap (``parser/wrap.py``) moves the root statements of a source
    without a ``program def`` into a synthetic entry after the root
    declarations; they are put back among those, at their own source
    positions.
    """
    statements: tuple[Item, ...] = ()
    declarations: list[Item] = []
    for item in items:
        if isinstance(item, FuncDef) and item.is_synthetic and isinstance(item.body, Block):
            statements = item.body.items
        else:
            declarations.append(item)
    return tuple(sorted((*declarations, *statements), key=_start_offset)), statements


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


def _constraints_related(a: ConstraintKind, b: ConstraintKind) -> bool:
    """True if *a* and *b* are the same kind or one implies the other."""
    return a in close_constraints(frozenset({b})) or b in close_constraints(frozenset({a}))


def _unknown_member(chain: QualifierChain, member: str) -> UnknownMemberError:
    """Return the one verdict for ``chain::member``, as written, selecting no member."""
    return UnknownMemberError(render_qualified_name(chain, member), span=chain.span)


def _unknown_qualifier(chain: QualifierChain) -> UnknownQualifierError:
    """Return the one verdict for *chain*, as written, naming nothing that qualifies."""
    return UnknownQualifierError(render_qualifier_path(chain), span=chain.span)


def _item_order(named: ItemDeclaration) -> PathAtom:
    """Order import items naming declarations by the path they spell."""
    return named.item


_Declaration = (
    FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias | BuiltinVarDecl | LetDecl | VarDecl
)
"""A declaration claiming a path."""


def _written_scope(item: _Declaration) -> ScopePath:
    """The scope path *item* is written at, its enclosing regions' included."""
    return tuple(segment.name for segment in item.scope_path)


def _declaring_items(
    items: Iterable[Item], scope: ScopePath = ()
) -> Iterator[tuple[ScopeRegion | _Declaration, ScopePath]]:
    """Yield every region and declaration among *items*, nested ones too.

    Each with the path it declares beneath: a region's own, a declaration's
    scope path.
    """
    for item in items:
        if isinstance(item, ScopeRegion):
            path = (*scope, item.segment.name)
            yield item, path
            yield from _declaring_items(item.items, path)
        elif isinstance(
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
        ):
            yield item, _written_scope(item)


def _beneath_alias_error(alias_path: ScopePath, span: SourceSpan) -> AglScopeError:
    """The rejection of declaring beneath *alias_path*, which the module declares as an alias."""
    return AglScopeError(
        f"Nothing can be declared beneath '{'::'.join(alias_path)}', "
        "which this module declares as a type alias.",
        span=span,
    )


def reject_declaring_beneath_retained(
    program: Program,
    scope_nodes: Mapping[ScopePath, ScopeNode],
    type_paths: Mapping[ScopePath, TypeOwner],
) -> None:
    """Reject an alias REPL entry *program* declares where an earlier entry declared beneath it.

    Any retained scope path (*scope_nodes*) at or beneath its path completes
    the pair, but for those of an earlier type at its path (*type_paths*):
    that type's own scope, holding only its inline members, and their empty
    scopes.
    """
    for item, path in _declaring_items(_root_items_as_written(program.body.items)[0]):
        if not isinstance(item, TypeAlias):
            continue
        alias_path = (*path, item.name)
        owner = type_paths.get(alias_path)
        inline = frozenset(() if owner is None else owner.members)
        for scope_path, node in scope_nodes.items():
            beneath = scope_path[len(alias_path) :]
            if scope_path[: len(alias_path)] != alias_path:
                continue
            own = owner is not None and (
                not beneath or (len(beneath) == 1 and beneath[0] in inline)
            )
            allowed = frozenset() if beneath else inline
            if (
                not own
                or not allowed.issuperset(node.members)
                or node.bare_contributions
                or node.uses
            ):
                raise _beneath_alias_error(alias_path, item.span)


def _head_arguments(head: TypeExpr | None) -> tuple[TypeExpr, ...]:
    """The type arguments ``def`` head *head* is written applied to, which its type slots lead."""
    if isinstance(head, ArrayT):
        return (head.elem,)
    if isinstance(head, DictT):
        return (head.key, head.value)
    return head.args if isinstance(head, AppliedT) else ()


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------


class _Resolver(ModuleSources):
    """Stateful resolver that builds the scope tree and resolution tables.

    Implements explicit ``isinstance`` dispatch for each node kind; what its
    lookups read by full path is :class:`ModuleSources`', and what its uses
    expose is its :class:`UseReader`'s. Use ``resolve_program`` — the public
    whole-program entry point — rather than instantiating this class directly.

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
        super().__init__()
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
        # Types, whose members are constructor bindings, are excluded.
        self._repl_session_ordinary_member_paths = {
            (*path, name)
            for path, node in self._repl_session_scope_nodes.items()
            for name, ref in node.members.items()
            if ref.kind is not BinderKind.constructor_binding
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
        # Structured method identity -> nominal receiver owner. This is
        # scope's single receiver classification artifact for later passes.
        self._method_declarations: dict[DeclarationKey, ReceiverOwner] = {}
        self._scoped_constructor_candidates: dict[tuple[ScopePath, str], list[ConstructorRef]] = {}
        # Constructor candidates: name -> ordered list of ConstructorRef.
        self._constructor_candidates: dict[str, list[ConstructorRef]] = {}
        # The members this module's enums inject bare: (step, name) -> constructors.
        self._injected_constructors: dict[tuple[ScopePath, str], list[ConstructorRef]] = {}
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
        self._reject_declaring_beneath_aliases(program.body.items)

        # Pre-pass 1: collect every named declaration and scope mention. This
        # establishes path-keyed membership before qualifier validation and the
        # legacy root worker build their compatibility tables.
        self._collect_declarations(program)

        # Pre-pass 2: collect top-level def names for mutual recursion.
        self._collect_func_decls()

        self._scope_nodes = self._build_scope_nodes(self._root_scope)
        self._uses = UseReader(
            self, self._module_id, self._scope_nodes, self._scope_entity_kinds, self._type_owners
        )
        self._at_root = True
        self._resolve_headers(program.body.items)

    def _reject_declaring_beneath_aliases(self, items: tuple[Item, ...]) -> None:
        """Reject declaring beneath a name this module declares as an alias, in either order.

        A region opening an alias's path, or a declaration beneath it, meets
        an alias written here or one an earlier REPL entry retained. Of the
        pairs, the one completed first is reported: at the alias when it is
        the later, else at the segment completing its path.
        """
        declaring = list(_declaring_items(_root_items_as_written(items)[0]))
        aliases = {
            (*path, item.name): item for item, path in declaring if isinstance(item, TypeAlias)
        }
        retained = [
            path
            for path, owner in self._repl_session_type_paths.items()
            if owner.alias is not None and path not in aliases
        ]
        completed: tuple[int, ScopePath, SourceSpan] | None = None
        for item, path in declaring:
            for alias_path in (*aliases, *retained):
                count = len(alias_path)
                if path[:count] != alias_path or (
                    isinstance(item, ScopeRegion) and len(path) != count
                ):
                    continue
                alias = aliases.get(alias_path)
                if alias is not None and alias.span.start_offset > item.span.start_offset:
                    pair = (alias.span.start_offset, alias_path, alias.span)
                else:
                    segment = (
                        item.segment
                        if isinstance(item, ScopeRegion)
                        else item.scope_path[count - 1]
                    )
                    pair = (item.span.start_offset, alias_path, segment.span)
                if completed is None or pair[0] < completed[0]:
                    completed = pair
        if completed is not None:
            raise _beneath_alias_error(completed[1], completed[2])

    def resolve(
        self,
        *,
        ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
    ) -> ModuleResolution:
        """Resolve the prepared program's constructors and bodies; the second phase.

        Runs once every module of the program is constructed, since the
        type-owner index answers from all of their headers. It collects this
        module's constructors, then walks the bodies.

        *ambient_constructor_candidates* carries the other modules'
        constructors that import tails make bare here.
        """
        program = self._program
        root = self._root_scope
        type_owners = self._declared_type_owners()
        self._reject_alias_cycles()
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
        # A tail or ``hiding`` item written through an alias must name a
        # declaration whether or not anything reads it.
        for node_id in self._import_env.decl_hiding:
            self._import_hidden(node_id)
        for exposures in self._import_env.decl_tail_beneath.values():
            for named in sorted(
                {named for items in exposures.values() for named in items}, key=_item_order
            ):
                self._named_by(named)
        if ambient_constructor_candidates:
            for cname, crefs in ambient_constructor_candidates.items():
                for cref in crefs:
                    self._add_constructor_candidate(cname, cref)
        # Pre-pass 3: collect constructor candidates from the module's current
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
        # A static root binds every declaration up front: its walk changes no path read.
        if self._is_static_root_module:
            self._keep_readings()
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
            replaced_uses=self._replaced_uses(),
            declared_segments=self.reader().declared,
            spelled_types=self.spelled_types(),
        )

    # ------------------------------------------------------------------
    # Pre-passes
    # ------------------------------------------------------------------

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
        An inline entry's root statements bind at the root
        (:meth:`_resolve_root_items`): their ``let``/``var`` binders are
        collected like the root's own, in source order.
        """
        for item in _root_items_as_written(program.body.items)[0]:
            self._collect_item_declaration(item, ())

    def _collect_item_declaration(self, item: Item, scope: ScopePath) -> None:
        """Collect *item*, written in the regions opening *scope*."""
        if isinstance(item, ScopeRegion):
            scope = (*scope, item.segment.name)
            self._ensure_scope_path(scope, item.segment.node_id, item.span)
            for child in item.items:
                self._collect_item_declaration(child, scope)
            return
        if isinstance(item, ImportDecl):
            self._import_decl_scope_paths[item.node_id] = scope
            return
        if isinstance(item, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            self._register_declaration(item, self._declaration_scope(item))
            return
        if isinstance(item, BuiltinVarDecl):
            self._register_builtin_var_declaration(item, self._declaration_scope(item))
            return
        if isinstance(item, (LetDecl, VarDecl)):
            # A binder's scope layer is order-independent even though its
            # membership is not: create the path here so a binder with no
            # sibling declaration still gets a scope node.
            path = self._declaration_scope(item)
            name = static_binding_name(item)
            if name == "_":
                return
            self._register_static_binding_declaration(item, path, name)

    def _declaration_scope(
        self,
        item: FuncDef
        | RecordDef
        | EnumDef
        | ExceptionDef
        | TypeAlias
        | BuiltinVarDecl
        | LetDecl
        | VarDecl,
    ) -> ScopePath:
        """Create the scope path *item* is written at and return it."""
        written = _written_scope(item)
        self._ensure_scope_path(written, item.node_id, item.span)
        return written

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

    def _claim_ordinary(self, key: DeclarationKey, span: SourceSpan) -> None:
        """Claim *key* for an ordinary declaration at *span*, rejecting any other there."""
        if self._scope_entity_kinds.get(key) is not None:
            raise DuplicateDeclarationError(key[2], span=span)
        self._scope_entity_kinds[key] = "ordinary"

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
            self._claim_ordinary(key, item.span)

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
        type_scope = path + (item.name,)
        self._type_paths.add(type_scope)
        if isinstance(item, TypeAlias):
            return
        self._scope_paths.add(type_scope)
        self._scope_node_ids.setdefault(type_scope, item.node_id)
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
        self._claim_ordinary(key, item.span)
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
        self._claim_ordinary(key, item.span)
        self._declarations[key] = self._binder_ref(
            item, decl_node_id=item.node_id, name=item.name, scope_path=path
        )
        self._declaration_items[key] = item

    def _classify_function_head(self, declaration: FuncDef, base: ScopePath) -> None:
        """Classify one ``def``'s head, written in scope *base*, and its receiver,
        after preceding lexical contributions are visible."""
        head = declaration.receiver_type
        if not declaration.is_method:
            if head is not None:
                raise TypeArgumentsError(declaration.scope_path[-1].name, None, span=head.span)
            return
        receiver = declaration.params[0]
        path = _written_scope(declaration)
        own = declaration.scope_path[len(base) :]
        self._reject_alias_receiver(path, own, receiver.span)
        owner = (
            self._receiver_head(path, head, _head_arguments(head))
            if head is not None
            else self._receiver_owner(declaration.scope_path, receiver, own)
        )
        if owner is None:
            return
        if receiver.default is not None:
            raise AglScopeError(
                f"Receiver 'self' for method '{declaration.name}' cannot have a default value.",
                span=receiver.span,
            )
        self._method_declarations[(self._module_id, path, declaration.name)] = owner

    @staticmethod
    def _head_chain(written: Sequence[ScopeSegment]) -> QualifierChain | None:
        """Return the qualifier chain several *written* head segments spell; none for fewer."""
        if len(written) < 2:
            return None
        return QualifierChain(
            None,
            tuple(
                QualifierSegment(segment.name, None, segment.span, segment.node_id)
                for segment in written[:-1]
            ),
            written[-1].name,
            span_covering(written[0].span, written[-1].span),
            written[-1].node_id,
        )

    def _reject_alias_receiver(
        self, path: ScopePath, own: Sequence[ScopeSegment], span: SourceSpan
    ) -> None:
        """Reject a method, receiver at *span*, whose path *path* selects a type through an alias.

        Each prefix of *path*, the whole one included, is read as the
        qualifier it is (:func:`lookup_declared`), the ``def``'s *own*
        segments -- those beyond its enclosing regions -- as written.
        """
        start = len(path) - len(own)
        for end in range(1, len(path) + 1):
            found = lookup_declared(
                self,
                path[:end],
                self._head_chain(own[: end - start]),
                LookupKind.TYPE,
                span=span,
            )
            if not isinstance(found, QualifiedTarget) or found.key is None:
                continue
            owner = self._type_owners.owner(_key_qname(found.key))
            if owner is not None and owner.alias is not None:
                raise AglScopeError(
                    f"'self' cannot declare a method in alias scope '{found.key[2]}', "
                    f"which targets '{render_type_expr(owner.alias.type_expr)}'.",
                    span=span,
                )

    def _builtin_receiver(self, name: str, span: SourceSpan) -> ReceiverOwner:
        """Classify a method on built-in type *name*, spelled bare at *span*.

        A generic receiver binds its type parameters only in applied form.
        """
        if name in ("array", "dict"):
            raise AglScopeError(
                "Builtin method receivers must use their bare generic form "
                "(array[T] or dict[K, V]).",
                span=span,
            )
        return ReceiverOwner(self._module_id, (name,), BuiltinMethodReceiver(name))

    def _receiver_head(
        self, owner_path: ScopePath, head: TypeExpr, written: tuple[TypeExpr, ...]
    ) -> ReceiverOwner:
        """Classify method head *head*, applied to type arguments *written*, at *owner_path*.

        Only a builtin receiver in its bare generic form (``array[E]``,
        ``dict[K, V]``) or ``dict[text, V]`` takes them, each of its own a
        distinct one of *written*'s names; any other head declares beneath a
        type application.
        """
        slots = [argument for argument in written if isinstance(argument, NameT)]
        arguments: tuple[TypeExpr, ...]
        if isinstance(head, ArrayT):
            name, arguments = "array", (head.elem,)
        elif isinstance(head, DictT):
            name = "dict"
            arguments = (head.value,) if isinstance(head.key, TextT) else (head.key, head.value)
        else:
            raise TypeArgumentsError(owner_path[-1], None, span=head.span)
        indices = {slot.node_id: index for index, slot in enumerate(slots)}
        parameters = tuple(indices.get(argument.node_id, -1) for argument in arguments)
        if -1 in parameters or len(set(parameters)) != len(parameters):
            raise AglScopeError(
                "Builtin method receivers must use their bare generic form.", span=head.span
            )
        return ReceiverOwner(
            self._module_id, owner_path, BuiltinMethodReceiver(name, parameters, len(slots))
        )

    def _receiver_owner(
        self,
        segments: tuple[ScopeSegment, ...],
        receiver: Param,
        own: tuple[ScopeSegment, ...],
    ) -> ReceiverOwner | None:
        """Return the type method path *segments*' receiver attaches to, if any.

        An empty path has none. The receiver takes the type declared at the
        whole path (:meth:`_receiver_type_owner`); the ``def``'s *own*
        qualifier -- the segments beyond the enclosing regions -- is what a
        miss is reported on. An unannotated receiver's rejection is raised;
        an annotated one only makes the ``def`` an ordinary function.
        """
        owner: ReceiverOwner | AglError | None = None
        if segments:
            owner = self._receiver_type_owner(segments, own, receiver.span)
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

        The whole path is read at its parent step alone
        (:func:`lookup_declared`); a qualifier *written* in the ``def`` itself
        gets the verdict of a miss. A path declaring no type may end in a
        constructor its parent step reads bare, whose owner the bare value
        decision there selects; a one-segment head -- as written, else the
        whole path -- may name a built-in receiver type.
        """
        path = tuple(segment.name for segment in segments)
        found = lookup_declared(self, path, self._head_chain(written), LookupKind.TYPE, span=span)
        if isinstance(found, AglError):
            return found
        owner = None if found is None or found.key is None else self._receiver_key_owner(found.key)
        if owner is not None:
            return owner
        if len(written or segments) == 1 and path[-1] in BUILTIN_METHOD_RECEIVER_NAMES:
            return self._builtin_receiver(path[-1], (written or segments)[-1].span)
        found = lookup_declared(self, path, None, LookupKind.VALUE, span=span)
        constructor = found.constructor if isinstance(found, QualifiedTarget) else None
        if constructor is None:
            return found if isinstance(found, AglError) else None
        return ReceiverOwner(
            constructor.owner_module_id, (*constructor.owner_path, constructor.owner_name)
        )

    def _receiver_key_owner(self, key: DeclarationKey) -> ReceiverOwner | None:
        """Return the receiver owner type *key* names.

        Never an alias: :meth:`_reject_alias_receiver` rejects those first.
        """
        module_id, path, name = key
        if module_id == self._module_id:
            return (
                ReceiverOwner(module_id, (*path, name))
                if self._type_owners.is_declared(_key_qname(key))
                else None
            )
        return self._cross_module_type_owners.get(_key_qname(key))

    def _build_scope_nodes(self, root: ScopeNode) -> dict[ScopePath, ScopeNode]:
        """Build named-scope layers and populate their declaration memberships."""
        nodes: dict[ScopePath, ScopeNode] = {(): root}
        for path in sorted(self._scope_paths, key=scope_path_sort_key):
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
            if type_path in nodes:
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
            _written_scope(node),
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
        """Intern *ref* as its member declaration's canonical metadata.

        A member an alias selects is the alias's constructor with that member,
        one per member, so it is its own.
        """
        if ref.member is not None:
            return ref
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
        routes) joins once: under one name, a declaration contributes one
        constructor.
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
        for step, name, constructor in injected_members(self._module_id, owners):
            self._injected_constructors[(step, name)] = self._place_candidate(
                self._injected_constructors.get((step, name), []),
                self._canonical_constructor_ref(constructor),
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
            scope.define(name, constructor_binding(name, rep))

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
                self._uses.validate(self._scope.scope_path, item)
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

        The statements resolve at their own source positions among the root
        declarations (:func:`_root_items_as_written`), in the root scope, with
        a function body's flags: the text reads -- and reports its first
        error -- exactly as written, and a statement's ``let``/``var`` is a
        root binding ``::name`` reaches.
        """
        written, statements = _root_items_as_written(items)
        statement_ids = {id(statement) for statement in statements}
        for item in written:
            if id(item) not in statement_ids:
                self._resolve_block_items((item,))
                continue
            with self._resolution_flags_ctx(in_loop=False, in_function=True):
                self._resolve_block_items((item,))

    def _resolve_headers(self, items: tuple[Item, ...]) -> None:
        """Record the ``use`` and contribute the region-scoped ``import`` headers of *items*.

        Its regions' headers too. A use is read where it is written whenever
        it is used; an import contributes bindings at once, whose constructors
        are read from them when looked up. Every import here is in its
        sequence's header: placement was checked first.
        """
        for item in items:
            if isinstance(item, UseDecl):
                self._uses.write(self._scope, item)
            elif isinstance(item, ImportDecl) and item.scope_path:
                self._contribute_regional_import_bare(item)
            elif isinstance(item, ScopeRegion):
                with self._named_scope((*self._scope.scope_path, item.segment.name)):
                    self._resolve_headers(item.items)

    # ------------------------------------------------------------------
    # Declaration handlers
    # ------------------------------------------------------------------

    def _contribute_regional_import_bare(self, decl: ImportDecl) -> None:
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

    # -- ``use`` declarations: read where written, whenever used --

    def _replaced_uses(self) -> dict[int, frozenset[int]]:
        """Each retained use this REPL entry replaces, with the uses replacing it."""
        return self._uses.replacements(
            (*self._scope_nodes.values(), *self._layer_chain(self._root_scope))
        )

    def _resolve_scope_region(self, region: ScopeRegion) -> None:
        """Resolve a named region in the member layer of the scope path it opens."""
        with self._named_scope((*self._scope.scope_path, region.segment.name)):
            self._resolve_block_items(region.items)

    def _resolve_funcdef(self, node: FuncDef) -> None:
        """Resolve a ``def`` declaration (body + params).

        At the root, the pre-pass already defined the function binding, so we
        just resolve the body with a fresh param scope. Named-scope members
        use their collected member layer; a nested block holds no ``def``
        (placement is checked first).
        """
        if node.scope_path:
            base = self._scope.scope_path
            with self._named_scope(_written_scope(node)):
                self._classify_function_head(node, base)
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
        path = _written_scope(node) if node.scope_path else self._named_scope_path()
        if isinstance(node, TypeAlias):
            self._validate_alias(path, node)
        else:
            with self._named_scope(path):
                self._validate_type_decl(node)

    def _reject_alias_cycles(self) -> None:
        """Reject the own aliases whose targets lead back to them: they denote no type.

        Reported where it is declared before anything reads it, by this module or another.
        """
        cyclic = [
            CycleAlias(
                (declaration.span.start_offset,),
                "::".join((*path, declaration.name)),
                declaration.span,
            )
            for declaration, path in self._type_declarations
            if isinstance(declaration, TypeAlias)
            and self._type_owners.cyclic((self._module_id, _bare_atom((*path, declaration.name))))
        ]
        if cyclic:
            raise alias_cycle_error(AglScopeError, cyclic)

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
        with self._uses.view(every_use=True), self._named_scope(path):
            found = self._type_name_target(spelling)
        return found.selection if isinstance(found, QualifiedTarget) else None

    def _validate_alias(self, path: ScopePath, alias: TypeAlias) -> None:
        """Validate *alias*, declared at scope *path*, once.

        Its target is checked here, where the alias is declared, whoever reads it.
        """
        if alias.node_id not in self._validated_aliases:
            self._validated_aliases.add(alias.node_id)
            with self._uses.view(every_use=True), self._named_scope(path):
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
        with self._named_scope(_written_scope(node)):
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
            self._bare_assign_target(name, target.span)
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

    def _bare_assign_target(self, name: str, span: SourceSpan) -> BindingRef | None:
        """Return the binding bare target *name* selects: the same spelling's value reading."""
        found = self._bare_value_target(name, span)
        if isinstance(found, AglError):
            raise found
        return None if found is None else self._value_binding(found, span)

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
        return None if isinstance(found, Misfit) else self._value_binding(found, span)

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
        """Resolve a name reference to its binding and, when it names one, its constructor.

        A qualified reference is read by its whole path
        (:meth:`_qualified_lookup`); a bare one by the bare-value decision
        (:meth:`_bare_value_target`).

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
        self._record_bare_value(
            node,
            self._bare_value_target(node.name, node.span),
            is_call_target=is_call_target,
        )

    def _unknown_bare_value(self, name: str, span: SourceSpan) -> AglError:
        """Why bare value *name*, at *span*, selecting nothing is an error."""
        return (
            self._bare_type_misfit(name, span)
            or self._spaced_qualifier_repair(self._spaced_qualifier_around(span), span)
            or AglScopeError(undefined_name_message(name), span=span)
        )

    def _bare_value_target(
        self, name: str, span: SourceSpan, *, layer: ScopeNode | None = None
    ) -> QualifiedTarget | AglError | None:
        """Decide what bare value *name*, written at *span* in *layer*, selects.

        The one bare-value decision behind a value, an assignment target, a
        field access's object and a pattern name's reading outside its
        pattern; *layer* is the current one unless given. A binding of an
        enclosing block or function is nearest; then the whole-path lookup
        (:func:`lookup_bare`) decides, a binding and the constructor it names
        being one target. A built-in's spelling stays reserved: only this
        module's own declarations other than ordinary functions are read
        before the built-in. Constructors alone competing are an ambiguous
        constructor.
        """
        start = self._scope if layer is None else layer
        for lexical in self._lexical_layers(start):
            ref = self._level_value((lexical,), name)
            if ref is not None:
                return QualifiedTarget(None, ref, None)
        reserved = name in _BUILTIN_CALL_NAMES
        found = lookup_bare(
            self,
            name,
            self._named_scope_path(start),
            LookupKind.VALUE,
            span=span,
            contributions=not reserved,
            constructors=partial(self._ambiguous_bare_constructor, name, span),
        )
        if not reserved or isinstance(found, AglError):
            return found
        if found is not None and not self._is_ordinary_function(found.ref):
            return found
        builtin = self._bare_builtin_ref(name)
        return None if builtin is None else QualifiedTarget(None, builtin, None)

    def _is_ordinary_function(self, ref: BindingRef | None) -> bool:
        """Whether *ref* is a ``def`` other than a ``builtin def``."""
        return (
            ref is not None
            and ref.kind is BinderKind.function_binding
            and not self._is_builtin_function_ref(ref)
        )

    def _record_bare_value(
        self, node: VarRef, target: QualifiedTarget | AglError | None, *, is_call_target: bool
    ) -> None:
        """Record what bare *node* selects (*target*), or raise why nothing.

        Its binding, and for a constructor the constructor.
        """
        if not isinstance(target, QualifiedTarget):
            raise target or self._unknown_bare_value(node.name, node.span)
        constructor = target.constructor
        # A bare value always selects a binding, a constructor's included.
        ref = cast(BindingRef, self._value_binding(target, node.span))
        self._reject_builtin_value_ref(node, ref, is_call_target=is_call_target)
        self._resolution[node.node_id] = ref
        if constructor is not None and ref.kind is BinderKind.constructor_binding:
            self._constructor_refs[node.node_id] = constructor

    def _ambiguous_bare_constructor(
        self, name: str, span: SourceSpan, candidates: Mapping[ConstructorRef, Layers]
    ) -> AmbiguousConstructorError:
        """Report bare *name*, written at *span*, selecting several *candidates*.

        One per declaration they construct, repaired by the first.
        """
        distinct = self._one_per_declaration(candidates)
        ordered = sorted(distinct, key=constructor_candidate_sort_key)
        return self._ambiguous_constructor(
            name,
            {candidate: distinct[candidate] for candidate in ordered},
            self._bare_constructor_repair(ordered[0], name, span),
            span,
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

    def _bare_constructor_repair(
        self, candidate: ConstructorRef, name: str, span: SourceSpan
    ) -> str:
        """Spell *candidate*, which bare *name* written at *span* selects among others.

        An imported member is qualified by the owner name a root import tail
        makes bare, renamed as that import exposes it, while that spelling
        selects *candidate* there; else by its shortest route that does
        (:meth:`_routed_spelling`). Anything else is spelled by its
        declaration path.
        """
        origin = candidate.selected_qname
        if candidate.owner_module_id == self._module_id:
            path = (*_bare_path(origin[1])[:-1], name)
            return spell_declaration(self._module_id, path, reader=self.reader())
        unqualified = self._import_env.unqualified
        owner = next(
            (
                atom[0]
                for atom, qnames in unqualified.items()
                if isinstance(atom, tuple)
                and atom[1:] == (name,)
                and origin in qnames
                and len(unqualified.get(atom[0], ())) == 1
            ),
            None,
        )
        if owner is not None and self._spelling_selects((owner,), name, candidate, span):
            return f"{owner}::{name}"
        return self._routed_spelling(candidate, origin, span)

    def _spelling_selects(
        self,
        qualifier: tuple[str, ...],
        member: str,
        candidate: ConstructorRef,
        span: SourceSpan,
        *,
        anchored: bool = False,
    ) -> bool:
        """Whether ``qualifier::member``, written at *span*, selects *candidate*."""
        found = self._qualified_lookup(
            self._probe_chain(qualifier, member, span, anchored=anchored),
            member,
            LookupKind.VALUE,
            span,
        )
        return isinstance(found, QualifiedTarget) and found.constructor == candidate

    def _routed_spelling(self, candidate: ConstructorRef, origin: QName, span: SourceSpan) -> str:
        """Spell *candidate*, imported as *origin*, by its shortest route selecting it at *span*.

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
                ("/".join(route), *written[:-1]), written[-1], candidate, span, anchored=anchored
            )
        )
        return min(spellings, key=len, default=None) or spell_declaration(
            origin[0], _bare_path(origin[1]), reader=self.reader()
        )

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
        is a built-in type's fallback name, else unknown.
        """
        found = self._type_target(None, name, span)
        if found is None and not is_builtin_type_name(name):
            raise unknown_type(name, span)
        self._record_type_selection(node_id, found)

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

        A qualified spelling selecting only a value names no type; a bare one
        selecting nothing yields ``None``.
        """
        if chain is None:
            return self._bare_lookup(name, LookupKind.TYPE, span)
        qualified = self._qualified_lookup(chain, name, LookupKind.TYPE, span)
        if isinstance(qualified, Misfit):
            return not_a_type(render_qualified_name(chain, name), span)
        return qualified

    def _bare_lookup(
        self, name: str, kind: LookupKind, span: SourceSpan
    ) -> QualifiedTarget | AglError | None:
        """Return what bare *name* of *kind* selects in the current named scope (:mod:`lookup`)."""
        return lookup_bare(self, name, self._named_scope_path(), kind, span=span)

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

    def _named_scope_path(self, start: ScopeNode | None = None) -> ScopePath:
        """Return the path of the nearest named scope enclosing *start*, else the current layer."""
        layer: ScopeNode | None = self._scope if start is None else start
        while layer is not None and not layer.scope_path:
            layer = layer.parent
        return () if layer is None else layer.scope_path

    def _lexical_layers(self, start: ScopeNode) -> Iterator[ScopeNode]:
        """Yield *start* and the block and function layers enclosing it, innermost first."""
        layer: ScopeNode | None = start
        while layer is not None and layer is not self._root_scope and not layer.scope_path:
            yield layer
            layer = layer.parent

    def _spaced_qualifier_repair(
        self, advisory: SpacedQualifier | None, failure_span: SourceSpan
    ) -> SpacedQualifierError | None:
        """Explain a reference that whitespace split from its qualifier.

        A run such as ``app/config ::x`` never becomes a qualifier — the lexer
        requires byte adjacency — so it parses as an unrelated expression whose
        parts then fail to resolve on their own.  The lexer recorded the run it
        saw; the repair is offered only when the tight spelling selects the
        intended member where it is written (and, for ``Type::Ctor``, only when
        the member names a constructible type).
        """
        if advisory is None:
            return None
        found = lookup_qualified(
            self,
            self._probe_chain(
                ("/".join(advisory.segments),),
                advisory.member,
                failure_span,
                anchored=advisory.anchored,
            ),
            advisory.member,
            self._named_scope_path(),
            LookupKind.TYPE if advisory.type_qualified else LookupKind.VALUE,
            span=failure_span,
        )
        if not isinstance(found, QualifiedTarget) or (
            advisory.type_qualified
            and (found.key is None or not self._is_constructible_type_ref(_key_qname(found.key)))
        ):
            return None
        return SpacedQualifierError(
            advisory.member_text,
            render_route_member(
                advisory.segments, (advisory.member_text,), anchored=advisory.anchored
            ),
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
        self,
        scope_path: ScopePath,
        spelling: NameT | AppliedT,
        paths: Collection[ScopePath],
        *,
        every_use: bool,
    ) -> frozenset[ScopePath]:
        """Return the paths among *paths* ``<spelling>::path``, at *scope_path*, reaches.

        A path is left out when its whole-path type lookup (:mod:`lookup`)
        finds it hidden: a ``hiding`` removed the path and nothing else
        reaches it. Another declaration at that path hides nothing. Every use
        is read when *every_use*; otherwise those the read in progress sees.
        """
        with self._uses.view(every_use), self._named_scope(scope_path), self._reaching():
            return frozenset(
                path
                for path in paths
                if not isinstance(
                    self._type_target(member_chain(spelling, path), path[-1], spelling.span),
                    HiddenMemberError,
                )
            )

    def scopes_named_at(
        self, scope_path: ScopePath, name: str, *, every_use: bool
    ) -> frozenset[QName]:
        """Return the scope paths *name*, a qualifier at *scope_path*, names.

        Every use is read when *every_use*; otherwise those the read in progress sees.
        """
        chain = QualifierChain(None, (), name, self._program.span, self._program.node_id)
        with self._uses.view(every_use):
            return lookup_origins(self, chain, scope_path)

    def type_name_selection_at(
        self,
        scope_path: ScopePath,
        spelling: NameT | AppliedT | VariantRef,
        *,
        every_use: bool,
    ) -> TypeSelection | None:
        """Return what type name or member reference *spelling*, at *scope_path*, selects now.

        Scope's own type-position decision (:meth:`_type_name_target`), read
        without recording it; ``None`` when the spelling selects no
        declaration there, the decision's own rejections included. Every use
        is read when *every_use*; otherwise those the read in progress sees.
        """
        with self._uses.view(every_use), self._named_scope(scope_path):
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
        """Resolve a field-access expression by resolving its object as a value.

        A bare object reading a type's constructor, or nothing, where it
        names a type is that type misused as a value.
        """
        obj = expr.obj
        if not isinstance(obj, VarRef) or obj.qualifier is not None:
            self._resolve_expr(obj)
            return
        found = self._bare_value_target(obj.name, obj.span)
        if (
            found is None or (isinstance(found, QualifiedTarget) and found.constructor is not None)
        ) and isinstance(self._bare_lookup(obj.name, LookupKind.TYPE, obj.span), QualifiedTarget):
            raise AglScopeError(
                f"'{obj.name}' is a type name, not a value; use '::' for "
                f"constructor qualification (for example, '{obj.name}::{expr.field}').",
                span=obj.span,
            )
        self._record_bare_value(obj, found, is_call_target=False)

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
                        else self._routed_spelling(constructor, origin, chain.span),
                    )
        return injected

    def _pattern_constructors(self, name: str) -> tuple[ConstructorRef, ...]:
        """Return the constructor candidates a bare pattern or ``is`` spelling *name* reaches.

        Its scrutinee selects among them: every constructor so spelled at
        every step -- own, contributed and injected -- that no ``hiding`` removed.
        """
        found: dict[ConstructorRef, Layers] = {}
        for step in lookup_steps(self._named_scope_path()):
            path = (*step, name)
            reading = (
                self.own_at(path, LookupKind.CONSTRUCTOR)
                + self.contributed_at(step, path, LookupKind.CONSTRUCTOR)
                + self.injected_at(step, name)
            )
            for candidate in reading.candidates:
                constructor = candidate.target.constructor
                if constructor is not None and not is_removed(candidate, self):
                    add_layers(found, constructor, (candidate.layer,))
        return tuple(self._one_per_declaration(found))

    def _one_per_declaration(
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
            denoted = self._type_owners.denotation(named.selected_qname)
            grouped.setdefault(named if denoted is None else denoted, {})[candidate] = layers
        return {
            (
                named if named in reached else min(reached, key=constructor_candidate_sort_key)
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
            outside = (
                None
                if scope.parent is None
                else self._bare_value_target(name, span, layer=scope.parent)
            )
            self._pattern_slots[slot_id] = PatternSlot(
                slot_id=slot_id,
                name=name,
                candidates=(),
                alternative=(
                    self._value_binding(outside, span)
                    if isinstance(outside, QualifiedTarget)
                    else None
                ),
                match_site_node_id=match_site_node_id,
                alternative_constructor=(
                    outside.constructor if isinstance(outside, QualifiedTarget) else None
                ),
                outside_ambiguity=(
                    Ambiguity.of(outside)
                    if isinstance(outside, AmbiguousQualificationError)
                    else None
                ),
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
