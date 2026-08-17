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
   *qualified* target is settled here: only ``builtin var`` is assignable
   across a module boundary, and no qualified name is ever a pattern slot.
   An *indexed* target (``target[index] := value``) has no binding of its
   own: ``obj`` and ``index`` are resolved as ordinary expressions, and no
   ``resolution`` entry is recorded for the ``AssignStmt``.
3. Reading (``VarRef``) a name not visible in the current scope chain → error.
   ``_`` is always a discard wildcard and never resolves as a readable name.
4. Pattern variables and catch binders are immutable and branch-local.
5. ``loop`` body bindings are visible to the ``until`` condition but not after.
6. ``param`` declarations are valid at the module root and in named scope
   regions in every module; ``program def`` follows the same placement rule.
7. ``def`` declarations are valid at the module root and in named scope
   regions; a pre-pass collects them by path so root and same-scope members
   support mutual recursion.

Built-in call classification
-----------------------------
Contextual built-ins are reached through ordinary name resolution. In call
position, the resolver records ``Call.node_id → BuiltinKind`` only when the
resolved function declaration has ``builtin`` provenance. A reference to such
a declaration outside call position raises ``AglScopeError`` because host
built-ins are not first-class values in AgL.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, TypeVar, cast

from agm.agl.diagnostics import static_root_message
from agm.agl.modules.ids import STD_CONFIG_ID, STD_CORE_ID, ModuleId, spell_declaration
from agm.agl.scope.imports import (
    BareRoute,
    NameAtom,
    QName,
    QualResolution,
    QualResolutionAmbiguous,
    QualResolutionFound,
    qualification_repair_guidance,
    qualifier_candidates,
    qualifier_contributes,
    qualifier_members,
    qualifier_scope_paths,
    render_qualifier,
    resolve_alias_target,
    resolve_qualified,
    resolve_qualified_member,
    sibling_qname,
    try_resolve_qualified_member,
)
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    AglScopeError,
    BinderKind,
    BindingRef,
    BuiltinKind,
    ConstructorRef,
    DeclarationKey,
    LocalUseContribution,
    ModuleResolution,
    PatternSlot,
    ResolvedUseTarget,
    ScopeNode,
    ScopePath,
    SlotCandidate,
    alias_denotes_constructible_type,
    duplicate_binder_message,
    immutable_binder_phrase,
)
from agm.agl.scope.symbols import import_item_path as _item_path
from agm.agl.scope.symbols import to_bare_atom as _bare_atom
from agm.agl.scope.symbols import to_bare_path as _bare_path
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS
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
    ParamDecl,
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
    VarPattern,
    VarRef,
    WildcardPattern,
    declares_source_entry,
    pattern_binder_candidates,
    simple_let_pattern_name,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TYPE_PARAMETER_WILDCARD, AppliedT, NameT, render_type_expr
from agm.agl.syntax.visitor import walk

_T = TypeVar("_T")


def _relative_under(atom: NameAtom, target: ScopePath) -> ScopePath | None:
    """Return *atom*'s path relative to *target*, or ``None`` if it is not under it.

    The one definition of the prefix test every use-target expansion performs
    when it re-spells a module or scope surface relative to the target named
    by a ``use``.
    """
    path = _bare_path(atom)
    return path[len(target) :] if path[: len(target)] == target else None


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
class _PatternResolutionPolicy:
    """Name-resolution policy for one pattern-owning syntax site."""

    root_bare_binds: bool
    binder_kind: BinderKind


_CASE_PATTERN_POLICY = _PatternResolutionPolicy(
    root_bare_binds=False, binder_kind=BinderKind.pattern_binding
)
_LET_PATTERN_POLICY = _PatternResolutionPolicy(
    root_bare_binds=True, binder_kind=BinderKind.let_binding
)


@dataclass(frozen=True, slots=True)
class _LocalScopeRoute:
    """Identity of a local scope exposed through ``use``."""

    path: ScopePath


@dataclass(frozen=True, slots=True)
class _ImportedUseContribution:
    """One imported surface exposed by a resolved ``use`` in a lexical region."""

    members: Mapping[NameAtom, QName]
    scope_routes: Mapping[NameAtom, frozenset[BareRoute]]


# Built-in call names: recognised in call position, not bindable as values.
# Sourced from ``symbols.BUILTIN_CALL_NAMES`` (the single source of truth).
_BUILTIN_CALL_NAMES = BUILTIN_CALL_NAMES

# Sentinel ``owner_decl_node_id`` for built-in constructor candidates (exceptions
# and prelude types), which have no source-level declaration node.  User
# declarations may shadow bindings carrying this sentinel.
_BUILTIN_CONSTRUCTOR_NODE_ID = -1

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


def _supersedes(candidate: ConstructorRef, cref: ConstructorRef) -> bool:
    """Whether *cref* is a later declaration of *candidate*'s own name path.

    An ambient candidate seeded from an earlier REPL entry names the very
    path this entry's own declaration now owns, so a fresh constructor
    reference must reach the newest declaration rather than the one the
    session happened to seed first.  Only a full name path (module, scope
    path, and owner name) identifies the same owner; two distinct owners
    never share one, so genuine constructor overloading — two types' same
    named variant, or one name declared in different modules — is untouched.
    A built-in seeded without a declaration node carries no identity to
    supersede or be superseded by, and the same declaration contributed
    twice (over two overlapping import routes) is not a later one.
    """
    return (
        candidate.owner_decl_node_id != cref.owner_decl_node_id
        and _BUILTIN_CONSTRUCTOR_NODE_ID
        not in (candidate.owner_decl_node_id, cref.owner_decl_node_id)
        and (candidate.owner_module_id, candidate.owner_path, candidate.owner_name)
        == (cref.owner_module_id, cref.owner_path, cref.owner_name)
    )


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------


class _Resolver:
    """Stateful resolver that builds the scope tree and resolution tables.

    Implements explicit ``isinstance`` dispatch for each node kind.
    Use ``resolve_program`` — the public whole-program entry point — rather
    than instantiating this class directly.
    """

    def __init__(
        self,
        module_id: ModuleId,
        import_env: ImportEnv,
        all_public_types: dict[
            tuple[ModuleId, NameAtom], RecordDef | EnumDef | ExceptionDef | TypeAlias
        ],
        decl_info: dict[tuple[ModuleId, NameAtom], tuple[int, SourceSpan, BinderKind, bool]]
        | None = None,
        cross_module_constructor_refs: Mapping[tuple[ModuleId, NameAtom], ConstructorRef]
        | None = None,
        cross_module_constructible_types: frozenset[tuple[ModuleId, NameAtom]] = frozenset(),
        cross_module_type_scopes: frozenset[tuple[ModuleId, NameAtom]] = frozenset(),
        allow_root_statements: bool = False,
        repl_session_scope: ScopeNode | None = None,
        repl_session_scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
        repl_session_type_paths: Mapping[ScopePath, str | None] | None = None,
        retained_use_targets: Mapping[int, ResolvedUseTarget] | None = None,
        origin_path: Path | None = None,
        spaced_qualifiers: tuple[SpacedQualifier, ...] = (),
    ) -> None:
        # Program parameters. One _Resolver is built per module of a whole
        # program (see scope/program.py::resolve_program); the import
        # environment and whole-program public-type table are always real,
        # never absent, since a program always builds them (an empty
        # ImportEnv/dict for a module with no imports or no public types).
        self._module_id: ModuleId = module_id
        self._import_env: ImportEnv = import_env
        # Maps (module_id, name) → (node_id, span, kind, is_builtin) for
        # cross-module refs.
        self._decl_info: dict[
            tuple[ModuleId, NameAtom], tuple[int, SourceSpan, BinderKind, bool]
        ] = decl_info if decl_info is not None else {}
        self._cross_module_constructor_refs: Mapping[tuple[ModuleId, NameAtom], ConstructorRef] = (
            cross_module_constructor_refs if cross_module_constructor_refs is not None else {}
        )
        self._cross_module_constructible_types = cross_module_constructible_types
        # Public type declarations establish scope paths even when they have
        # no separately public child members.
        self._cross_module_type_scopes = cross_module_type_scopes
        # Whole-program public-type table, used to follow a type alias's
        # target across an import when deciding whether the alias has a
        # variant-less constructor.
        self._all_public_types: dict[
            tuple[ModuleId, NameAtom], RecordDef | EnumDef | ExceptionDef | TypeAlias
        ] = all_public_types
        # The REPL is an incremental host and intentionally retains root
        # statements. File and inline exec entries use static roots.
        self._allow_root_statements = allow_root_statements
        # Optional REPL session scope for ``::name`` self-ref fallback.
        # When set, ``_lookup_own_root`` falls back to this scope for names not
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
        # Each retained type-owned path maps to its rendered alias target, or
        # None for a nominal type: receiver classification cannot tell the two
        # apart from a path alone.
        self._repl_session_type_paths = dict(repl_session_type_paths or {})
        self._retained_use_targets = retained_use_targets or {}
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
        self._use_targets: dict[int, ResolvedUseTarget] = {}
        self._imported_use_contributions: dict[int, list[_ImportedUseContribution]] = {}
        # Scope stack — top is the current scope.
        self._scope: ScopeNode | None = None
        # The module's root ScopeNode (set in run()); used by _lookup_own_root
        # to bypass lexical shadows introduced by nested scopes for ::name.
        self._root_scope: ScopeNode | None = None
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
            DeclarationKey, FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias
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
        # Scoped externs are implemented by their member names in one module
        # companion, so distinct scope paths cannot share a Python symbol.
        self._scoped_extern_symbols: dict[str, FuncDef] = {}
        self._scope_paths: set[ScopePath] = {(), *self._repl_session_scope_nodes}
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
        # Structured method identity -> nominal receiver owner path. This is
        # scope's single receiver classification artifact for later passes.
        self._method_declarations: dict[DeclarationKey, ScopePath] = {}
        self._scoped_constructor_candidates: dict[tuple[ScopePath, str], list[ConstructorRef]] = {}
        # Constructor candidates: name -> ordered list of ConstructorRef.
        self._constructor_candidates: dict[str, list[ConstructorRef]] = {}
        # Nominal aliases whose chain provably ends at an enum: they occupy
        # their name (so a bare reference resolves instead of raising "not
        # defined") but have no constructor candidate, since an enum's variants
        # are its constructors, not the enum type itself.
        self._nonconstructible_type_decls: dict[str, TypeAlias] = {}
        # Resolved single-candidate constructor refs: VarRef.node_id -> ConstructorRef.
        self._constructor_refs: dict[int, ConstructorRef] = {}
        # Scope records constructor candidates for bare pattern names. The
        # checker classifies them after constructor fields have been mapped;
        # candidates do not depend on ordinary lexical value bindings.
        self._pattern_constructor_candidates: dict[int, tuple[ConstructorRef, ...]] = {}
        self._pattern_constructor_spellings: dict[int, str] = {}
        # Bare ``is`` spellings remain candidate sets until typecheck knows the
        # nominal type of the left operand.
        self._is_test_constructor_candidates: dict[int, tuple[ConstructorRef, ...]] = {}
        # Each pattern-owning case branch or let declaration creates one shared
        # slot per binding name. The checker selects its final target after the
        # match site has been classified.
        self._pattern_slots: dict[int, PatternSlot] = {}
        self._match_site_pattern_slots_by_node: dict[int, tuple[int, ...]] = {}
        self._active_match_site_pattern_slots: dict[str, int] | None = None
        self._active_match_site_node_id: int | None = None
        self._active_match_site_binder_kind: BinderKind | None = None
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
        self._declares_program_entry: bool = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        program: Program,
        *,
        parent_scope: ScopeNode | None = None,
        ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
        ambient_type_names: frozenset[str] = frozenset(),
        use_targets_only: bool = False,
    ) -> ModuleResolution:
        """Execute the resolution pass over *program*.

        When *parent_scope* is given, the entry's root scope is parented to it
        so name lookups fall through to session bindings (incremental REPL
        sessions).  New declarations live in the entry's own root scope and
        shadow parent bindings without a duplicate-declaration error.

        *ambient_constructor_candidates* carries constructor candidates from
        prior REPL entries so that constructor references to types declared in
        earlier entries resolve correctly in subsequent entries.

        *ambient_type_names* carries type names from prior entries so that
        qualified constructor access (``Owner::variant``) resolves for types
        declared in earlier REPL entries.
        """
        # Seed ambient constructor candidates (from prior REPL entries) before
        # running the local pre-passes so local declarations can shadow them.
        if ambient_constructor_candidates:
            for cname, crefs in ambient_constructor_candidates.items():
                for cref in crefs:
                    # A candidate from another module was selected by an import
                    # and is exposed here under *cname* alone; its owner path
                    # describes its declaring module, not a scope of this one.
                    # Only a retained same-module member belongs to a named
                    # scope here and must stay out of the bare table.
                    if cref.owner_path and cref.owner_module_id == self._module_id:
                        owner_scope = cref.owner_path + (
                            (cref.owner_name,) if cref.variant is not None else ()
                        )
                        self._add_constructor_candidate(
                            cname, cref, scope_path=owner_scope, inject_bare=False
                        )
                    else:
                        self._add_constructor_candidate(cname, cref)
        # Seed ambient type names (from prior REPL entries).
        if ambient_type_names:
            self._declared_type_names.update(ambient_type_names)

        self._declares_program_entry = declares_source_entry(program.body.items)

        # Pre-pass 1: collect every named declaration and scope mention. This
        # establishes path-keyed membership before qualifier validation and the
        # legacy root worker build their compatibility tables.
        self._collect_declarations(program)
        self._classify_method_declarations()
        self._validate_function_names()
        self._validate_non_method_type_params()
        self._validate_extern_backing()

        # Pre-pass 2: collect top-level def names for mutual recursion.
        self._collect_func_decls(program)
        # Pre-pass 3: collect type-declaration names and validate type_params.
        self._collect_type_decl_names(program)
        # Pre-pass 4: collect constructor candidates from RecordDef/EnumDef.
        self._collect_constructor_candidates(program)

        root = ScopeNode(node_id=program.node_id, parent=parent_scope, scope_path=())
        self._push_scope(root)
        self._root_scope = root
        self._scope_nodes = self._build_scope_nodes(root)
        self._at_root = True

        # Define root functions as value bindings; scoped members are already
        # present in their named-scope layers.
        self._define_function_bindings()
        # Define constructor bindings in root scope.
        self._define_constructor_bindings()

        # Main walk: resolve either the complete module or just enough header
        # context for an incremental host to classify use replacement keys.
        if use_targets_only:
            self._resolve_use_context(program.body.items)
        else:
            self._resolve_block_items(program.body.items)
            self._validate_local_use_contributions()

        self._at_root = False
        self._pop_scope()

        return ModuleResolution(
            program=program,
            resolution=self._resolution,
            builtin_calls=self._builtin_calls,
            root_scope=root,
            declarations=dict(self._declarations),
            scope_nodes=dict(self._scope_nodes),
            declared_functions=dict(self._declared_functions),
            allows_root_statements=self._allow_root_statements,
            origin_path=self._origin_path,
            declared_type_names=frozenset(self._declared_type_names),
            declared_type_paths=frozenset(self._type_paths),
            constructor_candidates={
                name: tuple(refs) for name, refs in self._constructor_candidates.items()
            },
            constructor_candidates_by_path={
                key: tuple(refs) for key, refs in self._scoped_constructor_candidates.items()
            },
            constructor_refs=dict(self._constructor_refs),
            pattern_constructor_candidates=dict(self._pattern_constructor_candidates),
            pattern_constructor_spellings=dict(self._pattern_constructor_spellings),
            is_test_constructor_candidates=dict(self._is_test_constructor_candidates),
            pattern_slots=dict(self._pattern_slots),
            match_site_pattern_slots=dict(self._match_site_pattern_slots_by_node),
            method_declarations=dict(self._method_declarations),
            use_targets=dict(self._use_targets),
        )

    # ------------------------------------------------------------------
    # Pre-passes
    # ------------------------------------------------------------------

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
        if isinstance(item, (LetDecl, VarDecl)):
            path = tuple(segment.name for segment in item.scope_path) or enclosing_path
            if path:
                # A binder's scope layer is order-independent even though its
                # membership is not: create the path here so a binder with no
                # sibling declaration still gets a scope node, but leave the
                # member itself to be registered during the body walk, which is
                # what makes textual precedence fall out of the mechanism.
                self._ensure_scope_path(path, item.node_id, item.span)
                name = (
                    item.name
                    if isinstance(item, VarDecl)
                    else simple_let_pattern_name(item.pattern)
                )
                if name is not None and name != "_":
                    self._ordered_binding_paths.add((*path, name))

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
        )
        self._declaration_items[key] = item
        if isinstance(item, FuncDef):
            if item.is_extern and path:
                prior = self._scoped_extern_symbols.get(item.name)
                if prior is not None:
                    raise AglScopeError(
                        f"Scoped extern declarations map to companion symbol '{item.name}'; "
                        "declare it in only one scope.",
                        span=item.span,
                    )
                self._scoped_extern_symbols[item.name] = item
            self._validate_type_params(item)
        else:
            self._validate_type_params(item)
        if not is_type:
            return

        type_item = cast(RecordDef | EnumDef | ExceptionDef | TypeAlias, item)
        self._type_declarations.append((type_item, path))
        self._type_declarations_by_path.setdefault(path, []).append(type_item)
        type_scope = path + (item.name,)
        self._scope_paths.add(type_scope)
        self._scope_node_ids.setdefault(type_scope, item.node_id)
        self._type_paths.add(type_scope)
        if isinstance(item, EnumDef):
            for variant in item.variants:
                variant_key = (self._module_id, type_scope, variant.name)
                if self._scope_entity_kinds.get(variant_key) is not None:
                    raise AglScopeError(
                        f"Name '{variant.name}' is already declared in this scope.",
                        span=variant.span,
                    )
                self._scope_entity_kinds[variant_key] = "ordinary"
                self._declarations[variant_key] = BindingRef(
                    name=variant.name,
                    mutable=False,
                    decl_span=variant.span,
                    decl_node_id=variant.node_id,
                    kind=BinderKind.constructor_binding,
                    module_id=self._module_id,
                    scope_path=type_scope,
                )

    def _classify_method_declarations(self) -> None:
        """Classify receiver parameters after every declaration path is complete.

        ``alias_targets`` maps each alias-owned type path to its target: a
        rendered string, retained from a prior REPL entry, or this entry's own
        ``TypeAlias`` declaration, rendered lazily only if a method is actually
        rejected on that path. A path this entry redeclares as a nominal type
        (record/enum/exception) is popped so a stale retained alias target
        cannot reject a method on that redeclaration; a path this entry
        redeclares as an alias always overrides whatever the retained state
        held, in either direction.
        """
        alias_targets: dict[ScopePath, str | TypeAlias] = {
            path: target
            for path, target in self._repl_session_type_paths.items()
            if target is not None
        }
        for type_decl, path in self._type_declarations:
            type_scope = path + (type_decl.name,)
            if isinstance(type_decl, TypeAlias):
                alias_targets[type_scope] = type_decl
            else:
                alias_targets.pop(type_scope, None)
        for key, declaration in self._declaration_items.items():
            if not isinstance(declaration, FuncDef) or not declaration.params:
                continue
            receiver = declaration.params[0]
            if receiver.name != "self":
                continue
            owner_path = key[1]
            alias_target = alias_targets.get(owner_path)
            if alias_target is not None:
                target_text = (
                    alias_target
                    if isinstance(alias_target, str)
                    else render_type_expr(alias_target.type_expr)
                )
                raise AglScopeError(
                    f"'self' cannot declare a method in alias scope '{owner_path[-1]}', "
                    f"which targets '{target_text}'.",
                    span=receiver.span,
                )
            if owner_path in self._type_paths:
                if receiver.default is not None:
                    raise AglScopeError(
                        f"Receiver 'self' for method '{declaration.name}' "
                        "cannot have a default value.",
                        span=receiver.span,
                    )
                self._method_declarations[key] = owner_path
            elif receiver.type_expr is None:
                if self._is_orphan_receiver(owner_path):
                    raise AglScopeError(
                        f"'self' cannot declare a method on '{owner_path[-1]}', which is "
                        "declared in another module; methods must be declared in the module "
                        "that declares their receiver type.",
                        span=receiver.span,
                    )
                raise AglScopeError("'self' requires an enclosing type scope.", span=receiver.span)

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

    def _is_orphan_receiver(self, owner_path: ScopePath) -> bool:
        """Return whether *owner_path* names a type imported from another module.

        Consults the import environment's bare contributions — the same
        names that make a tailed-imported type name callable
        without qualification — together with the whole-program type table,
        so a receiver on a type this module can see but does not declare
        reports the "no orphans" rule instead of the generic "no enclosing
        type scope" diagnostic. *owner_path*'s last segment is the receiver's
        own bare name; every segment before it is the region the receiver is
        declared in, whose reach is exactly a scoped import declared there or
        in one of its ancestors -- the same reach an ordinary bare reference
        there already gets from ``resolve_bare_contribution_layer`` -- unioned with
        the module-wide root table.
        """
        if not owner_path:
            return False
        name = owner_path[-1]
        candidates: set[QName] = set(self._import_env.unqualified.get(name, frozenset()))
        for _node_id, bare in self._reachable_decl_contributions(
            self._import_env.decl_bare, owner_path[:-1]
        ):
            candidates.update(bare.get(name, frozenset()))
        return any(
            module != self._module_id and (module, source) in self._cross_module_type_scopes
            for module, source in candidates
        )

    def _build_scope_nodes(self, root: ScopeNode) -> dict[ScopePath, ScopeNode]:
        """Build named-scope layers and populate their declaration memberships."""
        nodes: dict[ScopePath, ScopeNode] = {(): root}
        for path in sorted(self._scope_paths, key=_scope_path_sort_key):
            if not path:
                continue
            retained = self._repl_session_scope_nodes.get(path)
            node = ScopeNode(
                node_id=self._scope_node_ids[path],
                parent=nodes[path[:-1]],
                scope_path=path,
            )
            if retained is not None:
                # A retained member is registered like any freshly declared
                # one, so a replayed REPL scope layer is indistinguishable
                # from a layer this program declared itself.
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
        self,
        declaration_type: type[FuncDef]
        | type[RecordDef]
        | type[EnumDef]
        | type[ExceptionDef]
        | type[TypeAlias],
    ) -> Iterator[FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias]:
        """Yield collected declarations of *declaration_type* at the root path."""
        for (module_id, path, _name), item in self._declaration_items.items():
            if module_id == self._module_id and not path and isinstance(item, declaration_type):
                yield item

    def _collect_func_decls(self, program: Program) -> None:
        """Populate the legacy function table from root-path declarations only."""
        for item in self._root_declaration_items(FuncDef):
            assert isinstance(item, FuncDef)
            self._declared_functions[item.name] = item

    def _validate_function_names(self) -> None:
        """Reject a built-in call name reused as an ordinary function name.

        Runs after :meth:`_classify_method_declarations` so that a method
        (its key is in ``self._method_declarations``) is exempt: methods live
        in their receiver type's own member namespace, distinct from the
        namespace this rule protects. Iterates ``self._declaration_items`` in
        declaration order for deterministic diagnostics.
        """
        for key, declaration in self._declaration_items.items():
            if not isinstance(declaration, FuncDef) or key in self._method_declarations:
                continue
            self._validate_function_decl(declaration)

    def _validate_function_decl(self, decl: FuncDef) -> None:
        """Apply function declaration validation independently of its scope path."""
        if decl.name in _RESERVED_NAMES and not decl.is_builtin:
            raise AglScopeError(
                f"'{decl.name}' is a built-in name and cannot be used as a function name.",
                span=decl.span,
            )

    def _validate_non_method_type_params(self) -> None:
        """Reject a '_' type-parameter slot on a function that is not a method.

        Runs after :meth:`_classify_method_declarations` so that a method's
        key (present in ``self._method_declarations``) is exempt: a method's
        receiver-prefix '_' slots are validated later, against the receiver's
        arity, during type checking. Every other function declaration —
        including a plain ``def`` inside a type scope that lacks a ``self``
        receiver — may not use '_' at all.
        """
        for key, declaration in self._declaration_items.items():
            if not isinstance(declaration, FuncDef) or key in self._method_declarations:
                continue
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

    def _collect_type_decl_names(self, program: Program) -> None:
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

        The ``owner_decl_node_id`` is set to -1 (a sentinel) because these types have
        no AST declaration node.
        """
        exception_names: frozenset[str] = frozenset(BUILTIN_EXCEPTIONS)

        for exc_name in BUILTIN_EXCEPTIONS:
            cref = ConstructorRef(
                owner_name=exc_name,
                variant=None,
                owner_decl_node_id=_BUILTIN_CONSTRUCTOR_NODE_ID,
                type_params=(),
                owner_module_id=STD_CORE_ID,
            )
            self._add_constructor_candidate(exc_name, cref)

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
                for variant_name, _vfields in typedef.variants:
                    if variant_name not in exception_names:
                        cref = ConstructorRef(
                            owner_name=type_name,
                            variant=variant_name,
                            owner_decl_node_id=_BUILTIN_CONSTRUCTOR_NODE_ID,
                            type_params=(),
                            owner_module_id=STD_CORE_ID,
                            can_match_bare_pattern=not _vfields,
                        )
                        self._add_constructor_candidate(variant_name, cref)
            else:
                cref = ConstructorRef(
                    owner_name=type_name,
                    variant=None,
                    owner_decl_node_id=_BUILTIN_CONSTRUCTOR_NODE_ID,
                    type_params=(),
                    owner_module_id=STD_CORE_ID,
                )
                self._add_constructor_candidate(type_name, cref)

    def _add_constructor_candidate(
        self,
        ctor_key: str,
        cref: ConstructorRef,
        *,
        scope_path: ScopePath = (),
        inject_bare: bool = True,
    ) -> None:
        """Add *cref* to the candidates list for *ctor_key*, superseding a stale owner.

        A candidate that *cref* supersedes (see :func:`_supersedes`) is
        replaced in place; otherwise a candidate already claiming this
        spelling keeps it and *cref* is dropped.  The spelling is claimed by
        another candidate with the same owner name in the same module
        (a duplicate type declaration — the type-builder pass raises a clear
        "already declared" error for that; we must not conflate it with
        genuine constructor overloading across distinct types), by the same
        owner at the same path in the scoped table, and, in the bare table,
        by a host-seeded built-in of the same name.
        """
        scoped_key = (scope_path, ctor_key)
        self._scoped_constructor_candidates[scoped_key] = self._place_candidate(
            self._scoped_constructor_candidates.get(scoped_key, []),
            cref,
            claims_spelling=lambda candidate: (
                (
                    candidate.owner_module_id,
                    candidate.owner_path,
                    candidate.owner_name,
                )
                == (cref.owner_module_id, cref.owner_path, cref.owner_name)
            ),
        )
        if not inject_bare:
            return
        self._constructor_candidates[ctor_key] = self._place_candidate(
            self._constructor_candidates.get(ctor_key, []),
            cref,
            claims_spelling=lambda candidate: (
                (candidate.owner_module_id, candidate.owner_name)
                == (cref.owner_module_id, cref.owner_name)
                or (
                    (
                        candidate.owner_decl_node_id == _BUILTIN_CONSTRUCTOR_NODE_ID
                        or cref.owner_decl_node_id == _BUILTIN_CONSTRUCTOR_NODE_ID
                    )
                    and candidate.owner_name == cref.owner_name
                )
            ),
        )

    @staticmethod
    def _place_candidate(
        existing: list[ConstructorRef],
        cref: ConstructorRef,
        *,
        claims_spelling: Callable[[ConstructorRef], bool],
    ) -> list[ConstructorRef]:
        """Return *existing* with *cref* superseding, yielding to, or joining it.

        Replacement happens in place so candidate order — which decides the
        representative binding and the reported one on an ambiguity — does
        not depend on when a declaration was superseded.
        """
        for index, candidate in enumerate(existing):
            if _supersedes(candidate, cref):
                return [*existing[:index], cref, *existing[index + 1 :]]
        if any(claims_spelling(candidate) for candidate in existing):
            return existing
        return [*existing, cref]

    def _alias_target_lookup(
        self, name: str, qualifier: QualifierChain | None, *, scope_path: ScopePath = ()
    ) -> RecordDef | EnumDef | ExceptionDef | TypeAlias | None:
        """Resolve one alias-target type reference for :func:`alias_denotes_constructible_type`.

        Tries this module's own declarations first — the alias's own scope path,
        then the module root (only for an unqualified name, since a qualified
        reference can never name a local declaration) — then falls back to the
        shared import-environment resolution
        (:func:`~agm.agl.scope.imports.resolve_alias_target`), which covers a
        target reached through a bare tail contribution or a module qualifier.
        """
        if qualifier is None or not qualifier.segments:
            local_paths: tuple[ScopePath, ...] = (scope_path, ()) if scope_path else ((),)
            for path in local_paths:
                local = self._declaration_items.get((self._module_id, path, name))
                if isinstance(local, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                    return local
        return resolve_alias_target(
            name,
            qualifier,
            self_module_id=None,
            import_env=self._import_env,
            all_public_types=self._all_public_types,
            scope_path=scope_path,
        )

    def _collect_constructor_candidates(self, program: Program) -> None:
        """Build constructor candidates for every collected declaration path."""
        self._seed_builtin_constructor_candidates()
        for item, path in self._type_declarations:
            if isinstance(item, RecordDef):
                cref = ConstructorRef(
                    owner_name=item.name,
                    variant=None,
                    owner_decl_node_id=item.node_id,
                    type_params=item.type_params,
                    owner_module_id=self._module_id,
                    owner_path=path,
                )
                self._add_constructor_candidate(
                    item.name, cref, scope_path=path, inject_bare=not path
                )
            elif isinstance(item, EnumDef):
                type_scope = path + (item.name,)
                for variant in item.variants:
                    cref = ConstructorRef(
                        owner_name=item.name,
                        variant=variant.name,
                        owner_decl_node_id=item.node_id,
                        type_params=item.type_params,
                        owner_module_id=self._module_id,
                        can_match_bare_pattern=not variant.fields,
                        owner_path=path,
                    )
                    self._add_constructor_candidate(
                        variant.name,
                        cref,
                        scope_path=type_scope,
                        inject_bare=not path,
                    )
            elif isinstance(item, ExceptionDef):
                cref = ConstructorRef(
                    owner_name=item.name,
                    variant=None,
                    owner_decl_node_id=item.node_id,
                    type_params=(),
                    owner_module_id=self._module_id,
                    owner_path=path,
                )
                self._add_constructor_candidate(
                    item.name, cref, scope_path=path, inject_bare=not path
                )
            elif isinstance(item, TypeAlias) and isinstance(item.type_expr, (NameT, AppliedT)):
                if alias_denotes_constructible_type(
                    item, partial(self._alias_target_lookup, scope_path=path)
                ):
                    cref = ConstructorRef(
                        owner_name=item.name,
                        variant=None,
                        owner_decl_node_id=item.node_id,
                        type_params=item.type_params,
                        owner_module_id=self._module_id,
                        owner_path=path,
                    )
                    self._add_constructor_candidate(
                        item.name, cref, scope_path=path, inject_bare=not path
                    )
                elif not path:
                    self._nonconstructible_type_decls[item.name] = item

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
            if cref.variant is None and cref.owner_module_id == self._module_id
        )

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
        scope = self._current_scope()
        for name, crefs in self._constructor_candidates.items():
            if name in scope.bindings:
                # Path-keyed collection has already rejected same-module
                # declaration collisions. An enum variant's bare spelling yields
                # to a value binding that already claimed it; `Owner::variant`
                # still reaches the variant and pattern position still classifies it.
                continue
            parent_ref = scope.parent.lookup(name) if scope.parent is not None else None
            if parent_ref is not None and parent_ref.kind is not BinderKind.constructor_binding:
                if self._root_declaring_candidates(name):
                    raise AglScopeError(
                        f"Name '{name}' is already declared in this scope.",
                        span=None,
                    )
                # A REPL entry's new variants remain available to pattern
                # classification, but an ordinary session binding retains its
                # expression-position meaning.
                continue
            # Use the first candidate's decl as the representative binding.
            rep = crefs[0]
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
        self._define_nonconstructible_type_bindings(scope)

    def _define_nonconstructible_type_bindings(self, scope: ScopeNode) -> None:
        """Bind a nominal alias whose chain ends at an enum to its own declaration.

        The alias has no constructor candidate — an enum's variants are its
        constructors, not the enum type itself — but its name is still
        occupied: a bare value reference resolves to a ``constructor_binding``
        so type checking reports its "type name, not a value" diagnostic
        instead of scope raising a bare "not defined" error. Same-module
        collisions were already rejected by path-keyed collection; a REPL
        parent binding is shadowed without error.
        """
        for name, decl in self._nonconstructible_type_decls.items():
            parent_ref = scope.parent.lookup(name) if scope.parent is not None else None
            if parent_ref is not None and parent_ref.kind is not BinderKind.constructor_binding:
                continue
            scope.define(
                name,
                BindingRef(
                    name=name,
                    mutable=False,
                    decl_span=decl.span,
                    decl_node_id=decl.node_id,
                    kind=BinderKind.constructor_binding,
                    module_id=self._module_id,
                ),
            )

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
            )
            self._current_scope().define(name, ref)

    def _resolve_builtin_var(self, node: BuiltinVarDecl) -> None:
        """Resolve a standard-library engine setting into a mutable register binding.

        ``builtin var`` is reserved to the canonical ``std/config`` module, but
        is otherwise a member like any other: legal at the module root and
        inside a named scope region (both keep ``_at_root`` set), rejected
        only inside a nested block. The typecheck pass then validates that the
        name is a known engine key with its canonical type.
        """
        if not self._at_root:
            raise AglScopeError(
                f"'builtin var' declarations are only allowed at the module root, "
                f"not inside a nested block (found 'builtin var {node.name}' here).",
                span=node.span,
            )
        if self._module_id != STD_CONFIG_ID:
            raise AglScopeError(
                "'builtin var' declarations are only allowed in the standard-library "
                "module 'std/config'.",
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

    def _push_scope(self, scope: ScopeNode) -> None:
        self._scope = scope

    def _pop_scope(self) -> None:
        assert self._scope is not None
        self._scope = self._scope.parent

    def _current_scope(self) -> ScopeNode:
        assert self._scope is not None, "resolver used outside of run()"
        return self._scope

    @contextmanager
    def _named_scope(self, path: ScopePath) -> Iterator[ScopeNode]:
        """Resolve a region or shorthand body in its named lexical layer."""
        named = self._scope_nodes[path]
        previous = self._current_scope()
        self._push_scope(named)
        try:
            yield named
        finally:
            self._push_scope(previous)

    @contextmanager
    def _child_scope(self, node_id: int) -> Iterator[ScopeNode]:
        """Open a fresh child scope and yield it.

        Clears the root flag for its lifetime (only the program root is
        ``_at_root``) and restores on exit.
        """
        child = ScopeNode(node_id=node_id, parent=self._current_scope())
        self._push_scope(child)
        was_root = self._at_root
        self._at_root = False
        try:
            yield child
        finally:
            self._at_root = was_root
            self._pop_scope()

    @contextmanager
    def _loop_body_ctx(self) -> Iterator[None]:
        """Context manager that sets ``_in_loop`` to ``True`` for the duration.

        Used when resolving a loop's interior (while_cond, body, until_cond)
        so that ``break``/``continue`` inside are accepted.  Save/restore so
        nested loops and post-loop scope both behave correctly.
        """
        prev = self._in_loop
        self._in_loop = True
        try:
            yield
        finally:
            self._in_loop = prev

    @contextmanager
    def _fn_boundary_ctx(self) -> Iterator[None]:
        """Reset enclosing loop/function flags while crossing a function boundary.

        Parameter defaults resolve in the enclosing lexical scope but outside the
        new function body, so neither loop exits nor returns cross this boundary.
        """
        prev_loop = self._in_loop
        prev_function = self._in_function
        self._in_loop = False
        self._in_function = False
        try:
            yield
        finally:
            self._in_loop = prev_loop
            self._in_function = prev_function

    @contextmanager
    def _function_body_ctx(self) -> Iterator[None]:
        """Mark resolution as occurring inside the current function body."""
        prev = self._in_function
        self._in_function = True
        try:
            yield
        finally:
            self._in_function = prev

    def _define(self, name: str, ref: BindingRef) -> None:
        """Define *name* in the current scope; error on redeclaration.

        A named scope's own body — reached while resolving a scope region or
        the pushed layer of a root-position binder-path shorthand — is a
        members layer, not a lexical bindings layer: a binder resolved there
        is registered into ``members`` instead, so it becomes a member
        reachable both bare (inside, via the outward walk) and by path
        (outside). It is checked and recorded against the same
        ``_scope_entity_kinds`` registry every other member at that path
        uses (declarations in the pre-pass, nested scope layers via
        ``_ensure_scope_path``), so one check governs collisions between a
        binding and a ``def``, a type, or a nested scope at the same path,
        regardless of textual order. ``_scope_entity_kinds`` is a fresh dict
        per entry, so the check only ever sees THIS entry's own members: a
        member the REPL session retained from a prior entry pre-populates
        ``scope.members`` (for lookup) but leaves no ``_scope_entity_kinds``
        trace, so redeclaring it is a replacement, matching how a redeclared
        scoped ``def`` or type behaves across entries. A retained *scope's
        own path*, unlike a retained member, is pre-seeded into the fresh
        dict at construction, so a member or declaration this entry tries to
        register there still collides — the cross-entry counterpart of
        ``_ensure_scope_path``'s same-entry seeding. This is the only
        reachable site: parameters, pattern/catch binders, and loop
        variables always resolve inside a fresh child scope with an empty
        path.

        At the true root, a declaration may claim the bare spelling of a
        constructor declared in another module (e.g. the prelude
        ``Retry``/``ExecResult`` names) or of a same-module enum variant,
        since ``Owner::variant`` and a qualified module path still reach
        those.  A same-module record, exception, or alias constructor
        declares the bare name itself, so it collides.
        """
        scope = self._current_scope()
        if scope.scope_path:
            key = (self._module_id, scope.scope_path, name)
            if self._scope_entity_kinds.get(key) is not None:
                raise AglScopeError(
                    f"Name '{name}' is already declared in this scope.",
                    span=ref.decl_span,
                )
            self._scope_entity_kinds[key] = "ordinary"
            scope.register_member(name, replace(ref, scope_path=scope.scope_path))
            return
        existing = scope.bindings.get(name)
        if existing is not None and (
            existing.kind is not BinderKind.constructor_binding
            or self._root_declaring_candidates(name)
        ):
            raise AglScopeError(
                f"Name '{name}' is already declared in this scope.",
                span=ref.decl_span,
            )
        scope.define(name, ref)

    def _check_not_reserved(self, name: str, span: object) -> None:
        """Raise if *name* is a built-in contextual name."""
        if name in _RESERVED_NAMES:
            sp = span if isinstance(span, SourceSpan) else None
            raise AglScopeError(
                f"'{name}' is a reserved contextual keyword and cannot be "
                "used as a variable or param name.",
                span=sp,
            )

    # ------------------------------------------------------------------
    # Block item resolution
    # ------------------------------------------------------------------

    def _resolve_use_context(self, items: tuple[Item, ...]) -> None:
        """Resolve use targets without letting entry bodies observe retained uses."""
        for item in items:
            if isinstance(item, UseDecl):
                self._resolve_use_decl(item)
            elif isinstance(item, ImportDecl) and item.scope_path:
                self._contribute_regional_import_bare(item)
            elif isinstance(item, ScopeRegion):
                path = self._current_scope().scope_path + (item.segment.name,)
                with self._named_scope(path):
                    self._resolve_use_context(cast(tuple[Item, ...], item.items))

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
                self._resolve_use_decl(item)
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
                if isinstance(item, ImportDecl) and item.scope_path:
                    self._contribute_regional_import_bare(item)
                # The program module-system pass processes imports/exports; this pass skips them.
                continue
            if isinstance(item, InfixDecl):
                if not self._at_root or self._current_scope().scope_path:
                    raise AglScopeError(
                        "infix declarations are only allowed at the program root.",
                        span=item.span,
                    )
                seen_non_import_item = True
                continue
            # Header enforcement: track that a non-import item has been seen.
            seen_non_import_item = True
            # Named declarations switch to their declaration's lexical scope
            # in their own handlers. Every other item belongs to the current
            # layer, so validate its qualifier chains here.
            if not isinstance(
                item,
                (ScopeRegion, FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias),
            ):
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
            elif isinstance(item, ParamDecl):
                self._resolve_param(item)
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

    # ------------------------------------------------------------------
    # Declaration handlers
    # ------------------------------------------------------------------

    def _cross_module_member_ref(
        self, atom: NameAtom, qname: QName, span: SourceSpan
    ) -> tuple[BindingRef, ConstructorRef | None]:
        """Build a bare/member ``BindingRef`` for one exposed atom's origin ``QName``.

        Shared by every site that turns a (atom, origin) pair reached through
        an import into a reference: promotes the ordinary cross-module
        ``BindingRef`` to a constructor binding -- with the constructor's own
        declaration id, kind, and owner path -- whenever *qname* names a
        record, exception, enum variant, or generic-type constructor.
        """
        module, source = qname
        constructor = self._cross_module_constructor_refs.get(qname)
        ref = self._make_cross_module_ref(module, _bare_path(atom)[-1], source, span)
        if constructor is not None:
            ref = replace(
                ref,
                decl_node_id=constructor.owner_decl_node_id,
                kind=BinderKind.constructor_binding,
                scope_path=constructor.owner_path,
            )
        return ref, constructor

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
        selected_qnames = frozenset(qname for qnames in bare.values() for qname in qnames)
        for atom, qnames in bare.items():
            for qname in qnames:
                ref, constructor = self._cross_module_member_ref(atom, qname, decl.span)
                self._current_scope().contribute_bare(atom, ref)
                if constructor is not None:
                    self._current_scope().contribute_bare_constructor(atom, constructor)
                if isinstance(atom, str):
                    self._contribute_regional_enum_variants(
                        qname, decl.span, selected_qnames=selected_qnames
                    )

    def _contribute_regional_enum_variants(
        self, qname: QName, span: SourceSpan, *, selected_qnames: Collection[QName]
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
        scope = self._current_scope()
        for variant in declaration.variants:
            # A same-named exception beside the enum already owns the bare
            # name; the exception's own bare contribution stands alone.
            exception_qname = sibling_qname(qname, variant.name)
            if exception_qname in selected_qnames and isinstance(
                self._all_public_types.get(exception_qname), ExceptionDef
            ):
                continue
            variant_qname = (module, _bare_atom((*owner_path, variant.name)))
            if variant_qname not in selected_qnames:
                continue
            ref, constructor = self._cross_module_member_ref(variant.name, variant_qname, span)
            scope.contribute_bare(variant.name, ref)
            # Every enum variant has a constructor: `_cross_module_constructor_refs`
            # is built from the same `_all_public_types` table by iterating this
            # same declaration's variants unconditionally (see
            # `_cross_module_constructor_refs` in scope/program.py).
            assert constructor is not None
            scope.contribute_bare_constructor(variant.name, constructor)

    def _resolve_use_decl(self, decl: UseDecl) -> None:
        """Inject the selected members of one already-nameable route bare."""
        retained_target = self._retained_use_targets.get(decl.node_id)
        if retained_target is not None and retained_target.local_path is not None:
            retained_local = retained_target.local_path
            self._use_targets[decl.node_id] = retained_target
            self._current_scope().contribute_local_use(
                LocalUseContribution(declaration=decl, source=self._scope_nodes[retained_local])
            )
            return
        decl = self._reinterpret_single_member_use_alias(decl)
        target = tuple(segment.name for segment in decl.target)
        local: ScopePath | None = None
        if not (retained_target and retained_target.imported_routes):
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
        imported = self._merge_use_import_targets(direct_imports, bare_imports, used_imports)
        available_import_routes = frozenset(route for route, _members in imported)
        facade_declarations = (
            self._import_env.facade_aliases.get(route[0], {}) if len(route) == 1 else {}
        )
        facade_origin_node_id = (
            retained_target.wildcard_facade_origin_node_id if retained_target is not None else None
        )
        if facade_origin_node_id is None and not decl.anchored and len(route) == 1:
            direct_modules = frozenset(module for module, _members in direct_candidates)
            matching_origins = tuple(
                origin
                for origin, modules in facade_declarations.items()
                if modules == direct_modules
            )
            if len(matching_origins) == 1:
                facade_origin_node_id = matching_origins[0]
        if retained_target is not None and retained_target.imported_routes:
            available = {**dict(imported), **dict(direct_imports)}
            replay_routes: dict[BareRoute, None] = {
                imported_route: None for imported_route in retained_target.imported_routes
            }
            if facade_origin_node_id is not None:
                origin_modules = facade_declarations.get(facade_origin_node_id, frozenset())
                for imported_route, _members in direct_imports:
                    if imported_route[0] in origin_modules:
                        replay_routes.setdefault(imported_route, None)
            replayed: list[tuple[BareRoute, Mapping[NameAtom, QName]]] = []
            for imported_route in replay_routes:
                members = available.get(imported_route)
                if members is None:
                    module, source = imported_route
                    contribution = self._import_env.contributions.get(module)
                    if contribution is None:
                        continue
                    members = self._relative_use_import_members(
                        contribution.members,
                        source,
                        target_exists=(imported_route in self._import_env.scope_origins_by_route),
                    )
                if members is not None:
                    replayed.append((imported_route, members))
            imported = tuple(replayed)
        direct_routes = {imported_route for imported_route, _members in direct_imports}
        shared_alias_facade = (
            not decl.anchored
            and len(direct_candidates) > 1
            and tuple(facade_declarations.values())
            == (frozenset(module for module, _members in direct_candidates),)
            and {imported_route for imported_route, _members in imported} == direct_routes
        ) or bool(
            retained_target
            and len(imported) > 1
            and (facade_origin_node_id is not None or len(retained_target.imported_routes) > 1)
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
            raise AglScopeError(
                f"use target '{rendered}' is ambiguous between local scope '{'::'.join(local)}' "
                f"and imported module route(s): {candidates}. Use {module_targets} to select the "
                f"module route or ::{rendered} to select the local scope.",
                span=decl.span,
            )
        if len(imported) > 1 and not shared_alias_facade:
            rendered = "/".join(route)
            candidates = ", ".join(
                f"{module.display()}::{'::'.join(root)}" if root else module.display()
                for (module, root), _members in imported
            )
            raise AglScopeError(
                f"use target '{rendered}' is ambiguous across imported modules: {candidates}. "
                f"Use a longer suffix, a /-anchored path, or as to name one import distinctly.",
                span=decl.span,
            )
        if local is None and not imported:
            rendered = "/".join(route) if route else "::".join(target)
            raise AglScopeError(
                f"use target '{rendered}' is not nameable. Import its module before using it.",
                span=decl.span,
            )
        if decl.alias is not None and not (decl.alias[0] == "_" or decl.alias[0].isalpha()):
            raise AglScopeError(
                "a whole-target use alias must be an identifier.",
                span=decl.span,
            )
        if local is not None:
            self._use_targets[decl.node_id] = ResolvedUseTarget(local_path=local)
            self._current_scope().contribute_local_use(
                LocalUseContribution(declaration=decl, source=self._scope_nodes[local])
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
            if retained_target is not None and imported_route not in available_import_routes:
                return {**self._import_scope_routes(imported_route), **selected}
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

    def _reinterpret_single_member_use_alias(self, decl: UseDecl) -> UseDecl:
        """Disambiguate ``use Scope::member as Alias`` from a whole-target alias."""
        if decl.alias is None or len(decl.target) < 2:
            return decl
        target = tuple(segment.name for segment in decl.target)
        parent = target[:-1]
        member = target[-1]
        ordinary = False
        if not decl.anchored or decl.current_module:
            bases = [()] if decl.current_module else self._scope_bases_for_use()
            ordinary = any(
                (scope := self._scope_nodes.get(base + parent)) is not None
                and (member in scope.members or base + target in self._ordered_binding_paths)
                and base + target not in self._scope_nodes
                for base in bases
            )
        if not ordinary and not decl.anchored:
            local_parent = self._use_contributed_local_target(parent, decl.span)
            if local_parent is not None:
                source_path = (*local_parent, member)
                ordinary = source_path not in self._scope_nodes and (
                    member in self._scope_nodes[local_parent].members
                    or source_path in self._ordered_binding_paths
                )
        if not ordinary and not decl.anchored:
            ordinary = any(
                (qname := members.get(member)) is not None
                and qname not in self._cross_module_type_scopes
                for _route, members in self._used_import_targets(parent)
            )
        if not ordinary and not decl.current_module:
            route = tuple(target[0].split("/"))
            source = _bare_atom(target[1:])
            ordinary = any(
                (qname := members.get(source)) is not None
                and qname not in self._cross_module_type_scopes
                for _module, members in qualifier_members(
                    self._import_env, route, anchored=decl.anchored
                )
            )
        if not ordinary and not decl.anchored:
            exposed = _bare_atom(target)
            ordinary = any(
                qname not in self._cross_module_type_scopes
                for qname in self._import_env.unqualified.get(exposed, ())
            )
        if not ordinary:
            return decl
        segment = decl.target[-1]
        selected = ImportItem(
            name=segment.name,
            rename=decl.alias,
            span=segment.span,
            node_id=segment.node_id,
        )
        return replace(decl, target=decl.target[:-1], tail=(selected,), alias=None)

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
                exists = exists or qname in self._cross_module_type_scopes
            elif path[: len(target)] == target:
                exists = True
                relative_members[_bare_atom(path[len(target) :])] = qname
        return relative_members if exists else None

    def _import_scope_routes(
        self, imported_route: BareRoute
    ) -> dict[NameAtom, frozenset[BareRoute]]:
        """Return all scope identities beneath one exact imported route."""
        contribution = self._import_env.contributions.get(imported_route[0])
        assert contribution is not None
        scope_paths = set(contribution.path_scope_paths)
        for alias_paths in contribution.alias_scope_paths.values():
            scope_paths.update(alias_paths)
        return self._relative_use_import_scope_routes(scope_paths, imported_route)

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
                    self._import_env.decl_bare, self._current_scope().scope_path
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
                        and not any(qname in self._cross_module_type_scopes for qname in selected)
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

        scope_routes = (
            self._import_env.unqualified_scope_routes,
            *(
                routes
                for _node_id, routes in self._reachable_decl_contributions(
                    self._import_env.decl_bare_scope_routes, self._current_scope().scope_path
                )
            ),
        )
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
        provenances = (
            self._import_env.unqualified_scope_routes,
            *(
                routes
                for _node_id, routes in self._reachable_decl_contributions(
                    self._import_env.decl_bare_scope_routes, self._current_scope().scope_path
                )
            ),
        )
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
        layer: ScopeNode | None = self._current_scope()
        exposed_target = _bare_atom(target)
        while layer is not None:
            candidates: list[tuple[BareRoute, Mapping[NameAtom, QName]]] = []
            for contribution in self._imported_use_contributions.get(layer.node_id, ()):
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
        layer: ScopeNode | None = self._current_scope()
        exposed_target = _bare_atom(target)
        while layer is not None:
            contributions = self._imported_use_contributions.get(layer.node_id, ())
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
        bases = [()] if decl.current_module else self._scope_bases_for_use()
        return next((base + target for base in bases if base + target in self._scope_nodes), None)

    def _scope_bases_for_use(self) -> list[ScopePath]:
        """Return lexical scope bases while resolving a header declaration."""
        bases: list[ScopePath] = []
        scope: ScopeNode | None = self._current_scope()
        while scope is not None:
            if scope.scope_path not in bases:
                bases.append(scope.scope_path)
            scope = scope.parent
        return bases

    def _use_contributed_local_target(
        self, target: ScopePath, span: SourceSpan
    ) -> ScopePath | None:
        """Resolve a scope route exposed by an earlier local ``use``."""
        layer: ScopeNode | None = self._current_scope()
        while layer is not None:
            candidates: set[ScopePath] = set()
            for contribution in layer.local_use_contributions:
                for exposed, source in self._local_use_exposures(contribution):
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
                rendered = "::".join(target)
                options = ", ".join("::".join(candidate) for candidate in sorted(candidates))
                raise AglScopeError(
                    f"use target '{rendered}' is ambiguous across local scopes: {options}.",
                    span=span,
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
        """Validate local use selections after all target members are collected."""
        for scope in self._scope_nodes.values():
            for contribution in scope.local_use_contributions:
                for exposed, source in self._local_use_exposures(contribution, validate=True):
                    if not isinstance(source, BindingRef):
                        continue
                    scope.contribute_bare(exposed, source)
                    for constructor in self._declaring_constructor_candidates(source.name, source):
                        scope.contribute_bare_constructor(exposed, constructor)

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
        scope = self._current_scope()
        exposed_scope_routes = {
            atom: route for atom, route in selected_scope_routes.items() if _bare_path(atom)
        }
        if exposed_scope_routes:
            self._imported_use_contributions.setdefault(scope.node_id, []).append(
                _ImportedUseContribution(
                    members=dict(selected),
                    scope_routes=exposed_scope_routes,
                )
            )

        def contribute(exposed: NameAtom, source: QName) -> None:
            ref, constructor = self._cross_module_member_ref(exposed, source, decl.span)
            scope.contribute_bare(exposed, ref)
            if constructor is not None:
                scope.contribute_bare_constructor(exposed, constructor)

        selected_qnames = frozenset(selected.values())
        for exposed, qname in selected.items():
            contribute(exposed, qname)
            if isinstance(exposed, str):
                self._contribute_regional_enum_variants(
                    qname, decl.span, selected_qnames=selected_qnames
                )
        for exposed, source in self._use_renamed_members(decl, members):
            contribute(exposed, source)

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
            matches = tuple(atom for atom in members if _bare_path(atom)[: len(prefix)] == prefix)
            if not matches and validate:
                raise AglScopeError(
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

        if decl.tail == ():
            selected.update(members)
        else:
            for item in cast(tuple[ImportItem, ...], decl.tail):
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
        path = self._current_scope().scope_path + (region.segment.name,)
        with self._named_scope(path):
            self._resolve_block_items(cast(tuple[Item, ...], region.items))

    def _resolve_funcdef(self, node: FuncDef) -> None:
        """Resolve a ``def`` declaration (body + params).

        At the root, the pre-pass already defined the function binding, so we
        just resolve the body with a fresh param scope. Named-scope members
        use their collected member layer; ordinary blocks reject ``def``.
        """
        if node.scope_path:
            with self._named_scope(tuple(segment.name for segment in node.scope_path)):
                self._validate_qualifier_chains(node)
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
        self._validate_qualifier_chains(node)
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
                self._validate_qualifier_chains(node)
        else:
            self._validate_qualifier_chains(node)

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
            yield

    def _resolve_let(self, node: LetDecl) -> None:
        with self._binder_scope(node, "let"):
            # A let initializer is non-recursive: no part of its pattern exists
            # until its RHS has resolved. A simple name is still a pattern
            # binder; only its slot handling differs from a broader pattern.
            self._resolve_expr(node.value)
            if isinstance(node.pattern, VarPattern):
                candidates = tuple(self._constructor_candidates.get(node.pattern.name, ()))
                if candidates:
                    self._pattern_constructor_candidates[node.pattern.node_id] = candidates
                self._check_not_reserved(node.pattern.name, node.span)
                self._define_binder(
                    node,
                    decl_node_id=node.pattern.node_id,
                    name=node.pattern.name,
                    mutable=False,
                    kind=BinderKind.let_binding,
                )
                return
            if isinstance(node.pattern, WildcardPattern):
                return
            with self._match_site_pattern_slots(node.node_id, _LET_PATTERN_POLICY):
                self._bind_pattern_vars(node.pattern, self._current_scope(), _LET_PATTERN_POLICY)

    def _resolve_var(self, node: VarDecl) -> None:
        with self._binder_scope(node, "var"):
            self._check_not_reserved(node.name, node.span)
            # Resolve RHS before defining the name (lambda non-recursion).
            self._resolve_expr(node.value)
            self._define_binder(
                node,
                decl_node_id=node.node_id,
                name=node.name,
                mutable=True,
                kind=BinderKind.var_binding,
            )

    def _define_binder(
        self,
        node: LetDecl | VarDecl,
        *,
        decl_node_id: int,
        name: str,
        mutable: bool,
        kind: BinderKind,
    ) -> None:
        """Define a resolved simple-name binder using its own identity node.

        Let callers pass the pattern node; var callers pass the declaration node
        because ``var`` has no pattern.
        """
        if name == "_":
            return
        self._define(
            name,
            BindingRef(
                name=name,
                mutable=mutable,
                decl_span=node.span,
                decl_node_id=decl_node_id,
                kind=kind,
                module_id=self._module_id,
            ),
        )

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
        if target.qualifier is not None:
            self._resolve_qualified_assign(node, target)
            return
        name = target.name
        ref = self._current_scope().lookup(name)
        if ref is None:
            ref = self._lookup_bare_contribution(name, node.span)
            # Try structured bare import contributions as a fallback, so a bare target
            # reaches an imported mutable binding just like a read.
            if ref is None:
                ref = self._lookup_import_env_unqualified(name, node.span)
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

    def _resolve_qualified_assign(self, node: AssignStmt, target: NameTarget) -> None:
        """Resolve a qualified assignment target (``PATH::name := expr``).

        A local scope path is consulted first, through the same per-path
        member namespace ``_resolve_local_scope_member`` consults for a
        qualified read -- including the same ambiguity guard
        (``_check_local_scope_route_ambiguity``) when the leading qualifier
        segment is also a module route -- so a scoped ``var`` is assignable
        through its path while a scoped ``let`` -- or a ``def``, a type, or
        an agent sharing its path -- reuses the immutable-binder diagnostic
        below. Only when the qualifier does not name a local scope path is a
        cross-module target attempted; only a ``builtin var`` binding is
        assignable across a module boundary (the sole mutable exported
        binding kind).
        """
        assert target.qualifier is not None
        qualifier = target.qualifier
        local_path = self._validate_local_scope_chain(qualifier)
        if local_path is not None:
            ref = self._scope_nodes[local_path].members.get(target.name)
            if ref is None:
                raise AglScopeError(
                    f"Unknown member '{target.name}' in scope path '{'::'.join(local_path)}'.",
                    span=node.span,
                )
            self._check_local_scope_route_ambiguity(qualifier, target.name, local_path)
        else:
            ref = self._lookup_qualified_use_contribution(qualifier, target.name, node.span)
            if ref is None:
                if not qualifier.route_segments:
                    raise AglScopeError(
                        f"'{target.name}' is not declared; assignment requires an existing "
                        f"mutable binding.",
                        span=node.span,
                    )
                ref = self._lookup_qualified_binding(qualifier, target.name, node.span)
        self._require_textually_visible(ref, node.span)
        if not ref.mutable:
            raise AglScopeError(
                f"Cannot assign to '{target.name}': "
                f"{immutable_binder_phrase(ref.kind)} (immutable).",
                span=node.span,
            )
        self._resolution[node.node_id] = ref
        self._resolve_expr(node.value)

    def _resolve_param(self, node: ParamDecl) -> None:
        """Resolve a ``param`` declaration in any module, root or region alike.

        A region does not clear ``_at_root`` (see ``_resolve_scope_region``),
        so this single check admits both a root ``param`` and one declared
        inside a scope region while still rejecting one nested in an ordinary
        block. There is no declaration-path shorthand for ``param``, so unlike
        ``let``/``var`` this needs no wrapper branching on ``node.scope_path``:
        ``_define`` below already routes to the current scope's member layer
        when a region pushed one.
        """
        if not self._at_root:
            raise AglScopeError(
                f"'param' declarations are only allowed at a static module root or "
                f"inside a named scope region (found 'param {node.name}' in a nested block).",
                span=node.span,
            )
        self._check_not_reserved(node.name, node.span)
        if node.default is not None:
            self._resolve_expr(node.default)
        ref = BindingRef(
            name=node.name,
            mutable=False,
            decl_span=node.span,
            decl_node_id=node.node_id,
            kind=BinderKind.param_binding,
            module_id=self._module_id,
        )
        self._define(node.name, ref)

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
        if isinstance(expr, VarRef):
            self._resolve_varref(expr)
        elif isinstance(expr, Call):
            self._resolve_call(expr)
        elif isinstance(expr, Template):
            self._resolve_template(expr)
        elif isinstance(expr, Block):
            with self._child_scope(expr.node_id):
                self._resolve_block_items(expr.items)
        elif isinstance(expr, If):
            self._resolve_if(expr)
        elif isinstance(expr, Case):
            self._resolve_case(expr)
        elif isinstance(expr, Loop):
            self._resolve_loop(expr)
        elif isinstance(expr, Try):
            self._resolve_try(expr)
        elif isinstance(expr, Lambda):
            self._resolve_lambda(expr)
        elif isinstance(expr, Raise):
            self._resolve_expr(expr.exc)
        elif isinstance(expr, Return):
            if not self._in_function:
                raise AglScopeError(
                    "'return' used outside a function.",
                    span=expr.span,
                )
            if expr.value is not None:
                self._resolve_expr(expr.value)
        elif isinstance(expr, Break):
            if not self._in_loop:
                raise AglScopeError(
                    "'break' used outside a loop.",
                    span=expr.span,
                )
        elif isinstance(expr, Continue):
            if not self._in_loop:
                raise AglScopeError(
                    "'continue' used outside a loop.",
                    span=expr.span,
                )
        elif isinstance(expr, FieldAccess):
            self._resolve_field_access(expr)
        elif isinstance(expr, RecordUpdate):
            self._resolve_expr(expr.target)
            for update in expr.updates:
                self._resolve_expr(update.value)
        elif isinstance(expr, IndexAccess):
            self._resolve_expr(expr.obj)
            self._resolve_expr(expr.index)
        elif isinstance(expr, BinaryOp):
            self._resolve_expr(expr.left)
            self._resolve_expr(expr.right)
        elif isinstance(expr, UnaryNot):
            self._resolve_expr(expr.operand)
        elif isinstance(expr, UnaryNeg):
            self._resolve_expr(expr.operand)
        elif isinstance(expr, IsTest):
            if expr.qualifier is None:
                candidates = self._bare_constructor_candidates(expr.variant)
                self._is_test_constructor_candidates[expr.node_id] = candidates
                if len(candidates) == 1:
                    self._constructor_refs[expr.node_id] = candidates[0]
            else:
                self._resolve_constructor_chain(
                    expr.node_id, expr.qualifier, expr.variant, defer_route_diagnostics=True
                )
            self._resolve_expr(expr.expr)
        elif isinstance(expr, Cast):
            self._resolve_expr(expr.expr)
        elif isinstance(expr, TypeApply):
            self._resolve_expr(expr.expr)
        elif isinstance(expr, ArrayLit):
            for elem in expr.elements:
                self._resolve_expr(elem)
        elif isinstance(expr, DictLit):
            for entry in expr.entries:
                self._resolve_expr(entry.key)
                self._resolve_expr(entry.value)
        else:
            assert isinstance(
                expr,
                (IntLit, DecimalLit, BoolLit, NullLit, StringLit, UnitLit, Placeholder),
            ), f"unhandled expr node: {type(expr)}"  # pragma: no cover

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
        it) but no value at all otherwise — a body-less declaration lowering
        cannot give a symbol to (see :meth:`_reject_builtin_value_ref`).
        """
        if node.name == "_":
            raise AglScopeError("'_' is not defined.", span=node.span)
        if (
            node.qualifier is not None
            and node.qualifier.anchor is QualifierAnchor.CURRENT_MODULE
            and not node.qualifier.segments
        ):
            ref = self._lookup_own_root(node.name)
            if ref is None:
                raise self._spaced_qualifier_repair(
                    self._spaced_qualifier_at(node.qualifier.span), node.qualifier.span
                ) or AglScopeError(f"'{node.name}' is not defined in this module.", span=node.span)
            self._reject_builtin_value_ref(node, ref, is_call_target=is_call_target)
            self._record_varref_binding(node, ref)
            return
        if self._resolve_local_scope_member(node):
            self._reject_builtin_value_ref(
                node, self._resolution.get(node.node_id), is_call_target=is_call_target
            )
            return
        if node.qualifier is not None:
            self._resolve_qualified_chain(node)
            self._reject_builtin_value_ref(
                node, self._resolution.get(node.node_id), is_call_target=is_call_target
            )
            return
        # Standard lexical lookup. A receiver method may legally share a host
        # built-in's spelling, but bare call syntax still denotes the visible
        # builtin declaration; methods are selected by qualification or member
        # access. Discard that method candidate so the ordinary contribution
        # lookup below can recover the builtin with its real provenance.
        ref = self._current_scope().lookup(node.name)
        if (
            is_call_target
            and node.name in _BUILTIN_CALL_NAMES
            and ref is not None
            and ref.kind is BinderKind.function_binding
            and not ref.is_builtin
        ):
            ref = None
        # Only a missing or constructor binding can consume regional candidates:
        # any other kind leaves *ref* untouched below and returns from
        # ``_record_varref_binding`` before the candidates are read.
        regional_candidates = (
            self._regional_constructor_candidates(node.name)
            if ref is None or ref.kind is BinderKind.constructor_binding
            else None
        )
        if ref is None or (
            ref.kind is BinderKind.constructor_binding and regional_candidates is not None
        ):
            contributed = self._lookup_bare_contribution(node.name, node.span)
            if contributed is not None:
                ref = contributed
        if ref is None:
            # Try structured bare import contributions as a fallback.
            ref = self._lookup_import_env_unqualified(node.name, node.span)
            if ref is None:
                raise self._spaced_qualifier_repair(
                    self._spaced_qualifier_around(node.span), node.span
                ) or AglScopeError(
                    f"'{node.name}' is not defined.",
                    span=node.span,
                )
        self._reject_builtin_value_ref(node, ref, is_call_target=is_call_target)
        self._record_varref_binding(node, ref, candidates=regional_candidates)

    def _reject_builtin_value_ref(
        self, node: VarRef, ref: BindingRef | None, *, is_call_target: bool
    ) -> None:
        """Raise if *ref* is a scoped ``builtin def``'s binding used as a value.

        Mirrors :meth:`_resolve_call`'s own resolved-binding classification:
        once a reference resolves to a ``function_binding`` for a built-in
        declaration, a call site turns it into a host-dispatch classification
        (*is_call_target* is set), while every other value context has no
        host dispatch to offer. Qualified methods may legitimately share a
        built-in's spelling, so declaration provenance rather than the
        reference name distinguishes the host implementation.
        """
        if (
            not is_call_target
            and self._is_builtin_function_ref(ref)
            and ref is not None
            and ref.name in _BUILTIN_CALL_NAMES
        ):
            raise AglScopeError(
                f"Built-in function '{ref.name}' cannot be used as a value.", span=node.span
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
            owner_names = ", ".join(
                f"'{candidate.owner_name}'" for candidate in resolved_candidates
            )
            raise AglScopeError(
                f"'{node.name}' is ambiguous: it is declared as a constructor "
                f"in multiple types ({owner_names}). "
                f"Qualify the reference, e.g. '{resolved_candidates[0].owner_name}::{node.name}'.",
                span=node.span,
            )
        if len(resolved_candidates) == 1:
            self._constructor_refs[node.node_id] = resolved_candidates[0]

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
        if ref.scope_path and ref.module_id == self._module_id:
            scoped = tuple(self._scoped_constructor_candidates.get((ref.scope_path, name), ()))
            if scoped:
                return scoped
            return tuple(
                candidate
                for candidate in self._constructor_candidates.get(name, ())
                if candidate.owner_module_id == ref.module_id
                and candidate.owner_path == ref.scope_path[:-1]
                and candidate.owner_name == ref.scope_path[-1]
            )
        return tuple(self._constructor_candidates.get(name, ()))

    def _validate_qualifier_chains(self, program: object) -> None:
        """Validate qualifier syntax in the current lexical scope layer."""

        def validate(node: object) -> None:
            chain = (
                node.qualifier
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
            self._validate_local_scope_chain(chain)

        walk(program, validate)

    def _scope_bases(self, chain: QualifierChain) -> list[ScopePath]:
        """Return lexical bases from which an unanchored scope path may start."""
        if chain.anchor is QualifierAnchor.CURRENT_MODULE:
            return [()]
        bases: list[ScopePath] = []
        scope: ScopeNode | None = self._current_scope()
        while scope is not None:
            if scope.scope_path not in bases:
                bases.append(scope.scope_path)
            scope = scope.parent
        return bases

    def _validate_local_scope_chain(
        self, chain: QualifierChain | None, *, bases: list[ScopePath] | None = None
    ) -> ScopePath | None:
        """Find a local scope path and reject applications to plain scopes.

        The caller supplies lexical bases when one is meaningful.  The
        declaration pre-pass uses the module root, while expression and
        ``is`` resolution use the active lexical layers.  A non-local chain is
        deliberately left for module-route and constructor compatibility
        resolution.
        """
        if chain is None or chain.anchor is QualifierAnchor.MODULE or not chain.segments:
            return None
        relative_path = tuple(segment.name for segment in chain.segments)
        paths = [base + relative_path for base in (bases or self._scope_bases(chain))]
        known_paths = self._scope_nodes if hasattr(self, "_scope_nodes") else self._scope_paths
        path = next((candidate for candidate in paths if candidate in known_paths), None)
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
        """Raise if *chain* names both a local scope path and a module route.

        Shared by qualified value resolution and qualified assignment so a
        read and a write of the same spelling agree: a leading qualifier
        segment that is genuinely both a local scope (or type name) and an
        imported module route is ambiguous regardless of which direction the
        reference goes.
        """
        relative_path = tuple(segment.name for segment in chain.segments)
        if not (
            chain.anchor is None and qualifier_contributes(self._import_env, relative_path, name)
        ):
            return
        local_kind = "a type name" if path in self._type_paths else "a local scope"
        raise AglScopeError(
            f"Qualifier '{relative_path[0]}' is both {local_kind} and a module route for "
            f"'{name}'. {qualification_repair_guidance()}",
            span=chain.span,
        )

    def _resolve_local_scope_member(self, node: VarRef) -> bool:
        """Resolve a scoped member through ordered lexical scope layers.

        Local paths are considered before import routes.  A type-owned member
        absent from its scope deliberately falls through to the established
        constructor resolver, which owns constructor-specific diagnostics until
        constructor-chain migration is complete.
        """
        chain = node.qualifier
        if chain is None or chain.anchor is QualifierAnchor.MODULE or not chain.segments:
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
            known_segment = any(relative_path[0] in scope_path for scope_path in self._scope_nodes)
            if (chain.anchor is QualifierAnchor.CURRENT_MODULE and len(chain.segments) > 1) or (
                known_segment
                and not has_module_route
                and not has_leading_route
                and not has_opened_member
            ):
                raise AglScopeError(
                    f"Unknown scope path '{'::'.join(relative_path)}'.", span=chain.span
                )
            return False

        ref = self._scope_nodes[path].members.get(node.name)
        rendered = "::".join(path)
        if ref is None:
            if path in self._type_paths:
                return False
            raise AglScopeError(
                f"Unknown member '{node.name}' in scope path '{rendered}'.", span=node.span
            )
        self._check_local_scope_route_ambiguity(chain, node.name, path)
        self._require_textually_visible(ref, node.span)
        if ref.kind is BinderKind.constructor_binding:
            candidates = self._scoped_constructor_candidates.get((path, node.name), ())
            if len(candidates) == 1:
                self._constructor_refs[node.node_id] = candidates[0]
                return True
            return False
        self._resolution[node.node_id] = ref
        return True

    def _resolve_qualified_chain(self, node: VarRef) -> None:
        """Resolve a qualified value through one ordered scope-or-route chain.

        An imported route owns the complete requested path atom. Resolving it
        first keeps ordinary members of a type-owned scope distinct from enum
        variants and record construction, while also ensuring selection filters
        apply to the final atom rather than only its type owner.
        """
        chain = node.qualifier
        assert chain is not None
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
            route = tuple(part for part in chain.segments[0].name.split("/"))
            try:
                self._resolve_varref_qualified(node, chain)
                return
            except AglScopeError as error:
                direct_error = error
                atom_path = (*tuple(segment.name for segment in chain.segments[1:]), node.name)
                if len(atom_path) > 1:
                    owner_atom: NameAtom = atom_path[0] if len(atom_path) == 2 else atom_path[:-1]
                    owner = try_resolve_qualified_member(
                        self._import_env, route, owner_atom, anchored=chain.anchored
                    )
                    if owner in self._cross_module_constructible_types:
                        raise
        if self._resolve_constructor_chain(node.node_id, chain, node.name):
            return
        if direct_error is not None:
            raise direct_error
        # Only a ``CURRENT_MODULE``-anchored chain reaches here: every other
        # anchor either resolves above or leaves ``direct_error`` set. Its
        # empty-segment self-reference form is handled in ``_resolve_varref``,
        # so the chain names one segment this module does not declare.
        segment = chain.segments[0]
        missing_error = AglScopeError(
            f"'{segment.name}' is not defined in this module.", span=segment.span
        )
        raise (
            self._spaced_qualifier_repair(
                self._spaced_qualifier_at(chain.span) or self._spaced_qualifier_around(node.span),
                node.span,
            )
            or missing_error
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
        opened = self._lookup_bare_contribution(atom, span)
        if opened is None:
            return None
        imported = self._qualified_import_resolution(chain, name)
        if isinstance(imported, QualResolutionAmbiguous):
            raise AglScopeError(
                f"'{chain.render()}::{name}' is ambiguous across use and import routes. "
                f"{qualification_repair_guidance()}",
                span=span,
            )
        if isinstance(imported, QualResolutionFound):
            imported_ref = self._cross_module_member_ref(name, imported.qname, span)[0]
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
                raise AglScopeError(
                    f"'{chain.render()}::{name}' is ambiguous between a use contribution "
                    f"and an import route. {qualification_repair_guidance()}",
                    span=span,
                )
        return opened

    def _resolve_constructor_chain(
        self,
        node_id: int,
        chain: QualifierChain,
        variant: str,
        *,
        defer_route_diagnostics: bool = False,
    ) -> bool:
        """Record a constructor result when *chain* ends at a type owner.

        Patterns and ``is`` tests retain type-checker qualification verdicts.
        They resolve a valid owner here, but defer a bad imported route or a
        local/import clash until the checker can assess it against the enum
        being matched.
        """
        opened = self._use_constructor_candidates(chain, variant)
        if opened is not None:
            if len(opened) == 1:
                self._constructor_refs[node_id] = next(iter(opened))
                return True
            rendered = "::".join((*tuple(segment.name for segment in chain.segments), variant))
            raise AglScopeError(
                f"Constructor '{rendered}' is not visible through this use route.",
                span=chain.span,
            )
        local_path = self._validate_local_scope_chain(chain)
        if local_path in self._type_paths:
            if chain.anchor is None and qualifier_contributes(
                self._import_env,
                tuple(segment.name for segment in chain.segments),
                variant,
            ):
                if defer_route_diagnostics:
                    return False
                rendered = "::".join(segment.name for segment in chain.segments)
                raise AglScopeError(
                    f"Qualifier '{rendered}' is both a type name and a module route for "
                    f"'{variant}'. {qualification_repair_guidance()}",
                    span=chain.span,
                )
            self._constructor_refs[node_id] = self._constructor_for_type_path(
                local_path, variant, chain.span
            )
            return True
        try:
            owner = self._imported_chain_owner(chain, variant)
        except AglScopeError:
            if defer_route_diagnostics:
                return False
            raise
        if owner is not None:
            self._constructor_refs[node_id] = owner
            return True
        if (
            chain.anchor is not QualifierAnchor.MODULE
            and len(chain.segments) == 1
            and chain.segments[0].name in self._declared_type_names
        ):
            type_name = chain.segments[0].name
            candidate = next(
                (
                    candidate
                    for candidates in self._constructor_candidates.values()
                    for candidate in candidates
                    if candidate.owner_name == type_name
                    and (
                        chain.anchor is not QualifierAnchor.CURRENT_MODULE
                        or candidate.owner_module_id == self._module_id
                    )
                ),
                None,
            )
            if candidate is not None:
                self._constructor_refs[node_id] = replace(candidate, variant=variant)
                return True
        return False

    def _use_constructor_candidates(
        self, chain: QualifierChain, variant: str
    ) -> set[ConstructorRef] | None:
        """Return a use route's exact constructor selection, including hidden verdicts."""
        if chain.anchor is not None:
            return None
        relative_path = tuple(segment.name for segment in chain.segments)
        opened = self._regional_constructor_candidates(_bare_atom((*relative_path, variant)))
        if opened:
            if len(opened) > 1:
                rendered = "::".join((*relative_path, variant))
                raise AglScopeError(
                    f"'{rendered}' is ambiguous across use routes. "
                    f"{qualification_repair_guidance()}",
                    span=chain.span,
                )
            imported = self._qualified_import_resolution(chain, variant)
            imported_constructor = (
                self._cross_module_constructor_refs.get(imported.qname)
                if isinstance(imported, QualResolutionFound)
                else None
            )
            if isinstance(imported, QualResolutionAmbiguous) or (
                isinstance(imported, QualResolutionFound) and opened != {imported_constructor}
            ):
                rendered = "::".join((*relative_path, variant))
                raise AglScopeError(
                    f"'{rendered}' is ambiguous between a use contribution and an import "
                    f"route. {qualification_repair_guidance()}",
                    span=chain.span,
                )
        if opened is not None:
            return opened
        prefix_atom = _bare_atom(relative_path)
        if self._bare_contribution_candidates(prefix_atom) is None:
            return None
        owner_ref = cast(BindingRef, self._lookup_bare_contribution(prefix_atom, chain.span))
        if not isinstance(self._type_declaration_for_owner(owner_ref), TypeAlias):
            return set()
        declared_child = _bare_atom((*owner_ref.scope_path, owner_ref.name, variant))
        if (owner_ref.module_id, declared_child) in self._decl_info:
            return set()
        return {self._constructor_for_owner_ref(owner_ref, variant)}

    def _constructor_for_type_path(
        self, path: ScopePath, variant: str, span: SourceSpan
    ) -> ConstructorRef:
        """Return the constructor-shaped chain result owned by local *path*.

        The owner's scope path stays in ``owner_path`` and never merges into
        ``owner_name``: a constructor identity is structured, so a display
        spelling must not double as one.
        """
        candidates = self._scoped_constructor_candidates.get((path, variant), ())
        if candidates:
            return candidates[0]
        declaration = self._declaration_items.get((self._module_id, path[:-1], path[-1]))
        if declaration is None:
            retained = next(
                (
                    candidate
                    for candidate in self._constructor_candidates.get(variant, ())
                    if candidate.owner_module_id == self._module_id
                    and candidate.owner_path == path[:-1]
                    and candidate.owner_name == path[-1]
                ),
                None,
            )
            if retained is None:
                # A retained type path can go stale across REPL entries when a
                # later entry's own declaration (not itself a type) claims the
                # same path a prior entry's type left behind — the path is
                # still in ``_type_paths``, but no constructor backs it any
                # more. A clean diagnostic beats a crash for a condition
                # reachable from ordinary REPL input.
                raise AglScopeError(
                    f"'{variant}' is not a member of '{'::'.join(path)}'.", span=span
                )
            return retained
        assert isinstance(declaration, (RecordDef, EnumDef, ExceptionDef, TypeAlias))
        return ConstructorRef(
            owner_name=path[-1],
            variant=variant,
            owner_decl_node_id=declaration.node_id,
            type_params=declaration.type_params,
            owner_module_id=self._module_id,
            owner_path=path[:-1],
        )

    def _type_declaration_for_owner(
        self, owner_ref: BindingRef
    ) -> RecordDef | EnumDef | ExceptionDef | TypeAlias | None:
        """Return the program type declaration denoted by an owner binding."""
        atom = _bare_atom((*owner_ref.scope_path, owner_ref.name))
        return self._all_public_types.get((owner_ref.module_id, atom))

    def _constructor_for_owner_ref(self, owner_ref: BindingRef, variant: str) -> ConstructorRef:
        """Return a constructor-shaped result owned by a resolved type binding."""
        candidate = next(
            (
                candidate
                for candidates in self._constructor_candidates.values()
                for candidate in candidates
                if candidate.owner_module_id == owner_ref.module_id
                and candidate.owner_path == owner_ref.scope_path
                and candidate.owner_name == owner_ref.name
            ),
            None,
        )
        if candidate is not None:
            return replace(candidate, variant=variant)
        return ConstructorRef(
            owner_name=owner_ref.name,
            variant=variant,
            owner_decl_node_id=owner_ref.decl_node_id,
            type_params=(),
            owner_module_id=owner_ref.module_id,
            owner_path=owner_ref.scope_path,
        )

    def _imported_chain_owner(self, chain: QualifierChain, variant: str) -> ConstructorRef | None:
        """Resolve a type-owning segment reached through imports.

        The final chain segment is selected as a normal imported member; any
        preceding segments are its route.  A one-segment chain may instead
        name a type exposed by an import tail. This is the same chain walk used for
        ordinary qualified values, with only the resulting member kind
        determining whether it owns a constructor.
        """
        if chain.anchor is QualifierAnchor.CURRENT_MODULE or not chain.segments:
            return None
        owner_ref: BindingRef | None = None
        if len(chain.segments) == 1 and not chain.anchored:
            owner_ref = self._lookup_import_env_unqualified(chain.segments[0].name, chain.span)
            if owner_ref is not None and owner_ref.kind is not BinderKind.constructor_binding:
                return None
        elif len(chain.segments) > 1:
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
            except AglScopeError:
                # The final scope segment can be part of a selected path atom
                # rather than a separately selected type owner. Let the normal
                # module-path resolver consume the complete atom.
                return None
            if owner_ref.kind is not BinderKind.constructor_binding:
                rendered = chain.render()
                raise AglScopeError(f"'{rendered}' is not a constructible type.", span=chain.span)
        if owner_ref is None:
            return None
        return self._constructor_for_owner_ref(owner_ref, variant)

    def _nearest_bare_contribution_layer(
        self,
        name: NameAtom,
        *,
        binding_predicate: Callable[[BindingRef], bool] | None = None,
        constructors_only: bool = False,
    ) -> tuple[ScopeNode, set[BindingRef], set[ConstructorRef]] | None:
        """Return the nearest static and live-use candidates in one namespace."""
        layer: ScopeNode | None = self._current_scope()
        while layer is not None:
            bindings = set(layer.bare_contributions.get(name, ()))
            constructors = set(layer.bare_constructor_contributions.get(name, ()))
            for contribution in layer.local_use_contributions:
                for exposed, source in self._local_use_exposures(contribution):
                    if exposed != name or not isinstance(source, BindingRef):
                        continue
                    bindings.add(source)
                    constructors.update(self._declaring_constructor_candidates(source.name, source))
            if binding_predicate is not None:
                bindings = {ref for ref in bindings if binding_predicate(ref)}
                constructors = {
                    constructor
                    for constructor in constructors
                    if any(
                        constructor.owner_module_id == ref.module_id
                        and constructor.owner_decl_node_id == ref.decl_node_id
                        for ref in bindings
                    )
                }
            if constructors or (bindings and not constructors_only):
                return layer, bindings, constructors
            layer = layer.parent
        return None

    def _is_value_contribution(self, ref: BindingRef) -> bool:
        """Whether a shared contribution denotes a value in addition to any type."""
        atom = _bare_atom((*ref.scope_path, ref.name))
        return (
            ref.kind is not BinderKind.constructor_binding
            or (ref.module_id, atom) in self._cross_module_constructor_refs
            or bool(self._declaring_constructor_candidates(ref.name, ref))
        )

    def _bare_contribution_candidates(
        self, name: NameAtom, *, values_only: bool = False
    ) -> set[BindingRef] | None:
        """Return the nearest region's bare contributions in the requested namespace."""
        nearest = self._nearest_bare_contribution_layer(
            name,
            binding_predicate=self._is_value_contribution if values_only else None,
        )
        return None if nearest is None else nearest[1]

    def _regional_constructor_candidates(self, name: NameAtom) -> set[ConstructorRef] | None:
        """Return the nearest region's constructor candidates, including live local uses."""
        nearest = self._nearest_bare_contribution_layer(name, constructors_only=True)
        return None if nearest is None else nearest[2]

    def _lookup_bare_contribution(self, name: NameAtom, span: SourceSpan) -> BindingRef | None:
        """Resolve one region's bare contributions, deferring clashes to use sites."""
        nearest = self._nearest_bare_contribution_layer(
            name, binding_predicate=self._is_value_contribution
        )
        value_layer_found = nearest is not None
        if nearest is None:
            # Keep a lone type-only spelling available for the checker's
            # dedicated "type name, not a value" diagnostic.
            nearest = self._nearest_bare_contribution_layer(name)
            if nearest is None:
                return None
        selected_layer, resolved, _constructors = nearest
        assert self._root_scope is not None
        imported_refs = {
            ref
            for qname in self._import_env.unqualified.get(name, frozenset())
            if self._is_value_contribution(
                ref := self._cross_module_member_ref(name, qname, span)[0]
            )
        }
        if imported_refs and not value_layer_found:
            resolved = imported_refs
        elif selected_layer is self._root_scope:
            resolved.update(imported_refs)
        distinct = {
            (ref.module_id, ref.scope_path, ref.decl_node_id, ref.kind): ref for ref in resolved
        }
        if len(distinct) == 1:
            return next(iter(distinct.values()))
        qualifiers = ", ".join(
            sorted(
                spell_declaration(
                    ref.module_id, (*ref.scope_path, ref.name), local_to=self._module_id
                )
                for ref in distinct.values()
            )
        )
        rendered = "::".join(_bare_path(name))
        raise AglScopeError(
            f"'{rendered}' is ambiguous: contributed by multiple use declarations. "
            f"Use a qualified reference to disambiguate: {qualifiers}",
            span=span,
        )

    def _lookup_import_env_unqualified(self, name: str, span: SourceSpan) -> BindingRef | None:
        """Look up a bare name in the tailed-import environment.

        Returns a ``BindingRef`` if exactly one ``QName`` matches, or raises
        ``AglScopeError`` on ambiguity (clash-on-use). Returns ``None`` if the
        name is not contributed by a tailed import. Shared by bare value
        references and bare assignment targets.
        """
        exposed = self._import_env.unqualified.get(name)
        if exposed is None:
            return None
        qnames = {
            qname
            for qname in exposed
            if self._is_value_contribution(self._cross_module_member_ref(name, qname, span)[0])
        }
        if not qnames:
            qnames = set(exposed)
        if len(qnames) > 1:
            # Clash-on-use: more than one module exposes this name.
            qualifiers = sorted(
                qn[0].display() + "::" + "::".join((qn[1],) if isinstance(qn[1], str) else qn[1])
                for qn in qnames
            )
            hint = ", ".join(qualifiers)
            raise AglScopeError(
                f"'{name}' is ambiguous: imported from multiple modules. "
                f"Use a qualified reference to disambiguate: {hint}",
                span=span,
            )
        # Exactly one QName.
        qname = next(iter(qnames))
        return self._make_cross_module_ref(qname[0], name, qname[1], span)

    def _resolve_varref_qualified(self, node: VarRef, module_qualifier: QualifierChain) -> None:
        """Resolve an imported qualified VarRef."""
        qname = self._resolve_qualified_qname(module_qualifier, node.name, node.span)
        constructor = self._cross_module_constructor_refs.get(qname)
        if constructor is not None and (constructor.owner_path or constructor.variant is not None):
            self._constructor_refs[node.node_id] = constructor
            return
        self._resolution[node.node_id] = self._make_cross_module_ref(
            qname[0], node.name, qname[1], node.span
        )

    def _lookup_qualified_binding(
        self, qualifier: QualifierChain, name: str, span: SourceSpan
    ) -> BindingRef:
        """Resolve a qualified value or assignment target through the shared resolver."""
        qname = self._resolve_qualified_qname(qualifier, name, span)
        return self._make_cross_module_ref(qname[0], name, qname[1], span)

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
                members.get(_bare_atom(cumulative)) in self._cross_module_type_scopes
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
            unknown_qualifier=lambda rendered: AglScopeError(
                f"No module imported under qualifier '{rendered}'.", span=span
            ),
            missing_member=lambda rendered: AglScopeError(
                f"'{name}' is not a public member of imported module '{rendered}' or is hidden.",
                span=span,
            ),
            ambiguous=lambda message: AglScopeError(message, span=span),
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
        assert self._root_scope is not None, "_lookup_own_root called outside of run()"
        ref = self._root_scope.bindings.get(name)
        if ref is None and self._repl_session_scope is not None:
            ref = self._repl_session_scope.bindings.get(name)
        return ref

    def _make_cross_module_ref(
        self,
        owning_module: ModuleId,
        exposed_name: str,
        src_name: NameAtom,
        span: SourceSpan,
    ) -> BindingRef:
        """Build a ``BindingRef`` for a cross-module name resolution.

        Parameters
        ----------
        owning_module:
            The ``ModuleId`` of the module that declares the name.
        exposed_name:
            The name as written in this module (after any rename).
        src_name:
            The original name in the owning module.
        span:
            Source span of the reference site (for synthetic decl_span).
        """
        key = (owning_module, src_name)
        decl_node_id, decl_span, kind, is_builtin = self._decl_info.get(
            key, (-1, span, BinderKind.function_binding, False)
        )
        path = (src_name,) if isinstance(src_name, str) else src_name
        return BindingRef(
            name=path[-1],
            # Only ``builtin var`` bindings are mutable across a module boundary;
            # every other exported binding (functions, constructors, …) is
            # immutable at the reference site.
            mutable=kind is BinderKind.builtin_var_binding,
            decl_span=decl_span,
            decl_node_id=decl_node_id,
            kind=kind,
            module_id=owning_module,
            scope_path=path[:-1],
            is_builtin=is_builtin,
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
        if advisory.type_qualified:
            _node_id, _span, kind, _is_builtin = self._decl_info.get(
                result.qname, (-1, advisory.dcolon_span, BinderKind.let_binding, False)
            )
            if kind is not BinderKind.constructor_binding:
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
            if ref is not None and self._is_builtin_function_ref(ref):
                kind = _BUILTIN_CALL_NAMES.get(ref.name)
                if kind is not None:
                    self._builtin_calls[node.node_id] = kind
        elif isinstance(callee, FieldAccess):
            self._resolve_field_access(callee)
            # Member selection is type-directed, so scope cannot yet know
            # whether this spelling names a builtin method or an ordinary
            # method with the same name. Record the possible host route; the
            # checker confirms it only after selecting the method declaration.
            if callee.field in _BUILTIN_CALL_NAMES:
                self._builtin_calls[node.node_id] = _BUILTIN_CALL_NAMES[callee.field]
        else:
            self._resolve_expr(callee)
        # Resolve positional args.
        for arg in node.args:
            self._resolve_expr(arg)
        # Resolve named-arg values.
        for named in node.named_args:
            self._resolve_expr(named.value)

    def _resolve_field_access(self, expr: FieldAccess) -> None:
        """Resolve a field-access expression by resolving its object as a value."""
        if isinstance(expr.obj, VarRef) and expr.obj.qualifier is None:
            existing = self._current_scope().lookup(expr.obj.name)
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
                with self._match_site_pattern_slots(branch.node_id, _CASE_PATTERN_POLICY):
                    self._bind_pattern_vars(branch.pattern, branch_scope, _CASE_PATTERN_POLICY)
                self._resolve_expr_or_block(branch.body)

    def _resolve_loop(self, node: Loop) -> None:
        """Resolve a unified loop expression.

        Resolution order (all in the ENCLOSING scope, before the loop variable
        is bound, so none of these can reference the loop variable):
        - ``bound`` (if any)
        - ``for_iter`` (if any) — the range start value for a range ``for``
        - ``for_range_to`` (if any) — the range upper/lower bound
        - ``for_range_by`` (if any) — the range step

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
        if node.for_range_by is not None:
            self._resolve_expr(node.for_range_by)
        with self._child_scope(node.node_id) as loop_scope:
            with self._loop_body_ctx():
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
        # A bare receiver marker is meaningful only on a definition owned by
        # a nominal type. Annotated ``self`` remains an ordinary lambda param.
        if node.params and node.params[0].type_expr is None:
            raise AglScopeError(
                "'self' requires an enclosing type scope; lambdas cannot declare methods.",
                span=node.params[0].span,
            )
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
        with self._fn_boundary_ctx():
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
                    with self._function_body_ctx():
                        self._resolve_expr_or_block(node.body)

    # ------------------------------------------------------------------
    # Pattern variable binding
    # ------------------------------------------------------------------

    def _qualified_pattern_constructor_candidates(
        self, node: ConstructorPattern
    ) -> tuple[ConstructorRef, ...]:
        """Select the constructors a qualified pattern spelling can match.

        Ordering mirrors qualified value resolution: a scope-use contribution or
        a local scope member is considered before an import route owning the
        complete atom, and only then is the chain read as a type owner with the
        pattern name as its variant. A spelling that resolves to nothing yields
        an empty tuple, deferring the diagnostic to type checking.
        """
        chain = node.qualifier
        assert chain is not None
        relative_path = tuple(segment.name for segment in chain.segments)
        if chain.anchor is QualifierAnchor.CURRENT_MODULE and not relative_path:
            # ``::Name`` names this module's own root declaration and must not
            # reach an imported constructor of the same spelling.
            return tuple(
                candidate
                for candidate in self._constructor_candidates.get(node.name, ())
                if candidate.owner_module_id == self._module_id and not candidate.owner_path
            )
        opened = self._use_constructor_candidates(chain, node.name)
        if opened is not None:
            if not opened:
                rendered = "::".join((*relative_path, node.name))
                raise AglScopeError(
                    f"Constructor '{rendered}' is not visible through this use route.",
                    span=chain.span,
                )
            return tuple(opened)
        local_path = self._validate_local_scope_chain(chain)
        if local_path is not None:
            if local_path in self._type_paths:
                return (self._constructor_for_type_path(local_path, node.name, chain.span),)
            scoped = self._scoped_constructor_candidates.get((local_path, node.name), ())
            if scoped:
                return tuple(scoped)
        if chain.anchor is not QualifierAnchor.CURRENT_MODULE and chain.segments:
            qname = self._try_resolve_qualified_qname(chain, node.name)
            if qname is not None:
                imported = self._cross_module_constructor_refs.get(qname)
                if imported is not None:
                    return (imported,)
        if self._resolve_constructor_chain(
            node.node_id, chain, node.name, defer_route_diagnostics=True
        ):
            return (self._constructor_refs[node.node_id],)
        return ()

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
        scope: ScopeNode | None = self._current_scope()
        while scope is not None:
            if scope.scope_path:
                owned = self._owned_scope_constructor_candidates(scope.scope_path, name)
                if owned:
                    return owned
            scope = scope.parent
        return tuple(self._constructor_candidates.get(name, ()))

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
        self, pattern: Pattern, scope: ScopeNode, policy: _PatternResolutionPolicy
    ) -> None:
        """Bind candidates according to a case or let match-site policy.

        The syntax helper is the sole pattern walker. Case roots remain
        constructor-only, let roots always bind, nested bare names remain
        field-directed, and ``as`` names always bind.
        """

        def record_constructor_candidates(node: object) -> None:
            if not isinstance(node, ConstructorPattern):
                return
            self._pattern_constructor_spellings[node.node_id] = node.name
            if node.qualifier is None:
                self._pattern_constructor_candidates[node.node_id] = (
                    self._bare_constructor_candidates(node.name)
                )
                return
            # A qualified spelling never falls back to the bare one: every
            # failure yields an empty set so the checker can judge the
            # spelling against the type actually being matched.
            candidates = self._qualified_pattern_constructor_candidates(node)
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
                self._pattern_constructor_spellings[candidate.node_id] = candidate.name
            binds = candidate.is_as_pattern or candidate.nested or policy.root_bare_binds
            if not binds:
                if not constructor_candidates:
                    raise AglScopeError(
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
                can_match_bare_pattern=any(
                    constructor.can_match_bare_pattern for constructor in constructor_candidates
                ),
            )

    @contextmanager
    def _match_site_pattern_slots(
        self, match_site_node_id: int, policy: _PatternResolutionPolicy
    ) -> Iterator[None]:
        """Collect slots owned by one case branch or let declaration."""
        parent_slots = self._active_match_site_pattern_slots
        parent_node_id = self._active_match_site_node_id
        parent_binder_kind = self._active_match_site_binder_kind
        self._active_match_site_pattern_slots = {}
        self._active_match_site_node_id = match_site_node_id
        self._active_match_site_binder_kind = policy.binder_kind
        try:
            yield
        finally:
            assert self._active_match_site_pattern_slots is not None
            self._match_site_pattern_slots_by_node[match_site_node_id] = tuple(
                sorted(self._active_match_site_pattern_slots.values())
            )
            self._active_match_site_pattern_slots = parent_slots
            self._active_match_site_node_id = parent_node_id
            self._active_match_site_binder_kind = parent_binder_kind

    def _add_pattern_slot_candidate(
        self,
        name: str,
        span: SourceSpan,
        pattern_node_id: int,
        scope: ScopeNode,
        *,
        can_match_bare_pattern: bool,
    ) -> None:
        """Join a candidate to its match-site-local shared binding.

        Reject duplicate binders immediately only when neither the arriving
        candidate nor any prior slot candidate can match a bare pattern.
        """
        self._check_not_reserved(name, span)
        match_site_slots = self._active_match_site_pattern_slots
        match_site_node_id = self._active_match_site_node_id
        binder_kind = self._active_match_site_binder_kind
        assert match_site_slots is not None
        assert match_site_node_id is not None
        assert binder_kind is not None
        slot_id = match_site_slots.get(name)
        if slot_id is None:
            slot_id = self._next_pattern_slot_id
            self._next_pattern_slot_id += 1
            match_site_slots[name] = slot_id
            alternative = scope.parent.lookup(name) if scope.parent is not None else None
            self._pattern_slots[slot_id] = PatternSlot(
                slot_id=slot_id,
                name=name,
                candidates=(),
                alternative=alternative,
                match_site_node_id=match_site_node_id,
                binder_kind=binder_kind,
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
