"""Static name-resolution pass for the AgL pipeline.

``resolve_module(program)`` performs a full single-pass walk over the AST,
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
6. ``param`` and ``program`` declarations are only valid at the module root.
7. ``def`` declarations are valid at the module root and in named scope
   regions; a pre-pass collects them by path so root and same-scope members
   support mutual recursion.
8. ``agent`` declarations are valid in entry-module root and named scope
   regions; a pre-pass collects structured path identities and their value
   bindings.

Built-in call classification
-----------------------------
``print`` / ``exec`` / ``ask`` are contextual built-ins.  They cannot be
declared (``let``/``var``/``def``/``param``/``agent``/param/pattern/catch).
In **call position** (a ``Call`` whose callee is a bare ``VarRef`` with one
of these names), the resolver records ``Call.node_id → BuiltinKind`` in the
``builtin_calls`` side table and does NOT attempt to resolve the callee as a
normal binding.  A bare ``VarRef("print")`` (not in call position) raises
``AglScopeError`` — they are not first-class values in AgL.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, cast

from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, STD_CONFIG_ID, ModuleId, spell_declaration
from agm.agl.scope.imports import (
    NameAtom,
    QName,
    QualResolutionFound,
    qualification_repair_guidance,
    qualifier_candidates,
    qualifier_contributes,
    render_qualifier,
    resolve_alias_target,
    resolve_qualified,
    resolve_qualified_member,
    try_resolve_qualified_member,
)
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    AgentKey,
    AglScopeError,
    BinderKind,
    BindingRef,
    BuiltinKind,
    ConstructorRef,
    DeclarationKey,
    ModuleResolution,
    PatternSlot,
    ScopeNode,
    ScopePath,
    SlotCandidate,
    alias_denotes_constructible_type,
    duplicate_binder_message,
    immutable_binder_phrase,
)
from agm.agl.semantics.engine_keys import RESERVED_PROGRAM_NAMES
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
    AgentDecl,
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
    OpenDecl,
    ParamDecl,
    Pattern,
    Placeholder,
    Program,
    ProgramDecl,
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
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
    pattern_binder_candidates,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import AppliedT, ImportMode, NameT
from agm.agl.syntax.visitor import walk

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

# Built-in call names: recognised in call position, not bindable as values.
# Sourced from ``symbols.BUILTIN_CALL_NAMES`` (the single source of truth).
_BUILTIN_CALL_NAMES = BUILTIN_CALL_NAMES

# Sentinel ``owner_decl_node_id`` for built-in constructor candidates (exceptions
# and prelude types), which have no source-level declaration node.  User
# declarations may shadow bindings carrying this sentinel.
_BUILTIN_CONSTRUCTOR_NODE_ID = -1

# The set of names that may NOT be used as any kind of binding.
_RESERVED_NAMES: frozenset[str] = frozenset(_BUILTIN_CALL_NAMES)


def _scope_path_sort_key(path: ScopePath) -> tuple[int, ScopePath]:
    """Order scope paths by depth, then lexical spelling."""
    return (len(path), path)


def _bare_atom(path: ScopePath) -> NameAtom:
    """Keep single bare names compact while retaining nested paths."""
    return path[0] if len(path) == 1 else path


def _bare_path(atom: NameAtom) -> ScopePath:
    """Normalize a bare contribution atom to its structured path."""
    return (atom,) if isinstance(atom, str) else atom


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------


class _Resolver:
    """Stateful resolver that builds the scope tree and resolution tables.

    Implements explicit ``isinstance`` dispatch for each node kind.
    Use ``resolve_module(program)`` — the public function — rather than instantiating
    this class directly.
    """

    def __init__(
        self,
        module_id: ModuleId = ENTRY_ID,
        import_env: ImportEnv | None = None,
        decl_info: dict[tuple[ModuleId, NameAtom], tuple[int, SourceSpan, BinderKind]]
        | None = None,
        cross_module_constructor_refs: Mapping[tuple[ModuleId, NameAtom], ConstructorRef]
        | None = None,
        cross_module_constructible_types: frozenset[tuple[ModuleId, NameAtom]] = frozenset(),
        cross_module_type_scopes: frozenset[tuple[ModuleId, NameAtom]] = frozenset(),
        all_public_types: dict[
            tuple[ModuleId, NameAtom], RecordDef | EnumDef | ExceptionDef | TypeAlias
        ]
        | None = None,
        is_entry: bool = True,
        repl_session_scope: ScopeNode | None = None,
        repl_session_scope_nodes: Mapping[ScopePath, ScopeNode] | None = None,
        repl_session_type_paths: frozenset[ScopePath] = frozenset(),
        origin_path: Path | None = None,
        spaced_qualifiers: tuple[SpacedQualifier, ...] = (),
    ) -> None:
        # Program parameters (None = module mode).
        self._module_id: ModuleId = module_id
        self._import_env: ImportEnv | None = import_env
        # Maps (module_id, name) → (node_id, span, kind) for cross-module refs.
        self._decl_info: dict[tuple[ModuleId, NameAtom], tuple[int, SourceSpan, BinderKind]] = (
            decl_info if decl_info is not None else {}
        )
        self._cross_module_constructor_refs: Mapping[tuple[ModuleId, NameAtom], ConstructorRef] = (
            cross_module_constructor_refs if cross_module_constructor_refs is not None else {}
        )
        self._cross_module_constructible_types = cross_module_constructible_types
        # Public type declarations establish scope paths even when they have
        # no separately public child members.
        self._cross_module_type_scopes = cross_module_type_scopes
        # Whole-program public-type table (program context only), used to
        # follow a type alias's target across an import when deciding whether
        # the alias has a variant-less constructor. ``None`` in module mode.
        self._all_public_types: (
            dict[tuple[ModuleId, NameAtom], RecordDef | EnumDef | ExceptionDef | TypeAlias] | None
        ) = all_public_types
        # Whether this module is the entry module (program context only).
        self._is_entry: bool = is_entry
        # Optional REPL session scope for ``::name`` self-ref fallback.
        # When set, ``_lookup_own_root`` falls back to this scope for names not
        # in the entry's own root scope, allowing ``::name`` to resolve to a
        # prior session binding.
        self._repl_session_scope: ScopeNode | None = repl_session_scope
        # Named scope layers retained by prior REPL entries. They are copied
        # into this entry's fresh resolver image and committed only after a
        # successful evaluation, so a failed entry cannot mutate session state.
        self._repl_session_scope_nodes = dict(repl_session_scope_nodes or {})
        self._repl_session_type_paths = repl_session_type_paths
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
        # Scope stack — top is the current scope.
        self._scope: ScopeNode | None = None
        # The module's root ScopeNode (set in run()); used by _lookup_own_root
        # to bypass lexical shadows introduced by nested scopes for ::name.
        self._root_scope: ScopeNode | None = None
        # Whether we are at the program root (for root-only checks).
        self._at_root: bool = False
        # Agents keyed by their structured declaration paths.
        self._declared_agents: dict[AgentKey, AgentDecl] = {}
        # Ambient agents from the host (not in declared_agents).
        self._ambient_agents: frozenset[str] = frozenset()
        # Top-level function defs.
        self._declared_functions: dict[str, FuncDef] = {}
        # Program-declared agent identities referenced as a VarRef.
        self._referenced_agents: set[AgentKey] = set()
        # Header-only tracking for imports in non-entry modules (program context).
        self._seen_non_import_item: bool = False
        # Source-declared program name.
        self._program_name: str | None = None
        # Names of all root-level type declarations (RecordDef/EnumDef/TypeAlias).
        self._declared_type_names: set[str] = set()
        # Named declarations are keyed uniformly by module and scope path; the
        # root is simply the empty path.
        self._declarations: dict[DeclarationKey, BindingRef] = {}
        # The declaration item accompanies its path-keyed binding so legacy
        # root-only tables can be derived from the same collection without
        # admitting scoped members.
        self._declaration_items: dict[
            DeclarationKey, FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias | AgentDecl
        ] = {}
        self._scope_entity_kinds: dict[DeclarationKey, str] = {}
        # Scoped externs are implemented by their member names in one module
        # companion, so distinct scope paths cannot share a Python symbol.
        self._scoped_extern_symbols: dict[str, FuncDef] = {}
        self._scope_paths: set[ScopePath] = {(), *self._repl_session_scope_nodes}
        self._scope_node_ids: dict[ScopePath, int] = {
            path: node.node_id for path, node in self._repl_session_scope_nodes.items() if path
        }
        self._type_paths: set[ScopePath] = set(self._repl_session_type_paths)
        self._type_declarations: list[
            tuple[RecordDef | EnumDef | ExceptionDef | TypeAlias, ScopePath]
        ] = []
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        program: Program,
        *,
        parent_scope: ScopeNode | None = None,
        ambient_agents: frozenset[str] = frozenset(),
        ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
        ambient_type_names: frozenset[str] = frozenset(),
    ) -> ModuleResolution:
        """Execute the resolution pass over *program*.

        When *parent_scope* is given, the entry's root scope is parented to it
        so name lookups fall through to session bindings (incremental REPL
        sessions).  New declarations live in the entry's own root scope and
        shadow parent bindings without a duplicate-declaration error.

        *ambient_agents* are agent names the host already backs.  They count
        as valid call targets alongside this program's own ``agent``
        declarations, but never appear in ``ModuleResolution.declared_agents``
        and never trigger an unused-agent warning.

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

        # Pre-pass 1: collect every named declaration and scope mention. This
        # establishes path-keyed membership before qualifier validation and the
        # legacy root worker build their compatibility tables.
        self._collect_declarations(program)
        self._validate_extern_backing()

        # Pre-pass 2: collect path-keyed agent declarations before body
        # resolution; root bindings are installed below and scoped bindings
        # live in their member layers.
        self._ambient_agents = ambient_agents
        self._collect_agent_decls(program)
        # Pre-pass 3: collect top-level def names for mutual recursion.
        self._collect_func_decls(program)
        # Pre-pass 4: collect type-declaration names and validate type_params.
        self._collect_type_decl_names(program)
        # Pre-pass 5: collect constructor candidates from RecordDef/EnumDef.
        self._collect_constructor_candidates(program)

        root = ScopeNode(node_id=program.node_id, parent=parent_scope, scope_path=())
        self._push_scope(root)
        self._root_scope = root
        self._scope_nodes = self._build_scope_nodes(root)
        self._at_root = True

        # Define root agents and functions as value bindings; scoped members
        # are already present in their named-scope layers.
        self._define_agent_bindings()
        self._define_ambient_agent_bindings()
        self._define_function_bindings()
        # Define constructor bindings in root scope.
        self._define_constructor_bindings()

        # Main walk: resolve all block items in order.
        self._resolve_block_items(program.body.items)

        self._at_root = False
        self._pop_scope()

        return ModuleResolution(
            program=program,
            resolution=self._resolution,
            builtin_calls=self._builtin_calls,
            root_scope=root,
            declarations=dict(self._declarations),
            scope_nodes=dict(self._scope_nodes),
            declared_agents=dict(self._declared_agents),
            declared_functions=dict(self._declared_functions),
            program_name=self._program_name,
            warnings=self._unused_agent_warnings(),
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
            pattern_slots=dict(self._pattern_slots),
            match_site_pattern_slots=dict(self._match_site_pattern_slots_by_node),
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
        if isinstance(item, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias, AgentDecl)):
            path = tuple(segment.name for segment in item.scope_path)
            self._ensure_scope_path(path, item.node_id, item.span)
            self._register_declaration(item, path)

    def _ensure_scope_path(self, path: ScopePath, node_id: int, span: SourceSpan) -> None:
        """Create every scope layer in *path*, rejecting ordinary-name clashes."""
        for index, name in enumerate(path):
            parent_path = path[:index]
            scope_path = path[: index + 1]
            key = (self._module_id, parent_path, name)
            existing = self._scope_entity_kinds.get(key)
            if existing == "ordinary":
                raise AglScopeError(f"Name '{name}' is already declared in this scope.", span=span)
            self._scope_entity_kinds.setdefault(key, "scope")
            self._scope_paths.add(scope_path)
            self._scope_node_ids.setdefault(scope_path, node_id)

    def _register_declaration(
        self,
        item: FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias | AgentDecl,
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

        kind = (
            BinderKind.constructor_binding
            if is_type
            else BinderKind.function_binding
            if isinstance(item, FuncDef)
            else BinderKind.agent_binding
        )
        self._declarations[key] = BindingRef(
            name=item.name,
            mutable=False,
            decl_span=item.span,
            decl_node_id=item.node_id,
            kind=kind,
            module_id=self._module_id,
            scope_path=path,
        )
        self._declaration_items[key] = item
        if isinstance(item, FuncDef):
            self._validate_function_decl(item)
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
        elif isinstance(item, AgentDecl):
            self._validate_agent_decl(item)
        else:
            self._validate_type_params(item)
        if not is_type:
            return

        type_item = cast(RecordDef | EnumDef | ExceptionDef | TypeAlias, item)
        self._type_declarations.append((type_item, path))
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

    def _build_scope_nodes(self, root: ScopeNode) -> dict[ScopePath, ScopeNode]:
        """Build named-scope layers and populate their declaration memberships."""
        nodes: dict[ScopePath, ScopeNode] = {(): root}
        for path in sorted(self._scope_paths, key=_scope_path_sort_key):
            if not path:
                continue
            retained = self._repl_session_scope_nodes.get(path)
            nodes[path] = ScopeNode(
                node_id=self._scope_node_ids[path],
                parent=nodes[path[:-1]],
                scope_path=path,
                members={} if retained is None else dict(retained.members),
            )
        # A replacement type declaration owns a fresh member layer: stale enum
        # variants from its prior definition must not survive into this entry.
        for item, path in self._type_declarations:
            nodes[path + (item.name,)].members.clear()
        for (_module_id, path, name), declaration in self._declarations.items():
            nodes[path].members[name] = declaration
        return nodes

    def _root_declaration_items(
        self,
        declaration_type: type[FuncDef]
        | type[AgentDecl]
        | type[RecordDef]
        | type[EnumDef]
        | type[ExceptionDef]
        | type[TypeAlias],
    ) -> Iterator[FuncDef | AgentDecl | RecordDef | EnumDef | ExceptionDef | TypeAlias]:
        """Yield collected declarations of *declaration_type* at the root path."""
        for (module_id, path, _name), item in self._declaration_items.items():
            if module_id == self._module_id and not path and isinstance(item, declaration_type):
                yield item

    def _collect_agent_decls(self, program: Program) -> None:
        """Collect agents from every static scope by declaration identity."""
        for item in self._declaration_items.values():
            if isinstance(item, AgentDecl):
                path = tuple(segment.name for segment in item.scope_path)
                self._declared_agents[(path, item.name)] = item

    def _validate_agent_decl(self, decl: AgentDecl) -> None:
        """Apply agent declaration validation independently of its scope path."""
        if decl.name in _RESERVED_NAMES:
            raise AglScopeError(
                f"'{decl.name}' is built-in and cannot be declared as an agent.",
                span=decl.span,
            )

    def _collect_func_decls(self, program: Program) -> None:
        """Populate the legacy function table from root-path declarations only."""
        for item in self._root_declaration_items(FuncDef):
            assert isinstance(item, FuncDef)
            self._declared_functions[item.name] = item

    def _validate_function_decl(self, decl: FuncDef) -> None:
        """Apply function declaration validation independently of its scope path."""
        if decl.name in _RESERVED_NAMES and not decl.is_builtin:
            raise AglScopeError(
                f"'{decl.name}' is a built-in name and cannot be used as a function name.",
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
        """Raise AglScopeError if *decl* has duplicate type-parameter names."""
        seen: set[str] = set()
        for tp in decl.type_params:
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
                owner_module_id=PRELUDE_ID,
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
                            owner_module_id=PRELUDE_ID,
                            can_match_bare_pattern=not _vfields,
                        )
                        self._add_constructor_candidate(variant_name, cref)
            else:
                cref = ConstructorRef(
                    owner_name=type_name,
                    variant=None,
                    owner_decl_node_id=_BUILTIN_CONSTRUCTOR_NODE_ID,
                    type_params=(),
                    owner_module_id=PRELUDE_ID,
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
        """Add *cref* to the candidates list for *ctor_key*.

        Skips the entry if another candidate with the same ``owner_name`` is
        already present (duplicate type declaration — the type-builder pass will
        raise a clear "already declared" error for that; we must not conflate it
        with genuine constructor overloading across distinct types).
        """
        scoped_key = (scope_path, ctor_key)
        scoped_existing = self._scoped_constructor_candidates.get(scoped_key, [])
        if not any(
            (candidate.owner_module_id, candidate.owner_path, candidate.owner_name)
            == (cref.owner_module_id, cref.owner_path, cref.owner_name)
            for candidate in scoped_existing
        ):
            scoped_existing.append(cref)
            self._scoped_constructor_candidates[scoped_key] = scoped_existing
        if not inject_bare:
            return
        existing = self._constructor_candidates.get(ctor_key, [])
        if any(
            (c.owner_module_id, c.owner_name) == (cref.owner_module_id, cref.owner_name)
            or (
                (
                    c.owner_decl_node_id == _BUILTIN_CONSTRUCTOR_NODE_ID
                    or cref.owner_decl_node_id == _BUILTIN_CONSTRUCTOR_NODE_ID
                )
                and c.owner_name == cref.owner_name
            )
            for c in existing
        ):
            return
        existing.append(cref)
        self._constructor_candidates[ctor_key] = existing

    def _alias_target_lookup(
        self, name: str, qualifier: QualifierChain | None, *, scope_path: ScopePath = ()
    ) -> RecordDef | EnumDef | ExceptionDef | TypeAlias | None:
        """Resolve one alias-target type reference for :func:`alias_denotes_constructible_type`.

        Tries this module's own declarations first — the alias's own scope path,
        then the module root (only for an unqualified name, since a qualified
        reference can never name a local declaration) — then falls back to the
        shared import-environment resolution
        (:func:`~agm.agl.scope.imports.resolve_alias_target`), which covers a
        target reached through an open import or a module qualifier. In module
        mode (no import environment or public-types table), only the local step
        ever succeeds.
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

    def _define_agent_bindings(self) -> None:
        """Define root agents; scoped agents already live in scope memberships."""
        for (_path, name), decl in self._declared_agents.items():
            if decl.scope_path:
                continue
            ref = BindingRef(
                name=name,
                mutable=False,
                decl_span=decl.span,
                decl_node_id=decl.node_id,
                kind=BinderKind.agent_binding,
                module_id=self._module_id,
            )
            self._current_scope().define(name, ref)

    def _define_ambient_agent_bindings(self) -> None:
        """Define ambient agent names as value bindings so VarRefs resolve.

        Ambient agents come from the host (e.g. earlier REPL entries) and are
        NOT in ``_declared_agents``; they are never reported in
        ``declared_agents`` and never trigger unused-agent warnings.

        We use a synthetic span pointing to the program root node and the root
        node_id as the decl_node_id (no real declaration AST node exists).
        Only define if not already defined (declared agents take precedence).
        """
        scope = self._current_scope()
        for name in self._ambient_agents:
            if name in scope.bindings:
                continue  # Already defined by a local agent declaration
            # Use a sentinel span/node_id — the parent scope may have a real
            # binding for this name (REPL session), so only add if absent.
            if scope.lookup(name) is not None:
                continue
            # Create a synthetic binding ref for the ambient agent.
            synthetic_span = SourceSpan(
                start_line=0,
                start_col=0,
                end_line=0,
                end_col=0,
                start_offset=0,
                end_offset=0,
            )
            ref = BindingRef(
                name=name,
                mutable=False,
                decl_span=synthetic_span,
                decl_node_id=-1,
                kind=BinderKind.agent_binding,
                module_id=self._module_id,
            )
            scope.define(name, ref)

    def _define_function_bindings(self) -> None:
        """Define each collected function as a value binding in the current scope."""
        for name, decl in self._declared_functions.items():
            if name in self._current_scope().bindings:
                # An ambient agent with the same name as this def was defined
                # first (_define_ambient_agent_bindings runs before this method).
                raise AglScopeError(
                    f"Name '{name}' is already declared in this scope.",
                    span=decl.span,
                )
            ref = BindingRef(
                name=name,
                mutable=False,
                decl_span=decl.span,
                decl_node_id=decl.node_id,
                kind=BinderKind.function_binding,
                module_id=self._module_id,
            )
            self._current_scope().define(name, ref)

    # ------------------------------------------------------------------
    # Unused-agent warnings
    # ------------------------------------------------------------------

    def _unused_agent_warnings(self) -> tuple[Diagnostic, ...]:
        """Warn for each program-declared agent never referenced."""
        warnings: list[Diagnostic] = []
        for key, decl in self._declared_agents.items():
            if key not in self._referenced_agents:
                warnings.append(
                    Diagnostic(
                        message=f"agent '{decl.name}' is declared but never called.",
                        line=decl.span.start_line,
                        column=decl.span.start_col,
                        end_line=decl.span.end_line,
                        end_column=decl.span.end_col,
                        severity="warning",
                    )
                )
        return tuple(warnings)

    def _resolve_builtin_var(self, node: BuiltinVarDecl) -> None:
        """Resolve a standard-library engine setting into a mutable register binding.

        ``builtin var`` is reserved to the canonical ``std/config`` module.
        The typecheck pass then validates that the name is a known engine key
        with its canonical type.
        """
        if not self._at_root or self._current_scope().scope_path:
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

        A declaration may claim the bare spelling of a constructor declared in
        another module (e.g. the prelude ``Retry``/``ExecResult`` names) or of a
        same-module enum variant, since ``Owner::variant`` and a qualified
        module path still reach those.  A same-module record, exception, or
        alias constructor declares the bare name itself, so it collides.
        """
        scope = self._current_scope()
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

    def _resolve_block_items(self, items: tuple[Item, ...]) -> None:
        """Resolve items in order; each binder adds to the current scope.

        This is the core sequencing logic.  Binders (``LetDecl``, ``VarDecl``,
        ``AssignStmt``) and declarations (``FuncDef``, ``AgentDecl``, etc.) that
        are not pure expressions are handled first; everything else is treated
        as an expression item.

        In program context (``_import_env is not None``), additional enforcement:

        - Non-entry modules: only ``FuncDef``, ``RecordDef``, ``EnumDef``,
          ``TypeAlias``, ``InfixDecl``, ``ImportDecl``, and ``ExportDecl`` are
          allowed at the module root.
          ``LetDecl``, ``VarDecl``, ``AssignStmt``, bare expressions, and
          entry-only constructs (``AgentDecl``, ``ParamDecl``, ``ProgramDecl``)
          are rejected with a scope error.
        - Non-entry modules: ``ImportDecl`` and ``ExportDecl`` must precede all declarations
          (header-only; ``_seen_non_import_item`` tracks this).
        """
        has_import_context = self._import_env is not None
        is_non_entry_root = has_import_context and not self._is_entry and self._at_root

        for item in items:
            if isinstance(item, OpenDecl):
                self._resolve_open_decl(item)
                continue
            if isinstance(item, (ImportDecl, ExportDecl)):
                if not self._at_root:
                    kind = "import" if isinstance(item, ImportDecl) else "export"
                    raise AglScopeError(
                        f"'{kind}' declarations are only allowed at the program root, "
                        "not inside a nested block.",
                        span=item.span,
                    )
                if is_non_entry_root and self._seen_non_import_item:
                    raise AglScopeError(
                        "Import and export declarations must appear before any other "
                        "declarations in a library module.",
                        span=item.span,
                    )
                # The program module-system pass processes imports/exports; this pass skips them.
                continue
            if isinstance(item, InfixDecl):
                if not self._at_root or self._current_scope().scope_path:
                    raise AglScopeError(
                        "infix declarations are only allowed at the program root.",
                        span=item.span,
                    )
                if is_non_entry_root:
                    self._seen_non_import_item = True
                continue
            # Non-entry enforcement: track that a non-import item has been seen.
            if is_non_entry_root:
                self._seen_non_import_item = True
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
            elif isinstance(item, AgentDecl):
                if is_non_entry_root:
                    raise AglScopeError(
                        f"'agent' declarations are only allowed in the entry module, "
                        f"not in library modules (found 'agent {item.name}' here).",
                        span=item.span,
                    )
                self._resolve_agent_decl(item)
            elif isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                self._resolve_type_decl(item)
            elif isinstance(item, LetDecl):
                if is_non_entry_root:
                    raise AglScopeError(
                        "Library modules may only contain declarations "
                        "('def', 'record', 'enum', 'type', 'import'); "
                        "'let' bindings are not allowed at the top level of a "
                        "library module.",
                        span=item.span,
                    )
                self._resolve_let(item)
            elif isinstance(item, VarDecl):
                if is_non_entry_root:
                    raise AglScopeError(
                        "Library modules may only contain declarations "
                        "('def', 'record', 'enum', 'type', 'import'); "
                        "'var' bindings are not allowed at the top level of a "
                        "library module.",
                        span=item.span,
                    )
                self._resolve_var(item)
            elif isinstance(item, AssignStmt):
                if is_non_entry_root:
                    raise AglScopeError(
                        "Library modules may only contain declarations; "
                        "assignment statements are not allowed at the top level.",
                        span=item.span,
                    )
                self._resolve_assign(item)
            elif isinstance(item, ParamDecl):
                if is_non_entry_root:
                    raise AglScopeError(
                        "'param' declarations are only allowed in the entry module, "
                        f"not in library modules (found 'param {item.name}' here).",
                        span=item.span,
                    )
                self._resolve_param(item)
            elif isinstance(item, ProgramDecl):
                if is_non_entry_root:
                    raise AglScopeError(
                        "'program' declarations are only allowed in the entry module, "
                        f"not in library modules (found 'program {item.name}' here).",
                        span=item.span,
                    )
                self._resolve_program_decl(item)
            else:
                # Pure expression item (Expr union).
                if is_non_entry_root:
                    raise AglScopeError(
                        "Library modules may only contain declarations "
                        "('def', 'record', 'enum', 'type', 'import'); "
                        "bare expressions are not allowed at the top level of a "
                        "library module.",
                        span=item.span,
                    )
                self._resolve_expr(item)

    # ------------------------------------------------------------------
    # Declaration handlers
    # ------------------------------------------------------------------

    def _resolve_open_decl(self, decl: OpenDecl) -> None:
        """Contribute a selected scope subtree to the current region."""
        target = self._open_target_members(decl)
        selected: list[tuple[NameAtom, BindingRef, ConstructorRef | None]] = []
        if decl.mode is ImportMode.ALL:
            selected = [(atom, ref, constructor) for atom, (ref, constructor) in target.items()]
        else:
            selected_paths: set[ScopePath] = set()
            for item in decl.items:
                prefix = (*tuple(segment.name for segment in item.scope_path), item.name)
                matches = [atom for atom in target if _bare_path(atom)[: len(prefix)] == prefix]
                if not matches:
                    raise AglScopeError(
                        f"'{'::'.join(prefix)}' is not a public member of the opened scope.",
                        span=item.span,
                    )
                selected_paths.add(prefix)
            if decl.mode is ImportMode.USING:
                for item in decl.items:
                    prefix = (*tuple(segment.name for segment in item.scope_path), item.name)
                    for atom, value in target.items():
                        path = _bare_path(atom)
                        if path[: len(prefix)] == prefix:
                            exposed = _bare_atom(
                                (item.rename, *path[len(prefix) :])
                                if item.rename is not None
                                else path
                            )
                            ref, constructor = value
                            selected.append((exposed, ref, constructor))
            else:
                selected = [
                    (atom, ref, constructor)
                    for atom, (ref, constructor) in target.items()
                    if not any(
                        _bare_path(atom)[: len(prefix)] == prefix for prefix in selected_paths
                    )
                ]

        for atom, ref, constructor in selected:
            self._current_scope().contribute_bare(atom, ref)
            if constructor is not None:
                self._current_scope().contribute_bare_constructor(atom, constructor)

    def _open_target_members(
        self, decl: OpenDecl
    ) -> dict[NameAtom, tuple[BindingRef, ConstructorRef | None]]:
        """Return the relative public subtree of an opened local or imported scope."""
        requested_path = tuple(segment.name for segment in decl.scope_ref.scope_path)
        # An explicit module route (``open geo/shapes::Point``) names its target
        # unambiguously; a same-named local scope must not intercept it.
        local_path = (
            None if decl.scope_ref.module_route else self._open_local_scope_path(requested_path)
        )
        if local_path is not None:
            local_members: dict[NameAtom, tuple[BindingRef, ConstructorRef | None]] = {}
            for path, node in self._scope_nodes.items():
                if path[: len(local_path)] != local_path:
                    continue
                for name, ref in node.members.items():
                    relative = (*path[len(local_path) :], name)
                    atom = _bare_atom(relative)
                    local_members[atom] = (
                        ref,
                        next(iter(self._scoped_constructor_candidates.get((path, name), ())), None),
                    )
            return local_members

        route = decl.scope_ref.module_route
        scope_path = requested_path
        if not route and len(requested_path) > 1:
            route, scope_path = (requested_path[0],), requested_path[1:]
        if not route and self._import_env is not None:
            # A prior ``open import`` / ``using`` already contributed this
            # scope's members bare; reach the source scope they came from
            # before giving up on an explicit route.
            bare_members = self._open_bare_imported_scope_members(
                requested_path, decl.scope_ref.span
            )
            if bare_members is not None:
                return bare_members
        if self._import_env is None or not route or not scope_path:
            raise AglScopeError(
                f"Unknown scope path '{'::'.join(requested_path)}'.", span=decl.scope_ref.span
            )

        matching: list[dict[NameAtom, tuple[BindingRef, ConstructorRef | None]]] = []
        for module in qualifier_candidates(self._import_env, route, anchored=False):
            contribution = self._import_env.contributions[module]
            imported_members, is_public_type_scope = self._scope_subtree_members(
                contribution.members.items(), scope_path, decl.span
            )
            if imported_members or is_public_type_scope:
                matching.append(imported_members)
        if not matching:
            rendered = "::".join(scope_path)
            raise AglScopeError(
                f"Unknown or non-public scope '{rendered}'.", span=decl.scope_ref.span
            )
        if len(matching) > 1:
            raise AglScopeError(
                f"Opened scope '{'::'.join(scope_path)}' is ambiguous across imported modules.",
                span=decl.scope_ref.span,
            )
        return matching[0]

    def _scope_subtree_members(
        self,
        entries: Iterable[tuple[NameAtom, QName]],
        scope_path: ScopePath,
        span: SourceSpan,
    ) -> tuple[dict[NameAtom, tuple[BindingRef, ConstructorRef | None]], bool]:
        """Materialize one contribution's members below *scope_path*, relative to it.

        Returns the relative-atom member dict and whether *scope_path* itself
        names a public type scope with no separately public child members
        (still a valid, if empty, selection).
        """
        members: dict[NameAtom, tuple[BindingRef, ConstructorRef | None]] = {}
        is_public_type_scope = False
        for exposed, qname in entries:
            path = _bare_path(exposed)
            if path == scope_path and qname in self._cross_module_type_scopes:
                is_public_type_scope = True
                continue
            if path[: len(scope_path)] != scope_path or len(path) == len(scope_path):
                continue
            atom = _bare_atom(path[len(scope_path) :])
            constructor = self._cross_module_constructor_refs.get(qname)
            ref = self._make_cross_module_ref(qname[0], _bare_path(atom)[-1], qname[1], span)
            if constructor is not None:
                ref = replace(
                    ref,
                    decl_node_id=constructor.owner_decl_node_id,
                    kind=BinderKind.constructor_binding,
                    scope_path=constructor.owner_path,
                )
            members[atom] = (ref, constructor)
        return members, is_public_type_scope

    def _open_bare_imported_scope_members(
        self, requested_path: ScopePath, span: SourceSpan
    ) -> dict[NameAtom, tuple[BindingRef, ConstructorRef | None]] | None:
        """Resolve an unrouted ``open`` target against already-bare imports.

        ``open import lib`` and ``import lib using A`` both inject a module's
        selected members bare (see ``ImportEnv``/``build_import_env``); a
        later ``open A`` names the source scope those bare members came from,
        not an individual member, so it is resolved against every
        contribution's bare-exposed names rather than a module route.
        """
        assert self._import_env is not None
        matching: list[dict[NameAtom, tuple[BindingRef, ConstructorRef | None]]] = []
        for contribution in self._import_env.contributions.values():
            entries = (
                (exposed, contribution.members[exposed]) for exposed in contribution.bare_names
            )
            members, _is_public_type_scope = self._scope_subtree_members(
                entries, requested_path, span
            )
            if members:
                matching.append(members)
        if not matching:
            return None
        if len(matching) > 1:
            raise AglScopeError(
                f"Opened scope '{'::'.join(requested_path)}' is ambiguous across imported modules.",
                span=span,
            )
        return matching[0]

    def _open_local_scope_path(self, requested_path: ScopePath) -> ScopePath | None:
        """Find an exact lexical scope target for an ``open`` declaration."""
        scope: ScopeNode | None = self._current_scope()
        while scope is not None:
            candidate = scope.scope_path + requested_path
            if candidate in self._scope_nodes:
                return candidate
            scope = scope.parent
        return None

    def _resolve_scope_region(self, region: ScopeRegion) -> None:
        """Resolve a named region in its member layer."""
        if not self._is_entry:

            def reject_agent(node: object) -> None:
                if isinstance(node, AgentDecl):
                    raise AglScopeError(
                        "'agent' declarations are only allowed in the entry module, "
                        f"not in library modules (found 'agent {node.name}' here).",
                        span=node.span,
                    )

            walk(region, reject_agent)
        path = self._current_scope().scope_path + (region.segment.name,)
        with self._named_scope(path):
            self._resolve_block_items(cast(tuple[Item, ...], region.items))

    def _resolve_agent_decl(self, node: AgentDecl) -> None:
        """Reject an unscoped agent declaration nested in an ordinary block."""
        if node.scope_path:
            return
        if not self._at_root:
            raise AglScopeError(
                f"'agent' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'agent {node.name}' here).",
                span=node.span,
            )

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
        self._resolve_params_and_body(node)

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

    def _resolve_let(self, node: LetDecl) -> None:
        # A let initializer is non-recursive: no part of its pattern exists
        # until its RHS has resolved. A simple name is still a pattern binder;
        # only its slot handling differs from a broader pattern.
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
            # In program context with import_env, try structured open imports as fallback,
            # so a bare target reaches an imported mutable binding just like a read.
            if ref is None and self._import_env is not None:
                ref = self._lookup_import_env_unqualified(name, node.span)
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

    def _resolve_qualified_assign(self, node: AssignStmt, target: NameTarget) -> None:
        """Resolve a qualified assignment target (``MODQUAL::name := expr``).

        Only ``builtin var`` bindings are assignable across a module boundary
        (they are the sole mutable exported bindings); assigning any other
        cross-module binding is rejected as immutable.
        """
        assert target.qualifier is not None
        qualifier = target.qualifier
        if self._import_env is None or not qualifier.route_segments:
            raise AglScopeError(
                f"'{target.name}' is not declared; assignment requires an existing "
                f"mutable binding.",
                span=node.span,
            )
        ref = self._lookup_qualified_binding(qualifier, target.name, node.span)
        if not ref.mutable:
            raise AglScopeError(
                f"Cannot assign to '{target.name}': "
                f"{immutable_binder_phrase(ref.kind)} (immutable).",
                span=node.span,
            )
        self._resolution[node.node_id] = ref
        self._resolve_expr(node.value)

    def _resolve_param(self, node: ParamDecl) -> None:
        if not self._at_root:
            raise AglScopeError(
                f"'param' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'param {node.name}' here).",
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

    def _resolve_program_decl(self, node: ProgramDecl) -> None:
        if not self._at_root:
            raise AglScopeError(
                f"'program' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'program {node.name}' here).",
                span=node.span,
            )
        if node.name in RESERVED_PROGRAM_NAMES:
            raise AglScopeError(
                f"'program {node.name}' is not allowed: '{node.name}' is a reserved AGM "
                f"command or config-section name.",
                span=node.span,
            )
        if self._program_name is not None:
            raise AglScopeError(
                f"'program' is already declared as '{self._program_name}'; "
                "at most one 'program' declaration is allowed per program.",
                span=node.span,
            )
        self._program_name = node.name

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

    def _resolve_varref(self, node: VarRef) -> None:
        """Resolve a name reference.

        When a VarRef resolves to a constructor_binding, look up the candidate set:
        - Exactly 1 candidate → record in constructor_refs.
        - ≥ 2 candidates → ambiguity error.

        In program context (when ``_import_env`` is set), a bare reference uses
        lexical scope then open imports; a current-module chain uses the own
        root scope; and a module-route chain uses qualified cross-module access.
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
            self._record_varref_binding(node, ref)
            return
        if self._resolve_local_scope_member(node):
            return
        if node.qualifier is not None:
            self._resolve_qualified_chain(node)
            return
        if node.name in _BUILTIN_CALL_NAMES and self._current_scope().lookup(node.name) is None:
            raise AglScopeError(
                f"Built-in function '{node.name}' cannot be used as a value.", span=node.span
            )

        # Standard lexical lookup (module mode or bare name in program context).
        ref = self._current_scope().lookup(node.name)
        regional_candidates = self._current_scope().bare_constructor_candidates(node.name)
        if ref is None or (
            ref.kind is BinderKind.constructor_binding and regional_candidates is not None
        ):
            contributed = self._lookup_bare_contribution(node.name, node.span)
            if contributed is not None:
                ref = contributed
        if ref is None:
            # In program context with import_env, try structured open imports as fallback.
            if self._import_env is not None:
                ref = self._lookup_import_env_unqualified(node.name, node.span)
            if ref is None:
                raise self._spaced_qualifier_repair(
                    self._spaced_qualifier_around(node.span), node.span
                ) or AglScopeError(
                    f"'{node.name}' is not defined.",
                    span=node.span,
                )
        self._record_varref_binding(node, ref, candidates=regional_candidates)

    def _record_varref_binding(
        self,
        node: VarRef,
        ref: BindingRef,
        *,
        candidates: Collection[ConstructorRef] | None = None,
    ) -> None:
        """Record an ordinary value binding and its constructor metadata."""
        self._track_agent_reference(ref)
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

    def _declaring_constructor_candidates(
        self, name: str, ref: BindingRef
    ) -> tuple[ConstructorRef, ...]:
        """Return the candidates declared where *ref* was found.

        A member of a named scope is selected by that scope's structured
        identity; only a module-root binding uses the root-only candidate map.
        """
        if ref.scope_path and ref.module_id == self._module_id:
            return tuple(self._scoped_constructor_candidates.get((ref.scope_path, name), ()))
        return tuple(self._constructor_candidates.get(name, ()))

    def _track_agent_reference(self, ref: BindingRef) -> None:
        """Preserve agent usage hidden behind constructor-selected pattern slots."""
        while ref.kind is BinderKind.pattern_slot:
            assert ref.slot_id is not None
            alternative = self._pattern_slots[ref.slot_id].alternative
            if alternative is None:
                return
            ref = alternative
        agent_key = (ref.scope_path, ref.name)
        if ref.kind is BinderKind.agent_binding and agent_key in self._declared_agents:
            self._referenced_agents.add(agent_key)

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
            has_module_route = self._import_env is not None and bool(
                qualifier_candidates(self._import_env, relative_path, anchored=chain.anchored)
            )
            # A route needs only its leading segment to be real: `lib::A::f()`
            # is a genuine import route through `lib` even though `A::f` (or a
            # longer suffix) does not itself name a complete module route.
            has_leading_route = self._import_env is not None and bool(
                qualifier_candidates(self._import_env, relative_path[:1], anchored=chain.anchored)
            )
            has_opened_member = (
                self._current_scope().bare_candidates(_bare_atom((*relative_path, node.name)))
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
        if (
            chain.anchor is None
            and self._import_env is not None
            and qualifier_contributes(self._import_env, relative_path, node.name)
        ):
            local_kind = "a type name" if path in self._type_paths else "a local scope"
            raise AglScopeError(
                f"Qualifier '{relative_path[0]}' is both {local_kind} and a module route for "
                f"'{node.name}'. {qualification_repair_guidance()}",
                span=chain.span,
            )
        if ref.kind is BinderKind.constructor_binding:
            candidates = self._scoped_constructor_candidates.get((path, node.name), ())
            if len(candidates) == 1:
                self._constructor_refs[node.node_id] = candidates[0]
                return True
            return False
        self._track_agent_reference(ref)
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
        if chain.anchor is None:
            atom = _bare_atom((*tuple(segment.name for segment in chain.segments), node.name))
            ref = self._lookup_bare_contribution(atom, node.span)
            if ref is not None:
                self._record_varref_binding(
                    node,
                    ref,
                    candidates=self._current_scope().bare_constructor_candidates(atom),
                )
                return
        direct_error: AglScopeError | None = None
        local_type_path = self._validate_local_scope_chain(chain)
        if (
            self._import_env is not None
            and chain.anchor is not QualifierAnchor.CURRENT_MODULE
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
                    route_is_complete = all(
                        self._import_env.contributions[module].complete_public_set
                        for module in qualifier_candidates(
                            self._import_env, route, anchored=chain.anchored
                        )
                    )
                    if owner in self._cross_module_constructible_types and not route_is_complete:
                        raise
        if self._resolve_constructor_chain(node.node_id, chain, node.name):
            return
        if direct_error is not None:
            raise direct_error
        if (
            chain.anchor is QualifierAnchor.CURRENT_MODULE
            and len(chain.segments) == 1
            and not any(
                candidate.owner_name == chain.segments[0].name
                and candidate.owner_module_id in (self._module_id, PRELUDE_ID)
                for candidates in self._constructor_candidates.values()
                for candidate in candidates
            )
        ):
            segment = chain.segments[0]
            missing_error = AglScopeError(
                f"'{segment.name}' is not defined in this module.", span=segment.span
            )
            raise (
                self._spaced_qualifier_repair(
                    self._spaced_qualifier_at(chain.span)
                    or self._spaced_qualifier_around(node.span),
                    node.span,
                )
                or missing_error
            )

        if len(chain.segments) == 1 and chain.segments[0].type_args is not None:
            raise AglScopeError(
                f"'{chain.segments[0].name}' is not a known type.", span=chain.segments[0].span
            )
        qualifier_str = chain.render()
        raise AglScopeError(
            f"No module imported under qualifier '{qualifier_str}' or type named "
            f"'{qualifier_str}'.",
            span=chain.span,
        )

    def _resolve_constructor_chain(
        self,
        node_id: int,
        chain: QualifierChain | None,
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
        if chain is None:
            return False
        local_path = self._validate_local_scope_chain(chain)
        if local_path in self._type_paths:
            if (
                chain.anchor is None
                and self._import_env is not None
                and qualifier_contributes(
                    self._import_env,
                    tuple(segment.name for segment in chain.segments),
                    variant,
                )
            ):
                rendered = "::".join(segment.name for segment in chain.segments)
                if defer_route_diagnostics:
                    return False
                raise AglScopeError(
                    f"Qualifier '{rendered}' is both a type name and a module route for "
                    f"'{variant}'. {qualification_repair_guidance()}",
                    span=chain.span,
                )
            self._constructor_refs[node_id] = self._constructor_for_type_path(local_path, variant)
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
                        or candidate.owner_module_id in (self._module_id, PRELUDE_ID)
                    )
                ),
                None,
            )
            if candidate is not None:
                self._constructor_refs[node_id] = replace(candidate, variant=variant)
                return True
        return False

    def _constructor_for_type_path(self, path: ScopePath, variant: str) -> ConstructorRef:
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
            assert retained is not None
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

    def _imported_chain_owner(self, chain: QualifierChain, variant: str) -> ConstructorRef | None:
        """Resolve a type-owning segment reached through imports.

        The final chain segment is selected as a normal imported member; any
        preceding segments are its route.  A one-segment chain may instead
        name an open-imported type.  This is the same chain walk used for
        ordinary qualified values, with only the resulting member kind
        determining whether it owns a constructor.
        """
        if chain.anchor is QualifierAnchor.CURRENT_MODULE or not chain.segments:
            return None
        if len(chain.segments) == 1 and not chain.anchored:
            bare_atom = _bare_atom((chain.segments[0].name, variant))
            bare_candidates = self._current_scope().bare_constructor_candidates(bare_atom)
            # Several opened scopes contributing the same owner fall through to
            # the shared lookup below, which owns the ambiguity diagnostic.
            if bare_candidates is not None and len(bare_candidates) == 1:
                return next(iter(bare_candidates))
        if self._import_env is None:
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
        if candidate is None:
            return ConstructorRef(
                owner_name=owner_ref.name,
                variant=variant,
                owner_decl_node_id=owner_ref.decl_node_id,
                type_params=(),
                owner_module_id=owner_ref.module_id,
                owner_path=owner_ref.scope_path,
            )
        return replace(candidate, variant=variant)

    def _lookup_bare_contribution(self, name: NameAtom, span: SourceSpan) -> BindingRef | None:
        """Resolve one region's bare contributions, deferring clashes to use sites."""
        candidates = self._current_scope().bare_candidates(name)
        if candidates is None:
            return None
        assert self._root_scope is not None
        resolved = set(candidates)
        if name in self._root_scope.bare_contributions and self._import_env is not None:
            for module, source in self._import_env.unqualified.get(name, frozenset()):
                constructor = self._cross_module_constructor_refs.get((module, source))
                ref = self._make_cross_module_ref(module, _bare_path(name)[-1], source, span)
                if constructor is not None:
                    ref = replace(
                        ref,
                        decl_node_id=constructor.owner_decl_node_id,
                        kind=BinderKind.constructor_binding,
                        scope_path=constructor.owner_path,
                    )
                if not any(
                    (existing.module_id, existing.scope_path, existing.decl_node_id, existing.kind)
                    == (ref.module_id, ref.scope_path, ref.decl_node_id, ref.kind)
                    for existing in resolved
                ):
                    resolved.add(ref)
        if len(resolved) == 1:
            return next(iter(resolved))
        qualifiers = ", ".join(
            sorted(
                spell_declaration(
                    ref.module_id, (*ref.scope_path, ref.name), local_to=self._module_id
                )
                for ref in resolved
            )
        )
        rendered = "::".join(_bare_path(name))
        raise AglScopeError(
            f"'{rendered}' is ambiguous: contributed by multiple opened scopes. "
            f"Use a qualified reference to disambiguate: {qualifiers}",
            span=span,
        )

    def _lookup_import_env_unqualified(self, name: str, span: SourceSpan) -> BindingRef | None:
        """Look up a bare name in the open-import environment (program context).

        Returns a ``BindingRef`` if exactly one ``QName`` matches, or raises
        ``AglScopeError`` on ambiguity (clash-on-use).  Returns ``None`` if the
        name is not found in any open import.  Shared by bare value references
        and bare assignment targets, so an open import exposes a binding the
        same way for reads and writes.
        """
        assert self._import_env is not None
        qnames = self._import_env.unqualified.get(name)
        if qnames is None:
            return None
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
        """Resolve an imported qualified VarRef in program context."""
        assert self._import_env is not None
        qname = self._resolve_qualified_qname(module_qualifier, node.name, node.span)
        constructor = self._cross_module_constructor_refs.get(qname)
        if constructor is not None:
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
        assert self._import_env is not None
        route_segment = qualifier.segments[0]
        if route_segment.type_args is not None:
            raise AglScopeError(
                f"Type arguments cannot be applied to module route '{route_segment.name}'.",
                span=route_segment.span,
            )
        route = tuple(part for part in route_segment.name.split("/"))
        candidate_modules = qualifier_candidates(
            self._import_env, route, anchored=qualifier.anchored
        )
        cumulative: ScopePath = ()
        for segment in qualifier.segments[1:]:
            cumulative = (*cumulative, segment.name)
            if segment.type_args is not None and not any(
                (module, _bare_atom(cumulative)) in self._cross_module_type_scopes
                for module in candidate_modules
            ):
                raise AglScopeError(
                    f"Type arguments cannot be applied to scope segment '{segment.name}'.",
                    span=segment.span,
                )
        atom_path = (*cumulative, name)
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
                f"'{name}' is not in the imported set of '{rendered}'.", span=span
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
        decl_node_id, decl_span, kind = self._decl_info.get(
            key, (-1, span, BinderKind.function_binding)
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
        if advisory is None or self._import_env is None:
            return None
        result = resolve_qualified(
            self._import_env, advisory.segments, advisory.member, anchored=advisory.anchored
        )
        if not isinstance(result, QualResolutionFound):
            return None
        if advisory.type_qualified:
            _node_id, _span, kind = self._decl_info.get(
                result.qname, (-1, advisory.dcolon_span, BinderKind.let_binding)
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

    def _resolve_call(self, node: Call) -> None:
        """Resolve a ``Call`` node.

        If the callee is a bare ``VarRef`` whose name is a built-in, classify
        the call in ``builtin_calls`` and skip normal callee resolution.  For
        all other callees, resolve the callee expression normally (it must
        resolve to a binding).
        """
        callee = node.callee
        if (
            isinstance(callee, VarRef)
            and callee.name in _BUILTIN_CALL_NAMES
            and self._current_scope().lookup(callee.name) is None
        ):
            self._builtin_calls[node.node_id] = _BUILTIN_CALL_NAMES[callee.name]
        else:
            self._resolve_expr(callee)
            if isinstance(callee, VarRef) and callee.name in _BUILTIN_CALL_NAMES:
                ref = self._resolution.get(callee.node_id)
                if ref is not None and ref.kind is BinderKind.function_binding:
                    self._builtin_calls[node.node_id] = _BUILTIN_CALL_NAMES[callee.name]
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

        Ordering mirrors qualified value resolution: an opened contribution or
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
                if candidate.owner_module_id in (self._module_id, PRELUDE_ID)
                and not candidate.owner_path
            )
        if chain.anchor is None:
            opened = self._current_scope().bare_constructor_candidates(
                _bare_atom((*relative_path, node.name))
            )
            if opened:
                return tuple(opened)
        local_path = self._validate_local_scope_chain(chain)
        if local_path is not None:
            if local_path in self._type_paths:
                return (self._constructor_for_type_path(local_path, node.name),)
            scoped = self._scoped_constructor_candidates.get((local_path, node.name), ())
            if scoped:
                return tuple(scoped)
        if (
            self._import_env is not None
            and chain.anchor is not QualifierAnchor.CURRENT_MODULE
            and chain.segments
        ):
            qname = self._try_resolve_qualified_qname(chain, node.name)
            if qname is not None:
                imported = self._cross_module_constructor_refs.get(qname)
                if imported is not None:
                    return (imported,)
                # A root record, exception, or alias constructor is reached as an
                # ordinary imported member; its declaration kind is what makes it
                # a constructor spelling.
                atom_path = _bare_path(qname[1])
                decl_node_id, _decl_span, kind = self._decl_info.get(
                    qname, (-1, node.span, BinderKind.let_binding)
                )
                if kind is BinderKind.constructor_binding:
                    return (
                        ConstructorRef(
                            owner_name=atom_path[-1],
                            variant=None,
                            owner_decl_node_id=decl_node_id,
                            type_params=(),
                            owner_module_id=qname[0],
                            owner_path=atom_path[:-1],
                        ),
                    )
        if self._resolve_constructor_chain(
            node.node_id, chain, node.name, defer_route_diagnostics=True
        ):
            return (self._constructor_refs[node.node_id],)
        return ()

    def _try_resolve_qualified_qname(
        self, chain: QualifierChain, name: str
    ) -> tuple[ModuleId, NameAtom] | None:
        """Resolve ``chain::name`` as one imported atom, or ``None`` on any failure."""
        assert self._import_env is not None
        route = tuple(part for part in chain.segments[0].name.split("/"))
        atom = _bare_atom((*(segment.name for segment in chain.segments[1:]), name))
        return try_resolve_qualified_member(self._import_env, route, atom, anchored=chain.anchored)

    def _bare_constructor_candidates(self, name: str) -> tuple[ConstructorRef, ...]:
        """Return the constructor candidates an unqualified *name* can select.

        A region's opened contributions take precedence over the module-wide
        candidate table, so a bare spelling inside a scope selects the same
        constructor a qualified reference would.
        """
        regional = self._current_scope().bare_constructor_candidates(name)
        if regional is not None:
            return tuple(regional)
        return tuple(self._constructor_candidates.get(name, ()))

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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_module(
    program: Program,
    *,
    parent_scope: ScopeNode | None = None,
    ambient_agents: frozenset[str] = frozenset(),
    ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
    ambient_type_names: frozenset[str] = frozenset(),
    origin_path: Path | None = None,
) -> ModuleResolution:
    """Run the full static name-resolution pass over *program*.

    Parameters
    ----------
    program:
        A parsed ``syntax.Program`` AST.
    parent_scope:
        When given, the entry's root ``ScopeNode`` is parented to it, so name
        lookups (``VarRef``, ``:=``) fall through to session bindings.  New
        declarations live in the entry's own root scope and *shadow* parent
        bindings without raising a duplicate-declaration error.  Default
        ``None`` → standalone behaviour.
    ambient_agents:
        Agent names the host already backs.  They are valid call targets
        alongside this program's own ``agent`` declarations, but are not
        reported in ``ModuleResolution.declared_agents`` and never produce an
        unused-agent warning.  Default empty → only in-program declarations
        are valid.
    ambient_constructor_candidates:
        Constructor candidates from prior REPL entries.  Seeded before the
        local pre-passes so that references to constructors declared in earlier
        entries resolve correctly.  Default ``None`` → no ambient candidates.
    ambient_type_names:
        Type names from prior REPL entries, used for qualified constructor
        access (``Owner::variant``).  Default empty.
    origin_path:
        *program*'s canonical source file, or ``None`` when it has no backing
        file (inline ``-c`` sources, direct REPL entries).  ``extern def`` is
        rejected unless a real path is given.  Default ``None``.

    Returns
    -------
    ModuleResolution
        The program annotated with resolution side tables.

    Raises
    ------
    AglScopeError
        On the first static scope violation (first-error abort).
    """
    return _Resolver(origin_path=origin_path).run(
        program,
        parent_scope=parent_scope,
        ambient_agents=ambient_agents,
        ambient_constructor_candidates=ambient_constructor_candidates,
        ambient_type_names=ambient_type_names,
    )
