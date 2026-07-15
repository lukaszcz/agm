"""Static name-resolution pass for the AgL pipeline.

``resolve(program)`` performs a full single-pass walk over the AST,
building the lexical scope chain and populating side tables:

- ``resolution``:     ``VarRef.node_id`` / ``AssignStmt.node_id`` → ``BindingRef``
- ``builtin_calls``:  ``Call.node_id`` → ``BuiltinKind``  (for contextual built-ins)

Scope rules
-----------
1. ``let``/``var``/``def`` bind in the current scope; redeclaration in the
   *same* scope is an error.
2. ``:=`` resolves to the nearest visible **mutable** binding; ``:=`` on an
   immutable binding → error; ``:=`` on an undeclared name → error.
3. Reading (``VarRef``) a name not visible in the current scope chain → error.
4. Pattern variables and catch binders are immutable and branch-local.
5. ``loop`` body bindings are visible to the ``until`` condition but not after.
6. ``param`` and ``program`` declarations are only valid at the program root.
7. ``def`` declarations are only valid at the program root; a pre-pass
   collects them all first so every def is in scope for every other def
   (mutual recursion).
8. ``agent`` declarations are only valid at the program root; a pre-pass
   collects them as value bindings (type ``agent``) + ``declared_agents`` map.

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

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, ModuleId
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    AglScopeError,
    BinderKind,
    BindingRef,
    BuiltinKind,
    ConstructorRef,
    ResolvedProgram,
    ScopeNode,
)
from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES, RESERVED_PROGRAM_NAMES
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPES,
    COMPATIBILITY_PRELUDE_TYPE_NAMES,
    EnumType,
)

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.scope.imports import ImportEnv
    from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    BinaryOp,
    Block,
    BoolLit,
    Break,
    Call,
    Case,
    Cast,
    CatchClause,
    ConfigDecl,
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
    ListLit,
    Loop,
    NullLit,
    ParamDecl,
    Pattern,
    PatternField,
    Placeholder,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    Return,
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
    assign_target_root_name,
)
from agm.agl.syntax.spans import SourceSpan

# ---------------------------------------------------------------------------
# Built-in names and reserved-name enforcement
# ---------------------------------------------------------------------------

# Built-in call names: recognised in call position, not bindable as values.
# Sourced from ``symbols.BUILTIN_CALL_NAMES`` (the single source of truth).
_BUILTIN_CALL_NAMES = BUILTIN_CALL_NAMES

# Sentinel ``owner_decl_node_id`` for built-in constructor candidates (exceptions
# and prelude types), which have no source-level declaration node.  User
# declarations may shadow bindings carrying this sentinel.
_BUILTIN_CONSTRUCTOR_NODE_ID = -1

# The set of names that may NOT be used as any kind of binding.
_RESERVED_NAMES: frozenset[str] = frozenset(_BUILTIN_CALL_NAMES)

# Per-binder phrasing for the ``:=``-on-immutable rejection.
_IMMUTABLE_BINDER_PHRASES: dict[BinderKind, str] = {
    BinderKind.let_binding: "it was declared with 'let'",
    BinderKind.catch_binder: "it is a catch binder",
    BinderKind.pattern_binding: "it is a pattern binding",
    BinderKind.function_binding: "it is a function (def) binding",
    BinderKind.agent_binding: "it is an agent binding",
    BinderKind.param_binding: "it is a parameter binding",
    BinderKind.config_binding: "it is a config binding",
    BinderKind.constructor_binding: "it is a constructor binding",
    BinderKind.loop_var_binding: "it is a for-loop variable binding",
}


def _immutable_binder_phrase(kind: BinderKind) -> str:
    """Return the ``:=``-rejection phrase naming *kind*'s binder."""
    return _IMMUTABLE_BINDER_PHRASES[kind]


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------


class _Resolver:
    """Stateful resolver that builds the scope tree and resolution tables.

    Implements explicit ``isinstance`` dispatch for each node kind.
    Use ``resolve(program)`` — the public function — rather than instantiating
    this class directly.
    """

    def __init__(
        self,
        module_id: ModuleId = ENTRY_ID,
        import_env: ImportEnv | None = None,
        decl_info: dict[tuple[ModuleId, str], tuple[int, SourceSpan, BinderKind]]
        | None = None,
        private_info: dict[tuple[ModuleId, str], bool] | None = None,
        is_entry: bool = True,
        repl_session_scope: ScopeNode | None = None,
        origin_path: Path | None = None,
    ) -> None:
        # Graph-mode parameters (None = single-program mode).
        self._module_id: ModuleId = module_id
        self._import_env: ImportEnv | None = import_env
        # Maps (module_id, name) → (node_id, span, kind) for cross-module refs.
        self._decl_info: dict[tuple[ModuleId, str], tuple[int, SourceSpan, BinderKind]] = (
            decl_info if decl_info is not None else {}
        )
        # Maps (module_id, name) → True for private declarations.
        self._private_info: dict[tuple[ModuleId, str], bool] = (
            private_info if private_info is not None else {}
        )
        # Whether this module is the entry module (graph mode only).
        self._is_entry: bool = is_entry
        # Optional REPL session scope for ``::name`` self-ref fallback.
        # When set, ``_lookup_own_root`` falls back to this scope for names not
        # in the entry's own root scope, allowing ``::name`` to resolve to a
        # prior session binding.
        self._repl_session_scope: ScopeNode | None = repl_session_scope
        # This module's canonical source file, or None for a module with no
        # backing file (inline `-c` sources, direct REPL entries). Drives the
        # `extern def` placement check — externs require a file-backed module.
        self._origin_path: Path | None = origin_path

        self._resolution: dict[int, BindingRef] = {}
        self._builtin_calls: dict[int, BuiltinKind] = {}
        # Scope stack — top is the current scope.
        self._scope: ScopeNode | None = None
        # The module's root ScopeNode (set in run()); used by _lookup_own_root
        # to bypass lexical shadows introduced by nested scopes for ::name.
        self._root_scope: ScopeNode | None = None
        # Whether we are at the program root (for root-only checks).
        self._at_root: bool = False
        # Agents declared at the program root.
        self._declared_agents: dict[str, AgentDecl] = {}
        # Ambient agents from the host (not in declared_agents).
        self._ambient_agents: frozenset[str] = frozenset()
        # Top-level function defs.
        self._declared_functions: dict[str, FuncDef] = {}
        # Program-declared agent names that have been referenced as a VarRef.
        self._referenced_agents: set[str] = set()
        # Names of config keys declared so far (duplicate detection).
        self._declared_config_names: set[str] = set()
        # Header-only tracking for imports in non-entry modules (graph mode).
        self._seen_non_import_item: bool = False
        # Source-declared program name.
        self._program_name: str | None = None
        # Names of all root-level type declarations (RecordDef/EnumDef/TypeAlias).
        self._declared_type_names: set[str] = set()
        # Constructor candidates: name -> ordered list of ConstructorRef.
        self._constructor_candidates: dict[str, list[ConstructorRef]] = {}
        # Resolved single-candidate constructor refs: VarRef.node_id -> ConstructorRef.
        self._constructor_refs: dict[int, ConstructorRef] = {}
        # Qualified constructor refs: VarRef.node_id -> (owner_name, member, owner_module_id).
        self._qualified_constructor_refs: dict[int, tuple[str, str, ModuleId | None]] = {}
        # VarPattern.node_id of bare names that denote a constructor (nullary
        # variant patterns), not variable binders.
        self._bare_variant_patterns: set[int] = set()
        # Case.node_id -> exact lexical scope active at the case site.
        self._case_scopes: dict[int, ScopeNode] = {}
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
    ) -> ResolvedProgram:
        """Execute the resolution pass over *program*.

        When *parent_scope* is given, the entry's root scope is parented to it
        so name lookups fall through to session bindings (incremental REPL
        sessions).  New declarations live in the entry's own root scope and
        shadow parent bindings without a duplicate-declaration error.

        *ambient_agents* are agent names the host already backs.  They count
        as valid call targets alongside this program's own ``agent``
        declarations, but never appear in ``ResolvedProgram.declared_agents``
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
                    self._add_constructor_candidate(cname, cref)
        # Seed ambient type names (from prior REPL entries).
        if ambient_type_names:
            self._declared_type_names.update(ambient_type_names)

        # Pre-pass 1: collect root-level agent declarations into the declared
        # table and define as value bindings (before body resolution).
        self._ambient_agents = ambient_agents
        self._collect_agent_decls(program)
        # Pre-pass 2: collect top-level def names for mutual recursion.
        self._collect_func_decls(program)
        # Pre-pass 3: collect type-declaration names and validate type_params.
        self._collect_type_decl_names(program)
        # Pre-pass 4: collect constructor candidates from RecordDef/EnumDef.
        self._collect_constructor_candidates(program)

        root = ScopeNode(node_id=program.node_id, parent=parent_scope)
        self._push_scope(root)
        self._root_scope = root
        self._at_root = True

        # Define all collected agents and functions as value bindings in root.
        self._define_agent_bindings()
        self._define_ambient_agent_bindings()
        self._define_function_bindings()
        # Define constructor bindings in root scope.
        self._define_constructor_bindings()

        # Main walk: resolve all block items in order.
        self._resolve_block_items(program.body.items)

        self._at_root = False
        self._pop_scope()

        return ResolvedProgram(
            program=program,
            resolution=self._resolution,
            builtin_calls=self._builtin_calls,
            root_scope=root,
            declared_agents=dict(self._declared_agents),
            declared_functions=dict(self._declared_functions),
            program_name=self._program_name,
            warnings=self._unused_agent_warnings(),
            declared_type_names=frozenset(self._declared_type_names),
            constructor_candidates={
                name: tuple(refs)
                for name, refs in self._constructor_candidates.items()
            },
            constructor_refs=dict(self._constructor_refs),
            qualified_constructor_refs=dict(self._qualified_constructor_refs),
            bare_variant_patterns=frozenset(self._bare_variant_patterns),
            case_scopes=dict(self._case_scopes),
        )

    # ------------------------------------------------------------------
    # Pre-passes
    # ------------------------------------------------------------------

    def _collect_agent_decls(self, program: Program) -> None:
        """Collect root-level ``agent`` declarations into the declared table."""
        for item in program.body.items:
            if isinstance(item, AgentDecl):
                self._declare_agent(item)

    def _declare_agent(self, decl: AgentDecl) -> None:
        """Validate and record an agent declaration."""
        if decl.name in _RESERVED_NAMES:
            raise AglScopeError(
                f"'{decl.name}' is built-in and cannot be declared as an agent.",
                span=decl.span,
            )
        if decl.name in self._declared_agents:
            raise AglScopeError(
                f"agent '{decl.name}' is already declared.",
                span=decl.span,
            )
        self._declared_agents[decl.name] = decl

    def _collect_func_decls(self, program: Program) -> None:
        """Collect root-level ``def`` declarations for the mutual-recursion pre-pass."""
        for item in program.body.items:
            if isinstance(item, FuncDef):
                self._declare_function(item)

    def _declare_function(self, decl: FuncDef) -> None:
        """Validate and record a function declaration."""
        if decl.name in _RESERVED_NAMES and not decl.is_builtin:
            raise AglScopeError(
                f"'{decl.name}' is a built-in name and cannot be used as a "
                f"function name.",
                span=decl.span,
            )
        if decl.is_extern and self._origin_path is None:
            raise AglScopeError(
                f"'extern def {decl.name}' requires a file-backed module; "
                f"externs are not allowed in inline sources or REPL entries.",
                span=decl.span,
            )
        if decl.name in self._declared_functions:
            raise AglScopeError(
                f"Name '{decl.name}' is already declared in this scope.",
                span=decl.span,
            )
        if decl.name in self._declared_agents:
            raise AglScopeError(
                f"Name '{decl.name}' is already declared in this scope.",
                span=decl.span,
            )
        self._declared_functions[decl.name] = decl

    def _collect_type_decl_names(self, program: Program) -> None:
        """Collect names of root-level type declarations and validate type_params.

        Populates ``_declared_type_names`` and raises ``AglScopeError`` when
        a declaration has duplicate type-parameter names or when two local
        type declarations use the same name.

        Builtin prelude type names are seeded first so that qualified
        constructor access (e.g. ``ParsePolicy::Abort``) resolves correctly.
        """
        for type_name in BUILTIN_PRELUDE_TYPES:
            self._declared_type_names.add(type_name)

        local_type_names: set[str] = set()
        for item in program.body.items:
            if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
                if item.name in local_type_names:
                    raise AglScopeError(
                        f"Type name '{item.name}' is already declared in this scope.",
                        span=item.span,
                    )
                local_type_names.add(item.name)
                self._declared_type_names.add(item.name)
                self._validate_type_params(item)
            elif isinstance(item, FuncDef):
                self._validate_type_params(item)

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

    def _add_constructor_candidate(self, ctor_key: str, cref: ConstructorRef) -> None:
        """Add *cref* to the candidates list for *ctor_key*.

        Skips the entry if another candidate with the same ``owner_name`` is
        already present (duplicate type declaration — the type-builder pass will
        raise a clear "already declared" error for that; we must not conflate it
        with genuine constructor overloading across distinct types).
        """
        existing = self._constructor_candidates.get(ctor_key, [])
        if any(
            (c.owner_module_id, c.owner_name)
            == (cref.owner_module_id, cref.owner_name)
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

    def _collect_constructor_candidates(self, program: Program) -> None:
        """Build the constructor-candidates map from root-level RecordDef/EnumDef.

        For each RecordDef, the record NAME is a constructor candidate.
        For each EnumDef, each VARIANT NAME is a candidate (the enum name is NOT).
        Multiple candidates for the same name form an ordered overload set.
        Builtin exception and prelude types are seeded first.
        """
        self._seed_builtin_constructor_candidates()
        for item in program.body.items:
            if isinstance(item, RecordDef):
                cref = ConstructorRef(
                    owner_name=item.name,
                    variant=None,
                    owner_decl_node_id=item.node_id,
                    type_params=item.type_params,
                    owner_module_id=self._module_id,
                )
                self._add_constructor_candidate(item.name, cref)
            elif isinstance(item, EnumDef):
                for variant in item.variants:
                    cref = ConstructorRef(
                        owner_name=item.name,
                        variant=variant.name,
                        owner_decl_node_id=item.node_id,
                        type_params=item.type_params,
                        owner_module_id=self._module_id,
                    )
                    self._add_constructor_candidate(variant.name, cref)
            elif isinstance(item, ExceptionDef):
                cref = ConstructorRef(
                    owner_name=item.name,
                    variant=None,
                    owner_decl_node_id=item.node_id,
                    type_params=(),
                    owner_module_id=self._module_id,
                )
                self._add_constructor_candidate(item.name, cref)

    def _define_constructor_bindings(self) -> None:
        """Define each constructor name as a value binding in the current (root) scope.

        Collision rules:
        - Constructor-vs-constructor at the same scope: allowed (overload set).
        - Constructor-vs-non-constructor at the same scope: duplicate error,
          because the non-constructor binding was already defined before this
          method is called (agents and defs are defined first).
        """
        scope = self._current_scope()
        for name, crefs in self._constructor_candidates.items():
            if name in scope.bindings:
                # Built-in constructor candidates (sentinel decl_node_id, e.g. the
                # prelude `Retry`/`ExecResult` names) are conveniences and yield to
                # any user declaration that already claimed the name — skip rather
                # than raise so a `def Retry()`/`agent Retry` is legal.
                if all(c.owner_decl_node_id == _BUILTIN_CONSTRUCTOR_NODE_ID for c in crefs):
                    continue
                # A user-declared constructor colliding with a non-constructor
                # binding (def/agent) of the same name is a genuine duplicate.
                raise AglScopeError(
                    f"Name '{name}' is already declared in this scope.",
                    span=None,
                )
            # Use the first candidate's decl as the representative binding.
            rep = crefs[0]
            ref = BindingRef(
                name=name,
                mutable=False,
                decl_span=SourceSpan(
                    start_line=0, start_col=0, end_line=0, end_col=0,
                    start_offset=0, end_offset=0,
                ),
                decl_node_id=rep.owner_decl_node_id,
                kind=BinderKind.constructor_binding,
                module_id=rep.owner_module_id,
            )
            scope.define(name, ref)

    def _define_agent_bindings(self) -> None:
        """Define each collected agent as a value binding in the current scope."""
        for name, decl in self._declared_agents.items():
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
                start_line=0, start_col=0, end_line=0, end_col=0,
                start_offset=0, end_offset=0,
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
        for name, decl in self._declared_agents.items():
            if name not in self._referenced_agents:
                warnings.append(
                    Diagnostic(
                        message=f"agent '{name}' is declared but never called.",
                        line=decl.span.start_line,
                        column=decl.span.start_col,
                        end_line=decl.span.end_line,
                        end_column=decl.span.end_col,
                        severity="warning",
                    )
                )
        return tuple(warnings)

    # ------------------------------------------------------------------
    # Config declarations
    # ------------------------------------------------------------------

    def _resolve_config(self, node: ConfigDecl) -> None:
        """Resolve a ``config`` declaration into a readable runtime binding.

        A ``config`` declaration names a fixed engine key (kebab-case) and binds
        it as an immutable, runtime-resolved value (like ``param``).  The value
        expression — when present — is resolved here so any names it references
        are checked; an absent value (bare ``config KEY``) is also legal and
        resolves from the host's configured default at runtime.  Type checking of
        the value against the engine-key type happens in the typecheck pass.
        """
        if not self._at_root:
            raise AglScopeError(
                f"'config' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'config {node.name}' here).",
                span=node.span,
            )
        if node.name not in ENGINE_KEY_NAMES:
            allowed = ", ".join(sorted(ENGINE_KEY_NAMES))
            raise AglScopeError(
                f"Unknown config key '{node.name}'. "
                f"Allowed keys: {allowed}.",
                span=node.span,
            )
        if node.name in self._declared_config_names:
            raise AglScopeError(
                f"Duplicate config declaration '{node.name}'.",
                span=node.span,
            )
        self._declared_config_names.add(node.name)
        # Resolve the value expression (if any) before defining the binding, so a
        # config value cannot reference the binding it introduces.
        if node.value is not None:
            self._resolve_expr(node.value)
        # Define an immutable readable binding so the config key is visible in
        # the surrounding scope and resolved at runtime.
        ref = BindingRef(
            name=node.name,
            mutable=False,
            decl_span=node.span,
            decl_node_id=node.node_id,
            kind=BinderKind.config_binding,
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

    @staticmethod
    def _snapshot_scope(scope: ScopeNode) -> ScopeNode:
        """Copy a lexical scope chain at one source position.

        Scope nodes continue accumulating sequential block bindings during
        resolution.  Case provenance must instead retain precisely the names
        visible when the case was entered, so later declarations cannot
        retroactively change diagnostic spellings.
        """
        parent = (
            None
            if scope.parent is None
            else _Resolver._snapshot_scope(scope.parent)
        )
        return ScopeNode(scope.node_id, parent, dict(scope.bindings))

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

        A user declaration may shadow a built-in constructor binding (e.g. the
        prelude ``Retry``/``ExecResult`` names), which carry the sentinel
        ``decl_node_id`` and have no source declaration; genuine user-declared
        names still collide.
        """
        scope = self._current_scope()
        existing = scope.bindings.get(name)
        if existing is not None and not (
            existing.kind is BinderKind.constructor_binding
            and existing.decl_node_id == _BUILTIN_CONSTRUCTOR_NODE_ID
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

        In graph mode (``_import_env is not None``), additional enforcement:

        - Non-entry modules: only ``FuncDef``, ``RecordDef``, ``EnumDef``,
          ``TypeAlias``, ``InfixDecl``, ``ImportDecl``, and ``ExportDecl`` are
          allowed at the module root.
          ``LetDecl``, ``VarDecl``, ``AssignStmt``, bare expressions, and
          entry-only constructs (``AgentDecl``, ``ParamDecl``, ``ProgramDecl``)
          are rejected with a scope error.
        - Non-entry modules: ``ImportDecl`` and ``ExportDecl`` must precede all declarations
          (header-only; ``_seen_non_import_item`` tracks this).
        """
        is_graph_mode = self._import_env is not None
        is_non_entry_root = is_graph_mode and not self._is_entry and self._at_root

        for item in items:
            if isinstance(item, ConfigDecl):
                if is_non_entry_root:
                    # config declarations are not allowed in non-entry modules.
                    raise AglScopeError(
                        f"'config' declarations are only allowed in the entry module, "
                        f"not inside a library module (found 'config {item.name}' here).",
                        span=item.span,
                    )
                self._resolve_config(item)
                # A config declaration is not a non-config item.
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
                # The graph module-system pass processes imports/exports; this pass skips them.
                continue
            if isinstance(item, InfixDecl):
                if not self._at_root:
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
            if isinstance(item, FuncDef):
                self._resolve_funcdef(item)
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

    def _resolve_agent_decl(self, node: AgentDecl) -> None:
        """Enforce root-only placement for an ``agent`` declaration.

        At the root the pre-pass already recorded and defined the binding,
        so this branch is a no-op there.
        """
        if not self._at_root:
            raise AglScopeError(
                f"'agent' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'agent {node.name}' here).",
                span=node.span,
            )

    def _resolve_funcdef(self, node: FuncDef) -> None:
        """Resolve a ``def`` declaration (body + params).

        At the root: the pre-pass already defined the function binding, so we
        just resolve the body with a fresh param scope.
        Nested in a block: rejected (def is root-only).
        """
        if not self._at_root:
            raise AglScopeError(
                f"'def' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'def {node.name}' here).",
                span=node.span,
            )
        # Defaults are resolved in the enclosing (root) scope — they are
        # evaluated in the function's definition scope.
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
        # At root: ignored here; the typecheck pass handles type names.

    # ------------------------------------------------------------------
    # Binder handlers
    # ------------------------------------------------------------------

    def _resolve_let(self, node: LetDecl) -> None:
        self._check_not_reserved(node.name, node.span)
        # Resolve RHS before defining the name (lambda non-recursion).
        self._resolve_expr(node.value)
        ref = BindingRef(
            name=node.name,
            mutable=False,
            decl_span=node.span,
            decl_node_id=node.node_id,
            kind=BinderKind.let_binding,
            module_id=self._module_id,
        )
        self._define(node.name, ref)

    def _resolve_var(self, node: VarDecl) -> None:
        self._check_not_reserved(node.name, node.span)
        self._resolve_expr(node.value)
        ref = BindingRef(
            name=node.name,
            mutable=True,
            decl_span=node.span,
            decl_node_id=node.node_id,
            kind=BinderKind.var_binding,
            module_id=self._module_id,
        )
        self._define(node.name, ref)

    def _resolve_assign(self, node: AssignStmt) -> None:
        name = assign_target_root_name(node.target)
        if name is None:
            raise AglScopeError(
                "indexed assignment requires a variable list or dict root.",
                span=node.target.span,
            )
        ref = self._current_scope().lookup(name)
        if ref is None:
            raise AglScopeError(
                f"'{name}' is not declared; assignment requires an existing "
                f"mutable binding.",
                span=node.span,
            )
        if not ref.mutable:
            raise AglScopeError(
                f"Cannot assign to '{name}': "
                f"{_immutable_binder_phrase(ref.kind)} (immutable). "
                f"Declare with 'var' to make the variable mutable.",
                span=node.span,
            )
        self._resolution[node.node_id] = ref
        self._resolve_assign_target_indexes(node.target)
        self._resolve_expr(node.value)

    def _resolve_assign_target_indexes(self, target: object) -> None:
        if isinstance(target, IndexTarget):
            self._resolve_expr(target.obj)
            self._resolve_expr(target.index)

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
            self._resolve_expr(expr.expr)
        elif isinstance(expr, Cast):
            self._resolve_expr(expr.expr)
        elif isinstance(expr, TypeApply):
            self._resolve_expr(expr.expr)
        elif isinstance(expr, ListLit):
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

        In graph mode (when ``_import_env`` is set):
        - ``node.module_qualifier is None`` → lexical scope first, then open imports.
        - ``node.module_qualifier.segments == ()`` (``::name``) → self-ref to own scope.
        - ``node.module_qualifier.segments != ()`` → qualified cross-module access.
        """
        if node.type_qualifier is not None:
            self._resolve_type_qualified_constructor(node)
            return

        # Single-segment qualifiers can name either a type or an import handle.
        if node.module_qualifier is not None:
            if self._resolve_single_qualifier_constructor(node):
                return
            if self._import_env is not None:
                self._resolve_varref_qualified(node)
                return
            if node.module_qualifier.segments != ():
                qualifier_str = ".".join(node.module_qualifier.segments)
                raise AglScopeError(
                    f"No module imported under qualifier '{qualifier_str}' or type named "
                    f"'{qualifier_str}'.",
                    span=node.module_qualifier.span,
                )

        # Standard lexical lookup (single-program mode or bare name in graph mode).
        ref = self._current_scope().lookup(node.name)
        if ref is None:
            # In graph mode with import_env, try open imports as fallback.
            if self._import_env is not None:
                ref = self._lookup_import_env_unqualified(node)
            if ref is None:
                if node.name in _RESERVED_NAMES:
                    raise AglScopeError(
                        f"'{node.name}' is a built-in and cannot be used as a value.",
                        span=node.span,
                    )
                raise AglScopeError(
                    f"'{node.name}' is not defined.",
                    span=node.span,
                )
        # Track agent references for the unused-agent warning.
        if ref.kind == BinderKind.agent_binding and node.name in self._declared_agents:
            self._referenced_agents.add(node.name)
        self._resolution[node.node_id] = ref
        # If the resolved binding is a constructor, check for overload ambiguity.
        if ref.kind == BinderKind.constructor_binding:
            candidates = self._constructor_candidates.get(node.name, [])
            if len(candidates) >= 2:
                owner_names = ", ".join(
                    f"'{c.owner_name}'" for c in candidates
                )
                raise AglScopeError(
                    f"'{node.name}' is ambiguous: it is declared as a constructor "
                    f"in multiple types ({owner_names}). "
                    f"Qualify the reference, e.g. '{candidates[0].owner_name}::{node.name}'.",
                    span=node.span,
                )
            elif len(candidates) == 1:
                self._constructor_refs[node.node_id] = candidates[0]

    def _resolve_type_qualified_constructor(self, node: VarRef) -> None:
        """Resolve an explicit ``[module::]Type[args]::Ctor`` reference."""
        assert node.type_qualifier is not None
        type_name = node.type_qualifier.name
        if node.module_qualifier is None:
            if type_name not in self._declared_type_names:
                raise AglScopeError(
                    f"'{type_name}' is not a known type.",
                    span=node.type_qualifier.span,
                )
            self._qualified_constructor_refs[node.node_id] = (type_name, node.name, None)
            return
        if node.module_qualifier.segments == ():
            if type_name not in self._declared_type_names:
                raise AglScopeError(
                    f"'{type_name}' is not defined in this module.",
                    span=node.type_qualifier.span,
                )
            owner_module = self._module_id if self._import_env is not None else None
            self._qualified_constructor_refs[node.node_id] = (type_name, node.name, owner_module)
            return
        if self._import_env is None:
            qualifier_str = ".".join(node.module_qualifier.segments)
            raise AglScopeError(
                f"No module imported under qualifier '{qualifier_str}'.",
                span=node.module_qualifier.span,
            )
        src_name, owning_module = self._resolve_qualified_type_name(
            node.module_qualifier.segments, type_name, node.type_qualifier.span
        )
        self._qualified_constructor_refs[node.node_id] = (src_name, node.name, owning_module)

    def _resolve_single_qualifier_constructor(self, node: VarRef) -> bool:
        """Resolve ``Type::Ctor`` when the qualifier denotes a type name."""
        assert node.module_qualifier is not None
        segments = node.module_qualifier.segments
        if len(segments) != 1:
            return False
        type_name = segments[0]
        type_match = type_name in self._declared_type_names
        handle_match = (
            self._import_env is not None
            and self._import_env.qualified.get(segments) is not None
        )
        if type_match and handle_match:
            raise AglScopeError(
                f"Qualifier '{type_name}' is both a type name and an import handle; "
                "rename the import alias to disambiguate.",
                span=node.module_qualifier.span,
            )
        if not type_match:
            return False
        self._qualified_constructor_refs[node.node_id] = (type_name, node.name, None)
        return True

    def _resolve_qualified_type_name(
        self, handle: tuple[str, ...], type_name: str, span: SourceSpan
    ) -> tuple[str, ModuleId]:
        """Resolve ``handle::type_name`` as a constructible type owner."""
        assert self._import_env is not None
        qual_map = self._import_env.qualified.get(handle)
        qualifier_str = ".".join(handle)
        if qual_map is None:
            raise AglScopeError(
                f"No module imported under qualifier '{qualifier_str}'.",
                span=span,
            )
        qname = qual_map.get(type_name)
        if qname is None:
            owning_module: ModuleId = next(iter(qual_map.values()))[0]
            if self._private_info.get((owning_module, type_name)):
                raise AglScopeError(
                    f"'{type_name}' in module '{owning_module.dotted()}' is declared private "
                    f"and cannot be accessed from outside the module.",
                    span=span,
                )
            raise AglScopeError(
                f"'{type_name}' is not in the imported set of '{qualifier_str}'.",
                span=span,
            )
        owning_module, src_name = qname[0], qname[1]
        _decl_node_id, _decl_span, kind = self._decl_info.get(
            (owning_module, src_name), (-1, span, BinderKind.let_binding)
        )
        if kind is not BinderKind.constructor_binding:
            raise AglScopeError(
                f"'{qualifier_str}::{type_name}' is not a constructible type.",
                span=span,
            )
        return (src_name, owning_module)

    def _lookup_import_env_unqualified(self, node: VarRef) -> BindingRef | None:
        """Look up a bare name in the open-import environment (graph mode).

        Returns a ``BindingRef`` if exactly one ``QName`` matches, or raises
        ``AglScopeError`` on ambiguity (clash-on-use).  Returns ``None`` if the
        name is not found in any open import.
        """
        assert self._import_env is not None
        qnames = self._import_env.unqualified.get(node.name)
        if qnames is None:
            return None
        if len(qnames) > 1:
            # Clash-on-use: more than one module exposes this name.
            qualifiers = sorted(
                qn[0].dotted() + "::" + qn[1] for qn in qnames
            )
            hint = ", ".join(qualifiers)
            raise AglScopeError(
                f"'{node.name}' is ambiguous: imported from multiple modules. "
                f"Use a qualified reference to disambiguate: {hint}",
                span=node.span,
            )
        # Exactly one QName.
        qname = next(iter(qnames))
        return self._make_cross_module_ref(qname[0], node.name, qname[1], node.span)

    def _resolve_varref_qualified(self, node: VarRef) -> None:
        """Resolve a qualified VarRef (``::name`` or ``MODQUAL::name``) in graph mode."""
        assert self._import_env is not None
        assert node.module_qualifier is not None

        if node.module_qualifier.segments == ():
            # Self-reference: ::name — look up in own root scope.
            ref = self._lookup_own_root(node.name)
            if ref is None:
                raise AglScopeError(
                    f"'{node.name}' is not defined in this module.",
                    span=node.span,
                )
            self._resolution[node.node_id] = ref
            return

        # Qualified access: MODQUAL::name
        handle = node.module_qualifier.segments
        qual_map = self._import_env.qualified.get(handle)
        if qual_map is None:
            qualifier_str = ".".join(handle)
            raise AglScopeError(
                f"No module imported under qualifier '{qualifier_str}'.",
                span=node.span,
            )
        qname = qual_map.get(node.name)
        if qname is None:
            qualifier_str = ".".join(handle)
            # Determine the owning module for this handle: take any entry from the
            # qual_map (all entries for this handle belong to at most one source module
            # Wildcard handles may cover multiple modules, but each name maps
            # to exactly one QName).  A non-None qual_map always has at least one entry
            # because handles are only registered when names are added to them.
            owning_module: ModuleId = next(iter(qual_map.values()))[0]
            # Check if the name is private in the OWNING module (gives better error).
            if self._private_info.get((owning_module, node.name)):
                raise AglScopeError(
                    f"'{node.name}' in module '{owning_module.dotted()}' is declared private "
                    f"and cannot be accessed from outside the module.",
                    span=node.span,
                )
            raise AglScopeError(
                f"'{node.name}' is not in the imported set of '{qualifier_str}'.",
                span=node.span,
            )
        ref = self._make_cross_module_ref(qname[0], node.name, qname[1], node.span)
        self._resolution[node.node_id] = ref

    def _lookup_own_root(self, name: str) -> BindingRef | None:
        """Look up *name* in the module's own root scope bindings only.

        ``::name`` must resolve to the current module's OWN top-level declaration,
        bypassing any lexical shadows introduced by nested scopes (params, let, etc.).
        We look ONLY in the root frame's direct ``bindings`` dict — we do NOT call
        ``lookup()`` (which walks the parent chain and would fall through to a session
        parent scope or find nested shadows first).

        In the REPL graph mode, if *name* is not in the entry's own root scope,
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
        src_name: str,
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
        return BindingRef(
            name=src_name,
            mutable=False,
            decl_span=decl_span,
            decl_node_id=decl_node_id,
            kind=kind,
            module_id=owning_module,
        )

    def _resolve_call(self, node: Call) -> None:
        """Resolve a ``Call`` node.

        If the callee is a bare ``VarRef`` whose name is a built-in, classify
        the call in ``builtin_calls`` and skip normal callee resolution.  For
        all other callees, resolve the callee expression normally (it must
        resolve to a binding).
        """
        callee = node.callee
        if (
            self._import_env is None
            and isinstance(callee, VarRef)
            and callee.name in _BUILTIN_CALL_NAMES
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
        if isinstance(expr.obj, VarRef) and expr.obj.module_qualifier is None:
            existing = self._current_scope().lookup(expr.obj.name)
            if (
                expr.obj.name in self._declared_type_names
                and (existing is None or existing.kind is BinderKind.constructor_binding)
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
        self._case_scopes[node.node_id] = self._snapshot_scope(self._current_scope())
        self._resolve_expr(node.subject)
        for branch in node.branches:
            with self._child_scope(branch.node_id) as branch_scope:
                self._bind_pattern_vars(branch.pattern, branch_scope)
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

    def _bind_pattern_vars(self, pattern: Pattern, scope: ScopeNode) -> None:
        """Recursively bind variables introduced by *pattern* into *scope*.

        Raises ``AglScopeError`` on duplicate names within the same pattern.
        """
        if isinstance(pattern, VarPattern):
            if self._pattern_name_is_constructor(pattern, scope):
                # A bare name that denotes an in-scope constructor is a nullary
                # constructor pattern, not a variable binder.  The checker
                # validates it is a nullary variant of the scrutinee enum.
                self._bare_variant_patterns.add(pattern.node_id)
                return
            self._check_not_reserved(pattern.name, pattern.span)
            ref = BindingRef(
                name=pattern.name,
                mutable=False,
                decl_span=pattern.span,
                decl_node_id=pattern.node_id,
                kind=BinderKind.pattern_binding,
                module_id=self._module_id,
            )
            if pattern.name in scope.bindings:
                raise AglScopeError(
                    f"Name '{pattern.name}' is bound more than once in this pattern.",
                    span=pattern.span,
                )
            scope.define(pattern.name, ref)
        elif isinstance(pattern, ConstructorPattern):
            for p in pattern.positional:
                self._bind_pattern_vars(p, scope)
            for pf in pattern.named:
                self._bind_pattern_field_vars(pf, scope)
        # WildcardPattern, LiteralPattern — no bindings introduced.

    def _bind_pattern_field_vars(self, pf: PatternField, scope: ScopeNode) -> None:
        self._bind_pattern_vars(pf.pattern, scope)

    def _pattern_name_is_constructor(self, pattern: VarPattern, scope: ScopeNode) -> bool:
        """True when a bare pattern name denotes an in-scope constructor binding.

        A bare name in pattern position is a constructor pattern (not a variable
        binder) exactly when, as a value reference, it would denote a constructor
        — a lexically visible or open-imported record/enum constructor.  A nearer
        non-constructor binding shadows the constructor, in which case the bare
        name is an ordinary binder.  The checker then verifies the name is a
        nullary variant of the scrutinee enum.
        """
        ref = scope.lookup(pattern.name)
        if ref is not None:
            # A nearer ordinary binding shadows the constructor → binder.
            return ref.kind is BinderKind.constructor_binding
        # Not lexically visible: an open-imported constructor still counts — its
        # candidates are seeded into the candidate table from the import env.
        return bool(self._constructor_candidates.get(pattern.name))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve(
    program: Program,
    *,
    parent_scope: ScopeNode | None = None,
    ambient_agents: frozenset[str] = frozenset(),
    ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
    ambient_type_names: frozenset[str] = frozenset(),
    origin_path: Path | None = None,
) -> ResolvedProgram:
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
        reported in ``ResolvedProgram.declared_agents`` and never produce an
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
    ResolvedProgram
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
