"""Static name-resolution pass (Component 4) for the AgL v2 pipeline.

``resolve(program)`` performs a full single-pass walk over the AST,
building the lexical scope chain and populating side tables:

- ``resolution``:     ``VarRef.node_id`` / ``AssignStmt.node_id`` → ``BindingRef``
- ``builtin_calls``:  ``Call.node_id`` → ``BuiltinKind``  (for print/exec/ask/ask-request)

Scope rules
-----------
1. ``let``/``var``/``def`` bind in the current scope; redeclaration in the
   *same* scope is an error.
2. ``:=`` resolves to the nearest visible **mutable** binding; ``:=`` on an
   immutable binding → error; ``:=`` on an undeclared name → error.
3. Reading (``VarRef``) a name not visible in the current scope chain → error.
4. Pattern variables and catch binders are immutable and branch-local.
5. ``do`` body bindings are visible to the ``until`` condition but not after.
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
``AglScopeError`` — they are not first-class values in v1 (D6).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    AglScopeError,
    BinderKind,
    BindingRef,
    BuiltinKind,
    ResolvedProgram,
    ScopeNode,
)

if TYPE_CHECKING:
    from agm.agl.scope.imports import ImportEnv
    from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    BinaryOp,
    Block,
    BoolLit,
    Call,
    Case,
    Cast,
    CatchClause,
    ConfigPragma,
    Constructor,
    ConstructorPattern,
    DecimalLit,
    DictLit,
    Do,
    EnumDef,
    Expr,
    FieldAccess,
    FuncDef,
    If,
    ImportDecl,
    IndexAccess,
    IndexTarget,
    InterpSegment,
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    NullLit,
    ParamDecl,
    Pattern,
    PatternField,
    PragmaValue,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    StringLit,
    Template,
    Try,
    TypeAlias,
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

# The set of names that may NOT be used as any kind of binding.
_RESERVED_NAMES: frozenset[str] = frozenset(_BUILTIN_CALL_NAMES)

# Allowed config pragma keys and their expected value kinds.
_PRAGMA_KEY_KINDS: dict[str, str] = {
    "log": "bool",
    "strict_json": "bool",
    "max_iters": "int_pos",
    "runner": "str_nonempty",
    "log_file": "str_nonempty",
    "timeout": "str_or_int",
}
_ALLOWED_PRAGMA_KEYS: frozenset[str] = frozenset(_PRAGMA_KEY_KINDS)

# Per-binder phrasing for the ``:=``-on-immutable rejection.
_IMMUTABLE_BINDER_PHRASES: dict[BinderKind, str] = {
    BinderKind.let_binding: "it was declared with 'let'",
    BinderKind.catch_binder: "it is a catch binder",
    BinderKind.pattern_binding: "it is a pattern binding",
    BinderKind.function_binding: "it is a function (def) binding",
    BinderKind.agent_binding: "it is an agent binding",
    BinderKind.param_binding: "it is a parameter binding",
}


def _immutable_binder_phrase(kind: BinderKind) -> str:
    """Return the ``:=``-rejection phrase naming *kind*'s binder."""
    return _IMMUTABLE_BINDER_PHRASES[kind]


# ---------------------------------------------------------------------------
# Config pragma value validation
# ---------------------------------------------------------------------------


def _validate_pragma_value(key: str, value: PragmaValue, span: object) -> None:
    """Validate that *value* matches the expected kind for *key*.

    Raises ``AglScopeError`` on a mismatch.
    """
    sp = span if isinstance(span, SourceSpan) else None
    kind = _PRAGMA_KEY_KINDS[key]

    if kind == "bool":
        if not isinstance(value, bool):
            raise AglScopeError(
                f"config pragma '{key}' requires a bool value (true or false), "
                f"got {type(value).__name__!r}.",
                span=sp,
            )
    elif kind == "int_pos":
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AglScopeError(
                f"config pragma '{key}' requires a positive integer value (> 0), "
                f"got {value!r}.",
                span=sp,
            )
    elif kind == "str_nonempty":
        if not isinstance(value, str) or not value:
            raise AglScopeError(
                f"config pragma '{key}' requires a non-empty string value, "
                f"got {value!r}.",
                span=sp,
            )
    else:
        # kind == "str_or_int": string or positive integer (e.g. timeout)
        if isinstance(value, bool):
            raise AglScopeError(
                f"config pragma '{key}' requires a string or positive integer value, "
                f"got {value!r}.",
                span=sp,
            )
        if isinstance(value, int):
            if value <= 0:
                raise AglScopeError(
                    f"config pragma '{key}' requires a positive integer value (> 0), "
                    f"got {value!r}.",
                    span=sp,
                )
        elif not isinstance(value, str) or not value:
            raise AglScopeError(
                f"config pragma '{key}' requires a non-empty string or positive "
                f"integer value, got {value!r}.",
                span=sp,
            )


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

        self._resolution: dict[int, BindingRef] = {}
        self._builtin_calls: dict[int, BuiltinKind] = {}
        # Scope stack — top is the current scope.
        self._scope: ScopeNode | None = None
        # The module's root ScopeNode (set in run()); used by _lookup_own_root
        # to bypass lexical shadows introduced by nested scopes (D9 / ::name).
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
        # Validated config pragmas.
        self._config_pragmas: dict[str, PragmaValue] = {}
        # Header-only tracking for config pragmas.
        self._seen_non_pragma: bool = False
        # Header-only tracking for imports in non-entry modules (graph mode).
        self._seen_non_import_item: bool = False
        # Source-declared program name.
        self._program_name: str | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        program: Program,
        *,
        parent_scope: ScopeNode | None = None,
        ambient_agents: frozenset[str] = frozenset(),
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
        """
        # Pre-pass 1: collect root-level agent declarations into the declared
        # table and define as value bindings (before body resolution).
        self._ambient_agents = ambient_agents
        self._collect_agent_decls(program)
        # Pre-pass 2: collect top-level def names for mutual recursion.
        self._collect_func_decls(program)

        root = ScopeNode(node_id=program.node_id, parent=parent_scope)
        self._push_scope(root)
        self._root_scope = root
        self._at_root = True

        # Define all collected agents and functions as value bindings in root.
        self._define_agent_bindings()
        self._define_ambient_agent_bindings()
        self._define_function_bindings()

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
            config_pragmas=dict(self._config_pragmas),
            program_name=self._program_name,
            warnings=self._unused_agent_warnings(),
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
        if decl.name in _RESERVED_NAMES:
            raise AglScopeError(
                f"'{decl.name}' is a built-in name and cannot be used as a "
                f"function name.",
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
    # Config pragmas
    # ------------------------------------------------------------------

    def _resolve_config_pragma(self, node: ConfigPragma) -> None:
        """Validate a ``config`` pragma and collect it."""
        if not self._at_root:
            raise AglScopeError(
                f"'config' pragmas are only allowed at the program root, "
                f"not inside a nested block (found 'config {node.key}' here).",
                span=node.span,
            )
        if self._seen_non_pragma:
            raise AglScopeError(
                f"'config' pragmas must appear before any other statements "
                f"(found 'config {node.key}' after a non-pragma statement).",
                span=node.span,
            )
        if node.key not in _ALLOWED_PRAGMA_KEYS:
            allowed = ", ".join(sorted(_ALLOWED_PRAGMA_KEYS))
            raise AglScopeError(
                f"Unknown config pragma key '{node.key}'. "
                f"Allowed keys: {allowed}.",
                span=node.span,
            )
        if node.key in self._config_pragmas:
            raise AglScopeError(
                f"Duplicate config pragma '{node.key}'.",
                span=node.span,
            )
        _validate_pragma_value(node.key, node.value, node.span)
        self._config_pragmas[node.key] = node.value

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

    def _define(self, name: str, ref: BindingRef) -> None:
        """Define *name* in the current scope; error on redeclaration."""
        scope = self._current_scope()
        if name in scope.bindings:
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
          ``TypeAlias``, and ``ImportDecl`` are allowed at the module root.
          ``LetDecl``, ``VarDecl``, ``AssignStmt``, bare expressions, and
          entry-only constructs (``AgentDecl``, ``ParamDecl``, ``ProgramDecl``)
          are rejected with a scope error.
        - Non-entry modules: ``ImportDecl`` must precede all declarations
          (header-only; ``_seen_non_import_item`` tracks this).
        """
        is_graph_mode = self._import_env is not None
        is_non_entry_root = is_graph_mode and not self._is_entry and self._at_root

        for item in items:
            if isinstance(item, ConfigPragma):
                if is_non_entry_root:
                    # config pragmas are not allowed in non-entry modules.
                    raise AglScopeError(
                        f"'config' pragmas are only allowed in the entry module, "
                        f"not inside a library module (found 'config {item.key}' here).",
                        span=item.span,
                    )
                self._resolve_config_pragma(item)
                # A pragma is not a non-pragma item.
                continue
            if isinstance(item, ImportDecl):
                if is_non_entry_root and self._seen_non_import_item:
                    raise AglScopeError(
                        "Import declarations must appear before any other "
                        "declarations in a library module.",
                        span=item.span,
                    )
                # Module-system pass (M2+) processes imports; resolver skips them.
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
            elif isinstance(item, (RecordDef, EnumDef, TypeAlias)):
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
            # Track that we've seen a non-pragma item.
            if self._at_root:
                self._seen_non_pragma = True

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
        # evaluated in the function's DEFINITION scope (plan D5).
        self._resolve_params_and_body(node)

    def _resolve_type_decl(self, node: RecordDef | EnumDef | TypeAlias) -> None:
        """Reject type declarations outside the program root."""
        if not self._at_root:
            kind_word = (
                "record"
                if isinstance(node, RecordDef)
                else "enum"
                if isinstance(node, EnumDef)
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
        elif isinstance(expr, Do):
            self._resolve_do(expr)
        elif isinstance(expr, Try):
            self._resolve_try(expr)
        elif isinstance(expr, Lambda):
            self._resolve_lambda(expr)
        elif isinstance(expr, Raise):
            self._resolve_expr(expr.exc)
        elif isinstance(expr, FieldAccess):
            self._resolve_expr(expr.obj)
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
        elif isinstance(expr, Constructor):
            for arg in expr.args:
                self._resolve_expr(arg.value)
        elif isinstance(expr, ListLit):
            for elem in expr.elements:
                self._resolve_expr(elem)
        elif isinstance(expr, DictLit):
            for entry in expr.entries:
                self._resolve_expr(entry.key)
                self._resolve_expr(entry.value)
        else:
            # Literals: IntLit, DecimalLit, BoolLit, NullLit, StringLit,
            # UnitLit, TextSegment — no names to resolve.
            assert isinstance(
                expr,
                (IntLit, DecimalLit, BoolLit, NullLit, StringLit, UnitLit),
            ), f"unhandled expr node: {type(expr)}"  # pragma: no cover

    def _resolve_varref(self, node: VarRef) -> None:
        """Resolve a name reference.

        Built-in names (print/exec/ask/ask-request) are only valid in call position; a
        bare VarRef to them is an error (D6: they are not first-class values).

        In graph mode (when ``_import_env`` is set):
        - ``node.module_qualifier is None`` → lexical scope first, then open imports.
        - ``node.module_qualifier.segments == ()`` (``::name``) → self-ref to own scope.
        - ``node.module_qualifier.segments != ()`` → qualified cross-module access.
        """
        if node.name in _RESERVED_NAMES:
            raise AglScopeError(
                f"'{node.name}' is a built-in and cannot be used as a value.",
                span=node.span,
            )

        # Graph mode: handle module_qualifier and ImportEnv lookup.
        if self._import_env is not None and node.module_qualifier is not None:
            self._resolve_varref_qualified(node)
            return

        # Standard lexical lookup (single-program mode or bare name in graph mode).
        ref = self._current_scope().lookup(node.name)
        if ref is None:
            # In graph mode with import_env, try open imports as fallback.
            if self._import_env is not None:
                ref = self._lookup_import_env_unqualified(node)
            if ref is None:
                raise AglScopeError(
                    f"'{node.name}' is not defined.",
                    span=node.span,
                )
        # Track agent references for the unused-agent warning.
        if ref.kind == BinderKind.agent_binding and node.name in self._declared_agents:
            self._referenced_agents.add(node.name)
        self._resolution[node.node_id] = ref

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
                ".".join(qn[0].segments) + "::" + qn[1] for qn in qnames
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
            # per D3; wildcard handles may cover multiple modules but each name maps
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
        """Look up *name* in the module's own root scope bindings only (D9 / ::name).

        ``::name`` must resolve to the current module's OWN top-level declaration,
        bypassing any lexical shadows introduced by nested scopes (params, let, etc.).
        We look ONLY in the root frame's direct ``bindings`` dict — we do NOT call
        ``lookup()`` (which walks the parent chain and would fall through to a session
        parent scope or find nested shadows first).
        """
        assert self._root_scope is not None, "_lookup_own_root called outside of run()"
        return self._root_scope.bindings.get(name)

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

        If the callee is a bare ``VarRef`` whose name is a built-in
        (print/exec/ask/ask-request), classify the call in ``builtin_calls`` and skip
        normal callee resolution.  For all other callees, resolve the callee
        expression normally (it must resolve to a binding).
        """
        callee = node.callee
        if isinstance(callee, VarRef) and callee.name in _BUILTIN_CALL_NAMES:
            # Built-in call: record classification; do NOT resolve callee VarRef.
            self._builtin_calls[node.node_id] = _BUILTIN_CALL_NAMES[callee.name]
        else:
            # User-defined callee: resolve normally.
            self._resolve_expr(callee)
        # Resolve positional args.
        for arg in node.args:
            self._resolve_expr(arg)
        # Resolve named-arg values.
        for named in node.named_args:
            self._resolve_expr(named.value)

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
                self._bind_pattern_vars(branch.pattern, branch_scope)
                self._resolve_expr_or_block(branch.body)

    def _resolve_do(self, node: Do) -> None:
        """Resolve a ``do[limit] body until condition`` loop.

        Opens ONE child scope; if the body is a ``Block``, resolves its items
        DIRECTLY in that scope (no nested block scope) so bindings defined in
        the body are visible to the ``until`` condition.  A non-Block body is
        resolved directly in the child scope.  The condition is resolved in the
        same child scope.
        """
        with self._child_scope(node.node_id):
            if isinstance(node.body, Block):
                # Inline block items directly — no extra block scope.
                self._resolve_block_items(node.body.items)
            else:
                self._resolve_expr(node.body)
            # Condition sees all body bindings.
            self._resolve_expr(node.condition)

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
        """
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
            self._resolve_expr_or_block(node.body)

    # ------------------------------------------------------------------
    # Pattern variable binding
    # ------------------------------------------------------------------

    def _bind_pattern_vars(self, pattern: Pattern, scope: ScopeNode) -> None:
        """Recursively bind variables introduced by *pattern* into *scope*.

        Raises ``AglScopeError`` on duplicate names within the same pattern.
        """
        if isinstance(pattern, VarPattern):
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
            for pf in pattern.fields:
                self._bind_pattern_field_vars(pf, scope)
        # WildcardPattern, LiteralPattern — no bindings introduced.

    def _bind_pattern_field_vars(self, pf: PatternField, scope: ScopeNode) -> None:
        self._bind_pattern_vars(pf.pattern, scope)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve(
    program: Program,
    *,
    parent_scope: ScopeNode | None = None,
    ambient_agents: frozenset[str] = frozenset(),
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

    Returns
    -------
    ResolvedProgram
        The program annotated with resolution side tables.

    Raises
    ------
    AglScopeError
        On the first static scope violation (first-error abort).
    """
    return _Resolver().run(
        program, parent_scope=parent_scope, ambient_agents=ambient_agents
    )
