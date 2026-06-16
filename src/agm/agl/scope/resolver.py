"""Static name-resolution pass (Component 4) for the AgL pipeline.

``resolve(program)`` performs a full single-pass walk over the AST,
building the lexical scope chain and populating three side tables:

- ``resolution``:  ``VarRef.node_id`` / ``SetStmt.node_id``  →  ``BindingRef``
- ``call_kinds``:  ``AgentCall.node_id``  →  ``CallKind``

The resolver walks the AST using explicit ``isinstance`` dispatch in
``_resolve_stmt`` / ``_resolve_expr``.  Each scope-introducing construct has a
dedicated ``_resolve_*`` method.  Adding support for a new node kind means
adding an ``isinstance`` branch in ``_resolve_stmt`` or ``_resolve_expr`` and,
if the construct introduces a scope, a new ``_resolve_*`` helper method.

Scope rules (design §9, plan §6.2)
------------------------------------
1. ``let``/``var`` bind in the current scope; redeclaration in the *same*
   scope is an error.
2. ``set`` resolves to the nearest visible **mutable** binding; ``set`` on an
   immutable binding → error; ``set`` on an undeclared name → error.
3. Reading (``VarRef``) a name not visible in the current scope chain → error.
4. Pattern variables and catch binders are immutable and branch-local.
5. ``do`` body bindings are visible to the ``until`` condition but not after
   the loop.
6. ``input`` declarations are only valid at the program root; they bind an
   immutable root-scope name.  Redeclaration at the root like any other name.

Contextual keywords (design §2.1, §4.12)
-----------------------------------------
``ask`` and ``exec`` cannot be declared with ``let``/``var``/``input`` and
cannot shadow existing builtins.  In *call position* (``AgentCall.agent``)
they always resolve to ``CallKind.default_agent`` and ``CallKind.shell_exec``
respectively.  Any other ``AgentCall.agent`` name resolves to
``CallKind.agent``.

M2/M3/M4 scope handling
-------------------------
Scope-introducing constructs from later milestones (``do``, ``if``, ``case``,
``try``/``catch``) are handled by dedicated ``_resolve_*`` methods that will be
called from ``_resolve_stmt`` once the parser produces those nodes.  The
scope-chain infrastructure (``ScopeNode``, ``_push_scope``, ``_pop_scope``) is
already in place.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from agm.agl.diagnostics import Diagnostic
from agm.agl.scope.symbols import (
    AglScopeError,
    BinderKind,
    BindingRef,
    CallKind,
    ResolvedProgram,
    ScopeNode,
)
from agm.agl.syntax.nodes import (
    AgentCall,
    AgentDecl,
    CaseExprBranch,
    CaseStmt,
    CaseStmtBranch,
    CatchClause,
    ConstructorPattern,
    DoUntil,
    EnumDef,
    IfBranch,
    IfExprBranch,
    IfStmt,
    InputDecl,
    InterpSegment,
    LetDecl,
    PatternField,
    PrintStmt,
    Program,
    Raise,
    RecordDef,
    SetStmt,
    Stmt,
    Template,
    TryCatch,
    TypeAlias,
    VarDecl,
    VarPattern,
    VarRef,
)

# Reserved contextual keyword names that may not be used as variable names.
_RESERVED_NAMES: frozenset[str] = frozenset({"ask", "exec"})

# Per-binder phrasing for the ``set``-on-immutable rejection (F8).  Each phrase
# completes ``Cannot assign to 'x': <phrase> (immutable).`` and names the ACTUAL
# binder kind so a catch binder or pattern binding is not mislabelled as a
# ``let``.  ``var`` is mutable so it never reaches this table.
_IMMUTABLE_BINDER_PHRASES: dict[BinderKind, str] = {
    BinderKind.let_binding: "it was declared with 'let'",
    BinderKind.input_binding: "it is an 'input' binding",
    BinderKind.catch_binder: "it is a catch binder",
    BinderKind.pattern_binding: "it is a pattern binding",
}


def _immutable_binder_phrase(kind: BinderKind) -> str:
    """Return the ``set``-rejection phrase naming *kind*'s binder (F8)."""
    return _IMMUTABLE_BINDER_PHRASES[kind]


class _Resolver:
    """Stateful resolver that builds the scope tree and resolution tables.

    Uses explicit ``isinstance`` dispatch via ``_resolve_stmt`` /
    ``_resolve_expr`` and dedicated ``_resolve_*`` methods per scope-introducing
    construct.  Use ``resolve(program)`` — the public function — rather than
    instantiating this class directly.
    """

    def __init__(self) -> None:
        self._resolution: dict[int, BindingRef] = {}
        self._call_kinds: dict[int, CallKind] = {}
        # Scope stack — top is the current scope.
        self._scope: ScopeNode | None = None
        # Whether we are at the program root (for input-only-at-root check).
        self._at_root: bool = False
        # Agents declared at the program root (name → decl node).
        self._declared_agents: dict[str, AgentDecl] = {}
        # Valid agent names for call validation: declared ∪ ambient.
        self._valid_agents: frozenset[str] = frozenset()
        # Program-declared agent names referenced by an agent call.
        self._referenced_agents: set[str] = set()

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
        sessions); new declarations live in the entry's own root scope and
        shadow parent bindings without a duplicate-declaration error.

        *ambient_agents* are agent names the host already backs (e.g. session
        agents from earlier REPL entries).  They count as valid call targets
        alongside this program's own ``agent`` declarations, but they never
        appear in ``ResolvedProgram.declared_agents`` and never trigger an
        unused-agent warning.
        """
        # Pre-pass: collect root-level agent declarations (mirrors how input
        # is root-only) so a call may precede the declaration textually.
        self._collect_agent_decls(program)
        self._valid_agents = frozenset(self._declared_agents) | ambient_agents

        root = ScopeNode(node_id=program.node_id, parent=parent_scope)
        self._push_scope(root)
        self._at_root = True
        for stmt in program.body:
            self._resolve_stmt(stmt)
        self._at_root = False
        self._pop_scope()
        return ResolvedProgram(
            program=program,
            resolution=self._resolution,
            call_kinds=self._call_kinds,
            root_scope=root,
            declared_agents=dict(self._declared_agents),
            warnings=self._unused_agent_warnings(),
        )

    # ------------------------------------------------------------------
    # Agent declarations
    # ------------------------------------------------------------------

    def _collect_agent_decls(self, program: Program) -> None:
        """Collect root-level ``agent`` declarations into the declared table.

        Validates duplicates and reserved names here; root-only placement is
        enforced in ``_resolve_stmt`` during the walk (a declaration nested in
        a block is never collected by this root-only pre-pass).
        """
        for stmt in program.body:
            if isinstance(stmt, AgentDecl):
                self._declare_agent(stmt)

    def _declare_agent(self, decl: AgentDecl) -> None:
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

    def _resolve_agent_decl(self, stmt: AgentDecl) -> None:
        """Enforce root-only placement for an ``agent`` declaration.

        At the root the pre-pass already recorded and validated the
        declaration, so this branch is a no-op there; nested in a block it is a
        static error (mirroring ``input``).
        """
        if not self._at_root:
            raise AglScopeError(
                f"'agent' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'agent {stmt.name}' here).",
                span=stmt.span,
            )

    def _unused_agent_warnings(self) -> tuple[Diagnostic, ...]:
        """Warn for each program-declared agent never referenced by a call."""
        warnings: list[Diagnostic] = []
        for name, decl in self._declared_agents.items():
            if name not in self._referenced_agents:
                warnings.append(
                    Diagnostic(
                        message=f"agent '{name}' is declared but never called.",
                        line=decl.span.start_line,
                        severity="warning",
                    )
                )
        return tuple(warnings)

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
        """Open a fresh child scope (not the root) and yield it.

        Pushes a new ``ScopeNode`` parented to the current scope, clears the
        root flag for its lifetime (only the program root is ``_at_root``), and
        restores/pops on exit.
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
        """Raise if *name* is a contextual keyword (ask/exec)."""
        from agm.agl.syntax.spans import SourceSpan

        if name in _RESERVED_NAMES:
            sp = span if isinstance(span, SourceSpan) else None
            raise AglScopeError(
                f"'{name}' is a reserved contextual keyword and cannot be "
                "used as a variable or input name.",
                span=sp,
            )

    # ------------------------------------------------------------------
    # Statement resolution
    # ------------------------------------------------------------------

    def _resolve_stmt(self, stmt: Stmt) -> None:
        """Dispatch to the appropriate resolver for *stmt*."""
        if isinstance(stmt, LetDecl):
            self._resolve_let(stmt)
        elif isinstance(stmt, VarDecl):
            self._resolve_var(stmt)
        elif isinstance(stmt, SetStmt):
            self._resolve_set(stmt)
        elif isinstance(stmt, InputDecl):
            self._resolve_input(stmt)
        elif isinstance(stmt, AgentDecl):
            self._resolve_agent_decl(stmt)
        elif isinstance(stmt, PrintStmt):
            self._resolve_expr(stmt.value)
        elif isinstance(stmt, (RecordDef, EnumDef, TypeAlias)):
            self._resolve_type_decl(stmt)
        elif isinstance(stmt, DoUntil):
            self._resolve_do_until(stmt)
        elif isinstance(stmt, IfStmt):
            self._resolve_if(stmt)
        elif isinstance(stmt, CaseStmt):
            self._resolve_case_stmt(stmt)
        elif isinstance(stmt, TryCatch):
            self._resolve_try_catch(stmt)
        elif isinstance(stmt, Raise):
            self._resolve_expr(stmt.exc)
        else:
            from agm.agl.syntax.nodes import ExprStmt, PassStmt

            if isinstance(stmt, ExprStmt):
                self._resolve_expr(stmt.expr)
            else:
                assert isinstance(stmt, PassStmt)  # closed Stmt union

    def _resolve_let(self, stmt: LetDecl) -> None:
        self._check_not_reserved(stmt.name, stmt.span)
        self._resolve_expr(stmt.value)
        ref = BindingRef(
            name=stmt.name,
            mutable=False,
            decl_span=stmt.span,
            decl_node_id=stmt.node_id,
            kind=BinderKind.let_binding,
        )
        self._define(stmt.name, ref)

    def _resolve_var(self, stmt: VarDecl) -> None:
        self._check_not_reserved(stmt.name, stmt.span)
        self._resolve_expr(stmt.value)
        ref = BindingRef(
            name=stmt.name,
            mutable=True,
            decl_span=stmt.span,
            decl_node_id=stmt.node_id,
            kind=BinderKind.var_binding,
        )
        self._define(stmt.name, ref)

    def _resolve_set(self, stmt: SetStmt) -> None:
        ref = self._current_scope().lookup(stmt.target)
        if ref is None:
            raise AglScopeError(
                f"'{stmt.target}' is not declared; 'set' requires an existing mutable binding.",
                span=stmt.span,
            )
        if not ref.mutable:
            raise AglScopeError(
                f"Cannot assign to '{stmt.target}': "
                f"{_immutable_binder_phrase(ref.kind)} (immutable).",
                span=stmt.span,
            )
        self._resolution[stmt.node_id] = ref
        self._resolve_expr(stmt.value)

    def _resolve_input(self, stmt: InputDecl) -> None:
        if not self._at_root:
            raise AglScopeError(
                f"'input' declarations are only allowed at the program root, "
                f"not inside a nested block (found 'input {stmt.name}' here).",
                span=stmt.span,
            )
        self._check_not_reserved(stmt.name, stmt.span)
        ref = BindingRef(
            name=stmt.name,
            mutable=False,
            decl_span=stmt.span,
            decl_node_id=stmt.node_id,
            kind=BinderKind.input_binding,
        )
        self._define(stmt.name, ref)

    def _resolve_type_decl(self, stmt: RecordDef | EnumDef | TypeAlias) -> None:
        """Reject type declarations that appear outside the program root."""
        if not self._at_root:
            kind_word = (
                "record" if isinstance(stmt, RecordDef)
                else "enum" if isinstance(stmt, EnumDef)
                else "type"
            )
            raise AglScopeError(
                f"Type declarations are only allowed at the top level of the "
                f"program, not inside a nested block (found '{kind_word}' here).",
                span=stmt.span,
            )
        # At root: ignored here; the typecheck pass handles type names.

    def _resolve_do_until(self, stmt: DoUntil) -> None:
        # Open a fresh iteration scope for the body + condition.
        with self._child_scope(stmt.node_id):
            for s in stmt.body:
                self._resolve_stmt(s)
            # until condition is evaluated in the same iteration scope.
            self._resolve_expr(stmt.condition)

    def _resolve_if(self, stmt: IfStmt) -> None:
        for branch in stmt.branches:
            self._resolve_if_branch(branch)

    def _resolve_if_branch(self, branch: IfBranch) -> None:
        from agm.agl.syntax.nodes import ElseSentinel

        if not isinstance(branch.cond, ElseSentinel):
            self._resolve_expr(branch.cond)
        with self._child_scope(branch.node_id):
            for s in branch.body:
                self._resolve_stmt(s)

    def _resolve_case_stmt(self, stmt: CaseStmt) -> None:
        self._resolve_expr(stmt.subject)
        for branch in stmt.branches:
            self._resolve_case_stmt_branch(branch)

    def _resolve_case_stmt_branch(self, branch: CaseStmtBranch) -> None:
        with self._child_scope(branch.node_id) as child:
            # Bind pattern variables into the branch scope before the body.
            self._bind_pattern_vars(branch.pattern, child)
            for s in branch.body:
                self._resolve_stmt(s)

    def _resolve_try_catch(self, stmt: TryCatch) -> None:
        # Try body — its own scope.
        with self._child_scope(stmt.node_id):
            for s in stmt.body:
                self._resolve_stmt(s)
        # Each catch clause gets its own scope.
        for clause in stmt.handlers:
            self._resolve_catch_clause(clause)

    def _resolve_catch_clause(self, clause: CatchClause) -> None:
        with self._child_scope(clause.node_id) as catch_scope:
            # Bind the catch binder (immutable) if present.
            if clause.binding is not None:
                self._check_not_reserved(clause.binding, clause.span)
                ref = BindingRef(
                    name=clause.binding,
                    mutable=False,
                    decl_span=clause.span,
                    decl_node_id=clause.node_id,
                    kind=BinderKind.catch_binder,
                )
                catch_scope.define(clause.binding, ref)
            for s in clause.body:
                self._resolve_stmt(s)

    # ------------------------------------------------------------------
    # Pattern variable binding
    # ------------------------------------------------------------------

    def _bind_pattern_vars(self, pattern: object, scope: ScopeNode) -> None:
        """Recursively bind variables introduced by *pattern* into *scope*.

        Raises ``AglScopeError`` if the same name is bound more than once within
        the pattern (§9 rule 1: redeclaration in the same scope is an error).
        A pattern variable that shadows an outer-scope name is still legal.
        """
        if isinstance(pattern, VarPattern):
            self._check_not_reserved(pattern.name, pattern.span)
            ref = BindingRef(
                name=pattern.name,
                mutable=False,
                decl_span=pattern.span,
                decl_node_id=pattern.node_id,
                kind=BinderKind.pattern_binding,
            )
            # Check for intra-pattern duplicate (same scope, already bound by
            # an earlier field of this pattern).
            if pattern.name in scope.bindings:
                raise AglScopeError(
                    f"Name '{pattern.name}' is bound more than once in this pattern.",
                    span=pattern.span,
                )
            scope.define(pattern.name, ref)
        elif isinstance(pattern, ConstructorPattern):
            for pf in pattern.fields:
                self._bind_pattern_field_vars(pf, scope)
        else:
            pass  # WildcardPattern, LiteralPattern — no bindings introduced

    def _bind_pattern_field_vars(self, pf: PatternField, scope: ScopeNode) -> None:
        self._bind_pattern_vars(pf.pattern, scope)

    # ------------------------------------------------------------------
    # Expression resolution
    # ------------------------------------------------------------------

    def _resolve_expr(self, expr: object) -> None:
        """Recursively resolve all names in *expr*."""
        if isinstance(expr, VarRef):
            self._resolve_varref(expr)
        elif isinstance(expr, AgentCall):
            self._resolve_agent_call(expr)
        elif isinstance(expr, Template):
            self._resolve_template(expr)
        else:
            self._resolve_expr_inner(expr)

    def _resolve_expr_inner(self, expr: object) -> None:
        """Handle the remaining expression node kinds."""
        from agm.agl.syntax.nodes import (
            BinaryOp,
            CaseExpr,
            Constructor,
            DictLit,
            FieldAccess,
            IfExpr,
            IsTest,
            ListLit,
            UnaryNeg,
            UnaryNot,
        )

        if isinstance(expr, FieldAccess):
            self._resolve_expr(expr.obj)
        elif isinstance(expr, BinaryOp):
            self._resolve_expr(expr.left)
            self._resolve_expr(expr.right)
        elif isinstance(expr, UnaryNot):
            self._resolve_expr(expr.operand)
        elif isinstance(expr, UnaryNeg):
            self._resolve_expr(expr.operand)
        elif isinstance(expr, IsTest):
            self._resolve_expr(expr.expr)
        elif isinstance(expr, CaseExpr):
            self._resolve_expr(expr.subject)
            for branch in expr.branches:
                self._resolve_case_expr_branch(branch)
        elif isinstance(expr, IfExpr):
            for if_branch in expr.branches:
                self._resolve_if_expr_branch(if_branch)
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
            pass  # IntLit, DecimalLit, BoolLit, NullLit, StringLit, TextSegment — no names

    def _resolve_varref(self, node: VarRef) -> None:
        ref = self._current_scope().lookup(node.name)
        if ref is None:
            raise AglScopeError(
                f"'{node.name}' is not defined.",
                span=node.span,
            )
        self._resolution[node.node_id] = ref

    def _resolve_agent_call(self, node: AgentCall) -> None:
        agent = node.agent
        if agent == "ask":
            kind = CallKind.default_agent
        elif agent == "exec":
            kind = CallKind.shell_exec
        else:
            if agent not in self._valid_agents:
                raise AglScopeError(
                    f"Unknown agent '{agent}'; declare it with `agent {agent}`.",
                    span=node.span,
                )
            kind = CallKind.agent
            if agent in self._declared_agents:
                self._referenced_agents.add(agent)
        self._call_kinds[node.node_id] = kind
        # Resolve expressions inside the template.
        self._resolve_template(node.template)

    def _resolve_template(self, template: Template) -> None:
        for seg in template.segments:
            if isinstance(seg, InterpSegment):
                self._resolve_expr(seg.expr)

    def _resolve_case_expr_branch(self, branch: CaseExprBranch) -> None:
        with self._child_scope(branch.node_id) as child:
            self._bind_pattern_vars(branch.pattern, child)
            self._resolve_expr(branch.body)

    def _resolve_if_expr_branch(self, branch: IfExprBranch) -> None:
        """Resolve names in a single ``IfExprBranch``.

        Mirrors ``_resolve_if_branch`` (stmt form) and
        ``_resolve_case_expr_branch`` (case expr form): resolve the non-``else``
        condition in the current scope, then resolve the branch body inside a
        fresh child scope.
        """
        from agm.agl.syntax.nodes import ElseSentinel

        if not isinstance(branch.cond, ElseSentinel):
            self._resolve_expr(branch.cond)
        with self._child_scope(branch.node_id):
            self._resolve_expr(branch.body)


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
        lookups (``VarRef``, ``set``) fall through to session bindings.  New
        declarations live in the entry's own root scope and *shadow* parent
        bindings without raising a duplicate-declaration error.  Default
        ``None`` → today's standalone behaviour (``agm exec`` unchanged).
    ambient_agents:
        Agent names the host already backs (e.g. agents declared in earlier
        REPL entries of the same session).  They are valid call targets
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
