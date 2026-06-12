"""Static name-resolution pass (Component 4) for the AgL pipeline.

``resolve(program)`` performs a full single-pass walk over the AST,
building the lexical scope chain and populating three side tables:

- ``resolution``:  ``VarRef.node_id`` / ``SetStmt.node_id``  →  ``BindingRef``
- ``call_kinds``:  ``AgentCall.node_id``  →  ``CallKind``

The resolver is structured as a ``Visitor`` subclass so that adding new node
kinds in M2/M3 is additive: override the new ``visit_*`` method and the
existing resolution machinery is untouched.

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
``prompt`` and ``exec`` cannot be declared with ``let``/``var``/``input`` and
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
    CaseExprBranch,
    CaseStmt,
    CaseStmtBranch,
    CatchClause,
    ConstructorPattern,
    DoUntil,
    EnumDef,
    IfBranch,
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
from agm.agl.syntax.visitor import Visitor

# Reserved contextual keyword names that may not be used as variable names.
_RESERVED_NAMES: frozenset[str] = frozenset({"prompt", "exec"})

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


class _Resolver(Visitor):
    """Stateful visitor that builds the scope tree and resolution tables.

    Use ``resolve(program)`` — the public function — rather than
    instantiating this class directly.
    """

    def __init__(self) -> None:
        self._resolution: dict[int, BindingRef] = {}
        self._call_kinds: dict[int, CallKind] = {}
        # Scope stack — top is the current scope.
        self._scope: ScopeNode | None = None
        # Whether we are at the program root (for input-only-at-root check).
        self._at_root: bool = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, program: Program) -> ResolvedProgram:
        """Execute the resolution pass over *program*."""
        root = ScopeNode(node_id=program.node_id)
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
        )

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
        """Raise if *name* is a contextual keyword (prompt/exec)."""
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
        elif isinstance(stmt, PrintStmt):
            self._resolve_expr(stmt.value)
        elif isinstance(stmt, (RecordDef, EnumDef, TypeAlias)):
            # Type declarations — resolved in the typecheck pass; ignored here
            # for scope purposes (type names live in a separate namespace).
            pass
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

    def _resolve_do_until(self, stmt: DoUntil) -> None:
        # Open a fresh iteration scope for the body + condition.
        child = ScopeNode(node_id=stmt.node_id, parent=self._current_scope())
        self._push_scope(child)
        was_root = self._at_root
        self._at_root = False
        for s in stmt.body:
            self._resolve_stmt(s)
        # until condition is evaluated in the same iteration scope.
        self._resolve_expr(stmt.condition)
        self._at_root = was_root
        self._pop_scope()

    def _resolve_if(self, stmt: IfStmt) -> None:
        for branch in stmt.branches:
            self._resolve_if_branch(branch)

    def _resolve_if_branch(self, branch: IfBranch) -> None:
        from agm.agl.syntax.nodes import ElseSentinel

        if not isinstance(branch.cond, ElseSentinel):
            self._resolve_expr(branch.cond)
        child = ScopeNode(node_id=branch.node_id, parent=self._current_scope())
        self._push_scope(child)
        was_root = self._at_root
        self._at_root = False
        for s in branch.body:
            self._resolve_stmt(s)
        self._at_root = was_root
        self._pop_scope()

    def _resolve_case_stmt(self, stmt: CaseStmt) -> None:
        self._resolve_expr(stmt.subject)
        for branch in stmt.branches:
            self._resolve_case_stmt_branch(branch)

    def _resolve_case_stmt_branch(self, branch: CaseStmtBranch) -> None:
        child = ScopeNode(node_id=branch.node_id, parent=self._current_scope())
        self._push_scope(child)
        was_root = self._at_root
        self._at_root = False
        # Bind pattern variables into the branch scope before the body.
        self._bind_pattern_vars(branch.pattern, child)
        for s in branch.body:
            self._resolve_stmt(s)
        self._at_root = was_root
        self._pop_scope()

    def _resolve_try_catch(self, stmt: TryCatch) -> None:
        # Try body — its own scope.
        try_scope = ScopeNode(node_id=stmt.node_id, parent=self._current_scope())
        self._push_scope(try_scope)
        was_root = self._at_root
        self._at_root = False
        for s in stmt.body:
            self._resolve_stmt(s)
        self._at_root = was_root
        self._pop_scope()
        # Each catch clause gets its own scope.
        for clause in stmt.handlers:
            self._resolve_catch_clause(clause)

    def _resolve_catch_clause(self, clause: CatchClause) -> None:
        catch_scope = ScopeNode(node_id=clause.node_id, parent=self._current_scope())
        self._push_scope(catch_scope)
        was_root = self._at_root
        self._at_root = False
        # Bind the catch binder (immutable) if present.
        if clause.binding is not None:
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
        self._at_root = was_root
        self._pop_scope()

    # ------------------------------------------------------------------
    # Pattern variable binding
    # ------------------------------------------------------------------

    def _bind_pattern_vars(self, pattern: object, scope: ScopeNode) -> None:
        """Recursively bind variables introduced by *pattern* into *scope*."""
        if isinstance(pattern, VarPattern):
            ref = BindingRef(
                name=pattern.name,
                mutable=False,
                decl_span=pattern.span,
                decl_node_id=pattern.node_id,
                kind=BinderKind.pattern_binding,
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
        if agent == "prompt":
            kind = CallKind.default_agent
        elif agent == "exec":
            kind = CallKind.shell_exec
        else:
            kind = CallKind.agent
        self._call_kinds[node.node_id] = kind
        # Resolve expressions inside the template.
        self._resolve_template(node.template)

    def _resolve_template(self, template: Template) -> None:
        for seg in template.segments:
            if isinstance(seg, InterpSegment):
                self._resolve_expr(seg.expr)

    def _resolve_case_expr_branch(self, branch: CaseExprBranch) -> None:
        child = ScopeNode(node_id=branch.node_id, parent=self._current_scope())
        self._push_scope(child)
        was_root = self._at_root
        self._at_root = False
        self._bind_pattern_vars(branch.pattern, child)
        self._resolve_expr(branch.body)
        self._at_root = was_root
        self._pop_scope()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve(program: Program) -> ResolvedProgram:
    """Run the full static name-resolution pass over *program*.

    Parameters
    ----------
    program:
        A parsed ``syntax.Program`` AST.

    Returns
    -------
    ResolvedProgram
        The program annotated with resolution side tables.

    Raises
    ------
    AglScopeError
        On the first static scope violation (first-error abort).
    """
    return _Resolver().run(program)
