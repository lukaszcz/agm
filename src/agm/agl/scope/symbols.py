"""Symbol and scope-tree data types for the AgL v2 resolution pass.

Data model
----------
- ``BindingRef`` — a resolved variable reference: which scope introduced the
  binding, whether it is mutable, and its declaration span.
- ``ScopeNode`` — a node in the scope tree (one per scope-introducing
  construct).  The root ``ScopeNode`` is always present; nested scopes form a
  tree for visibility analysis.
- ``ResolvedProgram`` — the frozen output of the scope pass: the original
  ``Program`` plus side tables.
- ``BuiltinKind`` — enum classifying a built-in Call node (print/exec/ask/ask-request).
- ``AglScopeError`` — fatal scope error raised by the resolver.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.syntax.nodes import AgentDecl, FuncDef, PragmaValue, Program
from agm.agl.syntax.spans import SourceSpan

# ---------------------------------------------------------------------------
# BuiltinKind — classification of a built-in Call node
# ---------------------------------------------------------------------------


class BuiltinKind(enum.Enum):
    """Classification of a resolved built-in call.

    Attached to ``Call.node_id`` in ``ResolvedProgram.builtin_calls`` when the
    callee is one of the three special built-in names.

    ``PRINT``
        ``print(expr)`` — outputs a value; yields ``unit``.
    ``EXEC``
        ``exec(command, ...)`` — shell execution; yields ``ExecResult`` or
        a context-typed value.
    ``ASK``
        ``ask(prompt, ...)`` — invokes an agent; yields a context-typed value.
    ``ASK_REQUEST``
        ``ask-request[prompt, ...)`` — builds the ``AgentRequest`` that the
        corresponding ``ask`` call would dispatch, without invoking the agent;
        yields an ``AgentRequest`` record.
    """

    PRINT = "PRINT"
    EXEC = "EXEC"
    ASK = "ASK"
    ASK_REQUEST = "ASK_REQUEST"
    PARSE_JSON = "PARSE_JSON"


# The single source of truth for the built-in call names and their kinds.
# The resolver classifies calls by this mapping; the checker and any other
# layer that needs the set of built-in names derives it from here.
BUILTIN_CALL_NAMES: dict[str, BuiltinKind] = {
    "print": BuiltinKind.PRINT,
    "exec": BuiltinKind.EXEC,
    "ask": BuiltinKind.ASK,
    "ask-request": BuiltinKind.ASK_REQUEST,
    "parse_json": BuiltinKind.PARSE_JSON,
}


# ---------------------------------------------------------------------------
# BinderKind — how a binding was introduced
# ---------------------------------------------------------------------------


class BinderKind(enum.Enum):
    """How an immutable (or mutable) binding was introduced.

    Used to phrase a precise ``:=`` rejection message that names the ACTUAL
    binder kind, rather than always blaming ``let`` (F8).

    ``let_binding``
        A ``let`` declaration (immutable).
    ``var_binding``
        A ``var`` declaration (mutable).
    ``param_binding``
        A ``param`` declaration or function/lambda parameter (immutable).
    ``catch_binder``
        The binder introduced by a ``catch e`` clause (immutable, branch-local).
    ``pattern_binding``
        A variable introduced by a ``case``/``match`` pattern (immutable).
    ``function_binding``
        A top-level ``def`` declaration (immutable value binding).
    ``agent_binding``
        An ``agent`` declaration (immutable value binding of type ``agent``).
    ``param_binding``
        A function/lambda parameter binding (immutable, function-local).
    """

    let_binding = "let_binding"
    var_binding = "var_binding"
    catch_binder = "catch_binder"
    pattern_binding = "pattern_binding"
    function_binding = "function_binding"
    agent_binding = "agent_binding"
    param_binding = "param_binding"


# ---------------------------------------------------------------------------
# BindingRef — a resolved variable reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BindingRef:
    """A resolved reference to a scope binding.

    ``name``
        The variable name.
    ``mutable``
        ``True`` for ``var`` bindings; ``False`` for all others.
    ``decl_span``
        Source span of the declaration statement (used in error messages).
    ``decl_node_id``
        The ``node_id`` of the declaration node.
    ``kind``
        How the binding was introduced.  Drives the precise ``:=`` rejection
        message so a mutation of a catch binder is not mislabelled as a
        ``let`` (F8).
    """

    name: str
    mutable: bool
    decl_span: SourceSpan
    decl_node_id: int
    kind: BinderKind


# ---------------------------------------------------------------------------
# ScopeNode — one node in the lexical scope tree
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScopeNode:
    """A lexical scope in the scope tree.

    Each ``ScopeNode`` tracks:
    - ``bindings``: the names introduced *directly* in this scope.
    - ``parent``: the enclosing scope (``None`` for the root scope).
    - ``node_id``: the ``node_id`` of the AST construct that opened this scope.

    Lookup walks the parent chain.
    """

    node_id: int
    parent: ScopeNode | None = None
    bindings: dict[str, BindingRef] = field(default_factory=dict)

    def lookup(self, name: str) -> BindingRef | None:
        """Search upward through the scope chain for *name*."""
        scope: ScopeNode | None = self
        while scope is not None:
            ref = scope.bindings.get(name)
            if ref is not None:
                return ref
            scope = scope.parent
        return None

    def define(self, name: str, ref: BindingRef) -> None:
        """Add *name* → *ref* to this scope's binding table."""
        self.bindings[name] = ref


# ---------------------------------------------------------------------------
# ResolvedProgram — output of the scope pass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedProgram:
    """Immutable output of the scope resolution pass.

    ``program``
        The original ``Program`` AST node (never mutated).
    ``resolution``
        Maps every ``VarRef.node_id`` and ``AssignStmt.node_id`` to the
        ``BindingRef`` it resolved to.
    ``builtin_calls``
        Maps every ``Call.node_id`` whose callee is a built-in name
        (``print``/``exec``/``ask``/``ask-request``) to its ``BuiltinKind``.  Calls whose
        callee resolves to a user-defined binding have no entry here.
    ``root_scope``
        The root ``ScopeNode`` (tree root).  Nested scopes are linked via
        ``ScopeNode.parent``.
    ``declared_agents``
        Maps each agent name declared in THIS program (via an ``agent``
        declaration) to its :class:`AgentDecl` node.  Always populated by the
        resolver.
    ``declared_functions``
        Maps each top-level ``def`` name to its :class:`FuncDef` node.
        Populated in the pre-pass; useful for downstream typecheck and eval.
    ``config_pragmas``
        Validated config pragma key→value map collected in the header pass.
    ``program_name``
        The source-declared program name from a ``program NAME`` declaration,
        or ``None`` when undeclared.
    ``warnings``
        Non-fatal scope-pass diagnostics (severity ``"warning"``), e.g. an
        agent that is declared but never referenced.  Empty by default.
    """

    program: Program
    resolution: dict[int, BindingRef]
    builtin_calls: dict[int, BuiltinKind]
    root_scope: ScopeNode
    declared_agents: dict[str, AgentDecl] = field(default_factory=dict)
    declared_functions: dict[str, FuncDef] = field(default_factory=dict)
    config_pragmas: dict[str, PragmaValue] = field(default_factory=dict)
    program_name: str | None = None
    warnings: tuple[Diagnostic, ...] = ()


# ---------------------------------------------------------------------------
# AglScopeError — fatal scope error
# ---------------------------------------------------------------------------


class AglScopeError(AglError):
    """A fatal name-resolution error.

    Raised by the scope resolver on the first static scope violation
    (first-error abort policy).  Carries an optional ``SourceSpan`` for
    precise source location.
    """
