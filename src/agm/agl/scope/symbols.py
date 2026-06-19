"""Symbol and scope-tree data types for the AgL v2 resolution pass.

Data model
----------
- ``BindingRef`` — a resolved variable reference: which scope introduced the
  binding, whether it is mutable, and its declaration span.
- ``ConstructorRef`` — metadata about a resolved constructor reference (record
  or enum variant).
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


# The single source of truth for the built-in call names and their kinds.
# The resolver classifies calls by this mapping; the checker and any other
# layer that needs the set of built-in names derives it from here.
BUILTIN_CALL_NAMES: dict[str, BuiltinKind] = {
    "print": BuiltinKind.PRINT,
    "exec": BuiltinKind.EXEC,
    "ask": BuiltinKind.ASK,
    "ask-request": BuiltinKind.ASK_REQUEST,
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
    ``constructor_binding``
        A record constructor or enum variant binding (immutable value binding).
    """

    let_binding = "let_binding"
    var_binding = "var_binding"
    catch_binder = "catch_binder"
    pattern_binding = "pattern_binding"
    function_binding = "function_binding"
    agent_binding = "agent_binding"
    param_binding = "param_binding"
    constructor_binding = "constructor_binding"


# ---------------------------------------------------------------------------
# ConstructorRef — metadata about a resolved constructor reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstructorRef:
    """A resolved constructor reference's owner metadata.

    ``owner_name``
        The record or enum TYPE name.
    ``variant``
        The enum variant name; ``None`` for a record constructor.
    ``owner_decl_node_id``
        The ``node_id`` of the ``RecordDef`` / ``EnumDef`` that declares this
        constructor.
    ``type_params``
        The owner's declared type parameters (empty tuple if non-generic).
    """

    owner_name: str
    variant: str | None
    owner_decl_node_id: int
    type_params: tuple[str, ...]


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
    ``declared_type_names``
        Names of all root-level ``RecordDef`` / ``EnumDef`` / ``TypeAlias``
        declarations.  Used by the scope pass to classify qualified
        ``Owner.member`` field accesses.
    ``constructor_candidates``
        Maps each constructor name to an ordered tuple of all
        :class:`ConstructorRef` candidates (one per record/enum that declares
        it).  A single entry means the name is unambiguous; two or more mean
        an overload set requiring qualification.
    ``constructor_refs``
        Maps a ``VarRef.node_id`` (or ``Call.node_id`` whose callee was a
        constructor ``VarRef``) to the single :class:`ConstructorRef` it
        resolved to (only present when the candidate set has exactly one entry
        and no nearer non-constructor binding shadows it).
    ``qualified_constructor_refs``
        Maps a ``FieldAccess.node_id`` to ``(owner_name, member)`` when the
        field access is a type-qualified constructor reference
        (``Option.some``).  The checker validates that the owner is really an
        enum and the member a real variant.
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
    declared_type_names: frozenset[str] = frozenset()
    constructor_candidates: dict[str, tuple[ConstructorRef, ...]] = field(default_factory=dict)
    constructor_refs: dict[int, ConstructorRef] = field(default_factory=dict)
    qualified_constructor_refs: dict[int, tuple[str, str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AglScopeError — fatal scope error
# ---------------------------------------------------------------------------


class AglScopeError(AglError):
    """A fatal name-resolution error.

    Raised by the scope resolver on the first static scope violation
    (first-error abort policy).  Carries an optional ``SourceSpan`` for
    precise source location.
    """
