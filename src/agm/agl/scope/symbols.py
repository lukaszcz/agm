"""Symbol and scope-tree data types for the AgL resolution pass.

Data model
----------
- ``BindingRef`` — a resolved variable reference: which scope introduced the
  binding, whether it is mutable, and its declaration span.
- ``ScopeNode`` — a node in the scope tree (one per scope-introducing
  construct).  The root ``ScopeNode`` is always present; nested scopes form a
  tree for visibility analysis.
- ``ResolvedProgram`` — the frozen output of the scope pass: the original
  ``Program`` plus three side tables.
- ``CallKind`` — enum distinguishing how an ``AgentCall`` node resolved.
- ``AglScopeError`` — fatal scope error raised by the resolver.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.syntax.nodes import AgentDecl, Program
from agm.agl.syntax.spans import SourceSpan

# ---------------------------------------------------------------------------
# CallKind — how an AgentCall was resolved
# ---------------------------------------------------------------------------


class BinderKind(enum.Enum):
    """How an immutable (or mutable) binding was introduced.

    Used to phrase a precise ``set`` rejection message that names the ACTUAL
    binder kind, rather than always blaming ``let`` (F8).

    ``let_binding``
        A ``let`` declaration (immutable).
    ``var_binding``
        A ``var`` declaration (mutable).
    ``param_binding``
        A ``param`` declaration (immutable, root-scope).
    ``catch_binder``
        The binder introduced by a ``catch e`` clause (immutable, branch-local).
    ``pattern_binding``
        A variable introduced by a ``case``/``match`` pattern (immutable).
    """

    let_binding = "let_binding"
    var_binding = "var_binding"
    param_binding = "param_binding"
    catch_binder = "catch_binder"
    pattern_binding = "pattern_binding"


class CallKind(enum.Enum):
    """Classification of a resolved agent call.

    ``agent``
        A named custom agent (registered with the host runtime).
    ``default_agent``
        The ``ask`` contextual keyword → the runtime's default agent.
    ``shell_exec``
        The ``exec`` contextual keyword → shell execution.
    """

    agent = "agent"
    default_agent = "default_agent"
    shell_exec = "shell_exec"


# ---------------------------------------------------------------------------
# BindingRef — a resolved variable reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BindingRef:
    """A resolved reference to a scope binding.

    ``name``
        The variable name.
    ``mutable``
        ``True`` for ``var`` bindings; ``False`` for ``let`` and ``input``
        bindings, and for pattern/catch binders.
    ``decl_span``
        Source span of the declaration statement (used in error messages).
    ``decl_node_id``
        The ``node_id`` of the declaration node (``LetDecl``, ``VarDecl``,
        ``ParamDecl``, ``VarPattern``, or ``CatchClause``).
    ``kind``
        How the binding was introduced (``let``/``var``/``input``/catch
        binder/pattern binding).  Drives the precise ``set`` rejection message
        so a mutation of a catch binder is not mislabelled as a ``let`` (F8).
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
    - ``bindings``: the names introduced *directly* in this scope (by
      ``let``/``var``/``input`` declarations, pattern variables, or catch
      binders).
    - ``parent``: the enclosing scope (``None`` for the root scope).
    - ``node_id``: the ``node_id`` of the AST construct that opened this scope
      (the ``Program.node_id`` for the root scope).

    Lookup walks the parent chain.  ``set`` and VarRef resolution both use
    ``lookup``.  The scope chain is built bottom-up: the resolver opens a new
    ``ScopeNode`` on entering a scope-introducing construct and restores the
    previous one on exit.
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
        Maps every ``VarRef.node_id`` and ``SetStmt.node_id`` to the
        ``BindingRef`` it resolved to.
    ``call_kinds``
        Maps every ``AgentCall.node_id`` to its ``CallKind``.
    ``root_scope``
        The root ``ScopeNode`` (tree root).  Nested scopes are linked via
        ``ScopeNode.parent``.
    ``declared_agents``
        Maps each agent name declared in THIS program (via an ``agent``
        declaration) to its :class:`AgentDecl` node.  Ambient agents supplied
        by the host (see ``resolve(..., ambient_agents=...)``) are NOT included
        here.  Always populated by the resolver.
    ``program_name``
        The source-declared program name from a ``program NAME`` declaration,
        or ``None`` when undeclared.  Used by the runtime/config layer for
        ``[params.<name>]`` keying.  At most one ``program`` declaration is
        allowed per program; a duplicate is a scope error.
    ``warnings``
        Non-fatal scope-pass diagnostics (severity ``"warning"``), e.g. an
        agent that is declared but never called.  Empty by default.
    """

    program: Program
    resolution: dict[int, BindingRef]
    call_kinds: dict[int, CallKind]
    root_scope: ScopeNode
    declared_agents: dict[str, AgentDecl] = field(default_factory=dict)
    program_name: str | None = None
    warnings: tuple[Diagnostic, ...] = ()


# ---------------------------------------------------------------------------
# AglScopeError — fatal scope error
# ---------------------------------------------------------------------------


class AglScopeError(AglError):
    """A fatal name-resolution error.

    Raised by the scope resolver on the first static scope violation
    (Q4: first-error abort policy).  Carries an optional ``SourceSpan`` for
    precise source location.
    """
