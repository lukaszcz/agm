"""Interpreter subclass that captures a bare-expression entry's value.

A bare-expression REPL entry must echo its evaluated value, but the base
``Interpreter`` discards ``ExprStmt`` results.  Re-evaluating the expression
afterward would re-fire any agent call (violating the REPL's exactly-once
guarantee), so :class:`EchoInterpreter` captures the value of a designated
trailing ``ExprStmt`` (matched by node id) as it is executed.
"""

from __future__ import annotations

from agm.agl.eval.interpreter import Interpreter
from agm.agl.eval.scope import Scope
from agm.agl.eval.values import Value
from agm.agl.syntax.nodes import ExprStmt, ParamDecl, Stmt


class EchoInterpreter(Interpreter):
    """Interpreter that records the value of one trailing ``ExprStmt``.

    Set :attr:`echo_node_id` to the node id of the entry's last statement when it
    is a bare expression (``None`` otherwise).  After ``execute`` the captured
    value is available on :attr:`captured`.

    ``param`` declarations are handled out-of-band by ``ReplSession`` (via
    ``_declared_inputs`` / ``set_input`` / ``_value_scope``).  The REPL never
    pre-converts external values into ``param_values`` — the session manages
    the binding lifecycle itself — so ``_exec_param`` must be a no-op here to
    preserve the existing REPL incremental contract.
    """

    echo_node_id: int | None = None
    captured: Value | None = None

    def _exec_param(self, stmt: ParamDecl, scope: Scope) -> None:
        # In the REPL, param declarations are registered by ReplSession._promote
        # (into _declared_inputs) and values are bound via set_input / _value_scope.
        # The interpreter must not touch them here.
        pass

    def _exec_stmt(self, stmt: Stmt, scope: Scope) -> None:
        if isinstance(stmt, ExprStmt) and stmt.node_id == self.echo_node_id:
            self.captured = self._eval_expr(stmt.expr, scope)
            return
        super()._exec_stmt(stmt, scope)
