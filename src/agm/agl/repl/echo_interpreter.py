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
from agm.agl.syntax.nodes import ExprStmt, Stmt


class EchoInterpreter(Interpreter):
    """Interpreter that records the value of one trailing ``ExprStmt``.

    Set :attr:`echo_node_id` to the node id of the entry's last statement when it
    is a bare expression (``None`` otherwise).  After ``execute`` the captured
    value is available on :attr:`captured`.
    """

    echo_node_id: int | None = None
    captured: Value | None = None

    def _exec_stmt(self, stmt: Stmt, scope: Scope) -> None:
        if isinstance(stmt, ExprStmt) and stmt.node_id == self.echo_node_id:
            self.captured = self._eval_expr(stmt.expr, scope)
            return
        super()._exec_stmt(stmt, scope)
