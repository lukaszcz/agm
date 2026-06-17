"""Interpreter subclass that captures a bare-expression entry's value.

A bare-expression REPL entry must echo its evaluated value, but the base
``Interpreter`` discards the result of items that are pure expressions at the
root block level.  Re-evaluating the expression afterward would re-fire any
agent call (violating the REPL's exactly-once guarantee), so
:class:`EchoInterpreter` captures the value of a designated trailing bare
expression (matched by node id) as it is evaluated.

In v2, the trailing item of the root block is a bare ``Expr`` node (not
wrapped in an ``ExprStmt``).  The echo is captured by overriding ``_eval_item``
and intercepting when the item being evaluated has the designated ``node_id``.
"""

from __future__ import annotations

from agm.agl.eval.interpreter import Interpreter
from agm.agl.eval.scope import Scope
from agm.agl.eval.values import Value
from agm.agl.syntax.nodes import Binder, Declaration, Expr


class EchoInterpreter(Interpreter):
    """Interpreter that records the value of one trailing bare expression.

    Set :attr:`echo_node_id` to the node id of the entry's last item when it
    is a bare expression (i.e. not a ``Binder`` or ``Declaration``).  After
    ``execute`` the captured value is available on :attr:`captured`.
    """

    echo_node_id: int | None = None
    captured: Value | None = None

    def _eval_item(self, item: object, scope: Scope) -> Value:
        # Intercept the designated trailing-expression item before delegating.
        # An Item is either a Binder, a Declaration, or an Expr.  We only
        # capture bare Expr items (not binders or declarations), matching by
        # the globally-unique node_id set by the session before execute().
        if (
            self.echo_node_id is not None
            and not isinstance(item, (Binder, Declaration))
            and isinstance(item, Expr)
            and item.node_id == self.echo_node_id
        ):
            self.captured = self._eval_expr(item, scope)
            return self.captured
        return super()._eval_item(item, scope)
