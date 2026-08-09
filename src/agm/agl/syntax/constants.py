"""Syntax-level detection of AgL constant expressions.

A constant expression is built only from literal syntax, container literals, and
constructor applications. This is a pure predicate over the AST — it depends only
on :mod:`agm.agl.syntax.nodes` — leaving type checking responsible for deciding
which references are constructors and for validating the expression's declared
type.
"""

from __future__ import annotations

from collections.abc import Callable

from agm.agl.syntax.nodes import (
    ArrayLit,
    BoolLit,
    Call,
    DecimalLit,
    DictLit,
    Expr,
    IntLit,
    NullLit,
    StringLit,
    TypeApply,
    UnitLit,
    VarRef,
)

__all__ = ["is_constant_expression"]


def is_constant_expression(
    expr: Expr,
    *,
    is_constructor: Callable[[int], bool],
    is_constant_builtin: Callable[[int], bool] = lambda _node_id: False,
) -> bool:
    """Whether *expr* contains only literal construction.

    ``is_constructor`` is supplied by the checked frontend artifact, keeping
    this syntax-level predicate independent of scope and typecheck internals.
    """
    if isinstance(expr, (BoolLit, DecimalLit, IntLit, NullLit, StringLit, UnitLit)):
        return True
    if isinstance(expr, ArrayLit):
        return all(
            is_constant_expression(
                element, is_constructor=is_constructor, is_constant_builtin=is_constant_builtin
            )
            for element in expr.elements
        )
    if isinstance(expr, DictLit):
        return all(
            is_constant_expression(
                entry.value, is_constructor=is_constructor, is_constant_builtin=is_constant_builtin
            )
            for entry in expr.entries
        )
    if isinstance(expr, VarRef):
        return is_constructor(expr.node_id)
    if isinstance(expr, TypeApply):
        return is_constant_expression(
            expr.expr, is_constructor=is_constructor, is_constant_builtin=is_constant_builtin
        )
    if isinstance(expr, Call):
        return is_constant_builtin(expr.node_id) or (
            is_constant_expression(
                expr.callee, is_constructor=is_constructor, is_constant_builtin=is_constant_builtin
            )
            and all(
                is_constant_expression(
                    argument, is_constructor=is_constructor, is_constant_builtin=is_constant_builtin
                )
                for argument in expr.args
            )
            and all(
                is_constant_expression(
                    argument.value,
                    is_constructor=is_constructor,
                    is_constant_builtin=is_constant_builtin,
                )
                for argument in expr.named_args
            )
        )
    return False
