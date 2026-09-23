"""Syntax-level detection of AgL constant expressions.

A constant expression is built only from literal syntax, container literals,
templates over constants, unary operators over constants, constructor
applications, designated root builtin calls, and references to the declaring
module's own constants. This is a pure predicate over the AST — it depends
only on :mod:`agm.agl.syntax.nodes` — leaving type checking responsible for
deciding which references are constructors or module constants and for
validating the expression's declared type.
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
    InterpSegment,
    IntLit,
    NullLit,
    StringLit,
    Template,
    TypeApply,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    VarRef,
)

__all__ = ["is_constant_expression"]


def is_constant_expression(
    expr: Expr,
    *,
    is_constructor: Callable[[int], bool],
    is_constant_builtin: Callable[[int], bool] = lambda _node_id: False,
    is_module_constant: Callable[[int], bool] = lambda _node_id: False,
) -> bool:
    """Whether *expr* contains only constant construction.

    A unary operator over a constant operand is itself constant, so a negative
    number reads as the literal it looks like. A template is constant when
    every hole is — an environment hole is an ordinary call, so it is not —
    and a reference is constant when it names a constant of its own module.

    ``is_constructor``, ``is_constant_builtin`` and ``is_module_constant`` are
    supplied by the checked frontend artifact, keeping this syntax-level
    predicate independent of scope and typecheck internals. A constant builtin
    must be called through a :class:`VarRef`; a type-directed member call
    cannot prove builtin provenance.
    """

    def recur(sub_expr: Expr) -> bool:
        return is_constant_expression(
            sub_expr,
            is_constructor=is_constructor,
            is_constant_builtin=is_constant_builtin,
            is_module_constant=is_module_constant,
        )

    if isinstance(expr, (BoolLit, DecimalLit, IntLit, NullLit, StringLit, UnitLit)):
        return True
    if isinstance(expr, ArrayLit):
        return all(recur(element) for element in expr.elements)
    if isinstance(expr, DictLit):
        return all(recur(entry.value) for entry in expr.entries)
    if isinstance(expr, (UnaryNeg, UnaryNot)):
        return recur(expr.operand)
    if isinstance(expr, Template):
        return all(
            recur(segment.expr) for segment in expr.segments if isinstance(segment, InterpSegment)
        )
    if isinstance(expr, VarRef):
        return is_constructor(expr.node_id) or is_module_constant(expr.node_id)
    if isinstance(expr, TypeApply):
        return recur(expr.expr)
    if isinstance(expr, Call):
        # Builtin provenance is attached to calls speculatively for member
        # selection, so only a VarRef-rooted call can prove constancy here.
        is_root_builtin = isinstance(expr.callee, VarRef) and is_constant_builtin(expr.node_id)
        return is_root_builtin or (
            recur(expr.callee)
            and all(recur(argument) for argument in expr.args)
            and all(recur(argument.value) for argument in expr.named_args)
        )
    return False
