"""Pure arithmetic and comparison helpers for the AgL evaluator.

Used by the IR evaluator.
This module is the single source of truth for operator semantics.

IMPORTANT: Only imports from stdlib, agm.agl.semantics.values, agm.agl.ir.operations,
and agm.util.decimal. No syntax, scope, or typecheck imports are permitted here.

Every DECIMAL-kind arithmetic operand here is already a ``DecimalValue``: mixed
int/decimal operands are widened at compile time by the lowerer's
``IntToDecimal`` coercion (``lower.lowerer``), labelled with the triggering
operator, so this module never widens an int itself. Comparisons instead take
a mixed int/decimal pair unwidened and compare it exactly
(:func:`~agm.util.decimal.compare_numbers`). A trapped ``decimal.DecimalException``
(overflow, division by zero, invalid operation) propagates uncaught to the
caller, which classifies and labels it (``eval.ir_interpreter``).
"""

from __future__ import annotations

import decimal
from typing import TypeVar, assert_never, cast

from agm.agl.ir.operations import ArithKind, CmpOp, ContainsKind, NumericKind
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    DecimalValue,
    DictValue,
    IntValue,
    TextValue,
    Value,
    value_equal,
)
from agm.util.decimal import compare_numbers, exact_decimal, int_in_range

__all__ = [
    "add",
    "contains",
    "div",
    "logical_not",
    "mul",
    "negate",
    "order",
    "sub",
    "value_eq",
]


def value_eq(left: Value, right: Value) -> bool:
    """Value equality; an int and a decimal compare exactly.

    Delegates to ``value_equal``, whose structural comparison is cycle-safe
    and co-inductive, so ``==`` on a cyclic structured value terminates
    instead of recursing forever.
    """
    return value_equal(left, right)


_Ordered = TypeVar("_Ordered", int, decimal.Decimal, str)


def _cmp(op: CmpOp, lv: _Ordered, rv: _Ordered) -> bool:
    if op == CmpOp.LT:
        return lv < rv
    if op == CmpOp.LE:
        return lv <= rv
    if op == CmpOp.GT:
        return lv > rv
    return lv >= rv


def order(op: CmpOp, left: Value, right: Value) -> bool:
    """Ordering comparison (LT/LE/GT/GE) of two texts, or of two numbers compared exactly."""
    if isinstance(left, IntValue) and isinstance(right, IntValue):
        return _cmp(op, left.value, right.value)
    if isinstance(left, DecimalValue) and isinstance(right, DecimalValue):
        return _cmp(op, left.value, right.value)
    if isinstance(left, TextValue):
        right = cast(TextValue, right)
        return _cmp(op, left.value, right.value)
    left = cast("IntValue | DecimalValue", left)
    right = cast("IntValue | DecimalValue", right)
    return _cmp(op, compare_numbers(left.value, right.value), 0)


def contains(kind: ContainsKind, item: Value, container: Value) -> bool:
    """Containment check for array/dict/text."""
    match kind:
        case ContainsKind.ARRAY:
            container = cast(ArrayValue, container)
            return any(value_eq(item, elem) for elem in container.elements)
        case ContainsKind.DICT:
            container = cast(DictValue, container)
            return container.lookup(item) is not None
        case ContainsKind.DICT_INT_NEEDLE:
            container = cast(DictValue, container)
            needle = cast(IntValue, item).value
            return (
                int_in_range(needle)
                and container.lookup(DecimalValue(exact_decimal(needle))) is not None
            )
        case ContainsKind.TEXT:
            container = cast(TextValue, container)
            item = cast(TextValue, item)
            return item.value in container.value
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def add(kind: ArithKind, left: Value, right: Value) -> Value:
    """Addition: INT or DECIMAL."""
    match kind:
        case ArithKind.INT:
            left = cast(IntValue, left)
            right = cast(IntValue, right)
            return IntValue(left.value + right.value)
        case ArithKind.DECIMAL:
            left = cast(DecimalValue, left)
            right = cast(DecimalValue, right)
            return DecimalValue(left.value + right.value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def sub(kind: ArithKind, left: Value, right: Value) -> Value:
    """Subtraction: INT or DECIMAL."""
    match kind:
        case ArithKind.INT:
            left = cast(IntValue, left)
            right = cast(IntValue, right)
            return IntValue(left.value - right.value)
        case ArithKind.DECIMAL:
            left = cast(DecimalValue, left)
            right = cast(DecimalValue, right)
            return DecimalValue(left.value - right.value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def mul(kind: ArithKind, left: Value, right: Value) -> Value:
    """Multiplication: INT or DECIMAL."""
    match kind:
        case ArithKind.INT:
            left = cast(IntValue, left)
            right = cast(IntValue, right)
            return IntValue(left.value * right.value)
        case ArithKind.DECIMAL:
            left = cast(DecimalValue, left)
            right = cast(DecimalValue, right)
            return DecimalValue(left.value * right.value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def div(left: DecimalValue, right: DecimalValue) -> Value:
    """Division: always DECIMAL. A trapped signal (zero divisor, overflow) propagates.

    A zero divisor raises ``decimal.DivisionByZero`` up front, so ``0 / 0`` is
    a division by zero too -- the context itself reports it only as the
    coarse ``InvalidOperation``.
    """
    if right.value.is_zero():
        raise decimal.DivisionByZero()
    return DecimalValue(left.value / right.value)


def negate(kind: NumericKind, value: Value) -> Value:
    """Unary negation: INT or DECIMAL. Decimal negation is exact and cannot overflow."""
    match kind:
        case NumericKind.INT:
            value = cast(IntValue, value)
            return IntValue(-value.value)
        case NumericKind.DECIMAL:
            value = cast(DecimalValue, value)
            # `copy_negate` flips the sign bit only: context-free, never rounds,
            # never raises -- unlike `-value.value`, which rounds to the
            # ambient context's precision and could overflow on a carry.
            return DecimalValue(value.value.copy_negate())
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def logical_not(value: BoolValue) -> BoolValue:
    """Logical NOT: BoolValue → BoolValue."""
    return BoolValue(not value.value)
