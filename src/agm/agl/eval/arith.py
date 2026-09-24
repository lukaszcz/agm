"""Pure arithmetic and comparison helpers for the AgL evaluator.

Used by the IR evaluator.
This module is the single source of truth for operator semantics.

IMPORTANT: Only imports from stdlib, agm.agl.semantics.values, and agm.agl.ir.operations.
No syntax, scope, or typecheck imports are permitted here.

Every DECIMAL-kind operand here is already a ``DecimalValue``: mixed int/decimal
operands are widened at compile time by the lowerer's ``IntToDecimal``
coercion (``lower.lowerer``), labelled with the triggering operator, so this
module never widens an int itself. A trapped ``decimal.DecimalException``
(overflow, division by zero, invalid operation) propagates uncaught to the
caller, which classifies and labels it (``eval.ir_interpreter``).
"""

from __future__ import annotations

import decimal
from typing import TypeVar, assert_never

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
    """Value equality with int↔decimal widening.

    Delegates the structural comparison to ``values_equal``, which is
    cycle-safe and co-inductive, so ``==`` on a cyclic structured value
    terminates instead of recursing forever. The widening here only applies
    at this top level, never inside a container (unchanged from before).
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
    """Ordering comparison (LT/LE/GT/GE). Operands are already same-typed."""
    if op not in (CmpOp.LT, CmpOp.LE, CmpOp.GT, CmpOp.GE):
        raise AssertionError(f"order: non-ordering op {op!r}")
    if isinstance(left, IntValue) and isinstance(right, IntValue):
        return _cmp(op, left.value, right.value)
    if isinstance(left, DecimalValue) and isinstance(right, DecimalValue):
        return _cmp(op, left.value, right.value)
    if isinstance(left, TextValue) and isinstance(right, TextValue):
        return _cmp(op, left.value, right.value)
    raise AssertionError(f"order: cannot compare {type(left).__name__} and {type(right).__name__}")


def contains(kind: ContainsKind, item: Value, container: Value) -> bool:
    """Containment check for array/dict/text."""
    match kind:
        case ContainsKind.ARRAY:
            if not isinstance(container, ArrayValue):
                raise AssertionError(
                    f"contains ARRAY: expected ArrayValue, got {type(container).__name__}"
                )
            return any(value_eq(item, elem) for elem in container.elements)
        case ContainsKind.DICT:
            if not isinstance(container, DictValue):
                raise AssertionError(
                    f"contains DICT: expected DictValue, got {type(container).__name__}"
                )
            return container.lookup(item) is not None
        case ContainsKind.TEXT:
            if not isinstance(container, TextValue) or not isinstance(item, TextValue):
                raise AssertionError("contains TEXT: expected TextValue+TextValue")
            return item.value in container.value
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def add(kind: ArithKind, left: Value, right: Value) -> Value:
    """Addition: INT or DECIMAL."""
    match kind:
        case ArithKind.INT:
            if not isinstance(left, IntValue) or not isinstance(right, IntValue):
                raise AssertionError(
                    f"add INT: expected IntValue+IntValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return IntValue(left.value + right.value)
        case ArithKind.DECIMAL:
            if not isinstance(left, DecimalValue) or not isinstance(right, DecimalValue):
                raise AssertionError(
                    f"add DECIMAL: expected DecimalValue+DecimalValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return DecimalValue(left.value + right.value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def sub(kind: ArithKind, left: Value, right: Value) -> Value:
    """Subtraction: INT or DECIMAL."""
    match kind:
        case ArithKind.INT:
            if not isinstance(left, IntValue) or not isinstance(right, IntValue):
                raise AssertionError(
                    f"sub INT: expected IntValue+IntValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return IntValue(left.value - right.value)
        case ArithKind.DECIMAL:
            if not isinstance(left, DecimalValue) or not isinstance(right, DecimalValue):
                raise AssertionError(
                    f"sub DECIMAL: expected DecimalValue+DecimalValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return DecimalValue(left.value - right.value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def mul(kind: ArithKind, left: Value, right: Value) -> Value:
    """Multiplication: INT or DECIMAL."""
    match kind:
        case ArithKind.INT:
            if not isinstance(left, IntValue) or not isinstance(right, IntValue):
                raise AssertionError(
                    f"mul INT: expected IntValue+IntValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return IntValue(left.value * right.value)
        case ArithKind.DECIMAL:
            if not isinstance(left, DecimalValue) or not isinstance(right, DecimalValue):
                raise AssertionError(
                    f"mul DECIMAL: expected DecimalValue+DecimalValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return DecimalValue(left.value * right.value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def div(left: Value, right: Value) -> Value:
    """Division: always DECIMAL. A trapped signal (zero divisor, overflow) propagates.

    A zero divisor raises ``decimal.DivisionByZero`` up front, so ``0 / 0`` is
    a division by zero too -- the context itself reports it only as the
    coarse ``InvalidOperation``.
    """
    if not isinstance(left, DecimalValue) or not isinstance(right, DecimalValue):
        raise AssertionError(
            f"div: expected DecimalValue+DecimalValue, got"
            f" {type(left).__name__}+{type(right).__name__}"
        )
    if right.value.is_zero():
        raise decimal.DivisionByZero()
    return DecimalValue(left.value / right.value)


def negate(kind: NumericKind, value: Value) -> Value:
    """Unary negation: INT or DECIMAL. Decimal negation is exact and cannot overflow."""
    match kind:
        case NumericKind.INT:
            if not isinstance(value, IntValue):
                raise AssertionError(f"negate INT: expected IntValue, got {type(value).__name__}")
            return IntValue(-value.value)
        case NumericKind.DECIMAL:
            if not isinstance(value, DecimalValue):
                raise AssertionError(
                    f"negate DECIMAL: expected DecimalValue, got {type(value).__name__}"
                )
            # `copy_negate` flips the sign bit only: context-free, never rounds,
            # never raises -- unlike `-value.value`, which rounds to the
            # ambient context's precision and could overflow on a carry.
            return DecimalValue(value.value.copy_negate())
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def logical_not(value: Value) -> BoolValue:
    """Logical NOT: BoolValue → BoolValue."""
    if not isinstance(value, BoolValue):
        raise AssertionError(f"logical_not: expected BoolValue, got {type(value).__name__}")
    return BoolValue(not value.value)
