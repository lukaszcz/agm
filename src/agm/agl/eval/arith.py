"""Pure arithmetic and comparison helpers for the AgL evaluator.

Used by the IR evaluator.
This module is the single source of truth for operator semantics.

IMPORTANT: Only imports from stdlib, agm.agl.eval.values, and agm.agl.ir.operations.
No syntax, scope, or typecheck imports are permitted here.
"""

from __future__ import annotations

import decimal
from typing import TypeVar, assert_never

from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    IntValue,
    ListValue,
    TextValue,
    Value,
)
from agm.agl.ir.operations import ArithKind, CmpOp, ContainsKind, NumericKind

__all__ = [
    "AglDivisionByZero",
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


class AglDivisionByZero(Exception):
    """Sentinel raised by div() on a zero divisor.

    Each evaluator catches this and wraps it into its own AglRaise with its
    own trace_id (preserving legacy behavior).
    """


def _to_decimal(value: IntValue | DecimalValue) -> decimal.Decimal:
    if isinstance(value, IntValue):
        return decimal.Decimal(value.value)
    return value.value


def value_eq(left: Value, right: Value) -> bool:
    """Value equality with int↔decimal widening."""
    if isinstance(left, IntValue) and isinstance(right, DecimalValue):
        return decimal.Decimal(left.value) == right.value
    if isinstance(left, DecimalValue) and isinstance(right, IntValue):
        return left.value == decimal.Decimal(right.value)
    return left == right


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
    """Ordering comparison (LT/LE/GT/GE) with int↔decimal widening."""
    if op not in (CmpOp.LT, CmpOp.LE, CmpOp.GT, CmpOp.GE):
        raise AssertionError(f"order: non-ordering op {op!r}")
    # Widen int for mixed numeric ordering.
    if isinstance(left, IntValue) and isinstance(right, DecimalValue):
        left = DecimalValue(decimal.Decimal(left.value))
    elif isinstance(left, DecimalValue) and isinstance(right, IntValue):
        right = DecimalValue(decimal.Decimal(right.value))

    if isinstance(left, IntValue) and isinstance(right, IntValue):
        return _cmp(op, left.value, right.value)
    if isinstance(left, DecimalValue) and isinstance(right, DecimalValue):
        return _cmp(op, left.value, right.value)
    if isinstance(left, TextValue) and isinstance(right, TextValue):
        return _cmp(op, left.value, right.value)
    raise AssertionError(
        f"order: cannot compare {type(left).__name__} and {type(right).__name__}"
    )


def contains(kind: ContainsKind, item: Value, container: Value) -> bool:
    """Containment check for list/dict/text."""
    match kind:
        case ContainsKind.LIST:
            if not isinstance(container, ListValue):
                raise AssertionError(
                    f"contains LIST: expected ListValue, got {type(container).__name__}"
                )
            return any(value_eq(item, elem) for elem in container.elements)
        case ContainsKind.DICT:
            if not isinstance(container, DictValue):
                raise AssertionError(
                    f"contains DICT: expected DictValue, got {type(container).__name__}"
                )
            if isinstance(item, TextValue):
                return item.value in container.entries
            return False
        case ContainsKind.TEXT:
            if not isinstance(container, TextValue) or not isinstance(item, TextValue):
                raise AssertionError(
                    "contains TEXT: expected TextValue+TextValue"
                )
            return item.value in container.value
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def add(kind: ArithKind, left: Value, right: Value) -> Value:
    """Addition/concatenation: INT, DECIMAL (with widening), or TEXT."""
    match kind:
        case ArithKind.INT:
            if not isinstance(left, IntValue) or not isinstance(right, IntValue):
                raise AssertionError(
                    f"add INT: expected IntValue+IntValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return IntValue(left.value + right.value)
        case ArithKind.DECIMAL:
            if not isinstance(left, (IntValue, DecimalValue)) or not isinstance(
                right, (IntValue, DecimalValue)
            ):
                raise AssertionError(
                    f"add DECIMAL: expected numeric+numeric, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return DecimalValue(_to_decimal(left) + _to_decimal(right))
        case ArithKind.TEXT:
            if not isinstance(left, TextValue) or not isinstance(right, TextValue):
                raise AssertionError(
                    f"add TEXT: expected TextValue+TextValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return TextValue(left.value + right.value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def sub(kind: ArithKind, left: Value, right: Value) -> Value:
    """Subtraction: INT or DECIMAL only (TEXT is not valid)."""
    match kind:
        case ArithKind.INT:
            if not isinstance(left, IntValue) or not isinstance(right, IntValue):
                raise AssertionError(
                    f"sub INT: expected IntValue+IntValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return IntValue(left.value - right.value)
        case ArithKind.DECIMAL:
            if not isinstance(left, (IntValue, DecimalValue)) or not isinstance(
                right, (IntValue, DecimalValue)
            ):
                raise AssertionError(
                    f"sub DECIMAL: expected numeric+numeric, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return DecimalValue(_to_decimal(left) - _to_decimal(right))
        case ArithKind.TEXT:
            raise AssertionError("sub: TEXT kind is not valid for subtraction")
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def mul(kind: ArithKind, left: Value, right: Value) -> Value:
    """Multiplication: INT or DECIMAL only (TEXT is not valid)."""
    match kind:
        case ArithKind.INT:
            if not isinstance(left, IntValue) or not isinstance(right, IntValue):
                raise AssertionError(
                    f"mul INT: expected IntValue+IntValue, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return IntValue(left.value * right.value)
        case ArithKind.DECIMAL:
            if not isinstance(left, (IntValue, DecimalValue)) or not isinstance(
                right, (IntValue, DecimalValue)
            ):
                raise AssertionError(
                    f"mul DECIMAL: expected numeric+numeric, got"
                    f" {type(left).__name__}+{type(right).__name__}"
                )
            return DecimalValue(_to_decimal(left) * _to_decimal(right))
        case ArithKind.TEXT:
            raise AssertionError("mul: TEXT kind is not valid for multiplication")
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def div(left: Value, right: Value) -> Value:
    """Division: always returns DECIMAL. Raises AglDivisionByZero on zero divisor."""
    if not isinstance(left, (IntValue, DecimalValue)) or not isinstance(
        right, (IntValue, DecimalValue)
    ):
        raise AssertionError(
            f"div: expected numeric+numeric, got {type(left).__name__}+{type(right).__name__}"
        )
    rd = _to_decimal(right)
    if rd == decimal.Decimal(0):
        raise AglDivisionByZero()
    return DecimalValue(_to_decimal(left) / rd)


def negate(kind: NumericKind, value: Value) -> Value:
    """Unary negation: INT or DECIMAL."""
    match kind:
        case NumericKind.INT:
            if not isinstance(value, IntValue):
                raise AssertionError(
                    f"negate INT: expected IntValue, got {type(value).__name__}"
                )
            return IntValue(-value.value)
        case NumericKind.DECIMAL:
            if not isinstance(value, DecimalValue):
                raise AssertionError(
                    f"negate DECIMAL: expected DecimalValue, got {type(value).__name__}"
                )
            return DecimalValue(-value.value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def logical_not(value: Value) -> BoolValue:
    """Logical NOT: BoolValue → BoolValue."""
    if not isinstance(value, BoolValue):
        raise AssertionError(
            f"logical_not: expected BoolValue, got {type(value).__name__}"
        )
    return BoolValue(not value.value)
