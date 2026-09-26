"""AgL arithmetic signals: converting a trapped decimal exception (or an
out-of-range creation-time check) into the catchable ``ArithmeticError``.

The pinned decimal context and the pure range predicates (:data:`AGL_DECIMAL_CONTEXT`,
``decimal_in_range``, ``int_in_range``) live in the agm-import-free leaf
:mod:`agm.util.decimal`; this module builds the AgL-specific signal/exception
machinery on top of them.

Two independent layers enforce the range invariant:

- AgL-native arithmetic (``eval.arith``, ``eval.ir_interpreter``) runs under
  the pinned context as its ambient ``decimal.localcontext``, so each
  operator's own result is rounded to 28 significant digits and raises on a
  trapped signal (``InvalidOperation``, ``DivisionByZero``, ``Overflow``).
  :func:`signal_kind_for` classifies the trapped Python exception; a caller
  converts the sentinel into the catchable ``ArithmeticError`` via
  :func:`arithmetic_signal_raise`.
- Every other creation site -- int-to-decimal widening, inbound boundaries
  (JSON/wire decoding, casts, host arguments, extern return values), decimal
  literals -- validates against ``decimal_in_range`` (or, for an ``int`` not
  yet converted, the cheaper ``int_in_range``) *without* rounding, and
  rejects a violation with that site's own error. :func:`checked_decimal`
  is the single shared validator these sites call.
"""

from __future__ import annotations

import decimal
import enum
from typing import assert_never

from agm.agl.ir.builtin_nominals import BuiltinNominals
from agm.agl.ir.operations import AS_DECIMAL_OPERATION
from agm.agl.semantics.exceptions import AglRaise, make_builtin_exception
from agm.agl.semantics.values import TextValue
from agm.util.decimal import decimal_in_range, int_in_range

__all__ = [
    "AglArithmeticSignal",
    "ArithmeticSignalKind",
    "arithmetic_message",
    "arithmetic_signal_raise",
    "checked_decimal",
    "int_to_decimal",
    "signal_kind_for",
]


class ArithmeticSignalKind(enum.Enum):
    """Which trapped ``decimal`` signal an :class:`AglArithmeticSignal` carries."""

    OVERFLOW = "overflow"
    DIVISION_BY_ZERO = "division_by_zero"
    INVALID_OPERATION = "invalid_operation"


class AglArithmeticSignal(Exception):
    """Sentinel: a ``decimal`` operation raised a trapped Python signal, or a
    creation-time range check failed.

    *operation* names the operator or conversion that failed (e.g. ``"+"``,
    ``"/"``, ``"as decimal"``, a ``std/math`` function name). *kind* selects
    the ``ArithmeticError`` message text. A caller converts this into a
    catchable ``ArithmeticError`` via :func:`arithmetic_signal_raise`.
    """

    def __init__(self, operation: str, kind: ArithmeticSignalKind) -> None:
        super().__init__(operation)
        self.operation = operation
        self.kind = kind


def signal_kind_for(exc: decimal.DecimalException) -> ArithmeticSignalKind:
    """Classify a trapped Python ``decimal`` exception into its signal kind.

    ``ZeroDivisionError`` covers every division-by-zero-family signal,
    including ``decimal.DivisionUndefined``.
    """
    if isinstance(exc, ZeroDivisionError):
        return ArithmeticSignalKind.DIVISION_BY_ZERO
    if isinstance(exc, decimal.Overflow):
        return ArithmeticSignalKind.OVERFLOW
    return ArithmeticSignalKind.INVALID_OPERATION


def arithmetic_message(operation: str, kind: ArithmeticSignalKind) -> str:
    """Build the ``ArithmeticError`` message text for *operation* and *kind*."""
    match kind:
        case ArithmeticSignalKind.DIVISION_BY_ZERO:
            return f"Division by zero in {operation!r}"
        case ArithmeticSignalKind.OVERFLOW:
            return f"Arithmetic operation {operation!r} is out of range"
        case ArithmeticSignalKind.INVALID_OPERATION:
            return f"Arithmetic operation {operation!r} is invalid"
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def checked_decimal(value: int | decimal.Decimal) -> decimal.Decimal:
    """Return *value* as an exact decimal within the pinned range, or raise ``ValueError``.

    The single shared creation-time validator for every non-arithmetic
    decimal creation site: int-to-decimal widening, JSON/wire decoding,
    casts, host arguments, and extern return values. Never rounds. The
    message states the bound that was violated, never the rejected value
    itself: an out-of-range value can carry well over a million digits, and
    the span or raw input that produced it already identifies what was
    rejected.
    """
    if isinstance(value, int):
        if not int_in_range(value):
            raise ValueError("int is outside the decimal range")
        return decimal.Decimal(value)
    if not decimal_in_range(value):
        raise ValueError("decimal is outside the pinned range")
    return value


def int_to_decimal(n: int, operation: str = AS_DECIMAL_OPERATION) -> decimal.Decimal:
    """Convert an AgL int to an exact, range-checked decimal.

    Raises :class:`AglArithmeticSignal` (kind ``OVERFLOW``, labelled
    *operation*) when *n* is outside the pinned context's range -- decided
    cheaply via ``int_in_range``, never by constructing ``Decimal(n)`` for a
    huge out-of-range *n* just to reject it. An in-range *n* widens exactly,
    with no rounding. The single shared conversion for every
    AgL-reachable int-to-decimal site: the ``as``/``as?`` cast and the
    implicit coercion default to *operation* :data:`AS_DECIMAL_OPERATION`; a mixed
    binary operator's own int operand instead labels *operation* with the
    operator itself (``lower.lowerer._lower_arith`` and friends).
    """
    try:
        return checked_decimal(n)
    except ValueError as exc:
        raise AglArithmeticSignal(operation, ArithmeticSignalKind.OVERFLOW) from exc


def arithmetic_signal_raise(exc: AglArithmeticSignal, *, nominals: BuiltinNominals) -> AglRaise:
    """Build the catchable ``AglRaise(ArithmeticError)`` for a decimal signal.

    Single shared constructor so every caller that converts an
    :class:`AglArithmeticSignal` sentinel produces identically shaped fields.
    """
    return AglRaise(
        make_builtin_exception(
            "ArithmeticError",
            arithmetic_message(exc.operation, exc.kind),
            nominals=nominals,
            fields={"operation": TextValue(exc.operation)},
        )
    )
