"""Pure, context-free number helpers: the pinned decimal context, its range, and exact conversions.

Agm-import-free leaf: only ``decimal``/``math``/``typing`` from the standard library.
Every AgL decimal value, at every point it is created, is
finite and lies within :data:`AGL_DECIMAL_CONTEXT`'s exponent range --
:func:`decimal_in_range` (or, for an ``int`` not yet converted,
:func:`int_in_range`) is the shared test every creation site validates
against. Alongside: exact JSON-number parsing, decimal-to-int narrowing, and
exact int/decimal comparison. The AgL-specific signal/raise machinery built
on top of this lives in ``agm.agl.semantics.arithmetic``.
"""

from __future__ import annotations

import decimal
import math
from typing import cast

__all__ = [
    "AGL_DECIMAL_CONTEXT",
    "compare_numbers",
    "decimal_in_range",
    "exact_decimal",
    "int_in_range",
    "integral_to_int",
    "narrows_to_int",
    "parse_json_decimal",
    "reject_json_constant",
    "strip_trailing_zeros",
]

#: Pinned decimal context for all AgL arithmetic and decimal creation.
#:
#: Design: a host that lowers ``getcontext().prec`` would otherwise change
#: results such as ``1 / 3``. Every field is passed explicitly -- nothing is
#: inherited from ``decimal.DefaultContext`` -- so the context is part of the
#: language specification, not an artifact of the Python decimal module's own
#: defaults. Applied as the ambient context for the whole program run by
#: ``eval.ir_interpreter.IrInterpreter.run``. ``Underflow``/``Subnormal``/
#: ``Inexact``/``Rounded``/``Clamped`` are deliberately untrapped: an
#: arithmetic result too small for the range underflows toward zero instead
#: of raising.
AGL_DECIMAL_CONTEXT: decimal.Context = decimal.Context(
    prec=28,
    rounding=decimal.ROUND_HALF_EVEN,
    Emax=999999,
    Emin=-999999,
    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
    clamp=0,
)

_ETINY: int = AGL_DECIMAL_CONTEXT.Etiny()


def _pow10(exponent: int) -> int:
    """``10 ** exponent`` for a non-negative *exponent*, exact.

    Exponentiation by squaring using only ``*`` -- unlike ``10 ** exponent``
    or ``pow(10, exponent)``, whose typeshed signature types a non-literal
    int exponent as ``Any`` (the sign-dependent overload can't be picked
    statically), which the pinned project mypy configuration disallows.
    """
    result = 1
    base = 10
    while exponent > 0:
        if exponent & 1:
            result *= base
        base *= base
        exponent >>= 1
    return result


def _bits_for_power_of_ten(exponent: int) -> int:
    """Exact bit length of ``10**exponent``, without constructing it.

    ``10**exponent`` is never itself a power of two, so its bit length is
    ``floor(exponent * log2(10)) + 1`` -- verified against the exact value by
    a unit test for the one *exponent* this module calls it with. Computing
    the actual ``10**exponent`` integer just to measure it costs O(digits^2)
    and, for the pinned range's own size, measurable time and memory at
    import.
    """
    return math.floor(exponent * math.log2(10)) + 1


#: Exclusive upper bound, in bits, on the magnitude an in-range integer can
#: have: ``|n|`` is in range iff ``|n|.bit_length() <= _INT_RANGE_THRESHOLD_BITS``,
#: with the tie broken by :func:`_int_range_threshold` (see :func:`int_in_range`).
_INT_RANGE_THRESHOLD_BITS: int = _bits_for_power_of_ten(AGL_DECIMAL_CONTEXT.Emax + 1)


_int_range_threshold_cache: int | None = None


def _int_range_threshold() -> int:
    """The exact ``10**(Emax+1)`` integer: ``|n|`` is in range iff ``|n| < this``.

    Built lazily, once (a manual cache, not ``functools.cache``: the pinned
    project mypy configuration disallows the ``Any`` its decorator
    introduces), only when :func:`int_in_range` actually falls into the
    narrow band where the bit-length estimate alone cannot decide -- every
    other call answers from ``bit_length()`` alone, without ever
    constructing this value.
    """
    global _int_range_threshold_cache
    cached = _int_range_threshold_cache
    if cached is None:
        cached = _pow10(AGL_DECIMAL_CONTEXT.Emax + 1)
        _int_range_threshold_cache = cached
    return cached


def strip_trailing_zeros(value: decimal.Decimal) -> decimal.Decimal:
    """Return *value* with trailing coefficient zeros dropped, exactly and context-free.

    The unique minimal-coefficient representation of the same exact value:
    never rounds, never raises, and never consults the ambient decimal
    context (unlike :meth:`decimal.Decimal.normalize`). Shared by every
    renderer that must emit a decimal without a context dependency -- AgL
    surface/value-syntax spelling and exact JSON-number text -- and by
    :func:`decimal_in_range`, whose Etiny test is about the value's true
    least-significant digit, not merely how it happened to be spelled.
    """
    sign, digits, raw_exponent = value.as_tuple()
    # Finite: only NaN and infinity carry a letter exponent.
    exponent = cast(int, raw_exponent)
    if digits == (0,):
        return decimal.Decimal((sign, (0,), 0))
    stripped = list(digits)
    while len(stripped) > 1 and stripped[-1] == 0:
        stripped.pop()
        exponent += 1
    return decimal.Decimal((sign, tuple(stripped), exponent))


def decimal_in_range(value: decimal.Decimal) -> bool:
    """Whether *value* is finite and within the pinned context's exponent range.

    Range only, never precision: a value with more than 28 significant
    digits is in range as long as its most-significant digit's exponent does
    not exceed ``Emax`` and its least-significant digit's exponent is not
    below ``Etiny``. Zero is always in range, regardless of how it is
    spelled (``0e2000000`` is zero, not an out-of-range magnitude). The
    Etiny test is against *value*'s trailing-zero-stripped exponent, not its
    literal spelling: ``1.000e-1000026`` and ``1e-1000026`` carry the same
    information and must be judged the same way. Never rounds.
    """
    if not value.is_finite():
        return False
    if value.is_zero():
        return True
    stripped = strip_trailing_zeros(value)
    # Finite: only NaN and infinity carry a letter exponent.
    exponent = cast(int, stripped.as_tuple().exponent)
    return stripped.adjusted() <= AGL_DECIMAL_CONTEXT.Emax and exponent >= _ETINY


def int_in_range(n: int) -> bool:
    """Whether AgL int *n* widens to a decimal within the pinned range.

    Decided from ``n.bit_length()`` wherever possible -- ``Decimal(n)``
    construction is quadratic in digit count, and the pinned range needs
    well over a million digits to overflow, so a huge out-of-range *n* must
    never be converted just to find that out. Only the narrow band where
    ``n``'s bit length ties the threshold's own falls back to
    :func:`_int_range_threshold` (cheap: integer comparison is linear, not
    quadratic, in digit count).
    """
    magnitude = abs(n)
    bit_len = magnitude.bit_length()
    if bit_len < _INT_RANGE_THRESHOLD_BITS:
        return True
    if bit_len > _INT_RANGE_THRESHOLD_BITS:
        return False
    return magnitude < _int_range_threshold()


def parse_json_decimal(text: str) -> decimal.Decimal:
    """Parse a JSON/TOML number token exactly, as the ``parse_float`` hook of every decoder.

    Raises :exc:`ValueError` -- which every decoder boundary maps to its own
    parse error -- for a non-finite token and for an exponent no decimal can
    hold (``1e99999999999999999999``). Range is not checked: an out-of-range
    finite value is a valid ``json`` number, rejected only where it is narrowed.
    """
    try:
        value = decimal.Decimal(text, AGL_DECIMAL_CONTEXT)
    except decimal.DecimalException as exc:
        raise ValueError(f"unrepresentable number {text!r}") from exc
    if not value.is_finite():
        raise ValueError(f"non-finite number {text!r}")
    return value


def reject_json_constant(token: str) -> object:
    """Raise :exc:`ValueError` for ``NaN``/``Infinity``/``-Infinity``, wherever it appears.

    The ``parse_constant`` hook of every JSON decoder.
    """
    raise ValueError(f"non-finite JSON constant {token!r}")


def narrows_to_int(value: decimal.Decimal) -> bool:
    """Whether *value* is an integral decimal within the pinned range, so it narrows to an int.

    The range bounds the narrowed int's size: a ``json`` or wire number is
    exempt from it, and ``1e999999999999`` must not become a
    trillion-digit int.
    """
    return decimal_in_range(value) and value == value.to_integral_value()


def integral_to_int(value: decimal.Decimal) -> int:
    """The exact ``int`` equal to the finite, integral *value*.

    Converts through decimal text: ``int(Decimal)`` is quadratic in digit
    count, text conversion is subquadratic both ways.
    """
    return int(format(value.to_integral_value(), "f"))


def exact_decimal(n: int) -> decimal.Decimal:
    """The exact ``Decimal`` equal to *n*, through text as in :func:`integral_to_int`."""
    return decimal.Decimal(str(n))


_LOG10_2: float = math.log10(2)

#: Up to this bit length an int is compared with a decimal natively: CPython
#: converts it exactly and cheaply. Beyond it, :func:`_compare_magnitudes`.
_NATIVE_COMPARISON_BITS = 10_000


def _compare_magnitudes(magnitude: int, value: decimal.Decimal) -> int:
    """Sign of ``magnitude - value`` for a positive int and a positive finite decimal.

    Decided from ``bit_length()`` and ``adjusted()`` whenever the orders of
    magnitude differ (with a one-digit margin for float error). Otherwise
    *value*'s integer part, no longer than *magnitude* by more than a couple
    of digits, decides; on a tie, a nonzero fraction makes *value* larger.
    Never scales either side by a power of ten.
    """
    bits = magnitude.bit_length()
    adjusted = value.adjusted()
    if adjusted + 1 < math.floor((bits - 1) * _LOG10_2):
        return 1
    if adjusted > math.floor(bits * _LOG10_2) + 1:
        return -1
    floor = value.to_integral_value(rounding=decimal.ROUND_FLOOR)
    whole = integral_to_int(floor)
    if magnitude != whole:
        return 1 if magnitude > whole else -1
    return 0 if floor == value else -1


def compare_numbers(left: int | decimal.Decimal, right: int | decimal.Decimal) -> int:
    """Sign (-1/0/1) of the exact difference ``left - right`` of two finite numbers.

    Never rounds and never raises: a mixed int/decimal pair is compared
    without widening the int, whatever its magnitude, and in time
    subquadratic in both operands' digit counts.
    """
    if isinstance(left, int) and isinstance(right, decimal.Decimal):
        return -compare_numbers(right, left)
    if (
        isinstance(left, decimal.Decimal)
        and isinstance(right, int)
        and right.bit_length() > _NATIVE_COMPARISON_BITS
    ):
        right_sign = 1 if right > 0 else -1
        left_sign = 0 if left.is_zero() else (-1 if left.is_signed() else 1)
        if left_sign != right_sign:
            return 1 if left_sign > right_sign else -1
        return -right_sign * _compare_magnitudes(abs(right), left.copy_abs())
    return (left > right) - (left < right)
