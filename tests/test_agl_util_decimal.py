"""Unit tests for the pure, context-free decimal helpers in ``agm.util.decimal``.

These are private-module tests by exception: each function here is a pure,
stable contract (range predicates over the pinned decimal context) that is
clearer to verify directly than through an AgL program.
"""

from __future__ import annotations

import decimal

from agm.util.decimal import (
    AGL_DECIMAL_CONTEXT,
    decimal_in_range,
    int_in_range,
    strip_trailing_zeros,
)


class TestPinnedContext:
    def test_precision_and_rounding(self) -> None:
        assert AGL_DECIMAL_CONTEXT.prec == 28
        assert AGL_DECIMAL_CONTEXT.rounding == decimal.ROUND_HALF_EVEN

    def test_traps_only_invalid_division_overflow(self) -> None:
        assert AGL_DECIMAL_CONTEXT.traps[decimal.InvalidOperation] is True
        assert AGL_DECIMAL_CONTEXT.traps[decimal.DivisionByZero] is True
        assert AGL_DECIMAL_CONTEXT.traps[decimal.Overflow] is True
        assert AGL_DECIMAL_CONTEXT.traps[decimal.Underflow] is False
        assert AGL_DECIMAL_CONTEXT.traps[decimal.Subnormal] is False


class TestStripTrailingZeros:
    def test_drops_trailing_coefficient_zeros(self) -> None:
        result = strip_trailing_zeros(decimal.Decimal("1.2300"))
        assert result.as_tuple() == decimal.Decimal("1.23").as_tuple()

    def test_no_trailing_zeros_is_unchanged(self) -> None:
        value = decimal.Decimal("1.23")
        assert strip_trailing_zeros(value) == value

    def test_zero_normalizes_to_canonical_zero(self) -> None:
        result = strip_trailing_zeros(decimal.Decimal("0.000"))
        assert result == decimal.Decimal("0")
        assert result.as_tuple() == (0, (0,), 0)

    def test_never_rounds_past_the_ambient_context(self) -> None:
        # 40 significant digits, all non-zero: stripping keeps every one, even
        # though the ambient context's own precision is only 28.
        with decimal.localcontext(AGL_DECIMAL_CONTEXT):
            value = decimal.Decimal("1" * 40)
            result = strip_trailing_zeros(value)
        assert result == value


class TestDecimalInRange:
    def test_ordinary_value_in_range(self) -> None:
        assert decimal_in_range(decimal.Decimal("1.5")) is True

    def test_non_finite_is_rejected(self) -> None:
        assert decimal_in_range(decimal.Decimal("Infinity")) is False
        assert decimal_in_range(decimal.Decimal("-Infinity")) is False
        assert decimal_in_range(decimal.Decimal("NaN")) is False

    def test_zero_is_always_in_range_regardless_of_spelling(self) -> None:
        """`0e2000000` is zero, not an out-of-range magnitude."""
        assert decimal_in_range(decimal.Decimal("0")) is True
        assert decimal_in_range(decimal.Decimal((0, (0,), 2_000_000))) is True
        assert decimal_in_range(decimal.Decimal((1, (0,), -2_000_000))) is True

    def test_adjusted_exponent_above_emax_is_rejected(self) -> None:
        emax = AGL_DECIMAL_CONTEXT.Emax
        assert decimal_in_range(decimal.Decimal((0, (1,), emax))) is True
        assert decimal_in_range(decimal.Decimal((0, (1,), emax + 1))) is False

    def test_etiny_boundary_after_stripping_trailing_zeros(self) -> None:
        """A value whose *stored* exponent understates its true minimal
        exponent (because of trailing coefficient zeros) is judged by the
        stripped exponent, not the literal spelling."""
        etiny = AGL_DECIMAL_CONTEXT.Etiny()
        # Stored exponent is one below Etiny, but the trailing zero strips
        # away, leaving the true (in-range) exponent at Etiny.
        spelled_with_trailing_zero = decimal.Decimal((0, (1, 0), etiny - 1))
        assert decimal_in_range(spelled_with_trailing_zero) is True
        # No trailing zero to strip: genuinely below Etiny.
        genuinely_below_etiny = decimal.Decimal((0, (1,), etiny - 1))
        assert decimal_in_range(genuinely_below_etiny) is False


class TestIntInRange:
    def test_small_int_in_range(self) -> None:
        assert int_in_range(0) is True
        assert int_in_range(-1) is True
        assert int_in_range(10**27) is True

    def test_huge_int_out_of_range(self) -> None:
        assert int_in_range(10**1_000_000) is False
        assert int_in_range(-(10**1_000_000)) is False

    def test_boundary_around_the_exact_threshold(self) -> None:
        threshold = 10 ** (AGL_DECIMAL_CONTEXT.Emax + 1)
        assert int_in_range(threshold - 1) is True
        assert int_in_range(threshold) is False
        assert int_in_range(-(threshold - 1)) is True
        assert int_in_range(-threshold) is False

    def test_matches_decimal_in_range_at_the_boundary(self) -> None:
        """int_in_range(n) agrees with decimal_in_range on the same boundary
        magnitude -- the bit-length shortcut must not disagree with the exact
        predicate it approximates. The boundary decimals are built directly
        from a ``(sign, digits, exponent)`` tuple rather than by converting a
        million-digit int: that conversion is quadratic in digit count, and
        this test must stay cheap (see int_in_range's own docstring)."""
        emax = AGL_DECIMAL_CONTEXT.Emax
        threshold = 10 ** (emax + 1)
        just_in_range = decimal.Decimal((0, (9,) * (emax + 1), 0))  # 10**(emax+1) - 1
        just_out_of_range = decimal.Decimal((0, (1,), emax + 1))  # 10**(emax+1)
        assert int_in_range(threshold - 1) == decimal_in_range(just_in_range) is True
        assert int_in_range(threshold) == decimal_in_range(just_out_of_range) is False


def test_bit_length_threshold_is_exact() -> None:
    """The cached bit-length threshold this module derives arithmetically
    matches the exact bit length of ``10**(Emax+1)``, computed directly."""
    from agm.util.decimal import _INT_RANGE_THRESHOLD_BITS

    exact = (10 ** (AGL_DECIMAL_CONTEXT.Emax + 1)).bit_length()
    assert _INT_RANGE_THRESHOLD_BITS == exact
