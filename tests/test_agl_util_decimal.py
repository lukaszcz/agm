"""Unit tests for the pure, context-free decimal helpers in ``agm.util.decimal``.

These are private-module tests by exception: each function here is a pure,
stable contract (range predicates over the pinned decimal context) that is
clearer to verify directly than through an AgL program.
"""

from __future__ import annotations

import decimal
import fractions
import random

import pytest

from agm.util.decimal import (
    AGL_DECIMAL_CONTEXT,
    compare_numbers,
    decimal_in_range,
    exact_decimal,
    int_in_range,
    integral_to_int,
    narrows_to_int,
    parse_json_decimal,
    reject_json_constant,
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


class TestParseJsonDecimal:
    def test_parses_a_number_token_exactly(self) -> None:
        value = parse_json_decimal("1.50e3")
        assert value.as_tuple() == decimal.Decimal("1.50e3").as_tuple()

    def test_keeps_an_exponent_outside_the_pinned_range(self) -> None:
        assert parse_json_decimal("1e9999999") == decimal.Decimal("1e9999999")

    @pytest.mark.parametrize("token", ["1e99999999999999999999", "-1e-99999999999999999999"])
    def test_exponent_no_decimal_can_hold_is_a_value_error(self, token: str) -> None:
        with pytest.raises(ValueError):
            parse_json_decimal(token)

    @pytest.mark.parametrize("token", ["inf", "-inf", "+inf", "nan", "-nan", "sNaN"])
    def test_non_finite_token_is_a_value_error(self, token: str) -> None:
        with pytest.raises(ValueError):
            parse_json_decimal(token)

    def test_does_not_depend_on_the_ambient_context(self) -> None:
        with decimal.localcontext() as context:
            context.traps[decimal.InvalidOperation] = False
            with pytest.raises(ValueError):
                parse_json_decimal("1e99999999999999999999")


class TestRejectJsonConstant:
    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_every_constant_is_a_value_error(self, token: str) -> None:
        with pytest.raises(ValueError):
            reject_json_constant(token)


class TestNarrowsToInt:
    @pytest.mark.parametrize(
        "text", ["0", "-0.0", "2.0e3", "1e4300", "-1e999999", "0e99999999999", "-12"]
    )
    def test_integral_decimal_within_the_range_narrows(self, text: str) -> None:
        assert narrows_to_int(decimal.Decimal(text)) is True

    @pytest.mark.parametrize("text", ["2.5", "1e-3", "Infinity", "-Infinity", "NaN"])
    def test_fractional_or_non_finite_decimal_does_not_narrow(self, text: str) -> None:
        assert narrows_to_int(decimal.Decimal(text)) is False

    @pytest.mark.parametrize("text", ["1e1000000", "-1e999999999999"])
    def test_integral_decimal_outside_the_range_does_not_narrow(self, text: str) -> None:
        assert narrows_to_int(decimal.Decimal(text)) is False


class TestExactConversions:
    @pytest.mark.parametrize("text", ["0", "-0.00", "3.000", "-12", "1.5e3", "4e5000"])
    def test_integral_to_int_is_exact(self, text: str) -> None:
        value = decimal.Decimal(text)
        assert fractions.Fraction(integral_to_int(value)) == fractions.Fraction(value)

    def test_integral_to_int_never_rounds_under_the_pinned_context(self) -> None:
        digits = "123456789" * 10
        with decimal.localcontext(AGL_DECIMAL_CONTEXT):
            assert integral_to_int(decimal.Decimal(digits + ".000")) == int(digits)

    @pytest.mark.parametrize("n", [0, -7, 10**40 + 1, -(7**9000)])
    def test_exact_decimal_is_exact(self, n: int) -> None:
        with decimal.localcontext(AGL_DECIMAL_CONTEXT):
            assert fractions.Fraction(exact_decimal(n)) == n


class TestCompareNumbers:
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (1, 2, -1),
            (2, 2, 0),
            (decimal.Decimal("1.5"), decimal.Decimal("1.50"), 0),
            (decimal.Decimal("-1.5"), decimal.Decimal("1.5"), -1),
            (2, decimal.Decimal("2.0"), 0),
            (2, decimal.Decimal("1.5"), 1),
            (1, decimal.Decimal("1.5"), -1),
            (0, decimal.Decimal("-0.0"), 0),
            (0, decimal.Decimal("0.5"), -1),
            (0, decimal.Decimal("-0.5"), 1),
            (-1, decimal.Decimal("0"), -1),
            (-2, decimal.Decimal("-1.5"), -1),
            (-1, decimal.Decimal("-1.5"), 1),
            (-1, decimal.Decimal("1.5"), -1),
            (1, decimal.Decimal("0.001"), 1),
            (1, decimal.Decimal("1e-999999999999"), 1),
            (1, decimal.Decimal("1e999999999999"), -1),
            (10**30 + 1, decimal.Decimal("1000000000000000000000000000001.0"), 0),
            (10**30 + 1, decimal.Decimal("1000000000000000000000000000000.5"), 1),
            (10**30 + 1, decimal.Decimal("1000000000000000000000000000001.5"), -1),
            (10**30, decimal.Decimal("1e30"), 0),
            (10**30, decimal.Decimal("1.0000000000000000000000000000001e30"), -1),
        ],
    )
    def test_sign_of_the_exact_difference(
        self, left: int | decimal.Decimal, right: int | decimal.Decimal, expected: int
    ) -> None:
        assert compare_numbers(left, right) == expected
        assert compare_numbers(right, left) == -expected

    def test_int_far_outside_the_decimal_range_is_decided_by_magnitude(self) -> None:
        huge = 2**3_400_000
        assert compare_numbers(huge, decimal.Decimal("1.5")) == 1
        assert compare_numbers(-huge, decimal.Decimal("1.5")) == -1
        assert compare_numbers(decimal.Decimal("9.9e999999"), huge) == -1

    def test_large_int_near_a_large_decimal_compares_exactly(self) -> None:
        big = 10**5000
        assert compare_numbers(big, decimal.Decimal("1e5000")) == 0
        assert compare_numbers(big + 1, decimal.Decimal("1e5000")) == 1
        assert compare_numbers(big - 1, decimal.Decimal("1e5000")) == -1

    def test_never_rounds_under_the_pinned_context(self) -> None:
        big = 7**9000
        with decimal.localcontext(AGL_DECIMAL_CONTEXT):
            assert compare_numbers(10**40 + 1, decimal.Decimal(10**40)) == 1
            assert compare_numbers(big + 1, exact_decimal(big)) == 1
            assert compare_numbers(-big, exact_decimal(-big) - decimal.Decimal("0.5")) == 1

    @pytest.mark.parametrize("n", [2, 7**9000])
    def test_long_fraction_compares_without_scaling_by_its_exponent(self, n: int) -> None:
        # A two-million-digit fraction: scaling the int by 10**2000000, or
        # converting the coefficient to an int, would take minutes.
        fraction = "0" * 2_000_000 + "1"
        value = decimal.Decimal(f"{n}.{fraction}")
        assert compare_numbers(n, value) == -1
        assert compare_numbers(n + 1, value) == 1
        assert compare_numbers(-n, value.copy_negate()) == 1
        assert compare_numbers(n, decimal.Decimal(f"{n}.{'0' * 2_000_000}")) == 0

    def test_random_pairs_agree_with_exact_fractions(self) -> None:
        generator = random.Random(20260926)
        for _ in range(400):
            bits = generator.choice([8, 64, 9_990, 10_010, 20_000])
            n = generator.getrandbits(bits) * generator.choice([1, -1])
            shift = generator.randint(-3, 3)
            scale = generator.randint(0, 40)
            offset = generator.randint(-(10**scale), 10**scale)
            coefficient = n * 10**scale * 10 ** max(shift, 0) // 10 ** max(-shift, 0) + offset
            value = decimal.Decimal(f"{coefficient}E{-scale}")
            expected_difference = fractions.Fraction(n) - fractions.Fraction(value)
            expected = (expected_difference > 0) - (expected_difference < 0)
            assert compare_numbers(n, value) == expected
            assert compare_numbers(value, n) == -expected
