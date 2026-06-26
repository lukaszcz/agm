"""Tests for the surviving helpers in agm.agl.runtime.convert.

Covers:
1. parse_json_strict: valid scalars/objects/arrays with Decimal floats.
2. parse_json_strict: rejects trailing junk, NaN/Infinity (including nested),
   empty, malformed.
3. normalize_integral_decimals: integral Decimal→int; non-integral preserved;
   nested containers; bool not confused with int.
4. _clean_validation_message: Decimal repr is stripped from jsonschema messages.
5. decode_value / _decode_scalar: all ScalarKind branches, list/dict/record/enum
   happy-path and every ValueError branch for 100% coverage.
"""

from __future__ import annotations

import sys
from decimal import Decimal

import pytest
from jsonschema import ValidationError as JsonschemaValidationError

from agm.agl.ir.contracts import (
    DictDecode,
    EnumDecode,
    ListDecode,
    RecordDecode,
    ScalarDecode,
    ScalarKind,
    VariantDecode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.convert import (
    StrictJsonParseError,
    _clean_validation_message,
    _decode_scalar,
    decode_value,
    normalize_integral_decimals,
    parse_json_strict,
)
from agm.agl.semantics.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
)

# ---------------------------------------------------------------------------
# 1. parse_json_strict — valid inputs
# ---------------------------------------------------------------------------


class TestParseJsonStrict:
    def test_parses_integer(self) -> None:
        result = parse_json_strict("42")
        assert result == 42
        assert isinstance(result, int)

    def test_parses_negative_integer(self) -> None:
        result = parse_json_strict("-7")
        assert result == -7

    def test_parses_float_as_decimal(self) -> None:
        result = parse_json_strict("1.5")
        assert result == Decimal("1.5")
        assert isinstance(result, Decimal)

    def test_parses_float_preserves_precision(self) -> None:
        result = parse_json_strict("3.141592653589793")
        assert isinstance(result, Decimal)
        assert result == Decimal("3.141592653589793")

    def test_parses_bool_true(self) -> None:
        assert parse_json_strict("true") is True

    def test_parses_bool_false(self) -> None:
        assert parse_json_strict("false") is False

    def test_parses_null(self) -> None:
        assert parse_json_strict("null") is None

    def test_parses_string(self) -> None:
        assert parse_json_strict('"hello"') == "hello"

    def test_parses_empty_string(self) -> None:
        assert parse_json_strict('""') == ""

    def test_parses_object(self) -> None:
        result = parse_json_strict('{"x": 1, "y": 2}')
        assert result == {"x": 1, "y": 2}

    def test_parses_array(self) -> None:
        result = parse_json_strict("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parses_nested(self) -> None:
        result = parse_json_strict('{"items": [1, 2.5]}')
        assert isinstance(result, dict)
        d = result
        assert isinstance(d, dict)
        items = d["items"]
        assert isinstance(items, list)
        assert items[0] == 1
        assert items[1] == Decimal("2.5")

    def test_allows_surrounding_whitespace(self) -> None:
        assert parse_json_strict("  42  ") == 42

    def test_parses_integral_float_as_decimal(self) -> None:
        # 1.0 is parsed as Decimal("1.0"), not int — normalization is separate
        result = parse_json_strict("1.0")
        assert isinstance(result, Decimal)
        assert result == Decimal("1.0")

    # --- Rejected inputs ---

    def test_rejects_trailing_junk(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("42 extra")

    def test_rejects_leading_junk(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("extra 42")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("   ")

    def test_rejects_malformed_json(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("{invalid}")

    def test_rejects_unclosed_object(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict('{"a": 1')

    def test_rejects_nan(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("NaN")

    def test_rejects_infinity(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("Infinity")

    def test_rejects_negative_infinity(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("-Infinity")

    def test_rejects_prose_wrapped_json(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("Here is the answer: 42")

    def test_rejects_fenced_json(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("```json\n42\n```")

    def test_rejects_two_values(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("1 2")

    def test_rejects_integer_over_python_digit_limit(self) -> None:
        previous_limit = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(640)
            with pytest.raises(StrictJsonParseError):
                parse_json_strict("1" * 641)
        finally:
            sys.set_int_max_str_digits(previous_limit)


# ---------------------------------------------------------------------------
# 2. parse_json_strict — NaN/Infinity nested inside containers
# ---------------------------------------------------------------------------


class TestParseJsonStrictNested:
    """parse_json_strict must reject NaN/Infinity even when nested inside containers."""

    def test_rejects_nan_in_list(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("[NaN]")

    def test_rejects_infinity_in_object(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict('{"x": Infinity}')

    def test_rejects_negative_infinity_in_list(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("[-Infinity]")

    def test_accepts_nan_as_json_string(self) -> None:
        result = parse_json_strict('"NaN"')
        assert result == "NaN"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. normalize_integral_decimals
# ---------------------------------------------------------------------------


class TestNormalizeIntegralDecimals:
    def test_integral_decimal_becomes_int(self) -> None:
        assert normalize_integral_decimals(Decimal("1.0")) == 1
        assert isinstance(normalize_integral_decimals(Decimal("1.0")), int)

    def test_non_integral_decimal_preserved(self) -> None:
        result = normalize_integral_decimals(Decimal("1.5"))
        assert result == Decimal("1.5")
        assert isinstance(result, Decimal)

    def test_int_unchanged(self) -> None:
        assert normalize_integral_decimals(42) == 42
        assert isinstance(normalize_integral_decimals(42), int)

    def test_str_unchanged(self) -> None:
        assert normalize_integral_decimals("hello") == "hello"

    def test_none_unchanged(self) -> None:
        assert normalize_integral_decimals(None) is None

    def test_bool_not_treated_as_int(self) -> None:
        # bool is a subclass of int; must not be normalized as Decimal
        result = normalize_integral_decimals(True)
        assert result is True
        assert isinstance(result, bool)

    def test_nested_list(self) -> None:
        result = normalize_integral_decimals([Decimal("2.0"), Decimal("3.5")])
        assert result == [2, Decimal("3.5")]
        assert isinstance(result, list)
        items = result
        assert isinstance(items, list)
        assert isinstance(items[0], int)
        assert isinstance(items[1], Decimal)

    def test_nested_dict(self) -> None:
        result = normalize_integral_decimals({"a": Decimal("4.0"), "b": Decimal("1.1")})
        assert isinstance(result, dict)
        d = result
        assert isinstance(d, dict)
        assert d["a"] == 4
        assert isinstance(d["a"], int)
        assert d["b"] == Decimal("1.1")


# ---------------------------------------------------------------------------
# 4. _clean_validation_message
# ---------------------------------------------------------------------------


class TestCleanValidationMessage:
    def _make_error(self, message: str) -> JsonschemaValidationError:
        err = JsonschemaValidationError(message)
        return err

    def test_no_decimal_repr_unchanged(self) -> None:
        err = self._make_error("3.5 is not of type 'integer'")
        assert _clean_validation_message(err) == "3.5 is not of type 'integer'"

    def test_decimal_repr_stripped(self) -> None:
        err = self._make_error("Decimal('3.5') is not of type 'integer'")
        assert _clean_validation_message(err) == "3.5 is not of type 'integer'"

    def test_multiple_decimal_reprs_stripped(self) -> None:
        err = self._make_error(
            "Decimal('1.5') and Decimal('2.5') are not integers"
        )
        assert _clean_validation_message(err) == "1.5 and 2.5 are not integers"


# ---------------------------------------------------------------------------
# 5. decode_value / _decode_scalar — happy paths
# ---------------------------------------------------------------------------


class TestDecodeValueHappy:
    def test_scalar_text(self) -> None:
        result = decode_value(ScalarDecode(kind=ScalarKind.TEXT), "hello")
        assert result == TextValue("hello")

    def test_scalar_int(self) -> None:
        result = decode_value(ScalarDecode(kind=ScalarKind.INT), 42)
        assert result == IntValue(42)

    def test_scalar_int_from_integral_decimal(self) -> None:
        # _decode_scalar has a special branch for integral Decimal → IntValue
        result = _decode_scalar(ScalarKind.INT, Decimal("3.0"))
        assert result == IntValue(3)

    def test_scalar_decimal_from_decimal(self) -> None:
        result = decode_value(ScalarDecode(kind=ScalarKind.DECIMAL), Decimal("1.5"))
        assert result == DecimalValue(Decimal("1.5"))

    def test_scalar_decimal_from_int(self) -> None:
        result = decode_value(ScalarDecode(kind=ScalarKind.DECIMAL), 5)
        assert result == DecimalValue(Decimal(5))

    def test_scalar_bool(self) -> None:
        assert decode_value(ScalarDecode(kind=ScalarKind.BOOL), True) == BoolValue(True)
        assert decode_value(ScalarDecode(kind=ScalarKind.BOOL), False) == BoolValue(False)

    def test_scalar_json(self) -> None:
        result = decode_value(ScalarDecode(kind=ScalarKind.JSON), {"a": 1})
        assert result == JsonValue({"a": 1})

    def test_list(self) -> None:
        schema = ListDecode(elem=ScalarDecode(kind=ScalarKind.INT))
        result = decode_value(schema, [1, 2, 3])
        assert result == ListValue((IntValue(1), IntValue(2), IntValue(3)))

    def test_dict(self) -> None:
        schema = DictDecode(value=ScalarDecode(kind=ScalarKind.INT))
        result = decode_value(schema, {"a": 1, "b": 2})
        assert result == DictValue(entries={"a": IntValue(1), "b": IntValue(2)})

    def test_record(self) -> None:
        nominal = NominalId(ENTRY_ID, "Point")
        schema = RecordDecode(
            nominal=nominal,
            display_name="Point",
            fields=(
                ("x", ScalarDecode(kind=ScalarKind.INT)),
                ("y", ScalarDecode(kind=ScalarKind.INT)),
            ),
        )
        result = decode_value(schema, {"x": 1, "y": 2})
        assert result == RecordValue(
            nominal=nominal,
            display_name="Point",
            fields={"x": IntValue(1), "y": IntValue(2)},
        )

    def test_enum_nullary(self) -> None:
        nominal = NominalId(ENTRY_ID, "Color")
        schema = EnumDecode(
            nominal=nominal,
            display_name="Color",
            variants=(
                VariantDecode(name="Red", fields=()),
                VariantDecode(name="Blue", fields=()),
            ),
        )
        result = decode_value(schema, {"$case": "Red"})
        assert result == EnumValue(
            nominal=nominal, display_name="Color", variant="Red", fields={}
        )

    def test_enum_with_payload(self) -> None:
        nominal = NominalId(ENTRY_ID, "Result")
        schema = EnumDecode(
            nominal=nominal,
            display_name="Result",
            variants=(
                VariantDecode(name="Ok", fields=()),
                VariantDecode(
                    name="Err",
                    fields=(("code", ScalarDecode(kind=ScalarKind.INT)),),
                ),
            ),
        )
        result = decode_value(schema, {"$case": "Err", "code": 42})
        assert result == EnumValue(
            nominal=nominal,
            display_name="Result",
            variant="Err",
            fields={"code": IntValue(42)},
        )


# ---------------------------------------------------------------------------
# 6. decode_value / _decode_scalar — error branches (100% coverage)
# ---------------------------------------------------------------------------


class TestDecodeValueErrors:
    def test_text_type_got_non_string(self) -> None:
        with pytest.raises(ValueError, match="string"):
            decode_value(ScalarDecode(kind=ScalarKind.TEXT), 42)

    def test_int_type_got_bool(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            decode_value(ScalarDecode(kind=ScalarKind.INT), True)

    def test_int_type_got_non_integer_decimal(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            decode_value(ScalarDecode(kind=ScalarKind.INT), Decimal("1.5"))

    def test_decimal_type_got_bool(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            decode_value(ScalarDecode(kind=ScalarKind.DECIMAL), True)

    def test_decimal_type_got_string(self) -> None:
        with pytest.raises(ValueError, match="decimal"):
            decode_value(ScalarDecode(kind=ScalarKind.DECIMAL), "not a number")

    def test_bool_type_got_int(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            decode_value(ScalarDecode(kind=ScalarKind.BOOL), 1)

    def test_list_type_got_non_list(self) -> None:
        schema = ListDecode(elem=ScalarDecode(kind=ScalarKind.TEXT))
        with pytest.raises(ValueError, match="array"):
            decode_value(schema, "not a list")

    def test_dict_type_got_non_dict(self) -> None:
        schema = DictDecode(value=ScalarDecode(kind=ScalarKind.TEXT))
        with pytest.raises(ValueError, match="object"):
            decode_value(schema, [1, 2])

    def test_dict_non_string_key(self) -> None:
        schema = DictDecode(value=ScalarDecode(kind=ScalarKind.TEXT))
        with pytest.raises(ValueError, match="Dict key must be string"):
            decode_value(schema, {1: "val"})

    def test_record_type_got_non_dict(self) -> None:
        schema = RecordDecode(
            nominal=NominalId(ENTRY_ID, "R"),
            display_name="R",
            fields=(("x", ScalarDecode(kind=ScalarKind.INT)),),
        )
        with pytest.raises(ValueError, match="record"):
            decode_value(schema, [1, 2])

    def test_record_missing_field(self) -> None:
        schema = RecordDecode(
            nominal=NominalId(ENTRY_ID, "R"),
            display_name="R",
            fields=(("x", ScalarDecode(kind=ScalarKind.INT)),),
        )
        with pytest.raises(ValueError, match="Missing field"):
            decode_value(schema, {})

    def test_enum_type_got_non_dict(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(VariantDecode(name="A", fields=()),),
        )
        with pytest.raises(ValueError, match="object for enum"):
            decode_value(schema, "oops")

    def test_enum_missing_case_tag(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(VariantDecode(name="A", fields=()),),
        )
        with pytest.raises(ValueError, match=r"\$case"):
            decode_value(schema, {})

    def test_enum_case_tag_not_string(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(VariantDecode(name="A", fields=()),),
        )
        with pytest.raises(ValueError, match=r"\$case"):
            decode_value(schema, {"$case": 42})

    def test_enum_unknown_variant(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(VariantDecode(name="A", fields=()),),
        )
        with pytest.raises(ValueError, match="Unknown enum variant"):
            decode_value(schema, {"$case": "X"})

    def test_enum_missing_payload_field(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(
                VariantDecode(
                    name="B",
                    fields=(("x", ScalarDecode(kind=ScalarKind.INT)),),
                ),
            ),
        )
        with pytest.raises(ValueError, match="missing field"):
            decode_value(schema, {"$case": "B"})
