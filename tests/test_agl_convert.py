"""Tests for agm.agl.runtime.convert — the shared strict-parse/validate conversion core.

Covers (TDD for M4):
1. parse_json_strict: valid scalars/objects/arrays with Decimal floats.
2. parse_json_strict: rejects trailing junk, NaN/Infinity, empty, malformed.
3. json_obj_to_value: builds typed Values from JSON objects.
4. json_obj_to_value: integral 1.0→int ok; 1.5→int fails; schema mismatches.
5. convert_value: across the full matrix (D1).
6. Guardrail: existing test_agl_codec tests still pass (verified separately).
7. D9: text as json embeds (wraps text as JSON string), not parses.
"""

from __future__ import annotations

import sys
from decimal import Decimal

import pytest

from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
)
from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
from agm.agl.runtime.convert import (
    CastConversionError,
    StrictJsonParseError,
    convert_value,
    json_obj_to_value,
    parse_json_strict,
)
from agm.agl.typecheck.types import (
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
)

# ---------------------------------------------------------------------------
# parse_json_strict
# ---------------------------------------------------------------------------


class TestParseJsonStrict:
    # --- Valid inputs ---

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
        # No fence stripping or repair — strict is strict
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
# json_obj_to_value
# ---------------------------------------------------------------------------


class TestJsonObjToValue:
    # --- Scalars ---

    def test_int_obj_to_int_value(self) -> None:
        result = json_obj_to_value(42, IntType())
        assert result == IntValue(42)

    def test_integral_decimal_to_int_value(self) -> None:
        # Decimal("1.0") is integral → normalized to int, then IntValue
        result = json_obj_to_value(Decimal("1.0"), IntType())
        assert result == IntValue(1)

    def test_nonintegral_decimal_to_int_fails(self) -> None:
        with pytest.raises(CastConversionError):
            json_obj_to_value(Decimal("1.5"), IntType())

    def test_decimal_to_decimal_value(self) -> None:
        result = json_obj_to_value(Decimal("3.14"), DecimalType())
        assert result == DecimalValue(Decimal("3.14"))

    def test_int_to_decimal_widens(self) -> None:
        result = json_obj_to_value(5, DecimalType())
        assert result == DecimalValue(Decimal(5))

    def test_bool_to_bool_value(self) -> None:
        assert json_obj_to_value(True, BoolType()) == BoolValue(True)
        assert json_obj_to_value(False, BoolType()) == BoolValue(False)

    def test_bool_rejected_for_int_target(self) -> None:
        with pytest.raises(CastConversionError):
            json_obj_to_value(True, IntType())

    def test_string_to_text_value(self) -> None:
        result = json_obj_to_value("hello", TextType())
        assert result == TextValue("hello")

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(CastConversionError):
            json_obj_to_value("not a number", IntType())

    # --- List ---

    def test_list_to_list_value(self) -> None:
        result = json_obj_to_value([1, 2, 3], ListType(elem=IntType()))
        assert result == ListValue(elements=(IntValue(1), IntValue(2), IntValue(3)))

    def test_list_schema_mismatch_raises(self) -> None:
        with pytest.raises(CastConversionError):
            json_obj_to_value(["a", "b"], ListType(elem=IntType()))

    # --- Dict ---

    def test_dict_to_dict_value(self) -> None:
        result = json_obj_to_value({"a": 1, "b": 2}, DictType(value=IntType()))
        assert result == DictValue(entries={"a": IntValue(1), "b": IntValue(2)})

    # --- Record ---

    def test_record_obj_to_record_value(self) -> None:
        rec_type = RecordType(name="Point", fields={"x": IntType(), "y": IntType()})
        result = json_obj_to_value({"x": 1, "y": 2}, rec_type)
        assert result == RecordValue(
            nominal=NominalId(ENTRY_ID, "Point"),
            display_name="Point",
            fields={"x": IntValue(1), "y": IntValue(2)},
        )

    def test_record_missing_field_raises(self) -> None:
        rec_type = RecordType(name="Point", fields={"x": IntType(), "y": IntType()})
        with pytest.raises(CastConversionError):
            json_obj_to_value({"x": 1}, rec_type)

    def test_record_extra_field_raises(self) -> None:
        rec_type = RecordType(name="Point", fields={"x": IntType(), "y": IntType()})
        with pytest.raises(CastConversionError):
            json_obj_to_value({"x": 1, "y": 2, "z": 3}, rec_type)

    # --- Enum ---

    def test_enum_obj_to_enum_value(self) -> None:
        enum_type = EnumType(
            name="Color",
            variants={"Red": {}, "Blue": {}},
        )
        result = json_obj_to_value({"$case": "Red"}, enum_type)
        assert result == EnumValue(
            nominal=NominalId(ENTRY_ID, "Color"),
            display_name="Color",
            variant="Red",
            fields={},
        )

    def test_enum_bad_case_raises(self) -> None:
        enum_type = EnumType(name="Color", variants={"Red": {}, "Blue": {}})
        with pytest.raises(CastConversionError):
            json_obj_to_value({"$case": "Green"}, enum_type)

    # --- JsonType passthrough ---

    def test_any_value_to_json_value(self) -> None:
        result = json_obj_to_value({"a": 1}, JsonType())
        assert isinstance(result, JsonValue)


# ---------------------------------------------------------------------------
# convert_value — full matrix
# ---------------------------------------------------------------------------


class TestConvertValue:
    # --- → text (total) ---

    def test_int_to_text(self) -> None:
        result = convert_value(IntValue(42), IntType(), TextType())
        assert result == TextValue("42")

    def test_decimal_to_text(self) -> None:
        result = convert_value(DecimalValue(Decimal("3.14")), DecimalType(), TextType())
        assert result == TextValue("3.14")

    def test_bool_to_text(self) -> None:
        result = convert_value(BoolValue(True), BoolType(), TextType())
        assert result == TextValue("true")

    def test_text_to_text_noop(self) -> None:
        result = convert_value(TextValue("hello"), TextType(), TextType())
        assert result == TextValue("hello")

    def test_json_to_text(self) -> None:
        result = convert_value(JsonValue({"a": 1}), JsonType(), TextType())
        # Renders as pretty JSON
        assert isinstance(result, TextValue)
        assert '"a"' in result.value

    # --- → json (total) ---

    def test_int_to_json(self) -> None:
        result = convert_value(IntValue(5), IntType(), JsonType())
        assert result == JsonValue(5)

    def test_decimal_to_json(self) -> None:
        result = convert_value(DecimalValue(Decimal("1.5")), DecimalType(), JsonType())
        assert result == JsonValue(Decimal("1.5"))

    def test_bool_to_json(self) -> None:
        result = convert_value(BoolValue(False), BoolType(), JsonType())
        assert result == JsonValue(False)

    def test_list_to_json(self) -> None:
        lv = ListValue(elements=(IntValue(1), IntValue(2)))
        result = convert_value(lv, ListType(elem=IntType()), JsonType())
        assert result == JsonValue([1, 2])

    def test_dict_to_json(self) -> None:
        dv = DictValue(entries={"k": IntValue(9)})
        result = convert_value(dv, DictType(value=IntType()), JsonType())
        assert result == JsonValue({"k": 9})

    def test_text_to_json_embeds_as_string(self) -> None:
        # D9: text as json wraps the text as a JSON string, never parses it
        result = convert_value(TextValue("42"), TextType(), JsonType())
        # "42" as json → JsonValue("42"), NOT JsonValue(42)
        assert result == JsonValue("42")

    def test_text_to_json_vs_parse_json_strict(self) -> None:
        # Contrast: parse_json_strict("42") == 42 (the number)
        parsed = parse_json_strict("42")
        assert parsed == 42
        # But text as json gives the string "42" wrapped as json
        result = convert_value(TextValue("42"), TextType(), JsonType())
        assert result == JsonValue("42")
        assert result != JsonValue(42)

    def test_json_to_json_noop(self) -> None:
        jv = JsonValue({"x": 1})
        result = convert_value(jv, JsonType(), JsonType())
        assert result == jv

    # --- fallible: text → scalar ---

    def test_text_to_int_success(self) -> None:
        result = convert_value(TextValue("42"), TextType(), IntType())
        assert result == IntValue(42)

    def test_text_to_int_integral_float(self) -> None:
        # "1.0" parses as Decimal, normalized to int
        result = convert_value(TextValue("1.0"), TextType(), IntType())
        assert result == IntValue(1)

    def test_text_to_int_failure_nonintegral(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(TextValue("1.5"), TextType(), IntType())

    def test_text_to_int_failure_not_number(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(TextValue("hello"), TextType(), IntType())

    def test_text_to_int_over_python_digit_limit_is_cast_error(self) -> None:
        previous_limit = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(640)
            with pytest.raises(CastConversionError):
                convert_value(TextValue("1" * 641), TextType(), IntType())
        finally:
            sys.set_int_max_str_digits(previous_limit)

    def test_text_to_bool_true(self) -> None:
        result = convert_value(TextValue("true"), TextType(), BoolType())
        assert result == BoolValue(True)

    def test_text_to_bool_false(self) -> None:
        result = convert_value(TextValue("false"), TextType(), BoolType())
        assert result == BoolValue(False)

    def test_text_to_bool_failure(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(TextValue("yes"), TextType(), BoolType())

    def test_text_to_decimal_success(self) -> None:
        result = convert_value(TextValue("3.14"), TextType(), DecimalType())
        assert result == DecimalValue(Decimal("3.14"))

    def test_text_to_decimal_failure(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(TextValue("not a number"), TextType(), DecimalType())

    def test_text_to_list_success(self) -> None:
        result = convert_value(TextValue("[1, 2, 3]"), TextType(), ListType(elem=IntType()))
        assert result == ListValue(elements=(IntValue(1), IntValue(2), IntValue(3)))

    def test_text_to_list_failure(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(TextValue("not a list"), TextType(), ListType(elem=IntType()))

    def test_text_to_dict_success(self) -> None:
        result = convert_value(
            TextValue('{"a": 1}'), TextType(), DictType(value=IntType())
        )
        assert result == DictValue(entries={"a": IntValue(1)})

    def test_text_to_record_success(self) -> None:
        rec_type = RecordType(name="Point", fields={"x": IntType(), "y": IntType()})
        result = convert_value(TextValue('{"x": 1, "y": 2}'), TextType(), rec_type)
        assert result == RecordValue(
            nominal=NominalId(ENTRY_ID, "Point"),
            display_name="Point",
            fields={"x": IntValue(1), "y": IntValue(2)},
        )

    def test_text_to_record_failure(self) -> None:
        rec_type = RecordType(name="Point", fields={"x": IntType(), "y": IntType()})
        with pytest.raises(CastConversionError):
            convert_value(TextValue("not json"), TextType(), rec_type)

    # --- fallible: json → T ---

    def test_json_to_int_success(self) -> None:
        result = convert_value(JsonValue(42), JsonType(), IntType())
        assert result == IntValue(42)

    def test_json_to_int_failure_nonintegral(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(JsonValue(Decimal("1.5")), JsonType(), IntType())

    def test_json_to_bool_success(self) -> None:
        result = convert_value(JsonValue(True), JsonType(), BoolType())
        assert result == BoolValue(True)

    def test_json_to_bool_failure(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(JsonValue(42), JsonType(), BoolType())

    def test_json_to_decimal_success(self) -> None:
        result = convert_value(JsonValue(Decimal("3.14")), JsonType(), DecimalType())
        assert result == DecimalValue(Decimal("3.14"))

    def test_json_to_list_success(self) -> None:
        result = convert_value(JsonValue([1, 2]), JsonType(), ListType(elem=IntType()))
        assert result == ListValue(elements=(IntValue(1), IntValue(2)))

    def test_json_to_record_success(self) -> None:
        rec_type = RecordType(name="P", fields={"x": IntType()})
        result = convert_value(JsonValue({"x": 7}), JsonType(), rec_type)
        assert result == RecordValue(
            nominal=NominalId(ENTRY_ID, "P"),
            display_name="P",
            fields={"x": IntValue(7)},
        )

    def test_json_to_record_failure(self) -> None:
        rec_type = RecordType(name="P", fields={"x": IntType()})
        with pytest.raises(CastConversionError):
            convert_value(JsonValue("not an object"), JsonType(), rec_type)

    # --- fallible: decimal → int (D4) ---

    def test_decimal_to_int_integral(self) -> None:
        result = convert_value(DecimalValue(Decimal("3.0")), DecimalType(), IntType())
        assert result == IntValue(3)

    def test_decimal_to_int_nonintegral_fails(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(DecimalValue(Decimal("3.5")), DecimalType(), IntType())

    def test_decimal_to_int_large_integral(self) -> None:
        result = convert_value(
            DecimalValue(Decimal("1000000000000")), DecimalType(), IntType()
        )
        assert result == IntValue(1000000000000)

    # --- no-op / assignable cases (D6) ---

    def test_int_to_decimal_widens(self) -> None:
        result = convert_value(IntValue(7), IntType(), DecimalType())
        assert result == DecimalValue(Decimal(7))

    def test_int_to_int_noop(self) -> None:
        result = convert_value(IntValue(5), IntType(), IntType())
        assert result == IntValue(5)

    def test_decimal_to_decimal_noop(self) -> None:
        result = convert_value(DecimalValue(Decimal("1.5")), DecimalType(), DecimalType())
        assert result == DecimalValue(Decimal("1.5"))

    def test_bool_to_bool_noop(self) -> None:
        result = convert_value(BoolValue(True), BoolType(), BoolType())
        assert result == BoolValue(True)

    def test_list_to_list_noop(self) -> None:
        lv = ListValue(elements=(IntValue(1),))
        result = convert_value(lv, ListType(elem=IntType()), ListType(elem=IntType()))
        assert result == lv

    # --- CastConversionError carries context ---

    def test_cast_conversion_error_has_source_type(self) -> None:
        with pytest.raises(CastConversionError) as exc_info:
            convert_value(TextValue("bad"), TextType(), IntType())
        err = exc_info.value
        assert err.source_type == "text"

    def test_cast_conversion_error_has_target_type(self) -> None:
        with pytest.raises(CastConversionError) as exc_info:
            convert_value(TextValue("bad"), TextType(), IntType())
        err = exc_info.value
        assert err.target_type == "int"

    def test_cast_conversion_error_has_raw(self) -> None:
        with pytest.raises(CastConversionError) as exc_info:
            convert_value(TextValue("bad"), TextType(), IntType())
        err = exc_info.value
        assert err.raw == "bad"

    def test_cast_conversion_error_has_message(self) -> None:
        with pytest.raises(CastConversionError) as exc_info:
            convert_value(TextValue("not json"), TextType(), BoolType())
        err = exc_info.value
        assert isinstance(err.message, str)
        assert len(err.message) > 0


# ---------------------------------------------------------------------------
# Regression tests: nested NaN/Infinity bypass (CRITICAL fix)
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
        # The JSON string "NaN" (quoted) is valid and should parse as the Python string "NaN"
        result = parse_json_strict('"NaN"')
        assert result == "NaN"
        assert isinstance(result, str)


class TestConvertValueNestedNaN:
    """convert_value must raise CastConversionError (not bare ValueError) for nested NaN."""

    def test_text_nan_list_to_list_decimal_raises_cast_error(self) -> None:
        # "[NaN]" as list[decimal] must raise CastConversionError, not bare ValueError
        with pytest.raises(CastConversionError):
            convert_value(TextValue("[NaN]"), TextType(), ListType(elem=DecimalType()))

    def test_text_nan_list_to_list_int_raises_cast_error(self) -> None:
        with pytest.raises(CastConversionError):
            convert_value(TextValue("[NaN]"), TextType(), ListType(elem=IntType()))


class TestJsonObjToValueValueErrorCatch:
    """json_obj_to_value must translate ValueError from json_to_value to CastConversionError.

    This covers the defensive ValueError catch (lines after schema validation) to ensure
    no bare ValueError ever escapes the module boundary.  The only reachable scenario is a
    dict with a non-string key: the JSON object schema passes (any object is accepted) but
    json_to_value rejects the non-string key.
    """

    def test_dict_with_non_string_key_raises_cast_conversion_error(self) -> None:
        # {1: 42} passes schema validation ({"type": "object"}) but json_to_value
        # raises ValueError for the non-string key.  Must surface as CastConversionError.
        with pytest.raises(CastConversionError) as exc_info:
            json_obj_to_value({1: 42}, DictType(value=IntType()))
        err = exc_info.value
        assert "Dict key must be string" in err.message


# ---------------------------------------------------------------------------
# Regression tests: Decimal repr in error messages (NIT fix)
# ---------------------------------------------------------------------------


class TestDecimalErrorMessage:
    """CastConversionError messages must not leak Python Decimal(...) repr."""

    def test_nonintegral_decimal_error_message_no_decimal_repr(self) -> None:
        # "3.5" as int — message should not contain "Decimal("
        with pytest.raises(CastConversionError) as exc_info:
            convert_value(TextValue("3.5"), TextType(), IntType())
        err = exc_info.value
        assert "Decimal(" not in err.message

    def test_json_nonintegral_decimal_to_int_error_no_decimal_repr(self) -> None:
        # JsonValue with a Decimal — message should not contain "Decimal("
        with pytest.raises(CastConversionError) as exc_info:
            convert_value(JsonValue(Decimal("3.5")), JsonType(), IntType())
        err = exc_info.value
        assert "Decimal(" not in err.message


# ---------------------------------------------------------------------------
# M2: Nominal value → text produces AgL form (D1 / D10)
# ---------------------------------------------------------------------------


class TestConvertValueNominalToText:
    """convert_value(record/enum/exception, ..., TextType()) returns AgL-form text."""

    def test_record_to_text_agl_form(self) -> None:
        rec_type = RecordType(name="Point", fields={"x": IntType(), "y": IntType()})
        rv = RecordValue(
            nominal=NominalId(ENTRY_ID, "Point"),
            display_name="Point",
            fields={"x": IntValue(3), "y": IntValue(4)},
        )
        result = convert_value(rv, rec_type, TextType())
        assert result == TextValue("Point(x: 3, y: 4)")

    def test_enum_to_text_agl_form(self) -> None:
        enum_type = EnumType(name="Color", variants={"Red": {}, "Blue": {"n": IntType()}})
        ev = EnumValue(
            nominal=NominalId(ENTRY_ID, "Color"),
            display_name="Color",
            variant="Red",
            fields={},
        )
        result = convert_value(ev, enum_type, TextType())
        assert result == TextValue("Color.Red")

    def test_enum_with_fields_to_text_agl_form(self) -> None:
        enum_type = EnumType(name="Color", variants={"Red": {}, "Blue": {"n": IntType()}})
        ev = EnumValue(
            nominal=NominalId(ENTRY_ID, "Color"),
            display_name="Color",
            variant="Blue",
            fields={"n": IntValue(7)},
        )
        result = convert_value(ev, enum_type, TextType())
        assert result == TextValue("Color.Blue(n: 7)")

    def test_exception_to_text_agl_form(self) -> None:
        exc_type = ExceptionType(
            name="CastError",
            fields={"message": TextType(), "trace_id": TextType(),
                    "source_type": TextType(), "target_type": TextType(), "raw": TextType()},
        )
        exc_val = ExceptionValue(
            nominal=NominalId(PRELUDE_ID, "CastError"),
            display_name="CastError",
            fields={
                "message": TextValue("cannot parse"),
                "trace_id": TextValue("evt-1"),
                "source_type": TextValue("text"),
                "target_type": TextValue("int"),
                "raw": TextValue("x"),
            },
        )
        result = convert_value(exc_val, exc_type, TextType())
        assert isinstance(result, TextValue)
        assert result.value.startswith("CastError(")
        assert "trace_id" in result.value

    def test_list_to_text_agl_form(self) -> None:
        lv = ListValue(elements=(IntValue(1), IntValue(2), IntValue(3)))
        result = convert_value(lv, ListType(elem=IntType()), TextType())
        assert result == TextValue("[1, 2, 3]")

    def test_dict_to_text_agl_form(self) -> None:
        dv = DictValue(entries={"k": IntValue(9)})
        result = convert_value(dv, DictType(value=IntType()), TextType())
        assert result == TextValue('{"k": 9}')


# ---------------------------------------------------------------------------
# M2: Nominal value → json (D10 — TOTAL_JSON explicit cast)
# ---------------------------------------------------------------------------


class TestConvertValueNominalToJson:
    """convert_value(record/enum/exception, ..., JsonType()) produces structural JSON."""

    def test_record_to_json(self) -> None:
        rec_type = RecordType(name="Point", fields={"x": IntType(), "y": IntType()})
        rv = RecordValue(
            nominal=NominalId(ENTRY_ID, "Point"),
            display_name="Point",
            fields={"x": IntValue(3), "y": IntValue(4)},
        )
        result = convert_value(rv, rec_type, JsonType())
        # Record → json: field object (construction-order keys from fields dict)
        assert result == JsonValue({"x": 3, "y": 4})

    def test_enum_to_json(self) -> None:
        enum_type = EnumType(name="Color", variants={"Red": {}, "Blue": {"n": IntType()}})
        ev = EnumValue(
            nominal=NominalId(ENTRY_ID, "Color"),
            display_name="Color",
            variant="Blue",
            fields={"n": IntValue(7)},
        )
        result = convert_value(ev, enum_type, JsonType())
        # Enum → json: "$case"-tagged object
        assert result == JsonValue({"$case": "Blue", "n": 7})

    def test_enum_nullary_to_json(self) -> None:
        enum_type = EnumType(name="Color", variants={"Red": {}, "Blue": {}})
        ev = EnumValue(
            nominal=NominalId(ENTRY_ID, "Color"),
            display_name="Color",
            variant="Red",
            fields={},
        )
        result = convert_value(ev, enum_type, JsonType())
        assert result == JsonValue({"$case": "Red"})

    def test_exception_to_json(self) -> None:
        exc_type = ExceptionType(
            name="Abort",
            fields={"message": TextType(), "trace_id": TextType()},
        )
        exc_val = ExceptionValue(
            nominal=NominalId(ENTRY_ID, "Abort"),
            display_name="Abort",
            fields={"message": TextValue("stop"), "trace_id": TextValue("evt-2")},
        )
        result = convert_value(exc_val, exc_type, JsonType())
        # Exception → json: all fields including trace_id
        assert result == JsonValue({"message": "stop", "trace_id": "evt-2"})

    # as? coverage: total casts always return True (no CastConversionError raised)

    def test_record_to_json_is_total(self) -> None:
        """record → json is TOTAL_JSON — convert_value never raises CastConversionError."""
        rec_type = RecordType(name="P", fields={"x": IntType()})
        rv = RecordValue(
            nominal=NominalId(ENTRY_ID, "P"),
            display_name="P",
            fields={"x": IntValue(1)},
        )
        # Must not raise
        result = convert_value(rv, rec_type, JsonType())
        assert isinstance(result, JsonValue)

    def test_enum_to_json_is_total(self) -> None:
        enum_type = EnumType(name="E", variants={"A": {}})
        ev = EnumValue(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variant="A",
            fields={},
        )
        result = convert_value(ev, enum_type, JsonType())
        assert isinstance(result, JsonValue)

    def test_exception_to_json_is_total(self) -> None:
        exc_type = ExceptionType(
            name="TypeError", fields={"message": TextType(), "trace_id": TextType()}
        )
        exc_val = ExceptionValue(
            nominal=NominalId(ENTRY_ID, "TypeError"),
            display_name="TypeError",
            fields={"message": TextValue("oops"), "trace_id": TextValue("t1")},
        )
        result = convert_value(exc_val, exc_type, JsonType())
        assert isinstance(result, JsonValue)
