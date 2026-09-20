"""Tests for the surviving helpers in agm.agl.runtime.convert.

Covers:
1. parse_json_strict: valid scalars/objects/arrays with Decimal floats.
2. parse_json_strict: rejects trailing junk, NaN/Infinity (including nested),
   empty, malformed.
3. validator_for_schema: the AgL Draft 2020-12 validator accepts an integral
   Decimal as ``integer`` (including through ``$ref``/``$defs``), rejects a
   non-integral Decimal and bool, and caches by schema string.
4. _clean_validation_message: Decimal repr is stripped from jsonschema messages.
5. decode_value / _decode_scalar: all ScalarKind branches, array/dict/record/enum
   happy-path and every ValueError branch for 100% coverage.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal

import pytest
from jsonschema import ValidationError as JsonschemaValidationError

from agm.agl.ir.contracts import (
    ArrayDecode,
    DictDecode,
    EnumDecode,
    FieldDecode,
    RecordDecode,
    RefDecode,
    ScalarDecode,
    ScalarKind,
    VariantDecode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.runtime.convert import (
    StrictJsonParseError,
    _clean_validation_message,
    _decode_scalar,
    decode_value,
    parse_json_strict,
    validator_for_schema,
)
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    DecimalValue,
    DictValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
)
from agm.agl.zones import ParamZone

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
        # 1.0 is parsed as Decimal("1.0"), not int — the validator, not the
        # parser, is what accepts it against an integer schema.
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

    def test_rejects_lone_surrogate_escape(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict('"\\ud800"')

    def test_combines_surrogate_escape_pair(self) -> None:
        # Built from parts so no tool between here and the file can fold the
        # adjacent escapes into the astral character they denote.
        high_escape = "\\" + "ud83d"
        low_escape = "\\" + "ude00"
        assert parse_json_strict(f'"{high_escape}{low_escape}"') == "\U0001f600"


# ---------------------------------------------------------------------------
# 2. parse_json_strict — NaN/Infinity nested inside containers
# ---------------------------------------------------------------------------


class TestParseJsonStrictNested:
    """parse_json_strict must reject NaN/Infinity even when nested inside containers."""

    def test_rejects_nan_in_array(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("[NaN]")

    def test_rejects_infinity_in_object(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict('{"x": Infinity}')

    def test_rejects_negative_infinity_in_array(self) -> None:
        with pytest.raises(StrictJsonParseError):
            parse_json_strict("[-Infinity]")

    def test_accepts_nan_as_json_string(self) -> None:
        result = parse_json_strict('"NaN"')
        assert result == "NaN"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. validator_for_schema — the AgL Decimal-aware "integer" type check
# ---------------------------------------------------------------------------


class TestAglValidator:
    def test_integral_decimal_accepted_as_integer(self) -> None:
        validator = validator_for_schema('{"type": "integer"}')
        assert list(validator.iter_errors(Decimal("1.0"))) == []

    def test_negative_zero_decimal_accepted_as_integer(self) -> None:
        validator = validator_for_schema('{"type": "integer"}')
        assert list(validator.iter_errors(Decimal("-0.0"))) == []

    def test_non_integral_decimal_rejected_as_integer(self) -> None:
        validator = validator_for_schema('{"type": "integer"}')
        assert len(list(validator.iter_errors(Decimal("1.5")))) == 1

    def test_int_still_accepted_as_integer(self) -> None:
        validator = validator_for_schema('{"type": "integer"}')
        assert list(validator.iter_errors(1)) == []

    def test_bool_rejected_as_integer(self) -> None:
        validator = validator_for_schema('{"type": "integer"}')
        assert len(list(validator.iter_errors(True))) == 1

    def test_integral_decimal_accepted_through_ref(self) -> None:
        schema = json.dumps({"$ref": "#/$defs/Num", "$defs": {"Num": {"type": "integer"}}})
        validator = validator_for_schema(schema)
        assert list(validator.iter_errors(Decimal("2.0"))) == []

    def test_non_integral_decimal_rejected_through_ref(self) -> None:
        schema = json.dumps({"$ref": "#/$defs/Num", "$defs": {"Num": {"type": "integer"}}})
        validator = validator_for_schema(schema)
        assert len(list(validator.iter_errors(Decimal("2.5")))) == 1

    def test_caches_by_schema_string(self) -> None:
        assert validator_for_schema('{"type": "integer"}') is validator_for_schema(
            '{"type": "integer"}'
        )


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
        err = self._make_error("Decimal('1.5') and Decimal('2.5') are not integers")
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

    def test_array(self) -> None:
        schema = ArrayDecode(elem=ScalarDecode(kind=ScalarKind.INT))
        result = decode_value(schema, [1, 2, 3])
        assert result == ArrayValue([IntValue(1), IntValue(2), IntValue(3)])

    def test_dict(self) -> None:
        schema = DictDecode(value=ScalarDecode(kind=ScalarKind.INT))
        result = decode_value(schema, {"a": 1, "b": 2})
        assert result == DictValue(entries={"a": IntValue(1), "b": IntValue(2)})

    def test_record(self) -> None:
        nominal = NominalId(1)
        schema = RecordDecode(
            nominal=nominal,
            display_name="Point",
            name="Point",
            fields=(
                FieldDecode(
                    "x", "x", ScalarDecode(kind=ScalarKind.INT), zone=ParamZone.STANDARD, alias=None
                ),
                FieldDecode(
                    "y", "y", ScalarDecode(kind=ScalarKind.INT), zone=ParamZone.STANDARD, alias=None
                ),
            ),
            alias=None,
        )
        result = decode_value(schema, {"x": 1, "y": 2})
        assert result == RecordValue(
            nominal=nominal,
            fields={"x": IntValue(1), "y": IntValue(2)},
        )

    def test_enum_nullary(self) -> None:
        nominal = NominalId(1)
        schema = EnumDecode(
            nominal=nominal,
            display_name="Color",
            name="Color",
            variants=(
                VariantDecode(
                    name="Red",
                    json_name="Red",
                    nominal=NominalId(2),
                    display_name="Color::Red",
                    fields=(),
                    alias=None,
                ),
                VariantDecode(
                    name="Blue",
                    json_name="Blue",
                    nominal=NominalId(3),
                    display_name="Color::Blue",
                    fields=(),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        result = decode_value(schema, {"$case": "Red"})
        assert result == RecordValue(nominal=NominalId(2), fields={})

    def test_enum_with_payload(self) -> None:
        nominal = NominalId(1)
        schema = EnumDecode(
            nominal=nominal,
            display_name="Result",
            name="Result",
            variants=(
                VariantDecode(
                    name="Ok",
                    json_name="Ok",
                    nominal=NominalId(2),
                    display_name="Result::Ok",
                    fields=(),
                    alias=None,
                ),
                VariantDecode(
                    name="Err",
                    json_name="Err",
                    nominal=NominalId(3),
                    display_name="Result::Err",
                    fields=(
                        FieldDecode(
                            "code",
                            "code",
                            ScalarDecode(kind=ScalarKind.INT),
                            zone=ParamZone.STANDARD,
                            alias=None,
                        ),
                    ),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        result = decode_value(schema, {"$case": "Err", "code": 42})
        assert result == RecordValue(nominal=NominalId(3), fields={"code": IntValue(42)})

    def test_record_reads_renamed_json_key_builds_declared_field(self) -> None:
        """A field's ``json_name`` is read from JSON; the built record keeps the declared name."""
        nominal = NominalId(1)
        schema = RecordDecode(
            nominal=nominal,
            display_name="Point",
            name="Point",
            fields=(
                FieldDecode(
                    "x",
                    "x-coord",
                    ScalarDecode(kind=ScalarKind.INT),
                    zone=ParamZone.STANDARD,
                    alias=None,
                ),
            ),
            alias=None,
        )
        result = decode_value(schema, {"x-coord": 1})
        assert result == RecordValue(nominal=nominal, fields={"x": IntValue(1)})

    def test_enum_matches_case_against_renamed_json_tag(self) -> None:
        """``$case`` is matched against ``VariantDecode.json_name``, not the declared name."""
        nominal = NominalId(1)
        schema = EnumDecode(
            nominal=nominal,
            display_name="Color",
            name="Color",
            variants=(
                VariantDecode(
                    name="Red",
                    json_name="RED",
                    nominal=NominalId(2),
                    display_name="Color::Red",
                    fields=(),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        result = decode_value(schema, {"$case": "RED"})
        assert result == RecordValue(nominal=NominalId(2), fields={})


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

    def test_array_type_got_non_array(self) -> None:
        schema = ArrayDecode(elem=ScalarDecode(kind=ScalarKind.TEXT))
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
            nominal=NominalId(1),
            display_name="R",
            name="R",
            fields=(
                FieldDecode(
                    "x", "x", ScalarDecode(kind=ScalarKind.INT), zone=ParamZone.STANDARD, alias=None
                ),
            ),
            alias=None,
        )
        with pytest.raises(ValueError, match="record"):
            decode_value(schema, [1, 2])

    def test_record_missing_field(self) -> None:
        schema = RecordDecode(
            nominal=NominalId(1),
            display_name="R",
            name="R",
            fields=(
                FieldDecode(
                    "x", "x", ScalarDecode(kind=ScalarKind.INT), zone=ParamZone.STANDARD, alias=None
                ),
            ),
            alias=None,
        )
        with pytest.raises(ValueError, match="Missing field"):
            decode_value(schema, {})

    def test_enum_type_got_non_dict(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(1),
            display_name="E",
            name="E",
            variants=(
                VariantDecode(
                    name="A",
                    json_name="A",
                    nominal=NominalId(2),
                    display_name="E::A",
                    fields=(),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        with pytest.raises(ValueError, match="object for enum"):
            decode_value(schema, "oops")

    def test_enum_missing_case_tag(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(1),
            display_name="E",
            name="E",
            variants=(
                VariantDecode(
                    name="A",
                    json_name="A",
                    nominal=NominalId(2),
                    display_name="E::A",
                    fields=(),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        with pytest.raises(ValueError, match=r"\$case"):
            decode_value(schema, {})

    def test_enum_case_tag_not_string(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(1),
            display_name="E",
            name="E",
            variants=(
                VariantDecode(
                    name="A",
                    json_name="A",
                    nominal=NominalId(2),
                    display_name="E::A",
                    fields=(),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        with pytest.raises(ValueError, match=r"\$case"):
            decode_value(schema, {"$case": 42})

    def test_enum_unknown_variant(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(1),
            display_name="E",
            name="E",
            variants=(
                VariantDecode(
                    name="A",
                    json_name="A",
                    nominal=NominalId(2),
                    display_name="E::A",
                    fields=(),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        with pytest.raises(ValueError, match="Unknown enum variant"):
            decode_value(schema, {"$case": "X"})

    def test_enum_missing_payload_field(self) -> None:
        schema = EnumDecode(
            nominal=NominalId(1),
            display_name="E",
            name="E",
            variants=(
                VariantDecode(
                    name="B",
                    json_name="B",
                    nominal=NominalId(2),
                    display_name="E::B",
                    fields=(
                        FieldDecode(
                            "x",
                            "x",
                            ScalarDecode(kind=ScalarKind.INT),
                            zone=ParamZone.STANDARD,
                            alias=None,
                        ),
                    ),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        with pytest.raises(ValueError, match="missing field"):
            decode_value(schema, {"$case": "B"})


# ---------------------------------------------------------------------------
# RefDecode resolution — the runtime mirror of a recursive type's $ref/$defs.
# ---------------------------------------------------------------------------


class TestDecodeValueRefDecode:
    """decode_value resolves RefDecode through the defs mapping threaded alongside the walk."""

    @staticmethod
    def _tree_defs() -> dict[str, "object"]:
        """A self-recursive `Tree` decode schema: Leaf | Node(value, left, right)."""
        nominal = NominalId(1)
        tree_decode = EnumDecode(
            nominal=nominal,
            display_name="Tree",
            name="Tree",
            variants=(
                VariantDecode(
                    name="Leaf",
                    json_name="Leaf",
                    nominal=NominalId(2),
                    display_name="Tree::Leaf",
                    fields=(),
                    alias=None,
                ),
                VariantDecode(
                    name="Node",
                    json_name="Node",
                    nominal=NominalId(3),
                    display_name="Tree::Node",
                    fields=(
                        FieldDecode(
                            "value",
                            "value",
                            ScalarDecode(kind=ScalarKind.INT),
                            zone=ParamZone.STANDARD,
                            alias=None,
                        ),
                        FieldDecode(
                            "left", "left", RefDecode("Tree"), zone=ParamZone.STANDARD, alias=None
                        ),
                        FieldDecode(
                            "right", "right", RefDecode("Tree"), zone=ParamZone.STANDARD, alias=None
                        ),
                    ),
                    alias=None,
                ),
            ),
            host_agent=False,
        )
        return {"Tree": tree_decode}

    def test_recursive_root_ref_decodes_deep_nested_tree(self) -> None:
        """A deeply nested tree payload decodes correctly through repeated RefDecode resolution."""
        defs = self._tree_defs()
        schema = RefDecode("Tree")
        # A tree 5 nodes deep, all leaning right.
        payload: object = {"$case": "Leaf"}
        for value in range(5):
            payload = {
                "$case": "Node",
                "value": value,
                "left": {"$case": "Leaf"},
                "right": payload,
            }
        result = decode_value(schema, payload, defs)
        assert isinstance(result, RecordValue)
        assert result.nominal == NominalId(3)
        assert result.fields["value"] == IntValue(4)
        # Walk down the "right" spine to confirm every level decoded.
        node = result
        for expected in range(4, -1, -1):
            assert isinstance(node, RecordValue)
            assert node.nominal == NominalId(3)
            assert node.fields["value"] == IntValue(expected)
            next_node = node.fields["right"]
            assert isinstance(next_node, RecordValue)
            node = next_node
        assert node.nominal == NominalId(2)

    def test_ref_nested_inside_array_and_record(self) -> None:
        """A RefDecode reachable through ArrayDecode/RecordDecode fields resolves the same way."""
        defs = self._tree_defs()
        category_nominal = NominalId(2)
        schema = RecordDecode(
            nominal=category_nominal,
            display_name="Wrapper",
            name="Wrapper",
            fields=(
                FieldDecode(
                    "trees",
                    "trees",
                    ArrayDecode(RefDecode("Tree")),
                    zone=ParamZone.STANDARD,
                    alias=None,
                ),
            ),
            alias=None,
        )
        node_payload = {
            "$case": "Node",
            "value": 1,
            "left": {"$case": "Leaf"},
            "right": {"$case": "Leaf"},
        }
        payload = {"trees": [{"$case": "Leaf"}, node_payload]}
        result = decode_value(schema, payload, defs)
        assert isinstance(result, RecordValue)
        trees = result.fields["trees"]
        assert isinstance(trees, ArrayValue)
        assert len(trees.elements) == 2
        first = trees.elements[0]
        second = trees.elements[1]
        assert isinstance(first, RecordValue) and first.nominal == NominalId(2)
        assert isinstance(second, RecordValue) and second.nominal == NominalId(3)

    def test_unknown_defs_key_is_internal_error(self) -> None:
        """An unresolvable RefDecode key is an internal-invariant violation, not a user error."""
        with pytest.raises(AssertionError, match="unknown \\$defs key"):
            decode_value(RefDecode("NoSuchKey"), {"$case": "Leaf"}, {})

    def test_ref_only_defs_cycle_is_internal_error(self) -> None:
        """A malformed defs table must not make RefDecode resolution recurse forever."""
        with pytest.raises(AssertionError, match=r"\$defs reference cycle"):
            decode_value(RefDecode("A"), {}, {"A": RefDecode("B"), "B": RefDecode("A")})

    def test_defs_defaults_to_empty_for_non_recursive_schemas(self) -> None:
        """Calling decode_value with the historical 2-arg form still works (defs defaults empty)."""
        schema = ScalarDecode(kind=ScalarKind.INT)
        assert decode_value(schema, 5) == IntValue(5)
