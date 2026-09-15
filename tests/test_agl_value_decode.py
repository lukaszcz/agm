"""AgL value syntax -> JSON-native decoding: ``runtime.value_decode``."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.contracts import ArrayDecode, DecodeSchema, DictDecode, ScalarDecode, ScalarKind
from agm.agl.runtime.value_decode import (
    ValueDecodeError,
    host_text_to_json,
    option_some_field_schema,
    option_some_json_name,
    value_node_to_json,
)
from agm.agl.semantics.type_table import create_seeded_type_table
from agm.agl.semantics.types import BoolType, DecimalType, IntType, JsonType, TextType, Type
from agm.agl.type_schema import build_decode_schema, build_param_decoder
from agm.agl.typecheck import CheckedModule
from agm.agl.value_syntax.reader import read_value
from tests.agl.module_graph import resolve_and_check_repl_entry

# ---------------------------------------------------------------------------
# Test support: real typechecked source -> DecodeSchema
# ---------------------------------------------------------------------------


def _capabilities() -> HostCapabilities:
    return HostCapabilities(
        supports_shell_exec=True,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}
            ),
        },
    )


def _typecheck(source: str) -> CheckedModule:
    return resolve_and_check_repl_entry(source, _capabilities())


def _last_expr_plan(source: str) -> tuple[DecodeSchema, dict[str, DecodeSchema]]:
    """Build a decode schema for the type of *source*'s trailing expression."""
    checked = _typecheck(source)
    last = checked.resolved.program.body.items[-1]
    typ = checked.node_types[last.node_id]
    plan = build_decode_schema(typ, checked.type_env.type_table)
    return plan.root, dict(plan.defs)


def _scalar_plan(typ: Type) -> tuple[DecodeSchema, dict[str, DecodeSchema]]:
    decoder = build_param_decoder(typ, create_seeded_type_table())
    return decoder.decode, dict(decoder.defs)


_RECORD_SRC = (
    "record Point\n"
    '  @name("px") @json-name("pixel") x: int\n'
    "  y: int\n"
    "let p: Point = Point(x = 1, y = 2)\n"
    "p"
)

_QUALIFIED_RECORD_SRC = (
    "scope Q\n  record Point(x: int)\nend Q\n\nlet p: Q::Point = Q::Point(x = 1)\np"
)

_INLINE_MEMBER_RECORD_SRC = (
    "scope S\n"
    "  enum Shape\n"
    "    | Square(side: int)\n"
    "end S\n"
    "\n"
    "let sq: S::Shape::Square = S::Shape::Square(side = 1)\n"
    "sq"
)

_NESTED_SCOPE_RECORD_SRC = (
    "scope A\n"
    "\n"
    "  scope B\n"
    "    record Point(x: int)\n"
    "  end B\n"
    "end A\n"
    "\n"
    "let p: A::B::Point = A::B::Point(x = 1)\n"
    "p"
)

_STANDALONE_RECORD_SRC = "record Point(x: int)\nlet p: Point = Point(x = 1)\np"

_REFERENCED_MEMBER_ENUM_SRC = (
    "scope Data\n"
    "  record Saved(id: int)\n"
    "end Data\n"
    "\n"
    "enum Stored = ::Data::Saved | Fresh(value: int)\n"
    "\n"
    "let x: Stored = Data::Saved(id = 1)\n"
    "x"
)

_SHAPE_SRC = (
    "enum Shape\n"
    '  | @name("sq") @arg-pos Square(side: int)\n'
    "  | @arg-named Circle(radius: int)\n"
    "  | Triangle(base: int, height: int)\n"
    "  | Empty\n"
    "let s: Shape = Triangle(1, 2)\n"
    "s"
)

_LINK_SRC = "enum Link\n  | End\n  | Cell(next: Link)\nlet l: Link = Cell(next = End)\nl"

_OPTION_INT_SRC = "let x: Option[int] = None\nx"

_OPTION_POINT_SRC = (
    "record Point\n"
    '  @name("px") @json-name("pixel") x: int\n'
    "  y: int\n"
    "let o: Option[Point] = None\n"
    "o"
)


# ---------------------------------------------------------------------------
# value_node_to_json: scalars
# ---------------------------------------------------------------------------


class TestScalarDecode:
    def test_int_matches(self) -> None:
        schema, defs = _scalar_plan(IntType())
        assert value_node_to_json(read_value("42"), schema, defs) == 42

    def test_int_rejects_text(self) -> None:
        schema, defs = _scalar_plan(IntType())
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value('"nope"'), schema, defs)

    def test_int_mismatch_names_a_constructor_node(self) -> None:
        schema, defs = _scalar_plan(IntType())
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Foo(x = 1)"), schema, defs)

    def test_int_rejects_a_decimal_node(self) -> None:
        schema, defs = _scalar_plan(IntType())
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("1.5"), schema, defs)

    def test_decimal_widens_from_an_int_literal(self) -> None:
        schema, defs = _scalar_plan(DecimalType())
        assert value_node_to_json(read_value("3"), schema, defs) == 3

    def test_decimal_matches_a_decimal_literal(self) -> None:
        schema, defs = _scalar_plan(DecimalType())
        assert value_node_to_json(read_value("3.5"), schema, defs) == Decimal("3.5")

    def test_bool_matches(self) -> None:
        schema, defs = _scalar_plan(BoolType())
        assert value_node_to_json(read_value("true"), schema, defs) is True

    def test_bool_rejects_int(self) -> None:
        schema, defs = _scalar_plan(BoolType())
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("1"), schema, defs)

    def test_decimal_rejects_bool(self) -> None:
        schema, defs = _scalar_plan(DecimalType())
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("true"), schema, defs)

    def test_text_matches(self) -> None:
        schema, defs = _scalar_plan(TextType())
        assert value_node_to_json(read_value('"hi"'), schema, defs) == "hi"

    def test_text_rejects_null(self) -> None:
        schema, defs = _scalar_plan(TextType())
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("null"), schema, defs)


class TestJsonScalarDecode:
    def test_accepts_heterogeneous_data(self) -> None:
        schema, defs = _scalar_plan(JsonType())
        result = value_node_to_json(read_value('[1, "a", null, true, 2.5, {x: 1}]'), schema, defs)
        assert result == [1, "a", None, True, Decimal("2.5"), {"x": 1}]

    def test_rejects_a_constructor(self) -> None:
        schema, defs = _scalar_plan(JsonType())
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Foo(x = 1)"), schema, defs)

    def test_rejects_a_duplicate_dict_key(self) -> None:
        schema, defs = _scalar_plan(JsonType())
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("{x: 1, x: 2}"), schema, defs)


# ---------------------------------------------------------------------------
# value_node_to_json: array / dict
# ---------------------------------------------------------------------------


class TestArrayDictDecode:
    def test_array_of_int(self) -> None:
        schema = ArrayDecode(elem=ScalarDecode(kind=ScalarKind.INT))
        assert value_node_to_json(read_value("[1, 2, 3]"), schema) == [1, 2, 3]

    def test_array_rejects_non_array(self) -> None:
        schema = ArrayDecode(elem=ScalarDecode(kind=ScalarKind.INT))
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("1"), schema)

    def test_dict_of_text(self) -> None:
        schema = DictDecode(value=ScalarDecode(kind=ScalarKind.TEXT))
        result = value_node_to_json(read_value('{a: "x", b: "y"}'), schema)
        assert result == {"a": "x", "b": "y"}

    def test_dict_rejects_non_dict(self) -> None:
        schema = DictDecode(value=ScalarDecode(kind=ScalarKind.TEXT))
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("[]"), schema)

    def test_dict_rejects_duplicate_key(self) -> None:
        schema = DictDecode(value=ScalarDecode(kind=ScalarKind.INT))
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("{a: 1, a: 2}"), schema)

    def test_array_mismatch_names_a_boolean_node(self) -> None:
        schema = ArrayDecode(elem=ScalarDecode(kind=ScalarKind.INT))
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("true"), schema)

    def test_array_mismatch_names_a_text_node(self) -> None:
        schema = ArrayDecode(elem=ScalarDecode(kind=ScalarKind.INT))
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value('"x"'), schema)

    def test_array_mismatch_names_a_dict_node(self) -> None:
        schema = ArrayDecode(elem=ScalarDecode(kind=ScalarKind.INT))
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("{a: 1}"), schema)


# ---------------------------------------------------------------------------
# value_node_to_json: records
# ---------------------------------------------------------------------------


class TestRecordDecode:
    def test_declared_name_and_field_aliases(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        result = value_node_to_json(read_value("Point(x = 1, y = 2)"), schema, defs)
        assert result == {"pixel": 1, "y": 2}

    def test_field_alias_spelling_is_also_accepted(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        result = value_node_to_json(read_value("Point(px = 1, y = 2)"), schema, defs)
        assert result == {"pixel": 1, "y": 2}

    def test_rejects_a_non_constructor_node(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("1"), schema, defs)

    def test_rejects_the_wrong_constructor_name(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Other(x = 1, y = 2)"), schema, defs)

    def test_rejects_an_unknown_field(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Point(x = 1, y = 2, z = 3)"), schema, defs)

    def test_rejects_a_missing_field(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Point(x = 1)"), schema, defs)

    def test_rejects_a_duplicate_field(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Point(1, x = 2, y = 3)"), schema, defs)

    def test_rejects_too_many_positional_arguments(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Point(1, 2, 3)"), schema, defs)

    def test_qualified_spelling_matching_its_scope_is_accepted(self) -> None:
        schema, defs = _last_expr_plan(_QUALIFIED_RECORD_SRC)
        assert value_node_to_json(read_value("Q::Point(x = 1)"), schema, defs) == {"x": 1}

    def test_qualified_spelling_not_matching_its_scope_is_rejected(self) -> None:
        schema, defs = _last_expr_plan(_QUALIFIED_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Other::Point(x = 1)"), schema, defs)

    def test_inline_member_record_accepts_its_enclosing_enum_as_qualifier(self) -> None:
        schema, defs = _last_expr_plan(_INLINE_MEMBER_RECORD_SRC)
        result = value_node_to_json(read_value("Shape::Square(side = 1)"), schema, defs)
        assert result == {"side": 1}

    def test_inline_member_record_rejects_the_outer_scope_as_qualifier(self) -> None:
        schema, defs = _last_expr_plan(_INLINE_MEMBER_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("S::Square(side = 1)"), schema, defs)

    def test_nested_scope_record_accepts_its_innermost_scope_as_qualifier(self) -> None:
        schema, defs = _last_expr_plan(_NESTED_SCOPE_RECORD_SRC)
        result = value_node_to_json(read_value("B::Point(x = 1)"), schema, defs)
        assert result == {"x": 1}

    def test_nested_scope_record_rejects_the_outer_scope_as_qualifier(self) -> None:
        schema, defs = _last_expr_plan(_NESTED_SCOPE_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("A::Point(x = 1)"), schema, defs)

    def test_standalone_record_rejects_any_qualifier(self) -> None:
        schema, defs = _last_expr_plan(_STANDALONE_RECORD_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Foo::Point(x = 1)"), schema, defs)


# ---------------------------------------------------------------------------
# value_node_to_json: enums (zones, aliases, qualifier, bare form)
# ---------------------------------------------------------------------------


class TestEnumDecode:
    def test_standard_zone_accepts_positional_and_named(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        positional = value_node_to_json(read_value("Triangle(1, 2)"), schema, defs)
        named = value_node_to_json(read_value("Triangle(base = 1, height = 2)"), schema, defs)
        assert positional == named == {"$case": "Triangle", "base": 1, "height": 2}

    def test_positional_only_field_rejects_the_named_spelling(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Square(side = 1)"), schema, defs)

    def test_positional_only_field_accepts_the_positional_spelling(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        assert value_node_to_json(read_value("Square(1)"), schema, defs) == {
            "$case": "sq",
            "side": 1,
        }

    def test_named_only_field_rejects_the_positional_spelling(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Circle(1)"), schema, defs)

    def test_named_only_field_accepts_the_named_spelling(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        assert value_node_to_json(read_value("Circle(radius = 1)"), schema, defs) == {
            "$case": "Circle",
            "radius": 1,
        }

    def test_member_alias_spelling_is_accepted(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        assert value_node_to_json(read_value("sq(1)"), schema, defs) == {
            "$case": "sq",
            "side": 1,
        }

    def test_qualified_member_spelling_is_accepted(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        assert value_node_to_json(read_value("Shape::Triangle(1, 2)"), schema, defs) == {
            "$case": "Triangle",
            "base": 1,
            "height": 2,
        }

    def test_qualified_spelling_with_the_wrong_enum_name_is_rejected(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Other::Triangle(1, 2)"), schema, defs)

    def test_unknown_member_name_is_rejected(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Rhombus(1, 2)"), schema, defs)

    def test_rejects_a_non_constructor_node(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("1"), schema, defs)

    def test_bare_form_zero_field_member(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        assert value_node_to_json(read_value("Empty"), schema, defs) == {"$case": "Empty"}

    def test_empty_call_zero_field_member(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        assert value_node_to_json(read_value("Empty()"), schema, defs) == {"$case": "Empty"}

    def test_bare_form_with_fields_is_rejected(self) -> None:
        schema, defs = _last_expr_plan(_SHAPE_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Triangle"), schema, defs)

    def test_referenced_member_accepts_the_enum_name_as_qualifier(self) -> None:
        schema, defs = _last_expr_plan(_REFERENCED_MEMBER_ENUM_SRC)
        result = value_node_to_json(read_value("Stored::Saved(id = 1)"), schema, defs)
        assert result == {"$case": "Saved", "id": 1}

    def test_referenced_member_also_accepts_its_own_scope_as_qualifier(self) -> None:
        schema, defs = _last_expr_plan(_REFERENCED_MEMBER_ENUM_SRC)
        result = value_node_to_json(read_value("Data::Saved(id = 1)"), schema, defs)
        assert result == {"$case": "Saved", "id": 1}

    def test_referenced_member_rejects_an_unrelated_qualifier(self) -> None:
        schema, defs = _last_expr_plan(_REFERENCED_MEMBER_ENUM_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Other::Saved(id = 1)"), schema, defs)


# ---------------------------------------------------------------------------
# Recursive types ($defs / RefDecode)
# ---------------------------------------------------------------------------


class TestRecursiveDecode:
    def test_recursive_field_resolves_through_defs(self) -> None:
        schema, defs = _last_expr_plan(_LINK_SRC)
        result = value_node_to_json(read_value("Link::Cell(next = End)"), schema, defs)
        assert result == {"$case": "Cell", "next": {"$case": "End"}}

    def test_recursive_field_rejects_a_shape_mismatch(self) -> None:
        schema, defs = _last_expr_plan(_LINK_SRC)
        with pytest.raises(ValueDecodeError):
            value_node_to_json(read_value("Link::Cell(next = 1)"), schema, defs)


# ---------------------------------------------------------------------------
# option_some_field_schema / option_some_json_name
# ---------------------------------------------------------------------------


class TestOptionSomeHelpers:
    def test_field_schema_is_the_element_type(self) -> None:
        schema, defs = _last_expr_plan(_OPTION_INT_SRC)
        inner = option_some_field_schema(schema, defs)
        assert value_node_to_json(read_value("42"), inner, defs) == 42

    def test_field_schema_for_a_record_element(self) -> None:
        schema, defs = _last_expr_plan(_OPTION_POINT_SRC)
        inner = option_some_field_schema(schema, defs)
        result = value_node_to_json(read_value("Point(x = 1, y = 2)"), inner, defs)
        assert result == {"pixel": 1, "y": 2}

    def test_json_name_is_the_some_variant_tag(self) -> None:
        schema, defs = _last_expr_plan(_OPTION_INT_SRC)
        assert option_some_json_name(schema, defs) == "Some"


# ---------------------------------------------------------------------------
# host_text_to_json
# ---------------------------------------------------------------------------


class TestHostTextToJson:
    def test_text_schema_is_taken_verbatim(self) -> None:
        schema, defs = _scalar_plan(TextType())
        text = "  not json at all {"
        assert host_text_to_json(text, schema, defs, agent_command_fallback=False) == text

    def test_strict_json_is_preferred_when_valid(self) -> None:
        schema, defs = _scalar_plan(IntType())
        assert host_text_to_json("42", schema, defs, agent_command_fallback=False) == 42

    def test_falls_back_to_value_syntax_when_not_json(self) -> None:
        schema, defs = _last_expr_plan(_RECORD_SRC)
        result = host_text_to_json(
            "Point(x = 1, y = 2)", schema, defs, agent_command_fallback=False
        )
        assert result == {"pixel": 1, "y": 2}

    def test_reports_both_reasons_when_json_and_value_syntax_both_fail(self) -> None:
        """When neither strict JSON nor value syntax reads the text, both fail."""
        schema, defs = _scalar_plan(IntType())
        with pytest.raises(ValueDecodeError):
            host_text_to_json("not valid", schema, defs, agent_command_fallback=False)
