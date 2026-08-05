"""Unit tests for the extern (Python FFI) boundary walkers and invocation.

Drives ``encode_boundary_value``/``decode_boundary_value``/``ExternRegistry.invoke``
directly, with ``ExternContract``s compiled from real checked signatures
(``build_extern_contract``) and real Python callables (plain functions, not
files — companion loading/resolution is covered separately in
``tests/test_agl_extern_loading.py``).  Covers: the full type-mapping
matrix in both directions, live container aliasing, JSON copy independence, sealed-handle mechanics,
and the three ``invoke`` failure classes.
"""

from __future__ import annotations

import decimal
import sys
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

from agm.agl.eval._decimal import AGL_DECIMAL_CONTEXT
from agm.agl.ir.contracts import (
    BoundaryEnum,
    BoundaryException,
    BoundaryRecord,
    BoundarySchema,
    ExternContract,
)
from agm.agl.ir.ids import FunctionId, NominalId
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.boundary import (
    AglArrayView,
    AglDictView,
    BoundaryScope,
    BoundaryViolation,
    SealedHandle,
    decode_boundary_value,
    encode_boundary_value,
)
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.render import render_value
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import (
    AgentValue,
    ArrayValue,
    BoolValue,
    ConstructorValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    IrClosureValue,
    JsonValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)
from tests.agl.module_graph import build_extern_contract_from_source

_PATH = Path("/virtual/extern_boundary.agl")

_BOX = "record Box\n  value: int\n  label: text\n"
_SHAPE = "enum Shape\n  | circle(radius: decimal)\n  | rect(width: int, height: int)\n"
_BAD_THING = "exception BadThing extends Exception\n  detail: text\n"
_BOX_WITH_ARRAY = "record BoxWithArray\n  items: array[int]\n"
_SHAPE_WITH_DICT = "enum ShapeWithDict\n  | tagged(values: dict[text, int])\n"
_BAD_THING_WITH_ARRAY = "exception BadThingWithArray extends Exception\n  items: array[int]\n"


def build_contract(source: str, fn_name: str = "f") -> ExternContract:
    """Compile *fn_name*'s boundary contract from real checked AgL source."""
    return build_extern_contract_from_source(source, origin_path=_PATH, fn_name=fn_name)


def _nominal(schema: BoundarySchema) -> NominalId:
    """Narrow a nominal boundary schema to read its ``NominalId`` for test fixtures."""
    assert isinstance(schema, (BoundaryRecord, BoundaryEnum, BoundaryException))
    return schema.nominal


def _sealed_handle(value: Value, scope: BoundaryScope) -> SealedHandle:
    """Mint a test handle through the real encoder, not the public constructor."""
    contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
    encoded = encode_boundary_value(contract.params[0].schema, value, scope)
    assert isinstance(encoded, SealedHandle)
    return encoded


# ---------------------------------------------------------------------------
# Encode direction — one test per type-mapping row
# ---------------------------------------------------------------------------


class TestEncodeScalarsAndContainers:
    def test_int(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")
        assert encode_boundary_value(contract.params[0].schema, IntValue(3), BoundaryScope()) == 3

    def test_decimal_never_becomes_float(self) -> None:
        contract = build_contract("extern def f(x: decimal) -> decimal\n0")
        result = encode_boundary_value(
            contract.params[0].schema, DecimalValue(Decimal("1.5")), BoundaryScope()
        )
        assert result == Decimal("1.5")
        assert isinstance(result, Decimal)

    def test_bool(self) -> None:
        contract = build_contract("extern def f(x: bool) -> bool\n0")
        assert (
            encode_boundary_value(contract.params[0].schema, BoolValue(True), BoundaryScope())
            is True
        )

    def test_text(self) -> None:
        contract = build_contract("extern def f(x: text) -> text\n0")
        assert (
            encode_boundary_value(contract.params[0].schema, TextValue("hi"), BoundaryScope())
            == "hi"
        )

    def test_unit(self) -> None:
        contract = build_contract("extern def f(x: unit) -> unit\n0")
        assert (
            encode_boundary_value(contract.params[0].schema, UnitValue(), BoundaryScope()) is None
        )

    def test_json_scalar(self) -> None:
        contract = build_contract("extern def f(x: json) -> json\n0")
        result = encode_boundary_value(
            contract.params[0].schema, JsonValue(Decimal("2.5")), BoundaryScope()
        )
        assert result == Decimal("2.5")

    def test_array(self) -> None:
        contract = build_contract("extern def f(x: array[int]) -> array[int]\n0")
        value = ArrayValue([IntValue(1), IntValue(2)])
        encoded = encode_boundary_value(contract.params[0].schema, value, BoundaryScope())
        assert isinstance(encoded, AglArrayView)
        assert list(encoded) == [1, 2]

    def test_dict(self) -> None:
        contract = build_contract("extern def f(x: dict[text, int]) -> dict[text, int]\n0")
        value = DictValue(entries={"a": IntValue(1), "b": IntValue(2)})
        encoded = encode_boundary_value(contract.params[0].schema, value, BoundaryScope())
        assert isinstance(encoded, AglDictView)
        assert dict(encoded) == {"a": 1, "b": 2}


class TestEncodeNominals:
    def test_record_with_nested_array(self) -> None:
        contract = build_contract(
            _BOX_WITH_ARRAY + "extern def f(b: BoxWithArray) -> BoxWithArray\n0"
        )
        value = RecordValue(
            nominal=_nominal(contract.result),
            display_name="BoxWithArray",
            fields={"items": ArrayValue([IntValue(1), IntValue(2)])},
        )
        result = encode_boundary_value(contract.params[0].schema, value, BoundaryScope())
        assert isinstance(result, dict)
        items = result["items"]
        assert isinstance(items, AglArrayView)
        assert list(items) == [1, 2]

    def test_enum_variant_with_nested_dict(self) -> None:
        contract = build_contract(
            _SHAPE_WITH_DICT + "extern def f(s: ShapeWithDict) -> ShapeWithDict\n0"
        )
        value = EnumValue(
            nominal=_nominal(contract.result),
            display_name="ShapeWithDict",
            variant="tagged",
            fields={"values": DictValue(entries={"width": IntValue(3), "height": IntValue(4)})},
        )
        result = encode_boundary_value(contract.params[0].schema, value, BoundaryScope())
        assert isinstance(result, dict)
        assert result["$case"] == "tagged"
        values = result["values"]
        assert isinstance(values, AglDictView)
        assert dict(values) == {"width": 3, "height": 4}

    def test_enum_variant_with_no_fields(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(s: Shape) -> Shape\n0")
        value = EnumValue(
            nominal=_nominal(contract.result),
            display_name="Shape",
            variant="circle",
            fields={"radius": DecimalValue(Decimal(2))},
        )
        result = encode_boundary_value(contract.params[0].schema, value, BoundaryScope())
        assert result == {"$case": "circle", "radius": Decimal(2)}

    def test_exception_with_nested_array(self) -> None:
        contract = build_contract(
            _BAD_THING_WITH_ARRAY + "extern def f(x: int) -> BadThingWithArray\n0"
        )
        value = ExceptionValue(
            nominal=_nominal(contract.result),
            display_name="BadThingWithArray",
            fields={
                "message": TextValue("m"),
                "trace_id": TextValue(""),
                "items": ArrayValue([IntValue(1), IntValue(2)]),
            },
        )
        result = encode_boundary_value(contract.result, value, BoundaryScope())
        assert isinstance(result, dict)
        assert result["message"] == "m"
        assert result["trace_id"] == ""
        items = result["items"]
        assert isinstance(items, AglArrayView)
        assert list(items) == [1, 2]


# ---------------------------------------------------------------------------
# Encode direction — container aliasing and JSON copy independence
# ---------------------------------------------------------------------------


class TestEncodeAliasingAndJsonCopy:
    def test_encoded_array_and_agl_value_observe_each_others_mutations(self) -> None:
        contract = build_contract("extern def f(x: array[int]) -> array[int]\n0")
        original = ArrayValue([IntValue(1), IntValue(2)])
        encoded = encode_boundary_value(contract.params[0].schema, original, BoundaryScope())
        assert isinstance(encoded, AglArrayView)
        encoded.append(99)
        assert original.elements == [IntValue(1), IntValue(2), IntValue(99)]
        original.elements[0] = IntValue(3)
        assert encoded[0] == 3

    def test_mutating_encoded_json_does_not_affect_agl_value(self) -> None:
        contract = build_contract("extern def f(x: json) -> json\n0")
        inner: dict[str, object] = {"a": [1, 2]}
        original = JsonValue(inner)
        encoded = encode_boundary_value(contract.params[0].schema, original, BoundaryScope())
        assert isinstance(encoded, dict)
        assert isinstance(encoded["a"], list)
        encoded["a"].append(3)
        assert inner["a"] == [1, 2]
        assert isinstance(original.raw, dict)
        assert original.raw["a"] == [1, 2]

    def test_mutating_encoded_nested_json_list_element_does_not_affect_agl_value(self) -> None:
        contract = build_contract("extern def f(x: array[json]) -> array[json]\n0")
        inner_list: list[object] = [1, 2]
        original = ArrayValue([JsonValue(inner_list)])
        encoded = encode_boundary_value(contract.params[0].schema, original, BoundaryScope())
        assert isinstance(encoded, AglArrayView)
        nested_json = encoded[0]
        assert isinstance(nested_json, list)
        nested_json.append(999)
        assert inner_list == [1, 2]


# ---------------------------------------------------------------------------
# Encode direction — shape mismatches (argument-conversion failures)
# ---------------------------------------------------------------------------


class TestEncodeShapeMismatch:
    def test_int_scalar_mismatch(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, TextValue("nope"), BoundaryScope())

    def test_text_scalar_mismatch(self) -> None:
        contract = build_contract("extern def f(x: text) -> text\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_decimal_scalar_mismatch(self) -> None:
        contract = build_contract("extern def f(x: decimal) -> decimal\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_bool_scalar_mismatch(self) -> None:
        contract = build_contract("extern def f(x: bool) -> bool\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_json_scalar_mismatch(self) -> None:
        contract = build_contract("extern def f(x: json) -> json\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_unit_mismatch(self) -> None:
        contract = build_contract("extern def f(x: unit) -> unit\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_array_mismatch(self) -> None:
        contract = build_contract("extern def f(x: array[int]) -> array[int]\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_dict_mismatch(self) -> None:
        contract = build_contract("extern def f(x: dict[text, int]) -> dict[text, int]\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_record_mismatch(self) -> None:
        contract = build_contract(_BOX + "extern def f(b: Box) -> Box\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_enum_mismatch(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(s: Shape) -> Shape\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, IntValue(1), BoundaryScope())

    def test_exception_mismatch(self) -> None:
        contract = build_contract(_BAD_THING + "extern def f(x: int) -> BadThing\n0")
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.result, IntValue(1), BoundaryScope())

    def test_enum_unknown_variant_rejected(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(s: Shape) -> Shape\n0")
        value = EnumValue(
            nominal=_nominal(contract.result),
            display_name="Shape",
            variant="triangle",
            fields={},
        )
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(contract.params[0].schema, value, BoundaryScope())


# ---------------------------------------------------------------------------
# Decode direction — happy path, tolerances, and strict rejections
# ---------------------------------------------------------------------------


class TestDecodeScalars:
    def test_int_accepts_int(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")
        assert decode_boundary_value(contract.result, 5, BoundaryScope()) == IntValue(5)

    def test_decimal_accepts_int_exactly(self) -> None:
        contract = build_contract("extern def f(x: int) -> decimal\n0")
        assert decode_boundary_value(contract.result, 7, BoundaryScope()) == DecimalValue(
            Decimal(7)
        )

    def test_decimal_accepts_decimal(self) -> None:
        contract = build_contract("extern def f(x: int) -> decimal\n0")
        assert decode_boundary_value(
            contract.result, Decimal("1.25"), BoundaryScope()
        ) == DecimalValue(Decimal("1.25"))

    def test_decimal_rejects_non_finite_decimal(self) -> None:
        contract = build_contract("extern def f(x: int) -> decimal\n0")
        for raw in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with pytest.raises(BoundaryViolation):
                decode_boundary_value(contract.result, raw, BoundaryScope())

    def test_bool_rejected_where_int_declared(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, True, BoundaryScope())

    def test_bool_rejected_where_decimal_declared(self) -> None:
        contract = build_contract("extern def f(x: int) -> decimal\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, True, BoundaryScope())

    def test_bool_accepted_where_bool_declared(self) -> None:
        contract = build_contract("extern def f(x: int) -> bool\n0")
        assert decode_boundary_value(contract.result, True, BoundaryScope()) == BoolValue(True)

    def test_bool_rejects_non_bool(self) -> None:
        contract = build_contract("extern def f(x: int) -> bool\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, 1, BoundaryScope())

    def test_float_rejected_at_int(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, 1.5, BoundaryScope())

    def test_float_rejected_at_decimal(self) -> None:
        contract = build_contract("extern def f(x: int) -> decimal\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, 1.5, BoundaryScope())

    def test_float_rejected_in_json(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, 1.5, BoundaryScope())

    def test_float_rejected_nested_in_json(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, {"a": [1.5]}, BoundaryScope())

    def test_text_wrong_type_rejected(self) -> None:
        contract = build_contract("extern def f(x: int) -> text\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, 1, BoundaryScope())


class TestDecodeContainers:
    def test_array(self) -> None:
        contract = build_contract("extern def f(x: int) -> array[int]\n0")
        assert decode_boundary_value(contract.result, [1, 2], BoundaryScope()) == ArrayValue(
            [IntValue(1), IntValue(2)]
        )

    def test_array_wrong_type_rejected(self) -> None:
        contract = build_contract("extern def f(x: int) -> array[int]\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, "not a list", BoundaryScope())

    def test_cyclic_array_crosses_lazily(self) -> None:
        contract = build_contract("extern def f(x: array[array[int]]) -> int\n0")
        value = ArrayValue([])
        value.elements.append(value)

        encoded = encode_boundary_value(contract.params[0].schema, value, BoundaryScope())

        assert isinstance(encoded, AglArrayView)

    def test_cyclic_python_array_return_rejected(self) -> None:
        contract = build_contract("extern def f(x: int) -> array[array[int]]\n0")
        value: list[object] = []
        value.append(value)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, value, BoundaryScope())

    def test_dict(self) -> None:
        contract = build_contract("extern def f(x: int) -> dict[text, int]\n0")
        assert decode_boundary_value(contract.result, {"a": 1}, BoundaryScope()) == DictValue(
            entries={"a": IntValue(1)}
        )

    def test_dict_non_string_key_rejected(self) -> None:
        contract = build_contract("extern def f(x: int) -> dict[text, int]\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, {1: 1}, BoundaryScope())

    def test_dict_wrong_type_rejected(self) -> None:
        contract = build_contract("extern def f(x: int) -> dict[text, int]\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, [1], BoundaryScope())


class TestDecodeNominals:
    def test_record_happy_path(self) -> None:
        contract = build_contract(_BOX + "extern def f(x: int) -> Box\n0")
        result = decode_boundary_value(
            contract.result, {"value": 1, "label": "hi"}, BoundaryScope()
        )
        assert result == RecordValue(
            nominal=_nominal(contract.result),
            display_name="Box",
            fields={"value": IntValue(1), "label": TextValue("hi")},
        )

    def test_record_wrong_type_rejected(self) -> None:
        contract = build_contract(_BOX + "extern def f(x: int) -> Box\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, "nope", BoundaryScope())

    def test_record_non_string_key_rejected(self) -> None:
        contract = build_contract(_BOX + "extern def f(x: int) -> Box\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, {1: "x", "label": "y"}, BoundaryScope())

    def test_record_missing_field_rejected(self) -> None:
        contract = build_contract(_BOX + "extern def f(x: int) -> Box\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, {"value": 1}, BoundaryScope())

    def test_record_extra_field_rejected(self) -> None:
        contract = build_contract(_BOX + "extern def f(x: int) -> Box\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(
                contract.result, {"value": 1, "label": "x", "extra": 1}, BoundaryScope()
            )

    def test_record_misnamed_field_rejected(self) -> None:
        contract = build_contract(_BOX + "extern def f(x: int) -> Box\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, {"value": 1, "lbl": "x"}, BoundaryScope())

    def test_enum_happy_path(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(x: int) -> Shape\n0")
        result = decode_boundary_value(
            contract.result, {"$case": "rect", "width": 3, "height": 4}, BoundaryScope()
        )
        assert result == EnumValue(
            nominal=_nominal(contract.result),
            display_name="Shape",
            variant="rect",
            fields={"width": IntValue(3), "height": IntValue(4)},
        )

    def test_enum_missing_case_rejected(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(x: int) -> Shape\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, {"width": 3, "height": 4}, BoundaryScope())

    def test_enum_unknown_variant_rejected(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(x: int) -> Shape\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, {"$case": "triangle"}, BoundaryScope())

    def test_enum_missing_field_rejected(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(x: int) -> Shape\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, {"$case": "rect", "width": 3}, BoundaryScope())

    def test_enum_extra_field_rejected(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(x: int) -> Shape\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(
                contract.result,
                {"$case": "rect", "width": 3, "height": 4, "extra": 1},
                BoundaryScope(),
            )

    def test_enum_wrong_type_rejected(self) -> None:
        contract = build_contract(_SHAPE + "extern def f(x: int) -> Shape\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, "nope", BoundaryScope())

    def test_exception_happy_path(self) -> None:
        contract = build_contract(_BAD_THING + "extern def f(x: int) -> BadThing\n0")
        result = decode_boundary_value(
            contract.result, {"message": "m", "trace_id": "t", "detail": "d"}, BoundaryScope()
        )
        assert result == ExceptionValue(
            nominal=_nominal(contract.result),
            display_name="BadThing",
            fields={
                "message": TextValue("m"),
                "trace_id": TextValue("t"),
                "detail": TextValue("d"),
            },
        )

    def test_exception_misnamed_field_rejected(self) -> None:
        contract = build_contract(_BAD_THING + "extern def f(x: int) -> BadThing\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(
                contract.result,
                {"message": "m", "trace_id": "t", "info": "d"},
                BoundaryScope(),
            )

    def test_exception_wrong_type_rejected(self) -> None:
        contract = build_contract(_BAD_THING + "extern def f(x: int) -> BadThing\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, 1, BoundaryScope())


class TestDecodeUnit:
    def test_unit_requires_none(self) -> None:
        contract = build_contract("extern def f(x: int) -> unit\n0")
        assert decode_boundary_value(contract.result, None, BoundaryScope()) == UnitValue()

    def test_unit_rejects_non_none(self) -> None:
        contract = build_contract("extern def f(x: int) -> unit\n0")
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, 0, BoundaryScope())


class TestDecodeJson:
    def test_accepts_decimal_and_nested_shapes(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")
        obj: dict[str, object] = {"a": [1, Decimal("2.5"), None, True, "s"]}
        result = decode_boundary_value(contract.result, obj, BoundaryScope())
        assert isinstance(result, JsonValue)
        assert result.raw == obj

    def test_rejects_non_finite_decimal_nested_in_json(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")
        for raw in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with pytest.raises(BoundaryViolation):
                decode_boundary_value(contract.result, {"bad": [raw]}, BoundaryScope())

    def test_rejects_arbitrary_object(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")

        class Opaque:
            pass

        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, Opaque(), BoundaryScope())

    def test_rejects_cyclic_json_list(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")
        value: list[object] = []
        value.append(value)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, value, BoundaryScope())

    def test_rejects_cyclic_json_dict(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")
        value: dict[str, object] = {}
        value["self"] = value
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, value, BoundaryScope())

    def test_json_deepcopy_failure_is_boundary_violation(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")

        class BadStr(str):
            def __reduce_ex__(self, protocol: int) -> object:
                raise RuntimeError("copy failed")

        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, BadStr("x"), BoundaryScope())

    def test_rejects_sealed_handle(self) -> None:
        contract = build_contract("extern def f(x: int) -> json\n0")
        handle = _sealed_handle(IntValue(1), BoundaryScope(seals={"T": object()}))
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, handle, BoundaryScope())


# ---------------------------------------------------------------------------
# Sealing mechanics
# ---------------------------------------------------------------------------


class TestSealing:
    def test_encode_wraps_type_var_position_in_handle(self) -> None:
        contract = build_contract(
            "extern def reverse[T](xs: array[T]) -> array[T]\n0", fn_name="reverse"
        )
        scope = BoundaryScope(seals={"T": object()})
        value = ArrayValue([IntValue(1)])
        encoded = encode_boundary_value(contract.params[0].schema, value, scope)
        assert isinstance(encoded[0], SealedHandle)

    def test_decode_accepts_this_calls_handle(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
        scope = BoundaryScope(seals={"T": object()})
        handle = _sealed_handle(IntValue(5), scope)
        assert decode_boundary_value(contract.result, handle, scope) == IntValue(5)

    def test_handle_does_not_expose_mutable_value_payload(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
        scope = BoundaryScope(seals={"T": object()})
        original = DictValue(entries={"a": IntValue(1)})
        encoded = encode_boundary_value(contract.params[0].schema, original, scope)
        assert isinstance(encoded, SealedHandle)

        assert not hasattr(encoded, "_SealedHandle__eq_key")
        assert decode_boundary_value(contract.result, encoded, scope) == original

    def test_handle_equality_keys_cover_all_sealable_value_shapes(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
        nominal = NominalId(ENTRY_ID, "Choice")
        values = (
            DecimalValue(Decimal("1.5")),
            BoolValue(True),
            JsonValue({"items": [True, 1, Decimal("2"), "x", None]}),
            ArrayValue([IntValue(1)]),
            EnumValue(nominal=nominal, display_name="Choice", variant="some", fields={}),
            ExceptionValue(nominal=nominal, display_name="Oops", fields={"msg": TextValue("x")}),
            UnitValue(),
            AgentValue("runner"),
            ConstructorValue(nominal=nominal, display_name="Choice", variant="some"),
        )

        for value in values:
            left = encode_boundary_value(
                contract.params[0].schema, value, BoundaryScope(seals={"T": object()})
            )
            right = encode_boundary_value(
                contract.params[0].schema, value, BoundaryScope(seals={"T": object()})
            )
            assert isinstance(left, SealedHandle)
            assert isinstance(right, SealedHandle)
            assert left == right
            assert hash(left) == hash(right)

    def test_decode_rejects_stale_handle_from_a_different_call(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
        stale_scope = BoundaryScope(seals={"T": object()})
        this_call_scope = BoundaryScope(seals={"T": object()})
        handle = _sealed_handle(IntValue(5), stale_scope)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, handle, this_call_scope)

    def test_decode_rejects_handle_with_fresh_implicit_scope(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
        handle = _sealed_handle(IntValue(5), BoundaryScope(seals={"T": object()}))
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, handle)

    def test_decode_rejects_cross_variable_handle(self) -> None:
        contract = build_contract("extern def pair[A, B](a: A, b: B) -> A\n0", fn_name="pair")
        scope = BoundaryScope(seals={"A": object(), "B": object()})
        encoded = encode_boundary_value(contract.params[1].schema, IntValue(1), scope)
        assert isinstance(encoded, SealedHandle)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, encoded, scope)

    def test_decode_rejects_raw_forged_value(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
        scope = BoundaryScope(seals={"T": object()})
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, 5, scope)

    def test_decode_rejects_uninitialized_handle_instance(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
        forged = object.__new__(SealedHandle)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(contract.result, forged, BoundaryScope(seals={"T": object()}))

    def test_uninitialized_handle_special_methods_are_defensive(self) -> None:
        forged = object.__new__(SealedHandle)
        assert forged != _sealed_handle(IntValue(1), BoundaryScope(seals={"T": object()}))
        with pytest.raises(TypeError):
            hash(forged)
        assert repr(forged) == "<sealed handle>"

    def test_decode_rejects_handle_with_forged_public_shape(self) -> None:
        from agm.agl.runtime import boundary as boundary_mod

        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")
        forged = object.__new__(SealedHandle)
        # Give the forged instance the public shape of a real handle (so it
        # hashes and reprs like one), yet it was never minted into any vault.
        boundary_mod._HANDLE_STATE[id(forged)] = boundary_mod._HandleState(IntValue(1))
        try:
            assert hash(forged) == hash(
                _sealed_handle(IntValue(1), BoundaryScope(seals={"T": object()}))
            )
            with pytest.raises(BoundaryViolation):
                decode_boundary_value(contract.result, forged, BoundaryScope(seals={"T": object()}))
        finally:
            boundary_mod._HANDLE_STATE.pop(id(forged), None)

    def test_public_constructor_cannot_forge_handle(self) -> None:
        with pytest.raises(TypeError):
            SealedHandle(IntValue(1), object())

    def test_handle_exposes_no_value_or_seal_attributes(self) -> None:
        h = _sealed_handle(IntValue(1), BoundaryScope(seals={"T": object()}))
        assert not hasattr(h, "_value")
        assert not hasattr(h, "_seal")

    def test_handle_equality_and_hash_delegate_to_wrapped_value(self) -> None:
        h1 = _sealed_handle(IntValue(1), BoundaryScope(seals={"T": object()}))
        h2 = _sealed_handle(IntValue(1), BoundaryScope(seals={"T": object()}))
        assert h1 == h2
        assert hash(h1) == hash(h2)
        assert h1 != _sealed_handle(IntValue(2), BoundaryScope(seals={"T": object()}))

    def test_handle_never_equals_a_non_handle(self) -> None:
        h = _sealed_handle(IntValue(1), BoundaryScope(seals={"T": object()}))
        assert (h == 1) is False
        assert h != 1

    def test_handles_work_in_sets_and_dicts(self) -> None:
        h1 = _sealed_handle(TextValue("a"), BoundaryScope(seals={"T": object()}))
        h2 = _sealed_handle(TextValue("a"), BoundaryScope(seals={"T": object()}))
        assert h2 in {h1}
        mapping = {h1: "found"}
        assert mapping[h2] == "found"

    def test_handle_repr_shows_rendered_value(self) -> None:
        h = _sealed_handle(TextValue("hi"), BoundaryScope(seals={"T": object()}))
        assert repr(h) == render_value(TextValue("hi"))

    def test_handle_repr_reflects_mutations_to_the_wrapped_value(self) -> None:
        original = ArrayValue([IntValue(1), IntValue(2)])
        h = _sealed_handle(original, BoundaryScope(seals={"T": object()}))
        assert repr(h) == "[1, 2]"

        original.elements[0] = IntValue(99)

        assert repr(h) == "[99, 2]"

    def test_identity_equality_values_seal_and_hash_without_error(self) -> None:
        closure = IrClosureValue(function_id=FunctionId(0), captures=())
        h1 = _sealed_handle(closure, BoundaryScope(seals={"T": object()}))
        h2 = _sealed_handle(closure, BoundaryScope(seals={"T": object()}))
        assert h1 == h2
        assert hash(h1) == hash(h2)

        other_closure = IrClosureValue(function_id=FunctionId(0), captures=())
        h3 = _sealed_handle(other_closure, BoundaryScope(seals={"T": object()}))
        assert h1 != h3  # distinct closure identity, even with identical fields


# ---------------------------------------------------------------------------
# invoke — call order and the three failure classes
# ---------------------------------------------------------------------------


class TestInvoke:
    def test_positional_call_order(self) -> None:
        contract = build_contract("extern def f(a: int, b: text) -> text\n0")
        captured: dict[str, tuple[object, ...]] = {}

        def fn(a: int, b: str) -> str:
            captured["args"] = (a, b)
            return f"{a}-{b}"

        registry = ExternRegistry()
        result = registry.invoke("f", contract, fn, [IntValue(1), TextValue("x")], "trace-1")
        assert captured["args"] == (1, "x")
        assert result == TextValue("1-x")

    def test_roundtrips_sealed_value_through_identity_function(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")

        def fn(x: object) -> object:
            return x

        registry = ExternRegistry()
        result = registry.invoke("identity", contract, fn, [IntValue(42)], "trace-2")
        assert result == IntValue(42)

    def test_companion_decimal_context_mutation_is_isolated(self) -> None:
        contract = build_contract("extern def f() -> int\n0")

        def fn() -> int:
            decimal.getcontext().prec = 2
            return 1

        registry = ExternRegistry()
        with decimal.localcontext(AGL_DECIMAL_CONTEXT):
            result = registry.invoke("f", contract, fn, [], "trace-decimal")
            assert result == IntValue(1)
            assert decimal.getcontext().prec == AGL_DECIMAL_CONTEXT.prec

    def test_python_exception_becomes_extern_error_with_class_name(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")

        def fn(x: int) -> int:
            raise ValueError("boom")

        registry = ExternRegistry()
        with pytest.raises(AglRaise) as excinfo:
            registry.invoke("f", contract, fn, [IntValue(1)], "trace-3")
        exc = excinfo.value.exc
        assert exc.display_name == "ExternError"
        assert exc.fields["function"] == TextValue("f")
        assert exc.fields["python_type"] == TextValue("ValueError")
        assert exc.fields["trace_id"] == TextValue("trace-3")

        from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS

        assert exc.nominal == NO_BUILTIN_DECLARATIONS.nominal("ExternError")

    def test_invoke_forwards_a_custom_nominals_table(self) -> None:
        """A caller-supplied ``nominals`` table (not the default) is honored."""
        from agm.agl.ir.builtin_nominals import BuiltinNominals

        contract = build_contract("extern def f(x: int) -> int\n0")

        def fn(x: int) -> int:
            raise ValueError("boom")

        registry = ExternRegistry()
        custom = BuiltinNominals(declared={"ExternError": NominalId(ENTRY_ID, "ExternError")})
        with pytest.raises(AglRaise) as excinfo:
            registry.invoke("f", contract, fn, [IntValue(1)], "trace-custom", nominals=custom)
        assert excinfo.value.exc.nominal == NominalId(ENTRY_ID, "ExternError")

    def test_return_contract_violation_has_empty_python_type(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")

        def fn(x: int) -> str:
            return "not an int"

        registry = ExternRegistry()
        with pytest.raises(AglRaise) as excinfo:
            registry.invoke("f", contract, fn, [IntValue(1)], "trace-4")
        exc = excinfo.value.exc
        assert exc.display_name == "ExternError"
        assert exc.fields["python_type"] == TextValue("")

    def test_list_subclass_return_is_a_contract_violation(self) -> None:
        contract = build_contract("extern def f() -> array[int]\n0")

        class BadList(list[object]):
            def __iter__(self) -> Iterator[object]:
                raise RuntimeError("iteration should not escape")

        def fn() -> object:
            return BadList([1])

        registry = ExternRegistry()
        with pytest.raises(AglRaise) as excinfo:
            registry.invoke("f", contract, fn, [], "trace-list-subclass")
        exc = excinfo.value.exc
        assert exc.display_name == "ExternError"
        assert exc.fields["python_type"] == TextValue("")

    def test_unexpected_return_decode_exception_becomes_extern_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract = build_contract("extern def f() -> int\n0")

        def boom(schema: BoundarySchema, obj: object, scope: BoundaryScope | None = None) -> Value:
            raise RuntimeError("decode exploded")

        monkeypatch.setattr("agm.agl.runtime.externs.decode_boundary_value", boom)

        registry = ExternRegistry()
        with pytest.raises(AglRaise) as excinfo:
            registry.invoke("f", contract, lambda: 1, [], "trace-decode-error")
        exc = excinfo.value.exc
        assert exc.display_name == "ExternError"
        assert exc.fields["python_type"] == TextValue("RuntimeError")

    def test_cyclic_python_return_becomes_extern_error(self) -> None:
        contract = build_contract("extern def f() -> json\n0")

        def fn() -> object:
            value: list[object] = []
            value.append(value)
            return value

        registry = ExternRegistry()
        with pytest.raises(AglRaise) as excinfo:
            registry.invoke("f", contract, fn, [], "trace-cycle")
        exc = excinfo.value.exc
        assert exc.display_name == "ExternError"
        assert exc.fields["python_type"] == TextValue("")

    def test_seal_violation_on_return_is_a_contract_violation(self) -> None:
        contract = build_contract("extern def identity[T](x: T) -> T\n0", fn_name="identity")

        def fn(x: object) -> object:
            return 999  # forged raw value instead of the handle it received

        registry = ExternRegistry()
        with pytest.raises(AglRaise) as excinfo:
            registry.invoke("identity", contract, fn, [IntValue(1)], "trace-5")
        assert excinfo.value.exc.fields["python_type"] == TextValue("")

    def test_argument_conversion_failure_has_empty_python_type(self) -> None:
        contract = build_contract("extern def f(x: int) -> int\n0")

        def fn(x: int) -> int:
            return x  # never reached

        registry = ExternRegistry()
        with pytest.raises(AglRaise) as excinfo:
            registry.invoke("f", contract, fn, [TextValue("not an int")], "trace-6")
        exc = excinfo.value.exc
        assert exc.display_name == "ExternError"
        assert exc.fields["python_type"] == TextValue("")


# ---------------------------------------------------------------------------
# ExternRegistry misuse contract
# ---------------------------------------------------------------------------


class TestExternRegistryMisuse:
    def test_entry_companion_synthetic_name_is_valid(self, tmp_path: Path) -> None:
        from agm.agl.modules.ids import ENTRY_ID

        companion = tmp_path / "entry.py"
        companion.write_text("captured_name = __name__\n")
        registry = ExternRegistry()

        module = registry.load_companion(ENTRY_ID, companion)

        assert getattr(module, "captured_name") == "agm_agl_extern_companion__entry__0"
        assert "\x00" not in module.__name__

    def test_companion_may_remove_its_synthetic_module_during_import(self, tmp_path: Path) -> None:
        from agm.agl.modules.ids import ModuleId

        companion = tmp_path / "lib.py"
        companion.write_text("import sys\ndel sys.modules[__name__]\ndef f():\n    return 1\n")
        registry = ExternRegistry()
        module_id = ModuleId.from_path("lib")

        module = registry.load_companion(module_id, companion)

        assert registry.resolve(module_id, "f")() == 1
        assert module.__name__ not in sys.modules

    def test_resolve_before_load_companion_is_a_programming_error(self) -> None:
        from agm.agl.modules.ids import ModuleId

        registry = ExternRegistry()
        with pytest.raises(AssertionError):
            registry.resolve(ModuleId.from_path("lib/mod"), "f")
