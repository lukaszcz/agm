"""Value-directed extern boundary behavior."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind, VariantDescriptor
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.boundary import (
    AglArrayView,
    AglDictView,
    AglJson,
    BoundaryTypeError,
    BoundaryViolation,
    decode_boundary_value,
    encode_boundary_value,
    pop_nominal_classes,
    push_nominal_classes,
    synthesize_nominal_classes,
)
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import (
    UNIT_VALUE,
    AgentValue,
    ArrayValue,
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
)
from tests.agl.ir_harness import evaluate_ir_raises_with_externs, evaluate_ir_with_externs


class TestValueDirectedBoundary:
    def test_json_scalars_remain_distinct_from_native_scalars(self, tmp_path: Path) -> None:
        source = (
            "extern def json_null() -> json\n"
            "extern def json_number() -> json\n"
            "let a = json_null()\n"
            "let b = json_number()\n"
            "a\n"
        )
        companion = (
            "from agl import json\n"
            "def json_null(): return json(None)\n"
            "def json_number(): return json(3)\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["a"] == JsonValue(None)
        assert result["b"] == JsonValue(3)

    def test_nominals_cross_as_program_specific_classes(self, tmp_path: Path) -> None:
        source = (
            "record Box\n  value: int\n"
            "extern def bump(box: Box) -> Box\n"
            "let result = bump(Box(value = 1))\n"
            "result\n"
        )
        companion = "from agl import Box\ndef bump(box): return Box(value=box.value + 1)\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert isinstance(result["result"], RecordValue)
        assert result["result"].fields == {"value": IntValue(2)}

    def test_bare_python_container_is_not_an_agl_value(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> array[int]\nlet _ = f()\n()\n",
            "def f(): return [1]\n",
            tmp_path,
        )
        assert exc.fields["python_type"].value == ""


def test_container_views_cover_mutating_sequence_and_mapping_operations() -> None:
    array_value = ArrayValue([IntValue(3), IntValue(1), IntValue(2)])
    array = AglArrayView(array_value)
    assert array[1:] == [1, 2]
    assert array[0] == 3
    array[1:] = [4, 5]
    del array[1:2]
    array.insert(1, 2)
    array.append(0)
    array.extend([6])
    array.reverse()
    array.sort()
    assert list(array) == [0, 2, 3, 5, 6]
    assert 2 in array
    assert array_value.elements == [IntValue(0), IntValue(2), IntValue(3), IntValue(5), IntValue(6)]
    with pytest.raises(BoundaryTypeError):
        array[0] = object()
    with pytest.raises(BoundaryTypeError):
        array.insert(0, object())
    with pytest.raises(BoundaryTypeError):
        array.extend([object()])
    with pytest.raises(TypeError):
        array[0:1] = object()

    dict_value = DictValue({"one": IntValue(1), "two": IntValue(2)})
    mapping = AglDictView(dict_value)
    assert mapping["one"] == 1
    assert "two" in mapping
    with pytest.raises(TypeError):
        mapping[1] = 1
    with pytest.raises(BoundaryTypeError):
        mapping["bad"] = object()
    assert mapping.popitem() == ("two", 2)
    del mapping["one"]
    mapping.clear()
    assert not mapping
    assert list(mapping) == []
    assert array != object()
    assert mapping != object()
    assert hash(array) == id(array_value)
    assert hash(mapping) == id(dict_value)
    assert repr(array) == "[0, 2, 3, 5, 6]"
    assert repr(mapping) == "{}"
    array.clear()
    assert not array


def test_value_mapping_and_synthesized_nominals() -> None:
    nominal = NominalId(ENTRY_ID, "Box")
    enum_nominal = NominalId(ENTRY_ID, "Choice")
    exception_nominal = NominalId(ENTRY_ID, "Problem")
    descriptors = {
        nominal: NominalDescriptor(nominal, "Box", NominalKind.RECORD, fields=("value",)),
        enum_nominal: NominalDescriptor(
            enum_nominal,
            "Choice",
            NominalKind.ENUM,
            variants=(VariantDescriptor("Some", ("value",)), VariantDescriptor("None", ())),
        ),
        exception_nominal: NominalDescriptor(
            exception_nominal, "Problem", NominalKind.EXCEPTION, fields=("detail",)
        ),
    }
    classes, by_type = synthesize_nominal_classes(descriptors.values())
    tokens = push_nominal_classes(classes, by_type)
    try:
        assert encode_boundary_value(UNIT_VALUE) is None
        assert encode_boundary_value(BoolValue(True)) is True
        assert encode_boundary_value(IntValue(1)) == 1
        assert encode_boundary_value(DecimalValue(Decimal("1.5"))) == Decimal("1.5")
        assert encode_boundary_value(TextValue("x")) == "x"
        assert encode_boundary_value(JsonValue(None)) == AglJson(None)
        assert isinstance(encode_boundary_value(ArrayValue([IntValue(1)])), AglArrayView)
        assert isinstance(encode_boundary_value(DictValue({"x": IntValue(1)})), AglDictView)
        assert decode_boundary_value(None) == UNIT_VALUE
        assert decode_boundary_value(True) == BoolValue(True)
        assert decode_boundary_value(1) == IntValue(1)
        assert decode_boundary_value(Decimal("1.5")) == DecimalValue(Decimal("1.5"))
        assert decode_boundary_value("x") == TextValue("x")

        box = classes[nominal](value=1)
        assert repr(box) == "Box(...)"
        with pytest.raises(AttributeError):
            box.value = 2
        with pytest.raises(TypeError):
            classes[nominal]()
        assert decode_boundary_value(box) == RecordValue(nominal, "Box", {"value": IntValue(1)})

        some = classes[enum_nominal].Some(value=2)
        assert isinstance(some, classes[enum_nominal])
        assert decode_boundary_value(some) == EnumValue(
            enum_nominal, "Choice", "Some", {"value": IntValue(2)}
        )
        assert isinstance(
            encode_boundary_value(EnumValue(enum_nominal, "Choice", "None", {})),
            getattr(classes[enum_nominal], "None"),
        )
        problem = classes[exception_nominal](detail="bad")
        assert decode_boundary_value(problem) == ExceptionValue(
            exception_nominal, "Problem", {"detail": TextValue("bad")}
        )
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(RecordValue(NominalId(ENTRY_ID, "Missing"), "Missing", {}))
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(AgentValue("agent"))
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(object())
        assert decode_boundary_value(AglJson({"x": []})) == JsonValue({"x": []})
        cyclic: list[object] = []
        cyclic.append(cyclic)
        with pytest.raises(BoundaryViolation):
            decode_boundary_value(AglJson(cyclic))
        array_view = AglArrayView(ArrayValue([]))
        dict_view = AglDictView(DictValue())
        assert decode_boundary_value(array_view) is array_view._value
        assert decode_boundary_value(dict_view) is dict_view._value
    finally:
        pop_nominal_classes(tokens)


def test_registry_wraps_unexpected_decode_errors_as_extern_errors() -> None:
    nominal = NominalId(ENTRY_ID, "Box")
    descriptor = NominalDescriptor(nominal, "Box", NominalKind.RECORD, fields=("value",))
    registry = ExternRegistry()
    registry.set_nominals({nominal: descriptor})
    broken = registry._nominal_classes[nominal](value=1)
    object.__delattr__(broken, "value")

    with pytest.raises(AglRaise) as excinfo:
        registry.invoke("broken", lambda: broken, (), "trace")

    assert excinfo.value.exc.fields["python_type"] == TextValue("AttributeError")

    def test_generic_aliases_need_no_schema_reconciliation(self, tmp_path: Path) -> None:
        source = (
            "extern def add[T, U](a: array[T], b: array[U]) -> unit\n"
            "var xs = [1]\n"
            "let _: unit = add(xs, xs)\n"
            "xs\n"
        )
        companion = "def add(a, b):\n    assert a == b\n    a.append(2)\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["xs"] == ArrayValue([IntValue(1), IntValue(2)])

    def test_cyclic_json_wrapper_is_rejected_without_recursing_forever(
        self, tmp_path: Path
    ) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> json\nlet _ = f()\n()\n",
            (
                "from agl import json\n"
                "def f():\n"
                "    value = []\n"
                "    value.append(value)\n"
                "    return json(value)\n"
            ),
            tmp_path,
        )
        assert exc.fields["python_type"].value == ""
