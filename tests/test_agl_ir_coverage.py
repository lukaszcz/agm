"""IR-owned helper edge cases retained after removal of the AST evaluator."""

from __future__ import annotations

import decimal

import pytest

from agm.agl.eval.arith import contains, div, order, value_eq
from agm.agl.ir.ids import NominalId
from agm.agl.ir.operations import CmpOp, ContainsKind
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.contract import materialize_contract
from agm.agl.runtime.params import convert_param_value
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.semantics.types import DecimalType, TextType, UnitType
from agm.agl.semantics.values import (
    ConstructorValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    IteratorValue,
    JsonValue,
    RecordValue,
    TextValue,
    _json_eq,
    _json_hash,
)
from agm.agl.typecheck.env import OutputContractSpec
from tests._agl_helpers import type_table_for


def test_arithmetic_mixed_and_defensive_edges() -> None:
    one = IntValue(1)
    decimal_one = DecimalValue(decimal.Decimal(1))
    assert value_eq(one, decimal_one)
    assert value_eq(decimal_one, one)
    assert order(CmpOp.LE, one, decimal_one)
    assert order(CmpOp.GE, decimal_one, one)
    assert not contains(ContainsKind.DICT, one, DictValue({"1": one}))
    with pytest.raises(AssertionError, match="cannot compare"):
        order(CmpOp.LT, TextValue("x"), one)
    with pytest.raises(AssertionError, match="expected numeric"):
        div(TextValue("x"), one)


def test_runtime_value_notimplemented_and_hash_edges() -> None:
    nominal = NominalId(ENTRY_ID, "Thing")
    values = [
        DictValue({"x": IntValue(1)}),
        RecordValue(nominal, "Thing", {"x": IntValue(1)}),
        EnumValue(nominal, "Thing", "Case", {"x": IntValue(1)}),
        ExceptionValue(nominal, "Thing", {"x": IntValue(1)}),
    ]
    for value in values:
        assert value.__eq__(object()) is NotImplemented
        assert isinstance(hash(value), int)


def test_json_value_helper_edges() -> None:
    assert not _json_eq([1], [1, 2])
    assert _json_hash(True) != _json_hash(1)
    assert isinstance(_json_hash([1, {"x": decimal.Decimal(2)}]), int)
    assert JsonValue(1).__eq__(object()) is NotImplemented


def test_constructor_render_and_serialization_edges() -> None:
    nominal = NominalId(ENTRY_ID, "Thing")
    record = ConstructorValue(nominal, "Thing", None)
    variant = ConstructorValue(nominal, "Thing", "Case")
    assert render_value(record) == "<constructor Thing>"
    assert render_value(variant) == "<constructor Thing::Case>"
    with pytest.raises(TypeError, match="ConstructorValue"):
        value_to_json_obj(record)
    with pytest.raises(TypeError, match="IteratorValue"):
        value_to_json_obj(IteratorValue(elements=[]))


def test_param_conversion_direct_success_edges() -> None:
    table = type_table_for()
    assert convert_param_value("text", "value", TextType(), table) == TextValue("value")
    assert convert_param_value("decimal", decimal.Decimal("1.5"), DecimalType(), table) == (
        DecimalValue(decimal.Decimal("1.5"))
    )
    with pytest.raises(ValueError, match="unsupported type"):
        convert_param_value("unit", None, UnitType(), table)


def test_structured_exec_contract_uses_passthrough_codec() -> None:
    contract = materialize_contract(
        OutputContractSpec(UnitType(), "unused", None, structured_exec=True), {}
    )
    assert contract.structured_exec
