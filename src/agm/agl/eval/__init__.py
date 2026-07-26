"""Typeless IR evaluator and runtime value support."""

from __future__ import annotations

from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import (
    UNIT_VALUE,
    VOID_VALUE,
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
    UnitValue,
    Value,
)

__all__ = [
    "AgentValue",
    "AglRaise",
    "ArrayValue",
    "BoolValue",
    "DecimalValue",
    "DictValue",
    "EnumValue",
    "ExceptionValue",
    "IntValue",
    "JsonValue",
    "RecordValue",
    "TextValue",
    "UNIT_VALUE",
    "UnitValue",
    "Value",
    "VOID_VALUE",
]
