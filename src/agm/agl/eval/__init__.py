"""Typeless IR evaluator and runtime value support."""

from __future__ import annotations

from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import (
    UNIT_VALUE,
    AgentValue,
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
    UnitValue,
    Value,
)

__all__ = [
    "AgentValue",
    "AglRaise",
    "BoolValue",
    "DecimalValue",
    "DictValue",
    "EnumValue",
    "ExceptionValue",
    "IntValue",
    "JsonValue",
    "ListValue",
    "RecordValue",
    "TextValue",
    "UNIT_VALUE",
    "UnitValue",
    "Value",
]
