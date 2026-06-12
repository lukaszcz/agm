"""AgL evaluator (Component 6).

Public API
----------
- :class:`Interpreter` — tree-walking evaluator.
- :class:`Scope` — runtime scope frame.
- :class:`Binding` — a single scope binding.
- :class:`AglRaise` — Python carrier for a propagating AgL exception.
- Value types: ``TextValue``, ``IntValue``, ``DecimalValue``, ``BoolValue``,
  ``JsonValue``, ``ListValue``, ``DictValue``, ``RecordValue``, ``EnumValue``,
  ``ExceptionValue``, ``Value``.
"""

from __future__ import annotations

from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.interpreter import Interpreter
from agm.agl.eval.scope import Binding, Scope
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
    Value,
)

__all__ = [
    "AglRaise",
    "Binding",
    "BoolValue",
    "DecimalValue",
    "DictValue",
    "EnumValue",
    "ExceptionValue",
    "IntValue",
    "Interpreter",
    "JsonValue",
    "ListValue",
    "RecordValue",
    "Scope",
    "TextValue",
    "Value",
]
