"""AgL evaluator (Component 6).

Public API
----------
- :class:`Interpreter` — tree-walking evaluator.
- :class:`Scope` — runtime scope frame.
- :class:`Binding` — a single scope binding.
- :class:`AglRaise` — Python carrier for a propagating AgL exception.
- Value types: ``TextValue``, ``IntValue``, ``DecimalValue``, ``BoolValue``,
  ``JsonValue``, ``ListValue``, ``DictValue``, ``RecordValue``, ``EnumValue``,
  ``ExceptionValue``, ``UnitValue``, ``AgentValue``, ``Closure``, ``Value``.
"""

from __future__ import annotations

from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.interpreter import Interpreter, execute_graph
from agm.agl.eval.scope import Binding, Scope
from agm.agl.eval.values import (
    UNIT_VALUE,
    AgentValue,
    BoolValue,
    Closure,
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
    "Binding",
    "BoolValue",
    "Closure",
    "DecimalValue",
    "DictValue",
    "EnumValue",
    "ExceptionValue",
    "execute_graph",
    "IntValue",
    "Interpreter",
    "JsonValue",
    "ListValue",
    "RecordValue",
    "Scope",
    "TextValue",
    "UNIT_VALUE",
    "UnitValue",
    "Value",
]
