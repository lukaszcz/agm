"""Shared match-error helpers for case-expression pattern matching.

This module provides ``_describe_value`` (maps a runtime ``Value`` to its
AgL type-name string) and ``make_match_error`` (builds a ``MatchError``
``ExceptionValue`` with ``scrutinee_type`` and ``scrutinee`` fields).

The IR evaluator delegates here so match diagnostics have one implementation.

Allowed imports: stdlib, ``agm.agl.eval.values``, ``agm.agl.eval.exceptions``,
``agm.agl.runtime.serialize``.  No syntax, scope, or typecheck imports.
"""

from __future__ import annotations

from agm.agl.eval.exceptions import make_builtin_exception
from agm.agl.eval.values import (
    AgentValue,
    BoolValue,
    ConstructorValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    IrClosureValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)
from agm.agl.runtime.serialize import value_to_json_obj

__all__ = ["_describe_value", "make_match_error"]


def _describe_value(value: Value) -> str:
    """Return the AgL type-name of *value* (design §8.1 ``scrutinee_type``).

    Mirrors ``interpreter._describe_value`` exactly — this is the single
    authoritative implementation.  Both evaluators call this function.
    """
    if isinstance(value, EnumValue):
        return value.display_name
    if isinstance(value, RecordValue):
        return value.display_name
    if isinstance(value, ExceptionValue):
        return value.display_name
    if isinstance(value, TextValue):
        return "text"
    if isinstance(value, IntValue):
        return "int"
    if isinstance(value, DecimalValue):
        return "decimal"
    if isinstance(value, BoolValue):
        return "bool"
    if isinstance(value, JsonValue):
        return "json"
    if isinstance(value, ListValue):
        return "list"
    if isinstance(value, DictValue):
        return "dict"
    if isinstance(value, UnitValue):
        return "unit"
    if isinstance(value, AgentValue):
        return "agent"
    if isinstance(value, ConstructorValue):
        # A first-class constructor's static type is a FunctionType.
        return "function"
    assert isinstance(value, IrClosureValue), (
        f"unexpected value kind: {type(value).__name__}"
    )
    return "function"


def make_match_error(subject: Value, *, trace_id: str = "") -> ExceptionValue:
    """Build a ``MatchError`` ``ExceptionValue`` for a non-matching *subject*.

    Fields:
    - ``message``: human-readable description including the scrutinee type.
    - ``trace_id``: caller-provided event id (minted per evaluator).
    - ``scrutinee_type``: ``TextValue`` of the AgL type name of *subject*.
    - ``scrutinee``: ``JsonValue`` of the JSON representation of *subject*.
    """
    scrutinee_type = _describe_value(subject)
    scrutinee_json = value_to_json_obj(subject)
    return make_builtin_exception(
        "MatchError",
        f"Non-exhaustive case: no pattern matched value of type {scrutinee_type!r}",
        trace_id=trace_id,
        scrutinee_type=TextValue(scrutinee_type),
        scrutinee=JsonValue(scrutinee_json),
    )
