"""Shared Value → JSON serialization for the AgL runtime.

This is the single source of truth for converting AgL ``Value`` objects to
JSON-shaped Python objects and for emitting them as JSON text.

Design constraint: a
``DecimalValue`` carries an exact :class:`decimal.Decimal`.  It is **never**
routed through :class:`float`.  Instead :func:`value_to_json_obj` preserves the
``Decimal`` in the JSON-shaped object, and :func:`dumps_exact` emits it as
unquoted numeric text using the ``Decimal``'s own exact string form.

Two entry points:

- :func:`value_to_json_obj` — ``Value`` → JSON-shaped object (``dict``/``list``/
  ``str``/``int``/``Decimal``/``bool``/``None``).  ``Decimal`` is preserved.
- :func:`dumps_exact` — render such an object as JSON text, emitting ``Decimal``
  as exact unquoted numeric text.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import assert_never

from agm.agl.semantics.cycles import enter_container
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
    IteratorValue,
    JsonValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)


class AglNonDataValue(Exception):
    """Sentinel: a value kind with no JSON representation reached the walk.

    Raised by :func:`value_to_json_obj` for ``unit``, ``agent``,
    ``constructor``, ``function``, and ``iterator`` values — none of these
    kinds has a JSON-shaped representation. ``kind`` is the user-facing kind
    name (not the Python class name), used to build the substituted marker
    text via :func:`non_data_value_marker`.

    No evaluator path can produce a catchable AgL exception from this
    sentinel: every reachable conversion to ``json`` is statically gated
    (``is_json_convertible``, see ``semantics/type_table.py``) to types that
    have a JSON representation, so a non-data value can only ever reach this
    walk through a caller that degrades it rather than propagating it — see
    ``runtime/trace.py``'s ``TraceSink.mutation`` and
    ``pipeline.py``'s ``exception_value_to_run_error``.
    """

    def __init__(self, kind: str) -> None:
        super().__init__(f"{kind} has no JSON representation")
        self.kind = kind


def non_data_value_marker(exc: AglNonDataValue) -> str:
    """Build the placeholder text substituted for a non-data value.

    Single shared constructor so every caller that converts an
    :class:`AglNonDataValue` sentinel produces byte-identical marker text —
    see ``runtime/trace.py`` and ``pipeline.py``.
    """
    return f"<{exc.kind} has no JSON representation>"


def value_to_json_obj(value: Value, active: "set[int] | None" = None) -> object:
    """Convert a ``Value`` to a JSON-shaped Python object.

    The result is drawn from the closed JSON-shape domain
    ``dict | list | str | int | Decimal | bool | None``.  ``DecimalValue`` is
    preserved as :class:`decimal.Decimal` (never converted to ``float``).

    Reference semantics makes a cyclic array/dict constructible; ``active``
    (an active-container-id set, allocated lazily on first use) detects a
    cycle and raises :class:`~agm.agl.semantics.cycles.AglCyclicValue` rather
    than recursing forever. Callers pass no *active* argument — it exists
    only to thread the walk's own recursive calls.

    A ``unit``, ``agent``, ``constructor``, ``function``, or ``iterator``
    value has no JSON representation at all; such a value raises
    :class:`AglNonDataValue` rather than a bare :class:`TypeError`, so a
    caller that can legitimately receive one (e.g. because it carries the
    field of an in-flight exception) can degrade it to a marker instead of
    crashing.
    """
    if isinstance(value, TextValue):
        return value.value
    if isinstance(value, IntValue):
        return value.value
    if isinstance(value, DecimalValue):
        return value.value
    if isinstance(value, BoolValue):
        return value.value
    if isinstance(value, JsonValue):
        return value.raw
    if isinstance(value, ArrayValue):
        active = enter_container(id(value), active)
        try:
            return [value_to_json_obj(e, active) for e in value.elements]
        finally:
            active.discard(id(value))
    if isinstance(value, DictValue):
        active = enter_container(id(value), active)
        try:
            return {k: value_to_json_obj(v, active) for k, v in value.entries.items()}
        finally:
            active.discard(id(value))
    if isinstance(value, RecordValue):
        return {k: value_to_json_obj(v, active) for k, v in value.fields.items()}
    if isinstance(value, EnumValue):
        result: dict[str, object] = {"$case": value.variant}
        result.update({k: value_to_json_obj(v, active) for k, v in value.fields.items()})
        return result
    if isinstance(value, ExceptionValue):
        return {k: value_to_json_obj(v, active) for k, v in value.fields.items()}
    if isinstance(value, UnitValue):
        raise AglNonDataValue("unit")
    if isinstance(value, AgentValue):
        raise AglNonDataValue("agent")
    if isinstance(value, ConstructorValue):
        raise AglNonDataValue("constructor")
    if isinstance(value, IrClosureValue):
        raise AglNonDataValue("function")
    if isinstance(value, IteratorValue):
        raise AglNonDataValue("iterator")
    assert_never(value)  # pragma: no cover


def dumps_exact(obj: object, *, indent: int | None = 2) -> str:
    """Serialize a JSON-shaped object to text, emitting decimals exactly.

    Operates over the closed JSON-shape domain produced by
    :func:`value_to_json_obj` (``dict``/``list``/``str``/``int``/``Decimal``/
    ``bool``/``None``).  A :class:`decimal.Decimal` is emitted as unquoted
    numeric text using its exact string form — it is never routed through
    :class:`float`.

    A small recursive emitter is used (rather than ``json.dumps``) because the
    stdlib encoder cannot serialize ``Decimal`` without a binary-float round
    trip.  ``str``/``bool``/``int``/``None`` leaves are still delegated to
    ``json.dumps`` so that escaping and formatting match the stdlib exactly.
    """
    return _emit(obj, indent=indent, level=0)


def _emit(obj: object, *, indent: int | None, level: int) -> str:
    # ``bool`` must be checked before ``int`` (bool is a subclass of int).
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, Decimal):
        return _decimal_text(obj)
    if isinstance(obj, (str, int)) or obj is None:
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        return _emit_array(obj, indent=indent, level=level)
    if isinstance(obj, dict):
        return _emit_dict(obj, indent=indent, level=level)
    # Defensive: anything outside the closed domain is rendered via json.dumps,
    # which raises a clear TypeError for genuinely unsupported objects.
    return json.dumps(obj, ensure_ascii=False)  # pragma: no cover


def _decimal_text(d: Decimal) -> str:
    """Exact unquoted numeric text for a ``Decimal`` (no float round trip)."""
    # ``str`` preserves the Decimal's exact value but can use scientific
    # notation (e.g. ``1E+2``); ``format(d, "f")`` forces plain fixed-point
    # while remaining exact.
    return format(d, "f")


def _emit_array(obj: list[object], *, indent: int | None, level: int) -> str:
    if not obj:
        return "[]"
    if indent is None:
        items = [_emit(e, indent=None, level=level) for e in obj]
        return "[" + ", ".join(items) + "]"
    pad = " " * (indent * (level + 1))
    close_pad = " " * (indent * level)
    items = [pad + _emit(e, indent=indent, level=level + 1) for e in obj]
    return "[\n" + ",\n".join(items) + "\n" + close_pad + "]"


def _emit_dict(obj: dict[object, object], *, indent: int | None, level: int) -> str:
    if not obj:
        return "{}"
    keys = [json.dumps(str(k), ensure_ascii=False) for k in obj]
    values = list(obj.values())
    if indent is None:
        items = [
            f"{k}: {_emit(v, indent=None, level=level)}" for k, v in zip(keys, values, strict=True)
        ]
        return "{" + ", ".join(items) + "}"
    pad = " " * (indent * (level + 1))
    close_pad = " " * (indent * level)
    items = [
        f"{pad}{k}: {_emit(v, indent=indent, level=level + 1)}"
        for k, v in zip(keys, values, strict=True)
    ]
    return "{\n" + ",\n".join(items) + "\n" + close_pad + "}"
