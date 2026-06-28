"""Shared Value → JSON serialization for the AgL runtime.

This is the single source of truth for converting AgL ``Value`` objects to
JSON-shaped Python objects and for emitting them as JSON text.

Design constraint (design §5.1: *no binary floating-point anywhere*): a
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

from agm.agl.semantics.values import (
    AgentValue,
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
    ListValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)


def value_to_json_obj(value: Value) -> object:
    """Convert a ``Value`` to a JSON-shaped Python object.

    The result is drawn from the closed JSON-shape domain
    ``dict | list | str | int | Decimal | bool | None``.  ``DecimalValue`` is
    preserved as :class:`decimal.Decimal` (never converted to ``float``).
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
    if isinstance(value, ListValue):
        return [value_to_json_obj(e) for e in value.elements]
    if isinstance(value, DictValue):
        return {k: value_to_json_obj(v) for k, v in value.entries.items()}
    if isinstance(value, RecordValue):
        return {k: value_to_json_obj(v) for k, v in value.fields.items()}
    if isinstance(value, EnumValue):
        result: dict[str, object] = {"$case": value.variant}
        result.update({k: value_to_json_obj(v) for k, v in value.fields.items()})
        return result
    if isinstance(value, ExceptionValue):
        return {k: value_to_json_obj(v) for k, v in value.fields.items()}
    if isinstance(value, UnitValue):
        raise TypeError("UnitValue has no JSON representation")
    if isinstance(value, AgentValue):
        raise TypeError("AgentValue has no JSON representation")
    if isinstance(value, ConstructorValue):
        raise TypeError("ConstructorValue has no JSON representation")
    if isinstance(value, IrClosureValue):
        raise TypeError("IrClosureValue has no JSON representation")
    if isinstance(value, IteratorValue):
        raise TypeError("IteratorValue has no JSON representation")
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
        return _emit_list(obj, indent=indent, level=level)
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


def _emit_list(obj: list[object], *, indent: int | None, level: int) -> str:
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
            f"{k}: {_emit(v, indent=None, level=level)}"
            for k, v in zip(keys, values, strict=True)
        ]
        return "{" + ", ".join(items) + "}"
    pad = " " * (indent * (level + 1))
    close_pad = " " * (indent * level)
    items = [
        f"{pad}{k}: {_emit(v, indent=indent, level=level + 1)}"
        for k, v in zip(keys, values, strict=True)
    ]
    return "{\n" + ",\n".join(items) + "\n" + close_pad + "}"
