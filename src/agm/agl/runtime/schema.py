"""JSON Schema derivation from AgL semantic types.

:func:`derive_schema` produces a JSON Schema ``dict[str, object]`` from a
semantic :class:`~agm.agl.typecheck.types.Type`.  The derived schema is used:

1. As ``OutputContract.json_schema`` so API-backed agents can request native
   structured output.
2. For schema validation via the ``jsonschema`` library inside
   :class:`~agm.agl.runtime.codec.JsonCodec`.

Derivation rules (design §7.3 / §7.4):
- ``text``    → ``{"type": "string"}``
- ``int``     → ``{"type": "integer"}``
- ``decimal`` → ``{"type": "number"}``
- ``bool``    → ``{"type": "boolean"}``
- ``json``    → ``{}``  (permissive — accepts any JSON value)
- ``list[T]`` → ``{"type": "array", "items": <schema for T>}``
- ``dict[text, V]`` → ``{"type": "object", "additionalProperties": <schema for V>}``
- ``record``  → object schema with ``additionalProperties: false``, ``required``,
                and per-field ``properties``.
- ``enum``    → ``{"oneOf": [...]}`` — one variant schema per variant, each an
                object with a ``"$case"`` const property and any payload fields.
"""

from __future__ import annotations

from typing import assert_never

from agm.agl.typecheck.types import (
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
)


def derive_schema(typ: Type) -> dict[str, object]:
    """Derive a JSON Schema from a semantic AgL *typ*.

    The returned dictionary is a valid JSON Schema object.  ``Decimal`` and
    ``int`` values round-trip correctly through JSON Schema validation (both
    are acceptable for ``"type": "number"``; ``"type": "integer"`` accepts
    only whole numbers).

    :raises TypeError: if *typ* is an ``ExceptionType`` (exceptions are not
        wire-serialised and have no JSON Schema).
    """
    if isinstance(typ, TextType):
        return {"type": "string"}
    if isinstance(typ, IntType):
        return {"type": "integer"}
    if isinstance(typ, DecimalType):
        return {"type": "number"}
    if isinstance(typ, BoolType):
        return {"type": "boolean"}
    if isinstance(typ, JsonType):
        # Permissive: accepts any JSON value.
        return {}
    if isinstance(typ, ListType):
        return {"type": "array", "items": derive_schema(typ.elem)}
    if isinstance(typ, DictType):
        return {"type": "object", "additionalProperties": derive_schema(typ.value)}
    if isinstance(typ, RecordType):
        return _record_schema(typ)
    if isinstance(typ, EnumType):
        return _enum_schema(typ)
    if isinstance(typ, ExceptionType):
        raise TypeError(
            f"ExceptionType {typ.name!r} has no JSON Schema; exceptions are not "
            "wire-serialised by the JSON codec."
        )
    assert_never(typ)  # pragma: no cover


def _record_schema(typ: RecordType) -> dict[str, object]:
    """Derive the JSON Schema for a record type (design §7.3)."""
    properties: dict[str, object] = {
        field_name: derive_schema(field_type)
        for field_name, field_type in typ.fields.items()
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(typ.fields.keys()),
        "properties": properties,
    }


def _enum_schema(typ: EnumType) -> dict[str, object]:
    """Derive the JSON Schema for an enum type (design §7.4).

    Each variant becomes a ``oneOf`` alternative.  The ``"$case"`` property
    is a ``const`` string that identifies the selected variant; payload fields
    follow alongside it.
    """
    variant_schemas: list[object] = []
    for variant_name, variant_fields in typ.variants.items():
        required: list[str] = ["$case"]
        properties: dict[str, object] = {
            "$case": {"const": variant_name},
        }
        for field_name, field_type in variant_fields.items():
            properties[field_name] = derive_schema(field_type)
            required.append(field_name)
        variant_schemas.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": properties,
            }
        )
    return {"oneOf": variant_schemas}
