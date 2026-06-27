"""Compile-time JSON Schema and decode-schema derivation.

:func:`derive_schema` produces a JSON Schema ``dict[str, object]`` from a
semantic :class:`~agm.agl.semantics.types.Type`.  The derived schema is used:

1. Embedded in ``OutputContract.format_instructions`` (pretty-printed) so the
   agent receives the precise shape, and as ``OutputContract.json_schema`` so
   API-backed agents can request native structured output.
2. For schema validation via the ``jsonschema`` library inside
   :class:`~agm.agl.runtime.codec.JsonCodec`.

:func:`build_decode_schema` compiles a ``Type`` into a typeless
:class:`~agm.agl.ir.contracts.DecodeSchema` used by the IR evaluator to
reconstruct typed ``Value`` objects from validated JSON without holding
checker ``Type`` references.

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

import json
from typing import assert_never

from agm.agl.ir.contracts import (
    DecodeSchema,
    DictDecode,
    EnumDecode,
    ListDecode,
    ParamDecoder,
    RecordDecode,
    ScalarDecode,
    ScalarKind,
    VariantDecode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
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
    if isinstance(typ, UnitType):
        raise TypeError("UnitType has no JSON Schema; unit is not wire-serialised.")
    if isinstance(typ, AgentType):
        raise TypeError("AgentType has no JSON Schema; agent values are not wire-serialised.")
    if isinstance(typ, FunctionType):
        raise TypeError(
            "FunctionType has no JSON Schema; function values are not wire-serialised."
        )
    if isinstance(typ, BottomType):
        raise TypeError("BottomType has no JSON Schema; bottom type is not wire-serialised.")
    if isinstance(typ, TypeVarType):
        raise TypeError(
            "TypeVarType has no JSON Schema; type variables are not wire-serialised."
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


def build_decode_schema(typ: Type) -> DecodeSchema:
    """Compile a checker ``Type`` into a typeless ``DecodeSchema``.

    Mirrors the type recursion of ``runtime.convert.decode_value`` so the
    evaluator can reconstruct the typed value without the checker ``Type``.
    """
    if isinstance(typ, TextType):
        return ScalarDecode(ScalarKind.TEXT)
    if isinstance(typ, IntType):
        return ScalarDecode(ScalarKind.INT)
    if isinstance(typ, DecimalType):
        return ScalarDecode(ScalarKind.DECIMAL)
    if isinstance(typ, BoolType):
        return ScalarDecode(ScalarKind.BOOL)
    if isinstance(typ, JsonType):
        return ScalarDecode(ScalarKind.JSON)
    if isinstance(typ, ListType):
        return ListDecode(build_decode_schema(typ.elem))
    if isinstance(typ, DictType):
        return DictDecode(build_decode_schema(typ.value))
    if isinstance(typ, RecordType):
        return RecordDecode(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            fields=tuple(
                (fname, build_decode_schema(ftype)) for fname, ftype in typ.fields.items()
            ),
        )
    if isinstance(typ, EnumType):
        return EnumDecode(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            variants=tuple(
                VariantDecode(
                    name=vname,
                    fields=tuple(
                        (fname, build_decode_schema(ftype)) for fname, ftype in vfields.items()
                    ),
                )
                for vname, vfields in typ.variants.items()
            ),
        )
    # Non-data targets (unit/agent/function/exception/bottom/typevar) are not
    # decodable from JSON and are rejected by the checker before lowering.
    raise AssertionError(  # pragma: no cover
        f"build_decode_schema: undecodable type {typ!r}"
    )


def build_param_decoder(typ: Type) -> ParamDecoder:
    """Compile a checker ``Type`` into the typeless ``ParamDecoder`` used to
    decode one host-supplied entry parameter.

    Single source of the param-decoder shape, shared by the lowerer (which
    embeds it in each ``IrParam.external_decoder``) and the REPL/config path
    (:func:`agm.agl.runtime.params.convert_param_value`).  ``text`` params are
    taken verbatim; every other type round-trips through the canonical JSON
    boundary (``derive_schema`` for validation, ``build_decode_schema`` for the
    typeless decode walk).

    :raises TypeError: if *typ* has no wire schema (unit/agent/exception/…);
        :func:`derive_schema` rejects such types.
    """
    return ParamDecoder(
        target_type_label=repr(typ),
        json_schema=json.dumps(derive_schema(typ), sort_keys=True),
        decode=build_decode_schema(typ),
        text_verbatim=isinstance(typ, TextType),
    )


def build_format_instructions(schema: dict[str, object]) -> str:
    """Build agent instructions embedding the authoritative JSON schema."""
    if not schema:
        return (
            "Return exactly one JSON value.\n"
            "Do not include Markdown, prose, or code fences."
        )
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    return (
        "Return exactly one JSON value conforming to the following JSON Schema.\n"
        "Do not include Markdown, prose, or code fences.\n"
        "\n"
        f"```json\n{schema_text}\n```"
    )
