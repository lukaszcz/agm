"""OutputContract materialization from OutputContractSpec.

``OutputContract`` is a materialized contract for a single agent call site:
it combines the static spec (codec name, target type, strict_json flag) with
the live codec implementation to produce the format instructions that will be
passed to agents and the parsing parameters for the codec.

M2: contracts for JSON-typed targets carry a ``json_schema`` and
``format_instructions`` (the latter embedding the schema) built by
``JsonCodec.make_contract``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from agm.agl.ir.contracts import (
    ContractRequest,
    DecodeSchema,
    DictDecode,
    EnumDecode,
    ListDecode,
    RecordDecode,
    ScalarDecode,
    ScalarKind,
)
from agm.agl.runtime.codec import OutputCodec
from agm.agl.semantics.types import (
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    UnitType,
)
from agm.agl.typecheck.env import OutputContractSpec


@dataclass(frozen=True, slots=True)
class TypelessOutputContract:
    """Agent-facing contract materialized from typeless execution IR metadata."""

    target_type: str
    codec_name: str
    strict_json: bool | None
    format_instructions: str
    json_schema: object
    structured_exec: bool = False


def _type_from_decode(schema: DecodeSchema) -> Type:
    match schema:
        case ScalarDecode(kind=ScalarKind.TEXT):
            return TextType()
        case ScalarDecode(kind=ScalarKind.INT):
            return IntType()
        case ScalarDecode(kind=ScalarKind.DECIMAL):
            return DecimalType()
        case ScalarDecode(kind=ScalarKind.BOOL):
            return BoolType()
        case ScalarDecode(kind=ScalarKind.JSON):
            return JsonType()
        case ListDecode(elem=elem):
            return ListType(_type_from_decode(elem))
        case DictDecode(value=value):
            return DictType(_type_from_decode(value))
        case RecordDecode(nominal=nominal, display_name=name, fields=fields):
            return RecordType(
                name=name,
                fields={field: _type_from_decode(decoder) for field, decoder in fields},
                module_id=nominal.module_id,
            )
        case EnumDecode(nominal=nominal, display_name=name, variants=variants):
            return EnumType(
                name=name,
                variants={
                    variant.name: {
                        field: _type_from_decode(decoder)
                        for field, decoder in variant.fields
                    }
                    for variant in variants
                },
                module_id=nominal.module_id,
            )
    raise AssertionError(f"Unknown decode schema: {schema!r}")


def materialize_ir_contract(
    request: ContractRequest, codecs: Mapping[str, OutputCodec]
) -> OutputContract | None:
    """Materialize host codec behavior from a typeless IR contract descriptor."""
    if request.is_unit:
        return None
    codec = codecs.get(request.codec_name)
    if codec is None:
        raise ValueError(
            f"No codec registered for codec_name={request.codec_name!r}. "
            "This is a host-configuration error."
        )
    target_type = (
        TextType()
        if request.decode is None and request.target_type_label == "text"
        else UnitType() if request.decode is None else _type_from_decode(request.decode)
    )
    base = codec.make_contract(target_type)
    schema: object = (
        base.json_schema
        if request.json_schema is None
        else cast(object, json.loads(request.json_schema))
    )
    return OutputContract(
        target_type=target_type,
        codec=codec,
        strict_json=request.strict_json,
        format_instructions=base.format_instructions,
        json_schema=schema,
        structured_exec=request.structured_exec,
    )


@dataclass(slots=True)
class OutputContract:
    """Materialized per-call output contract.

    ``target_type``         — the resolved semantic type for this call.
    ``codec``               — the live codec implementation.
    ``strict_json``         — effective strict-JSON flag (None if not JSON).
    ``format_instructions`` — text instructions appended to the agent message
                              (empty for the text codec and for structured
                              exec; for JSON targets a behavioural preamble
                              plus the pretty-printed JSON Schema).
    ``json_schema``         — JSON Schema dict for API-backed agents (None for
                              the text codec; populated by JsonCodec).
    """

    target_type: Type
    codec: OutputCodec
    strict_json: bool | None
    format_instructions: str
    json_schema: object  # dict[str, object] | None, but object keeps mypy happy
    structured_exec: bool = False


def materialize_contract(
    spec: OutputContractSpec,
    codecs: Mapping[str, OutputCodec],
) -> OutputContract:
    """Build an ``OutputContract`` from a static ``OutputContractSpec``.

    Looks up the codec by name in *codecs*, calls ``codec.make_contract`` to
    derive format instructions and JSON Schema, then overlays the per-call
    ``strict_json`` flag from the spec.

    For ``structured_exec`` specs, returns a passthrough text contract without
    consulting the codec table (the codec field is unused for structured exec).

    Raises ``ValueError`` if the codec is not found (host-configuration error,
    not an AgL exception).
    """
    if spec.structured_exec:
        from agm.agl.runtime.codec import TextCodec

        return OutputContract(
            target_type=spec.target_type,
            codec=TextCodec(),
            strict_json=None,
            format_instructions="",
            json_schema=None,
            structured_exec=True,
        )
    codec = codecs.get(spec.codec_name)
    if codec is None:
        raise ValueError(
            f"No codec registered for codec_name={spec.codec_name!r}. "
            "This is a host-configuration error."
        )
    # Delegate format_instructions and json_schema derivation to the codec.
    # (CARRY-IN 2: TypeEnvironment() was constructed here but never used.)
    base = codec.make_contract(spec.target_type)
    # Overlay the per-call strict_json from the spec (the codec's make_contract
    # sets a default; the static spec overrides it for the specific call site).
    return OutputContract(
        target_type=base.target_type,
        codec=base.codec,
        strict_json=spec.strict_json,
        format_instructions=base.format_instructions,
        json_schema=base.json_schema,
    )
