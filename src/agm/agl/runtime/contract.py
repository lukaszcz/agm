"""OutputContract materialization from OutputContractSpec.

``OutputContract`` is a materialized contract for a single agent call site:
it combines the codec name/target-type label with the live codec
implementation to produce the format instructions that will be passed to
agents and the parsing parameters (schema + decode) for the codec.

Two materialization entry points feed the same ``OutputContract`` shape:

- :func:`materialize_ir_contract` — the execution-time path.  It builds the
  contract entirely from a typeless IR ``ContractRequest`` (the lowerer's
  compiled ``json_schema``/``decode``/``format_instructions``): no checker
  ``Type`` is reconstructed, and no schema/decode is re-derived.
- :func:`materialize_contract` — the check-time / REPL contract-preview path.
  It runs while a real checker ``Type`` is still available
  (``OutputContractSpec.target_type``), so it may still call
  ``codec.make_contract(type)``.

Both produce ``OutputContract`` objects exposing the same runtime surface
(``target_type_label``, ``codec``, ``strict_json``, ``format_instructions``,
``json_schema``, ``decode``, ``structured_exec``).

``TypelessOutputContract`` is a separate, lighter carrier for the same
display fields (no live ``codec``) used where no host codec registry is in
scope at all (see ``eval/effects.py``'s ``ask``-display fallback).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from agm.agl.ir.contracts import ContractRequest, DecodeSchema
from agm.agl.runtime.codec import BUILTIN_CODEC_NAMES, OutputCodec
from agm.agl.semantics.types import TextType, Type, UnitType
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


@dataclass(slots=True)
class OutputContract:
    """Materialized per-call output contract.

    ``target_type_label``   — display label for the resolved target type
                              (``repr(Type)`` at check time / the IR
                              contract's own label at execution time); used
                              only for messages, never for type structure.
    ``codec``               — the live codec implementation.
    ``strict_json``         — effective strict-JSON flag (None if not JSON).
    ``format_instructions`` — text instructions appended to the agent message
                              (empty for the text codec and for structured
                              exec; for JSON targets a behavioural preamble
                              plus the pretty-printed JSON Schema).
    ``json_schema``         — JSON Schema dict for API-backed agents (None for
                              the text codec; populated by JsonCodec).
    ``decode``              — typeless ``DecodeSchema`` walk used to convert
                              validated JSON into a typed ``Value`` (None for
                              the text codec and for structured exec).
    ``structured_exec``     — True for the structured ``exec`` passthrough
                              contract (raw ``ExecResult``, no parsing).
    """

    target_type_label: str
    codec: OutputCodec
    strict_json: bool | None
    format_instructions: str
    json_schema: object  # dict[str, object] | None, but object keeps mypy happy
    decode: DecodeSchema | None = None
    structured_exec: bool = False


def materialize_ir_contract(
    request: ContractRequest, codecs: Mapping[str, OutputCodec]
) -> OutputContract | None:
    """Materialize host codec behavior from a typeless IR contract descriptor.

    For the built-in ``text``/``json`` codecs the result is built entirely
    from *request*'s own precomputed fields: ``json_schema`` is parsed back
    from its canonical JSON string, and ``format_instructions`` is reused
    verbatim from ``request.format_instructions`` (the lowerer already
    computed it from that same schema), so the result is byte-identical to
    what ``JsonCodec.make_contract``/``TextCodec.make_contract`` would have
    produced, without re-deriving anything from a checker ``Type``.

    Host-registered custom codecs (see ``PipelineDriver.register_codec``) are
    third-party ``OutputCodec`` implementations whose ``make_contract`` may
    run arbitrary logic against the real target type.  The lowerer only
    derives a decode walk for the built-in ``json`` codec (any other
    ``codec_name`` always has ``request.decode is None``), so the only target
    type information ever available for a custom codec is whether the call
    was declared ``text``.  A minimal placeholder ``Type`` (``text`` or
    ``unit``) is reconstructed for that one case, to keep a custom codec's own
    ``make_contract`` behavior working exactly as before.
    """
    if request.is_unit:
        return None
    codec = codecs.get(request.codec_name)
    if codec is None:
        raise ValueError(
            f"No codec registered for codec_name={request.codec_name!r}. "
            "This is a host-configuration error."
        )
    if request.codec_name not in BUILTIN_CODEC_NAMES:
        placeholder_type: Type = (
            TextType() if request.target_type_label == "text" else UnitType()
        )
        base = codec.make_contract(placeholder_type)
        return OutputContract(
            target_type_label=request.target_type_label,
            codec=codec,
            strict_json=request.strict_json,
            format_instructions=base.format_instructions,
            json_schema=base.json_schema,
            decode=request.decode,
            structured_exec=request.structured_exec,
        )
    schema: object = (
        None if request.json_schema is None else cast(object, json.loads(request.json_schema))
    )
    return OutputContract(
        target_type_label=request.target_type_label,
        codec=codec,
        strict_json=request.strict_json,
        format_instructions=request.format_instructions,
        json_schema=schema,
        decode=request.decode,
        structured_exec=request.structured_exec,
    )


def materialize_contract(
    spec: OutputContractSpec,
    codecs: Mapping[str, OutputCodec],
) -> OutputContract:
    """Build an ``OutputContract`` from a static ``OutputContractSpec``.

    Looks up the codec by name in *codecs*, calls ``codec.make_contract`` to
    derive format instructions, JSON Schema, and the decode walk (a real
    checker ``Type`` is in hand here — this runs at check time / REPL
    contract-preview time, before lowering erases it), then overlays the
    per-call ``strict_json`` flag from the spec.

    For ``structured_exec`` specs, returns a passthrough text contract without
    consulting the codec table (the codec field is unused for structured exec).

    Raises ``ValueError`` if the codec is not found (host-configuration error,
    not an AgL exception).
    """
    if spec.structured_exec:
        from agm.agl.runtime.codec import TextCodec

        return OutputContract(
            target_type_label=repr(spec.target_type),
            codec=TextCodec(),
            strict_json=None,
            format_instructions="",
            json_schema=None,
            decode=None,
            structured_exec=True,
        )
    codec = codecs.get(spec.codec_name)
    if codec is None:
        raise ValueError(
            f"No codec registered for codec_name={spec.codec_name!r}. "
            "This is a host-configuration error."
        )
    # Delegate format_instructions/json_schema/decode derivation to the codec.
    base = codec.make_contract(spec.target_type)
    # Overlay the per-call strict_json from the spec (the codec's make_contract
    # sets a default; the static spec overrides it for the specific call site).
    return OutputContract(
        target_type_label=base.target_type_label,
        codec=base.codec,
        strict_json=spec.strict_json,
        format_instructions=base.format_instructions,
        json_schema=base.json_schema,
        decode=base.decode,
    )
