"""OutputContract materialization from OutputContractSpec.

``OutputContract`` is a materialized contract for a single agent call site:
it combines the static spec (codec name, target type, strict_json flag) with
the live codec implementation to produce the format instructions that will be
passed to agents and the parsing parameters for the codec.

In M1 with only the text codec, contracts are trivial; the seam is real
and ready for M2 (json schemas, format instructions for structured output).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agm.agl.runtime.codec import OutputCodec
from agm.agl.typecheck.env import OutputContractSpec
from agm.agl.typecheck.types import Type


@dataclass(slots=True)
class OutputContract:
    """Materialized per-call output contract.

    ``target_type``         — the resolved semantic type for this call.
    ``codec``               — the live codec implementation.
    ``strict_json``         — effective strict-JSON flag (None if not JSON).
    ``format_instructions`` — human-readable instructions for agents (empty
                              for the text codec; populated by JsonCodec in M2).
    ``json_schema``         — JSON Schema dict for API-backed agents (None in
                              M1; populated by JsonCodec in M2).
    """

    target_type: Type
    codec: OutputCodec
    strict_json: bool | None
    format_instructions: str
    json_schema: object  # dict[str, object] | None, but object keeps mypy happy


def materialize_contract(
    spec: OutputContractSpec,
    codecs: Mapping[str, OutputCodec],
) -> OutputContract:
    """Build an ``OutputContract`` from a static ``OutputContractSpec``.

    Looks up the codec by name in *codecs* and constructs the contract.
    Raises ``ValueError`` if the codec is not found (host-configuration error,
    not an AgL exception).
    """
    codec = codecs.get(spec.codec_name)
    if codec is None:
        raise ValueError(
            f"No codec registered for codec_name={spec.codec_name!r}. "
            "This is a host-configuration error."
        )
    return OutputContract(
        target_type=spec.target_type,
        codec=codec,
        strict_json=spec.strict_json,
        format_instructions="",  # M1: text codec needs no instructions
        json_schema=None,  # M1: populated by JsonCodec in M2
    )
