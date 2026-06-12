"""OutputContract materialization from OutputContractSpec.

``OutputContract`` is a materialized contract for a single agent call site:
it combines the static spec (codec name, target type, strict_json flag) with
the live codec implementation to produce the format instructions that will be
passed to agents and the parsing parameters for the codec.

M2: contracts for JSON-typed targets carry a ``json_schema`` and
``format_instructions`` built by ``JsonCodec.make_contract``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agm.agl.runtime.codec import OutputCodec
from agm.agl.typecheck.env import OutputContractSpec, TypeEnvironment
from agm.agl.typecheck.types import Type


@dataclass(slots=True)
class OutputContract:
    """Materialized per-call output contract.

    ``target_type``         — the resolved semantic type for this call.
    ``codec``               — the live codec implementation.
    ``strict_json``         — effective strict-JSON flag (None if not JSON).
    ``format_instructions`` — human-readable instructions for agents (empty
                              for the text codec; populated by JsonCodec).
    ``json_schema``         — JSON Schema dict for API-backed agents (None for
                              the text codec; populated by JsonCodec).
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

    Looks up the codec by name in *codecs*, calls ``codec.make_contract`` to
    derive format instructions and JSON Schema, then overlays the per-call
    ``strict_json`` flag from the spec.

    Raises ``ValueError`` if the codec is not found (host-configuration error,
    not an AgL exception).
    """
    codec = codecs.get(spec.codec_name)
    if codec is None:
        raise ValueError(
            f"No codec registered for codec_name={spec.codec_name!r}. "
            "This is a host-configuration error."
        )
    # Delegate format_instructions and json_schema derivation to the codec.
    env = TypeEnvironment()
    base = codec.make_contract(spec.target_type, env)
    # Overlay the per-call strict_json from the spec (the codec's make_contract
    # sets a default; the static spec overrides it for the specific call site).
    return OutputContract(
        target_type=base.target_type,
        codec=base.codec,
        strict_json=spec.strict_json,
        format_instructions=base.format_instructions,
        json_schema=base.json_schema,
    )
