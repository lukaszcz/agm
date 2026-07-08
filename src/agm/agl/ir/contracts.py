"""Runtime descriptors for the AgL typeless execution IR.

This module holds closed tagged-data descriptors that the lowerer compiles
while checker types are still available, and that the evaluator executes
WITHOUT any checker ``Type``.  Currently it defines the cast/conversion
descriptors (``ConversionRecipe`` and the ``DecodeSchema`` union); host
contract / param-decoder descriptors arrive in future work.

Dependency rule: ``agm.agl.ir`` imports
only stdlib + ``ir.ids`` / ``ir.operations`` + ``modules.ids``.  It imports
nothing from ``typecheck``, ``eval``, or ``runtime``, and stores no callables —
every descriptor is immutable, runtime-neutral data.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from agm.agl.ir.ids import NominalId

__all__ = [
    "ContractPayload",
    "ContractRequest",
    "ConversionFailureMode",
    "ConversionRecipe",
    "ConversionStrategy",
    "DecodePlan",
    "DecodeSchema",
    "DictDecode",
    "EnumDecode",
    "ListDecode",
    "ParamDecoder",
    "RecordDecode",
    "RefDecode",
    "ScalarDecode",
    "ScalarKind",
    "VariantDecode",
]


# ---------------------------------------------------------------------------
# Decode schema — typeless mirror of the checker-type recursion that
# ``runtime.convert.decode_value`` performs.  Built at lowering from the cast
# target type; walked at evaluation to construct the typed value.
# ---------------------------------------------------------------------------


class ScalarKind(enum.Enum):
    """Leaf decode targets (scalars + opaque json passthrough)."""

    TEXT = "text"
    INT = "int"
    DECIMAL = "decimal"
    BOOL = "bool"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class ScalarDecode:
    """Decode a JSON scalar (or opaque json) into the matching leaf value."""

    kind: ScalarKind


@dataclass(frozen=True, slots=True)
class ListDecode:
    """Decode a JSON array, recursively decoding each element."""

    elem: "DecodeSchema"


@dataclass(frozen=True, slots=True)
class DictDecode:
    """Decode a JSON object as a homogeneous dict, recursing on each value."""

    value: "DecodeSchema"


@dataclass(frozen=True, slots=True)
class RecordDecode:
    """Decode a JSON object into a record with the given fields (in order)."""

    nominal: NominalId
    display_name: str
    fields: "tuple[tuple[str, DecodeSchema], ...]"


@dataclass(frozen=True, slots=True)
class VariantDecode:
    """One enum variant: its name and ordered field decoders."""

    name: str
    fields: "tuple[tuple[str, DecodeSchema], ...]"


@dataclass(frozen=True, slots=True)
class EnumDecode:
    """Decode a JSON object (with a ``$case`` discriminator) into an enum."""

    nominal: NominalId
    display_name: str
    variants: tuple[VariantDecode, ...]


@dataclass(frozen=True, slots=True)
class RefDecode:
    """Reference to a recursive instantiation's entry in an enclosing ``defs`` table.

    Mirrors a ``{"$ref": "#/$defs/<key>"}`` node in the JSON Schema derived by
    ``derive_schema`` (``type_schema.py``): both are emitted from the SAME
    recursion plan, so ``key`` matches the JSON Schema's own ``$defs`` key for
    the same instantiation one-to-one.  Resolved against the ``defs`` table
    carried alongside the decode schema (see ``DecodePlan``) wherever the walk
    encounters one — the root itself, if the whole type is recursive, or any
    field/variant/element position reachable from it.
    """

    key: str


#: Closed union of decode-schema nodes.  Dispatch with a structural ``match``
#: whose final arm is ``assert_never``.
DecodeSchema = ScalarDecode | ListDecode | DictDecode | RecordDecode | EnumDecode | RefDecode


@dataclass(frozen=True, slots=True)
class DecodePlan:
    """A decode schema paired with its ``$defs`` table, as ``build_decode_schema`` returns it.

    ``root`` is the decode schema for the requested type itself (a
    ``RefDecode`` when the type's own root instantiation is recursive).
    ``defs`` holds one entry per recursive instantiation reachable from
    *root*, keyed identically to ``derive_schema``'s own ``$defs`` keys for
    the same type (same recursion plan, see ``type_schema._plan_schema``) — a
    tuple of ``(key, schema)`` pairs (not a ``dict``) so the plan stays
    hashable like every other IR descriptor.  Empty for a non-recursive type,
    the representation-identical default.

    This bundling is a convenience for callers that need to build both parts
    together; carriers that persist a decode schema (``ContractRequest``,
    ``ConversionRecipe``, ``ParamDecoder``) store ``decode``/``defs`` as two
    sibling fields rather than one ``DecodePlan`` field, so non-recursive
    carriers built directly (in tests or elsewhere) with a bare
    ``DecodeSchema`` and no ``defs`` keyword continue to work unchanged.
    """

    root: DecodeSchema
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()


@dataclass(frozen=True, slots=True)
class ParamDecoder:
    """Typeless decoder for one host-supplied entry parameter."""

    target_type_label: str
    json_schema: str
    decode: DecodeSchema
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()
    text_verbatim: bool = False


# ---------------------------------------------------------------------------
# Conversion recipe — the executable descriptor carried by ``IrConvert``.
# ---------------------------------------------------------------------------


class ConversionStrategy(enum.Enum):
    """How a cast realizes its conversion (resolved at lowering)."""

    NOOP = "noop"  # identity / already-assignable (return value unchanged)
    WIDEN_INT_TO_DECIMAL = "widen_int_to_decimal"
    RENDER_TO_TEXT = "render_to_text"  # total
    TO_JSON = "to_json"  # total
    NARROW_DECIMAL_TO_INT = "narrow_decimal_to_int"  # fallible
    PARSE_TEXT_THEN_DECODE = "parse_text_then_decode"  # fallible
    DECODE_JSON = "decode_json"  # fallible


class ConversionFailureMode(enum.Enum):
    """What a failed fallible conversion does at runtime."""

    RAISE_CAST_ERROR = "raise_cast_error"  # `as`
    RETURN_BOOL = "return_bool"  # `as?` (fallible)


@dataclass(frozen=True, slots=True)
class ConversionRecipe:
    """Closed tagged-data describing one cast conversion.

    ``source_label`` / ``target_label`` are the user-facing type names used in
    ``CastError`` (the legacy ``repr(Type)``).  For the decode strategies
    (``NARROW_DECIMAL_TO_INT`` / ``PARSE_TEXT_THEN_DECODE`` / ``DECODE_JSON``)
    ``json_schema`` carries the JSON Schema derived from the target type —
    serialized as a canonical JSON **string** so the recipe stays frozen and
    hashable (a bare ``dict`` would break ``__hash__``, the invariant every IR
    node maintains) — and ``decode`` carries the typeless decode walk; ``defs``
    carries the ``$defs`` table for a recursive target type (empty for a
    non-recursive one, see ``DecodePlan``); all three are ``None``/empty for
    the total strategies.
    """

    strategy: ConversionStrategy
    source_label: str
    target_label: str
    json_schema: str | None = None
    decode: DecodeSchema | None = None
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()


# ---------------------------------------------------------------------------
# Contract request — per-call ask/ask-request descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContractPayload:
    """Typeless materialized-codec payload embedded in a contract request.

    Hosts may materialize custom codecs while checker types are still available
    and pass only these immutable runtime fields into lowering.  The linked IR
    never stores the checker ``Type`` or ``TypeTable`` used to derive them.
    """

    json_schema: str | None
    decode: "DecodeSchema | None"
    format_instructions: str
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()


@dataclass(frozen=True, slots=True)
class ContractRequest:
    """Typeless contract descriptor for an ask/ask-request call site.

    Built at lowering while checker types are available; evaluated WITHOUT any
    checker ``Type``.  The evaluator parses agent output using only this descriptor.

    ``codec_name``          — ``"text"`` or ``"json"`` (from ``OutputContractSpec``).
    ``strict_json``         — per-call strict_json override; ``None`` → use the
                              evaluator-level default.
    ``json_schema``         — canonical JSON string of the derived schema
                              (``json.dumps(..., sort_keys=True)``); ``None`` for
                              the text codec.
    ``decode``              — typeless ``DecodeSchema`` walk for the target type;
                              ``None`` for the text codec.
    ``target_type_label``   — ``repr(target_type)`` stored for ``AgentParseError``
                              field text and failure-message formatting.
    ``target_type_kind``    — semantic kind string (``int``, ``record``, …) kept
                              as typeless compatibility metadata for legacy
                              custom-codec parse hooks.
    ``target_type``         — opaque checker type retained only for legacy custom
                              codecs whose ``parse`` hook still accepts a
                              positional target type. Runtime-neutral code must
                              not inspect it.
    ``structured_exec``     — ``True`` for structured exec; ``False`` for ``ask``.
    ``format_instructions`` — pre-computed format instructions string (empty for
                              text codec and unit-typed asks).
    ``is_unit``             — ``True`` when the target type is ``unit`` (unit
                              target); the evaluator dispatches the agent call but
                              skips output parsing and returns ``UnitValue``
                              immediately.  For ``ask-request`` the result is always
                              an ``AgentRequest`` record, never unit — ``is_unit``
                              is always ``False`` for ``ask-request`` call sites.
    ``defs``                — ``$defs`` table for a recursive target type (empty
                              for a non-recursive one, see ``DecodePlan``); ``()``
                              for the text codec.
    """

    codec_name: str
    strict_json: bool | None
    json_schema: str | None
    decode: "DecodeSchema | None"
    target_type_label: str
    structured_exec: bool
    format_instructions: str
    is_unit: bool = False
    target_type_kind: str = ""
    target_type: object | None = None
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()
