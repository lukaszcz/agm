"""Runtime descriptors for the AgL typeless execution IR.

This module holds closed tagged-data descriptors that the lowerer compiles
while checker types are still available, and that the evaluator executes
WITHOUT any checker ``Type``.  It defines the cast/conversion descriptors
(``ConversionRecipe`` and the ``DecodeSchema`` union).

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
    "ArrayDecode",
    "ArrayEncode",
    "ContractPayload",
    "ContractRequest",
    "ConversionFailureMode",
    "ConversionRecipe",
    "ConversionStrategy",
    "DecodePlan",
    "DecodeSchema",
    "DictDecode",
    "DictEncode",
    "DynamicApplyEncode",
    "DynamicArrayEncode",
    "DynamicDictEncode",
    "DynamicEncodeDefinition",
    "DynamicEncodePlan",
    "DynamicEncodeSchema",
    "DynamicEnumEncode",
    "DynamicExceptionEncode",
    "DynamicRecordEncode",
    "DynamicTypeParameterEncode",
    "DynamicVariantEncode",
    "EncodePlan",
    "EncodeSchema",
    "EnumEncode",
    "ExceptionEncode",
    "ExceptionFieldEncode",
    "EnumDecode",
    "ParamDecoder",
    "RecordDecode",
    "RecordEncode",
    "RefDecode",
    "RefEncode",
    "ScalarDecode",
    "ScalarEncode",
    "ScalarKind",
    "VariantDecode",
    "VariantEncode",
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
class ArrayDecode:
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
    """One enum member's terminal tag, record identity, display name, and field decoders."""

    name: str
    nominal: NominalId
    display_name: str
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
DecodeSchema = ScalarDecode | ArrayDecode | DictDecode | RecordDecode | EnumDecode | RefDecode


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


# ---------------------------------------------------------------------------
# Encode schema — typeless mirror of static Value → JSON conversion.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScalarEncode:
    """Encode a scalar or opaque ``json`` value."""


@dataclass(frozen=True, slots=True)
class ArrayEncode:
    """Encode an array by recursively encoding each element."""

    elem: "EncodeSchema"


@dataclass(frozen=True, slots=True)
class DictEncode:
    """Encode a dict by recursively encoding each value."""

    value: "EncodeSchema"


@dataclass(frozen=True, slots=True)
class RecordEncode:
    """Encode a record as its statically ordered field object."""

    nominal: NominalId
    fields: "tuple[tuple[str, EncodeSchema], ...]"


@dataclass(frozen=True, slots=True)
class ExceptionEncode:
    """Encode an exception as its statically ordered field object."""

    nominal: NominalId
    fields: "tuple[tuple[str, EncodeSchema], ...]"


@dataclass(frozen=True, slots=True)
class VariantEncode:
    """One enum member's terminal tag, identity, and ordered field encoders."""

    name: str
    nominal: NominalId
    fields: "tuple[tuple[str, EncodeSchema], ...]"


@dataclass(frozen=True, slots=True)
class EnumEncode:
    """Encode an enum slot with its member-selected ``$case`` tag."""

    nominal: NominalId
    variants: tuple[VariantEncode, ...]


@dataclass(frozen=True, slots=True)
class RefEncode:
    """Reference to a recursive encode body in an enclosing defs table."""

    key: str


EncodeSchema = (
    ScalarEncode
    | ArrayEncode
    | DictEncode
    | RecordEncode
    | ExceptionEncode
    | EnumEncode
    | RefEncode
)


@dataclass(frozen=True, slots=True)
class EncodePlan:
    """An encode schema paired with recursive bodies keyed like decode-plan defs."""

    root: EncodeSchema
    defs: "tuple[tuple[str, EncodeSchema], ...]" = ()


# ---------------------------------------------------------------------------
# Dynamic encode schema — source-slot-directed encoding for growing generic
# recursion.  Unlike ``EncodePlan``, definitions bind generic type parameters
# and applications supply their runtime slot arguments.  This retains static
# record-versus-enum context without requiring an infinite instantiation plan.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DynamicArrayEncode:
    """Encode an array through a generic-template element slot."""

    elem: "DynamicEncodeSchema"


@dataclass(frozen=True, slots=True)
class DynamicDictEncode:
    """Encode a dict through a generic-template value slot."""

    value: "DynamicEncodeSchema"


@dataclass(frozen=True, slots=True)
class DynamicRecordEncode:
    """Encode a record through generic-template field slots."""

    nominal: NominalId
    fields: "tuple[tuple[str, DynamicEncodeSchema], ...]"


@dataclass(frozen=True, slots=True)
class DynamicExceptionEncode:
    """Encode an exception through generic-template field slots."""

    nominal: NominalId
    fields: "tuple[tuple[str, DynamicEncodeSchema], ...]"


@dataclass(frozen=True, slots=True)
class DynamicVariantEncode:
    """One enum member selected by a generic-template enum slot."""

    name: str
    nominal: NominalId
    fields: "tuple[tuple[str, DynamicEncodeSchema], ...]"


@dataclass(frozen=True, slots=True)
class DynamicEnumEncode:
    """Encode an enum slot while retaining its static member/tag relation."""

    nominal: NominalId
    variants: "tuple[DynamicVariantEncode, ...]"


@dataclass(frozen=True, slots=True)
class DynamicTypeParameterEncode:
    """The encoding shape supplied for one enclosing generic type parameter."""

    index: int


@dataclass(frozen=True, slots=True)
class DynamicApplyEncode:
    """Apply a nominal encoding definition to statically selected slot arguments."""

    nominal: NominalId
    arguments: "tuple[DynamicEncodeSchema, ...]"


DynamicEncodeSchema = (
    ScalarEncode
    | DynamicArrayEncode
    | DynamicDictEncode
    | DynamicRecordEncode
    | DynamicExceptionEncode
    | DynamicEnumEncode
    | DynamicTypeParameterEncode
    | DynamicApplyEncode
)


@dataclass(frozen=True, slots=True)
class DynamicEncodeDefinition:
    """One nominal template body used by :class:`DynamicEncodePlan`."""

    nominal: NominalId
    parameter_count: int
    body: DynamicEncodeSchema


@dataclass(frozen=True, slots=True)
class DynamicEncodePlan:
    """Finite generic-template encoding plan for a value-directed JSON cast."""

    root: DynamicEncodeSchema
    definitions: "tuple[DynamicEncodeDefinition, ...]"


@dataclass(frozen=True, slots=True)
class ExceptionFieldEncode:
    """Static encode provenance for one reportable exception field."""

    field_name: str
    plan: EncodePlan | DynamicEncodePlan


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
    TO_JSON = "to_json"  # total, static encode plan
    TO_JSON_VALUE_DIRECTED = "to_json_value_directed"  # total, finite-value fallback
    NARROW_DECIMAL_TO_INT = "narrow_decimal_to_int"  # fallible
    PARSE_TEXT_THEN_DECODE = "parse_text_then_decode"  # fallible
    DECODE_JSON = "decode_json"  # fallible


class ConversionFailureMode(enum.Enum):
    """What a failed fallible conversion does at runtime."""

    RAISE_CAST_ERROR = "raise_cast_error"  # `as`
    RETURN_OPTION = "return_option"  # `as?`


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
    non-recursive one, see ``DecodePlan``). ``TO_JSON`` instead carries the
    static encode walk and its recursive bodies. ``TO_JSON_VALUE_DIRECTED``
    is the fallback for a statically JSON-convertible source whose growing
    polymorphic recursion has no finite concrete-instantiation plan; its
    ``dynamic_encode`` carries a finite generic-template plan. The runtime
    follows statically selected slots (including record versus enum context)
    while instantiating template parameters from the value walk. All unrelated
    fields are ``None``/empty for each strategy.
    """

    strategy: ConversionStrategy
    source_label: str
    target_label: str
    json_schema: str | None = None
    decode: DecodeSchema | None = None
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()
    encode: EncodeSchema | None = None
    encode_defs: "tuple[tuple[str, EncodeSchema], ...]" = ()
    dynamic_encode: DynamicEncodePlan | None = None


# ---------------------------------------------------------------------------
# Contract request — per-call ask/exec descriptor
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
    """Typeless contract descriptor for an ask or exec call site.

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
                              target); the evaluator dispatches the call but skips
                              output parsing and returns ``UnitValue`` immediately.
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
