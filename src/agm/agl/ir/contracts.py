"""Runtime descriptors for the AgL typeless execution IR.

This module holds closed tagged-data descriptors that the lowerer compiles
while checker types are still available, and that the evaluator executes
WITHOUT any checker ``Type``.  It defines the cast/conversion descriptors
(``ConversionRecipe`` and the ``DecodeSchema`` union).

Dependency rule: ``agm.agl.ir`` imports
only stdlib + ``ir.ids`` / ``ir.operations`` + ``modules.ids`` + ``zones``
(the parameter-zone enum, a dependency-free shared leaf).  It imports nothing
from ``typecheck``, ``eval``, or ``runtime``, and stores no callables — every
descriptor is immutable, runtime-neutral data.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from agm.agl.ir.ids import NominalId
from agm.agl.zones import ParamZone

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
    "EncodeDefinition",
    "EncodePlan",
    "EncodeSchema",
    "EnumEncode",
    "ExceptionEncode",
    "ExceptionFieldEncode",
    "EnumDecode",
    "FieldDecode",
    "FieldEncode",
    "ParamDecoder",
    "RecordDecode",
    "RecordEncode",
    "RefDecode",
    "RefEncode",
    "ScalarDecode",
    "ScalarEncode",
    "ScalarKind",
    "TypeParameterEncode",
    "VariantDecode",
    "VariantEncode",
    "forwarded_encode_key",
    "resolve_schema_ref",
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
class FieldDecode:
    """One record field's declared name, JSON key, decoder, zone, and value-syntax alias.

    ``zone`` is the field's parameter zone (positional-only/standard/named-
    only), for a value-syntax reader binding constructor arguments with the
    shared zone binder. ``alias`` is the field's ``@name`` spelling when it
    differs from ``name`` (an additional legal value-syntax spelling), or
    ``None`` when the field carries no alias.
    """

    name: str
    json_name: str
    schema: "DecodeSchema"
    zone: ParamZone
    alias: str | None


@dataclass(frozen=True, slots=True)
class RecordDecode:
    """Decode a JSON object into a record with the given fields (in order).

    ``name`` is the record's terminal declared name (the handle's own
    ``name``, unqualified — unlike ``display_name``). ``alias`` is the
    record's own ``@name`` spelling when it differs from ``name``, or
    ``None`` when the record carries no alias.
    """

    nominal: NominalId
    display_name: str
    fields: tuple[FieldDecode, ...]
    name: str
    alias: str | None


@dataclass(frozen=True, slots=True)
class VariantDecode:
    """One enum member's terminal name, JSON ``$case`` tag, identity, display name, and fields.

    ``alias`` is the member's own ``@name`` spelling when it differs from
    ``name``, or ``None`` when the member carries no alias.
    """

    name: str
    json_name: str
    nominal: NominalId
    display_name: str
    fields: tuple[FieldDecode, ...]
    alias: str | None


@dataclass(frozen=True, slots=True)
class EnumDecode:
    """Decode a JSON object (with a ``$case`` discriminator) into an enum.

    ``name`` is the enum's terminal declared name (unqualified, unlike
    ``display_name``). ``host_agent`` is ``True`` exactly for the standard
    library's ``Agent`` enum (see
    ``semantics.types.is_standard_agent_enum``); an enum has no ``@name``
    alias of its own — only its members and their fields do.
    """

    nominal: NominalId
    display_name: str
    variants: tuple[VariantDecode, ...]
    name: str
    host_agent: bool


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
class FieldEncode:
    """One record or exception field's declared name, JSON key, and encoder."""

    name: str
    json_name: str
    schema: "EncodeSchema"


@dataclass(frozen=True, slots=True)
class RecordEncode:
    """Encode a record as its statically ordered field object."""

    nominal: NominalId
    fields: tuple[FieldEncode, ...]


@dataclass(frozen=True, slots=True)
class ExceptionEncode:
    """Encode an exception as its statically ordered field object."""

    nominal: NominalId
    fields: tuple[FieldEncode, ...]


@dataclass(frozen=True, slots=True)
class VariantEncode:
    """One enum member's terminal name, JSON ``$case`` tag, identity, and ordered field encoders."""

    name: str
    json_name: str
    nominal: NominalId
    fields: tuple[FieldEncode, ...]


@dataclass(frozen=True, slots=True)
class EnumEncode:
    """Encode an enum slot with its member-selected ``$case`` tag."""

    nominal: NominalId
    variants: tuple[VariantEncode, ...]


@dataclass(frozen=True, slots=True)
class RefEncode:
    """Reference to a plan definition, applied to its type-parameter arguments.

    A definition with no parameters is referenced with no *arguments*, and its
    ``key`` is the ``$defs`` key its instantiation also has in the decode plan
    and the JSON Schema (same recursion plan, see ``type_schema._plan_schema``)
    — the shape every finite plan emits.  A generic-template definition instead
    takes parameters, and each reference supplies one encode schema per
    parameter; that is how a growing polymorphic-recursive source, whose
    concrete instantiations never close, is still described by finitely many
    bodies.
    """

    key: str
    arguments: "tuple[EncodeSchema, ...]" = ()


@dataclass(frozen=True, slots=True)
class TypeParameterEncode:
    """The encoding shape supplied for one enclosing definition parameter."""

    index: int


EncodeSchema = (
    ScalarEncode
    | ArrayEncode
    | DictEncode
    | RecordEncode
    | ExceptionEncode
    | EnumEncode
    | RefEncode
    | TypeParameterEncode
)


@dataclass(frozen=True, slots=True)
class EncodeDefinition:
    """One reusable encode body, referenced by ``key`` from anywhere in its plan.

    ``parameter_count`` is zero for a concrete instantiation's body and
    positive for a generic template, whose ``body`` reaches its parameters
    through :class:`TypeParameterEncode`.
    """

    key: str
    parameter_count: int
    body: EncodeSchema


@dataclass(frozen=True, slots=True)
class EncodePlan:
    """An encode schema paired with the definitions its references resolve against."""

    root: EncodeSchema
    definitions: "tuple[EncodeDefinition, ...]" = ()


def forwarded_encode_key(definition: "EncodeDefinition") -> str | None:
    """Onward key when a definition's body is a bare reference, else ``None``.

    Shared by the runtime encoder and the IR validator so both agree on where a
    reference chain terminates.  A reference carrying arguments is a template
    application rather than a forwarding link, so it ends the chain.
    """
    body = definition.body
    if isinstance(body, RefEncode) and not body.arguments:
        return body.key
    return None


@dataclass(frozen=True, slots=True)
class ExceptionFieldEncode:
    """Static reporting provenance for one exception field, JSON-keyed.

    ``plan`` is ``None`` for a field with no JSON form (``unit``, ``agent``,
    function, ...); ``json_name`` is always its effective JSON name, so every
    field — JSON-convertible or not — is covered and uniquely keyed.
    """

    field_name: str
    json_name: str
    plan: EncodePlan | None


@dataclass(frozen=True, slots=True)
class ParamDecoder:
    """Typeless decoder for one host-supplied entry parameter.

    Whether a raw value is taken verbatim (``text``) or read through the
    Agent host-text conventions is derived from ``decode`` itself at decode
    time (see ``runtime.value_decode.host_text_to_json``), not stored here.
    """

    target_type_label: str
    json_schema: str
    decode: DecodeSchema
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()


# ---------------------------------------------------------------------------
# Conversion recipe — the executable descriptor carried by ``IrConvert``.
# ---------------------------------------------------------------------------


class ConversionStrategy(enum.Enum):
    """How a cast realizes its conversion (resolved at lowering)."""

    NOOP = "noop"  # identity / already-assignable (return value unchanged)
    WIDEN_INT_TO_DECIMAL = "widen_int_to_decimal"
    RENDER_TO_TEXT = "render_to_text"  # total
    TO_JSON = "to_json"  # total, encode plan
    NARROW_DECIMAL_TO_INT = "narrow_decimal_to_int"  # fallible
    PARSE_TEXT_THEN_DECODE = "parse_text_then_decode"  # fallible
    DECODE_JSON = "decode_json"  # fallible


class ConversionFailureMode(enum.Enum):
    """What a failed fallible conversion does at runtime."""

    RAISE_CAST_ERROR = "raise_cast_error"  # `as`
    RETURN_BOOL = "return_bool"  # `as?`


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
    encode walk and its ``encode_definitions``, whose parameters a source with
    growing polymorphic recursion binds at each reference. All unrelated
    fields are ``None``/empty for each strategy.
    """

    strategy: ConversionStrategy
    source_label: str
    target_label: str
    json_schema: str | None = None
    decode: DecodeSchema | None = None
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()
    encode: EncodeSchema | None = None
    encode_definitions: "tuple[EncodeDefinition, ...]" = ()


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


_SchemaT = TypeVar("_SchemaT")


def resolve_schema_ref(
    key: str,
    defs: Mapping[str, _SchemaT],
    forwarded_key: Callable[[_SchemaT], str | None],
    *,
    subject: str,
) -> _SchemaT:
    """Follow ``$defs`` references to the first non-reference body.

    *forwarded_key* returns the onward key of a reference node, or ``None``
    once the walk reaches a body. Encode and decode plans share the same
    ``$defs`` keying, so they share this walk.
    """
    seen: set[str] = set()
    current = key
    while True:
        if current in seen:
            raise AssertionError(f"{subject}: $defs reference cycle at key {current!r}")
        seen.add(current)
        resolved = defs.get(current)
        if resolved is None:
            raise AssertionError(f"{subject}: unknown $defs key {current!r}")
        onward = forwarded_key(resolved)
        if onward is None:
            return resolved
        current = onward
