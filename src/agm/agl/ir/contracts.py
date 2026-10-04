"""Runtime descriptors for the AgL typeless execution IR.

This module holds closed tagged-data descriptors that the lowerer compiles
while checker types are still available, and that the evaluator executes
WITHOUT any checker ``Type``.  It defines the cast/conversion descriptors
(``ConversionRecipe`` and the ``DecodeSchema`` union), and the ``TypeTree``
describing a type-directed extern's target.

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
from typing import ClassVar, Literal, TypeVar

from agm.agl.ir.ids import NominalId
from agm.agl.zones import ParamZone

__all__ = [
    "ArrayDecode",
    "ArrayEncode",
    "ContractPayload",
    "ContractRequest",
    "CustomContractRequest",
    "ConversionFailureMode",
    "ConversionRecipe",
    "ConversionStrategy",
    "DecodeConversionKind",
    "DecodeConversionRecipe",
    "DecodePlan",
    "DecodeSchema",
    "DictDecode",
    "DictEncode",
    "DictKeyForm",
    "EncodeDefinition",
    "EncodePlan",
    "EncodeSchema",
    "EnumEncode",
    "ExceptionEncode",
    "ExceptionFieldEncode",
    "EnumDecode",
    "FieldDecode",
    "FieldEncode",
    "JsonContractRequest",
    "ParamDecoder",
    "RecordDecode",
    "RecordEncode",
    "RefDecode",
    "RefEncode",
    "ScalarDecode",
    "ScalarEncode",
    "ScalarKind",
    "SimpleConversionKind",
    "SimpleConversionRecipe",
    "TargetContractRequest",
    "UnitContractRequest",
    "TextContractRequest",
    "ToJsonRecipe",
    "TypeNode",
    "TypeNodeField",
    "TypeNodeKind",
    "TypeNodeRef",
    "TypeParameterEncode",
    "TypeTree",
    "TypeTreeEntry",
    "VariantDecode",
    "VariantEncode",
    "dict_key_form",
    "forwarded_encode_key",
    "is_plain_enum",
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
    ``None`` when the field carries no alias. ``default_index`` is set exactly
    when the field's declaration carries a constant default, to the field's
    position in its declaring nominal's own
    ``NominalDescriptor.fields``/``field_defaults`` -- the key a decode-time
    default fill (``runtime.convert.decode_value``'s ``default_resolver``)
    uses to evaluate the real default expression. A missing JSON key or an
    omitted constructor argument is then legal rather than an error.
    ``None`` for a field with no default.
    """

    name: str
    json_name: str
    schema: "DecodeSchema"
    zone: ParamZone
    alias: str | None
    default_index: int | None = None


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
    """One enum member's terminal name, JSON tag, identity, display name, and fields.

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
    """Decode an enum: a member's JSON tag string when plain, else a ``$case``-tagged object.

    See :func:`is_plain_enum`. ``name`` is the enum's terminal declared name (unqualified, unlike
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
    ``derive_schema_and_decode`` (``type_schema.py``): both are emitted from the SAME
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
    """A decode schema paired with its ``$defs`` table, as ``derive_schema_and_decode`` returns it.

    ``root`` is the decode schema for the requested type itself (a
    ``RefDecode`` when the type's own root instantiation is recursive).
    ``defs`` holds one entry per recursive instantiation reachable from
    *root*, keyed identically to the JSON Schema's own ``$defs`` keys for
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

    kind: ScalarKind


@dataclass(frozen=True, slots=True)
class ArrayEncode:
    """Encode an array by recursively encoding each element."""

    elem: "EncodeSchema"


class DictKeyForm(enum.Enum):
    """Wire shape for a dict's key, decided by :func:`dict_key_form` off its own encode schema.

    ``OBJECT_TEXT``: ``text`` (and text aliases) key directly as a JSON object key.
    ``OBJECT_STRINGIFIED``: int/decimal/bool/all-nullary-enum keys stringified as a JSON
    object key. ``ENTRIES``: every other hashable key, as a ``{"key":..., "value":...}``
    array.
    """

    OBJECT_TEXT = "object_text"
    OBJECT_STRINGIFIED = "object_stringified"
    ENTRIES = "entries"


@dataclass(frozen=True, slots=True)
class DictEncode:
    """Encode a dict by recursively encoding each key and value.

    ``key_form`` is the key's ``DictKeyForm`` (see :func:`dict_key_form`, the
    ONE classifier used by plan building, schema derivation, and the runtime
    for a growing template's key parameters — decoding never consults it),
    filled once when the plan is built
    (``type_schema._emit_encode_body``/``_build_template_encode_plan``)
    — it is ``None`` ONLY when ``key`` is a growing template's own
    :class:`TypeParameterEncode`, whose concrete key type is not known until
    the schema is resolved (substituted or followed through a ``$defs``
    reference) at each call site; there ``runtime.serialize``'s
    ``_encode_dict`` applies the classifier itself, after resolving the key
    schema (``_resolve``), at encode time.
    """

    key_form: "DictKeyForm | None"
    key: "EncodeSchema"
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
    """Encode an exception-typed slot by its value's own runtime nominal.

    A leaf: a slot's runtime value may be a more-derived subtype than its
    static type and encodes with all of its own fields, so the field plans
    are selected at encode time from the program's ``exception_field_encodes``
    table by the value's ``nominal`` (see ``runtime.serialize.encode_value``).
    ``nominal`` is the slot's static exception type.
    """

    nominal: NominalId


@dataclass(frozen=True, slots=True)
class VariantEncode:
    """One enum member's terminal name, JSON tag, identity, and ordered field encoders."""

    name: str
    json_name: str
    nominal: NominalId
    fields: tuple[FieldEncode, ...]


@dataclass(frozen=True, slots=True)
class EnumEncode:
    """Encode an enum slot: the member's JSON tag string when plain, else a ``$case`` object.

    See :func:`is_plain_enum`.
    """

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


def dict_key_form(schema: EncodeSchema) -> DictKeyForm:
    """Classify a dict key's wire shape from its own, already-RESOLVED encode schema.

    *schema* must already be resolved past any ``TypeParameterEncode``/
    ``RefEncode`` indirection. The main site is plan-build time —
    ``type_schema``'s own schema-derivation walk and its encode-plan builders
    (``_emit_encode_body``/``_build_template_encode_plan``), which fill
    ``DictEncode.key_form`` once; ``runtime.serialize``'s ``_encode_dict``
    calls this only for a growing template's own key type-parameter, whose
    concrete shape is not known until encode time. This is the ONE
    classifier either site consults; see ``DictKeyForm``.
    """
    if isinstance(schema, ScalarEncode):
        if schema.kind is ScalarKind.TEXT:
            return DictKeyForm.OBJECT_TEXT
        if schema.kind is ScalarKind.JSON:
            return DictKeyForm.ENTRIES
        return DictKeyForm.OBJECT_STRINGIFIED
    if isinstance(schema, EnumEncode) and all(not variant.fields for variant in schema.variants):
        return DictKeyForm.OBJECT_STRINGIFIED
    return DictKeyForm.ENTRIES


def is_plain_enum(schema: "EnumDecode | EnumEncode") -> bool:
    """Whether every member of *schema* is fieldless.

    A plain enum crosses JSON as its member's tag string; any other enum as a
    ``$case``-tagged object.
    """
    return not any(variant.fields for variant in schema.variants)


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
    NARROW_DECIMAL_TO_INT = "narrow_decimal_to_int"  # fallible
    PARSE_TEXT_THEN_DECODE = "parse_text_then_decode"  # fallible
    DECODE_JSON = "decode_json"  # fallible


class ConversionFailureMode(enum.Enum):
    """What a failed fallible conversion does at runtime."""

    RAISE_CAST_ERROR = "raise_cast_error"  # `as`
    RAISE_VALUE_PARSE_ERROR = "raise_value_parse_error"  # `std/value::parse`
    RETURN_OPTION = "return_option"  # `as?`


#: Strategies that need no conversion payload beyond the source/target labels.
type SimpleConversionKind = Literal[
    ConversionStrategy.NOOP,
    ConversionStrategy.WIDEN_INT_TO_DECIMAL,
    ConversionStrategy.RENDER_TO_TEXT,
]

#: Fallible strategies that validate a JSON document against ``json_schema``
#: then walk ``decode`` to build the target value.
type DecodeConversionKind = Literal[
    ConversionStrategy.NARROW_DECIMAL_TO_INT,
    ConversionStrategy.PARSE_TEXT_THEN_DECODE,
    ConversionStrategy.DECODE_JSON,
]


@dataclass(frozen=True, slots=True)
class SimpleConversionRecipe:
    """A cast conversion with no payload beyond its strategy and labels.

    ``source_label`` / ``target_label`` are the user-facing type names used in
    ``CastError`` (the legacy ``repr(Type)``).
    """

    strategy: SimpleConversionKind
    source_label: str
    target_label: str


@dataclass(frozen=True, slots=True)
class ToJsonRecipe:
    """An ``as json`` cast conversion: an encode plan for the source type.

    ``encode_definitions`` carries the plan's ``$defs``, whose parameters a
    source with growing polymorphic recursion binds at each reference.
    """

    source_label: str
    target_label: str
    encode: EncodeSchema
    encode_definitions: "tuple[EncodeDefinition, ...]" = ()


@dataclass(frozen=True, slots=True)
class DecodeConversionRecipe:
    """A fallible cast conversion: JSON-Schema validate, then decode.

    ``json_schema`` carries the JSON Schema derived from the target type —
    serialized as a canonical JSON **string** so the recipe stays frozen and
    hashable (a bare ``dict`` would break ``__hash__``, the invariant every IR
    node maintains) — and ``decode`` carries the typeless decode walk; ``defs``
    carries the ``$defs`` table for a recursive target type (empty for a
    non-recursive one, see ``DecodePlan``).
    """

    strategy: DecodeConversionKind
    source_label: str
    target_label: str
    json_schema: str
    decode: DecodeSchema
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()


#: Closed tagged-data describing one cast conversion, one variant per strategy
#: family. All unrelated payload is structurally absent for each variant.
type ConversionRecipe = SimpleConversionRecipe | ToJsonRecipe | DecodeConversionRecipe


# ---------------------------------------------------------------------------
# Type tree — typeless description of a type-directed extern's target type.
# ---------------------------------------------------------------------------


class TypeNodeKind(enum.Enum):
    """Shape of one type-tree node."""

    TEXT = "text"
    INT = "int"
    DECIMAL = "decimal"
    BOOL = "bool"
    JSON = "json"
    ARRAY = "array"
    DICT = "dict"
    RECORD = "record"
    ENUM = "enum"
    MEMBER = "member"


@dataclass(frozen=True, slots=True)
class TypeNodeRef:
    """Reference to the ``TypeTree.defs`` entry keyed like the JSON Schema's ``$defs``."""

    key: str


@dataclass(frozen=True, slots=True)
class TypeNodeField:
    """One record or member field: declared name, JSON key, field ``@doc``, and type."""

    name: str
    json_name: str
    doc: str | None
    node: "TypeTreeEntry"


@dataclass(frozen=True, slots=True)
class TypeNode:
    """One described type.

    ``label`` is the type's schema-canonical spelling (shared by every
    occurrence of a hoisted definition); the root's own label is
    ``TargetContractRequest.target_type_label``. ``schema`` is the node's JSON Schema
    fragment as a JSON string, whose ``$ref``s resolve against the tree's ``defs``.
    ``doc`` is the declaration's ``@doc``. ``nominal`` identifies a record,
    enum, or member declaration. ``fields`` (records, members) and
    ``members`` (enums, as ``(json_tag, member)``) keep declaration order;
    ``items`` is an array's elements, ``keys``/``values`` are a dict's keys and values.
    """

    kind: TypeNodeKind
    label: str
    schema: str
    doc: str | None = None
    nominal: NominalId | None = None
    fields: tuple[TypeNodeField, ...] = ()
    members: "tuple[tuple[str, TypeNode], ...]" = ()
    items: "TypeTreeEntry | None" = None
    keys: "TypeTreeEntry | None" = None
    values: "TypeTreeEntry | None" = None


TypeTreeEntry = TypeNode | TypeNodeRef


@dataclass(frozen=True, slots=True)
class TypeTree:
    """A target type's description mirroring its ``DecodePlan``: root plus recursive ``defs``."""

    root: TypeTreeEntry
    defs: tuple[tuple[str, TypeNode], ...] = ()


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
class UnitContractRequest:
    """A ``unit`` target: the evaluator discards the output without parsing it.

    Its common fields are constants. See ``JsonContractRequest`` for the
    common-field documentation shared by every ``ContractRequest`` variant.
    """

    codec_name: ClassVar[Literal["none"]] = "none"
    strict_json: ClassVar[None] = None
    target_type_label: ClassVar[str] = "unit"
    structured_exec: ClassVar[bool] = False
    format_instructions: ClassVar[str] = ""


@dataclass(frozen=True, slots=True)
class TextContractRequest:
    """No-schema contract descriptor for the built-in ``text`` codec.

    See ``JsonContractRequest`` for the common-field documentation shared by
    every ``ContractRequest`` variant.
    """

    strict_json: bool | None
    target_type_label: str
    structured_exec: bool
    format_instructions: str
    codec_name: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class JsonContractRequest:
    """Contract descriptor for the built-in ``json`` codec: ``json_schema``
    and ``decode`` are always derived and present. ``codec_name`` is always
    ``"json"``.

    Built at lowering while checker types are available; evaluated WITHOUT any
    checker ``Type``.  The evaluator parses agent output using only this descriptor.

    ``strict_json``         — per-call strict_json override; ``None`` → use the
                              evaluator-level default.
    ``json_schema``         — canonical JSON string of the derived schema
                              (``json.dumps(..., sort_keys=True)``).
    ``decode``              — typeless ``DecodeSchema`` walk for the target type.
    ``target_type_label``   — ``repr(target_type)`` stored for ``AgentParseError``
                              field text and failure-message formatting.
    ``structured_exec``     — ``True`` for structured exec; ``False`` for ``ask``.
    ``format_instructions`` — pre-computed format instructions string.
    ``defs``                — ``$defs`` table for a recursive target type (empty
                              for a non-recursive one, see ``DecodePlan``).
    """

    strict_json: bool | None
    json_schema: str
    decode: "DecodeSchema"
    target_type_label: str
    structured_exec: bool
    format_instructions: str
    codec_name: Literal["json"] = "json"
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()


@dataclass(frozen=True, slots=True)
class CustomContractRequest:
    """Contract descriptor for a host-registered custom codec.

    ``json_schema``/``decode`` mirror whatever the codec's own
    ``make_contract`` hook produced while checker types were still available
    (see ``ContractPayload``): a custom codec may or may not derive a schema,
    so unlike ``JsonContractRequest`` these stay optional. The evaluator never
    parses output through this descriptor directly -- it dispatches to the
    codec's own ``parse`` hook (see ``eval.ir_interpreter._call_custom_codec_parse``).
    ``target_type`` is the opaque checker type, retained only for legacy
    custom codecs whose ``parse`` hook accepts a positional target type;
    runtime-neutral code must not inspect it.
    """

    codec_name: str
    strict_json: bool | None
    json_schema: str | None
    decode: "DecodeSchema | None"
    target_type_label: str
    structured_exec: bool
    format_instructions: str
    target_type: object
    defs: "tuple[tuple[str, DecodeSchema], ...]" = ()


#: Closed tagged-data describing one ask/exec output contract, one variant per
#: codec shape. ``codec_name`` alone never determines which fields are present;
#: match on the variant instead.
type ContractRequest = (
    UnitContractRequest | TextContractRequest | JsonContractRequest | CustomContractRequest
)


@dataclass(frozen=True, slots=True)
class TargetContractRequest:
    """A type-directed extern's target contract, passed to its companion.

    ``json_schema`` is the target's canonical JSON Schema string and
    ``type_tree`` its ``TypeTree``; ``target_type_label`` is the target's
    display label.
    """

    target_type_label: str
    json_schema: str
    type_tree: TypeTree


_SchemaT = TypeVar("_SchemaT")


def resolve_schema_ref(
    key: str,
    defs: Mapping[str, _SchemaT],
    forwarded_key: Callable[[_SchemaT], str | None],
) -> _SchemaT:
    """Follow ``$defs`` references to the first non-reference body.

    *forwarded_key* returns the onward key of a reference node, or ``None``
    once the walk reaches a body. Encode and decode plans share the same
    ``$defs`` keying, so they share this walk.
    """
    current = key
    while True:
        resolved = defs[current]
        onward = forwarded_key(resolved)
        if onward is None:
            return resolved
        current = onward
