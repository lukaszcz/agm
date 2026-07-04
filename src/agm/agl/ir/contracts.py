"""Runtime descriptors for the AgL typeless execution IR.

This module holds closed tagged-data descriptors that the lowerer compiles
while checker types are still available, and that the evaluator executes
WITHOUT any checker ``Type``.  It defines the cast/conversion descriptors
(``ConversionRecipe`` and the ``DecodeSchema`` union) and the extern boundary
contract (``BoundarySchema`` and ``ExternContract``); the boundary walkers
that consume the contract at runtime arrive in future work.

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
    "BoundaryDict",
    "BoundaryEnum",
    "BoundaryException",
    "BoundaryList",
    "BoundaryRecord",
    "BoundaryScalar",
    "BoundarySchema",
    "BoundarySealVar",
    "BoundaryUnit",
    "BoundaryVariantShape",
    "ContractRequest",
    "ConversionFailureMode",
    "ConversionRecipe",
    "ConversionStrategy",
    "DecodeSchema",
    "DictDecode",
    "EnumDecode",
    "ExternContract",
    "ExternParamSchema",
    "ListDecode",
    "ParamDecoder",
    "RecordDecode",
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


#: Closed union of decode-schema nodes.  Dispatch with a structural ``match``
#: whose final arm is ``assert_never``.
DecodeSchema = ScalarDecode | ListDecode | DictDecode | RecordDecode | EnumDecode


@dataclass(frozen=True, slots=True)
class ParamDecoder:
    """Typeless decoder for one host-supplied entry parameter."""

    target_type_label: str
    json_schema: str
    decode: DecodeSchema
    text_verbatim: bool = False


# ---------------------------------------------------------------------------
# Boundary schema — typeless shape of one value crossing the extern (Python
# FFI) boundary.  One schema serves both directions: encoding an outbound
# argument and strictly decoding an inbound return value walk the same
# nodes; the walker (a future runtime component) owns the direction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundaryScalar:
    """A scalar (or opaque json) leaf crossing the boundary unchanged."""

    kind: ScalarKind


@dataclass(frozen=True, slots=True)
class BoundaryUnit:
    """``unit`` crosses the boundary as Python ``None``."""


@dataclass(frozen=True, slots=True)
class BoundaryList:
    """A ``list[T]`` crossing as a Python ``list``, recursing on elements."""

    element: "BoundarySchema"


@dataclass(frozen=True, slots=True)
class BoundaryDict:
    """A ``dict[text, V]`` crossing as a Python ``dict``, recursing on values."""

    value: "BoundarySchema"


@dataclass(frozen=True, slots=True)
class BoundaryRecord:
    """A record crossing as a field dict, in declaration order."""

    nominal: NominalId
    display_name: str
    fields: "tuple[tuple[str, BoundarySchema], ...]"


@dataclass(frozen=True, slots=True)
class BoundaryVariantShape:
    """One enum variant: its ``$case`` name and ordered field schemas."""

    name: str
    fields: "tuple[tuple[str, BoundarySchema], ...]"


@dataclass(frozen=True, slots=True)
class BoundaryEnum:
    """An enum crossing as a ``{"$case": ..., ...fields}`` dict."""

    nominal: NominalId
    display_name: str
    variants: tuple[BoundaryVariantShape, ...]


@dataclass(frozen=True, slots=True)
class BoundaryException:
    """An exception crossing as a field dict, decoding into an exception value.

    Distinct from ``BoundaryRecord``: an exception is not a record type, and
    decoding it constructs an exception value rather than a plain record.
    """

    nominal: NominalId
    display_name: str
    fields: "tuple[tuple[str, BoundarySchema], ...]"


@dataclass(frozen=True, slots=True)
class BoundarySealVar:
    """A type-variable position: the runtime seals/unseals here.

    ``var`` is the declared type-parameter name (unique within one contract).
    """

    var: str


#: Closed union of boundary-schema nodes.  Dispatch with a structural
#: ``match`` whose final arm is ``assert_never``.
BoundarySchema = (
    BoundaryScalar
    | BoundaryUnit
    | BoundaryList
    | BoundaryDict
    | BoundaryRecord
    | BoundaryEnum
    | BoundaryException
    | BoundarySealVar
)


@dataclass(frozen=True, slots=True)
class ExternParamSchema:
    """One extern parameter's boundary schema plus its diagnostic label.

    ``label`` is the human-readable type label (``repr(type)``) used in
    ``ExternError`` messages and other diagnostics.
    """

    label: str
    schema: BoundarySchema


@dataclass(frozen=True, slots=True)
class ExternContract:
    """Typeless boundary contract for one ``extern def``.

    Built at lowering from the checked ``FunctionSignature`` while checker
    types are still available; the runtime encodes arguments and decodes the
    return value using only this descriptor, without any checker ``Type``.

    ``params``       — one schema per declared parameter, in declaration order.
    ``result``       — the boundary schema for the return value.
    ``type_params``  — the extern's declared type-parameter names, in
                        declaration order; each name may appear as a
                        ``BoundarySealVar`` leaf somewhere in ``params``/
                        ``result``.
    ``result_label`` — the human-readable return-type label (``repr(type)``)
                        used in ``ExternError`` messages.
    """

    params: tuple[ExternParamSchema, ...]
    result: BoundarySchema
    type_params: tuple[str, ...]
    result_label: str


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
    node maintains) — and ``decode`` carries the typeless decode walk; both are
    ``None`` for the total strategies.
    """

    strategy: ConversionStrategy
    source_label: str
    target_label: str
    json_schema: str | None = None
    decode: DecodeSchema | None = None


# ---------------------------------------------------------------------------
# Contract request — per-call ask/ask-request descriptor
# ---------------------------------------------------------------------------


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
    ``structured_exec``     — ``True`` for structured exec; ``False`` for ``ask``.
    ``format_instructions`` — pre-computed format instructions string (empty for
                              text codec and unit-typed asks).
    ``is_unit``             — ``True`` when the target type is ``unit`` (unit
                              target); the evaluator dispatches the agent call but
                              skips output parsing and returns ``UnitValue``
                              immediately.  For ``ask-request`` the result is always
                              an ``AgentRequest`` record, never unit — ``is_unit``
                              is always ``False`` for ``ask-request`` call sites.
    """

    codec_name: str
    strict_json: bool | None
    json_schema: str | None
    decode: "DecodeSchema | None"
    target_type_label: str
    structured_exec: bool
    format_instructions: str
    is_unit: bool = False
