"""Shared strict-parse/validate conversion core for AgL type casts.

This module provides the reusable building blocks for both the strict cast path
(``as`` / ``as?`` operators) and the ``parse_json`` built-in:

- :exc:`StrictJsonParseError` — raised by :func:`parse_json_strict` on any
  malformed or non-conforming input.
- :exc:`CastConversionError` — raised by :func:`json_obj_to_value` and
  :func:`convert_value` when a conversion fails for expected reasons (type
  mismatch, non-integral narrowing, schema violation, parse failure).  Carries
  enough context (``source_type``, ``target_type``, ``raw``, ``message``) to
  build an AgL ``CastError`` exception value in the evaluator (M5).
- :func:`parse_json_strict` — strict ``json.loads`` with ``parse_float=Decimal``
  and ``parse_constant`` that rejects non-standard constants
  (``NaN`` / ``Infinity`` / ``-Infinity``) even when nested inside containers.
  Also rejects any trailing/leading non-whitespace.  Returns the raw parsed
  Python object.
- :func:`normalize_integral_decimals` — walk a JSON-shaped tree and replace
  integral ``Decimal`` values with ``int`` (D4 / F2 rule).
- :func:`json_to_value` — convert a JSON-shaped Python object into the typed
  ``Value`` for a given ``Type``.  Raises ``ValueError`` on mismatch.
- :func:`json_obj_to_value` — normalize integral decimals, validate against the
  JSON Schema derived from ``typ``, then construct the typed ``Value``.
- :func:`convert_value` — the single conversion entry point realizing the full
  D1 matrix.  Translates only :exc:`StrictJsonParseError` and expected
  validation/conversion failures into :exc:`CastConversionError`; all other
  exceptions propagate unchanged.

The lenient agent/``exec`` parsing path in :mod:`agm.agl.runtime.codec` imports
:func:`normalize_integral_decimals` and :func:`json_to_value` from this module.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonschemaValidationError

from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.runtime.render import render_value
from agm.agl.runtime.schema import derive_schema
from agm.agl.runtime.serialize import dumps_exact, value_to_json_obj
from agm.agl.typecheck.types import (
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
)

# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class StrictJsonParseError(Exception):
    """Raised by :func:`parse_json_strict` when input is not a single valid JSON value.

    Covers:
    - Malformed JSON (syntax errors).
    - Non-standard constants: ``NaN``, ``Infinity``, ``-Infinity`` (including
      when nested inside containers such as ``[NaN]`` or ``{"x": Infinity}``).
    - Trailing or leading non-whitespace beyond the JSON value.
    - Empty / whitespace-only input.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CastConversionError(Exception):
    """Raised when a type conversion fails for an expected reason.

    Carries the context needed for M5 to build an AgL ``CastError`` value:

    ``source_type``  — display name of the source type (e.g. ``"text"``).
    ``target_type``  — display name of the target type (e.g. ``"int"``).
    ``raw``          — rendered source value (via :func:`render_value`).
    ``message``      — human-readable description of the failure.

    Only raised for conversion failures that are expected at runtime (type
    mismatch, non-integral narrowing, schema violation, strict-parse error).
    Unexpected internal errors and programming defects propagate unchanged.
    """

    def __init__(
        self,
        message: str,
        *,
        source_type: str,
        target_type: str,
        raw: str,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.source_type = source_type
        self.target_type = target_type
        self.raw = raw


# ---------------------------------------------------------------------------
# normalize_integral_decimals (D4 / F2) — canonical home in convert.py
# ---------------------------------------------------------------------------


def normalize_integral_decimals(obj: object) -> object:
    """Convert integral ``Decimal`` values to ``int`` throughout *obj*.

    Walks the JSON-shaped tree produced by ``json.loads(parse_float=Decimal)``
    and replaces any ``Decimal`` whose value is integral and lossless
    (``d == int(d)``) with the equivalent ``int``.  This lets a wire value of
    ``1.0`` satisfy an ``{"type": "integer"}`` schema; non-integral decimals
    such as ``1.5`` are preserved and continue to fail integer targets.

    Decimal targets re-widen ``int`` → ``Decimal`` via ``json_to_value`` and a
    ``json`` passthrough sees a JSON-equal value, so this is loss-free.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, Decimal):
        if obj == obj.to_integral_value():
            return int(obj)
        return obj
    if isinstance(obj, list):
        items: list[object] = obj
        return [normalize_integral_decimals(e) for e in items]
    if isinstance(obj, dict):
        mapping: dict[object, object] = obj
        return {k: normalize_integral_decimals(v) for k, v in mapping.items()}
    return obj


# ---------------------------------------------------------------------------
# json_to_value — canonical home in convert.py
# ---------------------------------------------------------------------------


def json_to_value(obj: object, typ: Type) -> Value:
    """Convert a JSON-shaped Python object to the appropriate typed ``Value``.

    ``obj`` is the result of ``json.loads(parse_float=Decimal)`` — it may be
    ``dict``, ``list``, ``str``, ``int``, ``Decimal``, ``bool``, or ``None``.
    ``Decimal`` is never converted to ``float`` (design §5.1).

    Raises ``ValueError`` for type mismatches (the caller handles these).
    """
    if isinstance(typ, TextType):
        if isinstance(obj, str):
            return TextValue(obj)
        raise ValueError(f"Expected string, got {type(obj).__name__}")

    if isinstance(typ, IntType):
        if isinstance(obj, bool):
            raise ValueError("Expected integer, got bool")
        if isinstance(obj, int):
            return IntValue(obj)
        # Integral Decimals are normalized to ``int`` before validation/conversion
        # (see ``normalize_integral_decimals``), so any Decimal reaching here is
        # non-integral and rejected for an int target.
        raise ValueError(f"Expected integer, got {type(obj).__name__} {obj!r}")

    if isinstance(typ, DecimalType):
        if isinstance(obj, bool):
            raise ValueError("Expected decimal, got bool")
        if isinstance(obj, Decimal):
            return DecimalValue(obj)
        if isinstance(obj, int):
            return DecimalValue(Decimal(obj))
        raise ValueError(f"Expected decimal, got {type(obj).__name__} {obj!r}")

    if isinstance(typ, BoolType):
        if isinstance(obj, bool):
            return BoolValue(obj)
        raise ValueError(f"Expected bool, got {type(obj).__name__}")

    if isinstance(typ, JsonType):
        # Accept any JSON-shaped value.
        return JsonValue(obj)

    if isinstance(typ, ListType):
        if not isinstance(obj, list):
            raise ValueError(f"Expected array, got {type(obj).__name__}")
        elements = tuple(json_to_value(e, typ.elem) for e in obj)
        return ListValue(elements=elements)

    if isinstance(typ, DictType):
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object, got {type(obj).__name__}")
        entries: dict[str, Value] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"Dict key must be string, got {type(k).__name__}")
            entries[k] = json_to_value(v, typ.value)
        return DictValue(entries=entries)

    if isinstance(typ, RecordType):
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object for record, got {type(obj).__name__}")
        fields: dict[str, Value] = {}
        for field_name, field_type in typ.fields.items():
            if field_name not in obj:
                raise ValueError(f"Missing field {field_name!r}")
            fields[field_name] = json_to_value(obj[field_name], field_type)
        return RecordValue(type_name=typ.name, fields=fields)

    if isinstance(typ, EnumType):
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object for enum, got {type(obj).__name__}")
        case_val = obj.get("$case")
        if not isinstance(case_val, str):
            raise ValueError("Enum object must have a string '$case' field")
        variant_fields = typ.variants.get(case_val)
        if variant_fields is None:
            raise ValueError(
                f"Unknown enum variant {case_val!r} for {typ.name!r}. "
                f"Valid variants: {list(typ.variants.keys())}"
            )
        payload: dict[str, Value] = {}
        for field_name, field_type in variant_fields.items():
            if field_name not in obj:
                raise ValueError(
                    f"Enum variant {case_val!r} is missing field {field_name!r}"
                )
            payload[field_name] = json_to_value(obj[field_name], field_type)
        return EnumValue(type_name=typ.name, variant=case_val, fields=payload)

    # ExceptionType is not wire-serialised by the JSON codec.
    raise ValueError(f"Cannot deserialise type {typ!r} from JSON")


# ---------------------------------------------------------------------------
# parse_json_strict
# ---------------------------------------------------------------------------


def _reject_constant(c: str) -> object:
    """Raise :exc:`StrictJsonParseError` for any non-standard JSON constant.

    Passed as ``parse_constant`` to :func:`json.loads` so that ``NaN``,
    ``Infinity``, and ``-Infinity`` are rejected even when they appear nested
    inside containers such as ``[NaN]`` or ``{"x": Infinity}``.
    """
    raise StrictJsonParseError(
        f"Non-standard JSON constant {c!r} is not permitted in strict mode"
    )


def parse_json_strict(text: str) -> object:
    """Parse *text* as a single strict JSON value.

    Rules:
    - Leading and trailing whitespace are ignored (as per JSON specification).
    - Exactly one JSON value must be present — no trailing junk.
    - Non-standard constants ``NaN``, ``Infinity``, and ``-Infinity`` are
      rejected even when nested inside containers (e.g. ``[NaN]``,
      ``{"x": Infinity}``).  They are not valid JSON.
    - Floating-point numbers are parsed as :class:`decimal.Decimal` (never
      ``float``), preserving exact precision.

    :returns: The parsed Python object (``dict``, ``list``, ``str``, ``int``,
              :class:`decimal.Decimal`, ``bool``, or ``None``).
    :raises StrictJsonParseError: On any malformed or non-conforming input.
    """
    stripped = text.strip()
    if not stripped:
        raise StrictJsonParseError("Empty input: no JSON value found")

    try:
        # json.JSONDecoder.decode (used by json.loads) calls raw_decode and then
        # verifies that only whitespace follows the first value — so trailing junk
        # such as "42 extra" is already rejected with JSONDecodeError.
        #
        # parse_constant=_reject_constant ensures NaN/Infinity/-Infinity raise
        # StrictJsonParseError even when nested inside containers like [NaN].
        obj: object = json.loads(
            stripped, parse_float=Decimal, parse_constant=_reject_constant
        )
    except StrictJsonParseError:
        raise
    except ValueError as exc:
        raise StrictJsonParseError(f"JSON parse error: {exc}") from exc

    return obj


# ---------------------------------------------------------------------------
# json_obj_to_value
# ---------------------------------------------------------------------------


def json_obj_to_value(obj: object, typ: Type) -> Value:
    """Convert a JSON-shaped Python object *obj* into the typed ``Value`` for *typ*.

    The *obj* is typically the result of :func:`parse_json_strict` (which uses
    ``parse_float=Decimal``) or the ``.raw`` attribute of a :class:`JsonValue`.

    Steps:
    1. Normalize integral ``Decimal`` values to ``int`` (D4 / F2 rule): a
       ``Decimal`` that equals its integral value is replaced with the
       equivalent ``int``.  This lets ``1.0`` satisfy an integer schema while
       ``1.5`` still fails.
    2. Validate the normalized object against the JSON Schema derived from
       *typ* via :func:`~agm.agl.runtime.schema.derive_schema`.
    3. Construct the typed ``Value`` via :func:`json_to_value`.

    :raises CastConversionError: On schema validation failure or type-mismatch
        in value construction.
    """
    normalized = normalize_integral_decimals(obj)

    schema = derive_schema(typ)
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(normalized))
    if errors:
        # Build a clean message that does not leak Python Decimal(...) reprs.
        # We use the clean rendering of the source object and the target type
        # name instead of the raw jsonschema instance repr.
        clean_raw = dumps_exact(obj, indent=None)
        msgs = "; ".join(_clean_validation_message(e) for e in errors)
        raise CastConversionError(
            f"Schema validation failed: {msgs}",
            source_type="json",
            target_type=repr(typ),
            raw=clean_raw,
        )

    try:
        return json_to_value(normalized, typ)
    except ValueError as exc:
        # Translate any ValueError from json_to_value into CastConversionError
        # so no bare ValueError escapes this module boundary.
        raise CastConversionError(
            f"Value conversion failed: {exc}",
            source_type="json",
            target_type=repr(typ),
            raw=dumps_exact(obj, indent=None),
        ) from exc


def _clean_validation_message(error: JsonschemaValidationError) -> str:
    """Return a clean validation error message without Python internal reprs.

    jsonschema error messages embed the Python repr of the ``instance``
    (e.g. ``Decimal('3.5') is not of type 'integer'``).  We reconstruct
    a cleaner message from the error's ``message`` with Decimal reprs
    replaced by their string equivalents.
    """
    # Replace occurrences of Decimal('...') with the bare number string.
    return re.sub(r"Decimal\('([^']*)'\)", r"\1", error.message)


# ---------------------------------------------------------------------------
# convert_value — the single conversion entry point (D1 matrix)
# ---------------------------------------------------------------------------


def _rewrap_conversion_error(
    exc: CastConversionError, src_name: str, tgt_name: str, raw: str
) -> CastConversionError:
    """Return a new CastConversionError with the correct source/target/raw context."""
    return CastConversionError(exc.message, source_type=src_name, target_type=tgt_name, raw=raw)


def convert_value(value: Value, source_type: Type, target_type: Type) -> Value:
    """Convert *value* of *source_type* to *target_type*, realizing the D1 matrix.

    Conversions by target type:

    ``→ text`` (total):
        :func:`~agm.agl.runtime.render.render_value` is used to produce a
        :class:`~agm.agl.eval.values.TextValue`.  Nominal sources render in
        declaration order (the interpreter normalizes their fields).

    ``→ json`` (total):
        :func:`~agm.agl.runtime.serialize.value_to_json_obj` is used to
        produce a :class:`~agm.agl.eval.values.JsonValue`.  Per D9, ``text``
        is already JSON-shaped (a JSON string), so ``"42" as json`` yields
        ``JsonValue("42")``, not ``JsonValue(42)``.  Nominal values
        (record/enum/exception) are accepted and serialised structurally.

    ``text → T`` (fallible, T not json):
        :func:`parse_json_strict` parses the text strictly, then
        :func:`json_obj_to_value` validates and constructs the typed value.

    ``json → T`` (fallible, T not json):
        :func:`json_obj_to_value` validates ``.raw`` directly (no re-parse).

    ``decimal → int`` (D4, fallible):
        Succeeds iff the decimal has no fractional part; yields the exact
        integer.  Routes through :func:`json_obj_to_value` for consistency
        with the integrality rule.

    No-op / assignable cases (D6):
        Exact-type identity returns the value unchanged.  ``int → decimal``
        widens to :class:`~agm.agl.eval.values.DecimalValue`.

    :raises CastConversionError: When a fallible conversion fails for an
        expected reason (parse error, type mismatch, non-integral narrowing).
        Carries ``source_type``, ``target_type``, ``raw``, and ``message``.
        All other exceptions propagate unchanged.
    """
    # ------------------------------------------------------------------
    # → text (total): render any value to text
    # ``raw`` is the rendered form and also the result here.
    # ------------------------------------------------------------------
    if isinstance(target_type, TextType):
        return TextValue(render_value(value))

    # ------------------------------------------------------------------
    # → json (total): canonicalize any JSON-shaped value
    # Per D9: text as json wraps the text as a JSON string (no parsing).
    # Nominal values (record/enum/exception) are serialised structurally.
    # ------------------------------------------------------------------
    if isinstance(target_type, JsonType):
        return JsonValue(value_to_json_obj(value))

    # ------------------------------------------------------------------
    # No-op / identity: source == target (return unchanged)
    # Note: must come after the total → text / → json branches.
    # ------------------------------------------------------------------
    if source_type == target_type:
        return value

    # ------------------------------------------------------------------
    # int → decimal widening (D6 coercion)
    # ------------------------------------------------------------------
    if isinstance(source_type, IntType) and isinstance(target_type, DecimalType):
        assert isinstance(value, IntValue)
        return DecimalValue(Decimal(value.value))

    # Fallible paths below need source/target names and raw for error context.
    # ``render_value`` is called here (lazily) so the total → json / identity /
    # widening branches above never pay to render the value to text.
    src_name = repr(source_type)
    tgt_name = repr(target_type)
    raw = render_value(value)

    # ------------------------------------------------------------------
    # decimal → int narrowing (D4: must be integral)
    # ------------------------------------------------------------------
    if isinstance(source_type, DecimalType) and isinstance(target_type, IntType):
        assert isinstance(value, DecimalValue)
        obj: object = value.value
        try:
            return json_obj_to_value(obj, target_type)
        except CastConversionError as exc:
            raise _rewrap_conversion_error(exc, src_name, tgt_name, raw) from exc

    # ------------------------------------------------------------------
    # text → T (fallible, T is not json — handled above)
    # ------------------------------------------------------------------
    if isinstance(source_type, TextType):
        assert isinstance(value, TextValue)
        try:
            parsed_obj = parse_json_strict(value.value)
        except StrictJsonParseError as exc:
            raise CastConversionError(
                f"Failed to parse text as JSON: {exc.message}",
                source_type=src_name,
                target_type=tgt_name,
                raw=raw,
            ) from exc
        try:
            return json_obj_to_value(parsed_obj, target_type)
        except CastConversionError as exc:
            raise _rewrap_conversion_error(exc, src_name, tgt_name, raw) from exc

    # ------------------------------------------------------------------
    # json → T (fallible, T is not json)
    # ------------------------------------------------------------------
    if isinstance(source_type, JsonType):
        assert isinstance(value, JsonValue)
        try:
            return json_obj_to_value(value.raw, target_type)
        except CastConversionError as exc:
            raise _rewrap_conversion_error(exc, src_name, tgt_name, raw) from exc

    # ------------------------------------------------------------------
    # Fallthrough: should not be reached for valid (typechecked) casts.
    # The typecheck pass (M3) rejects statically-impossible pairs, so
    # this path indicates an implementation bug — raise unconditionally.
    # ------------------------------------------------------------------
    raise ValueError(  # pragma: no cover
        f"convert_value: no conversion path from {src_name!r} to {tgt_name!r}; "
        "this is an implementation bug — the typecheck pass should have rejected "
        "this combination as a static cast error."
    )
