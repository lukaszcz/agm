"""Strict-parse/normalize primitives and the typeless decode walk for AgL.

This module is the canonical home for the reusable building blocks shared by
the cast path (``as`` / ``as?`` operators), the agent/exec output codec, and
host parameter decoding:

- :exc:`StrictJsonParseError` — raised by :func:`parse_json_strict` on any
  malformed or non-conforming input.
- :func:`parse_json_strict` — strict ``json.loads`` with ``parse_float=Decimal``
  and ``parse_constant`` that rejects non-standard constants
  (``NaN`` / ``Infinity`` / ``-Infinity``) even when nested inside containers.
  Also rejects any trailing/leading non-whitespace.  Returns the raw parsed
  Python object.
- :func:`normalize_integral_decimals` — walk a JSON-shaped tree and replace
  integral ``Decimal`` values with ``int`` (D4 / F2 rule).
- :func:`_clean_validation_message` — strip Python ``Decimal(...)`` reprs from
  jsonschema error messages before surfacing them to users.
- :func:`decode_value` / :func:`_decode_scalar` — the typeless
  ``DecodeSchema``-driven decode walk.  The single decode path shared by
  casts, the agent/exec codec, and host param decoding.  Raises ``ValueError``
  on any type mismatch; callers convert this into domain errors.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import assert_never

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonschemaValidationError

from agm.agl.ir.contracts import (
    DecodeSchema,
    DictDecode,
    EnumDecode,
    ListDecode,
    RecordDecode,
    ScalarDecode,
    ScalarKind,
)
from agm.agl.semantics.values import (
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


# ---------------------------------------------------------------------------
# Cached JSON-Schema validators
# ---------------------------------------------------------------------------

_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}


def validator_for_schema(json_schema: str) -> Draft202012Validator:
    """Compile (and cache) a JSON-Schema validator from its canonical JSON string.

    Shared by the cast path (``conversions``) and host param decoding
    (``params``) so identical schemas are compiled once.
    """
    validator = _VALIDATOR_CACHE.get(json_schema)
    if validator is None:
        schema_obj: object = json.loads(json_schema)
        validator = Draft202012Validator(schema_obj)
        _VALIDATOR_CACHE[json_schema] = validator
    return validator


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

    Decimal targets re-widen ``int`` → ``Decimal`` via ``decode_value`` and a
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
# _clean_validation_message
# ---------------------------------------------------------------------------


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
# decode_value / _decode_scalar — the typeless DecodeSchema-driven decode walk
# ---------------------------------------------------------------------------


def decode_value(schema: DecodeSchema, obj: object) -> Value:
    """Construct a typed ``Value`` from JSON-shaped *obj* per *schema*.

    The typeless ``DecodeSchema``-driven decode walk shared by the cast path
    (``as`` / ``as?``), the agent/exec output codec, and host param decoding.
    Raises ``ValueError`` on any type mismatch; callers convert this into the
    appropriate domain error (``AglCastConversion``, ``ValidationError``, etc.).
    """
    match schema:
        case ScalarDecode(kind=kind):
            return _decode_scalar(kind, obj)
        case ListDecode(elem=elem):
            if not isinstance(obj, list):
                raise ValueError(f"Expected array, got {type(obj).__name__}")
            return ListValue(tuple(decode_value(elem, e) for e in obj))
        case DictDecode(value=value_schema):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object, got {type(obj).__name__}")
            entries: dict[str, Value] = {}
            for k, v in obj.items():
                if not isinstance(k, str):
                    raise ValueError(f"Dict key must be string, got {type(k).__name__}")
                entries[k] = decode_value(value_schema, v)
            return DictValue(entries=entries)
        case RecordDecode(nominal=nominal, display_name=display_name, fields=fields):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object for record, got {type(obj).__name__}")
            record_fields: dict[str, Value] = {}
            for fname, fschema in fields:
                if fname not in obj:
                    raise ValueError(f"Missing field {fname!r}")
                record_fields[fname] = decode_value(fschema, obj[fname])
            return RecordValue(
                nominal=nominal, display_name=display_name, fields=record_fields
            )
        case EnumDecode(nominal=nominal, display_name=display_name, variants=variants):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object for enum, got {type(obj).__name__}")
            case_val = obj.get("$case")
            if not isinstance(case_val, str):
                raise ValueError("Enum object must have a string '$case' field")
            variant = next((v for v in variants if v.name == case_val), None)
            if variant is None:
                raise ValueError(
                    f"Unknown enum variant {case_val!r} for {display_name!r}. "
                    f"Valid variants: {[v.name for v in variants]}"
                )
            payload: dict[str, Value] = {}
            for fname, fschema in variant.fields:
                if fname not in obj:
                    raise ValueError(
                        f"Enum variant {case_val!r} is missing field {fname!r}"
                    )
                payload[fname] = decode_value(fschema, obj[fname])
            return EnumValue(
                nominal=nominal,
                display_name=display_name,
                variant=case_val,
                fields=payload,
            )
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _decode_scalar(kind: ScalarKind, obj: object) -> Value:
    """Decode a JSON scalar into the matching leaf ``Value``."""
    match kind:
        case ScalarKind.TEXT:
            if isinstance(obj, str):
                return TextValue(obj)
            raise ValueError(f"Expected string, got {type(obj).__name__}")
        case ScalarKind.INT:
            if isinstance(obj, bool):
                raise ValueError("Expected integer, got bool")
            if isinstance(obj, int):
                return IntValue(obj)
            if isinstance(obj, Decimal) and obj == obj.to_integral_value():
                return IntValue(int(obj))
            raise ValueError(f"Expected integer, got {type(obj).__name__} {obj!r}")
        case ScalarKind.DECIMAL:
            if isinstance(obj, bool):
                raise ValueError("Expected decimal, got bool")
            if isinstance(obj, Decimal):
                return DecimalValue(obj)
            if isinstance(obj, int):
                return DecimalValue(Decimal(obj))
            raise ValueError(f"Expected decimal, got {type(obj).__name__} {obj!r}")
        case ScalarKind.BOOL:
            if isinstance(obj, bool):
                return BoolValue(obj)
            raise ValueError(f"Expected bool, got {type(obj).__name__}")
        case ScalarKind.JSON:
            return JsonValue(obj)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
