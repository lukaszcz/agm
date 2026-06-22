"""Frontend-free cast/conversion executor for the AgL IR evaluator (M3e-2).

``run_recipe(recipe, value)`` executes a typeless ``ConversionRecipe`` against a
runtime ``Value`` and returns the converted ``Value``.  On an expected fallible
failure it raises the module-private ``AglCastConversion`` sentinel (carrying
the message + user-facing source/target labels + rendered raw value); the
caller wraps it into the appropriate ``CastError`` / ``BoolValue(False)``.

This module is the single source of truth for the typeless decode walk
(``_decode``), which mirrors ``runtime.convert.json_to_value`` so the two
evaluators stay in lock-step (the differential oracle proves it).  It reuses
the existing runtime leaf primitives (rendering, JSON serialization, strict
parse, integral-decimal normalization, JSON-Schema validation) rather than
reimplementing them.

Imports: stdlib + ``agm.agl.values``/``agm.agl.eval.values`` + ``agm.agl.ir``
contracts + ``agm.agl.runtime`` leaf helpers.  No ``syntax`` / ``scope`` /
``typecheck`` imports are permitted here.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import assert_never

from jsonschema import Draft202012Validator

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
from agm.agl.ir.contracts import (
    ConversionRecipe,
    ConversionStrategy,
    DecodeSchema,
    DictDecode,
    EnumDecode,
    ListDecode,
    RecordDecode,
    ScalarDecode,
    ScalarKind,
)
from agm.agl.runtime.convert import (
    StrictJsonParseError,
    _clean_validation_message,
    normalize_integral_decimals,
    parse_json_strict,
)
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import value_to_json_obj

__all__ = ["AglCastConversion", "decode_value", "run_recipe"]


class AglCastConversion(Exception):
    """Sentinel: a fallible cast conversion failed for an expected reason.

    Mirrors the field set of the legacy ``CastError`` so the IR evaluator can
    build an identical exception value.
    """

    def __init__(
        self, message: str, *, source_label: str, target_label: str, raw: str
    ) -> None:
        super().__init__(message)
        self.message = message
        self.source_label = source_label
        self.target_label = target_label
        self.raw = raw


_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}


def _validator_for(json_schema: str) -> Draft202012Validator:
    """Compile (and cache) a JSON-Schema validator from its canonical JSON string."""
    validator = _VALIDATOR_CACHE.get(json_schema)
    if validator is None:
        schema_obj: object = json.loads(json_schema)
        validator = Draft202012Validator(schema_obj)
        _VALIDATOR_CACHE[json_schema] = validator
    return validator


def run_recipe(recipe: ConversionRecipe, value: Value) -> Value:
    """Execute *recipe* against *value*; raise ``AglCastConversion`` on failure."""
    match recipe.strategy:
        case ConversionStrategy.NOOP:
            return value
        case ConversionStrategy.WIDEN_INT_TO_DECIMAL:
            if not isinstance(value, IntValue):
                raise AssertionError(  # pragma: no cover
                    f"WIDEN_INT_TO_DECIMAL expected IntValue, got {type(value).__name__}"
                )
            return DecimalValue(Decimal(value.value))
        case ConversionStrategy.RENDER_TO_TEXT:
            return TextValue(render_value(value))
        case ConversionStrategy.TO_JSON:
            return JsonValue(value_to_json_obj(value))
        case ConversionStrategy.NARROW_DECIMAL_TO_INT:
            if not isinstance(value, DecimalValue):
                raise AssertionError(  # pragma: no cover
                    f"NARROW_DECIMAL_TO_INT expected DecimalValue, got {type(value).__name__}"
                )
            return _decode_from_json(recipe, value.value, value)
        case ConversionStrategy.PARSE_TEXT_THEN_DECODE:
            if not isinstance(value, TextValue):
                raise AssertionError(  # pragma: no cover
                    f"PARSE_TEXT_THEN_DECODE expected TextValue, got {type(value).__name__}"
                )
            try:
                parsed = parse_json_strict(value.value)
            except StrictJsonParseError as exc:
                raise AglCastConversion(
                    f"Failed to parse text as JSON: {exc.message}",
                    source_label=recipe.source_label,
                    target_label=recipe.target_label,
                    raw=render_value(value),
                ) from exc
            return _decode_from_json(recipe, parsed, value)
        case ConversionStrategy.DECODE_JSON:
            if not isinstance(value, JsonValue):
                raise AssertionError(  # pragma: no cover
                    f"DECODE_JSON expected JsonValue, got {type(value).__name__}"
                )
            return _decode_from_json(recipe, value.raw, value)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _decode_from_json(recipe: ConversionRecipe, obj: object, value: Value) -> Value:
    """Normalize → JSON-Schema validate → decode, mirroring json_obj_to_value."""
    raw = render_value(value)
    normalized = normalize_integral_decimals(obj)

    if recipe.json_schema is None:  # pragma: no cover
        raise AssertionError("decode strategy requires a json_schema")
    errors = list(_validator_for(recipe.json_schema).iter_errors(normalized))
    if errors:
        msgs = "; ".join(_clean_validation_message(e) for e in errors)
        raise AglCastConversion(
            f"Schema validation failed: {msgs}",
            source_label=recipe.source_label,
            target_label=recipe.target_label,
            raw=raw,
        )

    if recipe.decode is None:  # pragma: no cover
        raise AssertionError("decode strategy requires a decode schema")
    try:
        return decode_value(recipe.decode, normalized)
    except ValueError as exc:
        raise AglCastConversion(
            f"Value conversion failed: {exc}",
            source_label=recipe.source_label,
            target_label=recipe.target_label,
            raw=raw,
        ) from exc


def decode_value(schema: DecodeSchema, obj: object) -> Value:
    """Construct a typed ``Value`` from JSON-shaped *obj* per *schema*.

    Mirrors ``runtime.convert.json_to_value`` (identical ``ValueError``
    messages) but walks the typeless ``DecodeSchema`` instead of a checker
    ``Type``.
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


# Transitional private name retained for the agent parser until M9 removes
# migration-era module boundaries.
_decode = decode_value


def _decode_scalar(kind: ScalarKind, obj: object) -> Value:
    """Decode a JSON scalar into the matching leaf ``Value`` (json_to_value parity)."""
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
