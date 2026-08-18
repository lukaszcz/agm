"""Frontend-free cast/conversion executor for the AgL IR evaluator.

``run_recipe(recipe, value)`` executes a typeless ``ConversionRecipe`` against a
runtime ``Value`` and returns the converted ``Value``.  On an expected fallible
failure it raises the module-private ``AglCastConversion`` sentinel (carrying
the message + user-facing source/target labels + rendered raw value); the
caller wraps it into the appropriate ``CastError`` / ``BoolValue(False)``.

It reuses the existing runtime leaf primitives (rendering, JSON serialization,
strict parse, integral-decimal normalization, JSON-Schema validation, and the
typeless ``decode_value`` decode walk) rather than reimplementing them.

Imports: stdlib + ``agm.agl.semantics.values`` + ``agm.agl.ir``
contracts + ``agm.agl.runtime`` leaf helpers.  No ``syntax`` / ``scope`` /
``typecheck`` imports are permitted here.
"""

from __future__ import annotations

from decimal import Decimal
from typing import assert_never

from agm.agl.ir.contracts import (
    ConversionRecipe,
    ConversionStrategy,
)
from agm.agl.runtime.convert import (
    StrictJsonParseError,
    _clean_validation_message,
    decode_value,
    normalize_integral_decimals,
    parse_json_strict,
    validator_for_schema,
)
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.semantics.values import (
    DecimalValue,
    IntValue,
    JsonValue,
    TextValue,
    Value,
)

__all__ = ["AglCastConversion", "run_recipe"]


class AglCastConversion(Exception):
    """Sentinel: a fallible cast conversion failed for an expected reason.

    Mirrors the field set of the legacy ``CastError`` so the IR evaluator can
    build an identical exception value.
    """

    def __init__(self, message: str, *, source_label: str, target_label: str, raw: str) -> None:
        super().__init__(message)
        self.message = message
        self.source_label = source_label
        self.target_label = target_label
        self.raw = raw


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
    """Normalize → JSON-Schema validate → decode."""
    normalized = normalize_integral_decimals(obj)

    if recipe.json_schema is None:  # pragma: no cover
        raise AssertionError("decode strategy requires a json_schema")
    errors = list(validator_for_schema(recipe.json_schema).iter_errors(normalized))
    if errors:
        msgs = "; ".join(_clean_validation_message(e) for e in errors)
        raise AglCastConversion(
            f"Schema validation failed: {msgs}",
            source_label=recipe.source_label,
            target_label=recipe.target_label,
            raw=render_value(value),
        )

    if recipe.decode is None:  # pragma: no cover
        raise AssertionError("decode strategy requires a decode schema")
    try:
        return decode_value(recipe.decode, normalized, dict(recipe.defs))
    except ValueError as exc:
        raise AglCastConversion(
            f"Value conversion failed: {exc}",
            source_label=recipe.source_label,
            target_label=recipe.target_label,
            raw=render_value(value),
        ) from exc
