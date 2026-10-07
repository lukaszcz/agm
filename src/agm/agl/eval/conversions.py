"""Frontend-free cast/conversion executor for the AgL IR evaluator.

``run_recipe(recipe, value)`` executes a typeless ``ConversionRecipe`` against a
runtime ``Value`` and returns the converted ``Value``.  On an expected fallible
failure it raises the module-private ``AglCastConversion`` sentinel (carrying
the message + user-facing source/target labels + rendered raw value); the
caller wraps it into the appropriate ``CastError`` / ``ValueParseError`` /
``BoolValue(False)``.  ``WIDEN_INT_TO_DECIMAL`` instead raises
``AglArithmeticSignal`` when the int is too large for the pinned decimal
context (see ``semantics.arithmetic.int_to_decimal``); the caller converts it
to the catchable ``ArithmeticError``.

It reuses the existing runtime leaf primitives (rendering, JSON serialization,
the host text/value-syntax decode boundary, JSON-Schema validation, and the
typeless ``decode_value`` decode walk) rather than reimplementing them.

Imports: stdlib + ``agm.agl.semantics.values`` + ``agm.agl.semantics.arithmetic``
+ ``agm.agl.ir`` contracts + ``agm.agl.runtime`` leaf helpers (including the
host text/value-syntax decode boundary).  No ``syntax`` / ``scope`` /
``typecheck`` imports are permitted here.
"""

from __future__ import annotations

from typing import assert_never, cast

from agm.agl.ir.contracts import (
    ConversionRecipe,
    ConversionStrategy,
    DecodeConversionRecipe,
    EncodePlan,
    SimpleConversionRecipe,
    ToJsonRecipe,
)
from agm.agl.ir.program import ValueDescriptors
from agm.agl.runtime.convert import (
    DefaultResolver,
    _clean_validation_message,
    decode_value,
    validator_for_schema,
)
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import encode_value
from agm.agl.runtime.value_decode import host_text_to_json
from agm.agl.semantics.arithmetic import int_to_decimal
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


def run_recipe(
    recipe: ConversionRecipe,
    value: Value,
    descriptors: ValueDescriptors,
    *,
    default_resolver: DefaultResolver | None = None,
) -> Value:
    """Execute *recipe* against *value*; raise ``AglCastConversion`` on failure.

    An ``as json`` encode instead propagates the serializer's own
    ``AglCyclicValue``/``AglNonDataValue`` sentinels for the caller to map.
    """
    match recipe:
        case SimpleConversionRecipe(strategy=simple_strategy):
            match simple_strategy:
                case ConversionStrategy.NOOP:
                    return value
                case ConversionStrategy.WIDEN_INT_TO_DECIMAL:
                    value = cast(IntValue, value)
                    return DecimalValue(int_to_decimal(value.value))
                case ConversionStrategy.RENDER_TO_TEXT:
                    return TextValue(render_value(value, descriptors))
                case _ as unreachable_simple:  # pragma: no cover
                    assert_never(unreachable_simple)
        case ToJsonRecipe(encode=encode, encode_definitions=encode_definitions):
            return JsonValue(
                encode_value(
                    EncodePlan(encode, encode_definitions),
                    value,
                    descriptors.exception_field_encodes,
                )
            )
        case DecodeConversionRecipe(strategy=decode_strategy):
            match decode_strategy:
                case ConversionStrategy.NARROW_DECIMAL_TO_INT:
                    value = cast(DecimalValue, value)
                    return _decode_from_json(
                        recipe, value.value, value, descriptors, default_resolver
                    )
                case ConversionStrategy.PARSE_TEXT_THEN_DECODE:
                    value = cast(TextValue, value)
                    try:
                        parsed = host_text_to_json(
                            value.value,
                            recipe.decode,
                            dict(recipe.defs),
                            agent_command_fallback=False,
                        )
                    except ValueError as exc:
                        raise AglCastConversion(
                            f"Failed to parse text: {exc}",
                            source_label=recipe.source_label,
                            target_label=recipe.target_label,
                            raw=render_value(value, descriptors),
                        ) from exc
                    return _decode_from_json(recipe, parsed, value, descriptors, default_resolver)
                case ConversionStrategy.DECODE_JSON:
                    value = cast(JsonValue, value)
                    return _decode_from_json(
                        recipe, value.raw, value, descriptors, default_resolver
                    )
                case _ as unreachable_decode:  # pragma: no cover
                    assert_never(unreachable_decode)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _decode_from_json(
    recipe: DecodeConversionRecipe,
    obj: object,
    value: Value,
    descriptors: ValueDescriptors,
    default_resolver: DefaultResolver | None,
) -> Value:
    """JSON-Schema validate → decode."""
    errors = list(validator_for_schema(recipe.json_schema).iter_errors(obj))
    if errors:
        msgs = "; ".join(_clean_validation_message(e) for e in errors)
        raise AglCastConversion(
            f"Schema validation failed: {msgs}",
            source_label=recipe.source_label,
            target_label=recipe.target_label,
            raw=render_value(value, descriptors),
        )

    try:
        return decode_value(
            recipe.decode, obj, dict(recipe.defs), default_resolver=default_resolver
        )
    except ValueError as exc:
        raise AglCastConversion(
            f"Value conversion failed: {exc}",
            source_label=recipe.source_label,
            target_label=recipe.target_label,
            raw=render_value(value, descriptors),
        ) from exc
