"""Conversion-recipe compiler for the AgL lowering phase.

``compile_recipe(source, target, kind)`` is the ONLY place that reads checker
``Type`` objects to produce a typeless ``ConversionRecipe`` for a cast.  Once
this returns, the conversion is fully pre-resolved: the evaluator switches on
the recipe's ``strategy`` and walks the typeless ``DecodeSchema`` / JSON schema
without ever sniffing checker types.

Strategy selection follows the cast matrix and the ``CastKind`` classification
(``semantics.type_table.cast_classification``):
total casts (``TOTAL_NOOP`` / ``TOTAL_RENDER`` / ``TOTAL_JSON``) never fail;
finite JSON sources carry a static encode plan, while a statically
JSON-convertible growing polymorphic-recursive source gets an explicit
generic-template strategy because it has no finite concrete plan. Fallible casts
(``decimal → int`` narrowing, ``text → T``, ``json → T``) carry
the derived JSON schema and the ``decode_value`` decode walk.

``derive_schema_and_decode`` lives in :mod:`agm.agl.type_schema` so both the lowerer and the runtime
codec can import it without a cycle; it derives the JSON schema and the
typeless decode plan (``DecodePlan`` — a decode schema plus its ``$defs``
table for a recursive target type) from one shared recursion plan.

"""

from __future__ import annotations

import json
from typing import assert_never

from agm.agl.ir.contracts import (
    ConversionRecipe,
    ConversionStrategy,
    DecodeConversionKind,
    DecodeConversionRecipe,
    SimpleConversionKind,
    SimpleConversionRecipe,
    ToJsonRecipe,
)
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import (
    BottomType,
    CastKind,
    DecimalType,
    IntType,
    JsonType,
    TextType,
    Type,
)
from agm.agl.type_schema import (
    build_encode_plan,
    derive_schema_and_decode,
)

__all__ = ["compile_recipe"]


def compile_recipe(
    source: Type, target: Type, kind: CastKind, type_table: TypeTable
) -> ConversionRecipe:
    """Compile a cast ``(source, target, kind)`` into a ``ConversionRecipe``.

    *type_table* resolves record/enum field/variant shapes for the fallible
    branch's ``derive_schema_and_decode`` call.
    """
    source_label = repr(source)
    target_label = repr(target)

    # Bottom never supplies a runtime value. Its cast may have been classified
    # against an expected target type, so make the unreachable conversion
    # explicit before any target-specific planning.
    if isinstance(source, BottomType):
        return SimpleConversionRecipe(
            strategy=ConversionStrategy.NOOP,
            source_label=source_label,
            target_label=target_label,
        )

    match kind:
        case CastKind.TOTAL_NOOP:
            # int → decimal is the only widening no-op; everything else returns
            # the value unchanged (identity / already-assignable).
            strategy: SimpleConversionKind
            if isinstance(source, IntType) and isinstance(target, DecimalType):
                strategy = ConversionStrategy.WIDEN_INT_TO_DECIMAL
            else:
                strategy = ConversionStrategy.NOOP
            return SimpleConversionRecipe(
                strategy=strategy, source_label=source_label, target_label=target_label
            )
        case CastKind.TOTAL_RENDER:
            return SimpleConversionRecipe(
                strategy=ConversionStrategy.RENDER_TO_TEXT,
                source_label=source_label,
                target_label=target_label,
            )
        case CastKind.TOTAL_JSON:
            # A growing polymorphic-recursive source is still known to have a
            # JSON representation even though no finite set of concrete
            # instantiations covers it; build_encode_plan picks the plan shape.
            encode_plan = build_encode_plan(source, type_table)
            return ToJsonRecipe(
                source_label=source_label,
                target_label=target_label,
                encode=encode_plan.root,
                encode_definitions=encode_plan.definitions,
            )
        case CastKind.FALLIBLE:
            decode_strategy: DecodeConversionKind
            if isinstance(source, DecimalType) and isinstance(target, IntType):
                decode_strategy = ConversionStrategy.NARROW_DECIMAL_TO_INT
            elif isinstance(source, TextType):
                decode_strategy = ConversionStrategy.PARSE_TEXT_THEN_DECODE
            else:
                # cast_classification only yields FALLIBLE for decimal→int or a
                # text/json source; the remaining case is a json source.
                assert isinstance(source, JsonType), f"unexpected fallible cast source {source!r}"
                decode_strategy = ConversionStrategy.DECODE_JSON
            schema, decode_plan = derive_schema_and_decode(target, type_table)
            return DecodeConversionRecipe(
                strategy=decode_strategy,
                source_label=source_label,
                target_label=target_label,
                # Serialize the schema to a canonical JSON string so the recipe
                # stays hashable (sort_keys → deterministic recipe equality).
                json_schema=json.dumps(schema, sort_keys=True),
                decode=decode_plan.root,
                defs=decode_plan.defs,
            )
        case CastKind.IDENTITY_UPCAST:
            return SimpleConversionRecipe(
                strategy=ConversionStrategy.NOOP,
                source_label=source_label,
                target_label=target_label,
            )
        case CastKind.NOMINAL_DOWNCAST:  # pragma: no cover
            raise AssertionError(
                f"nominal downcast reached recipe compilation: {source!r} as {target!r}"
            )
        case CastKind.STATIC_ERROR:  # pragma: no cover
            # The checker rejects statically-impossible casts before lowering.
            raise AssertionError(f"STATIC_ERROR cast reached lowering: {source!r} as {target!r}")
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
