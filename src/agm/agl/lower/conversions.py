"""Conversion-recipe compiler for the AgL lowering phase.

``compile_recipe(source, target, kind)`` is the ONLY place that reads checker
``Type`` objects to produce a typeless ``ConversionRecipe`` for a cast.  Once
this returns, the conversion is fully pre-resolved: the evaluator switches on
the recipe's ``strategy`` and walks the typeless ``DecodeSchema`` / JSON schema
without ever sniffing checker types.

Strategy selection follows the cast matrix and the ``CastKind`` classification
(``semantics.types.cast_classification``):
total casts (``TOTAL_NOOP`` / ``TOTAL_RENDER`` / ``TOTAL_JSON``) never fail;
fallible casts (``decimal → int`` narrowing, ``text → T``, ``json → T``) carry
the derived JSON schema and the ``decode_value`` decode walk.

``derive_schema_and_decode`` lives in :mod:`agm.agl.type_schema` (alongside
``derive_schema``/``build_decode_schema``) so both the lowerer and the runtime
codec can import it without a cycle; it derives the JSON schema and the
typeless decode plan (``DecodePlan`` — a decode schema plus its ``$defs``
table for a recursive target type) from one shared recursion plan.

"""

from __future__ import annotations

import json
from typing import assert_never

from agm.agl.ir.contracts import ConversionRecipe, ConversionStrategy
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import CastKind, DecimalType, IntType, JsonType, TextType, Type
from agm.agl.type_schema import derive_schema_and_decode

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

    match kind:
        case CastKind.TOTAL_NOOP:
            # int → decimal is the only widening no-op; everything else returns
            # the value unchanged (identity / already-assignable).
            if isinstance(source, IntType) and isinstance(target, DecimalType):
                strategy = ConversionStrategy.WIDEN_INT_TO_DECIMAL
            else:
                strategy = ConversionStrategy.NOOP
            return ConversionRecipe(
                strategy=strategy, source_label=source_label, target_label=target_label
            )
        case CastKind.TOTAL_RENDER:
            return ConversionRecipe(
                strategy=ConversionStrategy.RENDER_TO_TEXT,
                source_label=source_label,
                target_label=target_label,
            )
        case CastKind.TOTAL_JSON:
            return ConversionRecipe(
                strategy=ConversionStrategy.TO_JSON,
                source_label=source_label,
                target_label=target_label,
            )
        case CastKind.FALLIBLE:
            if isinstance(source, DecimalType) and isinstance(target, IntType):
                strategy = ConversionStrategy.NARROW_DECIMAL_TO_INT
            elif isinstance(source, TextType):
                strategy = ConversionStrategy.PARSE_TEXT_THEN_DECODE
            else:
                # cast_classification only yields FALLIBLE for decimal→int or a
                # text/json source; the remaining case is a json source.
                assert isinstance(source, JsonType), f"unexpected fallible cast source {source!r}"
                strategy = ConversionStrategy.DECODE_JSON
            schema, decode_plan = derive_schema_and_decode(target, type_table)
            return ConversionRecipe(
                strategy=strategy,
                source_label=source_label,
                target_label=target_label,
                # Serialize the schema to a canonical JSON string so the recipe
                # stays hashable (sort_keys → deterministic recipe equality).
                json_schema=json.dumps(schema, sort_keys=True),
                decode=decode_plan.root,
                defs=decode_plan.defs,
            )
        case CastKind.STATIC_ERROR:  # pragma: no cover
            # The checker rejects statically-impossible casts before lowering.
            raise AssertionError(f"STATIC_ERROR cast reached lowering: {source!r} as {target!r}")
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
