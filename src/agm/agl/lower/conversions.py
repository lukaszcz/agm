"""Conversion-recipe compiler for the AgL lowering phase (M3e-2).

``compile_recipe(source, target, kind)`` is the ONLY place that reads checker
``Type`` objects to produce a typeless ``ConversionRecipe`` for a cast.  Once
this returns, the conversion is fully pre-resolved: the evaluator switches on
the recipe's ``strategy`` and walks the typeless ``DecodeSchema`` / JSON schema
without ever sniffing checker types.

Strategy selection mirrors ``runtime.convert.convert_value`` + the
``CastKind`` classification (``typecheck.types.cast_classification``):
total casts (``TOTAL_NOOP`` / ``TOTAL_RENDER`` / ``TOTAL_JSON``) never fail;
fallible casts (``decimal → int`` narrowing, ``text → T``, ``json → T``) carry
the derived JSON schema and decode walk.

Note (tracked for M9): ``derive_schema`` is a pure compile-time ``Type →
JSON-Schema`` transformation (it imports only ``typecheck.types``); importing
it here is cycle-free.  In the end-state layering it should live in a
lowering-adjacent / neutral location rather than under ``runtime``.
"""

from __future__ import annotations

import json
from typing import assert_never

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
    VariantDecode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.runtime.schema import derive_schema
from agm.agl.typecheck.types import (
    BoolType,
    CastKind,
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

__all__ = ["build_decode_schema", "compile_recipe"]


def build_decode_schema(typ: Type) -> DecodeSchema:
    """Compile a checker ``Type`` into a typeless ``DecodeSchema``.

    Mirrors the type recursion of ``runtime.convert.json_to_value`` so the
    evaluator can reconstruct the typed value without the checker ``Type``.
    """
    if isinstance(typ, TextType):
        return ScalarDecode(ScalarKind.TEXT)
    if isinstance(typ, IntType):
        return ScalarDecode(ScalarKind.INT)
    if isinstance(typ, DecimalType):
        return ScalarDecode(ScalarKind.DECIMAL)
    if isinstance(typ, BoolType):
        return ScalarDecode(ScalarKind.BOOL)
    if isinstance(typ, JsonType):
        return ScalarDecode(ScalarKind.JSON)
    if isinstance(typ, ListType):
        return ListDecode(build_decode_schema(typ.elem))
    if isinstance(typ, DictType):
        return DictDecode(build_decode_schema(typ.value))
    if isinstance(typ, RecordType):
        return RecordDecode(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            fields=tuple(
                (fname, build_decode_schema(ftype)) for fname, ftype in typ.fields.items()
            ),
        )
    if isinstance(typ, EnumType):
        return EnumDecode(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            variants=tuple(
                VariantDecode(
                    name=vname,
                    fields=tuple(
                        (fname, build_decode_schema(ftype)) for fname, ftype in vfields.items()
                    ),
                )
                for vname, vfields in typ.variants.items()
            ),
        )
    # Non-data targets (unit/agent/function/exception/bottom/typevar) are not
    # decodable from JSON and are rejected by the checker before lowering.
    raise AssertionError(  # pragma: no cover
        f"build_decode_schema: undecodable type {typ!r}"
    )


def compile_recipe(source: Type, target: Type, kind: CastKind) -> ConversionRecipe:
    """Compile a cast ``(source, target, kind)`` into a ``ConversionRecipe``."""
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
                assert isinstance(source, JsonType), (
                    f"unexpected fallible cast source {source!r}"
                )
                strategy = ConversionStrategy.DECODE_JSON
            return ConversionRecipe(
                strategy=strategy,
                source_label=source_label,
                target_label=target_label,
                # Serialize the schema to a canonical JSON string so the recipe
                # stays hashable (sort_keys → deterministic recipe equality).
                json_schema=json.dumps(derive_schema(target), sort_keys=True),
                decode=build_decode_schema(target),
            )
        case CastKind.STATIC_ERROR:  # pragma: no cover
            # The checker rejects statically-impossible casts before lowering.
            raise AssertionError(f"STATIC_ERROR cast reached lowering: {source!r} as {target!r}")
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
