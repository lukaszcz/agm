"""Coercion compiler for the AgL lowering phase.

``compile_coercion(source, target, type_table)`` is the ONLY place that reads
checker ``Type`` objects to produce a ``Coercion`` descriptor.  Once this
function returns, the coercion is fully pre-resolved; the evaluator only
switches on the returned ``Coercion`` union and never sniffs value types at
runtime.

Ordering follows the contract exactly (mirrors legacy eval/interpreter._coerce):
  1. target is JsonType and source is not JsonType → ToJson
  2. target is DecimalType and source is IntType → IntToDecimal
  3. both ListType → recurse on elem; MapList if child is not None
  4. both DictType → recurse on value; MapDictValues if child is not None
  5. both RecordType → per shared field; MapRecordFields if any
  6. both EnumType → per variant/field; MapEnumFields if any
  7. otherwise → None (identity / opaque / no implicit coercion)

TypeVarType sources or targets, equal types, and the json→json identity all
return None.  Record/enum field and variant shapes are resolved through the
shared ``TypeTable`` (``table.record_fields``/``table.enum_variants``) rather
than the handle's own embedded maps.
"""

from __future__ import annotations

from agm.agl.ir.operations import (
    Coercion,
    IntToDecimal,
    MapDictValues,
    MapEnumFields,
    MapList,
    MapRecordFields,
    ToJson,
)
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import (
    DecimalType,
    DictType,
    EnumType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    Type,
    TypeVarType,
)

__all__ = ["compile_coercion"]


def compile_coercion(source: Type, target: Type, type_table: TypeTable) -> Coercion | None:
    """Compile an implicit coercion from *source* to *target*.

    Returns a ``Coercion`` descriptor to be wrapped in an ``IrCoerce`` node, or
    ``None`` when no coercion node is needed (identity / opaque / equal types).
    *type_table* resolves record/enum field and variant shapes.

    Ordering (mirrors legacy eval/interpreter._coerce):
      1. target is JsonType and source is not JsonType → ToJson
      2. target is DecimalType and source is IntType → IntToDecimal
      3. both ListType → recurse on elem; MapList if child is not None
      4. both DictType → recurse on value; MapDictValues if child is not None
      5. both RecordType → per shared field; MapRecordFields if any
      6. both EnumType → per variant/field; MapEnumFields if any
      7. otherwise → None

    Note: RecordType and EnumType equality is nominal (name + module_id +
    type_args; fields/variants are excluded from __eq__).  Therefore we handle
    records and enums BEFORE the ``source == target`` short-circuit, so that
    two nominally-equal types with different field types (possible for generic
    instantiations) still get their field coercions compiled correctly.
    """
    # Opaque: type variables can carry any value — no compile-time coercion.
    if isinstance(source, TypeVarType) or isinstance(target, TypeVarType):
        return None

    # 5. Both record → per shared field (before equality check; see note above).
    if isinstance(target, RecordType) and isinstance(source, RecordType):
        src_fields = type_table.record_fields(source)
        tgt_fields = type_table.record_fields(target)
        field_ops: list[tuple[str, Coercion]] = []
        for field_name, tgt_field_type in tgt_fields.items():
            src_field_type = src_fields.get(field_name)
            if src_field_type is None:
                continue
            child = compile_coercion(src_field_type, tgt_field_type, type_table)
            if child is not None:
                field_ops.append((field_name, child))
        return MapRecordFields(tuple(field_ops)) if field_ops else None

    # 6. Both enum → per variant, per field (before equality check).
    if isinstance(target, EnumType) and isinstance(source, EnumType):
        src_variants = type_table.enum_variants(source)
        tgt_variants = type_table.enum_variants(target)
        variant_ops: list[tuple[str, tuple[tuple[str, Coercion], ...]]] = []
        for variant_name, tgt_vfields in tgt_variants.items():
            src_vfields = src_variants.get(variant_name, {})
            field_ops_v: list[tuple[str, Coercion]] = []
            for field_name, tgt_ftype in tgt_vfields.items():
                src_ftype = src_vfields.get(field_name)
                if src_ftype is None:
                    continue
                child = compile_coercion(src_ftype, tgt_ftype, type_table)
                if child is not None:
                    field_ops_v.append((field_name, child))
            if field_ops_v:
                variant_ops.append((variant_name, tuple(field_ops_v)))
        return MapEnumFields(tuple(variant_ops)) if variant_ops else None

    # Equal types: identity, no coercion.  This subsumes json→json and all
    # other same-primitive-type pairs.
    if source == target:
        return None

    # 1. Target is json and source is not json → wrap in ToJson.
    if isinstance(target, JsonType):
        # source != target so source is not JsonType (equal check above)
        return ToJson()

    # 2. Target is decimal and source is int → widen.
    if isinstance(target, DecimalType) and isinstance(source, IntType):
        return IntToDecimal()

    # 3. Both list → recurse on element type.
    if isinstance(target, ListType) and isinstance(source, ListType):
        child = compile_coercion(source.elem, target.elem, type_table)
        return MapList(child) if child is not None else None

    # 4. Both dict → recurse on value type.
    if isinstance(target, DictType) and isinstance(source, DictType):
        child = compile_coercion(source.value, target.value, type_table)
        return MapDictValues(child) if child is not None else None

    # 7. Otherwise: no implicit coercion.
    return None
