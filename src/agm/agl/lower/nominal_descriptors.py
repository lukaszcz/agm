"""Shared ``NominalDescriptor``/``VariantDescriptor`` builders.

Both whole-program linking (``lower/program.py``, from a checked module's own
``TypeDef``) and reserved built-in registration (``lower/lowerer.py``, from a
prelude/enum/exception ``TypeDef`` looked up on the same ``TypeTable``) build
the same descriptor shapes from declaration data: :func:`exception_descriptor`
sets the ``base`` link (the runtime mirror of ``TypeTable.ancestor_defs``;
records and enums have none), :func:`enum_descriptor` builds its
``VariantDescriptor`` tuple via :func:`variant_descriptors`, and
:func:`record_descriptor`, :func:`exception_descriptor`, and
:func:`_variant_descriptor` all get their field/JSON-name pairing from
:func:`json_field_names`, computed in one place for all three.
"""

from __future__ import annotations

from collections.abc import Mapping

from agm.agl.ir.ids import NominalId
from agm.agl.ir.nodes import IrExpr
from agm.agl.ir.program import NominalDescriptor, NominalKind, VariantDescriptor
from agm.agl.semantics.arguments import positional_field_names
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import EnumType, ExceptionType, RecordType

__all__ = [
    "enum_descriptor",
    "exception_descriptor",
    "json_field_names",
    "record_descriptor",
    "variant_descriptors",
]


def json_field_names(
    type_table: TypeTable, handle: RecordType | ExceptionType
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return *handle*'s field names and their effective JSON names, both in declaration order.

    The one place ``TypeTable.json_fields`` is split into the parallel
    ``fields``/``field_json_names`` pair every ``NominalDescriptor`` and
    ``VariantDescriptor`` carries; ``type_schema._emit_field_encodes`` reads
    ``TypeTable.json_fields`` directly for ``FieldEncode``.
    """
    json_fields = type_table.json_fields(handle)
    return (
        tuple(fname for fname, _json_name, _ftype in json_fields),
        tuple(json_name for _name, json_name, _ftype in json_fields),
    )


def _variant_descriptor(name: str, member: RecordType, type_table: TypeTable) -> VariantDescriptor:
    """Build one enum member's ``VariantDescriptor``, fields JSON-keyed like its own descriptor."""
    fields, field_json_names = json_field_names(type_table, member)
    return VariantDescriptor(
        name=name,
        fields=fields,
        member=NominalId(member.decl_id),
        json_name=type_table.member_json_tag(member, name),
        field_json_names=field_json_names,
    )


def variant_descriptors(handle: EnumType, type_table: TypeTable) -> tuple[VariantDescriptor, ...]:
    """Build every member's ``VariantDescriptor`` for enum *handle*, in declaration order."""
    return tuple(
        _variant_descriptor(name, member, type_table)
        for name, member in type_table.enum_member_names(handle).items()
    )


def exception_descriptor(
    typedef: TypeDef,
    handle: ExceptionType,
    type_table: TypeTable,
    *,
    bears_name_path: bool,
    field_defaults: Mapping[NominalId, "tuple[IrExpr | None, ...]"],
) -> NominalDescriptor:
    """Build one exception descriptor from its authoritative declaration.

    *field_defaults* maps a declaration's own identity to its lowered
    per-field defaults (see ``_LinkState.field_defaults``); flattened base
    first across the ``extends`` chain — like ``fields`` itself — so an
    inherited field keeps its base's lowered default unless redeclared.
    A chain link absent from *field_defaults* contributes an
    all-``None`` run.
    """
    nominal = NominalId(typedef.decl_node_id)
    fields, field_json_names = json_field_names(type_table, handle)
    chain = type_table.exception_chain_defs(typedef.decl_node_id)
    defaults_by_field: dict[str, IrExpr | None] = {}
    for chain_typedef in chain:
        own_defaults = field_defaults.get(
            NominalId(chain_typedef.decl_node_id), (None,) * len(chain_typedef.fields)
        )
        defaults_by_field.update(
            zip((name for name, _ in chain_typedef.fields), own_defaults, strict=True)
        )
    flattened_defaults = tuple(defaults_by_field.values())
    return NominalDescriptor(
        nominal=nominal,
        module_id=typedef.module_id,
        scope_path=typedef.scope_path,
        declared_name=typedef.name,
        kind=NominalKind.EXCEPTION,
        base=NominalId(typedef.base) if typedef.base is not None else None,
        fields=fields,
        field_json_names=field_json_names,
        field_defaults=flattened_defaults,
        variants=(),
        positional_fields=positional_field_names(type_table.field_kinds(handle)),
        bears_name_path=bears_name_path,
    )


def enum_descriptor(
    typedef: TypeDef,
    handle: EnumType,
    type_table: TypeTable,
    *,
    bears_name_path: bool,
) -> NominalDescriptor:
    """Build one enum descriptor from its authoritative declaration."""
    return NominalDescriptor(
        nominal=NominalId(typedef.decl_node_id),
        module_id=typedef.module_id,
        scope_path=typedef.scope_path,
        declared_name=typedef.name,
        kind=NominalKind.ENUM,
        variants=variant_descriptors(handle, type_table),
        bears_name_path=bears_name_path,
    )


def record_descriptor(
    typedef: TypeDef,
    handle: RecordType,
    type_table: TypeTable,
    *,
    bears_name_path: bool,
    field_defaults: Mapping[NominalId, tuple[IrExpr | None, ...]],
) -> NominalDescriptor:
    """Build one record descriptor from its authoritative declaration."""
    nominal = NominalId(typedef.decl_node_id)
    fields, field_json_names = json_field_names(type_table, handle)
    return NominalDescriptor(
        nominal=nominal,
        module_id=typedef.module_id,
        scope_path=typedef.scope_path,
        declared_name=typedef.name,
        kind=NominalKind.RECORD,
        fields=fields,
        field_json_names=field_json_names,
        mutable_fields=typedef.mutable_fields,
        variants=(),
        positional_fields=positional_field_names(type_table.field_kinds(handle)),
        field_defaults=field_defaults.get(nominal, ()),
        bears_name_path=bears_name_path,
    )
