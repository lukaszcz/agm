"""Shared ``NominalDescriptor`` builder for exception declarations.

Both whole-program linking (``lower/program.py``, from a checked module's own
``TypeDef``) and reserved built-in registration (``lower/lowerer.py``, from
``semantics.type_table.BUILTIN_EXCEPTION_TYPE_DEFS`` via
``TypeTable.exception_def``) build the same descriptor shape from a
``TypeDef``/``ExceptionType`` pair. One builder keeps the ``base`` link logic
(the runtime mirror of ``TypeTable.ancestor_defs``) in one place.
"""

from __future__ import annotations

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.semantics.arguments import positional_field_names
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import ExceptionType

__all__ = ["exception_descriptor"]


def exception_descriptor(
    typedef: TypeDef,
    handle: ExceptionType,
    type_table: TypeTable,
    *,
    bears_name_path: bool,
) -> NominalDescriptor:
    """Build one exception descriptor from its authoritative declaration."""
    nominal = NominalId(typedef.decl_node_id)
    return NominalDescriptor(
        nominal=nominal,
        module_id=typedef.module_id,
        scope_path=typedef.scope_path,
        declared_name=typedef.name,
        kind=NominalKind.EXCEPTION,
        base=NominalId(typedef.base) if typedef.base is not None else None,
        fields=tuple(type_table.exception_fields(handle).keys()),
        variants=(),
        positional_fields=positional_field_names(type_table.field_kinds(handle)),
        bears_name_path=bears_name_path,
    )
