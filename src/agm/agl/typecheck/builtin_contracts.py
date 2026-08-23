"""Location-independent structural contracts for host-known builtin types.

The seeded ``std/core`` ``TypeDef`` objects are real nominal declarations used
by type resolution and runtime values.  They are not validation schemas: a
source ``builtin`` declaration may live in another module or scope, and an enum
may satisfy a member contract with a referenced record whose declaration path
necessarily differs from the canonical inline member's path.

This module projects nominal declarations into explicit contracts.  A contract
keeps declaration kind, parameters, fields, member names, captured member type
arguments and member fields, exception metadata, and normalized field/base type
references.  It deliberately does not keep the declaration location,
declaration identity, or enum-member record identity.  Validation can therefore
remain structural without changing the identity or path of the declaration
being validated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agm.agl.modules.ids import STD_CORE_ID
from agm.agl.semantics.type_table import (
    BUILTIN_EXCEPTION_TYPE_DEFS,
    BUILTIN_PRELUDE_TYPE_DEFS,
    OPTION_TYPE_DEF,
    TypeDef,
    TypeDefKind,
    TypeTable,
    create_seeded_type_table,
)
from agm.agl.semantics.types import ExceptionType, Type, reroot_type


@dataclass(frozen=True, slots=True)
class BuiltinMemberContract:
    """One enum member's location-independent name and record payload."""

    name: str
    type_args: tuple[Type, ...]
    fields: tuple[tuple[str, Type], ...]


@dataclass(frozen=True, slots=True)
class BuiltinTypeContract:
    """The structural host contract a source ``builtin`` type must satisfy."""

    kind: TypeDefKind
    name: str
    type_params: tuple[str, ...]
    fields: tuple[tuple[str, Type], ...]
    members: tuple[BuiltinMemberContract, ...]
    abstract: bool
    base: ExceptionType | None
    field_kinds: tuple[str, ...]


def contract_for_typedef(
    typedef: TypeDef,
    table: TypeTable,
    *,
    base_type: ExceptionType | None,
) -> BuiltinTypeContract:
    """Project *typedef* into its location-independent builtin contract.

    The declaration and its member records keep their real nominal identities
    in ``TypeTable``.  Only type references inside contract-bearing fields and
    the exception base are normalized onto the host contract namespace.  This
    lets a scoped builtin name a sibling scoped builtin while preserving a
    reference to an unrelated module as a genuine mismatch.
    """
    remap = (typedef.module_id, STD_CORE_ID)

    def normalize(typ: Type) -> Type:
        return reroot_type(typ, typedef.scope_path, remap_module=remap)

    fields = tuple((name, normalize(field_type)) for name, field_type in typedef.fields)
    members = tuple(
        BuiltinMemberContract(
            member.name,
            tuple(normalize(type_arg) for type_arg in member.type_args),
            tuple(
                (name, normalize(field_type))
                for name, field_type in table.record_fields(member).items()
            ),
        )
        for member in typedef.members
    )
    normalized_base = None if base_type is None else normalize(base_type)
    assert normalized_base is None or isinstance(normalized_base, ExceptionType)
    return BuiltinTypeContract(
        kind=typedef.kind,
        name=typedef.name,
        type_params=typedef.type_params,
        fields=fields,
        members=members,
        abstract=typedef.abstract,
        base=normalized_base,
        field_kinds=typedef.field_kinds,
    )


def _canonical_contracts(
    definitions: Mapping[str, TypeDef], table: TypeTable
) -> Mapping[str, BuiltinTypeContract]:
    """Project seeded definitions once; their nominal paths remain runtime-only."""
    contracts: dict[str, BuiltinTypeContract] = {}
    for name, typedef in definitions.items():
        base_type: ExceptionType | None = None
        if typedef.base is not None:
            base_def = table.get_by_id(typedef.base)
            assert base_def is not None
            base_handle = base_def.handle()
            assert isinstance(base_handle, ExceptionType)
            base_type = base_handle
        contracts[name] = contract_for_typedef(typedef, table, base_type=base_type)
    return contracts


_CANONICAL_TABLE = create_seeded_type_table()
_PRELUDE_CONTRACTS = _canonical_contracts(
    {**BUILTIN_PRELUDE_TYPE_DEFS, "Option": OPTION_TYPE_DEF}, _CANONICAL_TABLE
)

BUILTIN_RECORD_CONTRACTS: Mapping[str, BuiltinTypeContract] = {
    name: contract for name, contract in _PRELUDE_CONTRACTS.items() if contract.kind == "record"
}

BUILTIN_ENUM_CONTRACTS: Mapping[str, BuiltinTypeContract] = {
    name: contract for name, contract in _PRELUDE_CONTRACTS.items() if contract.kind == "enum"
}

BUILTIN_EXCEPTION_CONTRACTS: Mapping[str, BuiltinTypeContract] = _canonical_contracts(
    BUILTIN_EXCEPTION_TYPE_DEFS, _CANONICAL_TABLE
)
