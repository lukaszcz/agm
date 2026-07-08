"""Shared helpers for AgL test modules.

Provides a recursive ``node_id`` collector used by the seeded parsing and
seeded type-checking tests, plus ``ambient_agents_for`` — used by non-scope
unit tests (typecheck/eval/codec/trace) to resolve programs that *call* named
agents without forcing an explicit ``agent`` declaration in every test source.
The agent-declaration RULE itself is exercised by ``tests/test_agl_scope.py``
and the e2e suite; these other modules only need the calls to bind.

``type_table_for`` is the shared helper for tests that build ad-hoc
``RecordType``/``EnumType`` handles directly (rather than through the real
type builder, which populates the ``TypeTable``): since a handle carries no
field/variant data of its own, it registers every given ``TypeDef`` — one per
ad-hoc nominal type the test constructs, including any nested inside another
one's field/variant templates — into a fresh seeded table, so
``derive_schema``/``build_decode_schema``/``compile_coercion``/etc. resolve
field and variant shapes exactly as specified.

``record_type``/``enum_type`` are convenience factories that build an ad-hoc
handle and its matching ``TypeDef`` together, in one call, for tests that
need both (the handle to pass to the function under test, the ``TypeDef`` to
pass to ``type_table_for``).
"""

from __future__ import annotations

import dataclasses

from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.semantics.type_table import TypeDef, TypeTable, create_seeded_type_table
from agm.agl.semantics.types import EnumType, RecordType, Type
from agm.agl.syntax.nodes import Program


def all_node_ids(obj: object, seen: set[int] | None = None) -> set[int]:
    """Recursively collect every ``node_id`` reachable from *obj*."""
    if seen is None:
        seen = set()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        nid = getattr(obj, "node_id", None)
        if isinstance(nid, int):
            seen.add(nid)
        for f in dataclasses.fields(obj):
            all_node_ids(getattr(obj, f.name), seen)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            all_node_ids(item, seen)
    return seen


def ambient_agents_for(program: Program) -> frozenset[str]:
    """Return an empty frozenset — agent names must be declared via 'agent' in AgL."""
    return frozenset()


def type_table_for(*defs: TypeDef) -> TypeTable:
    """Return a fresh seeded ``TypeTable`` with every given ``TypeDef`` registered.

    Callers building an ad-hoc ``RecordType``/``EnumType`` handle for a test
    (rather than through the real type builder) pass its ``TypeDef`` here —
    including the ``TypeDef`` of any OTHER ad-hoc type nested inside a
    field/variant template (e.g. an outer record embedding an inner one) —
    since a handle carries no shape data of its own for this helper to
    discover automatically.
    """
    table = create_seeded_type_table()
    for typedef in defs:
        table.register(typedef)
    return table


def record_type(
    name: str,
    fields: dict[str, Type],
    *,
    type_args: tuple[Type, ...] = (),
    module_id: ModuleId = ENTRY_ID,
    type_params: tuple[str, ...] = (),
) -> tuple[RecordType, TypeDef]:
    """Build an ad-hoc ``RecordType`` handle and its matching ``TypeDef`` together.

    Returns ``(handle, typedef)``; pass ``typedef`` (and the ``TypeDef`` of
    any nested ad-hoc nominal type referenced in *fields*) to
    :func:`type_table_for` so the handle's field shape resolves.
    """
    typedef = TypeDef(
        kind="record",
        name=name,
        module_id=module_id,
        type_params=type_params,
        fields=tuple(fields.items()),
    )
    return typedef.handle(type_args), typedef


def enum_type(
    name: str,
    variants: dict[str, dict[str, Type]],
    *,
    type_args: tuple[Type, ...] = (),
    module_id: ModuleId = ENTRY_ID,
    type_params: tuple[str, ...] = (),
) -> tuple[EnumType, TypeDef]:
    """Build an ad-hoc ``EnumType`` handle and its matching ``TypeDef`` together.

    See :func:`record_type`.
    """
    typedef = TypeDef(
        kind="enum",
        name=name,
        module_id=module_id,
        type_params=type_params,
        variants=tuple((vname, tuple(vfields.items())) for vname, vfields in variants.items()),
    )
    return typedef.handle(type_args), typedef
