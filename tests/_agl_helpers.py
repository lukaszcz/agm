"""Shared helpers for AgL test modules.

Provides a recursive ``node_id`` collector used by the seeded parsing and
seeded type-checking tests, plus ``ambient_agents_for`` — used by non-scope
unit tests (typecheck/eval/codec/trace) to resolve programs that *call* named
agents without forcing an explicit ``agent`` declaration in every test source.
The agent-declaration RULE itself is exercised by ``tests/test_agl_scope.py``
and the e2e suite; these other modules only need the calls to bind.

``type_table_for`` is the shared helper for tests that build ad-hoc
``RecordType``/``EnumType`` handles directly (rather than through the real
type builder, which dual-writes the ``TypeTable``): it registers a matching
``TypeDef`` for every nominal type reachable from the given handles into a
fresh seeded table, so ``derive_schema``/``build_decode_schema``/
``compile_coercion``/etc. resolve field and variant shapes exactly as the
handles' own embedded maps already specify.
"""

from __future__ import annotations

import dataclasses

from agm.agl.semantics.type_table import TypeDef, TypeTable, create_seeded_type_table
from agm.agl.semantics.types import DictType, EnumType, ListType, RecordType, Type
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


def _register_ad_hoc_type_def(table: TypeTable, typ: RecordType | EnumType) -> None:
    """Register a ``TypeDef`` in *table* matching *typ*'s own embedded shape.

    Uses *typ*'s own ``module_id``/``name`` as the key and its embedded
    ``fields``/``variants`` verbatim as the template (no generic type
    parameters) — exactly what ``table.record_fields``/``table.enum_variants``
    need to reproduce the handle's own shape.  Idempotent: registering the
    same key with an identical shape twice (e.g. the same ad-hoc type built by
    two different test calls) is a no-op.
    """
    if isinstance(typ, RecordType):
        table.register(
            TypeDef(
                kind="record",
                name=typ.name,
                module_id=typ.module_id,
                fields=tuple(typ.fields.items()),
            )
        )
    else:
        table.register(
            TypeDef(
                kind="enum",
                name=typ.name,
                module_id=typ.module_id,
                variants=tuple(
                    (vname, tuple(vfields.items())) for vname, vfields in typ.variants.items()
                ),
            )
        )


def _walk_and_register(table: TypeTable, typ: Type, seen: set[tuple[object, str]]) -> None:
    """Recursively register every record/enum reachable from *typ* into *table*."""
    if isinstance(typ, RecordType):
        key = (typ.module_id, typ.name)
        if key in seen:
            return
        seen.add(key)
        _register_ad_hoc_type_def(table, typ)
        for ftype in typ.fields.values():
            _walk_and_register(table, ftype, seen)
    elif isinstance(typ, EnumType):
        key = (typ.module_id, typ.name)
        if key in seen:
            return
        seen.add(key)
        _register_ad_hoc_type_def(table, typ)
        for vfields in typ.variants.values():
            for ftype in vfields.values():
                _walk_and_register(table, ftype, seen)
    elif isinstance(typ, ListType):
        _walk_and_register(table, typ.elem, seen)
    elif isinstance(typ, DictType):
        _walk_and_register(table, typ.value, seen)
    # Scalars and other type kinds carry no nested record/enum to register.


def type_table_for(*types: Type) -> TypeTable:
    """Return a fresh seeded ``TypeTable`` with every record/enum reachable from *types*.

    For each ad-hoc ``RecordType``/``EnumType`` handle in *types* (and any
    nested inside a list/dict/field/variant), registers a ``TypeDef`` built
    from the handle's OWN embedded ``fields``/``variants`` — so table lookups
    reproduce exactly what the handle already carries.  Two of the given
    *types* sharing a ``(module_id, name)`` key must already carry the
    identical shape (construct them with distinct ``module_id``s to model two
    independent declarations under the same display name, e.g. coercion
    source/target pairs).
    """
    table = create_seeded_type_table()
    seen: set[tuple[object, str]] = set()
    for typ in types:
        _walk_and_register(table, typ, seen)
    return table
