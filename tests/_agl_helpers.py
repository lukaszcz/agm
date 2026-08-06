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
pass to ``type_table_for``). Each call stamps a fresh, distinct declaration
identity (``next_decl_id``) onto the ``TypeDef`` and derives the returned
handle's ``decl_id`` from that same identity (via ``TypeDef.handle``), so the
pair always names the same declaration and ``TypeTable.register`` (which
requires an identity) always accepts it. A self-referential or
mutually-recursive ad-hoc type reserves its identity up front (``next_decl_id``)
and passes it back in as *decl_id*, so a field embedded in its own body can
name it before the ``TypeDef``/handle pair exists.

``strip_decl_ids`` erases every embedded nominal handle's ``decl_id`` back to
``NO_DECL_ID``, for a test asserting an actual checked/looked-up type's SHAPE
against a hand-written literal that has no way to know the real declaration
identity — most equality assertions against a literal, since identity
participates in ``RecordType``/``EnumType``/``ExceptionType`` equality.

``agent_value`` builds a runtime ``std/core::Agent`` enum value directly, for
tests that need one as an expected value, a seeded host setting, or a request
payload without going through source parsing.
"""

from __future__ import annotations

import dataclasses
import itertools

from agm.agl.ir.ids import NominalId
from agm.agl.ir.nodes import IrBind, IrExpr, IrSequence
from agm.agl.ir.reserved_nominals import NO_DECL_ID, require_reserved_nominal_id
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.semantics.type_table import TypeDef, TypeTable, create_seeded_type_table
from agm.agl.semantics.types import EnumType, ExceptionType, RecordType, Type, transform_type
from agm.agl.semantics.values import EnumValue, TextValue

# Declaration identities for ad-hoc test TypeDefs, distinct from real AST node
# ids (which start at 0) and from every reserved identity (<= -2, see
# ir.reserved_nominals) so an ad-hoc type never collides with either.
_decl_ids = itertools.count(900_000)


def next_decl_id() -> int:
    """Return a fresh declaration identity, distinct across the whole test run."""
    return next(_decl_ids)


def let_root_capture(initializer: IrExpr) -> IrBind:
    """Return the binding IR for a simple or destructuring immutable let."""
    if isinstance(initializer, IrBind):
        return initializer
    assert isinstance(initializer, IrSequence)
    root_capture = initializer.items[0]
    assert isinstance(root_capture, IrBind)
    return root_capture


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


def strip_decl_ids(t: Type) -> Type:
    """Return *t* with every embedded nominal handle's ``decl_id`` reset to ``NO_DECL_ID``.

    ``RecordType``/``EnumType``/``ExceptionType`` equality includes
    ``decl_id`` (see ``semantics.types``), so an assertion comparing a real
    checked/looked-up type against a hand-written expected literal — which has
    no way to know the real declaration identity — needs both sides reduced to
    the identity-independent shape this compares: name, module, scope path,
    and (recursively, since a type argument may itself nest a nominal
    reference) type arguments.
    """

    def _strip(node: Type) -> Type:
        if isinstance(node, (RecordType, EnumType, ExceptionType)):
            return dataclasses.replace(node, decl_id=NO_DECL_ID)
        return node

    return transform_type(t, _strip)


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
    decl_id: int | None = None,
) -> tuple[RecordType, TypeDef]:
    """Build an ad-hoc ``RecordType`` handle and its matching ``TypeDef`` together.

    Returns ``(handle, typedef)``; pass ``typedef`` (and the ``TypeDef`` of
    any nested ad-hoc nominal type referenced in *fields*) to
    :func:`type_table_for` so the handle's field shape resolves.

    *decl_id* defaults to a freshly reserved identity; pass one explicitly
    (from :func:`next_decl_id`) for a SELF-referential or mutually-recursive
    ad-hoc type, whose own *fields* must embed a reference carrying this same
    identity before the ``TypeDef``/handle pair exists to read it off of.
    """
    typedef = TypeDef(
        kind="record",
        name=name,
        module_id=module_id,
        type_params=type_params,
        fields=tuple(fields.items()),
        decl_node_id=next_decl_id() if decl_id is None else decl_id,
    )
    return typedef.handle(type_args), typedef


def enum_type(
    name: str,
    variants: dict[str, dict[str, Type]],
    *,
    type_args: tuple[Type, ...] = (),
    module_id: ModuleId = ENTRY_ID,
    type_params: tuple[str, ...] = (),
    decl_id: int | None = None,
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
        decl_node_id=next_decl_id() if decl_id is None else decl_id,
    )
    return typedef.handle(type_args), typedef


def agent_value(variant: str, **fields: str) -> EnumValue:
    """Build the runtime ``std/core::Agent`` enum value for *variant*.

    Each keyword becomes a text-valued field, matching every ``Agent``
    variant's payload shape (``command``, ``model``/``thinking``, etc.); pass
    none for a variant with no payload.
    """
    return EnumValue(
        nominal=NominalId(require_reserved_nominal_id("Agent")),
        display_name="Agent",
        variant=variant,
        fields={name: TextValue(value) for name, value in fields.items()},
    )
