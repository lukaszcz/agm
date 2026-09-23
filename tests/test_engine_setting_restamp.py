"""Generic tests for the engine-setting nominal-identity restamp mechanism.

Covers ``agm.agl.runtime.engine_config.restamp_engine_setting``/
``_restamp_value_tree`` against their own contract, independent of any single
engine key. Key-specific behavior lives in ``test_default_agent_setting.py``
and ``test_default_sandbox_setting.py``.
"""

from __future__ import annotations

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.engine_keys import ENGINE_KEY_TYPES
from agm.agl.semantics.type_table import TypeTable, create_seeded_type_table
from agm.agl.semantics.types import ArrayType, DictType, EnumType, RecordType, Type
from agm.agl.semantics.values import RecordValue
from tests._agl_helpers import agent_value


def test_restamp_engine_setting_ignores_non_enum_backed_keys() -> None:
    """A boolean-kind or unrecognized key carries no host-enum identity to restamp.

    ``trace``/``strict-json`` are boolean-kind and an unrecognized name is not a
    key at all, so :func:`restamp_engine_setting` leaves such a value untouched
    regardless of the ``from``/``to`` tables given.
    """
    from agm.agl.runtime.engine_config import restamp_engine_setting

    stray = agent_value("AgentCommand", command="unused")
    for key in ("trace", "strict-json", "not-an-engine-key"):
        assert (
            restamp_engine_setting(
                key, stray, from_table=NO_BUILTIN_DECLARATIONS, to_table=NO_BUILTIN_DECLARATIONS
            )
            is stray
        )


def test_restamp_engine_setting_leaves_an_unrecognized_nested_field_identity_alone() -> None:
    """A nested field identity neither table recognizes crosses through unchanged.

    ``Sandbox``'s own fields are always host-known (``Optional``/``Option``),
    so this exercises the general fallback directly: any nested ``RecordValue``
    field ``_restamp_value_tree`` cannot place in ``from_table`` keeps its
    original nominal rather than being restamped.
    """
    from agm.agl.runtime.engine_config import restamp_engine_setting

    sandbox_nominal = NO_BUILTIN_DECLARATIONS.resolve_standard_member(
        "AgentSandbox", "Sandbox"
    ).nominal
    stray_field = RecordValue(nominal=NominalId(-999_999_999), fields={})
    value = RecordValue(nominal=sandbox_nominal, fields={"memory": stray_field})

    result = restamp_engine_setting(
        "default-sandbox",
        value,
        from_table=NO_BUILTIN_DECLARATIONS,
        to_table=NO_BUILTIN_DECLARATIONS,
    )

    assert isinstance(result, RecordValue)
    assert result.fields["memory"] == stray_field


def _contains_collection(typ: Type, type_table: TypeTable, seen: set[Type]) -> bool:
    """Return whether *typ*, or anything reachable through its fields/members, is a collection."""
    if isinstance(typ, (ArrayType, DictType)):
        return True
    if isinstance(typ, RecordType):
        if typ in seen:
            return False
        seen.add(typ)
        return any(
            _contains_collection(field_type, type_table, seen)
            for field_type in type_table.record_fields(typ).values()
        )
    if isinstance(typ, EnumType):
        if typ in seen:
            return False
        seen.add(typ)
        return any(
            _contains_collection(member, type_table, seen)
            for member in type_table.enum_members(typ)
        )
    return False


def test_no_engine_key_type_contains_a_collection_payload() -> None:
    """No ``ENGINE_KEY_TYPES`` value carries an array/dict anywhere in its shape.

    ``_restamp_value_tree`` (``agm.agl.runtime.engine_config``) only recurses
    into records: an engine-setting value is always a scalar, an ``Option``/
    ``Optional``, or a record built from those, never a collection. This is
    the reachable guard for that assumption, iterating the real engine-key
    catalog rather than a hand-built value: it fails the day a future engine
    key's type gains an array/dict payload, pointing at
    ``_restamp_value_tree``, which would then need extending to recurse into
    it too.
    """
    type_table = create_seeded_type_table()
    for name, typ in ENGINE_KEY_TYPES.items():
        assert not _contains_collection(typ, type_table, set()), (
            f"engine key {name!r} type {typ!r} carries a collection payload; "
            "_restamp_value_tree must be extended to recurse into it"
        )
