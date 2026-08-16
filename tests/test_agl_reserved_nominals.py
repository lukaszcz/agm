"""Tests for the host-known reserved nominal identity catalog.

Covers ``agm.agl.ir.reserved_nominals`` (a pure data leaf) and how
``agm.agl.semantics.types``/``agm.agl.semantics.type_table`` stamp its ids
onto the module-level built-in handle constants and canonical seeded
``TypeDef``s.
"""

from __future__ import annotations

from agm.agl.ir.reserved_nominals import (
    NO_DECL_ID,
    RESERVED_NOMINAL_IDS,
    RESERVED_NOMINAL_NAMES,
    reserved_nominal_id,
)
from agm.agl.semantics.type_table import (
    BUILTIN_EXCEPTION_TYPE_DEFS,
    BUILTIN_PRELUDE_TYPE_DEFS,
    OPTION_TYPE_DEF,
)
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
    EnumType,
    ExceptionType,
    RecordType,
    Type,
)


def _decl_id(t: Type) -> int:
    """Return *t*'s ``decl_id``, asserting it is a nominal handle that has one."""
    assert isinstance(t, (RecordType, EnumType, ExceptionType))
    return t.decl_id


class TestReservedNominalCatalog:
    def test_covers_exactly_builtin_exceptions_and_prelude_types_plus_option(self) -> None:
        expected = BUILTIN_EXCEPTION_NAMES | BUILTIN_PRELUDE_TYPE_NAMES | {"Option"}
        assert set(RESERVED_NOMINAL_NAMES) == expected

    def test_names_have_no_duplicates(self) -> None:
        assert len(RESERVED_NOMINAL_NAMES) == len(set(RESERVED_NOMINAL_NAMES))

    def test_ids_are_distinct(self) -> None:
        ids = list(RESERVED_NOMINAL_IDS.values())
        assert len(ids) == len(set(ids))

    def test_ids_never_collide_with_an_ast_node_id_or_the_absent_marker(self) -> None:
        """AST node ids are non-negative (``0`` included — it is the id of a
        program's very first node), and ``NO_DECL_ID`` marks a handle with no
        declaration identity, so every reserved id must sit below both."""
        assert all(value < NO_DECL_ID for value in RESERVED_NOMINAL_IDS.values())

    def test_reserved_nominal_id_matches_the_mapping(self) -> None:
        for name, value in RESERVED_NOMINAL_IDS.items():
            assert reserved_nominal_id(name) == value

    def test_reserved_nominal_id_returns_none_for_an_unreserved_name(self) -> None:
        assert reserved_nominal_id("NotARealType") is None


class TestSessionNominalWiring:
    def test_session_nominals_are_wired_through_every_prelude_catalog(self) -> None:
        for name in ("SessionTransport", "Session", "SessionStats", "SessionError"):
            assert name in BUILTIN_PRELUDE_TYPE_NAMES
            assert name in BUILTIN_PRELUDE_TYPES
            assert name in BUILTIN_PRELUDE_TYPE_DEFS
            assert name in RESERVED_NOMINAL_NAMES
            assert reserved_nominal_id(name) is not None


class TestBuiltinHandlesCarryReservedIds:
    def test_every_builtin_exception_handle_carries_its_reserved_id(self) -> None:
        for name, handle in BUILTIN_EXCEPTIONS.items():
            assert handle.decl_id == reserved_nominal_id(name)

    def test_every_builtin_prelude_handle_carries_its_reserved_id(self) -> None:
        for name, handle in BUILTIN_PRELUDE_TYPES.items():
            assert _decl_id(handle) == reserved_nominal_id(name)


class TestSeededTypeDefsCarryReservedIds:
    def test_every_builtin_exception_typedef_carries_its_reserved_decl_node_id(self) -> None:
        for name, typedef in BUILTIN_EXCEPTION_TYPE_DEFS.items():
            assert typedef.decl_node_id == reserved_nominal_id(name)

    def test_every_builtin_prelude_typedef_carries_its_reserved_decl_node_id(self) -> None:
        for name, typedef in BUILTIN_PRELUDE_TYPE_DEFS.items():
            assert typedef.decl_node_id == reserved_nominal_id(name)

    def test_option_typedef_carries_its_reserved_decl_node_id(self) -> None:
        assert OPTION_TYPE_DEF.decl_node_id == reserved_nominal_id("Option")

    def test_agent_request_embedded_agent_field_carries_agent_reserved_id(self) -> None:
        agent_request = BUILTIN_PRELUDE_TYPE_DEFS["AgentRequest"]
        fields = dict(agent_request.fields)
        assert _decl_id(fields["agent"]) == reserved_nominal_id("Agent")

    def test_agent_request_embedded_option_fields_carry_option_reserved_id(self) -> None:
        agent_request = BUILTIN_PRELUDE_TYPE_DEFS["AgentRequest"]
        fields = dict(agent_request.fields)
        for field_name in ("target_type", "format_instructions", "json_schema", "previous_error"):
            assert _decl_id(fields[field_name]) == reserved_nominal_id("Option")

    def test_output_contract_option_embedded_record_carries_reserved_id(self) -> None:
        variants = dict(BUILTIN_PRELUDE_TYPE_DEFS["OutputContractOption"].variants)
        value_field_type = dict(variants["Some"])["value"]
        assert _decl_id(value_field_type) == reserved_nominal_id("OutputContract")

    def test_agent_call_error_embedded_agent_field_carries_reserved_id(self) -> None:
        fields = dict(BUILTIN_EXCEPTION_TYPE_DEFS["AgentCallError"].fields)
        assert _decl_id(fields["agent"]) == reserved_nominal_id("Agent")
