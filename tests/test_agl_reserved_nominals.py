"""Tests for the host-known reserved nominal identity catalog.

Covers ``agm.agl.ir.reserved_nominals`` (a pure data leaf) and how
``agm.agl.semantics.types``/``agm.agl.semantics.type_table`` stamp its ids
onto the module-level built-in handle constants and canonical seeded
``TypeDef``s.
"""

from __future__ import annotations

import pytest

from agm.agl import PipelineDriver
from agm.agl.ir.reserved_nominals import (
    NO_DECL_ID,
    RESERVED_ENUM_MEMBER_IDS,
    RESERVED_NOMINAL_IDS,
    RESERVED_NOMINAL_NAMES,
    require_reserved_enum_member_id,
    reserved_nominal_id,
)
from agm.agl.runtime.engine_config import convert_host_value
from agm.agl.semantics.type_table import (
    BUILTIN_EXCEPTION_TYPE_DEFS,
    BUILTIN_PRELUDE_TYPE_DEFS,
    OPTION_TYPE_DEF,
    OPTIONAL_TYPE_DEF,
    RESERVED_FIELD_DEFAULT_VALUES,
    TypeDef,
    create_seeded_type_table,
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
from agm.agl.semantics.values import Value
from tests._agl_helpers import agl_roots, run_inline_code, run_program, shapes_match


def _decl_id(t: Type) -> int:
    """Return *t*'s ``decl_id``, asserting it is a nominal handle that has one."""
    assert isinstance(t, (RecordType, EnumType, ExceptionType))
    return t.decl_id


class TestReservedNominalCatalog:
    def test_covers_exactly_builtin_exceptions_and_prelude_types_plus_host_enums(self) -> None:
        expected = BUILTIN_EXCEPTION_NAMES | BUILTIN_PRELUDE_TYPE_NAMES | {"Option", "Optional"}
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


class TestReservedEnumMemberIds:
    def test_agent_sandbox_sandbox_member_aliases_the_standalone_sandbox_record(self) -> None:
        """A referenced member (enum-record unification) reuses the referenced
        record's own reserved id -- the one intentional alias in the table."""
        assert (
            require_reserved_enum_member_id("AgentSandbox", "Sandbox")
            == RESERVED_NOMINAL_IDS["Sandbox"]
        )
        assert ("AgentSandbox", "Sandbox") in RESERVED_ENUM_MEMBER_IDS

    def test_optional_none_and_some_members_alias_the_option_enum(self) -> None:
        """``Optional``'s declaration is ``Option::Some[T] | Option::None |
        Default``: its ``None``/``Some`` members are referenced members and so
        alias ``Option``'s own reserved ids, matching how the real
        standard-library declaration unifies their identity."""
        optional_none = require_reserved_enum_member_id("Optional", "None")
        optional_some = require_reserved_enum_member_id("Optional", "Some")
        assert optional_none == require_reserved_enum_member_id("Option", "None")
        assert optional_some == require_reserved_enum_member_id("Option", "Some")

    def test_mismatched_pair_is_rejected_rather_than_aliased_by_bare_member_name(self) -> None:
        """No name-keyed fallback: a member name that happens to match some
        other reserved type name (here ``Session``) must not silently alias
        that unrelated type's identity."""
        with pytest.raises(AssertionError, match="not a reserved enum member"):
            require_reserved_enum_member_id("SessionTransport", "Session")


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

    def test_optional_typedef_carries_its_reserved_decl_node_id(self) -> None:
        assert OPTIONAL_TYPE_DEF.decl_node_id == reserved_nominal_id("Optional")

    def test_agent_request_carries_the_selected_agent(self) -> None:
        agent_request = BUILTIN_PRELUDE_TYPE_DEFS["AgentRequest"]
        fields = dict(agent_request.fields)
        assert "agent" in fields

    def test_agent_request_embedded_option_fields_carry_option_reserved_id(self) -> None:
        agent_request = BUILTIN_PRELUDE_TYPE_DEFS["AgentRequest"]
        fields = dict(agent_request.fields)
        for field_name in ("target-type", "format-instructions", "json-schema", "previous-error"):
            assert _decl_id(fields[field_name]) == reserved_nominal_id("Option")

    def test_output_contract_option_embedded_record_carries_reserved_id(self) -> None:
        table = create_seeded_type_table()
        option = BUILTIN_PRELUDE_TYPE_DEFS["OutputContractOption"]
        some = next(member for member in option.members if member.name == "Some")
        value_field_type = dict(table.record_fields(some))["value"]
        assert _decl_id(value_field_type) == reserved_nominal_id("OutputContract")

    def test_agent_call_error_embedded_agent_field_carries_reserved_id(self) -> None:
        fields = dict(BUILTIN_EXCEPTION_TYPE_DEFS["AgentCallError"].fields)
        assert _decl_id(fields["agent"]) == reserved_nominal_id("Agent")


def _reserved_typedef(decl_id: int) -> TypeDef:
    """Return the seeded definition of the reserved record *decl_id*."""
    typedef = create_seeded_type_table().get_by_id(decl_id)
    assert typedef is not None
    return typedef


_RESERVED_DEFAULTED = pytest.mark.parametrize(
    "decl_id",
    [
        pytest.param(decl_id, id=_reserved_typedef(decl_id).name)
        for decl_id in sorted(RESERVED_FIELD_DEFAULT_VALUES)
    ],
)


def _defaults_omitting_call(decl_id: int) -> tuple[str, Value]:
    """Return a reserved record's constructor call omitting every defaulted field.

    Assumes every required field is ``text`` and supplies an empty text for
    it. Also returns the host's own value-syntax decode of that call against
    the seeded table.
    """
    type_table = create_seeded_type_table()
    typedef = _reserved_typedef(decl_id)
    required = ", ".join(
        f'{name} = ""'
        for (name, _type), has_default in zip(
            typedef.fields, typedef.field_has_default or (), strict=True
        )
        if not has_default
    )
    call = f"{typedef.name}({required})"
    return call, convert_host_value(typedef.name, call, typedef.handle(), type_table)


@_RESERVED_DEFAULTED
def test_reserved_field_defaults_match_the_stdlib_source(decl_id: int) -> None:
    """A reserved record's host-side default constants must match its own AgL source.

    ``RESERVED_FIELD_DEFAULT_VALUES`` hand-encodes each host-known record's
    constructor field defaults for the pre-execution CLI/config decode
    boundary (``runtime.engine_config.convert_host_value``'s
    ``default_resolver``), since no evaluator is reachable there. Nothing
    else compares those constants against the real stdlib source's own
    declared defaults, so this runs the real stdlib's constructor with every
    defaulted field omitted (through the ordinary evaluator) and the host's
    own value-syntax decode of the same constructor call side by side, and
    checks they agree field by field. Parametrized over every reserved record
    carrying host-side defaults, so a future one is covered automatically --
    this is the one guard against silent divergence between the stdlib's
    declared defaults and ``semantics/type_table.py``'s host-side constants.
    """
    call, host_value = _defaults_omitting_call(decl_id)

    result = run_program(f"let probe = {call}\nprobe\n")
    assert result.ok

    assert shapes_match(result.bindings["probe"], host_value)


@_RESERVED_DEFAULTED
def test_no_stdlib_constructor_fills_reserved_field_defaults(decl_id: int) -> None:
    """Without the stdlib, a reserved record's constructor fills its host-side defaults."""
    call, host_value = _defaults_omitting_call(decl_id)

    result = run_inline_code(
        PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None),
        f"let probe = {call}\nprobe\n",
        roots=agl_roots(include_stdlib=False),
        default_stdlib=False,
    )
    assert result.ok

    assert result.bindings["probe"] == host_value
