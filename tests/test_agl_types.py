"""Tests for the AgL v2 type model (S3a).

Imports ONLY ``agm.agl.typecheck.types`` and ``agm.agl.typecheck.env`` —
both import cleanly without depending on the checker (which is mid-rewrite).

Coverage:
- UnitType / AgentType / FunctionType: kind, repr, structural equality.
- is_json_shaped: False for all three new types.
- is_assignable: exact-only for the three new types (positive + negative).
- comparable_types: False for agent/unit/function; unchanged for scalars.
- TypeEnvironment: prelude types (ExecResult, ParsePolicy) and RecursionError
  exception registered in every fresh env.
- seed_from: does not duplicate/clobber prelude types.
- unregister_name: leaves prelude + exception names intact.
"""

from __future__ import annotations

import pytest

from agm.agl.typecheck.env import TypeEnvironment
from agm.agl.typecheck.types import (
    AgentType,
    BoolType,
    DecimalType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    RecordType,
    TextType,
    UnitType,
    comparable_types,
    is_assignable,
    is_json_shaped,
)

# ---------------------------------------------------------------------------
# UnitType
# ---------------------------------------------------------------------------


class TestUnitType:
    def test_kind(self) -> None:
        assert UnitType().kind == "unit"

    def test_repr(self) -> None:
        assert repr(UnitType()) == "unit"

    def test_equality(self) -> None:
        assert UnitType() == UnitType()

    def test_frozen(self) -> None:
        u = UnitType()
        with pytest.raises(Exception):
            u.kind = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AgentType
# ---------------------------------------------------------------------------


class TestAgentType:
    def test_kind(self) -> None:
        assert AgentType().kind == "agent"

    def test_repr(self) -> None:
        assert repr(AgentType()) == "agent"

    def test_equality(self) -> None:
        assert AgentType() == AgentType()

    def test_frozen(self) -> None:
        a = AgentType()
        with pytest.raises(Exception):
            a.kind = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FunctionType
# ---------------------------------------------------------------------------


class TestFunctionType:
    def test_kind(self) -> None:
        f = FunctionType(params=(IntType(),), result=TextType())
        assert f.kind == "function"

    def test_repr_no_params(self) -> None:
        f = FunctionType(params=(), result=IntType())
        assert repr(f) == "() -> int"

    def test_repr_one_param(self) -> None:
        f = FunctionType(params=(IntType(),), result=TextType())
        assert repr(f) == "(int) -> text"

    def test_repr_multiple_params(self) -> None:
        f = FunctionType(params=(IntType(), TextType()), result=BoolType())
        assert repr(f) == "(int, text) -> bool"

    def test_repr_nested_function(self) -> None:
        inner = FunctionType(params=(IntType(),), result=IntType())
        outer = FunctionType(params=(inner,), result=TextType())
        assert repr(outer) == "((int) -> int) -> text"

    def test_structural_equality_same(self) -> None:
        f1 = FunctionType(params=(IntType(), TextType()), result=BoolType())
        f2 = FunctionType(params=(IntType(), TextType()), result=BoolType())
        assert f1 == f2

    def test_structural_equality_different_params(self) -> None:
        f1 = FunctionType(params=(IntType(),), result=TextType())
        f2 = FunctionType(params=(TextType(),), result=TextType())
        assert f1 != f2

    def test_structural_equality_different_result(self) -> None:
        f1 = FunctionType(params=(IntType(),), result=TextType())
        f2 = FunctionType(params=(IntType(),), result=IntType())
        assert f1 != f2

    def test_structural_equality_different_arity(self) -> None:
        f1 = FunctionType(params=(IntType(),), result=TextType())
        f2 = FunctionType(params=(IntType(), IntType()), result=TextType())
        assert f1 != f2

    def test_structural_equality_no_params(self) -> None:
        f1 = FunctionType(params=(), result=UnitType())
        f2 = FunctionType(params=(), result=UnitType())
        assert f1 == f2


# ---------------------------------------------------------------------------
# is_json_shaped — new types must return False
# ---------------------------------------------------------------------------


class TestIsJsonShaped:
    def test_unit_not_json_shaped(self) -> None:
        assert is_json_shaped(UnitType()) is False

    def test_agent_not_json_shaped(self) -> None:
        assert is_json_shaped(AgentType()) is False

    def test_function_not_json_shaped(self) -> None:
        f = FunctionType(params=(IntType(),), result=TextType())
        assert is_json_shaped(f) is False

    def test_function_no_params_not_json_shaped(self) -> None:
        assert is_json_shaped(FunctionType(params=(), result=UnitType())) is False

    # Regression: existing types still work correctly.
    def test_text_is_json_shaped(self) -> None:
        assert is_json_shaped(TextType()) is True

    def test_int_is_json_shaped(self) -> None:
        assert is_json_shaped(IntType()) is True

    def test_record_not_json_shaped(self) -> None:
        r = RecordType(name="R", fields={"x": IntType()})
        assert is_json_shaped(r) is False


# ---------------------------------------------------------------------------
# is_assignable — exact-only for new types
# ---------------------------------------------------------------------------


class TestIsAssignable:
    # Positive cases — exact match.
    def test_unit_to_unit(self) -> None:
        assert is_assignable(UnitType(), UnitType()) is True

    def test_agent_to_agent(self) -> None:
        assert is_assignable(AgentType(), AgentType()) is True

    def test_function_to_same_function(self) -> None:
        f = FunctionType(params=(IntType(),), result=TextType())
        assert is_assignable(f, f) is True

    def test_function_exact_structural_match(self) -> None:
        f1 = FunctionType(params=(IntType(), BoolType()), result=DecimalType())
        f2 = FunctionType(params=(IntType(), BoolType()), result=DecimalType())
        assert is_assignable(f1, f2) is True

    # Negative cases — no widening, no variance.
    def test_unit_not_assignable_to_agent(self) -> None:
        assert is_assignable(UnitType(), AgentType()) is False

    def test_agent_not_assignable_to_unit(self) -> None:
        assert is_assignable(AgentType(), UnitType()) is False

    def test_unit_not_assignable_to_int(self) -> None:
        assert is_assignable(UnitType(), IntType()) is False

    def test_int_not_assignable_to_unit(self) -> None:
        assert is_assignable(IntType(), UnitType()) is False

    def test_agent_not_assignable_to_int(self) -> None:
        assert is_assignable(AgentType(), IntType()) is False

    def test_function_param_mismatch_not_assignable(self) -> None:
        f1 = FunctionType(params=(IntType(),), result=TextType())
        f2 = FunctionType(params=(TextType(),), result=TextType())
        assert is_assignable(f1, f2) is False

    def test_function_result_mismatch_not_assignable(self) -> None:
        f1 = FunctionType(params=(IntType(),), result=TextType())
        f2 = FunctionType(params=(IntType(),), result=IntType())
        assert is_assignable(f1, f2) is False

    def test_function_not_assignable_to_int(self) -> None:
        f = FunctionType(params=(IntType(),), result=TextType())
        assert is_assignable(f, IntType()) is False

    def test_function_not_assignable_to_json(self) -> None:
        # json accepts JSON-shaped values; function is not JSON-shaped.
        from agm.agl.typecheck.types import JsonType

        f = FunctionType(params=(IntType(),), result=TextType())
        assert is_assignable(f, JsonType()) is False

    # Regression: existing scalar coercion is unchanged.
    def test_int_widening_to_decimal(self) -> None:
        assert is_assignable(IntType(), DecimalType()) is True

    def test_decimal_not_narrowing_to_int(self) -> None:
        assert is_assignable(DecimalType(), IntType()) is False


# ---------------------------------------------------------------------------
# comparable_types — new types must return False; scalars unchanged
# ---------------------------------------------------------------------------


class TestComparableTypes:
    # New types: never comparable (even with themselves).
    def test_agent_vs_agent_not_comparable(self) -> None:
        assert comparable_types(AgentType(), AgentType()) is False

    def test_unit_vs_unit_not_comparable(self) -> None:
        assert comparable_types(UnitType(), UnitType()) is False

    def test_function_vs_same_function_not_comparable(self) -> None:
        f = FunctionType(params=(IntType(),), result=IntType())
        assert comparable_types(f, f) is False

    def test_function_vs_function_not_comparable(self) -> None:
        f1 = FunctionType(params=(IntType(),), result=IntType())
        f2 = FunctionType(params=(IntType(),), result=IntType())
        assert comparable_types(f1, f2) is False

    def test_agent_vs_int_not_comparable(self) -> None:
        assert comparable_types(AgentType(), IntType()) is False

    def test_int_vs_agent_not_comparable(self) -> None:
        assert comparable_types(IntType(), AgentType()) is False

    def test_unit_vs_text_not_comparable(self) -> None:
        assert comparable_types(UnitType(), TextType()) is False

    def test_function_vs_text_not_comparable(self) -> None:
        f = FunctionType(params=(), result=UnitType())
        assert comparable_types(f, TextType()) is False

    # Regression: existing scalar comparability is unchanged.
    def test_int_vs_int_comparable(self) -> None:
        assert comparable_types(IntType(), IntType()) is True

    def test_text_vs_text_comparable(self) -> None:
        assert comparable_types(TextType(), TextType()) is True

    def test_int_vs_decimal_comparable(self) -> None:
        assert comparable_types(IntType(), DecimalType()) is True

    def test_decimal_vs_int_comparable(self) -> None:
        assert comparable_types(DecimalType(), IntType()) is True

    def test_bool_vs_bool_comparable(self) -> None:
        assert comparable_types(BoolType(), BoolType()) is True

    def test_int_vs_text_not_comparable(self) -> None:
        assert comparable_types(IntType(), TextType()) is False


# ---------------------------------------------------------------------------
# TypeEnvironment — prelude types + RecursionError registered on init
# ---------------------------------------------------------------------------


class TestTypeEnvironmentPrelude:
    def test_exec_result_resolves(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("ExecResult")
        assert isinstance(t, RecordType)
        assert t.name == "ExecResult"

    def test_exec_result_fields(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("ExecResult")
        assert isinstance(t, RecordType)
        assert t.fields["stdout"] == TextType()
        assert t.fields["exit_code"] == IntType()
        assert t.fields["stderr"] == TextType()
        assert t.fields["timed_out"] == BoolType()

    def test_parse_policy_resolves(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("ParsePolicy")
        assert isinstance(t, EnumType)
        assert t.name == "ParsePolicy"

    def test_parse_policy_variants(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("ParsePolicy")
        assert isinstance(t, EnumType)
        # Abort has no fields.
        assert t.variants["Abort"] == {}
        # Retry has n: int.
        assert t.variants["Retry"] == {"n": IntType()}

    def test_recursion_error_resolves(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("RecursionError")
        assert isinstance(t, ExceptionType)
        assert t.name == "RecursionError"

    def test_recursion_error_fields(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("RecursionError")
        assert isinstance(t, ExceptionType)
        assert t.fields["message"] == TextType()
        assert t.fields["trace_id"] == TextType()
        assert t.fields["limit"] == IntType()

    def test_resolve_named_type_exec_result(self) -> None:
        env = TypeEnvironment()
        t = env.resolve_named_type("ExecResult")
        assert isinstance(t, RecordType)
        assert t.name == "ExecResult"

    def test_resolve_named_type_parse_policy(self) -> None:
        env = TypeEnvironment()
        t = env.resolve_named_type("ParsePolicy")
        assert isinstance(t, EnumType)
        assert t.name == "ParsePolicy"

    def test_has_type_exec_result(self) -> None:
        env = TypeEnvironment()
        assert env.has_type("ExecResult") is True

    def test_has_type_parse_policy(self) -> None:
        env = TypeEnvironment()
        assert env.has_type("ParsePolicy") is True

    def test_has_type_recursion_error(self) -> None:
        env = TypeEnvironment()
        assert env.has_type("RecursionError") is True


# ---------------------------------------------------------------------------
# seed_from — does not duplicate or clobber prelude types
# ---------------------------------------------------------------------------


class TestSeedFrom:
    def test_prelude_present_after_seed(self) -> None:
        source = TypeEnvironment()
        target = TypeEnvironment()
        target.seed_from(source)
        # Prelude types are still available.
        assert isinstance(target.get_type("ExecResult"), RecordType)
        assert isinstance(target.get_type("ParsePolicy"), EnumType)

    def test_seed_does_not_overwrite_prelude(self) -> None:
        # If source somehow had a different ExecResult, seed_from must NOT
        # copy it (prelude names are excluded from the copy loop).
        source = TypeEnvironment()
        # Manually inject a different type under the prelude name in source._types.
        # We reach inside _types to simulate a hypothetical collision.
        source._types["ExecResult"] = RecordType(  # type: ignore[attr-defined]
            name="ExecResult", fields={"x": IntType()}
        )
        target = TypeEnvironment()
        target.seed_from(source)
        # Target's ExecResult must be the original prelude, not the source's fake.
        t = target.get_type("ExecResult")
        assert isinstance(t, RecordType)
        assert "stdout" in t.fields  # original prelude has stdout

    def test_seed_copies_user_types(self) -> None:
        source = TypeEnvironment()
        user_record = RecordType(name="MyRecord", fields={"val": IntType()})
        source.register_type("MyRecord", user_record)
        target = TypeEnvironment()
        target.seed_from(source)
        assert target.get_type("MyRecord") == user_record

    def test_seed_does_not_copy_builtin_exceptions(self) -> None:
        # Built-in exceptions are present by default; seeding must not duplicate.
        source = TypeEnvironment()
        target = TypeEnvironment()
        target.seed_from(source)
        # Still exactly one ExecError (an ExceptionType), not duplicated.
        t = target.get_type("ExecError")
        assert isinstance(t, ExceptionType)
        assert t.name == "ExecError"


# ---------------------------------------------------------------------------
# unregister_name — prelude + exception names must never be removed
# ---------------------------------------------------------------------------


class TestUnregisterName:
    def test_cannot_unregister_exec_result(self) -> None:
        env = TypeEnvironment()
        env.unregister_name("ExecResult")
        # Still present.
        assert env.has_type("ExecResult") is True

    def test_cannot_unregister_parse_policy(self) -> None:
        env = TypeEnvironment()
        env.unregister_name("ParsePolicy")
        assert env.has_type("ParsePolicy") is True

    def test_cannot_unregister_builtin_exception(self) -> None:
        env = TypeEnvironment()
        env.unregister_name("ExecError")
        assert env.has_type("ExecError") is True

    def test_cannot_unregister_recursion_error(self) -> None:
        env = TypeEnvironment()
        env.unregister_name("RecursionError")
        assert env.has_type("RecursionError") is True

    def test_can_unregister_user_type(self) -> None:
        env = TypeEnvironment()
        env.register_type("UserRec", RecordType(name="UserRec", fields={}))
        env.unregister_name("UserRec")
        assert env.has_type("UserRec") is False
