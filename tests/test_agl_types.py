"""Tests for the AgL type model.

Covers ``agm.agl.semantics.types`` and ``agm.agl.typecheck.env`` —
both import cleanly without depending on the rest of the checker.

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

from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.type_table import TypeTable, comparable_types
from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    InferenceVarType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
    contains_inference_var,
    contains_type_var,
    free_type_vars,
    is_assignable,
    is_json_shaped,
    iter_type,
    substitute,
)
from agm.agl.typecheck.env import TypeEnvironment

# Comparability tests below only exercise scalar/agent/unit/function/typevar
# operands, whose comparable_types arms never consult the TypeTable; an empty
# table is a valid (unused) argument for those calls.
_EMPTY_TABLE = TypeTable()

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
            setattr(u, "kind", "x")


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
            setattr(a, "kind", "x")


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
        assert repr(f) == "int -> text"

    def test_repr_multiple_params(self) -> None:
        f = FunctionType(params=(IntType(), TextType()), result=BoolType())
        assert repr(f) == "(int, text) -> bool"

    def test_repr_nested_function_param(self) -> None:
        inner = FunctionType(params=(IntType(),), result=IntType())
        outer = FunctionType(params=(inner,), result=TextType())
        assert repr(outer) == "(int -> int) -> text"

    def test_repr_nested_function_result_is_right_associative(self) -> None:
        inner = FunctionType(params=(TextType(),), result=BoolType())
        outer = FunctionType(params=(IntType(),), result=inner)
        assert repr(outer) == "int -> text -> bool"

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
        r = RecordType(name="R")
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
        from agm.agl.semantics.types import JsonType

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
        assert comparable_types(AgentType(), AgentType(), _EMPTY_TABLE) is False

    def test_unit_vs_unit_not_comparable(self) -> None:
        assert comparable_types(UnitType(), UnitType(), _EMPTY_TABLE) is False

    def test_function_vs_same_function_not_comparable(self) -> None:
        f = FunctionType(params=(IntType(),), result=IntType())
        assert comparable_types(f, f, _EMPTY_TABLE) is False

    def test_function_vs_function_not_comparable(self) -> None:
        f1 = FunctionType(params=(IntType(),), result=IntType())
        f2 = FunctionType(params=(IntType(),), result=IntType())
        assert comparable_types(f1, f2, _EMPTY_TABLE) is False

    def test_agent_vs_int_not_comparable(self) -> None:
        assert comparable_types(AgentType(), IntType(), _EMPTY_TABLE) is False

    def test_int_vs_agent_not_comparable(self) -> None:
        assert comparable_types(IntType(), AgentType(), _EMPTY_TABLE) is False

    def test_unit_vs_text_not_comparable(self) -> None:
        assert comparable_types(UnitType(), TextType(), _EMPTY_TABLE) is False

    def test_function_vs_text_not_comparable(self) -> None:
        f = FunctionType(params=(), result=UnitType())
        assert comparable_types(f, TextType(), _EMPTY_TABLE) is False

    # Regression: existing scalar comparability is unchanged.
    def test_int_vs_int_comparable(self) -> None:
        assert comparable_types(IntType(), IntType(), _EMPTY_TABLE) is True

    def test_text_vs_text_comparable(self) -> None:
        assert comparable_types(TextType(), TextType(), _EMPTY_TABLE) is True

    def test_int_vs_decimal_comparable(self) -> None:
        assert comparable_types(IntType(), DecimalType(), _EMPTY_TABLE) is True

    def test_decimal_vs_int_comparable(self) -> None:
        assert comparable_types(DecimalType(), IntType(), _EMPTY_TABLE) is True

    def test_bool_vs_bool_comparable(self) -> None:
        assert comparable_types(BoolType(), BoolType(), _EMPTY_TABLE) is True

    def test_int_vs_text_not_comparable(self) -> None:
        assert comparable_types(IntType(), TextType(), _EMPTY_TABLE) is False


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
        fields = env.type_table.record_fields(t)
        assert fields["stdout"] == TextType()
        assert fields["exit_code"] == IntType()
        assert fields["stderr"] == TextType()
        assert fields["timed_out"] == BoolType()

    def test_parse_policy_resolves(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("ParsePolicy")
        assert isinstance(t, EnumType)
        assert t.name == "ParsePolicy"

    def test_parse_policy_variants(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("ParsePolicy")
        assert isinstance(t, EnumType)
        variants = env.type_table.enum_variants(t)
        # Abort has no fields.
        assert variants["Abort"] == {}
        # Retry has n: int.
        assert variants["Retry"] == {"n": IntType()}

    def test_recursion_error_resolves(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("RecursionError")
        assert isinstance(t, ExceptionType)
        assert t.name == "RecursionError"

    def test_recursion_error_fields(self) -> None:
        env = TypeEnvironment()
        t = env.get_type("RecursionError")
        assert isinstance(t, ExceptionType)
        fields = env.type_table.exception_fields(t)
        assert fields["message"] == TextType()
        assert fields["trace_id"] == TextType()
        assert fields["limit"] == IntType()

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
        source.seal()
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
        getattr(source, "_types")["ExecResult"] = RecordType(name="ExecResult")
        source.seal()
        target = TypeEnvironment()
        target.seed_from(source)
        # Target's ExecResult must be the original prelude, not the source's fake.
        t = target.get_type("ExecResult")
        assert isinstance(t, RecordType)
        assert "stdout" in target.type_table.record_fields(t)  # original prelude has stdout

    def test_seed_copies_user_types(self) -> None:
        source = TypeEnvironment()
        user_record = RecordType(name="MyRecord")
        source.register_type("MyRecord", user_record)
        source.seal()
        target = TypeEnvironment()
        target.seed_from(source)
        assert target.get_type("MyRecord") == user_record

    def test_seed_does_not_copy_builtin_exceptions(self) -> None:
        # Built-in exceptions are present by default; seeding must not duplicate.
        source = TypeEnvironment()
        source.seal()
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
        env.register_type("UserRec", RecordType(name="UserRec"))
        env.unregister_name("UserRec")
        assert env.has_type("UserRec") is False


# ---------------------------------------------------------------------------
# TypeVarType
# ---------------------------------------------------------------------------


class TestTypeVarType:
    def test_kind(self) -> None:
        assert TypeVarType("T").kind == "typevar"

    def test_repr(self) -> None:
        assert repr(TypeVarType("T")) == "T"

    def test_equality_same_name(self) -> None:
        assert TypeVarType("T") == TypeVarType("T")

    def test_inequality_different_name(self) -> None:
        assert TypeVarType("T") != TypeVarType("U")

    def test_frozen(self) -> None:
        tv = TypeVarType("T")
        with pytest.raises(Exception):
            setattr(tv, "name", "U")

    def test_not_json_shaped(self) -> None:
        assert is_json_shaped(TypeVarType("T")) is False

    def test_not_comparable_left(self) -> None:
        assert comparable_types(TypeVarType("T"), IntType(), _EMPTY_TABLE) is False

    def test_not_comparable_right(self) -> None:
        assert comparable_types(IntType(), TypeVarType("T"), _EMPTY_TABLE) is False

    def test_not_comparable_with_itself(self) -> None:
        assert comparable_types(TypeVarType("T"), TypeVarType("T"), _EMPTY_TABLE) is False

    def test_assignable_to_same_typevar(self) -> None:
        assert is_assignable(TypeVarType("T"), TypeVarType("T")) is True

    def test_not_assignable_to_different_typevar(self) -> None:
        assert is_assignable(TypeVarType("T"), TypeVarType("U")) is False

    def test_not_assignable_to_int(self) -> None:
        assert is_assignable(TypeVarType("T"), IntType()) is False

    def test_int_not_assignable_to_typevar(self) -> None:
        assert is_assignable(IntType(), TypeVarType("T")) is False

    def test_not_assignable_to_json(self) -> None:
        assert is_assignable(TypeVarType("T"), JsonType()) is False

    def test_bottom_assignable_to_typevar(self) -> None:
        assert is_assignable(BottomType(), TypeVarType("T")) is True


# ---------------------------------------------------------------------------
# InferenceVarType
# ---------------------------------------------------------------------------


class TestInferenceVarType:
    def test_default_hint_allocates_a_fresh_identity(self) -> None:
        assert InferenceVarType() != InferenceVarType()

    def test_same_hint_allocates_distinct_identities(self) -> None:
        first = InferenceVarType("T")
        second = InferenceVarType("T")

        assert first != second
        assert len({first, second}) == 2

    def test_aliases_have_the_same_identity(self) -> None:
        variable = InferenceVarType("T")
        alias = variable

        assert alias == variable
        assert hash(alias) == hash(variable)

    def test_display_hint_does_not_determine_identity(self) -> None:
        first = InferenceVarType("first")
        second = InferenceVarType("second")

        assert first != second

    def test_representation_is_private(self) -> None:
        variable = InferenceVarType("T")

        assert variable.kind == "inferencevar"
        assert repr(variable) == "<inference-var>"
        assert "T" not in repr(variable)

    def test_is_not_a_source_or_rigid_type_variable(self) -> None:
        variable = InferenceVarType("T")

        assert is_json_shaped(variable) is False
        assert free_type_vars(variable) == frozenset()
        assert contains_type_var(variable) is False
        assert contains_inference_var(variable) is True

    def test_is_not_registered_as_a_source_type(self) -> None:
        assert TypeEnvironment().has_type("InferenceVarType") is False

    def test_is_not_reexported_by_the_public_typecheck_package(self) -> None:
        import agm.agl.typecheck as typecheck

        assert "InferenceVarType" not in typecheck.__all__

    @pytest.mark.parametrize(
        "typ",
        [
            ListType(InferenceVarType("T")),
            DictType(InferenceVarType("T")),
            FunctionType(params=(InferenceVarType("T"),), result=InferenceVarType("T")),
            RecordType("Box", type_args=(InferenceVarType("T"),)),
            EnumType("Option", type_args=(InferenceVarType("T"),)),
        ],
    )
    def test_can_nest_under_every_structural_or_nominal_constructor(self, typ: Type) -> None:
        assert isinstance(typ, Type)
        assert contains_inference_var(typ) is True
        assert contains_type_var(typ) is False
        assert free_type_vars(typ) == frozenset()

    def test_iter_type_visits_flexible_variables_in_nested_types(self) -> None:
        variable = InferenceVarType("T")
        typ = FunctionType(
            params=(ListType(variable),),
            result=RecordType("Box", type_args=(DictType(variable),)),
        )

        assert tuple(item for item in iter_type(typ) if item == variable) == (variable, variable)

    def test_rigid_substitution_does_not_rewrite_flexible_variables(self) -> None:
        variable = InferenceVarType("T")
        typ = FunctionType(
            params=(TypeVarType("T"), ListType(variable)),
            result=EnumType("Option", type_args=(variable, TypeVarType("U"))),
        )

        result = substitute(typ, {"T": IntType(), "U": TextType()})

        assert result == FunctionType(
            params=(IntType(), ListType(variable)),
            result=EnumType("Option", type_args=(variable, TextType())),
        )
        assert contains_inference_var(result) is True

    def test_shared_semantic_analyses_ignore_flexible_variables_as_source_names(self) -> None:
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.semantics.analyses import compute_finite_closure, compute_uninhabited
        from agm.agl.semantics.type_table import TypeDef

        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Value",
                module_id=ENTRY_ID,
                fields=(("value", InferenceVarType("T")),),
            )
        )
        assert compute_uninhabited(table) == frozenset()

        finite_table = TypeTable()
        finite_table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("direct", TypeVarType("T")),
                    ("next", RecordType("Box", type_args=(InferenceVarType("T"),))),
                ),
            )
        )
        assert compute_finite_closure(finite_table).infinite == frozenset()

    def test_schema_and_extern_walkers_reject_flexible_variables(self) -> None:
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.type_schema import build_extern_contract, derive_schema
        from agm.agl.typecheck.env import FunctionSignature

        variable = InferenceVarType("T")
        table = create_seeded_type_table()

        with pytest.raises(TypeError):
            derive_schema(variable, table)
        with pytest.raises(TypeError):
            build_extern_contract(FunctionSignature(params=(), result=variable), table)


# ---------------------------------------------------------------------------
# Generic nominal identity (fields/variants excluded from equality)
# ---------------------------------------------------------------------------


class TestGenericNominalIdentity:
    """RecordType/EnumType handles carry no field/variant data; identity is
    purely ``(module_id, name, type_args)``."""

    def test_record_identity_by_name_and_type_args(self) -> None:
        r1 = RecordType("Box", type_args=(IntType(),))
        r2 = RecordType("Box", type_args=(IntType(),))
        assert r1 == r2

    def test_record_different_type_args_not_equal(self) -> None:
        r1 = RecordType("Box", type_args=(IntType(),))
        r2 = RecordType("Box", type_args=(TextType(),))
        assert r1 != r2

    def test_record_no_type_args_identity(self) -> None:
        r1 = RecordType("R")
        r2 = RecordType("R")
        assert r1 == r2  # name-based, no type_args

    def test_record_consistent_hash(self) -> None:
        r1 = RecordType("Box", type_args=(IntType(),))
        r2 = RecordType("Box", type_args=(IntType(),))
        assert hash(r1) == hash(r2)

    def test_enum_identity_by_name_and_type_args(self) -> None:
        e1 = EnumType("Option", type_args=(IntType(),))
        e2 = EnumType("Option", type_args=(IntType(),))
        assert e1 == e2

    def test_enum_different_type_args_not_equal(self) -> None:
        e1 = EnumType("Option", type_args=(IntType(),))
        e2 = EnumType("Option", type_args=(TextType(),))
        assert e1 != e2

    def test_repr_with_type_args(self) -> None:
        r = RecordType("Box", type_args=(IntType(),))
        assert repr(r) == "Box[int]"

    def test_repr_without_type_args(self) -> None:
        r = RecordType("R")
        assert repr(r) == "R"

    def test_enum_repr_with_type_args(self) -> None:
        e = EnumType("Option", type_args=(TextType(),))
        assert repr(e) == "Option[text]"


# ---------------------------------------------------------------------------
# Helper functions: free_type_vars, substitute, contains_type_var
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_free_type_vars_primitive(self) -> None:
        assert free_type_vars(IntType()) == frozenset()

    def test_free_type_vars_typevar(self) -> None:
        assert free_type_vars(TypeVarType("T")) == frozenset({"T"})

    def test_free_type_vars_list(self) -> None:
        assert free_type_vars(ListType(TypeVarType("T"))) == frozenset({"T"})

    def test_free_type_vars_dict(self) -> None:
        assert free_type_vars(DictType(TypeVarType("V"))) == frozenset({"V"})

    def test_free_type_vars_function(self) -> None:
        ft = FunctionType(params=(TypeVarType("A"),), result=TypeVarType("B"))
        assert free_type_vars(ft) == frozenset({"A", "B"})

    def test_free_type_vars_record_type_args(self) -> None:
        r = RecordType("Box", type_args=(TypeVarType("T"),))
        assert free_type_vars(r) == frozenset({"T"})

    def test_free_type_vars_no_vars(self) -> None:
        r = RecordType("R")
        assert free_type_vars(r) == frozenset()

    def test_free_type_vars_enum_variants(self) -> None:
        e = EnumType("Either", type_args=(TypeVarType("A"), TypeVarType("B")))
        assert free_type_vars(e) == frozenset({"A", "B"})

    def test_free_type_vars_enum_no_vars(self) -> None:
        e = EnumType("Status")
        assert free_type_vars(e) == frozenset()

    def test_substitute_typevar(self) -> None:
        t = TypeVarType("T")
        result = substitute(t, {"T": IntType()})
        assert result == IntType()

    def test_substitute_typevar_not_in_subst(self) -> None:
        t = TypeVarType("T")
        result = substitute(t, {"U": IntType()})
        assert result == TypeVarType("T")

    def test_substitute_list(self) -> None:
        t = ListType(TypeVarType("T"))
        result = substitute(t, {"T": IntType()})
        assert result == ListType(IntType())

    def test_substitute_dict(self) -> None:
        t = DictType(TypeVarType("V"))
        result = substitute(t, {"V": TextType()})
        assert result == DictType(TextType())

    def test_substitute_function(self) -> None:
        ft = FunctionType(params=(TypeVarType("A"),), result=TypeVarType("B"))
        result = substitute(ft, {"A": IntType(), "B": TextType()})
        assert result == FunctionType(params=(IntType(),), result=TextType())

    def test_substitute_record_type_args(self) -> None:
        r = RecordType("Box", type_args=(TypeVarType("T"),))
        result = substitute(r, {"T": IntType()})
        assert isinstance(result, RecordType)
        assert result.type_args == (IntType(),)
        assert result.name == "Box"

    def test_substitute_enum_type_args(self) -> None:
        e = EnumType("Option", type_args=(TypeVarType("T"),))
        result = substitute(e, {"T": TextType()})
        assert isinstance(result, EnumType)
        assert result.type_args == (TextType(),)
        assert result.name == "Option"

    def test_substitute_primitive_unchanged(self) -> None:
        assert substitute(IntType(), {"T": TextType()}) == IntType()

    def test_contains_type_var_true(self) -> None:
        assert contains_type_var(ListType(TypeVarType("T"))) is True

    def test_contains_type_var_false(self) -> None:
        assert contains_type_var(ListType(IntType())) is False

    def test_contains_type_var_record_type_args(self) -> None:
        assert contains_type_var(RecordType("Box", type_args=(TypeVarType("T"),))) is True
        assert contains_type_var(RecordType("Box", type_args=(IntType(),))) is False

    def test_contains_type_var_enum_type_args(self) -> None:
        assert contains_type_var(EnumType("Option", type_args=(TypeVarType("T"),))) is True
        assert contains_type_var(EnumType("Option", type_args=(IntType(),))) is False


# ---------------------------------------------------------------------------
# Capability gates for TypeVarType
# ---------------------------------------------------------------------------


class TestCapabilityGates:
    def test_typevar_not_json_shaped(self) -> None:
        assert is_json_shaped(TypeVarType("T")) is False

    def test_typevar_not_comparable_left(self) -> None:
        assert comparable_types(TypeVarType("T"), IntType(), _EMPTY_TABLE) is False

    def test_typevar_not_comparable_right(self) -> None:
        assert comparable_types(IntType(), TypeVarType("T"), _EMPTY_TABLE) is False

    def test_typevar_not_comparable_with_itself(self) -> None:
        assert comparable_types(TypeVarType("T"), TypeVarType("T"), _EMPTY_TABLE) is False

    def test_typevar_assignable_to_same(self) -> None:
        assert is_assignable(TypeVarType("T"), TypeVarType("T")) is True

    def test_typevar_not_assignable_to_json(self) -> None:
        assert is_assignable(TypeVarType("T"), JsonType()) is False

    def test_bottom_assignable_to_typevar(self) -> None:
        assert is_assignable(BottomType(), TypeVarType("T")) is True


# ---------------------------------------------------------------------------
# New builtin exceptions
# ---------------------------------------------------------------------------


class TestNewExceptions:
    def test_cast_error_in_builtin_exceptions(self) -> None:
        from agm.agl.semantics.types import BUILTIN_EXCEPTION_NAMES
        assert "CastError" in BUILTIN_EXCEPTION_NAMES

    def test_json_parse_error_in_builtin_exceptions(self) -> None:
        from agm.agl.semantics.types import BUILTIN_EXCEPTION_NAMES
        assert "JsonParseError" in BUILTIN_EXCEPTION_NAMES

    def test_cast_error_fields(self) -> None:
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.semantics.types import BUILTIN_EXCEPTIONS, TextType
        e = BUILTIN_EXCEPTIONS["CastError"]
        fields = create_seeded_type_table().exception_fields(e)
        assert "source_type" in fields
        assert "target_type" in fields
        assert "raw" in fields
        assert fields["source_type"] == TextType()

    def test_json_parse_error_fields(self) -> None:
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.semantics.types import BUILTIN_EXCEPTIONS, TextType
        e = BUILTIN_EXCEPTIONS["JsonParseError"]
        fields = create_seeded_type_table().exception_fields(e)
        assert "raw" in fields
        assert fields["raw"] == TextType()


class TestCastClassification:
    def test_bottom_source_is_assignable_to_cast_target(self) -> None:
        from agm.agl.semantics.types import (
            BottomType,
            CastKind,
            TextType,
            cast_classification,
        )

        assert cast_classification(BottomType(), TextType()) == CastKind.TOTAL_NOOP

    def test_text_to_text_noop(self) -> None:
        from agm.agl.semantics.types import CastKind, TextType, cast_classification
        assert cast_classification(TextType(), TextType()) == CastKind.TOTAL_NOOP

    def test_int_to_text_render(self) -> None:
        from agm.agl.semantics.types import CastKind, IntType, TextType, cast_classification
        assert cast_classification(IntType(), TextType()) == CastKind.TOTAL_RENDER

    def test_bool_to_text_render(self) -> None:
        from agm.agl.semantics.types import BoolType, CastKind, TextType, cast_classification
        assert cast_classification(BoolType(), TextType()) == CastKind.TOTAL_RENDER

    def test_int_to_json_total_json(self) -> None:
        from agm.agl.semantics.types import CastKind, IntType, JsonType, cast_classification
        assert cast_classification(IntType(), JsonType()) == CastKind.TOTAL_JSON

    def test_text_to_json_total_json(self) -> None:
        from agm.agl.semantics.types import CastKind, JsonType, TextType, cast_classification
        assert cast_classification(TextType(), JsonType()) == CastKind.TOTAL_JSON

    def test_json_to_json_noop(self) -> None:
        from agm.agl.semantics.types import CastKind, JsonType, cast_classification
        assert cast_classification(JsonType(), JsonType()) == CastKind.TOTAL_NOOP

    def test_text_to_int_fallible(self) -> None:
        from agm.agl.semantics.types import CastKind, IntType, TextType, cast_classification
        assert cast_classification(TextType(), IntType()) == CastKind.FALLIBLE

    def test_json_to_bool_fallible(self) -> None:
        from agm.agl.semantics.types import BoolType, CastKind, JsonType, cast_classification
        assert cast_classification(JsonType(), BoolType()) == CastKind.FALLIBLE

    def test_decimal_to_int_fallible(self) -> None:
        from agm.agl.semantics.types import CastKind, DecimalType, IntType, cast_classification
        assert cast_classification(DecimalType(), IntType()) == CastKind.FALLIBLE

    def test_int_to_decimal_noop(self) -> None:
        from agm.agl.semantics.types import CastKind, DecimalType, IntType, cast_classification
        assert cast_classification(IntType(), DecimalType()) == CastKind.TOTAL_NOOP

    def test_bool_to_int_static_error(self) -> None:
        from agm.agl.semantics.types import BoolType, CastKind, IntType, cast_classification
        assert cast_classification(BoolType(), IntType()) == CastKind.STATIC_ERROR

    def test_int_to_bool_static_error(self) -> None:
        from agm.agl.semantics.types import BoolType, CastKind, IntType, cast_classification
        assert cast_classification(IntType(), BoolType()) == CastKind.STATIC_ERROR

    def test_unit_source_static_error(self) -> None:
        from agm.agl.semantics.types import CastKind, TextType, UnitType, cast_classification
        assert cast_classification(UnitType(), TextType()) == CastKind.STATIC_ERROR

    def test_agent_target_static_error(self) -> None:
        from agm.agl.semantics.types import AgentType, CastKind, TextType, cast_classification
        assert cast_classification(TextType(), AgentType()) == CastKind.STATIC_ERROR

    def test_record_to_json_total(self) -> None:
        from agm.agl.semantics.types import CastKind, JsonType, RecordType, cast_classification
        r = RecordType(name="R")
        assert cast_classification(r, JsonType()) == CastKind.TOTAL_JSON

    def test_list_of_record_to_json_static_error(self) -> None:
        # A list whose elements are nominal is not json-shaped and not itself a
        # nominal source, so `list[record] as json` remains a static error: only
        # a direct record/enum/exception source supports the explicit json cast.
        from agm.agl.semantics.types import (
            CastKind,
            JsonType,
            ListType,
            RecordType,
            cast_classification,
        )
        lst = ListType(elem=RecordType(name="R"))
        assert cast_classification(lst, JsonType()) == CastKind.STATIC_ERROR

    def test_text_to_record_fallible(self) -> None:
        from agm.agl.semantics.types import CastKind, RecordType, TextType, cast_classification
        r = RecordType(name="R")
        assert cast_classification(TextType(), r) == CastKind.FALLIBLE

    def test_json_to_record_fallible(self) -> None:
        from agm.agl.semantics.types import CastKind, JsonType, RecordType, cast_classification
        r = RecordType(name="R")
        assert cast_classification(JsonType(), r) == CastKind.FALLIBLE

    def test_exception_as_target_static_error(self) -> None:
        from agm.agl.semantics.types import (
            CastKind,
            ExceptionType,
            TextType,
            cast_classification,
        )
        exc = ExceptionType(name="MyError")
        assert cast_classification(TextType(), exc) == CastKind.STATIC_ERROR

    def test_exception_source_to_text_render(self) -> None:
        from agm.agl.semantics.types import (
            CastKind,
            ExceptionType,
            TextType,
            cast_classification,
        )
        exc = ExceptionType(name="MyError")
        assert cast_classification(exc, TextType()) == CastKind.TOTAL_RENDER

    def test_json_to_text_render(self) -> None:
        from agm.agl.semantics.types import CastKind, JsonType, TextType, cast_classification
        assert cast_classification(JsonType(), TextType()) == CastKind.TOTAL_RENDER

    def test_int_to_list_static_error(self) -> None:
        from agm.agl.semantics.types import (
            CastKind,
            IntType,
            ListType,
            cast_classification,
        )
        assert cast_classification(IntType(), ListType(IntType())) == CastKind.STATIC_ERROR


# ---------------------------------------------------------------------------
# Nominal equality for RecordType / EnumType
# ---------------------------------------------------------------------------


class TestNominalEquality:
    """RecordType/EnumType are handles: identity is (module_id, name, type_args) only."""

    def _mod(self, name: str) -> "ModuleId":
        from agm.agl.modules.ids import ModuleId

        return ModuleId(segments=(name,))

    def test_record_same_name_and_module_equal(self) -> None:
        from agm.agl.semantics.types import RecordType

        r1 = RecordType(name="Point")
        r2 = RecordType(name="Point")
        assert r1 == r2

    def test_enum_same_name_and_module_equal(self) -> None:
        from agm.agl.semantics.types import EnumType

        e1 = EnumType(name="Color")
        e2 = EnumType(name="Color")
        assert e1 == e2

    def test_record_different_module_not_equal(self) -> None:
        from agm.agl.semantics.types import RecordType

        mod_foo = self._mod("foo")
        mod_bar = self._mod("bar")
        r1 = RecordType(name="Color", module_id=mod_foo)
        r2 = RecordType(name="Color", module_id=mod_bar)
        assert r1 != r2

    def test_enum_different_module_not_equal(self) -> None:
        from agm.agl.semantics.types import EnumType

        mod_foo = self._mod("foo")
        mod_bar = self._mod("bar")
        e1 = EnumType(name="Color", module_id=mod_foo)
        e2 = EnumType(name="Color", module_id=mod_bar)
        assert e1 != e2

    def test_record_different_name_not_equal(self) -> None:
        from agm.agl.semantics.types import RecordType

        r1 = RecordType(name="Point")
        r2 = RecordType(name="Color")
        assert r1 != r2

    def test_enum_different_name_not_equal(self) -> None:
        from agm.agl.semantics.types import EnumType

        e1 = EnumType(name="Color")
        e2 = EnumType(name="Shape")
        assert e1 != e2

    def test_record_hashable(self) -> None:
        from agm.agl.semantics.types import RecordType

        r = RecordType(name="Point")
        h = hash(r)
        assert isinstance(h, int)

    def test_enum_hashable(self) -> None:
        from agm.agl.semantics.types import EnumType

        e = EnumType(name="Color")
        h = hash(e)
        assert isinstance(h, int)

    def test_equal_records_hash_equal(self) -> None:
        from agm.agl.semantics.types import RecordType

        r1 = RecordType(name="Point")
        r2 = RecordType(name="Point")
        assert r1 == r2
        assert hash(r1) == hash(r2)

    def test_equal_enums_hash_equal(self) -> None:
        from agm.agl.semantics.types import EnumType

        e1 = EnumType(name="Color")
        e2 = EnumType(name="Color")
        assert e1 == e2
        assert hash(e1) == hash(e2)

    def test_record_usable_as_dict_key(self) -> None:
        from agm.agl.semantics.types import RecordType

        r1 = RecordType(name="Point")
        r2 = RecordType(name="Point")
        d: dict[RecordType, str] = {r1: "found"}
        assert d[r2] == "found"

    def test_enum_usable_in_set(self) -> None:
        from agm.agl.semantics.types import EnumType

        e1 = EnumType(name="Color")
        e2 = EnumType(name="Color")
        s = {e1, e2}
        assert len(s) == 1

    def test_list_type_stays_structural(self) -> None:
        assert ListType(IntType()) != ListType(TextType())

    def test_dict_type_stays_structural(self) -> None:
        assert DictType(IntType()) != DictType(TextType())

    def test_function_type_stays_structural(self) -> None:
        f1 = FunctionType(params=(IntType(),), result=TextType())
        f2 = FunctionType(params=(TextType(),), result=TextType())
        assert f1 != f2
