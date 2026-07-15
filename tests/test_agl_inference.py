"""Isolated tests for the checker-independent AgL inference engine."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agm.agl.modules.ids import ModuleId
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
)
from agm.agl.syntax.spans import SourceId, SourceSpan
from agm.agl.typecheck.inference import (
    ConstraintOrigin,
    ConstraintRole,
    InferenceEngine,
    InferenceError,
)


def _span(line: int, source: str = "<test>") -> SourceSpan:
    return SourceSpan(line, 1, line, 2, line - 1, line, SourceId(source))


def _origin(
    engine: InferenceEngine,
    line: int,
    *,
    role: ConstraintRole = ConstraintRole.FUNCTION_ARGUMENT,
    subject: str = "id",
    type_param: str | None = "T",
) -> ConstraintOrigin:
    return engine.origin(_span(line), role=role, subject=subject, type_param=type_param)


class TestInstantiation:
    def test_fresh_identity_and_ordered_independent_instantiations(self) -> None:
        engine = InferenceEngine()
        template = FunctionType(params=(TypeVarType("T"),), result=TypeVarType("T"))

        first = engine.instantiate(("T",), (template,))
        second = engine.instantiate(("T",), (template,))

        assert tuple(first.variables) == ("T",)
        assert first.variables["T"] != second.variables["T"]
        assert first.templates == (
            FunctionType(params=(first.variables["T"],), result=first.variables["T"]),
        )
        assert second.templates == (
            FunctionType(params=(second.variables["T"],), result=second.variables["T"]),
        )

    def test_instantiation_only_replaces_quantified_rigids(self) -> None:
        engine = InferenceEngine()
        instantiated = engine.instantiate(
            ("T",),
            (FunctionType((TypeVarType("T"), TypeVarType("U")), TypeVarType("T")),),
        )

        assert instantiated.templates == (
            FunctionType(
                (instantiated.variables["T"], TypeVarType("U")), instantiated.variables["T"]
            ),
        )

    def test_instantiation_freshens_every_supported_structural_constructor(self) -> None:
        engine = InferenceEngine()
        instantiated = engine.instantiate(
            ("T",),
            (
                ListType(TypeVarType("T")),
                DictType(TypeVarType("T")),
                RecordType("Box", (TypeVarType("T"),)),
                EnumType("Option", (TypeVarType("T"),)),
                IntType(),
            ),
        )

        variable = instantiated.variables["T"]
        assert instantiated.templates == (
            ListType(variable),
            DictType(variable),
            RecordType("Box", (variable,)),
            EnumType("Option", (variable,)),
            IntType(),
        )


class TestUnification:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (IntType(), IntType()),
            (TextType(), TextType()),
            (BoolType(), BoolType()),
            (DecimalType(), DecimalType()),
            (JsonType(), JsonType()),
            (UnitType(), UnitType()),
            (AgentType(), AgentType()),
            (ExceptionType("Problem"), ExceptionType("Problem")),
            (ListType(IntType()), ListType(IntType())),
            (DictType(IntType()), DictType(IntType())),
            (
                FunctionType((IntType(), ListType(TextType())), TextType()),
                FunctionType((IntType(), ListType(TextType())), TextType()),
            ),
            (
                RecordType("Box", (IntType(),), ModuleId.from_dotted("a")),
                RecordType("Box", (IntType(),), ModuleId.from_dotted("a")),
            ),
            (
                EnumType("Option", (IntType(),), ModuleId.from_dotted("a")),
                EnumType("Option", (IntType(),), ModuleId.from_dotted("a")),
            ),
            (TypeVarType("T"), TypeVarType("T")),
        ],
    )
    def test_exact_structural_types_unify(self, left: Type, right: Type) -> None:
        engine = InferenceEngine()
        engine.unify(left, right, _origin(engine, 1))

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (IntType(), TextType()),
            (ListType(IntType()), ListType(TextType())),
            (DictType(IntType()), DictType(TextType())),
            (FunctionType((IntType(),), IntType()), FunctionType((), IntType())),
            (TypeVarType("T"), TypeVarType("U")),
            (
                RecordType("Box", (IntType(),), ModuleId.from_dotted("one")),
                RecordType("Box", (IntType(),), ModuleId.from_dotted("two")),
            ),
            (RecordType("Box", (IntType(),)), EnumType("Box", (IntType(),))),
            (EnumType("Box", (IntType(),)), EnumType("Box", (IntType(), TextType()))),
            (EnumType("One", (IntType(),)), EnumType("Two", (IntType(),))),
        ],
    )
    def test_shape_and_identity_mismatches_fail(self, left: Type, right: Type) -> None:
        engine = InferenceEngine()
        with pytest.raises(InferenceError):
            engine.unify(left, right, _origin(engine, 1))

    def test_flexible_variables_merge_and_solve_to_a_rigid(self) -> None:
        engine = InferenceEngine()
        first = engine.fresh("T")
        second = engine.fresh("T")
        engine.unify(second, first, _origin(engine, 1))
        engine.unify(TypeVarType("T"), second, _origin(engine, 2))

        assert engine.zonk(first) == TypeVarType("T")
        assert engine.zonk(second) == TypeVarType("T")

    def test_structural_unification_descends_to_flexible_children(self) -> None:
        engine = InferenceEngine()
        list_variable = engine.fresh("list")
        dict_variable = engine.fresh("dict")
        function_variable = engine.fresh("function")
        record_variable = engine.fresh("record")
        enum_variable = engine.fresh("enum")

        engine.unify(ListType(list_variable), ListType(IntType()), _origin(engine, 1))
        engine.unify(DictType(dict_variable), DictType(TextType()), _origin(engine, 2))
        engine.unify(
            FunctionType((function_variable,), function_variable),
            FunctionType((IntType(),), IntType()),
            _origin(engine, 3),
        )
        engine.unify(
            RecordType("Box", (record_variable,)),
            RecordType("Box", (IntType(),)),
            _origin(engine, 4),
        )
        engine.unify(
            EnumType("Option", (enum_variable,)),
            EnumType("Option", (TextType(),)),
            _origin(engine, 5),
        )

        assert tuple(engine.zonk(variable) for variable in (list_variable, function_variable)) == (
            IntType(),
            IntType(),
        )
        assert tuple(engine.zonk(variable) for variable in (dict_variable, enum_variable)) == (
            TextType(),
            TextType(),
        )
        assert engine.zonk(record_variable) == IntType()

    def test_nominal_arguments_and_function_parts_are_invariant(self) -> None:
        engine = InferenceEngine()
        with pytest.raises(InferenceError):
            engine.unify(
                FunctionType((ListType(IntType()),), IntType()),
                FunctionType((ListType(TextType()),), IntType()),
                _origin(engine, 1),
            )
        with pytest.raises(InferenceError):
            engine.unify(
                RecordType("Box", (IntType(),)),
                RecordType("Box", (TextType(),)),
                _origin(engine, 2),
            )

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda variable: ListType(variable),
            lambda variable: DictType(variable),
            lambda variable: FunctionType((variable,), IntType()),
            lambda variable: RecordType("Box", (variable,)),
            lambda variable: EnumType("Option", (variable,)),
        ],
    )
    def test_occurs_check_rejects_each_structural_path(
        self, wrap: Callable[[InferenceVarType], Type]
    ) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")

        with pytest.raises(InferenceError, match="infinite"):
            engine.unify(variable, wrap(variable), _origin(engine, 1))

    def test_solver_rejects_flexible_variables_owned_by_another_engine(self) -> None:
        owner = InferenceEngine()
        foreign = owner.fresh("T")

        with pytest.raises(AssertionError, match="owned"):
            InferenceEngine().zonk(foreign)

    def test_occurs_check_never_expands_nominal_definitions(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        with pytest.raises(InferenceError):
            engine.unify(variable, RecordType("Recursive", (variable,)), _origin(engine, 1))

    def test_equal_flexible_variables_retain_evidence(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        engine.unify(variable, variable, _origin(engine, 1))
        engine.unify(variable, IntType(), _origin(engine, 2))

        with pytest.raises(InferenceError) as raised:
            engine.unify(variable, TextType(), _origin(engine, 3))
        assert raised.value.related[0][1] == _span(1)

    def test_bottom_succeeds_without_solving_a_flexible_variable(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        engine.unify(variable, BottomType(), _origin(engine, 1))
        engine.unify(BottomType(), ListType(variable), _origin(engine, 2))

        assert engine.zonk(variable) == variable
        assert engine.is_solved(variable) is False


class TestContextCompletion:
    def test_context_fills_only_unresolved_representatives(self) -> None:
        engine = InferenceEngine()
        first = engine.fresh("T")
        second = engine.fresh("U")
        engine.complete_from_context(
            FunctionType((ListType(first),), DictType(second)),
            FunctionType((ListType(IntType()),), DictType(TextType())),
            _origin(engine, 1, role=ConstraintRole.EXPECTED_RESULT),
        )

        assert engine.zonk(first) == IntType()
        assert engine.zonk(second) == TextType()

    def test_context_never_overrides_actual_equality_evidence(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        engine.unify(variable, IntType(), _origin(engine, 1))
        engine.complete_from_context(variable, TextType(), _origin(engine, 2))

        assert engine.zonk(variable) == IntType()

    def test_context_ignores_mismatched_shapes_and_bottom(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        engine.complete_from_context(ListType(variable), DictType(IntType()), _origin(engine, 1))
        engine.complete_from_context(variable, BottomType(), _origin(engine, 2))

        assert engine.is_solved(variable) is False

    def test_context_recurses_through_each_matching_shape(self) -> None:
        engine = InferenceEngine()
        list_variable = engine.fresh("list")
        dict_variable = engine.fresh("dict")
        function_variable = engine.fresh("function")
        record_variable = engine.fresh("record")
        enum_variable = engine.fresh("enum")
        engine.complete_from_context(
            ListType(list_variable), ListType(IntType()), _origin(engine, 1)
        )
        engine.complete_from_context(
            DictType(dict_variable), DictType(TextType()), _origin(engine, 2)
        )
        engine.complete_from_context(
            FunctionType((function_variable,), IntType()),
            FunctionType((TextType(),), IntType()),
            _origin(engine, 3),
        )
        engine.complete_from_context(
            RecordType("Box", (record_variable,)),
            RecordType("Box", (IntType(),)),
            _origin(engine, 4),
        )
        engine.complete_from_context(
            EnumType("Option", (enum_variable,)),
            EnumType("Option", (TextType(),)),
            _origin(engine, 5),
        )

        assert tuple(engine.zonk(variable) for variable in (list_variable, record_variable)) == (
            IntType(),
            IntType(),
        )
        assert tuple(
            engine.zonk(variable) for variable in (dict_variable, function_variable, enum_variable)
        ) == (TextType(), TextType(), TextType())

    def test_context_ignores_recursive_or_incompatible_matching_shapes(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        other = engine.fresh("U")
        engine.complete_from_context(variable, other, _origin(engine, 1))
        engine.complete_from_context(variable, variable, _origin(engine, 2))
        engine.complete_from_context(variable, ListType(variable), _origin(engine, 3))
        engine.complete_from_context(
            FunctionType((variable,), IntType()), FunctionType((), IntType()), _origin(engine, 4)
        )
        engine.complete_from_context(
            RecordType("Box", (variable,)), RecordType("Other", (IntType(),)), _origin(engine, 5)
        )
        engine.complete_from_context(
            EnumType("Option", (variable,)), EnumType("Option", (),), _origin(engine, 6)
        )
        engine.complete_from_context(
            EnumType("One", (engine.fresh("V"),)),
            EnumType("Two", (IntType(),)),
            _origin(engine, 7),
        )

        assert engine.is_solved(variable) is False
        assert engine.parent_of(variable) == engine.parent_of(other)


class TestFinalizationAndProvenance:
    def test_zonk_compresses_links_and_rebuilds_nested_types(self) -> None:
        engine = InferenceEngine()
        first = engine.fresh("T")
        second = engine.fresh("U")
        third = engine.fresh("V")
        engine.unify(first, second, _origin(engine, 1))
        engine.unify(second, third, _origin(engine, 2))
        engine.unify(third, IntType(), _origin(engine, 3))

        zonked = engine.zonk(FunctionType((first,), ListType(second)))

        assert zonked == FunctionType((IntType(),), ListType(IntType()))
        assert engine.zonk(first) == IntType()
        assert engine.parent_of(first) == engine.parent_of(third)

    def test_requirements_and_leak_assertions_are_owned_and_reusable(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        requirement = _origin(engine, 1, subject="id", type_param="T")
        engine.require_solved(variable, requirement)

        with pytest.raises(InferenceError, match="T") as raised:
            engine.check_requirements()
        assert "inference-var" not in str(raised.value)
        with pytest.raises(AssertionError):
            engine.assert_no_inference_vars((FunctionType((variable,), IntType()),))

        engine.unify(variable, IntType(), _origin(engine, 2))
        engine.check_requirements()
        engine.assert_no_inference_vars((FunctionType((variable,), IntType()),))

    def test_requirement_rejects_a_solution_with_nested_unresolved_variables(self) -> None:
        engine = InferenceEngine()
        outer = engine.fresh("T")
        inner = engine.fresh("U")
        engine.unify(outer, ListType(inner), _origin(engine, 1))
        engine.require_solved(outer, _origin(engine, 2, type_param="T"))

        with pytest.raises(InferenceError, match="T"):
            engine.check_requirements()
        with pytest.raises(AssertionError):
            engine.assert_no_inference_vars((outer,))

    def test_mismatch_uses_failing_origin_and_earliest_related_evidence(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        first = _origin(engine, 10, subject="id", type_param="T")
        second = _origin(engine, 20, subject="id", type_param="T")
        engine.unify(variable, IntType(), first)

        with pytest.raises(InferenceError) as raised:
            engine.unify(variable, TextType(), second)

        error = raised.value
        assert error.span == second.span
        assert error.related == (
            ("T was first constrained by function argument 'id'.", first.span),
        )
        assert "inference-var" not in str(error)

    def test_colliding_external_origin_sequences_keep_both_provenance_records(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        first = ConstraintOrigin(_span(1), 0, ConstraintRole.FUNCTION_ARGUMENT, "first", "T")
        second = ConstraintOrigin(_span(2), 0, ConstraintRole.FUNCTION_ARGUMENT, "second", "T")
        engine.unify(variable, IntType(), first)

        with pytest.raises(InferenceError) as raised:
            engine.unify(variable, TextType(), second)

        assert raised.value.related[0][1] == first.span

    def test_origin_sequence_is_stable_and_roles_are_typed(self) -> None:
        engine = InferenceEngine()
        first = engine.origin(_span(1), role=ConstraintRole.LITERAL_ELEMENT, subject="list")
        second = engine.origin(_span(2), role=ConstraintRole.EXPECTED_RESULT, subject="id")

        assert (first.sequence, second.sequence) == (0, 1)
        assert first.role is ConstraintRole.LITERAL_ELEMENT
