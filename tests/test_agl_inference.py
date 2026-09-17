"""Tests for the AgL inference engine and inference-dependent checker behavior."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import (
    ArrayType,
    BottomType,
    DictType,
    EnumType,
    FunctionType,
    InferenceVarType,
    IntType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
)
from agm.agl.syntax.nodes import FuncDef
from agm.agl.syntax.spans import SourceId, SourceSpan
from agm.agl.typecheck.env import AglTypeError
from agm.agl.typecheck.inference import (
    ConstraintOrigin,
    ConstraintRole,
    InferenceEngine,
    InferenceError,
)
from tests._agl_helpers import strip_decl_ids
from tests.agl.module_graph import resolve_and_check_inline_entry


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
                ArrayType(TypeVarType("T")),
                DictType(TypeVarType("T")),
                RecordType("Box", (TypeVarType("T"),)),
                EnumType("Option", (TypeVarType("T"),)),
                IntType(),
            ),
        )

        variable = instantiated.variables["T"]
        assert instantiated.templates == (
            ArrayType(variable),
            DictType(variable),
            RecordType("Box", (variable,)),
            EnumType("Option", (variable,)),
            IntType(),
        )


class TestUnification:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (IntType(), TextType()),
            (ArrayType(IntType()), ArrayType(TextType())),
            (DictType(IntType()), DictType(TextType())),
            (FunctionType((IntType(),), IntType()), FunctionType((), IntType())),
            (TypeVarType("T"), TypeVarType("U")),
            (
                RecordType("Box", (IntType(),), ModuleId.from_path("one"), decl_id=1),
                RecordType("Box", (IntType(),), ModuleId.from_path("two"), decl_id=2),
            ),
            (RecordType("Box", (IntType(),)), EnumType("Box", (IntType(),))),
            (EnumType("Box", (IntType(),)), EnumType("Box", (IntType(), TextType()))),
            (EnumType("One", (IntType(),), decl_id=1), EnumType("Two", (IntType(),), decl_id=2)),
        ],
    )
    def test_shape_and_identity_mismatches_fail(self, left: Type, right: Type) -> None:
        engine = InferenceEngine()
        with pytest.raises(InferenceError):
            engine.unify(left, right, _origin(engine, 1))

    def test_unification_does_not_unify_two_same_named_declarations(self) -> None:
        # Two distinct declarations sharing one name path (a REPL
        # redeclaration mints a fresh identity for the same name, so the two
        # coexist) must never unify with each other, even though name and
        # module agree -- only ``decl_id`` distinguishes them.
        engine = InferenceEngine()
        old_box = RecordType("Box", (IntType(),), decl_id=1)
        new_box = RecordType("Box", (IntType(),), decl_id=2)

        with pytest.raises(InferenceError):
            engine.unify(old_box, new_box, _origin(engine, 1))

        old_option = EnumType("Option", (IntType(),), decl_id=1)
        new_option = EnumType("Option", (IntType(),), decl_id=2)

        with pytest.raises(InferenceError):
            engine.unify(old_option, new_option, _origin(engine, 2))

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
        array_variable = engine.fresh("array")
        dict_variable = engine.fresh("dict")
        function_variable = engine.fresh("function")
        record_variable = engine.fresh("record")
        enum_variable = engine.fresh("enum")

        engine.unify(ArrayType(array_variable), ArrayType(IntType()), _origin(engine, 1))
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

        assert tuple(engine.zonk(variable) for variable in (array_variable, function_variable)) == (
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
                FunctionType((ArrayType(IntType()),), IntType()),
                FunctionType((ArrayType(TextType()),), IntType()),
                _origin(engine, 1),
            )
        with pytest.raises(InferenceError):
            engine.unify(
                RecordType("Box", (IntType(),)),
                RecordType("Box", (TextType(),)),
                _origin(engine, 2),
            )

    def test_member_record_unifies_only_with_a_rigid_enum_target(self) -> None:
        member = RecordType("Leaf", scope_path=("Tree",), decl_id=1)
        tree = EnumType("Tree", (IntType(),), decl_id=2)
        table = TypeTable()
        table.register(
            TypeDef(
                kind="enum",
                name="Tree",
                module_id=tree.module_id,
                type_params=("T",),
                members=(member,),
                decl_node_id=tree.decl_id,
            )
        )
        engine = InferenceEngine(table)

        engine.unify(member, tree, _origin(engine, 1))
        with pytest.raises(InferenceError):
            engine.unify(tree, member, _origin(engine, 2))
        with pytest.raises(InferenceError):
            engine.unify(
                RecordType("Other", scope_path=("Tree",), decl_id=3), tree, _origin(engine, 3)
            )
        with pytest.raises(InferenceError):
            engine.unify(ArrayType(member), ArrayType(tree), _origin(engine, 4))

    def test_member_record_unification_solves_captured_enum_arguments(self) -> None:
        table = TypeTable()
        tree = TypeDef(
            kind="enum",
            name="Tree",
            module_id=ModuleId.from_path("trees"),
            type_params=("T",),
            members=(RecordType("Leaf", (TypeVarType("T"),), scope_path=("Tree",), decl_id=1),),
            decl_node_id=2,
        )
        table.register(tree)
        engine = InferenceEngine(table)
        value_type = RecordType("Leaf", (engine.fresh("T"),), scope_path=("Tree",), decl_id=1)

        engine.unify(value_type, tree.handle((IntType(),)), _origin(engine, 1))

        assert engine.zonk(value_type.type_args[0]) == IntType()

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda variable: ArrayType(variable),
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
        engine.unify(BottomType(), ArrayType(variable), _origin(engine, 2))

        assert engine.zonk(variable) == variable
        assert engine.is_solved(variable) is False


class TestContextCompletion:
    def test_context_fills_only_unresolved_representatives(self) -> None:
        engine = InferenceEngine()
        first = engine.fresh("T")
        second = engine.fresh("U")
        engine.complete_from_context(
            FunctionType((ArrayType(first),), DictType(second)),
            FunctionType((ArrayType(IntType()),), DictType(TextType())),
            _origin(engine, 1, role=ConstraintRole.EXPECTED_RESULT),
        )

        assert engine.zonk(first) == IntType()
        assert engine.zonk(second) == TextType()

    def test_member_record_context_completes_captured_enum_arguments(self) -> None:
        table = TypeTable()
        tree = TypeDef(
            kind="enum",
            name="Tree",
            module_id=ModuleId.from_path("trees"),
            type_params=("T",),
            members=(RecordType("Leaf", (TypeVarType("T"),), scope_path=("Tree",), decl_id=1),),
            decl_node_id=2,
        )
        table.register(tree)
        engine = InferenceEngine(table)
        value_type = RecordType("Leaf", (engine.fresh("T"),), scope_path=("Tree",), decl_id=1)

        engine.complete_from_context(value_type, tree.handle((IntType(),)), _origin(engine, 1))

        assert engine.zonk(value_type.type_args[0]) == IntType()
        engine.complete_from_context(
            RecordType("Other", scope_path=("Tree",), decl_id=3),
            tree.handle((IntType(),)),
            _origin(engine, 2),
        )

    def test_context_never_overrides_actual_equality_evidence(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        engine.unify(variable, IntType(), _origin(engine, 1))
        engine.complete_from_context(variable, TextType(), _origin(engine, 2))

        assert engine.zonk(variable) == IntType()

    def test_context_ignores_mismatched_shapes_and_bottom(self) -> None:
        engine = InferenceEngine()
        variable = engine.fresh("T")
        engine.complete_from_context(ArrayType(variable), DictType(IntType()), _origin(engine, 1))
        engine.complete_from_context(variable, BottomType(), _origin(engine, 2))

        assert engine.is_solved(variable) is False

    def test_context_recurses_through_each_matching_shape(self) -> None:
        engine = InferenceEngine()
        array_variable = engine.fresh("array")
        dict_variable = engine.fresh("dict")
        function_variable = engine.fresh("function")
        record_variable = engine.fresh("record")
        enum_variable = engine.fresh("enum")
        engine.complete_from_context(
            ArrayType(array_variable), ArrayType(IntType()), _origin(engine, 1)
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

        assert tuple(engine.zonk(variable) for variable in (array_variable, record_variable)) == (
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
        engine.complete_from_context(variable, ArrayType(variable), _origin(engine, 3))
        engine.complete_from_context(
            FunctionType((variable,), IntType()), FunctionType((), IntType()), _origin(engine, 4)
        )
        engine.complete_from_context(
            RecordType("Box", (variable,), decl_id=1),
            RecordType("Other", (IntType(),), decl_id=2),
            _origin(engine, 5),
        )
        engine.complete_from_context(
            EnumType("Option", (variable,)),
            EnumType(
                "Option",
                (),
            ),
            _origin(engine, 6),
        )
        engine.complete_from_context(
            EnumType("One", (engine.fresh("V"),), decl_id=1),
            EnumType("Two", (IntType(),), decl_id=2),
            _origin(engine, 7),
        )

        assert engine.is_solved(variable) is False
        assert engine.zonk(variable) == engine.zonk(other)


def test_mixed_provisional_literal_elements_report_a_type_error() -> None:
    with pytest.raises(AglTypeError, match="Array literal elements"):
        resolve_and_check_inline_entry(
            "def id[T](value: T) -> T = value\nlet values = [id(?), id(1)]\nvalues",
            HostCapabilities(),
        )


def test_method_with_inferred_return_uses_receiver_header_type() -> None:
    """Candidate inference retains the receiver's rigid generic slot in its body."""
    checked = resolve_and_check_inline_entry(
        "record Box[T]\n"
        "  value: T\n"
        "def Box::get[E](self) = self.value\n"
        "let box = Box(value = 1)\n"
        "Box::get(box)",
        HostCapabilities(),
    )

    method = next(
        item
        for item in checked.resolved.program.body.items
        if isinstance(item, FuncDef) and item.name == "get"
    )
    signature = checked.type_env.get_function_signature_by_node_id(method.node_id)
    assert signature is not None
    assert signature.result == TypeVarType("E")
    assert strip_decl_ids(signature.params[0].type) == RecordType("Box", (TypeVarType("E"),))


def test_generic_lambda_widens_a_member_result_before_function_constraint() -> None:
    """A lambda result widens directly before its generic function slot is unified."""
    checked = resolve_and_check_inline_entry(
        "def apply[U](f: (int) -> Result[U, text]) -> Result[U, text] = f(1)\n"
        "apply(fn(value: int) => Result::Ok(value = value))",
        HostCapabilities(),
    )

    result = checked.resolved.program.body.items[-1]
    result_type = checked.node_types[result.node_id]
    assert isinstance(result_type, EnumType)
    assert result_type.name == "Result"
    assert result_type.type_args == (IntType(), TextType())


def test_generic_lambda_rejects_a_member_result_that_conflicts_with_prior_evidence() -> None:
    """Direct member widening still reports conflicts from an earlier generic argument."""
    with pytest.raises(AglTypeError):
        resolve_and_check_inline_entry(
            "def apply[T](value: T, f: () -> Result[T, text]) -> Result[T, text] = f()\n"
            'apply("text", fn() => Result::Ok(value = 1))',
            HostCapabilities(),
        )


def test_contextual_lambda_with_bottom_body_keeps_its_concrete_result_type() -> None:
    """A lambda which always raises still adopts its concrete function result context."""
    checked = resolve_and_check_inline_entry(
        "def apply(f: (int) -> Result[int, text]) -> Result[int, text] = f(1)\n"
        'apply(fn(value: int) => raise Abort(message = "failed"))',
        HostCapabilities(),
    )

    result = checked.resolved.program.body.items[-1]
    result_type = checked.node_types[result.node_id]
    assert isinstance(result_type, EnumType)
    assert result_type.name == "Result"
    assert result_type.type_args == (IntType(), TextType())


def test_bound_generic_method_pins_receiver_and_inferrs_own_type_parameter() -> None:
    """Only method parameters beyond the receiver are inferred at member access."""
    checked = resolve_and_check_inline_entry(
        "record Box[T]\n"
        "  value: T\n"
        "def Box::map[T, U](self, f: (T) -> U) -> Box[U] = Box(value = f(self.value))\n"
        "let box = Box(value = 1)\n"
        'box.map(fn(value: int) -> text => "value")',
        HostCapabilities(),
    )

    result = checked.resolved.program.body.items[-1]
    assert strip_decl_ids(checked.node_types[result.node_id]) == RecordType("Box", (TextType(),))


def test_bound_generic_method_accepts_explicit_own_type_parameter() -> None:
    """Explicit member instantiation supplies only parameters not pinned by the receiver."""
    checked = resolve_and_check_inline_entry(
        "record Box[T]\n"
        "  value: T\n"
        "def Box::map[T, U](self, f: (T) -> U) -> Box[U] = Box(value = f(self.value))\n"
        "let box = Box(value = 1)\n"
        'box.map::[text](fn(value: int) -> text => "value")',
        HostCapabilities(),
    )

    result = checked.resolved.program.body.items[-1]
    assert strip_decl_ids(checked.node_types[result.node_id]) == RecordType("Box", (TextType(),))


def test_qualified_generic_member_infers_without_shadowing_a_bare_builtin() -> None:
    """A qualified generic function owns its spelling, while bare ``render`` stays builtin."""
    checked = resolve_and_check_inline_entry(
        "use Codec::*\n"
        "\n"
        "scope Codec\n"
        "  def render[T](value: T) -> array[T] = [value]\n"
        "end Codec\n"
        "\n"
        "let values = Codec::render(1)\n"
        "let text: text = render(1)\n"
        "values",
        HostCapabilities(),
    )

    result = checked.resolved.program.body.items[-1]
    assert strip_decl_ids(checked.node_types[result.node_id]) == ArrayType(IntType())


class TestFinalizationAndProvenance:
    def test_solved_query_rejects_nested_unresolved_solution(self) -> None:
        engine = InferenceEngine()
        result = engine.fresh("result")
        element = engine.fresh("element")
        engine.unify(result, ArrayType(element), _origin(engine, 1))

        assert engine.is_solved(result) is False

        engine.unify(element, IntType(), _origin(engine, 2))

        assert engine.is_solved(result) is True

    def test_zonk_compresses_links_and_rebuilds_nested_types(self) -> None:
        engine = InferenceEngine()
        first = engine.fresh("T")
        second = engine.fresh("U")
        third = engine.fresh("V")
        engine.unify(first, second, _origin(engine, 1))
        engine.unify(second, third, _origin(engine, 2))
        engine.unify(third, IntType(), _origin(engine, 3))

        zonked = engine.zonk(FunctionType((first,), ArrayType(second)))

        assert zonked == FunctionType((IntType(),), ArrayType(IntType()))
        assert engine.zonk(first) == IntType()
        assert engine.zonk(third) == IntType()

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
        engine.unify(outer, ArrayType(inner), _origin(engine, 1))
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
        first = engine.origin(_span(1), role=ConstraintRole.LITERAL_ELEMENT, subject="array")
        second = engine.origin(_span(2), role=ConstraintRole.EXPECTED_RESULT, subject="id")

        assert (first.sequence, second.sequence) == (0, 1)
        assert first.role is ConstraintRole.LITERAL_ELEMENT
