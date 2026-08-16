"""Static JSON encode-plan derivation and runtime execution tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from agm.agl.eval.conversions import run_recipe
from agm.agl.eval.ir_interpreter import _apply_coercion
from agm.agl.ir import contracts
from agm.agl.ir.contracts import (
    ArrayEncode,
    ConversionRecipe,
    ConversionStrategy,
    DictEncode,
    DynamicApplyEncode,
    DynamicArrayEncode,
    DynamicDictEncode,
    DynamicEncodeDefinition,
    DynamicEncodePlan,
    DynamicEnumEncode,
    DynamicExceptionEncode,
    DynamicRecordEncode,
    DynamicTypeParameterEncode,
    DynamicVariantEncode,
    EncodePlan,
    EnumEncode,
    ExceptionEncode,
    RecordEncode,
    RefEncode,
    ScalarEncode,
    VariantEncode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.ir.operations import ToJson
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.params import decode_param_value
from agm.agl.runtime.serialize import (
    dumps_exact,
    encode_dynamic_value,
    encode_value,
    value_to_json_obj,
)
from agm.agl.semantics.type_table import TypeDef
from agm.agl.semantics.types import (
    ArrayType,
    DictType,
    ExceptionType,
    IntType,
    RecordType,
    UnitType,
)
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    DecimalValue,
    DictValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
)
from agm.agl.type_schema import build_dynamic_encode_plan, build_encode_plan, build_param_decoder
from tests._agl_helpers import enum_type, next_decl_id, record_type, type_table_for
from tests.agl.ir_harness import (
    evaluate_ir,
    evaluate_ir_with_agents,
    evaluate_ir_with_externs,
)


def test_contracts_exports_array_encode() -> None:
    assert "ArrayEncode" in contracts.__all__


def test_encode_plan_derives_enum_members_and_nested_records() -> None:
    item, item_def = record_type("Item", {"value": IntType()})
    choice, choice_def = enum_type(
        "Choice",
        {"One": {"item": item}, "Many": {"items": ArrayType(item)}},
    )

    plan = build_encode_plan(choice, type_table_for(item_def, choice_def))

    assert isinstance(plan.root, EnumEncode)
    assert [(variant.name, variant.nominal) for variant in plan.root.variants] == [
        ("One", NominalId(choice_def.members[0].decl_id)),
        ("Many", NominalId(choice_def.members[1].decl_id)),
    ]
    many = plan.root.variants[1]
    assert isinstance(many.fields[0][1], ArrayEncode)
    assert isinstance(many.fields[0][1].elem, RecordEncode)


def test_encode_plan_preserves_enum_tags_for_member_records() -> None:
    """Static enum slots add the legacy tag while plain records remain untagged."""
    item, item_def = record_type("Item", {"value": IntType()})
    choice, choice_def = enum_type(
        "Choice",
        {"One": {"item": item}, "Many": {"items": ArrayType(item)}},
    )
    problem_id = next_decl_id()
    problem = ExceptionType(name="Problem", decl_id=problem_id)
    problem_def = TypeDef(
        kind="exception",
        name="Problem",
        module_id=ENTRY_ID,
        fields=(("choice", choice),),
        decl_node_id=problem_id,
    )
    table = type_table_for(item_def, choice_def, problem_def)
    item_value = RecordValue(NominalId(item.decl_id), "Item", {"value": IntValue(7)})
    choice_value = RecordValue(
        nominal=NominalId(choice_def.members[1].decl_id),
        display_name="Choice::Many",
        fields={"items": ArrayValue([item_value])},
    )
    problem_value = ExceptionValue(NominalId(problem.decl_id), "Problem", {"choice": choice_value})

    assert value_to_json_obj(choice_value) == {"items": [{"value": 7}]}
    assert encode_value(build_encode_plan(choice, table), choice_value) == {
        "$case": "Many",
        "items": [{"value": 7}],
    }
    assert encode_value(build_encode_plan(problem, table), problem_value) == {
        "choice": {"$case": "Many", "items": [{"value": 7}]}
    }
    assert encode_dynamic_value(build_dynamic_encode_plan(problem, table), problem_value) == {
        "choice": {"$case": "Many", "items": [{"value": 7}]}
    }


def test_encode_plan_executes_all_shapes_and_member_identity() -> None:
    scalar = EncodePlan(ScalarEncode())
    assert [
        encode_value(scalar, value)
        for value in (
            TextValue("text"),
            IntValue(2),
            DecimalValue(Decimal("2.5")),
            BoolValue(True),
            JsonValue({"raw": None}),
        )
    ] == ["text", 2, Decimal("2.5"), True, {"raw": None}]

    assert encode_value(
        EncodePlan(ArrayEncode(ScalarEncode())), ArrayValue([IntValue(1), IntValue(2)])
    ) == [1, 2]
    assert encode_value(EncodePlan(DictEncode(ScalarEncode())), DictValue({"n": IntValue(3)})) == {
        "n": 3
    }

    enum = EncodePlan(EnumEncode(NominalId(1), (VariantEncode("Member", NominalId(2), ()),)))
    assert encode_value(
        enum, RecordValue(nominal=NominalId(2), display_name=f"{'E'}::{'Member'}", fields={})
    ) == {"$case": "Member"}


def test_to_json_coercion_uses_the_static_scalar_encoder() -> None:
    assert _apply_coercion(IntValue(1), ToJson()) == JsonValue(1)
    with pytest.raises(AssertionError, match="scalar encode"):
        _apply_coercion(ArrayValue([]), ToJson())


def test_to_json_recipe_requires_the_lowered_encode_plan() -> None:
    with pytest.raises(AssertionError, match="requires an encode plan"):
        run_recipe(
            ConversionRecipe(
                strategy=ConversionStrategy.TO_JSON,
                source_label="int",
                target_label="json",
            ),
            IntValue(1),
        )


def test_encode_plan_reports_malformed_static_plans() -> None:
    scalar = ScalarEncode()
    enum = EnumEncode(NominalId(1), (VariantEncode("A", NominalId(2), ()),))
    cases = (
        (scalar, ArrayValue([])),
        (ArrayEncode(scalar), IntValue(1)),
        (DictEncode(scalar), IntValue(1)),
        (RecordEncode(NominalId(1), ()), IntValue(1)),
        (RecordEncode(NominalId(1), ()), RecordValue(NominalId(2), "Other", {})),
        (ExceptionEncode(NominalId(1), ()), IntValue(1)),
        (ExceptionEncode(NominalId(1), ()), ExceptionValue(NominalId(2), "Other", {})),
        (enum, RecordValue(nominal=NominalId(1), display_name=f"{'E'}::{'Other'}", fields={})),
        (enum, RecordValue(nominal=NominalId(1), display_name=f"{'E'}::{'Other'}", fields={})),
        (enum, RecordValue(nominal=NominalId(3), display_name=f"{'Other'}::{'Member'}", fields={})),
        (enum, IntValue(1)),
    )
    for schema, value in cases:
        with pytest.raises(AssertionError):
            encode_value(EncodePlan(schema), value)

    for plan in (
        EncodePlan(RefEncode("missing")),
        EncodePlan(
            RefEncode("first"),
            (("first", RefEncode("second")), ("second", RefEncode("first"))),
        ),
    ):
        with pytest.raises(AssertionError):
            encode_value(plan, IntValue(1))


def test_encode_plan_preserves_legacy_json_bytes_for_a_complete_corpus() -> None:
    """Plans retain legacy member tags and declaration-order object bytes."""
    leaf, leaf_def = record_type("Leaf", {})
    box, box_def = record_type("Box", {"item": leaf})
    choice, choice_def = enum_type(
        "Choice",
        {"Empty": {}, "One": {"box": box}, "Many": {"boxes": ArrayType(box)}},
    )
    envelope, envelope_def = record_type("Envelope", {"choice": choice})
    tree_id = next_decl_id()
    tree = RecordType(name="Tree", decl_id=tree_id)
    tree_def = TypeDef(
        kind="record",
        name="Tree",
        module_id=ENTRY_ID,
        fields=(("children", ArrayType(tree)),),
        decl_node_id=tree_id,
    )
    table = type_table_for(leaf_def, box_def, choice_def, envelope_def, tree_def)
    leaf_value = RecordValue(NominalId(leaf.decl_id), "Leaf", {})
    box_value = RecordValue(NominalId(box.decl_id), "Box", {"item": leaf_value})
    one_value = RecordValue(
        nominal=NominalId(choice_def.members[1].decl_id),
        display_name=f"{'Choice'}::{'One'}",
        fields={"box": box_value},
    )
    cases: tuple[tuple[object, object, str], ...] = (
        (leaf, leaf_value, "{}"),
        (
            choice,
            RecordValue(
                nominal=NominalId(choice_def.members[0].decl_id),
                display_name=f"{'Choice'}::{'Empty'}",
                fields={},
            ),
            '{"$case": "Empty"}',
        ),
        (
            choice,
            one_value,
            '{"$case": "One", "box": {"item": {}}}',
        ),
        (
            envelope,
            RecordValue(NominalId(envelope.decl_id), "Envelope", {"choice": one_value}),
            '{"choice": {"$case": "One", "box": {"item": {}}}}',
        ),
        (
            ArrayType(choice),
            ArrayValue(
                [
                    RecordValue(
                        nominal=NominalId(choice_def.members[0].decl_id),
                        display_name=f"{'Choice'}::{'Empty'}",
                        fields={},
                    ),
                    one_value,
                ]
            ),
            '[{"$case": "Empty"}, {"$case": "One", "box": {"item": {}}}]',
        ),
        (
            DictType(choice),
            DictValue({"primary": one_value}),
            '{"primary": {"$case": "One", "box": {"item": {}}}}',
        ),
        (
            choice,
            RecordValue(
                nominal=NominalId(choice_def.members[2].decl_id),
                display_name=f"{'Choice'}::{'Many'}",
                fields={"boxes": ArrayValue([box_value])},
            ),
            '{"$case": "Many", "boxes": [{"item": {}}]}',
        ),
        (
            tree,
            RecordValue(
                NominalId(tree.decl_id),
                "Tree",
                {
                    "children": ArrayValue(
                        [RecordValue(NominalId(tree.decl_id), "Tree", {"children": ArrayValue([])})]
                    )
                },
            ),
            '{"children": [{"children": []}]}',
        ),
    )

    for typ, value, legacy_bytes in cases:
        assert (
            dumps_exact(encode_value(build_encode_plan(typ, table), value), indent=None)
            == legacy_bytes
        )


def test_encode_bytes_are_preserved_across_agent_request_and_parameter_boundaries() -> None:
    request_result = evaluate_ir(
        'let request = ask-request("hi", agent = AgentCommand(command = "fake"))\n'
        "let encoded = request as json\n"
        "()\n"
    )
    request_encoded = request_result["encoded"]
    assert isinstance(request_encoded, JsonValue)
    assert dumps_exact(request_encoded.raw, indent=None) == (
        '{"agent": {"$case": "AgentCommand", "command": "fake"}, "prompt": "hi", '
        '"target_type": {"$case": "Some", "value": "text"}, '
        '"format_instructions": {"$case": "None"}, "json_schema": {"$case": "None"}, '
        '"attempt": 0, "previous_error": {"$case": "None"}, '
        '"metadata": {"codec_name": "text", "strict_json": null, "structured_exec": false}}'
    )

    choice, choice_def = enum_type("Choice", {"None": {}, "One": {"value": IntType()}})
    table = type_table_for(choice_def)
    decoded = decode_param_value(
        build_param_decoder(choice, table),
        '{"$case": "One", "value": 2}',
    )
    assert dumps_exact(encode_value(build_encode_plan(choice, table), decoded), indent=None) == (
        '{"$case": "One", "value": 2}'
    )


def test_encode_bytes_are_preserved_across_mocked_agent_and_ffi_boundaries(
    tmp_path: Path,
) -> None:
    """Typed JSON casts retain the legacy wire bytes before host crossings."""
    agent_result = evaluate_ir_with_agents(
        "enum Choice | None | One(value: int)\n"
        "enum Tree | Leaf | Node(children: array[Tree])\n"
        "record Payload\n"
        "  choice: Choice\n"
        "  choices: array[Choice]\n"
        "  by_name: dict[text, Choice]\n"
        "  tree: Tree\n"
        'let payload: Payload = ask("payload", agent = AgentCommand(command = "fake"))\n'
        "let encoded = payload as json\n"
        "()\n",
        {
            "fake": [
                '{"choice":{"$case":"One","value":2},"choices":[{"$case":"None"},'
                '{"$case":"One","value":3}],"by_name":{"primary":{"$case":"One",'
                '"value":4}},"tree":{"$case":"Node","children":[{"$case":"Leaf"}]}}'
            ]
        },
    )
    agent_encoded = agent_result["encoded"]
    assert isinstance(agent_encoded, JsonValue)
    assert dumps_exact(agent_encoded.raw, indent=None) == (
        '{"choice": {"$case": "One", "value": 2}, "choices": [{"$case": "None"}, '
        '{"$case": "One", "value": 3}], "by_name": {"primary": {"$case": "One", '
        '"value": 4}}, "tree": {"$case": "Node", "children": [{"$case": "Leaf"}]}}'
    )

    ffi_result, _registry = evaluate_ir_with_externs(
        "enum Choice | None | One(value: int)\n"
        "enum Tree | Leaf | Node(children: array[Tree])\n"
        "record Payload\n"
        "  choice: Choice\n"
        "  choices: array[Choice]\n"
        "  by_name: dict[text, Choice]\n"
        "  tree: Tree\n"
        "extern def relay(value: json) -> json\n"
        "let payload = Payload(\n"
        "  choice = Choice::One(value = 2),\n"
        "  choices = [Choice::None, Choice::One(value = 3)],\n"
        '  by_name = {"primary": Choice::One(value = 4)},\n'
        "  tree = Tree::Node(children = [Tree::Leaf])\n"
        ")\n"
        "let encoded = relay(payload as json)\n"
        "()\n",
        "from agl import json\ndef relay(value): return value\n",
        tmp_path,
    )
    ffi_encoded = ffi_result["encoded"]
    assert isinstance(ffi_encoded, JsonValue)
    assert dumps_exact(ffi_encoded.raw, indent=None) == (
        '{"choice": {"$case": "One", "value": 2}, "choices": [{"$case": "None"}, '
        '{"$case": "One", "value": 3}], "by_name": {"primary": {"$case": "One", '
        '"value": 4}}, "tree": {"$case": "Node", "children": [{"$case": "Leaf"}]}}'
    )


def test_lowered_json_cast_preserves_legacy_bytes_for_a_recursive_enum() -> None:
    result = evaluate_ir(
        "enum Tree | Leaf | Node(children: array[Tree])\n"
        "let tree: Tree = Tree::Node(children = [Tree::Leaf, Tree::Node(children = [])])\n"
        "let encoded = tree as json\n"
        "()\n"
    )

    encoded = result["encoded"]
    assert isinstance(encoded, JsonValue)
    assert dumps_exact(encoded.raw, indent=None) == (
        '{"$case": "Node", "children": [{"$case": "Leaf"}, {"$case": "Node", "children": []}]}'
    )


def test_dynamic_plan_distinguishes_record_and_enum_slots_for_a_shared_member() -> None:
    """A member nominal gets a tag only where the static slot is an enum."""
    envelope = NominalId(1)
    enum = NominalId(2)
    member = NominalId(3)
    plan = DynamicEncodePlan(
        root=DynamicApplyEncode(envelope, ()),
        definitions=(
            DynamicEncodeDefinition(
                envelope,
                0,
                DynamicRecordEncode(
                    envelope,
                    (
                        ("plain", DynamicRecordEncode(member, (("value", ScalarEncode()),))),
                        (
                            "selected",
                            DynamicEnumEncode(
                                enum,
                                (
                                    DynamicVariantEncode(
                                        "Shared", member, (("value", ScalarEncode()),)
                                    ),
                                ),
                            ),
                        ),
                        ("items", DynamicArrayEncode(ScalarEncode())),
                        ("by_name", DynamicDictEncode(ScalarEncode())),
                    ),
                ),
            ),
        ),
    )
    shared = RecordValue(member, "other::name", {"value": IntValue(7)})

    assert encode_dynamic_value(
        plan,
        RecordValue(
            envelope,
            "Envelope",
            {
                "plain": shared,
                "selected": shared,
                "items": ArrayValue([IntValue(1)]),
                "by_name": DictValue({"n": IntValue(2)}),
            },
        ),
    ) == {
        "plain": {"value": 7},
        "selected": {"$case": "Shared", "value": 7},
        "items": [1],
        "by_name": {"n": 2},
    }


def test_dynamic_encode_plan_rejects_malformed_runtime_shapes() -> None:
    """Dynamic plans retain the same runtime shape checks as static plans."""
    nominal = NominalId(1)
    member = NominalId(2)
    cases = (
        (DynamicEncodePlan(DynamicTypeParameterEncode(0), ()), IntValue(1)),
        (DynamicEncodePlan(DynamicApplyEncode(nominal, ()), ()), IntValue(1)),
        (
            DynamicEncodePlan(
                DynamicApplyEncode(nominal, ()),
                (DynamicEncodeDefinition(nominal, 1, ScalarEncode()),),
            ),
            IntValue(1),
        ),
        (DynamicEncodePlan(DynamicArrayEncode(ScalarEncode()), ()), IntValue(1)),
        (DynamicEncodePlan(DynamicDictEncode(ScalarEncode()), ()), IntValue(1)),
        (DynamicEncodePlan(DynamicRecordEncode(nominal, ()), ()), IntValue(1)),
        (DynamicEncodePlan(DynamicExceptionEncode(nominal, ()), ()), IntValue(1)),
        (DynamicEncodePlan(DynamicEnumEncode(nominal, ()), ()), IntValue(1)),
        (
            DynamicEncodePlan(
                DynamicEnumEncode(nominal, (DynamicVariantEncode("Case", member, ()),)),
                (),
            ),
            RecordValue(NominalId(3), "Other", {}),
        ),
        (
            DynamicEncodePlan(
                DynamicApplyEncode(nominal, (DynamicTypeParameterEncode(0),)),
                (DynamicEncodeDefinition(nominal, 1, ScalarEncode()),),
            ),
            IntValue(1),
        ),
    )
    for plan, value in cases:
        with pytest.raises(AssertionError):
            encode_dynamic_value(plan, value)


def test_build_dynamic_encode_plan_rejects_unbound_or_unknown_types() -> None:
    from agm.agl.semantics.types import TypeVarType

    table = type_table_for()
    with pytest.raises(AssertionError):
        build_dynamic_encode_plan(TypeVarType("T"), table)
    with pytest.raises(AssertionError):
        build_dynamic_encode_plan(RecordType("Ghost", decl_id=999), table)
    with pytest.raises(AssertionError):
        build_dynamic_encode_plan(UnitType(), table)


def test_dynamic_application_instantiates_every_composite_argument_shape() -> None:
    root = NominalId(1)
    record = DynamicRecordEncode(NominalId(2), (("value", ScalarEncode()),))
    exception = DynamicExceptionEncode(NominalId(3), (("value", ScalarEncode()),))
    enum = DynamicEnumEncode(
        NominalId(4), (DynamicVariantEncode("Case", NominalId(5), (("value", ScalarEncode()),)),)
    )
    definition = DynamicEncodeDefinition(root, 1, DynamicTypeParameterEncode(0))
    cases = (
        (record, RecordValue(NominalId(2), "Record", {"value": IntValue(1)}), {"value": 1}),
        (
            exception,
            ExceptionValue(NominalId(3), "Problem", {"value": IntValue(2)}),
            {"value": 2},
        ),
        (
            enum,
            RecordValue(NominalId(5), "Case", {"value": IntValue(3)}),
            {"$case": "Case", "value": 3},
        ),
    )
    for argument, value, expected in cases:
        assert (
            encode_dynamic_value(
                DynamicEncodePlan(DynamicApplyEncode(root, (argument,)), (definition,)), value
            )
            == expected
        )


def test_growing_polymorphic_recursive_json_cast_lowers_and_evaluates() -> None:
    """A finite Perfect[int] value needs no finite plan for all possible values."""
    result = evaluate_ir(
        "record Pair[A, B]\n"
        "  first: A\n"
        "  second: B\n"
        "enum Perfect[T]\n"
        "  | Single(value: T)\n"
        "  | Succ(next: Perfect[Pair[array[T], dict[text, T]]])\n"
        'let p: Perfect[int] = Succ(next = Single(value = Pair(first = [1], second = {"x": 2})))\n'
        "let encoded = p as json\n"
        "()\n"
    )

    encoded = result["encoded"]
    assert isinstance(encoded, JsonValue)
    assert dumps_exact(encoded.raw, indent=None) == (
        '{"$case": "Succ", "next": {"$case": "Single", "value": '
        '{"first": [1], "second": {"x": 2}}}}'
    )


def test_encode_plan_rejects_a_non_json_type() -> None:
    with pytest.raises(AssertionError, match="unencodable"):
        build_encode_plan(UnitType(), type_table_for())


def test_encode_plan_handles_recursive_containers() -> None:
    recursive_id = next_decl_id()
    recursive = RecordType(name="Recursive", decl_id=recursive_id)
    recursive_def = TypeDef(
        kind="record",
        name="Recursive",
        module_id=ENTRY_ID,
        fields=(("children", ArrayType(recursive)),),
        decl_node_id=recursive_id,
    )
    value = RecordValue(
        NominalId(recursive_id),
        "Recursive",
        {
            "children": ArrayValue(
                [RecordValue(NominalId(recursive_id), "Recursive", {"children": ArrayValue([])})]
            )
        },
    )

    plan = build_encode_plan(recursive, type_table_for(recursive_def))

    assert isinstance(plan.root, RefEncode)
    assert encode_value(plan, value) == value_to_json_obj(value)
