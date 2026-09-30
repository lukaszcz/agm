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
    EncodeDefinition,
    EncodePlan,
    EnumEncode,
    ExceptionEncode,
    FieldEncode,
    RecordEncode,
    RefEncode,
    ScalarEncode,
    TypeParameterEncode,
    VariantEncode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.ir.operations import ToJson
from agm.agl.ir.program import ValueDescriptors
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.arguments import decode_param_value
from agm.agl.runtime.serialize import (
    dumps_exact,
    encode_value,
    value_to_json_obj,
)
from agm.agl.semantics.external_names import ExternalName
from agm.agl.semantics.type_table import TypeDef
from agm.agl.semantics.types import (
    ArrayType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    RecordType,
    TypeVarType,
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
from agm.agl.type_schema import (
    _build_template_encode_plan,
    build_encode_plan,
    build_param_decoder,
)
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
    # ``Item`` occurs in both variants, so it is emitted once into ``defs`` and
    # referenced from each occurrence rather than inlined twice.
    many = plan.root.variants[1]
    assert isinstance(many.fields[0].schema, ArrayEncode)
    assert many.fields[0].schema.elem == RefEncode("Item")
    one = plan.root.variants[0]
    assert one.fields[0].schema == RefEncode("Item")
    assert [definition.key for definition in plan.definitions] == ["Item"]
    (item_definition,) = plan.definitions
    assert item_definition.parameter_count == 0
    assert isinstance(item_definition.body, RecordEncode)


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
    item_value = RecordValue(NominalId(item.decl_id), {"value": IntValue(7)})
    choice_value = RecordValue(
        nominal=NominalId(choice_def.members[1].decl_id),
        fields={"items": ArrayValue([item_value])},
    )
    problem_value = ExceptionValue(NominalId(problem.decl_id), {"choice": choice_value})

    assert value_to_json_obj(choice_value) == {"items": [{"value": 7}]}
    assert encode_value(build_encode_plan(choice, table), choice_value) == {
        "$case": "Many",
        "items": [{"value": 7}],
    }
    assert encode_value(build_encode_plan(problem, table), problem_value) == {
        "choice": {"$case": "Many", "items": [{"value": 7}]}
    }
    # The template builder is what a growing source gets; on a source both
    # builders accept it must agree, exception root and enum slot alike.
    assert encode_value(_build_template_encode_plan(problem, table), problem_value) == {
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

    enum = EncodePlan(
        EnumEncode(NominalId(1), (VariantEncode("Member", "Member", NominalId(2), ()),))
    )
    assert encode_value(enum, RecordValue(nominal=NominalId(2), fields={})) == {"$case": "Member"}


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
            ValueDescriptors(nominals={}, functions={}),
        )


def test_encode_plan_reports_malformed_static_plans() -> None:
    scalar = ScalarEncode()
    enum = EnumEncode(NominalId(1), (VariantEncode("A", "A", NominalId(2), ()),))
    cases = (
        (scalar, ArrayValue([])),
        (ArrayEncode(scalar), IntValue(1)),
        (DictEncode(scalar), IntValue(1)),
        (RecordEncode(NominalId(1), ()), IntValue(1)),
        (RecordEncode(NominalId(1), ()), RecordValue(NominalId(2), {})),
        (ExceptionEncode(NominalId(1), ()), IntValue(1)),
        (ExceptionEncode(NominalId(1), ()), ExceptionValue(NominalId(2), {})),
        (enum, RecordValue(nominal=NominalId(1), fields={})),
        (enum, RecordValue(nominal=NominalId(1), fields={})),
        (enum, RecordValue(nominal=NominalId(3), fields={})),
        (enum, IntValue(1)),
    )
    for schema, value in cases:
        with pytest.raises(AssertionError):
            encode_value(EncodePlan(schema), value)

    for plan in (
        EncodePlan(RefEncode("missing")),
        EncodePlan(
            RefEncode("first"),
            (
                EncodeDefinition("first", 0, RefEncode("second")),
                EncodeDefinition("second", 0, RefEncode("first")),
            ),
        ),
    ):
        with pytest.raises(AssertionError):
            encode_value(plan, IntValue(1))


def test_encode_plan_binds_definition_parameters_at_each_reference() -> None:
    """A parameterized definition encodes its slots through the caller's arguments."""
    pair = NominalId(1)
    plan = EncodePlan(
        root=RefEncode("Pair", (ScalarEncode(), ArrayEncode(ScalarEncode()))),
        definitions=(
            EncodeDefinition(
                "Pair",
                2,
                RecordEncode(
                    pair,
                    (
                        FieldEncode("first", "first", TypeParameterEncode(0)),
                        FieldEncode("second", "second", TypeParameterEncode(1)),
                    ),
                ),
            ),
        ),
    )
    value = RecordValue(
        pair, {"first": IntValue(1), "second": ArrayValue([IntValue(2), IntValue(3)])}
    )

    assert encode_value(plan, value) == {"first": 1, "second": [2, 3]}


def test_encode_plan_substitutes_arguments_through_every_composite_shape() -> None:
    """An argument is rewritten out of the caller's parameter space before it binds."""
    outer = NominalId(1)
    record = NominalId(2)
    exception = NominalId(3)
    enum = NominalId(4)
    member = NominalId(5)
    # ``Outer`` hands ``Inner`` a composite written in terms of ITS OWN
    # parameter, and ``Inner``'s body is just that parameter — so each case
    # encodes correctly only if the substitution reached every leaf position
    # inside the composite it was given.
    cases = (
        (ArrayEncode(TypeParameterEncode(0)), ArrayValue([IntValue(4)]), [4]),
        (DictEncode(TypeParameterEncode(0)), DictValue({"n": IntValue(5)}), {"n": 5}),
        (
            RecordEncode(record, (FieldEncode("value", "value", TypeParameterEncode(0)),)),
            RecordValue(record, {"value": IntValue(1)}),
            {"value": 1},
        ),
        (
            ExceptionEncode(exception, (FieldEncode("value", "value", TypeParameterEncode(0)),)),
            ExceptionValue(exception, {"value": IntValue(2)}),
            {"value": 2},
        ),
        (
            EnumEncode(
                enum,
                (
                    VariantEncode(
                        "Case",
                        "Case",
                        member,
                        (FieldEncode("value", "value", TypeParameterEncode(0)),),
                    ),
                ),
            ),
            RecordValue(member, {"value": IntValue(3)}),
            {"$case": "Case", "value": 3},
        ),
        (RefEncode("Inner", (TypeParameterEncode(0),)), IntValue(6), 6),
        (TypeParameterEncode(0), IntValue(7), 7),
    )
    for composite, value, expected in cases:
        plan = EncodePlan(
            root=RefEncode("Outer", (ScalarEncode(),)),
            definitions=(
                EncodeDefinition("Inner", 1, TypeParameterEncode(0)),
                EncodeDefinition(
                    "Outer",
                    1,
                    RecordEncode(
                        outer, (FieldEncode("held", "held", RefEncode("Inner", (composite,))),)
                    ),
                ),
            ),
        )
        assert encode_value(plan, RecordValue(outer, {"held": value})) == {"held": expected}


def test_encode_plan_reports_malformed_parameterized_plans() -> None:
    """An unbound parameter or a mis-applied definition is an internal-invariant violation."""
    box = EncodeDefinition("Box", 1, ScalarEncode())
    cases = (
        # A parameter at the root, where nothing binds it.
        EncodePlan(TypeParameterEncode(0)),
        # A reference supplying no argument for the definition's parameter.
        EncodePlan(RefEncode("Box"), (box,)),
        # An argument naming a parameter the referring position does not have.
        EncodePlan(RefEncode("Box", (TypeParameterEncode(0),)), (box,)),
    )
    for plan in cases:
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
    leaf_value = RecordValue(NominalId(leaf.decl_id), {})
    box_value = RecordValue(NominalId(box.decl_id), {"item": leaf_value})
    one_value = RecordValue(
        nominal=NominalId(choice_def.members[1].decl_id),
        fields={"box": box_value},
    )
    cases: tuple[tuple[object, object, str], ...] = (
        (leaf, leaf_value, "{}"),
        (
            choice,
            RecordValue(
                nominal=NominalId(choice_def.members[0].decl_id),
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
            RecordValue(NominalId(envelope.decl_id), {"choice": one_value}),
            '{"choice": {"$case": "One", "box": {"item": {}}}}',
        ),
        (
            ArrayType(choice),
            ArrayValue(
                [
                    RecordValue(
                        nominal=NominalId(choice_def.members[0].decl_id),
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
                fields={"boxes": ArrayValue([box_value])},
            ),
            '{"$case": "Many", "boxes": [{"item": {}}]}',
        ),
        (
            tree,
            RecordValue(
                NominalId(tree.decl_id),
                {
                    "children": ArrayValue(
                        [RecordValue(NominalId(tree.decl_id), {"children": ArrayValue([])})]
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
        '"target-type": {"$case": "Some", "value": "text"}, '
        '"format-instructions": {"$case": "None"}, "json-schema": {"$case": "None"}, '
        '"attempt": 0, "previous-error": {"$case": "None"}, '
        '"metadata": {"codec_name": "text", "strict_json": null, "structured_exec": false, '
        '"max_attempts": 1}, '
        '"sandbox": {"$case": "Sandbox", "memory": {"$case": "Default"}, '
        '"swap": {"$case": "Default"}, "settings": {"$case": "None"}, "patch": true}}'
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
        "  by-name: dict[text, Choice]\n"
        "  tree: Tree\n"
        'let payload: Payload = ask("payload", agent = AgentCommand(command = "fake"))\n'
        "let encoded = payload as json\n"
        "()\n",
        {
            "fake": [
                '{"choice":{"$case":"One","value":2},"choices":[{"$case":"None"},'
                '{"$case":"One","value":3}],"by-name":{"primary":{"$case":"One",'
                '"value":4}},"tree":{"$case":"Node","children":[{"$case":"Leaf"}]}}'
            ]
        },
    )
    agent_encoded = agent_result["encoded"]
    assert isinstance(agent_encoded, JsonValue)
    assert dumps_exact(agent_encoded.raw, indent=None) == (
        '{"choice": {"$case": "One", "value": 2}, "choices": [{"$case": "None"}, '
        '{"$case": "One", "value": 3}], "by-name": {"primary": {"$case": "One", '
        '"value": 4}}, "tree": {"$case": "Node", "children": [{"$case": "Leaf"}]}}'
    )

    ffi_result, _registry = evaluate_ir_with_externs(
        "enum Choice | None | One(value: int)\n"
        "enum Tree | Leaf | Node(children: array[Tree])\n"
        "record Payload\n"
        "  choice: Choice\n"
        "  choices: array[Choice]\n"
        "  by-name: dict[text, Choice]\n"
        "  tree: Tree\n"
        "extern def relay(value: json) -> json\n"
        "let payload = Payload(\n"
        "  choice = Choice::One(value = 2),\n"
        "  choices = [Choice::None, Choice::One(value = 3)],\n"
        '  by-name = {"primary": Choice::One(value = 4)},\n'
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
        '{"$case": "One", "value": 3}], "by-name": {"primary": {"$case": "One", '
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


def test_encode_plan_distinguishes_record_and_enum_slots_for_a_shared_member() -> None:
    """A member nominal gets a tag only where the static slot is an enum."""
    envelope = NominalId(1)
    enum = NominalId(2)
    member = NominalId(3)
    plan = EncodePlan(
        root=RefEncode("Envelope"),
        definitions=(
            EncodeDefinition(
                "Envelope",
                0,
                RecordEncode(
                    envelope,
                    (
                        FieldEncode(
                            "plain",
                            "plain",
                            RecordEncode(member, (FieldEncode("value", "value", ScalarEncode()),)),
                        ),
                        FieldEncode(
                            "selected",
                            "selected",
                            EnumEncode(
                                enum,
                                (
                                    VariantEncode(
                                        "Shared",
                                        "Shared",
                                        member,
                                        (FieldEncode("value", "value", ScalarEncode()),),
                                    ),
                                ),
                            ),
                        ),
                        FieldEncode("items", "items", ArrayEncode(ScalarEncode())),
                        FieldEncode("by-name", "by-name", DictEncode(ScalarEncode())),
                    ),
                ),
            ),
        ),
    )
    shared = RecordValue(member, {"value": IntValue(7)})

    assert encode_value(
        plan,
        RecordValue(
            envelope,
            {
                "plain": shared,
                "selected": shared,
                "items": ArrayValue([IntValue(1)]),
                "by-name": DictValue({"n": IntValue(2)}),
            },
        ),
    ) == {
        "plain": {"value": 7},
        "selected": {"$case": "Shared", "value": 7},
        "items": [1],
        "by-name": {"n": 2},
    }


def test_encode_plan_detects_record_exception_and_enum_closed_cycles() -> None:
    from agm.agl.semantics.cycles import AglCyclicValue

    record = RecordValue(NominalId(1), {})
    record.fields["next"] = record
    exception = ExceptionValue(NominalId(2), {})
    exception.fields["cause"] = exception
    member = RecordValue(NominalId(4), {})
    member.fields["next"] = member

    plans = (
        (
            EncodePlan(
                RefEncode("Node"),
                (
                    EncodeDefinition(
                        "Node",
                        0,
                        RecordEncode(
                            NominalId(1), (FieldEncode("next", "next", RefEncode("Node")),)
                        ),
                    ),
                ),
            ),
            record,
        ),
        (
            EncodePlan(
                RefEncode("Problem"),
                (
                    EncodeDefinition(
                        "Problem",
                        0,
                        ExceptionEncode(
                            NominalId(2), (FieldEncode("cause", "cause", RefEncode("Problem")),)
                        ),
                    ),
                ),
            ),
            exception,
        ),
        (
            EncodePlan(
                RefEncode("Link"),
                (
                    EncodeDefinition(
                        "Link",
                        0,
                        EnumEncode(
                            NominalId(3),
                            (
                                VariantEncode(
                                    "Cell",
                                    "Cell",
                                    NominalId(4),
                                    (FieldEncode("next", "next", RefEncode("Link")),),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            member,
        ),
    )

    for plan, value in plans:
        with pytest.raises(AglCyclicValue):
            encode_value(plan, value)


def test_encode_plan_allows_a_record_diamond() -> None:
    leaf = NominalId(1)
    pair = NominalId(2)
    shared = RecordValue(leaf, {"value": IntValue(1)})
    value = RecordValue(pair, {"left": shared, "right": shared})
    plan = EncodePlan(
        RecordEncode(
            pair,
            (
                FieldEncode(
                    "left",
                    "left",
                    RecordEncode(leaf, (FieldEncode("value", "value", ScalarEncode()),)),
                ),
                FieldEncode(
                    "right",
                    "right",
                    RecordEncode(leaf, (FieldEncode("value", "value", ScalarEncode()),)),
                ),
            ),
        )
    )

    assert encode_value(plan, value) == {"left": {"value": 1}, "right": {"value": 1}}


def test_template_encode_plan_rejects_unbound_or_unknown_types() -> None:
    from agm.agl.semantics.types import TypeVarType

    table = type_table_for()
    with pytest.raises(AssertionError):
        _build_template_encode_plan(TypeVarType("T"), table)
    with pytest.raises(AssertionError):
        _build_template_encode_plan(RecordType("Ghost", decl_id=999), table)
    with pytest.raises(AssertionError):
        _build_template_encode_plan(UnitType(), table)


def test_template_encode_plan_uses_renamed_field() -> None:
    """The declaration-template plan for a growing source also JSON-keys by rename."""
    decl_id = next_decl_id()
    box = RecordType(name="Box", type_args=(IntType(),), decl_id=decl_id)
    box_def = TypeDef(
        kind="record",
        name="Box",
        module_id=ENTRY_ID,
        type_params=("T",),
        fields=(("value", TypeVarType("T")),),
        field_external_names=(("value", ExternalName(json_name="payload")),),
        decl_node_id=decl_id,
    )
    table = type_table_for(box_def)
    value = RecordValue(NominalId(decl_id), {"value": IntValue(9)})

    plan = _build_template_encode_plan(box, table)

    assert encode_value(plan, value) == {"payload": 9}


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
        {
            "children": ArrayValue(
                [RecordValue(NominalId(recursive_id), {"children": ArrayValue([])})]
            )
        },
    )

    plan = build_encode_plan(recursive, type_table_for(recursive_def))

    assert isinstance(plan.root, RefEncode)
    assert encode_value(plan, value) == value_to_json_obj(value)


def test_encode_plan_uses_effective_json_name_diverging_from_value_to_json_obj() -> None:
    """A renamed field's key diverges from ``value_to_json_obj``, which keeps declared names."""
    decl_id = next_decl_id()
    renamed = RecordType(name="Renamed", decl_id=decl_id)
    renamed_def = TypeDef(
        kind="record",
        name="Renamed",
        module_id=ENTRY_ID,
        fields=(("value", IntType()),),
        field_external_names=(("value", ExternalName(json_name="val")),),
        decl_node_id=decl_id,
    )
    value = RecordValue(NominalId(decl_id), {"value": IntValue(3)})

    plan = build_encode_plan(renamed, type_table_for(renamed_def))

    assert encode_value(plan, value) == {"val": 3}
    assert value_to_json_obj(value) == {"value": 3}
    assert encode_value(plan, value) != value_to_json_obj(value)


def test_encode_plan_uses_member_external_name_as_case_tag() -> None:
    """A renamed enum member's ``@name``/``@json-name`` becomes the ``$case`` tag."""
    enum_id = next_decl_id()
    member_id = next_decl_id()
    member = RecordType(name="One", module_id=ENTRY_ID, scope_path=("Choice",), decl_id=member_id)
    member_def = TypeDef(
        kind="record",
        name="One",
        module_id=ENTRY_ID,
        scope_path=("Choice",),
        external_name=ExternalName(json_name="uno"),
        decl_node_id=member_id,
    )
    choice = EnumType(name="Choice", decl_id=enum_id)
    choice_def = TypeDef(
        kind="enum", name="Choice", module_id=ENTRY_ID, members=(member,), decl_node_id=enum_id
    )
    table = type_table_for(member_def, choice_def)
    value = RecordValue(NominalId(member_id), {})

    plan = build_encode_plan(choice, table)

    assert encode_value(plan, value) == {"$case": "uno"}


def test_encode_plan_json_name_overrides_name_for_field() -> None:
    """``@json-name`` wins over ``@name`` for a field's JSON key."""
    decl_id = next_decl_id()
    renamed = RecordType(name="Renamed", decl_id=decl_id)
    renamed_def = TypeDef(
        kind="record",
        name="Renamed",
        module_id=ENTRY_ID,
        fields=(("value", IntType()),),
        field_external_names=(("value", ExternalName(name="alt", json_name="val")),),
        decl_node_id=decl_id,
    )
    value = RecordValue(NominalId(decl_id), {"value": IntValue(3)})

    plan = build_encode_plan(renamed, type_table_for(renamed_def))

    assert encode_value(plan, value) == {"val": 3}


def test_encode_plan_flattens_renamed_field_from_exception_base_chain() -> None:
    """An inherited field's rename from a base exception still applies at the derived type."""
    base_id = next_decl_id()
    derived_id = next_decl_id()
    base_def = TypeDef(
        kind="exception",
        name="Base",
        module_id=ENTRY_ID,
        fields=(("code", IntType()),),
        field_external_names=(("code", ExternalName(json_name="error-code")),),
        decl_node_id=base_id,
    )
    derived = ExceptionType(name="Derived", decl_id=derived_id)
    derived_def = TypeDef(
        kind="exception",
        name="Derived",
        module_id=ENTRY_ID,
        base=base_id,
        decl_node_id=derived_id,
    )
    table = type_table_for(base_def, derived_def)
    value = ExceptionValue(NominalId(derived_id), {"code": IntValue(4)})

    plan = build_encode_plan(derived, table)

    assert encode_value(plan, value) == {"error-code": 4}


def test_encode_plan_renames_field_in_generic_record() -> None:
    """A renamed field on a generic record keys its JSON output regardless of instantiation."""
    decl_id = next_decl_id()
    box = RecordType(name="Box", type_args=(IntType(),), decl_id=decl_id)
    box_def = TypeDef(
        kind="record",
        name="Box",
        module_id=ENTRY_ID,
        type_params=("T",),
        fields=(("value", TypeVarType("T")),),
        field_external_names=(("value", ExternalName(json_name="payload")),),
        decl_node_id=decl_id,
    )
    value = RecordValue(NominalId(decl_id), {"value": IntValue(9)})

    plan = build_encode_plan(box, type_table_for(box_def))

    assert encode_value(plan, value) == {"payload": 9}


def test_encode_plan_renames_field_in_recursive_hoisted_type() -> None:
    """A renamed field survives ``$defs`` hoisting for a recursive type."""
    recursive_id = next_decl_id()
    recursive = RecordType(name="Recursive", decl_id=recursive_id)
    recursive_def = TypeDef(
        kind="record",
        name="Recursive",
        module_id=ENTRY_ID,
        fields=(("children", ArrayType(recursive)),),
        field_external_names=(("children", ExternalName(json_name="kids")),),
        decl_node_id=recursive_id,
    )
    value = RecordValue(
        NominalId(recursive_id),
        {
            "children": ArrayValue(
                [RecordValue(NominalId(recursive_id), {"children": ArrayValue([])})]
            )
        },
    )

    plan = build_encode_plan(recursive, type_table_for(recursive_def))

    assert isinstance(plan.root, RefEncode)
    assert encode_value(plan, value) == {"kids": [{"kids": []}]}


def test_encode_definition_keys_match_the_schema_and_decode_defs_keys() -> None:
    """One recursion plan keys the JSON Schema, the decode walk, and the encode walk alike."""
    from tests._agl_helpers import build_decode_schema, derive_schema

    tree_id = next_decl_id()
    tree_ref = EnumType(name="Tree", decl_id=tree_id)
    tree, tree_def = enum_type(
        "Tree",
        {"Leaf": {}, "Node": {"left": tree_ref, "right": tree_ref}},
        decl_id=tree_id,
    )
    # ``Tree`` is both recursive and reached from two fields, so it is hoisted
    # once and referenced from every occurrence in all three derivations.
    wrapper, wrapper_def = record_type("Wrapper", {"first": tree, "second": tree})
    table = type_table_for(tree_def, wrapper_def)

    schema = derive_schema(wrapper, table)
    schema_defs = schema["$defs"]
    assert isinstance(schema_defs, dict)
    encode_keys = [definition.key for definition in build_encode_plan(wrapper, table).definitions]
    decode_keys = [key for key, _body in build_decode_schema(wrapper, table).defs]

    assert encode_keys == decode_keys == ["Tree"]
    assert set(encode_keys) == set(schema_defs)
    assert all(
        definition.parameter_count == 0
        for definition in build_encode_plan(wrapper, table).definitions
    )
