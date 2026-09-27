"""Static JSON encode-plan derivation and runtime execution tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from agm.agl.eval.ir_interpreter import _apply_coercion
from agm.agl.ir import contracts
from agm.agl.ir.contracts import (
    ArrayEncode,
    DictEncode,
    DictKeyForm,
    EncodeDefinition,
    EncodePlan,
    EnumEncode,
    ExceptionEncode,
    ExceptionFieldEncode,
    FieldEncode,
    RecordEncode,
    RefEncode,
    ScalarEncode,
    ScalarKind,
    TypeParameterEncode,
    VariantEncode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.ir.operations import ToJson
from agm.agl.ir.reserved_nominals import (
    require_reserved_enum_member_id,
    require_reserved_nominal_id,
)
from agm.agl.modules.ids import ENTRY_ID, RESERVED_ID
from agm.agl.runtime.arguments import decode_param_value
from agm.agl.runtime.serialize import (
    WalkTags,
    dumps_exact,
    encode_value,
    value_to_json_obj,
)
from agm.agl.semantics.external_names import ExternalName
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import (
    ArrayType,
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    TypeVarType,
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
    build_exception_field_encodes,
    build_param_decoder,
)
from agm.agl.zones import ParamZone
from tests._agl_helpers import (
    derive_schema,
    enum_type,
    enum_typedef,
    next_decl_id,
    record_type,
    type_table_for,
)
from tests.agl.ir_harness import (
    evaluate_ir,
    evaluate_ir_with_agents,
    evaluate_ir_with_externs,
)

#: The exception field-encode table of a program declaring no exceptions.
_NO_EXCEPTIONS: dict[NominalId, tuple[ExceptionFieldEncode, ...]] = {}


def _walk_tags_from(type_table: TypeTable, *handles: "RecordType | ExceptionType") -> WalkTags:
    """Build :class:`WalkTags` from *handles*' real fields/JSON-names, read off *type_table*."""
    from agm.agl.lower.nominal_descriptors import exception_descriptor, record_descriptor
    from agm.agl.runtime.serialize import _walk_tags

    nominals = {}
    for handle in handles:
        typedef = type_table.typedef_of(handle.decl_id)
        descriptor = (
            exception_descriptor(typedef, handle, type_table, bears_name_path=False)
            if isinstance(handle, ExceptionType)
            else record_descriptor(typedef, handle, type_table, bears_name_path=False)
        )
        nominals[descriptor.nominal] = descriptor
    return _walk_tags(nominals)


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
    exceptions = {NominalId(problem_id): build_exception_field_encodes(problem, table)}
    item_value = RecordValue(NominalId(item.decl_id), {"value": IntValue(7)})
    choice_value = RecordValue(
        nominal=NominalId(choice_def.members[1].decl_id),
        fields={"items": ArrayValue([item_value])},
    )
    problem_value = ExceptionValue(NominalId(problem.decl_id), {"choice": choice_value})

    tags = _walk_tags_from(table, item, choice_def.members[1])
    assert value_to_json_obj(choice_value, tags=tags) == {"items": [{"value": 7}]}
    assert encode_value(build_encode_plan(choice, table), choice_value, _NO_EXCEPTIONS) == {
        "$case": "Many",
        "items": [{"value": 7}],
    }
    assert encode_value(build_encode_plan(problem, table), problem_value, exceptions) == {
        "choice": {"$case": "Many", "items": [{"value": 7}]}
    }
    # The template builder is what a growing source gets; on a source both
    # builders accept it must agree, exception root and enum slot alike.
    assert encode_value(_build_template_encode_plan(problem, table), problem_value, exceptions) == {
        "choice": {"$case": "Many", "items": [{"value": 7}]}
    }


def test_encode_plan_executes_all_shapes_and_member_identity() -> None:
    scalar = EncodePlan(ScalarEncode(ScalarKind.INT))
    assert [
        encode_value(scalar, value, _NO_EXCEPTIONS)
        for value in (
            TextValue("text"),
            IntValue(2),
            DecimalValue(Decimal("2.5")),
            BoolValue(True),
            JsonValue({"raw": None}),
        )
    ] == ["text", 2, Decimal("2.5"), True, {"raw": None}]

    assert encode_value(
        EncodePlan(ArrayEncode(ScalarEncode(ScalarKind.INT))),
        ArrayValue([IntValue(1), IntValue(2)]),
        _NO_EXCEPTIONS,
    ) == [1, 2]
    assert encode_value(
        EncodePlan(
            DictEncode(
                DictKeyForm.OBJECT_TEXT, ScalarEncode(ScalarKind.TEXT), ScalarEncode(ScalarKind.INT)
            )
        ),
        DictValue({"n": IntValue(3)}),
        _NO_EXCEPTIONS,
    ) == {"n": 3}

    enum = EncodePlan(
        EnumEncode(NominalId(1), (VariantEncode("Member", "Member", NominalId(2), ()),))
    )
    assert encode_value(enum, RecordValue(nominal=NominalId(2), fields={}), _NO_EXCEPTIONS) == {
        "$case": "Member"
    }


def test_to_json_coercion_uses_the_static_scalar_encoder() -> None:
    assert _apply_coercion(IntValue(1), ToJson()) == JsonValue(1)


def test_encode_plan_binds_definition_parameters_at_each_reference() -> None:
    """A parameterized definition encodes its slots through the caller's arguments."""
    pair = NominalId(1)
    plan = EncodePlan(
        root=RefEncode(
            "Pair", (ScalarEncode(ScalarKind.INT), ArrayEncode(ScalarEncode(ScalarKind.INT)))
        ),
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

    assert encode_value(plan, value, _NO_EXCEPTIONS) == {"first": 1, "second": [2, 3]}


def test_encode_plan_substitutes_arguments_through_every_composite_shape() -> None:
    """An argument is rewritten out of the caller's parameter space before it binds."""
    outer = NominalId(1)
    record = NominalId(2)
    enum = NominalId(4)
    member = NominalId(5)
    # ``Outer`` hands ``Inner`` a composite written in terms of ITS OWN
    # parameter, and ``Inner``'s body is just that parameter — so each case
    # encodes correctly only if the substitution reached every leaf position
    # inside the composite it was given.
    cases = (
        (ArrayEncode(TypeParameterEncode(0)), ArrayValue([IntValue(4)]), [4]),
        (
            DictEncode(
                DictKeyForm.OBJECT_TEXT, ScalarEncode(ScalarKind.TEXT), TypeParameterEncode(0)
            ),
            DictValue({"n": IntValue(5)}),
            {"n": 5},
        ),
        (
            RecordEncode(record, (FieldEncode("value", "value", TypeParameterEncode(0)),)),
            RecordValue(record, {"value": IntValue(1)}),
            {"value": 1},
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
            root=RefEncode("Outer", (ScalarEncode(ScalarKind.INT),)),
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
        assert encode_value(plan, RecordValue(outer, {"held": value}), _NO_EXCEPTIONS) == {
            "held": expected
        }


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
            DictType(TextType(), choice),
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
            dumps_exact(
                encode_value(build_encode_plan(typ, table), value, _NO_EXCEPTIONS), indent=None
            )
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
        '"max_attempts": 1}}'
    )

    choice, choice_def = enum_type("Choice", {"None": {}, "One": {"value": IntType()}})
    table = type_table_for(choice_def)
    decoded = decode_param_value(
        build_param_decoder(choice, table),
        '{"$case": "One", "value": 2}',
    )
    assert dumps_exact(
        encode_value(build_encode_plan(choice, table), decoded, _NO_EXCEPTIONS), indent=None
    ) == ('{"$case": "One", "value": 2}')


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
                            RecordEncode(
                                member,
                                (FieldEncode("value", "value", ScalarEncode(ScalarKind.INT)),),
                            ),
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
                                        (
                                            FieldEncode(
                                                "value", "value", ScalarEncode(ScalarKind.INT)
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        FieldEncode("items", "items", ArrayEncode(ScalarEncode(ScalarKind.INT))),
                        FieldEncode(
                            "by-name",
                            "by-name",
                            DictEncode(
                                DictKeyForm.OBJECT_TEXT,
                                ScalarEncode(ScalarKind.TEXT),
                                ScalarEncode(ScalarKind.INT),
                            ),
                        ),
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
        _NO_EXCEPTIONS,
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
    exceptions = {
        NominalId(2): (
            ExceptionFieldEncode("cause", "cause", EncodePlan(ExceptionEncode(NominalId(2)))),
        )
    }

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
        (EncodePlan(ExceptionEncode(NominalId(2))), exception),
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
            encode_value(plan, value, exceptions)


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
                    RecordEncode(
                        leaf, (FieldEncode("value", "value", ScalarEncode(ScalarKind.INT)),)
                    ),
                ),
                FieldEncode(
                    "right",
                    "right",
                    RecordEncode(
                        leaf, (FieldEncode("value", "value", ScalarEncode(ScalarKind.INT)),)
                    ),
                ),
            ),
        )
    )

    assert encode_value(plan, value, _NO_EXCEPTIONS) == {
        "left": {"value": 1},
        "right": {"value": 1},
    }


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

    assert encode_value(plan, value, _NO_EXCEPTIONS) == {"payload": 9}


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


# ---------------------------------------------------------------------------
# `as json` encoding by dict key form (text object / stringified object / entries array)
# ---------------------------------------------------------------------------


class TestDictKeyFormJsonSchemaCrossCheck:
    """The actual ``as json`` output for each dict key form validates against
    ``derive_schema``'s own JSON Schema for the same ``(type, table)`` pair --
    an independent, external check (the ``jsonschema`` library) that the
    encoder and the schema deriver never silently drift apart.
    """

    def _cross_checked(self, typ: DictType, table: TypeTable, value: DictValue) -> object:
        schema = derive_schema(typ, table)
        Draft202012Validator.check_schema(schema)
        encoded = encode_value(build_encode_plan(typ, table), value, _NO_EXCEPTIONS)
        Draft202012Validator(schema).validate(encoded)
        return encoded

    def test_text_key_object_form(self) -> None:
        value = DictValue()
        value.insert(TextValue("k"), TextValue("v"))
        assert self._cross_checked(
            DictType(key=TextType(), value=TextType()), type_table_for(), value
        ) == {"k": "v"}

    def test_all_nullary_enum_key_stringified_form_honors_json_name_and_name(self) -> None:
        """A ``@json-name`` member and a ``@name``-only member both stringify by their
        effective tag (``@json-name`` direct; ``@name`` falling back to it)."""
        a_id = next_decl_id()
        b_id = next_decl_id()
        a_member = RecordType(name="A", module_id=ENTRY_ID, scope_path=("Choice",), decl_id=a_id)
        b_member = RecordType(name="B", module_id=ENTRY_ID, scope_path=("Choice",), decl_id=b_id)
        a_def = TypeDef(
            kind="record",
            name="A",
            module_id=ENTRY_ID,
            scope_path=("Choice",),
            external_name=ExternalName(json_name="x"),
            decl_node_id=a_id,
        )
        b_def = TypeDef(
            kind="record",
            name="B",
            module_id=ENTRY_ID,
            scope_path=("Choice",),
            external_name=ExternalName(name="bee"),
            decl_node_id=b_id,
        )
        enum_id = next_decl_id()
        choice_def = TypeDef(
            kind="enum",
            name="Choice",
            module_id=ENTRY_ID,
            members=(a_member, b_member),
            decl_node_id=enum_id,
        )
        table = type_table_for(a_def, b_def, choice_def)
        choice = EnumType(name="Choice", module_id=ENTRY_ID, decl_id=enum_id)
        value = DictValue()
        value.insert(RecordValue(NominalId(a_id), {}), TextValue("va"))
        value.insert(RecordValue(NominalId(b_id), {}), TextValue("vb"))
        assert self._cross_checked(DictType(key=choice, value=TextType()), table, value) == {
            "x": "va",
            "bee": "vb",
        }

    def test_int_key_object_form(self) -> None:
        value = DictValue()
        value.insert(IntValue(1), TextValue("a"))
        assert self._cross_checked(
            DictType(key=IntType(), value=TextType()), type_table_for(), value
        ) == {"1": "a"}

    def test_bool_key_object_form(self) -> None:
        value = DictValue()
        value.insert(BoolValue(True), TextValue("t"))
        assert self._cross_checked(
            DictType(key=BoolType(), value=TextType()), type_table_for(), value
        ) == {"true": "t"}

    def test_json_key_entries_form(self) -> None:
        value = DictValue()
        value.insert(JsonValue(5), TextValue("j"))
        assert self._cross_checked(
            DictType(key=JsonType(), value=TextType()), type_table_for(), value
        ) == [{"key": 5, "value": "j"}]

    def test_record_key_entries_form(self) -> None:
        point, point_def = record_type("Point", {"x": IntType(), "y": IntType()})
        table = type_table_for(point_def)
        value = DictValue()
        value.insert(
            RecordValue(NominalId(point.decl_id), {"x": IntValue(1), "y": IntValue(2)}),
            TextValue("p"),
        )
        assert self._cross_checked(DictType(key=point, value=TextType()), table, value) == [
            {"key": {"x": 1, "y": 2}, "value": "p"}
        ]

    def test_mixed_enum_key_entries_form(self) -> None:
        shape, shape_def = enum_type("Shape", {"Square": {}, "Circle": {"radius": IntType()}})
        table = type_table_for(shape_def)
        value = DictValue()
        value.insert(RecordValue(NominalId(shape_def.members[0].decl_id), {}), TextValue("s"))
        value.insert(
            RecordValue(NominalId(shape_def.members[1].decl_id), {"radius": IntValue(1)}),
            TextValue("c"),
        )
        assert self._cross_checked(DictType(key=shape, value=TextType()), table, value) == [
            {"key": {"$case": "Square"}, "value": "s"},
            {"key": {"$case": "Circle", "radius": 1}, "value": "c"},
        ]

    def test_decimal_key_stringified_form(self) -> None:
        value = DictValue()
        value.insert(DecimalValue(Decimal("1.50")), TextValue("a"))
        assert self._cross_checked(
            DictType(key=DecimalType(), value=TextType()), type_table_for(), value
        ) == {"1.50": "a"}

    def test_hoisted_enum_key_stringified_form_honors_json_name_and_name(self) -> None:
        """``Color`` occurs twice here (as both the dict's key and its value), so
        it is hoisted into its own ``$defs`` entry; the stringified wire form
        still honors each member's effective JSON tag (``@json-name`` direct,
        ``@name`` falling back to it) once resolved through the hoist."""
        a_id = next_decl_id()
        b_id = next_decl_id()
        a_member = RecordType(name="A", module_id=ENTRY_ID, scope_path=("Color",), decl_id=a_id)
        b_member = RecordType(name="B", module_id=ENTRY_ID, scope_path=("Color",), decl_id=b_id)
        a_def = TypeDef(
            kind="record",
            name="A",
            module_id=ENTRY_ID,
            scope_path=("Color",),
            external_name=ExternalName(json_name="x"),
            decl_node_id=a_id,
        )
        b_def = TypeDef(
            kind="record",
            name="B",
            module_id=ENTRY_ID,
            scope_path=("Color",),
            external_name=ExternalName(name="bleu"),
            decl_node_id=b_id,
        )
        enum_id = next_decl_id()
        color_def = TypeDef(
            kind="enum",
            name="Color",
            module_id=ENTRY_ID,
            members=(a_member, b_member),
            decl_node_id=enum_id,
        )
        color = EnumType(name="Color", module_id=ENTRY_ID, decl_id=enum_id)
        table = type_table_for(a_def, b_def, color_def)
        value = DictValue()
        value.insert(RecordValue(NominalId(a_id), {}), RecordValue(NominalId(b_id), {}))
        value.insert(RecordValue(NominalId(b_id), {}), RecordValue(NominalId(a_id), {}))
        assert self._cross_checked(DictType(key=color, value=color), table, value) == {
            "x": {"$case": "bleu"},
            "bleu": {"$case": "x"},
        }

    def test_option_enum_key_entries_form(self) -> None:
        """``Option[Color]``'s ``Some`` variant carries a field, so the key is
        not all-nullary even though ``Color`` alone would stringify -- giving
        the entries wire form."""
        a_id = next_decl_id()
        b_id = next_decl_id()
        a_member = RecordType(name="Red", module_id=ENTRY_ID, scope_path=("Color",), decl_id=a_id)
        b_member = RecordType(name="Blue", module_id=ENTRY_ID, scope_path=("Color",), decl_id=b_id)
        a_def = TypeDef(
            kind="record", name="Red", module_id=ENTRY_ID, scope_path=("Color",), decl_node_id=a_id
        )
        b_def = TypeDef(
            kind="record",
            name="Blue",
            module_id=ENTRY_ID,
            scope_path=("Color",),
            decl_node_id=b_id,
        )
        enum_id = next_decl_id()
        color_def = TypeDef(
            kind="enum",
            name="Color",
            module_id=ENTRY_ID,
            members=(a_member, b_member),
            decl_node_id=enum_id,
        )
        color = EnumType(name="Color", module_id=ENTRY_ID, decl_id=enum_id)
        option_color = EnumType(
            name="Option",
            type_args=(color,),
            module_id=RESERVED_ID,
            decl_id=require_reserved_nominal_id("Option"),
        )
        table = type_table_for(a_def, b_def, color_def)
        some_id = NominalId(require_reserved_enum_member_id("Option", "Some"))
        none_id = NominalId(require_reserved_enum_member_id("Option", "None"))
        value = DictValue()
        value.insert(
            RecordValue(some_id, {"value": RecordValue(NominalId(a_id), {})}), TextValue("r")
        )
        value.insert(RecordValue(none_id, {}), TextValue("n"))
        assert self._cross_checked(DictType(key=option_color, value=TextType()), table, value) == [
            {"key": {"$case": "Some", "value": {"$case": "Red"}}, "value": "r"},
            {"key": {"$case": "None"}, "value": "n"},
        ]

    def test_hoisted_record_key_entries_form(self) -> None:
        """A record key occurring twice (as both the dict's key and its value)
        is hoisted into its own ``$defs`` entry; the entries wire form still
        round-trips both occurrences correctly."""
        point, point_def = record_type("Point", {"x": IntType(), "y": IntType()})
        table = type_table_for(point_def)
        value = DictValue()
        value.insert(
            RecordValue(NominalId(point.decl_id), {"x": IntValue(1), "y": IntValue(2)}),
            RecordValue(NominalId(point.decl_id), {"x": IntValue(3), "y": IntValue(4)}),
        )
        assert self._cross_checked(DictType(key=point, value=point), table, value) == [
            {"key": {"x": 1, "y": 2}, "value": {"x": 3, "y": 4}}
        ]


def test_finite_encode_plan_fills_dict_key_form_once() -> None:
    """A finite plan's ``DictEncode`` stores its key's ``DictKeyForm`` at build time,
    not re-derived at encode time."""
    plan = build_encode_plan(DictType(key=IntType(), value=TextType()), type_table_for())

    assert isinstance(plan.root, DictEncode)
    assert plan.root.key_form is DictKeyForm.OBJECT_STRINGIFIED


def test_dict_key_resolves_through_a_ref_encode_before_choosing_its_wire_form() -> None:
    """A dict key stored via ``$defs`` (``RefEncode``) resolves to its concrete shape first."""
    point = NominalId(1)
    plan = EncodePlan(
        root=DictEncode(DictKeyForm.ENTRIES, RefEncode("Point"), ScalarEncode(ScalarKind.TEXT)),
        definitions=(
            EncodeDefinition(
                "Point",
                0,
                RecordEncode(
                    point,
                    (
                        FieldEncode("x", "x", ScalarEncode(ScalarKind.INT)),
                        FieldEncode("y", "y", ScalarEncode(ScalarKind.INT)),
                    ),
                ),
            ),
        ),
    )
    value = DictValue()
    value.insert(RecordValue(point, {"x": IntValue(1), "y": IntValue(2)}), TextValue("p"))

    assert encode_value(plan, value, _NO_EXCEPTIONS) == [{"key": {"x": 1, "y": 2}, "value": "p"}]


def test_growing_polymorphic_recursive_json_cast_own_key_parameter_uses_entries_form() -> None:
    """A growing template's dict field keyed by its own type parameter resolves at each depth.

    Here the parameter's instantiation (``Pair[int, int]``) is a record, so the
    dict's wire form is the entries array.
    """
    result = evaluate_ir(
        "record Pair[A, B]\n"
        "  first: A\n"
        "  second: B\n"
        "enum Perfect[T]\n"
        "  | Single(value: dict[T, text])\n"
        "  | Succ(next: Perfect[Pair[T, T]])\n"
        "let inner-key = Pair(first = 1, second = 2)\n"
        'let p: Perfect[int] = Succ(next = Single(value = {inner-key: "a"}))\n'
        "let encoded = p as json\n"
        "()\n"
    )

    encoded = result["encoded"]
    assert isinstance(encoded, JsonValue)
    assert encoded.raw == {
        "$case": "Succ",
        "next": {
            "$case": "Single",
            "value": [{"key": {"first": 1, "second": 2}, "value": "a"}],
        },
    }


def test_growing_polymorphic_recursive_json_cast_stringifies_its_own_key_parameter() -> None:
    """A growing template's own type-parameter key (``DictEncode.key_form is None``) still
    stringifies correctly once resolved at a depth where it instantiates to a scalar."""
    result = evaluate_ir(
        "record Pair[A, B]\n"
        "  first: A\n"
        "  second: B\n"
        "enum Perfect[T]\n"
        "  | Single(value: dict[T, text])\n"
        "  | Succ(next: Perfect[Pair[T, T]])\n"
        'let p: Perfect[int] = Single(value = {1: "a"})\n'
        "let encoded = p as json\n"
        "()\n"
    )

    encoded = result["encoded"]
    assert isinstance(encoded, JsonValue)
    assert encoded.raw == {"$case": "Single", "value": {"1": "a"}}


def test_growing_template_stringifies_an_all_nullary_enum_key_with_json_name() -> None:
    """A growing template's dict field keyed by a fixed (non-parameter) all-nullary
    enum stringifies by each member's effective JSON tag, both in the actual encoded
    output and in the ``DictEncode.key_form`` the template plan stores for it."""
    red_id = next_decl_id()
    blue_id = next_decl_id()
    red_member = RecordType(name="Red", module_id=ENTRY_ID, scope_path=("Color",), decl_id=red_id)
    blue_member = RecordType(
        name="Blue", module_id=ENTRY_ID, scope_path=("Color",), decl_id=blue_id
    )
    red_def = TypeDef(
        kind="record", name="Red", module_id=ENTRY_ID, scope_path=("Color",), decl_node_id=red_id
    )
    blue_def = TypeDef(
        kind="record",
        name="Blue",
        module_id=ENTRY_ID,
        scope_path=("Color",),
        external_name=ExternalName(json_name="bleu"),
        decl_node_id=blue_id,
    )
    color_id = next_decl_id()
    color_def = TypeDef(
        kind="enum",
        name="Color",
        module_id=ENTRY_ID,
        members=(red_member, blue_member),
        decl_node_id=color_id,
    )
    color = EnumType(name="Color", module_id=ENTRY_ID, decl_id=color_id)

    _, pair_def = record_type(
        "Pair", {"first": TypeVarType("A"), "second": TypeVarType("B")}, type_params=("A", "B")
    )
    pair_of_t = RecordType(
        name="Pair",
        type_args=(TypeVarType("T"), TypeVarType("T")),
        module_id=ENTRY_ID,
        decl_id=pair_def.decl_node_id,
    )

    perfect_id = next_decl_id()
    perfect_def = enum_typedef(
        "Perfect",
        {
            "Single": {"value": DictType(key=color, value=TypeVarType("T"))},
            "Succ": {
                "next": EnumType(
                    name="Perfect", type_args=(pair_of_t,), module_id=ENTRY_ID, decl_id=perfect_id
                )
            },
        },
        type_params=("T",),
        decl_id=perfect_id,
    )
    table = type_table_for(red_def, blue_def, color_def, pair_def, perfect_def)
    perfect = EnumType(
        name="Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=perfect_id
    )
    single_id = perfect_def.members[0].decl_id

    plan = _build_template_encode_plan(perfect, table)

    perfect_definition = next(
        d for d in plan.definitions if d.key == f"n{perfect_id}" and isinstance(d.body, EnumEncode)
    )
    single_variant = next(v for v in perfect_definition.body.variants if v.name == "Single")
    dict_encode = single_variant.fields[0].schema
    assert isinstance(dict_encode, DictEncode)
    assert dict_encode.key_form is DictKeyForm.OBJECT_STRINGIFIED

    value = DictValue()
    value.insert(RecordValue(NominalId(red_id), {}), IntValue(1))
    value.insert(RecordValue(NominalId(blue_id), {}), IntValue(2))
    single_value = RecordValue(NominalId(single_id), {"value": value})

    assert encode_value(plan, single_value, _NO_EXCEPTIONS) == {
        "$case": "Single",
        "value": {"Red": 1, "bleu": 2},
    }


def test_encode_plan_handles_recursive_containers() -> None:
    recursive_id = next_decl_id()
    recursive = RecordType(name="Recursive", decl_id=recursive_id)
    recursive_def = TypeDef(
        kind="record",
        name="Recursive",
        module_id=ENTRY_ID,
        fields=(("children", ArrayType(recursive)),),
        field_kinds=(ParamZone.STANDARD,),
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

    table = type_table_for(recursive_def)
    plan = build_encode_plan(recursive, table)

    assert isinstance(plan.root, RefEncode)
    tags = _walk_tags_from(table, recursive)
    assert encode_value(plan, value, _NO_EXCEPTIONS) == value_to_json_obj(value, tags=tags)


def test_encode_plan_and_untyped_walk_agree_on_effective_json_name() -> None:
    """A renamed field's effective JSON key is the same whether encoded or walked untyped."""
    decl_id = next_decl_id()
    renamed = RecordType(name="Renamed", decl_id=decl_id)
    renamed_def = TypeDef(
        kind="record",
        name="Renamed",
        module_id=ENTRY_ID,
        fields=(("value", IntType()),),
        field_kinds=(ParamZone.STANDARD,),
        field_external_names=(("value", ExternalName(json_name="val")),),
        decl_node_id=decl_id,
    )
    value = RecordValue(NominalId(decl_id), {"value": IntValue(3)})

    table = type_table_for(renamed_def)
    plan = build_encode_plan(renamed, table)
    tags = _walk_tags_from(table, renamed)

    assert encode_value(plan, value, _NO_EXCEPTIONS) == {"val": 3}
    assert value_to_json_obj(value, tags=tags) == {"val": 3}


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

    assert encode_value(plan, value, _NO_EXCEPTIONS) == {"$case": "uno"}


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

    assert encode_value(plan, value, _NO_EXCEPTIONS) == {"val": 3}


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
    exceptions = {NominalId(derived_id): build_exception_field_encodes(derived, table)}

    plan = build_encode_plan(derived, table)

    assert encode_value(plan, value, exceptions) == {"error-code": 4}


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

    assert encode_value(plan, value, _NO_EXCEPTIONS) == {"payload": 9}


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
    assert encode_value(plan, value, _NO_EXCEPTIONS) == {"kids": [{"kids": []}]}


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
