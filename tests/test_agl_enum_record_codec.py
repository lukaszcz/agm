"""Wire contracts for enum members represented by record values."""

from __future__ import annotations

import json

import pytest

from agm.agl.ir.contracts import DecodeSchema
from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.runtime.arguments import decode_param_value
from agm.agl.runtime.convert import decode_value
from agm.agl.runtime.serialize import dumps_exact, encode_value
from agm.agl.semantics.type_table import TypeDef
from agm.agl.semantics.types import (
    ArrayType,
    EnumType,
    IntType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
)
from agm.agl.semantics.values import (
    ArrayValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.type_schema import build_encode_plan, build_param_decoder, derive_schema_and_decode
from agm.agl.zones import ParamZone
from tests._agl_helpers import (
    build_decode_schema,
    derive_schema,
    enum_type,
    next_decl_id,
    record_type,
    type_table_for,
)
from tests.agl.ir_harness import (
    evaluate_ir_raises_with_agents,
    evaluate_ir_with_agents,
)


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(", ", ": "))


def _wire(*parts: str) -> str:
    return "".join(parts)


def test_inline_enum_wire_corpus_preserves_schema_and_value_bytes() -> None:
    """Inline enum declarations retain the established schema and wire spelling."""
    empty, empty_def = enum_type("Empty", {"None": {}})
    option, option_def = enum_type("Option", {"None": {}, "Some": {"value": IntType()}})
    generic, generic_def = enum_type(
        "Boxed",
        {"Empty": {}, "Item": {"value": TypeVarType("T")}},
        type_args=(TextType(),),
        type_params=("T",),
    )
    inner, inner_def = enum_type("Inner", {"Low": {}, "High": {"n": IntType()}})
    outer, outer_def = enum_type("Outer", {"Wrap": {"inner": inner}})
    tree_id = next_decl_id()
    tree_leaf_id = next_decl_id()
    tree_node_id = next_decl_id()
    tree = EnumType("Tree", decl_id=tree_id)
    tree_leaf = RecordType("Leaf", scope_path=("Tree",), decl_id=tree_leaf_id)
    tree_node = RecordType("Node", scope_path=("Tree",), decl_id=tree_node_id)
    tree_leaf_def = TypeDef(
        kind="record",
        name="Leaf",
        module_id=empty.module_id,
        scope_path=("Tree",),
        decl_node_id=tree_leaf_id,
    )
    tree_node_def = TypeDef(
        kind="record",
        name="Node",
        module_id=empty.module_id,
        scope_path=("Tree",),
        fields=(("children", ArrayType(tree)),),
        field_kinds=(ParamZone.STANDARD,),
        decl_node_id=tree_node_id,
    )
    tree_def = TypeDef(
        kind="enum",
        name="Tree",
        module_id=empty.module_id,
        members=(tree_leaf, tree_node),
        decl_node_id=tree_id,
    )
    table = type_table_for(
        empty_def,
        option_def,
        generic_def,
        inner_def,
        outer_def,
        tree_leaf_def,
        tree_node_def,
        tree_def,
    )

    empty_member = table.enum_member_names(empty)["None"]
    option_members = table.enum_member_names(option)
    generic_members = table.enum_member_names(generic)
    inner_members = table.enum_member_names(inner)
    outer_member = table.enum_member_names(outer)["Wrap"]
    tree_members = table.enum_member_names(tree)
    cases: tuple[tuple[Type, Value, str, str], ...] = (
        (
            empty,
            RecordValue(NominalId(empty_member.decl_id), {}),
            _wire(
                '{"oneOf": [{"type": "object", "additionalProperties": false, ',
                '"required": ["$case"], "properties": {"$case": {"const": "None"}}}]}',
            ),
            '{"$case": "None"}',
        ),
        (
            option,
            RecordValue(
                NominalId(option_members["Some"].decl_id),
                {"value": IntValue(3)},
            ),
            _wire(
                '{"oneOf": [{"type": "object", "additionalProperties": false, ',
                '"required": ["$case"], "properties": {"$case": {"const": "None"}}}, ',
                '{"type": "object", "additionalProperties": false, "required": ',
                '["$case", "value"], "properties": {"$case": {"const": "Some"}, ',
                '"value": {"type": "integer"}}}]}',
            ),
            '{"$case": "Some", "value": 3}',
        ),
        (
            generic,
            RecordValue(
                NominalId(generic_members["Item"].decl_id),
                {"value": TextValue("text")},
            ),
            _wire(
                '{"oneOf": [{"type": "object", "additionalProperties": false, ',
                '"required": ["$case"], "properties": {"$case": {"const": "Empty"}}}, ',
                '{"type": "object", "additionalProperties": false, "required": ',
                '["$case", "value"], "properties": {"$case": {"const": "Item"}, ',
                '"value": {"type": "string"}}}]}',
            ),
            '{"$case": "Item", "value": "text"}',
        ),
        (
            outer,
            RecordValue(
                NominalId(outer_member.decl_id),
                {
                    "inner": RecordValue(
                        NominalId(inner_members["High"].decl_id),
                        {"n": IntValue(8)},
                    )
                },
            ),
            _wire(
                '{"oneOf": [{"type": "object", "additionalProperties": false, ',
                '"required": ["$case", "inner"], "properties": {"$case": {"const": "Wrap"}, ',
                '"inner": {"oneOf": [{"type": "object", "additionalProperties": false, ',
                '"required": ["$case"], "properties": {"$case": {"const": "Low"}}}, ',
                '{"type": "object", "additionalProperties": false, "required": ',
                '["$case", "n"], "properties": {"$case": {"const": "High"}, ',
                '"n": {"type": "integer"}}}]}}}]}',
            ),
            '{"$case": "Wrap", "inner": {"$case": "High", "n": 8}}',
        ),
        (
            tree,
            RecordValue(
                NominalId(tree_members["Node"].decl_id),
                {
                    "children": ArrayValue(
                        [RecordValue(NominalId(tree_members["Leaf"].decl_id), {})]
                    )
                },
            ),
            _wire(
                '{"$ref": "#/$defs/Tree", "$defs": {"Tree": {"oneOf": [',
                '{"type": "object", "additionalProperties": false, "required": ["$case"], ',
                '"properties": {"$case": {"const": "Leaf"}}}, {"type": "object", ',
                '"additionalProperties": false, "required": ["$case", "children"], ',
                '"properties": {"$case": {"const": "Node"}, "children": {"type": "array", ',
                '"items": {"$ref": "#/$defs/Tree"}}}}]}}}',
            ),
            '{"$case": "Node", "children": [{"$case": "Leaf"}]}',
        ),
    )

    for typ, value, schema_bytes, value_bytes in cases:
        assert _compact_json(derive_schema(typ, table)) == schema_bytes
        encoded = encode_value(build_encode_plan(typ, table), value)
        assert dumps_exact(encoded, indent=None) == value_bytes


def test_record_and_enum_slots_keep_shared_members_distinct_on_the_wire() -> None:
    """One record value decodes through every declared static slot without retagging records."""
    shared, shared_def = record_type("Shared", {"value": IntType()})
    rr, _ = enum_type("RR", {"Ignored": {}})
    rr_prime, _ = enum_type("RRPrime", {"Ignored": {}})
    rr_def = TypeDef(
        kind="enum",
        name="RR",
        module_id=shared.module_id,
        members=(shared,),
        decl_node_id=rr.decl_id,
    )
    rr_prime_def = TypeDef(
        kind="enum",
        name="RRPrime",
        module_id=shared.module_id,
        members=(shared,),
        decl_node_id=rr_prime.decl_id,
    )
    table = type_table_for(shared_def, rr_def, rr_prime_def)
    value = RecordValue(NominalId(shared.decl_id), {"value": IntValue(7)})

    encoded_record = encode_value(build_encode_plan(shared, table), value)
    encoded_rr = encode_value(build_encode_plan(rr, table), value)
    encoded_rr_prime = encode_value(build_encode_plan(rr_prime, table), value)

    assert encoded_record == {"value": 7}
    assert encoded_rr == {"$case": "Shared", "value": 7}
    assert encoded_rr_prime == {"$case": "Shared", "value": 7}
    assert decode_value(build_decode_schema(shared, table).root, encoded_record) == value
    assert decode_value(build_decode_schema(rr, table).root, encoded_rr) == value
    assert decode_value(build_decode_schema(rr_prime, table).root, encoded_rr_prime) == value
    assert decode_param_value(build_param_decoder(shared, table), '{"value": 7}') == value
    assert (
        decode_param_value(build_param_decoder(rr, table), '{"$case": "Shared", "value": 7}')
        == value
    )
    assert (
        decode_param_value(build_param_decoder(rr_prime, table), '{"$case": "Shared", "value": 7}')
        == value
    )


def test_recursive_defs_share_a_non_generic_member_across_enum_instantiations() -> None:
    """A referenced non-generic member has one recursive definition at every use site."""
    leaf_id = next_decl_id()
    tree_id = next_decl_id()
    node_id = next_decl_id()
    forest_id = next_decl_id()
    leaf = RecordType("Leaf", decl_id=leaf_id)
    tree_int = EnumType("Tree", type_args=(IntType(),), decl_id=tree_id)
    tree_text = EnumType("Tree", type_args=(TextType(),), decl_id=tree_id)
    node = RecordType(
        "Node",
        type_args=(TypeVarType("T"),),
        scope_path=("Tree",),
        decl_id=node_id,
    )
    forest = EnumType("Forest", decl_id=forest_id)
    leaf_def = TypeDef(
        kind="record",
        name="Leaf",
        module_id=tree_int.module_id,
        fields=(("children", ArrayType(leaf)),),
        field_kinds=(ParamZone.STANDARD,),
        decl_node_id=leaf_id,
    )
    node_def = TypeDef(
        kind="record",
        name="Node",
        module_id=tree_int.module_id,
        scope_path=("Tree",),
        type_params=("T",),
        fields=(
            ("value", TypeVarType("T")),
            ("next", EnumType("Tree", type_args=(TypeVarType("T"),), decl_id=tree_id)),
        ),
        field_kinds=(ParamZone.STANDARD,) * 2,
        decl_node_id=node_id,
    )
    tree_def = TypeDef(
        kind="enum",
        name="Tree",
        module_id=tree_int.module_id,
        type_params=("T",),
        members=(leaf, node),
        decl_node_id=tree_id,
    )
    forest_def = TypeDef(
        kind="enum",
        name="Forest",
        module_id=tree_int.module_id,
        members=(leaf,),
        decl_node_id=forest_id,
    )
    holder, holder_def = record_type(
        "Holder",
        {"as_int": tree_int, "as_text": tree_text, "plain": leaf, "forest": forest},
    )
    table = type_table_for(leaf_def, node_def, tree_def, forest_def, holder_def)

    schema = derive_schema(holder, table)
    defs = schema["$defs"]

    assert isinstance(defs, dict)
    assert set(defs) == {"Tree_int", "Tree_text", "Leaf"}
    assert defs["Tree_int"]["oneOf"][1]["properties"]["value"] == {"type": "integer"}
    assert defs["Tree_text"]["oneOf"][1]["properties"]["value"] == {"type": "string"}
    assert defs["Leaf"] == {
        "type": "object",
        "additionalProperties": False,
        "required": ["children"],
        "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Leaf"}}},
    }


_AGENT_SOURCE = """\
record Shared(value: int)
enum RR = ::Shared | Other(label: text)
enum RRPrime = ::Shared
enum Tree = Leaf | Node(value: int, children: array[Tree])
record Answer(plain: Shared, selected: RR, prime: RRPrime, tree: Tree)
let worker: Agent = AgentCommand(\"worker\")
let answer: Answer = ask(\"answer\", agent = worker)
let encoded = answer as json
let decoded = encoded as Answer
()\n"""


@pytest.mark.parametrize(
    ("response", "expected"),
    (
        (
            _wire(
                '{"plain":{"value":1},"selected":{"$case":"Shared","value":2},',
                '"prime":{"$case":"Shared","value":3},"tree":{"$case":"Node","value":4,',
                '"children":[{"$case":"Leaf"}]}}',
            ),
            _wire(
                '{"plain": {"value": 1}, "selected": {"$case": "Shared", "value": 2}, ',
                '"prime": {"$case": "Shared", "value": 3}, "tree": {"$case": "Node", ',
                '"value": 4, "children": [{"$case": "Leaf"}]}}',
            ),
        ),
        (
            _wire(
                '{"plain":{"value":5},"selected":{"$case":"Other","label":"ok"},',
                '"prime":{"$case":"Shared","value":6},"tree":{"$case":"Leaf"}}',
            ),
            _wire(
                '{"plain": {"value": 5}, "selected": {"$case": "Other", "label": "ok"}, ',
                '"prime": {"$case": "Shared", "value": 6}, "tree": {"$case": "Leaf"}}',
            ),
        ),
    ),
)
def test_mocked_agent_decodes_member_records_for_multiple_enum_shapes(
    response: str, expected: str
) -> None:
    """Agent result decoding follows the record or enum type of each Answer field."""
    result = evaluate_ir_with_agents(_AGENT_SOURCE, {"worker": [response]})

    encoded = result["encoded"]
    decoded = result["decoded"]
    assert isinstance(encoded, JsonValue)
    assert dumps_exact(encoded.raw, indent=None) == expected
    assert decoded == result["answer"]


def test_mocked_agent_rejects_an_unknown_member_case() -> None:
    """An unknown enum case remains a decode error at the agent boundary."""
    response = _wire(
        '{"plain":{"value":1},"selected":{"$case":"Unknown","value":2},',
        '"prime":{"$case":"Shared","value":3},"tree":{"$case":"Leaf"}}',
    )

    error = evaluate_ir_raises_with_agents(_AGENT_SOURCE, {"worker": [response]})
    assert error.type_name == "AgentParseError"


# ---------------------------------------------------------------------------
# Constructor field defaults at the JSON decode boundary (decode_value)
# ---------------------------------------------------------------------------
#
# The decode plan never carries a default's VALUE (see type_schema.py); an
# omitted defaulted field is filled by calling the caller-supplied
# ``default_resolver(nominal, field_index)`` -- here a plain stub standing in
# for ``IrInterpreter.default_for_field`` (exercised end-to-end, against a
# real interpreter, by the e2e programs under tests/agl/programs/).


def test_record_omitted_defaulted_field_fills_via_the_default_resolver() -> None:
    typ, typedef = record_type(
        "Pair", {"x": IntType(), "y": IntType()}, field_has_default=(False, True)
    )
    nominal = NominalId(typedef.decl_node_id)
    table = type_table_for(typedef)
    _, plan = derive_schema_and_decode(typ, table)

    calls: list[tuple[NominalId, int]] = []

    def resolver(field_nominal: NominalId, field_index: int) -> Value:
        calls.append((field_nominal, field_index))
        return IntValue(9)

    value = decode_value(plan.root, {"x": 1}, dict(plan.defs), default_resolver=resolver)
    assert value == RecordValue(nominal, {"x": IntValue(1), "y": IntValue(9)})
    assert calls == [(nominal, 1)]


def test_record_omitted_defaulted_field_without_a_resolver_still_raises() -> None:
    """No ``default_resolver`` reachable (engine-config decode, schema preview): an omitted
    defaulted field reports the ordinary missing-field error rather than silently degrading."""
    typ, typedef = record_type(
        "Pair", {"x": IntType(), "y": IntType()}, field_has_default=(False, True)
    )
    table = type_table_for(typedef)
    _, plan = derive_schema_and_decode(typ, table)

    with pytest.raises(ValueError, match="Missing field"):
        decode_value(plan.root, {"x": 1}, dict(plan.defs))


def test_record_missing_required_field_still_raises() -> None:
    typ, typedef = record_type(
        "Pair", {"x": IntType(), "y": IntType()}, field_has_default=(False, True)
    )
    table = type_table_for(typedef)
    _, plan = derive_schema_and_decode(typ, table)

    with pytest.raises(ValueError, match="Missing field"):
        decode_value(
            plan.root, {}, dict(plan.defs), default_resolver=lambda nominal, index: IntValue(9)
        )


def _outcome_plan_with_defaulted_member() -> tuple[
    DecodeSchema, dict[str, DecodeSchema], NominalId
]:
    """``enum Outcome | Ok(tag: text = "ok")``, its member field's default left unresolved."""
    enum_id = next_decl_id()
    member_id = next_decl_id()
    member = RecordType(name="Ok", module_id=ENTRY_ID, scope_path=("Outcome",), decl_id=member_id)
    member_def = TypeDef(
        kind="record",
        name="Ok",
        module_id=ENTRY_ID,
        scope_path=("Outcome",),
        fields=(("tag", TextType()),),
        field_kinds=(ParamZone.STANDARD,),
        field_has_default=(True,),
        decl_node_id=member_id,
    )
    outcome_def = TypeDef(
        kind="enum", name="Outcome", module_id=ENTRY_ID, members=(member,), decl_node_id=enum_id
    )
    typ = EnumType(name="Outcome", decl_id=enum_id)
    table = type_table_for(member_def, outcome_def)
    _, plan = derive_schema_and_decode(typ, table)
    return plan.root, dict(plan.defs), NominalId(member_id)


def test_enum_variant_omitted_defaulted_field_fills_via_the_default_resolver() -> None:
    schema, defs, member_nominal = _outcome_plan_with_defaulted_member()

    def resolver(field_nominal: NominalId, field_index: int) -> Value:
        assert (field_nominal, field_index) == (member_nominal, 0)
        return TextValue("ok")

    value = decode_value(schema, {"$case": "Ok"}, defs, default_resolver=resolver)
    assert isinstance(value, RecordValue)
    assert value.fields == {"tag": TextValue("ok")}


def test_enum_variant_missing_required_field_still_raises() -> None:
    enum_id = next_decl_id()
    member_id = next_decl_id()
    member = RecordType(name="Item", module_id=ENTRY_ID, scope_path=("Outcome",), decl_id=member_id)
    member_def = TypeDef(
        kind="record",
        name="Item",
        module_id=ENTRY_ID,
        scope_path=("Outcome",),
        fields=(("value", IntType()),),
        field_kinds=(ParamZone.STANDARD,),
        decl_node_id=member_id,
    )
    outcome_def = TypeDef(
        kind="enum", name="Outcome", module_id=ENTRY_ID, members=(member,), decl_node_id=enum_id
    )
    typ = EnumType(name="Outcome", decl_id=enum_id)
    table = type_table_for(member_def, outcome_def)
    _, plan = derive_schema_and_decode(typ, table)

    with pytest.raises(ValueError, match="missing field"):
        decode_value(plan.root, {"$case": "Item"}, dict(plan.defs))
