"""Type-directed externs at run time: the ``agl.TypeContract`` a companion receives.

Each target parameter reaches the companion as an immutable ``TypeContract``
tree (kind, label, self-contained schema, doc, nominal class, fields, members,
items, values), built once per contract and resolved into a possibly cyclic
graph for a recursive target. A companion constructs its trusted result from
the tree's nominal classes. The underlying ``ContractValue`` is non-data.
"""

from __future__ import annotations

import dataclasses
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import ModuleType

import pytest

from agm.agl.ir.contracts import ContractRequest
from agm.agl.ir.ids import ContractId
from agm.agl.ir.program import ValueDescriptors
from agm.agl.runtime.boundary import BoundaryViolation, encode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import AglNonDataValue, value_to_json_obj
from agm.agl.runtime.type_contracts import TypeContract, build_type_contract
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import (
    ContractValue,
    IntValue,
    TextValue,
    UnitValue,
    Value,
    values_equal,
)
from tests._agl_helpers import repl_session_with_root
from tests.agl.ir_harness import (
    evaluate_ir_raises_with_externs,
    evaluate_ir_with_externs,
    lower_extern_program,
)

_PROBE = "agl_type_contract_probe"

_TEAM = (
    '@doc("The owning team.")\n'
    "enum Team\n"
    '  | @doc("Invoices and refunds.") @json-name("billing") Billing\n'
    "  | Technical\n"
)
_TRIAGE = (
    '@doc("A triaged message.")\n'
    "record Triage\n"
    '  @doc("Is it urgent?") urgent: bool\n'
    '  @doc("Who handles it?") @json-name("owner") team: Team\n'
    "  notes: array[text]\n"
)
_BILLING = "let billing: Team = Team::Billing\n"
_TREE = "enum Tree\n  | Leaf(value: int)\n  | Node(left: Tree, right: Tree)\n"

_DECLS = (
    "extern def capture[T](question: text) -> T\n"
    "extern def build[T](question: text) -> T\n"
    "extern def both[A, B](question: text) -> B\n"
)

#: Records every delivered contract in the probe; builds a value of any target.
_COMPANION = f"""
from decimal import Decimal
import agl
import {_PROBE} as probe

probe.agl = agl

def capture(contract, question):
    probe.seen.append(contract)
    return build(contract, question)

def both(first, second, question):
    probe.seen.extend([first, second])
    return first.label + "," + second.label

def build(contract, question):
    kind = contract.kind
    if kind == "int":
        return 7
    if kind == "text":
        return question
    if kind == "decimal":
        return Decimal("1.5")
    if kind == "bool":
        return True
    if kind == "json":
        return agl.json({{"q": question}})
    if kind == "array":
        return agl.array([build(contract.items, question)])
    if kind == "dict":
        return agl.dict({{"k": build(contract.values, question)}})
    if kind == "enum":
        return build(next(iter(contract.members.values())), question)
    return contract.nominal(
        **{{field.name: build(field.contract, question) for field in contract.fields.values()}}
    )

def leak(contract, question):
    return contract
"""


@pytest.fixture(autouse=True)
def probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """A module the companion imports and records every delivered contract into."""
    module = ModuleType(_PROBE)
    module.seen = []
    monkeypatch.setitem(sys.modules, _PROBE, module)
    yield module


def _run(source: str, tmp_path: Path) -> dict[str, object]:
    bindings, _output = evaluate_ir_with_externs(_DECLS + source, _COMPANION, tmp_path)
    return dict(bindings)


def _delivered(probe: ModuleType, source: str, tmp_path: Path) -> list[object]:
    _run(source, tmp_path)
    return list(probe.seen)


def _only(probe: ModuleType, source: str, tmp_path: Path) -> object:
    (contract,) = _delivered(probe, source, tmp_path)
    return contract


class TestDeliveredContents:
    @pytest.mark.parametrize(
        ("target", "schema"),
        [
            ("int", {"type": "integer"}),
            ("text", {"type": "string"}),
            ("decimal", {"type": "number"}),
            ("bool", {"type": "boolean"}),
            ("json", {}),
        ],
    )
    def test_scalar(self, probe: ModuleType, tmp_path: Path, target: str, schema: object) -> None:
        contract = _only(probe, f'let answer: {target} = capture("q")\n0', tmp_path)
        assert isinstance(contract, probe.agl.TypeContract)
        assert (contract.kind, contract.label, contract.schema) == (target, target, schema)
        assert (contract.doc, contract.nominal, contract.items, contract.values) == (
            None,
            None,
            None,
            None,
        )
        assert (dict(contract.fields), dict(contract.members)) == ({}, {})

    def test_enum_members_carry_tags_docs_and_member_classes(
        self, probe: ModuleType, tmp_path: Path
    ) -> None:
        contract = _only(probe, _TEAM + 'let team: Team = capture("q")\n0', tmp_path)
        team = probe.agl.Team
        assert (contract.kind, contract.label, contract.doc) == ("enum", "Team", "The owning team.")
        assert contract.nominal is team
        assert list(contract.members) == ["billing", "Technical"]
        billing = contract.members["billing"]
        assert (billing.kind, billing.doc, billing.nominal) == (
            "member",
            "Invoices and refunds.",
            team.Billing,
        )
        assert contract.members["Technical"].nominal is team.Technical
        assert billing.schema == contract.schema["oneOf"][0]

    def test_record_fields_keep_order_names_docs_and_nesting(
        self, probe: ModuleType, tmp_path: Path
    ) -> None:
        contract = _only(probe, _TEAM + _TRIAGE + 'let t: Triage = capture("q")\n0', tmp_path)
        assert (contract.kind, contract.label, contract.doc) == (
            "record",
            "Triage",
            "A triaged message.",
        )
        assert contract.nominal is probe.agl.Triage
        assert [(key, name, doc) for key, (name, doc, _) in contract.fields.items()] == [
            ("urgent", "urgent", "Is it urgent?"),
            ("owner", "team", "Who handles it?"),
            ("notes", "notes", None),
        ]
        team = contract.fields["owner"].contract
        assert (team.kind, team.nominal) == ("enum", probe.agl.Team)
        notes = contract.fields["notes"].contract
        assert (notes.kind, notes.label, notes.items.kind) == ("array", "array[text]", "text")
        assert list(contract.schema["properties"]) == ["urgent", "owner", "notes"]

    def test_dict_values(self, probe: ModuleType, tmp_path: Path) -> None:
        contract = _only(probe, 'let d: dict[text, int] = capture("q")\n0', tmp_path)
        assert (contract.kind, contract.values.kind, contract.items) == ("dict", "int", None)

    def test_root_label_is_the_target_spelling(self, probe: ModuleType, tmp_path: Path) -> None:
        contract = _only(
            probe,
            _TEAM + "record Choice[C](choice: C, confidence: decimal)\n"
            'let c = capture::[Choice[Team]]("q")\n0',
            tmp_path,
        )
        assert contract.label == "Choice[Team]"
        assert contract.nominal is probe.agl.Choice
        assert contract.fields["choice"].contract.nominal is probe.agl.Team

    def test_recursive_target_is_a_cyclic_graph_with_self_contained_schemas(
        self, probe: ModuleType, tmp_path: Path
    ) -> None:
        contract = _only(probe, _TREE + 'let trees: array[Tree] = capture("q")\n0', tmp_path)
        tree = contract.items
        node = tree.members["Node"]
        assert node.fields["left"].contract is tree
        assert node.fields["right"].contract is tree
        assert contract.schema["items"] == {"$ref": "#/$defs/Tree"}
        assert set(contract.schema["$defs"]) == {"Tree"}
        assert tree.schema["$defs"] == contract.schema["$defs"]
        assert "oneOf" in tree.schema
        assert node.schema["$defs"] == contract.schema["$defs"]

    def test_recursive_root_is_its_shared_definition(
        self, probe: ModuleType, tmp_path: Path
    ) -> None:
        contract = _only(probe, _TREE + 'let tree: Tree = capture("q")\n0', tmp_path)
        assert (contract.kind, contract.label) == ("enum", "Tree")
        assert contract.members["Node"].fields["left"].contract is contract
        assert contract.schema["$defs"]["Tree"]["oneOf"] == contract.schema["oneOf"]

    def test_recursive_root_keeps_its_own_label(self, tmp_path: Path) -> None:
        program = lower_extern_program(
            _DECLS + _TREE + 'let tree: Tree = capture("q")\n0', _COMPANION, tmp_path
        )
        (request,) = program.contracts.values()
        classes = defaultdict(lambda: object)
        contract = build_type_contract(
            dataclasses.replace(request, target_type_label="Forest"), classes
        )
        assert (contract.kind, contract.label) == ("enum", "Forest")
        recursive = contract.members["Node"].fields["left"].contract
        assert recursive is not contract
        assert recursive.label == "Tree"
        assert recursive.members["Node"].fields["left"].contract is recursive

    def test_contract_is_immutable(self, probe: ModuleType, tmp_path: Path) -> None:
        contract = _only(probe, _TEAM + _TRIAGE + 'let t: Triage = capture("q")\n0', tmp_path)
        with pytest.raises(AttributeError):
            contract.label = "other"
        with pytest.raises(AttributeError):
            del contract.label
        with pytest.raises(TypeError):
            probe.agl.TypeContract()
        with pytest.raises(TypeError):
            contract.fields["urgent"] = contract.fields["notes"]
        with pytest.raises(TypeError):
            contract.members["x"] = contract
        contract.schema["properties"].clear()
        assert list(contract.schema["properties"]) == ["urgent", "owner", "notes"]
        assert "Triage" in repr(contract)

    def test_multiple_targets_arrive_in_declaration_order(
        self, probe: ModuleType, tmp_path: Path
    ) -> None:
        bindings = _run('let labels = both::[int, text]("q")\nlabels', tmp_path)
        assert bindings["labels"] == TextValue("int,text")
        assert [contract.label for contract in probe.seen] == ["int", "text"]

    def test_one_occurrence_delivers_one_cached_contract(
        self, probe: ModuleType, tmp_path: Path
    ) -> None:
        first, second, other = _delivered(
            probe,
            'def ask-int() -> int = capture("q")\n'
            'let a = ask-int()\nlet b = ask-int()\nlet c: int = capture("q")\n0',
            tmp_path,
        )
        assert first is second
        assert other is not first
        assert other.label == first.label


class TestConstruction:
    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("int", "7"),
            ("text", "q"),
            ("decimal", "1.5"),
            ("bool", "true"),
            ("json", '{"q": "q"}'),
            ("array[int]", "[7]"),
            ("dict[text, int]", '{"k": 7}'),
            ("Option[int]", "Option::None"),
        ],
    )
    def test_builds_target(self, tmp_path: Path, target: str, expected: str) -> None:
        bindings = _run(f'let value: {target} = build("q")\nlet shown = "%{{value}}"\n0', tmp_path)
        assert bindings["shown"] == TextValue(expected)

    def test_builds_enum_member(self, tmp_path: Path) -> None:
        bindings = _run(
            _TEAM + _BILLING + 'let team: Team = build("q")\nlet ok = team == billing\n0', tmp_path
        )
        assert bindings["ok"].value is True

    def test_builds_nested_record(self, tmp_path: Path) -> None:
        bindings = _run(
            _TEAM + _TRIAGE + 'let t: Triage = build("q")\n'
            'let ok = t == Triage(urgent = true, team = Team::Billing, notes = ["q"])\n0',
            tmp_path,
        )
        assert bindings["ok"].value is True

    def test_builds_recursive_generic_record(self, tmp_path: Path) -> None:
        bindings = _run(
            _TREE + "record Choice[C](choice: C, confidence: decimal)\n"
            'let c: Choice[Tree] = build("q")\n'
            "let leaf: Tree = Tree::Leaf(value = 7)\n"
            "let expected = Choice(choice = leaf, confidence = 1.5)\n"
            "let ok = c == expected\n0",
            tmp_path,
        )
        assert bindings["ok"].value is True

    def test_reference_delivers_contract_at_call_time(self, tmp_path: Path) -> None:
        bindings = _run(
            _TEAM + _BILLING + 'let make = build::[Team]\nlet ok = make("q") == billing\n0',
            tmp_path,
        )
        assert bindings["ok"].value is True

    def test_partial_application_delivers_contract_at_call_time(self, tmp_path: Path) -> None:
        bindings = _run('let make = build::[array[text]](?)\nlet made = make("p")\n0', tmp_path)
        made = bindings["made"]
        assert [element.value for element in made.elements] == ["p"]

    def test_return_stays_unchecked(self, tmp_path: Path) -> None:
        companion = "def build(contract, question):\n    return question\n"
        bindings, _ = evaluate_ir_with_externs(
            'extern def build[T](question: text) -> T\nlet n: int = build("q")\n0',
            companion,
            tmp_path,
        )
        assert bindings["n"] == TextValue("q")


class TestRepl:
    def test_contracts_cross_repl_entries(self, probe: ModuleType, tmp_path: Path) -> None:
        (tmp_path / "lib.agl").write_text(_DECLS)
        (tmp_path / "lib.py").write_text(_COMPANION)
        session = repl_session_with_root(tmp_path)
        assert session.eval_entry("import lib::*").ok
        assert session.eval_entry(_TEAM).ok
        assert session.eval_entry('def pick() -> Team = capture("q")').ok
        assert session.eval_entry(_BILLING).ok
        first = session.eval_entry("pick() == billing")
        assert first.ok, first.diagnostics
        second = session.eval_entry('let n: int = capture("q")\nn')
        assert second.ok, second.diagnostics
        assert second.value == IntValue(7)
        assert [contract.label for contract in probe.seen] == ["Team", "int"]
        assert probe.seen[0].nominal.Billing is probe.seen[0].members["billing"].nominal


class TestContractValueIsNotData:
    def test_render_rejects(self) -> None:
        with pytest.raises(AglNonDataValue):
            render_value(ContractValue(ContractId(0)), ValueDescriptors(nominals={}, functions={}))

    def test_serialize_rejects(self) -> None:
        with pytest.raises(AglNonDataValue):
            value_to_json_obj(ContractValue(ContractId(0)))

    def test_compares_by_identity_only(self) -> None:
        contract = ContractValue(ContractId(0))
        assert values_equal(contract, contract)
        assert not values_equal(contract, ContractValue(ContractId(0)))

    def test_crosses_only_into_an_extern_call(self) -> None:
        with pytest.raises(BoundaryViolation):
            encode_boundary_value(
                ContractValue(ContractId(0)), ValueDescriptors(nominals={}, functions={})
            )

    def test_companion_cannot_return_a_contract(self, tmp_path: Path) -> None:
        error = evaluate_ir_raises_with_externs(
            _DECLS + 'extern def leak[T](question: text) -> T\nlet n: int = leak("q")\n0',
            _COMPANION,
            tmp_path,
        )
        assert error.type_name == "ExternError"


def test_nested_ordinary_extern_call_resolves_no_outer_contract(tmp_path: Path) -> None:
    """A contract reaching a nested non-type-directed call is a boundary violation."""
    program = lower_extern_program(_DECLS + 'let n: int = capture("q")\n0', _COMPANION, tmp_path)
    (contract_id,) = program.contracts
    registry = ExternRegistry()
    descriptors = ValueDescriptors(nominals={}, functions={})
    nested: list[object] = []

    def invoke(
        fn: Callable[..., object], contracts: Mapping[ContractId, ContractRequest] | None = None
    ) -> Value:
        return registry.invoke(
            "call",
            fn,
            [ContractValue(contract_id)],
            nominals=program.builtin_nominals,
            descriptors=descriptors,
            contracts=contracts,
        )

    def outer(contract: TypeContract) -> None:
        try:
            nested.append(invoke(lambda leaked: None))
        except AglRaise as exc:
            nested.append(exc)

    assert isinstance(invoke(outer, program.contracts), UnitValue)
    (outcome,) = nested
    assert isinstance(outcome, AglRaise)
