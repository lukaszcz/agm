"""Decision-DAG compilation and structured match-diagnostic contracts."""

from __future__ import annotations

import decimal
import itertools
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import agm.agl.matchcompile.compiler as compiler_module
from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.matchcompile import (
    BoolWitness,
    EnumWitness,
    EnumWitnessQualification,
    LiteralWitness,
    NonExhaustiveIssue,
    OpenComplementWitness,
    RedundantArmIssue,
    WildcardWitness,
    WitnessField,
    render_witness,
)
from agm.agl.matchcompile.compiler import (
    CompiledCase,
    compile_case,
    validate_decision_dag,
)
from agm.agl.matchcompile.matrix import (
    OccurrenceAllocator,
    OccurrenceIndex,
    matrix_from_normalized,
)
from agm.agl.matchcompile.model import (
    BinderAssignment,
    BoolConstructor,
    ClosedSignature,
    ConstructorCell,
    Decision,
    DecisionBranch,
    DecisionFail,
    DecisionLeaf,
    DecisionSwitch,
    EnumConstructor,
    EnumConstructorSpelling,
    FieldOccurrenceProvenance,
    LiteralKind,
    MatchCaseContext,
    Occurrence,
    OccurrenceId,
    Signature,
    WildcardCell,
)
from agm.agl.matchcompile.normalize import (
    MatchCompileInvariantError,
    normalize_case,
    signature_for_type,
)
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import EnumType, IntType, Type, TypeTemplate
from agm.agl.semantics.values import BoolValue, EnumValue, Value
from agm.agl.syntax.nodes import Case
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import (
    CheckedModule,
    EnumOwnerForm,
    EnumOwnerFormKind,
    check_module,
    check_program,
)
from tests.agl.ir_harness import make_graph_from_files
from tests.agl.match_reference import reference_action

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
    },
)


def _compile(source: str) -> tuple[CheckedModule, Case, CompiledCase]:
    checked = check_module(resolve_module(parse_program(source)), _CAPS)
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(checked.resolved.program, collect)
    assert len(cases) == 1
    case = cases[0]
    return checked, case, compile_case(normalize_case(case, checked))


def _compile_graph_case(tmp_path: Path, modules: dict[str, str]) -> CompiledCase:
    graph = make_graph_from_files(tmp_path, modules)
    checked = check_program(resolve_program(graph), _CAPS).modules[ENTRY_ID]
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(checked.resolved.program, collect)
    assert len(cases) == 1
    return compile_case(normalize_case(cases[0], checked))


def _compile_without_normalized_rows(source: str) -> tuple[Case, CompiledCase]:
    _, case, compiled = _compile(source)
    normalized = replace(compiled.normalized, rows=())
    return case, compile_case(normalized)


def _branch_matches(constructor: object, value: Value) -> bool:
    if isinstance(constructor, BoolConstructor):
        return isinstance(value, BoolValue) and value.value is constructor.value
    if isinstance(constructor, EnumConstructor):
        return (
            isinstance(value, EnumValue)
            and value.nominal.module_id == constructor.enum_type.module_id
            and value.nominal.declared_name == constructor.enum_type.name
            and value.variant == constructor.variant
        )
    raise AssertionError("finite generated tests only use boolean and enum constructors")


def _decision_action(compiled: CompiledCase, subject: Value) -> int | None:
    occurrences = {occurrence.id: occurrence for occurrence in compiled.occurrences}

    def evaluate(decision: Decision, values: dict[object, Value]) -> int | None:
        if isinstance(decision, DecisionFail):
            return None
        if isinstance(decision, DecisionLeaf):
            return decision.action_id
        value = values[decision.occurrence.id]
        for branch in decision.keyed_children:
            if not _branch_matches(branch.constructor, value):
                continue
            next_values = dict(values)
            if isinstance(branch.constructor, EnumConstructor):
                assert isinstance(value, EnumValue)

                def creation_order(occurrence: Occurrence) -> int:
                    return occurrence.creation_order

                children = sorted(
                    (
                        occurrence
                        for occurrence in occurrences.values()
                        if isinstance(occurrence.provenance, FieldOccurrenceProvenance)
                        and occurrence.provenance.parent == decision.occurrence.id
                        and occurrence.provenance.constructor == branch.constructor
                    ),
                    key=creation_order,
                )
                for field, child in zip(branch.constructor.fields, children, strict=True):
                    next_values[child.id] = value.fields[field.name]
            return evaluate(branch.decision, next_values)
        assert decision.default is not None
        return evaluate(decision.default, values)

    return evaluate(compiled.root, {compiled.normalized.root.id: subject})


def _has_fail(root: Decision) -> bool:
    seen: set[int] = set()

    def visit(decision: Decision) -> bool:
        if id(decision) in seen:
            return False
        seen.add(id(decision))
        if isinstance(decision, DecisionFail):
            return True
        if isinstance(decision, DecisionLeaf):
            return False
        return any(visit(branch.decision) for branch in decision.keyed_children) or (
            decision.default is not None and visit(decision.default)
        )

    return visit(root)


def _decision_references(root: Decision) -> tuple[int, int]:
    unique: set[int] = set()
    references = 0

    def visit(decision: Decision) -> None:
        nonlocal references
        references += 1
        if id(decision) in unique:
            return
        unique.add(id(decision))
        if isinstance(decision, DecisionSwitch):
            for branch in decision.keyed_children:
                visit(branch.decision)
            if decision.default is not None:
                visit(decision.default)

    visit(root)
    return len(unique), references


def _unique_decision_ids(root: Decision) -> frozenset[int]:
    unique: set[int] = set()

    def visit(decision: Decision) -> None:
        if id(decision) in unique:
            return
        unique.add(id(decision))
        if isinstance(decision, DecisionSwitch):
            for branch in decision.keyed_children:
                visit(branch.decision)
            if decision.default is not None:
                visit(decision.default)

    visit(root)
    return frozenset(unique)


def _wide_enum_source(size: int) -> str:
    variants = "".join(f"  | item{index}\n" for index in range(size))
    branches = "".join(f"  | item{index} => {index}\n" for index in range(size))
    return f"enum Choice\n{variants}let value: Choice = item0\ncase value of\n{branches}"


def _diagonal_source(size: int, *, exhaustive: bool) -> str:
    fields = ", ".join(f"slot{index}: Slot" for index in range(size))
    rows = [
        f"  | covered(value = vector(slot{index} = one(tail = end))) => {index}"
        for index in range(size)
    ]
    rows.append(f"  | covered(value = _) => {size}")
    if exhaustive:
        rows.append(f"  | missing => {size + 1}")
    return (
        "enum End\n"
        "  | end\n"
        "  | more\n"
        "enum Slot\n"
        "  | one(tail: End)\n"
        "  | empty\n"
        f"enum Vector\n  | vector({fields})\n"
        "enum Subject\n"
        "  | covered(value: Vector)\n"
        "  | missing\n"
        f"let value = missing\ncase value of\n{'\n'.join(rows)}"
    )


def _path_test_counts(root: Decision) -> tuple[int, ...]:
    counts: list[int] = []

    def visit(decision: Decision, count: int) -> None:
        if not isinstance(decision, DecisionSwitch):
            counts.append(count)
            return
        for branch in decision.keyed_children:
            visit(branch.decision, count + 1)
        if decision.default is not None:
            visit(decision.default, count + 1)

    visit(root, 0)
    return tuple(counts)


def _nested_pair_value(
    pair_type: EnumType,
    bit_type: EnumType,
    left: str,
    right: str,
) -> EnumValue:
    bit_nominal = NominalId(bit_type.module_id, bit_type.name)
    return EnumValue(
        NominalId(pair_type.module_id, pair_type.name),
        pair_type.name,
        "pair",
        {
            "left": EnumValue(bit_nominal, bit_type.name, left, {}),
            "right": EnumValue(bit_nominal, bit_type.name, right, {}),
        },
    )


def test_irrefutable_leaf_finalizes_binder_and_has_no_issues() -> None:
    _, case, compiled = _compile("let value = 1\ncase value of | captured => captured")

    assert isinstance(compiled.root, DecisionLeaf)
    assert compiled.root.action_id == case.branches[0].node_id
    assert compiled.root.binder_assignments[0].occurrence == compiled.normalized.root.id
    assert compiled.root.free_occurrences == (compiled.normalized.root.id,)
    assert compiled.reachable_action_ids == (case.branches[0].node_id,)
    assert compiled.issues == ()
    assert compiled.case_node_id == case.node_id
    assert compiled.actions == compiled.normalized.actions


def test_boolean_switch_is_complete_and_has_signature_order() -> None:
    _, case, compiled = _compile("let value = false\ncase value of | true => 1 | false => 0")

    assert isinstance(compiled.root, DecisionSwitch)
    assert [branch.constructor for branch in compiled.root.keyed_children] == [
        BoolConstructor(False),
        BoolConstructor(True),
    ]
    assert compiled.root.default is None
    assert compiled.issues == ()
    assert compiled.reachable_action_ids == tuple(branch.node_id for branch in case.branches)


def test_nested_missing_witness_is_structured_from_the_first_failure_path() -> None:
    _, case, compiled = _compile(
        "enum Box\n"
        "  | box(flag: bool)\n"
        "  | empty\n"
        "let value = box(flag = false)\n"
        "case value of | box(flag = false) => 1 | empty => 0"
    )

    assert len(compiled.issues) == 1
    issue = compiled.issues[0]
    assert isinstance(issue, NonExhaustiveIssue)
    assert issue.case_node_id == case.node_id
    assert issue.span == case.span
    assert isinstance(issue.witness, EnumWitness)
    assert issue.witness.variant == "box"
    assert issue.witness.fields[0].name == "flag"
    assert issue.witness.fields[0].witness == BoolWitness(True)
    assert "box" in render_witness(issue.witness)


def test_nested_enum_and_boolean_signatures_can_be_exhaustive_without_default() -> None:
    _, _, compiled = _compile(
        "enum Box\n"
        "  | box(flag: bool)\n"
        "  | empty\n"
        "let value = empty\n"
        "case value of\n"
        "  | box(flag = false) => 0\n"
        "  | box(flag = true) => 1\n"
        "  | empty => 2\n"
    )

    assert compiled.issues == ()
    assert not _has_fail(compiled.root)


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("let value = 1\ncase value of | 1 => 1 | 2 => 2", LiteralKind.NUMERIC),
        ('let value = "x"\ncase value of | "x" => 1', LiteralKind.TEXT),
    ],
)
def test_open_domains_require_catch_all_with_symbolic_complement(
    source: str, kind: LiteralKind
) -> None:
    _, _, compiled = _compile(source)

    issue = compiled.issues[0]
    assert isinstance(issue, NonExhaustiveIssue)
    assert isinstance(issue.witness, OpenComplementWitness)
    assert [constructor.kind for constructor in issue.witness.excluded] == [kind] * len(
        issue.witness.excluded
    )
    assert "other than" in render_witness(issue.witness)


def test_every_arm_on_a_bottom_scrutinee_is_redundant() -> None:
    _, case, compiled = _compile(
        "exception E extends Exception\n"
        "  code: int\n"
        'case (raise E(message = "x", code = 1)) of | _ => 1 | value => 2'
    )

    assert not any(isinstance(issue, NonExhaustiveIssue) for issue in compiled.issues)
    redundant = [issue for issue in compiled.issues if isinstance(issue, RedundantArmIssue)]
    assert [issue.action_id for issue in redundant] == [branch.node_id for branch in case.branches]


def test_duplicate_subsumed_and_uninhabited_arms_are_each_redundant() -> None:
    _, case, compiled = _compile(
        "let value: int = 1\ncase value of | 1 => 1 | 1.0 => 2 | _ => 3 | 1.5 => 4"
    )

    redundant = [issue for issue in compiled.issues if isinstance(issue, RedundantArmIssue)]
    assert [issue.action_id for issue in redundant] == [
        case.branches[1].node_id,
        case.branches[3].node_id,
    ]
    assert [issue.span for issue in redundant] == [
        case.branches[1].pattern.span,
        case.branches[3].pattern.span,
    ]
    assert not any(isinstance(issue, NonExhaustiveIssue) for issue in compiled.issues)


def test_partially_overlapping_arm_remains_reachable() -> None:
    _, case, compiled = _compile(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "let value = pair(left = false, right = false)\n"
        "case value of\n"
        "  | pair(left = false) => 1\n"
        "  | pair(right = false) => 2\n"
        "  | _ => 3\n"
    )

    assert compiled.issues == ()
    assert compiled.reachable_action_ids == tuple(branch.node_id for branch in case.branches)


def test_binder_in_specialized_default_row_targets_the_dominated_child_occurrence() -> None:
    _, case, compiled = _compile(
        "enum Box\n"
        "  | box(flag: bool)\n"
        "let value = box(flag = false)\n"
        "case value of | box(flag = false) => false | box(flag = captured) => captured"
    )
    root = cast(DecisionSwitch, compiled.root)
    nested = cast(DecisionSwitch, root.keyed_children[0].decision)
    fallback = cast(DecisionLeaf, nested.default)
    assignment = fallback.binder_assignments[0]

    assert fallback.action_id == case.branches[1].node_id
    assert assignment.occurrence != compiled.normalized.root.id
    assert isinstance(
        next(
            occurrence.provenance
            for occurrence in compiled.occurrences
            if occurrence.id == assignment.occurrence
        ),
        FieldOccurrenceProvenance,
    )


def test_multiple_issues_are_sorted_by_primary_source_span() -> None:
    _, case, compiled = _compile("let value = false\ncase value of | false => 1 | false => 2")

    assert [type(issue) for issue in compiled.issues] == [
        NonExhaustiveIssue,
        RedundantArmIssue,
    ]
    assert cast(RedundantArmIssue, compiled.issues[1]).action_id == case.branches[1].node_id


def test_qba_reordering_preserves_source_priority_for_every_pair_value() -> None:
    checked, case, compiled = _compile(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "let value = pair(left = false, right = false)\n"
        "case value of\n"
        "  | pair(right = false) => 1\n"
        "  | pair(left = false) => 2\n"
        "  | _ => 3\n"
    )
    root = cast(DecisionSwitch, compiled.root)
    pair = cast(EnumConstructor, root.keyed_children[0].constructor)
    nominal = NominalId(pair.enum_type.module_id, pair.enum_type.name)

    for left, right in itertools.product((False, True), repeat=2):
        value = EnumValue(
            nominal,
            pair.enum_type.name,
            pair.variant,
            {"left": BoolValue(left), "right": BoolValue(right)},
        )
        assert _decision_action(compiled, value) == reference_action(case, checked, value)


def test_generated_finite_matrices_match_reference_reachability_and_failure() -> None:
    row_patterns = (
        "pair(left = false, right = false)",
        "pair(left = false)",
        "pair(right = true)",
        "pair(left = true, right = true)",
        "_",
    )
    for indices in itertools.product(range(len(row_patterns)), repeat=3):
        patterns = [row_patterns[index] for index in indices]
        source = (
            "enum Pair\n  | pair(left: bool, right: bool)\n"
            "let value = pair(left = false, right = false)\ncase value of\n"
            + "\n".join(f"  | {pattern} => {action}" for action, pattern in enumerate(patterns))
        )
        checked, case, compiled = _compile(source)
        pair_type = cast(EnumType, compiled.normalized.root.type)
        nominal = NominalId(pair_type.module_id, pair_type.name)
        expected_actions: set[int] = set()
        unmatched = False
        for left, right in itertools.product((False, True), repeat=2):
            value = EnumValue(
                nominal,
                pair_type.name,
                "pair",
                {"left": BoolValue(left), "right": BoolValue(right)},
            )
            expected = reference_action(case, checked, value)
            actual = _decision_action(compiled, value)
            assert actual == expected
            unmatched |= expected is None
            if expected is not None:
                expected_actions.add(expected)
        assert set(compiled.reachable_action_ids) == expected_actions
        assert _has_fail(compiled.root) is unmatched


def test_generated_nested_multi_column_matrices_match_the_reference() -> None:
    row_patterns = (
        "pair(left = zero, right = zero)",
        "pair(left = zero)",
        "pair(right = one)",
        "pair(left = one, right = one)",
        "pair(left = left)",
        "pair(right = right)",
        "missing",
        "_",
    )
    generated_indices = (
        *itertools.product(range(len(row_patterns)), repeat=2),
        (0, 2, 7),
        (1, 2, 7),
        (2, 1, 7),
        (1, 3, 6),
        (4, 5, 6),
        (0, 0, 7),
        (7, 0, 6),
    )
    for indices in generated_indices:
        patterns = [row_patterns[index] for index in indices]
        source = (
            "enum Bit\n  | zero\n  | one\n"
            "enum Subject\n"
            "  | pair(left: Bit, right: Bit)\n"
            "  | missing\n"
            "let value: Subject = missing\ncase value of\n"
            + "\n".join(f"  | {pattern} => {action}" for action, pattern in enumerate(patterns))
        )
        checked, case, compiled = _compile(source)
        pair_type = cast(EnumType, compiled.normalized.root.type)
        pair_constructor = cast(
            EnumConstructor,
            cast(
                ClosedSignature, signature_for_type(pair_type, checked.type_env.type_table)
            ).constructors[0],
        )
        bit_type = cast(EnumType, pair_constructor.fields[0].type)
        values: tuple[Value, ...] = (
            *(
                _nested_pair_value(pair_type, bit_type, left, right)
                for left, right in itertools.product(("zero", "one"), repeat=2)
            ),
            EnumValue(
                NominalId(pair_type.module_id, pair_type.name),
                pair_type.name,
                "missing",
                {},
            ),
        )
        expected_actions: set[int] = set()
        unmatched = False
        for value in values:
            expected = reference_action(case, checked, value)
            assert _decision_action(compiled, value) == expected
            unmatched |= expected is None
            if expected is not None:
                expected_actions.add(expected)
        assert set(compiled.reachable_action_ids) == expected_actions
        assert _has_fail(compiled.root) is unmatched
        validate_decision_dag(compiled.root)


@pytest.mark.parametrize(
    ("size", "unique_nodes", "decision_references", "paths"),
    [(2, 10, 12, 8), (3, 13, 16, 16), (4, 16, 20, 32)],
)
def test_paper_diagonal_matrices_have_stable_structural_counts(
    size: int,
    unique_nodes: int,
    decision_references: int,
    paths: int,
) -> None:
    _, _, compiled = _compile(_diagonal_source(size, exhaustive=True))

    assert _decision_references(compiled.root) == (unique_nodes, decision_references)
    assert len(_path_test_counts(compiled.root)) == paths
    assert unique_nodes < decision_references
    validate_decision_dag(compiled.root)


def test_compiled_decisions_are_acyclic_single_test_paths_and_locally_shared() -> None:
    _, _, compiled = _compile(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "let value = pair(left = false, right = false)\n"
        "case value of | pair(left = false, right = false) => 1 | _ => 2"
    )

    validate_decision_dag(compiled.root)
    unique, references = _decision_references(compiled.root)
    assert (unique, references) == (5, 6)
    assert _path_test_counts(compiled.root) == (3, 3, 2)
    repeated: Counter[int] = Counter()

    def collect(decision: Decision) -> None:
        repeated[id(decision)] += 1
        if isinstance(decision, DecisionSwitch):
            for branch in decision.keyed_children:
                collect(branch.decision)
            if decision.default is not None:
                collect(decision.default)

    collect(compiled.root)
    assert max(repeated.values()) > 1
    assert unique < references
    root = cast(DecisionSwitch, compiled.root)
    left = cast(DecisionSwitch, root.keyed_children[0].decision)
    right = cast(DecisionSwitch, left.keyed_children[0].decision)
    assert right.default is left.default


def test_large_paper_diagonal_matrix_validates_shared_dag_without_path_expansion() -> None:
    _, _, compiled = _compile(_diagonal_source(15, exhaustive=True))

    validate_decision_dag(compiled.root)


def test_wide_enum_match_does_not_rebuild_its_signature_per_matrix_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = TypeTable.enum_variants

    def count_enum_variants(self: TypeTable, handle: EnumType) -> Mapping[str, Mapping[str, Type]]:
        nonlocal calls
        calls += 1
        return original(self, handle)

    monkeypatch.setattr(TypeTable, "enum_variants", count_enum_variants)

    size = 20
    _compile(_wide_enum_source(size))

    assert calls < 7 * size**2


@pytest.mark.parametrize("exhaustive", [False, True])
def test_failure_analysis_memoizes_paper_diagonal_dag_by_identity(
    exhaustive: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, compiled = _compile(_diagonal_source(10, exhaustive=exhaustive))
    unique = _unique_decision_ids(compiled.root)
    signature_calls = 0
    original = compiler_module.signature_for_type

    def counted_signature_for_type(subject_type: Type, table: TypeTable) -> Signature:
        nonlocal signature_calls
        signature_calls += 1
        return original(subject_type, table)

    monkeypatch.setattr(compiler_module, "signature_for_type", counted_signature_for_type)

    constraints = compiler_module._first_failure_constraints(
        compiled.root, compiled.normalized.type_table
    )

    # Each analyzed switch resolves its signature once, plus once more for a
    # default edge, so bounded signature work over a DAG with an order of
    # magnitude more paths than nodes means every node was analyzed once.
    assert signature_calls <= 2 * len(unique)
    assert len(_path_test_counts(compiled.root)) > len(unique) * 10
    assert (constraints is None) is exhaustive


def test_decisions_do_not_share_between_source_cases() -> None:
    _, _, first = _compile("let value = false\ncase value of | false => 0 | true => 1")
    _, _, second = _compile("let value = false\ncase value of | false => 0 | true => 1")

    assert first.root is not second.root
    assert isinstance(first.root, DecisionSwitch)
    assert isinstance(second.root, DecisionSwitch)
    assert first.root.keyed_children[0].decision is not second.root.keyed_children[0].decision


def test_validator_rejects_a_repeated_occurrence_test_on_one_path() -> None:
    _, _, compiled = _compile("let value = false\ncase value of | false => 0 | true => 1")
    root = cast(DecisionSwitch, compiled.root)
    invalid = DecisionSwitch(
        occurrence=root.occurrence,
        keyed_children=(type(root.keyed_children[0])(BoolConstructor(False), root),),
        default=None,
        free_occurrences=root.free_occurrences,
    )

    with pytest.raises(MatchCompileInvariantError, match="cycle|more than once"):
        validate_decision_dag(invalid)


def test_enum_witness_uses_wildcards_for_unconstrained_fields() -> None:
    _, _, compiled = _compile(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "  | empty\n"
        "let value = empty\n"
        "case value of | empty => 0"
    )
    issue = cast(NonExhaustiveIssue, compiled.issues[0])
    witness = cast(EnumWitness, issue.witness)

    assert witness.variant == "pair"
    assert [field.witness for field in witness.fields] == [
        WildcardWitness(),
        WildcardWitness(),
    ]
    assert render_witness(witness) == "pair(left = _, right = _)"
    assert witness.qualification is None


@pytest.mark.parametrize(
    ("covered_pattern", "rendered"),
    [
        ("Left::item(value = _)", "Left::empty"),
        ("Left::empty", "Left::item(value = _)"),
    ],
)
def test_ambiguous_local_enum_witnesses_use_type_qualification(
    covered_pattern: str, rendered: str
) -> None:
    _, _, compiled = _compile(
        "enum Left\n  | empty\n  | item(value: int)\n"
        "enum Right\n  | empty\n  | item(value: int)\n"
        "let value: Left = Left::empty\n"
        f"case value of | {covered_pattern} => 0"
    )
    issue = cast(NonExhaustiveIssue, compiled.issues[0])
    witness = cast(EnumWitness, issue.witness)

    assert witness.qualification == EnumWitnessQualification("Left", None)
    assert render_witness(witness) == rendered


def test_nested_ambiguous_enum_witnesses_are_qualified_recursively() -> None:
    _, _, compiled = _compile(
        "enum LeftInner\n  | missing\n  | present\n"
        "enum RightInner\n  | missing\n  | present\n"
        "enum LeftOuter\n  | wrap(value: LeftInner)\n  | finished\n"
        "enum RightOuter\n  | wrap(value: RightInner)\n  | finished\n"
        "let value: LeftOuter = LeftOuter::wrap(value = LeftInner::present)\n"
        "case value of\n"
        "  | LeftOuter::wrap(value = LeftInner::present) => 1\n"
        "  | LeftOuter::finished => 0\n"
    )
    outer = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)
    inner = cast(EnumWitness, outer.fields[0].witness)

    assert outer.qualification == EnumWitnessQualification("LeftOuter", None)
    assert inner.qualification == EnumWitnessQualification("LeftInner", None)
    assert render_witness(outer) == "LeftOuter::wrap(value = LeftInner::missing)"


@pytest.mark.parametrize("ambiguous", [False, True])
def test_imported_enum_witness_uses_full_module_qualification_only_when_ambiguous(
    tmp_path: Path, ambiguous: bool
) -> None:
    other_import = "import other\n" if ambiguous else ""
    pattern = "library.remote::Remote::empty" if ambiguous else "empty"
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": "enum Remote[T]\n  | empty\n  | item(value: T)",
            "other": "enum Other\n  | empty\n  | item(value: int)",
            "entry": (
                "import library.remote\n"
                f"{other_import}"
                "let value: library.remote::Remote[int] = "
                "library.remote::Remote::empty\n"
                f"case value of | {pattern} => 0\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    expected_qualification = EnumWitnessQualification("Remote", None) if ambiguous else None
    assert witness.qualification == expected_qualification
    assert render_witness(witness) == (
        "Remote::item(value = _)" if ambiguous else "item(value = _)"
    )


def test_qualified_only_imported_enum_witness_uses_source_alias(tmp_path: Path) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": "enum Remote\n  | empty\n  | item(value: int)",
            "entry": (
                "import library.remote qualified as r\n"
                "let value: r::Remote = r::Remote::empty\n"
                "case value of | r::Remote::empty => 0\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("Remote", ("r",))
    assert render_witness(witness) == "r::Remote::item(value = _)"


def test_aliased_qualified_imports_choose_target_source_handle(tmp_path: Path) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/left": "enum Shared\n  | empty\n  | item(value: int)",
            "library/right": "enum Shared\n  | empty\n  | item(value: int)",
            "entry": (
                "import library.left qualified as left\n"
                "import library.right qualified as right\n"
                "enum Shared\n  | empty\n  | item(value: int)\n"
                "let value: left::Shared = left::Shared::empty\n"
                "case value of | left::Shared::empty => 0\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("Shared", ("left",))
    assert render_witness(witness) == "left::Shared::item(value = _)"


def test_lexical_binding_shadowing_variant_forces_type_qualified_witness() -> None:
    _, _, compiled = _compile(
        "enum Choice\n  | empty\n  | item(value: int)\n"
        "def inspect(item: int, value: Choice) -> int =\n"
        "  case value of | Choice::empty => 0\n"
        "let result = inspect(1, Choice::empty)\n"
        "result\n"
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("Choice", None)
    assert render_witness(witness) == "Choice::item(value = _)"


def test_later_lexical_binding_does_not_retroactively_shadow_case_variant() -> None:
    _, _, compiled = _compile(
        "enum Choice\n  | empty\n  | item(value: int)\n"
        "def inspect(value: Choice) -> int =\n"
        "  let result = case value of | Choice::empty => 0\n"
        "  let item = 1\n"
        "  result\n"
        "inspect(Choice::empty)\n"
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification is None
    assert render_witness(witness) == "item(value = _)"


def test_renamed_open_import_uses_exposed_type_name_when_bare_variant_is_shadowed(
    tmp_path: Path,
) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": "enum Remote\n  | empty\n  | item(value: int)",
            "entry": (
                "import library.remote using Remote as R\n"
                "def inspect(item: int, value: R) -> int =\n"
                "  case value of | R::empty => 0\n"
                "let result = inspect(1, R::empty)\n"
                "result\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("R", None)
    assert render_witness(witness) == "R::item(value = _)"


def test_hidden_imported_type_allows_irrefutable_case_without_invented_spelling(
    tmp_path: Path,
) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": (
                "enum Remote\n  | empty\n  | item(value: int)\ndef make() -> Remote = empty\n"
            ),
            "entry": (
                "import library.remote using make\nlet value = make()\ncase value of | _ => 0\n"
            ),
        },
    )

    assert compiled.issues == ()

    unavailable = compile_case(replace(compiled.normalized, rows=()))
    issue = cast(NonExhaustiveIssue, unavailable.issues[0])
    assert issue.witness == WildcardWitness()
    assert render_witness(issue.witness) == "_"


@pytest.mark.parametrize(
    ("import_line", "owner", "module_qualifier", "rendered"),
    [
        (
            "import library.remote using Alias",
            "Alias",
            None,
            "Alias::item(value = _)",
        ),
        (
            "import library.remote using Alias as A",
            "A",
            None,
            "A::item(value = _)",
        ),
        (
            "import library.remote qualified as r using Alias",
            "Alias",
            ("r",),
            "r::Alias::item(value = _)",
        ),
    ],
)
def test_imported_transparent_alias_witness_uses_exposed_source_name(
    tmp_path: Path,
    import_line: str,
    owner: str,
    module_qualifier: tuple[str, ...] | None,
    rendered: str,
) -> None:
    prefix = "" if module_qualifier is None else f"{'.'.join(module_qualifier)}::"
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": (
                "enum Remote\n  | empty\n  | item(value: int)\ntype Alias = Remote\n"
            ),
            "entry": (
                f"{import_line}\n"
                f"let value: {prefix}{owner} = {prefix}{owner}::empty\n"
                f"case value of | {prefix}{owner}::empty => 0\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification(owner, module_qualifier)
    assert render_witness(witness) == rendered


def test_imported_transformed_generic_alias_matches_concrete_enum_owner(
    tmp_path: Path,
) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": (
                "record Pair[A, B]\n  first: A\n  second: B\n"
                "enum Remote[T]\n  | empty\n  | item(value: T)\n"
                "type Flipped[A, B] = Remote[Pair[B, A]]\n"
                "def make() -> Remote[Pair[int, text]] = Remote::empty\n"
            ),
            "entry": (
                "import library.remote using Flipped, make\n"
                "let value = make()\n"
                "case value of | Flipped::empty => 0\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("Flipped", None)
    assert render_witness(witness) == "Flipped::item(value = _)"


def test_local_generic_alias_is_retained_as_checked_template_candidate() -> None:
    _, _, compiled = _compile(
        "enum Remote[T]\n  | empty\n  | item(value: T)\n"
        "type Alias[T] = Remote[T]\n"
        "let value: Remote[int] = Remote::empty\n"
        "case value of | Remote::empty => 0\n"
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)
    alias_candidates = tuple(
        form
        for form in compiled.normalized.case_context.enum_owner_forms
        if form.owner_name == "Alias"
    )

    assert len(alias_candidates) == 2
    assert all(
        form.match(EnumType("Remote", (IntType(),))) is not None for form in alias_candidates
    )
    assert render_witness(witness) == "item(value = _)"


def test_rendered_local_generic_alias_owner_round_trips_through_checker() -> None:
    declaration = "enum Remote[T]\n  | empty\n  | item(value: T)\ntype Alias[T] = Remote[T]\n"
    _, _, compiled = _compile(
        declaration
        + "def inspect(item: int, value: Remote[int]) -> int =\n"
        + "  case value of | Remote::empty => 0\n"
        + "inspect(1, Remote::empty)\n"
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert render_witness(witness) == "Alias::item(value = _)"
    _compile(
        declaration
        + "def inspect(value: Remote[int]) -> int =\n"
        + "  case value of | Alias::item(value = _) => 0\n"
        + "inspect(Remote::empty)\n"
    )


def test_witness_prefers_shortest_valid_enum_owner_spelling() -> None:
    declaration = "enum R[T]\n  | empty\n  | item(value: T)\ntype LongAlias[T] = R[T]\n"
    _, _, compiled = _compile(
        declaration
        + "def inspect(item: int, value: R[int]) -> int =\n"
        + "  case value of | R::empty => 0\n"
        + "inspect(1, R::empty)\n"
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert render_witness(witness) == "R::item(value = _)"


def test_imported_generic_alias_with_fixed_argument_matches_enum_owner(
    tmp_path: Path,
) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": (
                "record Pair[A, B]\n  first: A\n  second: B\n"
                "enum Remote[T]\n  | empty\n  | item(value: T)\n"
                "type Fixed[T] = Remote[Pair[T, int]]\n"
                "def make() -> Remote[Pair[text, int]] = Remote::empty\n"
            ),
            "entry": (
                "import library.remote using Fixed, make\n"
                "let value = make()\n"
                "case value of | Fixed::empty => 0\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("Fixed", None)
    assert render_witness(witness) == "Fixed::item(value = _)"


def test_qualified_generic_identity_alias_uses_source_handle(tmp_path: Path) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": (
                "enum Remote[T]\n  | empty\n  | item(value: T)\n"
                "type Alias[T] = Remote[T]\n"
                "def make() -> Remote[int] = Remote::empty\n"
            ),
            "entry": (
                "import library.remote qualified as r using Alias, make\n"
                "let value = r::make()\n"
                "case value of | r::Alias::empty => 0\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("Alias", ("r",))
    assert render_witness(witness) == "r::Alias::item(value = _)"


def test_negative_alias_to_other_enum_is_not_selected_as_owner(tmp_path: Path) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": (
                "enum Remote\n  | empty\n  | item(value: int)\n"
                "enum Other\n  | empty\n  | item(value: int)\n"
                "type Wrong = Other\n"
                "def make() -> Remote = Remote::empty\n"
            ),
            "entry": (
                "import library.remote using Wrong, make\n"
                "let value = make()\n"
                "case value of | _ => 0\n"
            ),
        },
    )
    unavailable = compile_case(replace(compiled.normalized, rows=()))
    issue = cast(NonExhaustiveIssue, unavailable.issues[0])

    assert issue.witness == WildcardWitness()


def test_local_owner_form_blocks_shadowed_open_import_spelling(tmp_path: Path) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": (
                "enum Remote\n  | empty\n  | item(value: int)\n"
                "def make() -> Remote = Remote::empty\n"
            ),
            "entry": (
                "import library.remote using Remote as Clash, make\n"
                "enum Clash\n  | local\n"
                "def inspect(empty: int, item: int) -> int =\n"
                "  case make() of | _ => 0\n"
                "inspect(1, 2)\n"
            ),
        },
    )
    unavailable = compile_case(replace(compiled.normalized, rows=()))
    issue = cast(NonExhaustiveIssue, unavailable.issues[0])

    assert render_witness(issue.witness) == "library.remote::Clash::empty"
    assert not any(
        form.owner_name == "Clash" and form.kind is EnumOwnerFormKind.OPEN_IMPORT
        for form in compiled.normalized.case_context.enum_owner_forms
    )


def test_reexported_alias_chain_uses_final_exposed_name(tmp_path: Path) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/base": (
                "enum Remote\n  | empty\n  | item(value: int)\n"
                "type Alias = Remote\n"
                "type Chained = Alias\n"
            ),
            "library/facade": "export library.base using Chained as Public",
            "entry": (
                "import library.facade using Public\n"
                "let value: Public = Public::empty\n"
                "case value of | Public::empty => 0\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("Public", None)
    assert render_witness(witness) == "Public::item(value = _)"


def test_nested_witness_selects_alias_for_each_concrete_instantiation(
    tmp_path: Path,
) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": (
                "enum Remote[T]\n  | empty\n  | item(value: T)\n"
                "type IntRemote = Remote[int]\n"
                "type TextRemote = Remote[text]\n"
                "enum Pair\n"
                "  | pair(left: Remote[int], right: Remote[text])\n"
                "def make() -> Pair = Pair::pair(\n"
                "  left = Remote::empty, right = Remote::empty\n"
                ")\n"
            ),
            "entry": (
                "import library.remote using IntRemote, TextRemote, Pair, make\n"
                "let value = make()\n"
                "case value of\n"
                "  | Pair::pair(left = IntRemote::empty, right = _) => 0\n"
                "  | Pair::pair(left = _, right = TextRemote::empty) => 1\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert render_witness(witness) == (
        "pair(left = IntRemote::item(value = _), right = TextRemote::item(value = _))"
    )


def test_polymorphic_nested_instantiation_selects_generic_alias_template(
    tmp_path: Path,
) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/perfect": (
                "enum Box[T]\n  | box(value: T)\n"
                "enum Perfect[T]\n"
                "  | end\n"
                "  | value(item: T)\n"
                "  | next(value: Perfect[Box[T]])\n"
                "type Root[T] = Perfect[T]\n"
                "type Nested[T] = Perfect[Box[T]]\n"
                "def make() -> Perfect[int] = Perfect::end\n"
            ),
            "entry": (
                "import library.perfect using Root, Nested as N, make\n"
                "let value = make()\n"
                "case value of\n"
                "  | Root::end => 0\n"
                "  | Root::value(item = _) => 1\n"
                "  | Root::next(value = N::end) => 2\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert render_witness(witness) == "Root::next(value = N::value(item = _))"


def test_local_type_uses_self_qualification_when_import_handle_conflicts(
    tmp_path: Path,
) -> None:
    compiled = _compile_graph_case(
        tmp_path,
        {
            "library/remote": "def value() -> int = 1",
            "entry": (
                "import library.remote qualified as Choice\n"
                "enum Choice\n  | empty\n  | item(value: int)\n"
                "def inspect(item: int, value: ::Choice) -> int =\n"
                "  case value of | ::Choice::empty => 0\n"
                "let result = inspect(1, ::Choice::empty)\n"
                "result\n"
            ),
        },
    )
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert witness.qualification == EnumWitnessQualification("Choice", ())
    assert render_witness(witness) == "::Choice::item(value = _)"


def test_rendered_self_qualified_generic_owner_round_trips_under_handle_conflict(
    tmp_path: Path,
) -> None:
    modules = {
        "library/remote": "def value() -> int = 1",
        "entry": (
            "import library.remote qualified as Remote\n"
            "enum Remote[T]\n  | empty\n  | item(value: T)\n"
            "def inspect(item: int, value: ::Remote[int]) -> int =\n"
            "  case value of | ::Remote::empty => 0\n"
            "let result = inspect(1, ::Remote::empty)\n"
            "result\n"
        ),
    }
    compiled = _compile_graph_case(tmp_path, modules)
    witness = cast(EnumWitness, cast(NonExhaustiveIssue, compiled.issues[0]).witness)

    assert render_witness(witness) == "::Remote::item(value = _)"
    _compile_graph_case(
        tmp_path,
        {
            **modules,
            "entry": (
                "import library.remote qualified as Remote\n"
                "enum Remote[T]\n  | empty\n  | item(value: T)\n"
                "def inspect(value: ::Remote[int]) -> int =\n"
                "  case value of | ::Remote::item(value = _) => 0\n"
                "inspect(::Remote::empty)\n"
            ),
        },
    )


def test_whole_generic_enum_domain_uses_first_declaration_order_constructor() -> None:
    _, case, compiled = _compile(
        "enum Box[T]\n"
        "  | filled(value: T)\n"
        "  | empty\n"
        "let value: Box[int] = filled(value = 1)\n"
        "case value of | filled(value = 1.5) => 0"
    )

    assert compiled.normalized.rows == ()
    assert [type(issue) for issue in compiled.issues] == [
        NonExhaustiveIssue,
        RedundantArmIssue,
    ]
    assert cast(RedundantArmIssue, compiled.issues[1]).action_id == case.branches[0].node_id
    witness = cast(NonExhaustiveIssue, compiled.issues[0]).witness
    assert witness == EnumWitness(
        cast(EnumType, compiled.normalized.root.type),
        "filled",
        (WitnessField("value", WildcardWitness()),),
    )
    assert render_witness(witness) == "filled(value = _)"


def test_polymorphic_recursive_enum_discovers_enum_in_growing_type_argument() -> None:
    _, _, compiled = _compile(
        "enum Box[T]\n"
        "  | box(value: T)\n"
        "enum Perfect[T]\n"
        "  | end\n"
        "  | value(item: T)\n"
        "  | next(value: Perfect[Box[T]])\n"
        "let subject: Perfect[int] = Perfect::end\n"
        "case subject of\n"
        "  | Perfect::end => 0\n"
        "  | Perfect::value(item = _) => 1\n"
        "  | Perfect::next(value = Perfect::end) => 2\n"
        "  | Perfect::next(\n"
        "      value = Perfect::value(item = Box::box(value = 0))\n"
        "    ) => 3\n"
    )
    issue = cast(NonExhaustiveIssue, compiled.issues[0])

    assert "box(value = a int value other than 0)" in render_witness(issue.witness)


def test_source_owner_candidates_are_finite_over_visible_declarations() -> None:
    _, _, compiled = _compile(
        "enum Box[T]\n"
        "  | box(value: T)\n"
        "record Holder[T]\n"
        "  value: T\n"
        "enum Carrier[T]\n"
        "  | carrier\n"
        "let subject: Carrier[Holder[list[dict[text, (Box[int]) -> Box[int]]]]] = "
        "Carrier::carrier\n"
        "case subject of | _ => 0\n"
    )

    owners = {form.owner_name for form in compiled.normalized.case_context.enum_owner_forms}

    assert {"Box", "Carrier", "Holder"} <= owners


def test_whole_boolean_domain_uses_first_signature_constructor() -> None:
    case, compiled = _compile_without_normalized_rows(
        "let value = false\ncase value of | true => 1"
    )

    assert [type(issue) for issue in compiled.issues] == [
        NonExhaustiveIssue,
        RedundantArmIssue,
    ]
    assert cast(RedundantArmIssue, compiled.issues[1]).action_id == case.branches[0].node_id
    witness = cast(NonExhaustiveIssue, compiled.issues[0]).witness
    assert witness == BoolWitness(False)
    assert render_witness(witness) == "false"


@pytest.mark.parametrize(
    ("source", "rendered"),
    [
        ("let value: int = 1\ncase value of | 1.5 => 0", "a int value"),
        ("let value: decimal = 1.5\ncase value of | 1.5 => 0", "a decimal value"),
        ('let value = "x"\ncase value of | "x" => 0', "a text value"),
        ("let value: json = null\ncase value of | null => 0", "a json value"),
    ],
)
def test_whole_open_domain_uses_empty_exclusion_complement(source: str, rendered: str) -> None:
    case, compiled = _compile_without_normalized_rows(source)

    assert [type(issue) for issue in compiled.issues] == [
        NonExhaustiveIssue,
        RedundantArmIssue,
    ]
    assert cast(RedundantArmIssue, compiled.issues[1]).action_id == case.branches[0].node_id
    witness = cast(NonExhaustiveIssue, compiled.issues[0]).witness
    assert witness == OpenComplementWitness(compiled.normalized.root.type, ())
    assert render_witness(witness) == rendered


def test_nested_failure_path_can_contain_a_concrete_literal_witness() -> None:
    _, _, compiled = _compile(
        "enum Packet\n"
        "  | packet(code: int, flag: bool)\n"
        "let value = packet(code = 1, flag = false)\n"
        "case value of | packet(code = 1, flag = false) => 0"
    )
    issue = cast(NonExhaustiveIssue, compiled.issues[0])
    witness = cast(EnumWitness, issue.witness)

    assert witness.fields[0].witness == LiteralWitness(LiteralKind.NUMERIC, decimal.Decimal("1"))
    assert "1" in render_witness(witness)


def test_witness_renderer_covers_atomic_and_empty_complement_forms() -> None:
    _, _, compiled = _compile("let value: json = null\ncase value of | null => 0")
    issue = cast(NonExhaustiveIssue, compiled.issues[0])
    assert "null" in render_witness(issue.witness)
    assert render_witness(WildcardWitness()) == "_"
    assert render_witness(BoolWitness(False)) == "false"
    assert render_witness(LiteralWitness(LiteralKind.TEXT, "x")) == '"x"'
    assert render_witness(LiteralWitness(LiteralKind.TEXT, "\x1b")) == '"\\u001b"'
    assert render_witness(LiteralWitness(LiteralKind.TEXT, "${name}")) == '"\\${name}"'
    assert render_witness(LiteralWitness(LiteralKind.TEXT, "\\${name}")) == '"\\\\\\${name}"'
    assert (
        render_witness(LiteralWitness(LiteralKind.NUMERIC, decimal.Decimal("1E-7"))) == "0.0000001"
    )
    empty_enum = EnumWitness(EnumType("Empty"), "empty", ())
    assert render_witness(empty_enum) == "empty"
    synthetic_qualified = EnumWitness(
        EnumType("Empty", module_id=PRELUDE_ID),
        "empty",
        (),
        EnumWitnessQualification("Empty", None),
    )
    assert render_witness(synthetic_qualified) == "Empty::empty"
    assert "\x00" not in render_witness(synthetic_qualified)
    self_qualified = replace(
        synthetic_qualified,
        qualification=EnumWitnessQualification("Empty", ()),
    )
    assert render_witness(self_qualified) == "::Empty::empty"
    empty_complement = OpenComplementWitness(IntType(), ())
    assert render_witness(empty_complement) == "a int value"


def test_leaf_free_interface_deduplicates_an_occurrence() -> None:
    _, _, compiled = _compile("let value = 1\ncase value of | captured => captured")
    leaf = cast(DecisionLeaf, compiled.root)
    first = leaf.binder_assignments[0]
    duplicate_occurrence = replace(
        first,
        binder=replace(first.binder, node_id=first.binder.node_id + 1),
    )

    assert DecisionLeaf(leaf.action_id, (first, duplicate_occurrence)).free_occurrences == (
        first.occurrence,
    )


def test_decision_interning_does_not_recursively_hash_shared_children() -> None:
    _, _, compiled = _compile("let value = false\ncase value of | false => 0 | true => 1")
    root = cast(DecisionSwitch, compiled.root)
    compiler = compiler_module._CaseCompiler()
    decision: Decision = DecisionFail()
    for _ in range(2_000):
        decision = DecisionSwitch(
            root.occurrence,
            (DecisionBranch(BoolConstructor(False), decision),),
            None,
        )

    assert compiler.intern(decision) is decision


def test_private_compiler_guards_reject_malformed_internal_states() -> None:
    _, _, compiled = _compile("let value = false\ncase value of | false => 0 | true => 1")
    normalized = compiled.normalized
    matrix = matrix_from_normalized(normalized)
    allocator = OccurrenceAllocator.for_case(normalized)
    case_compiler = compiler_module._CaseCompiler()
    first_root, evolved = case_compiler.compile(matrix, allocator)
    second_root, same_allocator = case_compiler.compile(matrix, evolved)
    assert second_root is first_root
    assert same_allocator is evolved

    with pytest.raises(MatchCompileInvariantError, match="absent"):
        compiler_module._constructor_index(
            BoolConstructor(False), ClosedSignature((BoolConstructor(True),))
        )

    refutable_row = matrix.rows[0]
    assert isinstance(refutable_row.cells[0], ConstructorCell)
    with pytest.raises(MatchCompileInvariantError, match="irrefutable"):
        compiler_module._finalize_binders(matrix, refutable_row)

    binder_checked, _, binder_compiled = _compile(
        "let value = 1\ncase value of | captured => captured"
    )
    del binder_checked
    binder_matrix = matrix_from_normalized(binder_compiled.normalized)
    binder_row = binder_matrix.rows[0]
    binder_cell = cast(WildcardCell, binder_row.cells[0])
    assert binder_cell.binder is not None
    unavailable = replace(
        binder_row,
        cells=(replace(binder_cell, binder=None),),
        binder_assignments=(BinderAssignment(OccurrenceId(999), binder_cell.binder),),
    )
    with pytest.raises(MatchCompileInvariantError, match="unavailable"):
        compiler_module._finalize_binders(binder_matrix, unavailable)
    duplicate = replace(
        binder_row,
        binder_assignments=(
            BinderAssignment(binder_compiled.normalized.root.id, binder_cell.binder),
        ),
    )
    with pytest.raises(MatchCompileInvariantError, match="more than once"):
        compiler_module._finalize_binders(binder_matrix, duplicate)

    unknown_leaf = DecisionLeaf(
        1,
        (BinderAssignment(OccurrenceId(999), binder_cell.binder),),
    )
    with pytest.raises(MatchCompileInvariantError, match="unknown occurrence"):
        compiler_module._switch_free_occurrences(
            normalized.root,
            (DecisionBranch(BoolConstructor(False), unknown_leaf),),
            None,
            OccurrenceIndex.for_occurrences(normalized.occurrences),
        )

    _, _, enum_compiled = _compile(
        "enum Choice\n  | empty\n  | item(value: int)\n"
        "let value: Choice = Choice::empty\n"
        "case value of | Choice::empty => 0\n"
    )
    missing_spellings = replace(
        enum_compiled.normalized,
        case_context=MatchCaseContext(ENTRY_ID),
    )
    issue = cast(NonExhaustiveIssue, compile_case(missing_spellings).issues[0])

    assert issue.witness == WildcardWitness()


def test_strong_compiled_case_validator_rejects_internal_corruption() -> None:
    _, _, pair_compiled = _compile(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "let value = pair(left = false, right = false)\n"
        "case value of | pair(left = false, right = false) => 1 | _ => 2"
    )
    normalized = pair_compiled.normalized
    root_occurrence, left_occurrence, right_occurrence = pair_compiled.occurrences
    left_provenance = cast(FieldOccurrenceProvenance, left_occurrence.provenance)
    right_provenance = cast(FieldOccurrenceProvenance, right_occurrence.provenance)
    pair_constructor = left_provenance.constructor

    ledger_corruptions = (
        (
            root_occurrence,
            replace(left_occurrence, id=OccurrenceId(9)),
            right_occurrence,
        ),
        (
            root_occurrence,
            replace(left_occurrence, provenance=root_occurrence.provenance),
            right_occurrence,
        ),
        (
            root_occurrence,
            replace(
                left_occurrence,
                provenance=replace(left_provenance, parent=OccurrenceId(999)),
            ),
            right_occurrence,
        ),
        (
            root_occurrence,
            replace(
                left_occurrence,
                provenance=replace(
                    left_provenance,
                    constructor=BoolConstructor(False),
                ),
            ),
            right_occurrence,
        ),
        (
            root_occurrence,
            replace(
                left_occurrence,
                provenance=replace(left_provenance, field_index=99),
            ),
            right_occurrence,
        ),
        (
            root_occurrence,
            replace(
                left_occurrence,
                provenance=replace(left_provenance, field_name="wrong"),
            ),
            right_occurrence,
        ),
        (
            root_occurrence,
            left_occurrence,
            replace(
                right_occurrence,
                type=left_occurrence.type,
                provenance=replace(
                    right_provenance,
                    field_index=0,
                    field_name=left_provenance.field_name,
                ),
            ),
        ),
        (root_occurrence, left_occurrence),
    )
    for occurrences in ledger_corruptions:
        with pytest.raises(MatchCompileInvariantError):
            compiler_module._validate_occurrence_ledger(normalized, occurrences)

    open_root = replace(root_occurrence, type=IntType())
    open_normalized = replace(normalized, root=open_root, occurrences=(open_root,))
    with pytest.raises(MatchCompileInvariantError, match="open signature"):
        compiler_module._validate_occurrence_ledger(
            open_normalized,
            (open_root, left_occurrence),
        )

    with pytest.raises(MatchCompileInvariantError, match="incompatible"):
        compiler_module._canonical_switch_constructor(
            BoolConstructor(False),
            root_occurrence,
            normalized.type_table,
        )
    foreign_constructor = replace(pair_constructor, variant="foreign", fields=())
    with pytest.raises(MatchCompileInvariantError, match="absent"):
        compiler_module._canonical_switch_constructor(
            foreign_constructor,
            root_occurrence,
            normalized.type_table,
        )

    _, _, binder_compiled = _compile("let value = 1\ncase value of | captured => captured")
    binder_leaf = cast(DecisionLeaf, binder_compiled.root)
    binder_assignment = binder_leaf.binder_assignments[0]
    binder_ledger, binder_groups = compiler_module._validate_occurrence_ledger(
        binder_compiled.normalized,
        binder_compiled.occurrences,
    )
    leaf_corruptions = (
        DecisionLeaf(
            binder_leaf.action_id,
            (replace(binder_assignment, occurrence=OccurrenceId(999)),),
        ),
        DecisionLeaf(
            binder_leaf.action_id,
            (binder_assignment, binder_assignment),
        ),
        DecisionLeaf(
            binder_leaf.action_id,
            (
                replace(
                    binder_assignment,
                    binder=replace(
                        binder_assignment.binder,
                        node_id=binder_assignment.binder.node_id + 1,
                    ),
                ),
            ),
        ),
        DecisionLeaf(binder_leaf.action_id, ()),
    )
    for leaf in leaf_corruptions:
        with pytest.raises(MatchCompileInvariantError):
            compiler_module._validate_compiled_decisions(
                replace(binder_compiled, root=leaf),
                binder_ledger,
                binder_groups,
            )

    _, _, nested_binder_compiled = _compile(
        "enum Box\n"
        "  | box(value: int)\n"
        "  | empty\n"
        "let value = box(value = 1)\n"
        "case value of | box(value = captured) => captured"
    )
    nested_root = cast(DecisionSwitch, nested_binder_compiled.root)
    nested_leaf = cast(DecisionLeaf, nested_root.keyed_children[0].decision)
    nested_assignment = nested_leaf.binder_assignments[0]
    nested_ledger, nested_groups = compiler_module._validate_occurrence_ledger(
        nested_binder_compiled.normalized,
        nested_binder_compiled.occurrences,
    )
    wrong_target = DecisionLeaf(
        nested_leaf.action_id,
        (
            replace(
                nested_assignment,
                occurrence=nested_binder_compiled.normalized.root.id,
            ),
        ),
    )
    with pytest.raises(MatchCompileInvariantError, match="incompatible occurrence"):
        compiler_module._validate_compiled_decisions(
            replace(nested_binder_compiled, root=wrong_target),
            nested_ledger,
            nested_groups,
        )

    _, _, two_binder_compiled = _compile(
        "enum Pair\n"
        "  | pair(left: int, right: int)\n"
        "let value = pair(left = 1, right = 2)\n"
        "case value of | pair(left = left, right = right) => left"
    )
    two_root = cast(DecisionSwitch, two_binder_compiled.root)
    two_leaf = cast(DecisionLeaf, two_root.keyed_children[0].decision)
    two_ledger, two_groups = compiler_module._validate_occurrence_ledger(
        two_binder_compiled.normalized,
        two_binder_compiled.occurrences,
    )
    reversed_leaf = replace(
        two_leaf,
        binder_assignments=tuple(reversed(two_leaf.binder_assignments)),
    )
    with pytest.raises(MatchCompileInvariantError, match="stable occurrence order"):
        compiler_module._validate_compiled_decisions(
            replace(two_binder_compiled, root=reversed_leaf),
            two_ledger,
            two_groups,
        )

    bool_compiled = _compile("case true of | true => 1 | false => 2")[2]
    bool_root = cast(DecisionSwitch, bool_compiled.root)
    forged_occurrence = replace(bool_root.occurrence)
    switch_corruptions = (
        replace(bool_root, occurrence=forged_occurrence),
        replace(bool_root, keyed_children=tuple(reversed(bool_root.keyed_children))),
        replace(bool_root, keyed_children=bool_root.keyed_children[:1]),
    )
    bool_ledger, bool_groups = compiler_module._validate_occurrence_ledger(
        bool_compiled.normalized,
        bool_compiled.occurrences,
    )
    for switch in switch_corruptions:
        with pytest.raises(MatchCompileInvariantError):
            compiler_module._validate_compiled_decisions(
                replace(bool_compiled, root=switch),
                bool_ledger,
                bool_groups,
            )

    open_compiled = _compile("case 1 of | 1 => 1 | _ => 2")[2]
    open_switch = cast(DecisionSwitch, open_compiled.root)
    open_ledger, open_groups = compiler_module._validate_occurrence_ledger(
        open_compiled.normalized,
        open_compiled.occurrences,
    )
    with pytest.raises(MatchCompileInvariantError, match="retain a default"):
        compiler_module._validate_compiled_decisions(
            replace(open_compiled, root=replace(open_switch, default=None)),
            open_ledger,
            open_groups,
        )

    pair_ledger, pair_groups = compiler_module._validate_occurrence_ledger(
        normalized,
        pair_compiled.occurrences,
    )
    with pytest.raises(MatchCompileInvariantError, match="field group"):
        compiler_module._validate_compiled_decisions(pair_compiled, pair_ledger, {})
    nested_switch = cast(
        DecisionSwitch,
        cast(DecisionSwitch, pair_compiled.root).keyed_children[0].decision,
    )
    forged_nested = replace(
        nested_switch,
        free_occurrences=(
            nested_switch.occurrence.id,
            normalized.root.id,
        ),
    )
    forged_root = replace(
        cast(DecisionSwitch, pair_compiled.root),
        keyed_children=(
            replace(
                cast(DecisionSwitch, pair_compiled.root).keyed_children[0],
                decision=forged_nested,
            ),
        ),
    )
    with pytest.raises(MatchCompileInvariantError, match="forged free"):
        compiler_module._validate_compiled_decisions(
            replace(pair_compiled, root=forged_root),
            pair_ledger,
            pair_groups,
        )
    extra_groups = dict(pair_groups)
    extra_groups[(left_occurrence.id, pair_constructor)] = ()
    with pytest.raises(MatchCompileInvariantError, match="exactly match"):
        compiler_module._validate_compiled_decisions(
            pair_compiled,
            pair_ledger,
            extra_groups,
        )

    unavailable_root = replace(
        nested_root,
        default=nested_leaf,
        free_occurrences=(
            nested_binder_compiled.normalized.root.id,
            nested_assignment.occurrence,
        ),
    )
    with pytest.raises(MatchCompileInvariantError, match="unavailable"):
        compiler_module._validate_compiled_decisions(
            replace(nested_binder_compiled, root=unavailable_root),
            nested_ledger,
            nested_groups,
        )

    with pytest.raises(MatchCompileInvariantError, match="normalized source"):
        compiler_module.validate_compiled_case(
            pair_compiled,
            expected_normalized=replace(normalized, case_node_id=normalized.case_node_id + 1),
        )
    with pytest.raises(MatchCompileInvariantError, match="type context"):
        compiler_module.validate_compiled_case(
            pair_compiled,
            expected_normalized=replace(normalized, type_table=TypeTable()),
        )
    with pytest.raises(MatchCompileInvariantError, match="source context"):
        compiler_module.validate_compiled_case(
            pair_compiled,
            expected_normalized=replace(
                normalized,
                case_context=replace(normalized.case_context, module_id=PRELUDE_ID),
            ),
        )


def test_source_spelling_model_rejects_inconsistent_structures() -> None:
    with pytest.raises(ValueError, match="bare constructor"):
        EnumConstructorSpelling(None, ("module",))
    with pytest.raises(ValueError, match="type-qualified"):
        EnumConstructorSpelling("Choice", None, bare=True)
    with pytest.raises(ValueError, match="import handle"):
        EnumOwnerForm(
            "Choice",
            None,
            kind=EnumOwnerFormKind.QUALIFIED_IMPORT,
            type_template=TypeTemplate(EnumType("Choice")),
        )
    with pytest.raises(ValueError, match="unqualified enum owner"):
        EnumOwnerForm(
            "Choice",
            ("module",),
            kind=EnumOwnerFormKind.OPEN_IMPORT,
            type_template=TypeTemplate(EnumType("Choice")),
        )
    with pytest.raises(ValueError, match="self-qualified"):
        EnumOwnerForm(
            "Choice",
            None,
            kind=EnumOwnerFormKind.SELF,
            type_template=TypeTemplate(EnumType("Choice")),
        )
    assert EnumOwnerForm("Choice", None).match(EnumType("Choice")) is None
    assert EnumOwnerForm("Choice", ("module",)).kind is EnumOwnerFormKind.QUALIFIED_IMPORT
    assert EnumOwnerForm("Choice", ()).kind is EnumOwnerFormKind.SELF


def test_normalization_requires_case_scope_provenance() -> None:
    checked, case, _ = _compile(
        "enum Choice\n  | empty\nlet value: Choice = Choice::empty\ncase value of | _ => 0\n"
    )
    without_scope = replace(
        checked,
        resolved=replace(checked.resolved, case_scopes={}),
    )
    with pytest.raises(MatchCompileInvariantError, match="scope provenance"):
        normalize_case(case, without_scope)


def test_private_diagnostic_guards_reject_malformed_switches() -> None:
    _, _, compiled = _compile("let value = false\ncase value of | false => 0 | true => 1")
    normalized = compiled.normalized
    fail = DecisionFail()
    leaf = DecisionLeaf(normalized.actions[0].action_id, ())
    false_branch = DecisionBranch(BoolConstructor(False), fail)
    true_branch = DecisionBranch(BoolConstructor(True), leaf)
    complete_with_default = DecisionSwitch(
        normalized.root,
        (false_branch, true_branch),
        fail,
        (normalized.root.id,),
    )
    with pytest.raises(MatchCompileInvariantError, match="complete closed"):
        compiler_module._default_constraint(complete_with_default, normalized.type_table)

    int_occurrence = replace(normalized.root, type=IntType())
    malformed_open = DecisionSwitch(
        int_occurrence,
        (false_branch,),
        fail,
        (int_occurrence.id,),
    )
    with pytest.raises(MatchCompileInvariantError, match="non-literal"):
        compiler_module._default_constraint(malformed_open, normalized.type_table)

    repeated = DecisionSwitch(
        normalized.root,
        (DecisionBranch(BoolConstructor(False), fail),),
        None,
        (normalized.root.id,),
    )
    repeated_outer = DecisionSwitch(
        normalized.root,
        (DecisionBranch(BoolConstructor(False), repeated),),
        None,
        (normalized.root.id,),
    )
    with pytest.raises(MatchCompileInvariantError, match="more than once"):
        compiler_module._issues(normalized, repeated_outer, normalized.occurrences)


def test_validator_rejects_each_malformed_switch_shape_and_cycles() -> None:
    _, _, compiled = _compile("let value = false\ncase value of | false => 0 | true => 1")
    root = cast(DecisionSwitch, compiled.root)
    occurrence = root.occurrence
    leaf = root.keyed_children[0].decision
    false_branch = DecisionBranch(BoolConstructor(False), leaf)

    malformed = (
        (
            DecisionSwitch(occurrence, (), None, (occurrence.id,)),
            "non-empty",
        ),
        (
            DecisionSwitch(
                occurrence,
                (false_branch, false_branch),
                None,
                (occurrence.id,),
            ),
            "unique",
        ),
        (
            DecisionSwitch(
                occurrence,
                (false_branch,),
                None,
                (occurrence.id, occurrence.id),
            ),
            "duplicates",
        ),
        (DecisionSwitch(occurrence, (false_branch,), None, ()), "omits"),
    )
    for decision, message in malformed:
        with pytest.raises(MatchCompileInvariantError, match=message):
            validate_decision_dag(decision)

    cyclic = DecisionSwitch(
        occurrence,
        (false_branch,),
        None,
        (occurrence.id,),
    )
    object.__setattr__(
        cyclic,
        "keyed_children",
        (DecisionBranch(BoolConstructor(False), cyclic),),
    )
    with pytest.raises(MatchCompileInvariantError, match="cycle"):
        validate_decision_dag(cyclic)
    with pytest.raises(MatchCompileInvariantError, match="cycle"):
        compiler_module._first_failure_constraints(cyclic, compiled.normalized.type_table)
