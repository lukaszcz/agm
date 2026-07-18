"""Whole-program match-compilation artifact and diagnostic contracts."""

from __future__ import annotations

import decimal
from collections.abc import Callable, MutableMapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import agm.agl.matchcompile as matchcompile
import agm.agl.matchcompile.compiler as compiler_module
import agm.agl.matchcompile.stage as stage_module
from agm.agl.matchcompile import (
    MatchCompilationResult,
    MatchCompiledModule,
    MatchCompiledProgram,
    MatchIssue,
    NonExhaustiveIssue,
    RedundantArmIssue,
    compile_module_matches,
    compile_program_matches,
    diagnostic_from_match_issue,
    diagnostics_from_match_issues,
)
from agm.agl.matchcompile.compiler import (
    CompiledCase,
    compile_case,
    validate_compiled_case,
)
from agm.agl.matchcompile.diagnostics import issue_sort_key
from agm.agl.matchcompile.matrix import OccurrenceAllocator, PatternMatrix, Specialization
from agm.agl.matchcompile.model import (
    Constructor,
    DecisionBranch,
    DecisionFail,
    DecisionLeaf,
    DecisionSwitch,
    LiteralConstructor,
    NormalizedCase,
)
from agm.agl.matchcompile.normalize import MatchCompileInvariantError, normalize_case
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.pipeline import (
    PipelineDriver,
    PreparedProgram,
    _run_matchcompile_program,
)
from agm.agl.scope import resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.syntax.nodes import Case
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import EnumOwnerForm, check_module
from agm.agl.typecheck.env import CheckedModule
from agm.agl.typecheck.program import CheckedProgram, check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files


def test_matchcompile_public_exports_are_narrow_and_stable() -> None:
    assert set(matchcompile.__all__) == {
        "BoolConstructor",
        "BoolWitness",
        "CompiledCase",
        "Constructor",
        "Decision",
        "DecisionLeaf",
        "DecisionSwitch",
        "EnumConstructor",
        "EnumWitness",
        "EnumWitnessQualification",
        "FieldOccurrenceProvenance",
        "LiteralKind",
        "LiteralWitness",
        "MatchCompilationResult",
        "MatchCompiledArtifact",
        "MatchCompiledProgram",
        "MatchCompiledModule",
        "MatchIssue",
        "MatchWitness",
        "NonExhaustiveIssue",
        "Occurrence",
        "OccurrenceId",
        "OpenComplementWitness",
        "RedundantArmIssue",
        "WildcardWitness",
        "WitnessField",
        "compile_program_matches",
        "compile_module_matches",
        "diagnostic_from_match_issue",
        "diagnostics_from_match_issues",
        "render_witness",
        "validate_match_compiled_program",
        "validate_match_compiled_module",
    }
    assert not hasattr(matchcompile, "EnumOwnerForm")
    assert not hasattr(matchcompile, "EnumOwnerFormKind")
    assert hasattr(matchcompile, "FieldOccurrenceProvenance")
    assert matchcompile.EnumWitnessQualification is not EnumOwnerForm


def _checked(source: str) -> CheckedModule:
    return check_module(resolve_module(parse_program(source)), base_caps())


def _compiled(source: str) -> MatchCompiledModule:
    result = compile_module_matches(_checked(source))
    assert isinstance(result.compiled, MatchCompiledModule)
    return result.compiled


def _prepared_program(source: str, *, roots: frozenset[Path] = frozenset()) -> PreparedProgram:
    return PipelineDriver.prepare_program(
        source,
        entry_path=None,
        roots=RootSet(roots=roots),
        default_stdlib=False,
    )


def _first_case(checked: CheckedModule | CheckedModule) -> Case:
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(checked.resolved.program, collect)
    assert cases
    return cases[0]


def _semantic_replay_corruptions(
    cases: tuple[CompiledCase, ...],
) -> tuple[tuple[str, int, CompiledCase], ...]:
    bool_case, open_case, pair_case = cases
    bool_root = cast(DecisionSwitch, bool_case.root)
    open_root = cast(DecisionSwitch, open_case.root)
    pair_root = cast(DecisionSwitch, pair_case.root)
    pair_switch = cast(DecisionSwitch, pair_root.keyed_children[0].decision)
    pair_right = pair_case.occurrences[2]
    open_branch = open_root.keyed_children[0]

    return (
        (
            "swapped keyed leaves",
            0,
            replace(
                bool_case,
                root=replace(
                    bool_root,
                    keyed_children=(
                        replace(
                            bool_root.keyed_children[0],
                            decision=bool_root.keyed_children[1].decision,
                        ),
                        replace(
                            bool_root.keyed_children[1],
                            decision=bool_root.keyed_children[0].decision,
                        ),
                    ),
                ),
            ),
        ),
        (
            "swapped binder branch and default leaves",
            1,
            replace(
                open_case,
                root=replace(
                    open_root,
                    keyed_children=(
                        replace(open_branch, decision=cast(DecisionLeaf, open_root.default)),
                    ),
                    default=open_branch.decision,
                ),
            ),
        ),
        (
            "wrong observed literal head",
            1,
            replace(
                open_case,
                root=replace(
                    open_root,
                    keyed_children=(
                        replace(
                            open_branch,
                            constructor=replace(
                                cast(LiteralConstructor, open_branch.constructor),
                                value=decimal.Decimal(2),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        (
            "swapped binder-bearing keyed leaves",
            2,
            replace(
                pair_case,
                root=replace(
                    pair_root,
                    keyed_children=(
                        replace(
                            pair_root.keyed_children[0],
                            decision=replace(
                                pair_switch,
                                keyed_children=(
                                    replace(
                                        pair_switch.keyed_children[0],
                                        decision=pair_switch.keyed_children[1].decision,
                                    ),
                                    replace(
                                        pair_switch.keyed_children[1],
                                        decision=pair_switch.keyed_children[0].decision,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        (
            "wrong qba-selected occurrence",
            2,
            replace(
                pair_case,
                root=replace(
                    pair_root,
                    keyed_children=(
                        replace(
                            pair_root.keyed_children[0],
                            decision=replace(
                                pair_switch,
                                occurrence=pair_right,
                                free_occurrences=(pair_right.id,),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


_SEMANTIC_REPLAY_SOURCE = (
    "enum Pair\n"
    "  | pair(left: bool, right: bool)\n"
    "let pair_value = pair(left = false, right = true)\n"
    "case true of | true => 1 | false => 2\n"
    "case 1 of | 1 => 3 | captured => captured\n"
    "case pair_value of\n"
    "  | pair(left = true, right = when_true) => when_true\n"
    "  | pair(left = false, right = when_false) => when_false\n"
)


def test_compiles_every_nested_case_once_into_immutable_total_mapping() -> None:
    compiled = _compiled(
        "case true of\n"
        "  | true =>\n"
        "      case false of\n"
        "        | true => 1\n"
        "        | false => 2\n"
        "  | false => 3\n"
    )

    assert len(compiled.cases) == 2
    assert set(compiled.cases) == {case.case_node_id for case in compiled.cases.values()}
    mutable_view = cast(MutableMapping[int, CompiledCase], compiled.cases)
    with pytest.raises(TypeError):
        mutable_view[999] = next(iter(compiled.cases.values()))


def test_empty_case_program_produces_valid_empty_artifact() -> None:
    compiled = _compiled("let x = 1\nx")
    assert compiled.cases == {}


def test_source_issues_are_all_sorted_adapted_and_prevent_artifact() -> None:
    checked = _checked(
        "case true of\n  | true => 1\n  | true => 2\ncase false of\n  | false => 3\n"
    )
    result = compile_module_matches(checked)

    assert result.compiled is None
    assert len(result.issues) == 3
    diagnostics = diagnostics_from_match_issues(result.issues)
    assert [diagnostic.line for diagnostic in diagnostics] == sorted(
        diagnostic.line for diagnostic in diagnostics
    )
    assert all(diagnostic.severity == "error" for diagnostic in diagnostics)
    assert any("false" in diagnostic.message for diagnostic in diagnostics)
    assert any("Redundant" in diagnostic.message for diagnostic in diagnostics)

    with pytest.raises(AssertionError, match="unsupported"):
        diagnostic_from_match_issue(cast(MatchIssue, object()))


def test_program_match_compilation_orders_issues_by_source_location() -> None:
    checked = _checked(
        "case true of\n  | true => 1\n  | true => 2\ncase false of\n  | false => 3\n"
    )

    result = compile_module_matches(checked)

    assert len(result.issues) > 1
    assert list(result.issues) == sorted(result.issues, key=issue_sort_key)


def test_match_compilation_yields_an_artifact_or_issues_but_never_both(tmp_path: Path) -> None:
    accepted = compile_module_matches(_checked("case true of | true => 1 | false => 2"))
    assert accepted.compiled is not None
    assert accepted.issues == ()

    rejected = compile_module_matches(_checked("case true of | true => 1"))
    assert rejected.compiled is None
    assert rejected.issues != ()

    accepted_graph = compile_program_matches(
        check_program(
            resolve_program(
                make_graph_from_files(tmp_path / "ok", {"entry": "case true of | _ => 1"})
            ),
            base_caps(),
        )
    )
    assert accepted_graph.compiled is not None
    assert accepted_graph.issues == ()

    rejected_graph = compile_program_matches(
        check_program(
            resolve_program(
                make_graph_from_files(tmp_path / "bad", {"entry": "case true of | true => 1"})
            ),
            base_caps(),
        )
    )
    assert rejected_graph.compiled is None
    assert rejected_graph.issues != ()


def test_rejected_match_compilation_still_validates_the_cases_it_discards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected cases never reach an artifact, so the stage result is their only oracle."""
    real_compile_case = stage_module.compile_case

    def corrupting_compile_case(normalized: object) -> CompiledCase:
        compiled = real_compile_case(cast(NormalizedCase, normalized))
        if not compiled.issues:
            return compiled
        return replace(compiled, reachable_action_ids=())

    monkeypatch.setattr(stage_module, "compile_case", corrupting_compile_case)

    with pytest.raises(MatchCompileInvariantError, match="reachable action ids"):
        compile_module_matches(_checked("case true of | true => 1"))


def test_graph_issues_are_aggregated_and_sorted_across_module_sources(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import beta\nimport alpha\n()",
            "alpha": (
                "def alpha(x: bool) -> int =\n  case x of\n    | true => 1\n    | true => 2\n"
            ),
            "beta": ("def beta(x: bool) -> int =\n  case x of\n    | true => 1\n    | true => 2\n"),
        },
    )
    checked = check_program(resolve_program(graph), base_caps())

    result = compile_program_matches(checked)

    assert result.compiled is None
    assert [type(issue) for issue in result.issues] == [
        NonExhaustiveIssue,
        RedundantArmIssue,
        NonExhaustiveIssue,
        RedundantArmIssue,
    ]
    assert [Path(issue.span.source.label).name for issue in result.issues] == [
        "alpha.agl",
        "alpha.agl",
        "beta.agl",
        "beta.agl",
    ]
    assert [(issue.span.start_line, issue.span.start_col) for issue in result.issues] == [
        (2, 3),
        (4, 7),
        (2, 3),
        (4, 7),
    ]
    assert list(result.issues) == sorted(result.issues, key=issue_sort_key)

    diagnostics = diagnostics_from_match_issues(result.issues)
    assert all(diagnostic.severity == "error" for diagnostic in diagnostics)
    assert [
        (
            Path(diagnostic.source_label).name if diagnostic.source_label is not None else None,
            diagnostic.line,
            diagnostic.column,
        )
        for diagnostic in diagnostics
    ] == [
        ("alpha.agl", 2, 3),
        ("alpha.agl", 4, 7),
        ("beta.agl", 2, 3),
        ("beta.agl", 4, 7),
    ]


def test_program_artifact_rejects_missing_extra_mismatched_and_cross_program_cases() -> None:
    first = _compiled("case true of | true => 1 | false => 2")
    case_id, compiled_case = next(iter(first.cases.items()))

    with pytest.raises(MatchCompileInvariantError, match="missing"):
        MatchCompiledModule(first.checked, {})
    with pytest.raises(MatchCompileInvariantError, match="extra"):
        MatchCompiledModule(first.checked, {case_id: compiled_case, 999: compiled_case})

    mismatched = replace(
        compiled_case,
        normalized=replace(compiled_case.normalized, case_node_id=case_id + 1),
    )
    with pytest.raises(MatchCompileInvariantError, match="does not match"):
        MatchCompiledModule(first.checked, {case_id: mismatched})

    wrong_span = replace(
        compiled_case,
        normalized=replace(
            compiled_case.normalized,
            span=replace(
                compiled_case.normalized.span,
                end_col=compiled_case.normalized.span.end_col + 1,
            ),
        ),
    )
    with pytest.raises(MatchCompileInvariantError, match="source provenance"):
        MatchCompiledModule(first.checked, {case_id: wrong_span})

    second_checked = _checked("case true of | true => 1 | false => 2")
    with pytest.raises(MatchCompileInvariantError, match="different checked program"):
        MatchCompiledModule(second_checked, first.cases)

    invalid_checked = _checked("case true of | true => 1")
    invalid_case = _first_case(invalid_checked)
    invalid_compiled = compile_case(normalize_case(invalid_case, invalid_checked))
    with pytest.raises(MatchCompileInvariantError, match="contains issues"):
        MatchCompiledModule(invalid_checked, {invalid_case.node_id: invalid_compiled})


def test_program_artifact_rejects_compiled_case_semantic_corruption() -> None:
    compiled = _compiled(
        "case true of | true => 1 | false => 2\ncase false of | true => 3 | false => 4"
    )
    first_id, second_id = tuple(compiled.cases)
    first = compiled.cases[first_id]
    second = compiled.cases[second_id]
    first_root = cast(DecisionSwitch, first.root)

    corruptions = (
        replace(first, root=second.root, occurrences=second.occurrences),
        replace(
            first,
            root=DecisionLeaf(999, ()),
            reachable_action_ids=(999,),
        ),
        replace(first, reachable_action_ids=tuple(reversed(first.reachable_action_ids))),
        replace(first, reachable_action_ids=()),
        replace(
            first,
            root=replace(
                first_root,
                keyed_children=(
                    DecisionBranch(
                        first_root.keyed_children[0].constructor,
                        DecisionFail(),
                    ),
                    *first_root.keyed_children[1:],
                ),
            ),
        ),
    )
    for corrupted in corruptions:
        cases = dict(compiled.cases)
        cases[first_id] = corrupted
        with pytest.raises(MatchCompileInvariantError):
            MatchCompiledModule(compiled.checked, cases)

    invalid_checked = _checked("case true of | true => 1")
    invalid_case = _first_case(invalid_checked)
    invalid_compiled = compile_case(normalize_case(invalid_case, invalid_checked))
    stripped = replace(invalid_compiled, issues=())
    with pytest.raises(MatchCompileInvariantError):
        MatchCompiledModule(invalid_checked, {invalid_case.node_id: stripped})


def test_program_artifact_replays_source_semantics_against_every_decision_edge() -> None:
    compiled = _compiled(_SEMANTIC_REPLAY_SOURCE)
    case_ids = tuple(compiled.cases)
    cases = tuple(compiled.cases.values())

    MatchCompiledModule(compiled.checked, compiled.cases)
    for _, case_index, corrupted_case in _semantic_replay_corruptions(cases):
        corrupted_cases = dict(compiled.cases)
        corrupted_cases[case_ids[case_index]] = corrupted_case
        with pytest.raises(MatchCompileInvariantError, match="semantic replay"):
            MatchCompiledModule(compiled.checked, corrupted_cases)


def test_semantic_replay_rejects_terminal_and_ledger_rule_corruption() -> None:
    compiled = _compiled(_SEMANTIC_REPLAY_SOURCE)
    bool_case, _, pair_case = tuple(compiled.cases.values())
    bool_root = cast(DecisionSwitch, bool_case.root)
    leaf = cast(DecisionLeaf, bool_root.keyed_children[0].decision)
    empty_case = compile_case(replace(bool_case.normalized, rows=()))

    corruptions = (
        replace(empty_case, root=leaf),
        replace(bool_case, root=leaf),
        replace(bool_case, root=replace(bool_root, default=leaf)),
        replace(pair_case, occurrences=pair_case.occurrences[:-1]),
    )
    for corrupted in corruptions:
        with pytest.raises(MatchCompileInvariantError, match="semantic replay"):
            compiler_module._validate_semantic_replay(corrupted)


def test_semantic_replay_rejects_incompatible_shared_decision_identity() -> None:
    compiled = _compiled("case true of | true => 1 | false => 2")
    compiled_case = next(iter(compiled.cases.values()))
    root = cast(DecisionSwitch, compiled_case.root)
    shared_leaf = root.keyed_children[0].decision
    forged_root = replace(
        root,
        keyed_children=(
            root.keyed_children[0],
            replace(root.keyed_children[1], decision=shared_leaf),
        ),
    )

    with pytest.raises(MatchCompileInvariantError, match="incompatible states"):
        compiler_module._validate_semantic_replay(replace(compiled_case, root=forged_root))


def test_semantic_replay_memo_rejects_divergent_node_for_one_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled("case true of | true => 1 | false => 2")
    compiled_case = next(iter(compiled.cases.values()))
    root = cast(DecisionSwitch, compiled_case.root)

    def same_state_specialize(
        matrix: PatternMatrix,
        column: int,
        constructor: Constructor,
        allocator: OccurrenceAllocator,
    ) -> Specialization:
        del column, constructor
        return Specialization(matrix, allocator)

    monkeypatch.setattr(compiler_module, "specialize", same_state_specialize)
    with pytest.raises(MatchCompileInvariantError, match="divergent decision identities"):
        compiler_module._validate_semantic_replay(compiled_case)

    first_branch = replace(root.keyed_children[0])
    second_branch = replace(root.keyed_children[1])
    cyclic_root = replace(root, keyed_children=(first_branch, second_branch))
    object.__setattr__(first_branch, "decision", cyclic_root)
    object.__setattr__(second_branch, "decision", cyclic_root)
    compiler_module._validate_semantic_replay(replace(compiled_case, root=cyclic_root))


def test_valid_shared_dag_passes_strong_compiled_case_validation() -> None:
    compiled = _compiled(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "let value = pair(left = false, right = false)\n"
        "case value of | pair(left = false, right = false) => 1 | _ => 2"
    )
    compiled_case = next(iter(compiled.cases.values()))
    root = cast(DecisionSwitch, compiled_case.root)
    left = cast(DecisionSwitch, root.keyed_children[0].decision)
    right = cast(DecisionSwitch, left.keyed_children[0].decision)

    assert right.default is left.default
    validate_compiled_case(compiled_case)
    MatchCompiledModule(compiled.checked, compiled.cases)


def test_valid_shared_dag_passes_graph_semantic_replay_validation(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "enum Pair\n"
                "  | pair(left: bool, right: bool)\n"
                "let value = pair(left = false, right = false)\n"
                "case value of | pair(left = false, right = false) => 1 | _ => 2"
            )
        },
    )
    checked = check_program(resolve_program(graph), base_caps())
    result = compile_program_matches(checked)
    assert isinstance(result.compiled, MatchCompiledProgram)
    compiled = result.compiled
    compiled_case = next(iter(compiled.cases_by_module[ENTRY_ID].values()))
    root = cast(DecisionSwitch, compiled_case.root)
    left = cast(DecisionSwitch, root.keyed_children[0].decision)
    right = cast(DecisionSwitch, left.keyed_children[0].decision)

    assert right.default is left.default
    MatchCompiledProgram(checked, compiled.cases_by_module)


def test_duplicate_source_case_ids_are_rejected_by_compilation_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = _checked("case true of | true => 1 | false => 2")
    source_case = _first_case(checked)

    def walk_one_case_twice(_program: object, visit: Callable[[object], None]) -> None:
        visit(source_case)
        visit(source_case)

    monkeypatch.setattr(stage_module, "walk", walk_one_case_twice)

    with pytest.raises(MatchCompileInvariantError, match="duplicate"):
        stage_module._source_cases(checked.resolved.program)
    with pytest.raises(MatchCompileInvariantError, match="duplicate"):
        compile_module_matches(checked)


def test_graph_compiles_imported_cases_and_rejects_wrong_module_provenance(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib\n()",
            "lib": "def f(x: bool) -> int =\n  case x of | true => 1 | false => 2",
        },
    )
    checked = check_program(resolve_program(graph), base_caps())
    result = compile_program_matches(checked)
    assert isinstance(result.compiled, MatchCompiledProgram)
    compiled = result.compiled

    with pytest.raises(MatchCompileInvariantError, match="module mismatch"):
        MatchCompiledProgram(checked, {})

    library_id = next(module_id for module_id in checked.modules if module_id != ENTRY_ID)
    library_cases = compiled.cases_by_module[library_id]
    assert len(library_cases) == 1
    case_id, compiled_case = next(iter(library_cases.items()))
    wrong_context = replace(compiled_case.normalized.case_context, module_id=ENTRY_ID)
    wrong_case = replace(
        compiled_case,
        normalized=replace(compiled_case.normalized, case_context=wrong_context),
    )
    corrupted = dict(compiled.cases_by_module)
    corrupted[library_id] = {case_id: wrong_case}
    with pytest.raises(MatchCompileInvariantError, match="belongs to module"):
        MatchCompiledProgram(checked, corrupted)


def test_graph_artifact_rejects_compiled_case_semantic_corruption(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "case true of | true => 1 | false => 2\ncase false of | true => 3 | false => 4"
            )
        },
    )
    checked = check_program(resolve_program(graph), base_caps())
    result = compile_program_matches(checked)
    assert isinstance(result.compiled, MatchCompiledProgram)
    compiled = result.compiled
    entry_cases = compiled.cases_by_module[ENTRY_ID]
    first_id, second_id = tuple(entry_cases)
    first = entry_cases[first_id]
    second = entry_cases[second_id]
    first_root = cast(DecisionSwitch, first.root)

    corruptions = (
        replace(first, root=second.root, occurrences=second.occurrences),
        replace(
            first,
            root=DecisionLeaf(999, ()),
            reachable_action_ids=(999,),
        ),
        replace(first, reachable_action_ids=tuple(reversed(first.reachable_action_ids))),
        replace(first, reachable_action_ids=()),
        replace(
            first,
            root=replace(
                first_root,
                keyed_children=(
                    DecisionBranch(
                        first_root.keyed_children[0].constructor,
                        DecisionFail(),
                    ),
                    *first_root.keyed_children[1:],
                ),
            ),
        ),
    )
    for corrupted_case in corruptions:
        corrupted_modules = {
            module_id: dict(module_cases)
            for module_id, module_cases in compiled.cases_by_module.items()
        }
        corrupted_modules[ENTRY_ID][first_id] = corrupted_case
        with pytest.raises(MatchCompileInvariantError):
            MatchCompiledProgram(checked, corrupted_modules)

    invalid_graph = make_graph_from_files(
        tmp_path / "invalid",
        {"entry": "case true of | true => 1"},
    )
    invalid_checked = check_program(resolve_program(invalid_graph), base_caps())
    invalid_owner = invalid_checked.modules[ENTRY_ID]
    invalid_case = _first_case(invalid_owner)
    invalid_compiled = compile_case(normalize_case(invalid_case, invalid_owner))
    stripped = replace(invalid_compiled, issues=())
    with pytest.raises(MatchCompileInvariantError):
        MatchCompiledProgram(
            invalid_checked,
            {ENTRY_ID: {invalid_case.node_id: stripped}},
        )


def test_graph_artifact_replays_source_semantics_against_every_decision_edge(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(tmp_path, {"entry": _SEMANTIC_REPLAY_SOURCE})
    checked = check_program(resolve_program(graph), base_caps())
    result = compile_program_matches(checked)
    assert isinstance(result.compiled, MatchCompiledProgram)
    compiled = result.compiled
    entry_cases = compiled.cases_by_module[ENTRY_ID]
    case_ids = tuple(entry_cases)
    cases = tuple(entry_cases.values())

    MatchCompiledProgram(checked, compiled.cases_by_module)
    for _, case_index, corrupted_case in _semantic_replay_corruptions(cases):
        corrupted_modules = {
            module_id: dict(module_cases)
            for module_id, module_cases in compiled.cases_by_module.items()
        }
        corrupted_modules[ENTRY_ID][case_ids[case_index]] = corrupted_case
        with pytest.raises(MatchCompileInvariantError, match="semantic replay"):
            MatchCompiledProgram(checked, corrupted_modules)


def test_graph_reports_error_from_unexecuted_imported_module(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib\n()",
            "lib": "def f(x: bool) -> int =\n  case x of | true => 1",
        },
    )
    checked = check_program(resolve_program(graph), base_caps())
    result = compile_program_matches(checked)
    assert result.compiled is None
    assert len(result.issues) == 1


def test_pipeline_nonraising_helpers_defend_against_wrong_artifact_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    single = _compiled("let x = 1\nx")
    graph = make_graph_from_files(tmp_path, {"entry": "()"})
    checked = check_program(resolve_program(graph), base_caps())
    single_result = MatchCompilationResult(compiled=single, issues=())
    monkeypatch.setattr(
        "agm.agl.matchcompile.compile_program_matches", lambda _checked: single_result
    )
    compiled, diagnostics = _run_matchcompile_program(checked)
    assert compiled is None
    assert "module artifact" in diagnostics[0].message


def test_single_and_program_discovery_surface_match_errors() -> None:
    runtime = PipelineDriver()
    discovery = runtime.discover_params(
        runtime.prepare_program("param n: int = 1\ncase true of | true => n")
    )
    assert discovery.compiled is None
    assert any("Non-exhaustive" in item.message for item in discovery.diagnostics)

    prepared_program = PipelineDriver.prepare_program(
        "case true of | true => ()",
        entry_path=None,
        roots=RootSet(roots=frozenset()),
        default_stdlib=False,
    )
    program_discovery = runtime.discover_params(prepared_program)
    assert program_discovery.compiled is None
    assert any("Non-exhaustive" in item.message for item in program_discovery.diagnostics)


@pytest.mark.parametrize("check_only", [False, True])
def test_single_discovery_and_cached_run_compile_matches_once(
    monkeypatch: pytest.MonkeyPatch, check_only: bool
) -> None:
    compile_count = 0

    def counted_compile(checked: CheckedProgram) -> MatchCompilationResult:
        nonlocal compile_count
        compile_count += 1
        return compile_program_matches(checked)

    monkeypatch.setattr(
        "agm.agl.matchcompile.compile_program_matches",
        counted_compile,
    )
    runtime = PipelineDriver()
    prepared = runtime.prepare_program(
        "param selected: bool = true\ncase selected of | true => 1 | false => 0"
    )

    discovery = runtime.discover_params(prepared)
    assert discovery.compiled is not None
    result = runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
        check_only=check_only,
    )

    assert result.ok, result.diagnostics
    assert compile_count == 1


@pytest.mark.parametrize("check_only", [False, True])
def test_program_discovery_and_cached_run_compile_matches_once(
    monkeypatch: pytest.MonkeyPatch, check_only: bool
) -> None:
    compile_count = 0

    def counted_compile(checked: CheckedProgram) -> MatchCompilationResult:
        nonlocal compile_count
        compile_count += 1
        return compile_program_matches(checked)

    monkeypatch.setattr(
        "agm.agl.matchcompile.compile_program_matches",
        counted_compile,
    )
    runtime = PipelineDriver()
    prepared = _prepared_program(
        "param selected: bool = true\ncase selected of | true => 1 | false => 0"
    )

    discovery = runtime.discover_params(prepared)
    assert discovery.compiled is not None
    result = runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
        check_only=check_only,
    )

    assert result.ok, result.diagnostics
    assert compile_count == 1


def test_discovery_and_execution_reuse_one_graph_match_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_count = 0

    def counted_compile(checked: CheckedProgram) -> MatchCompilationResult:
        nonlocal compile_count
        compile_count += 1
        return compile_program_matches(checked)

    monkeypatch.setattr(
        "agm.agl.matchcompile.compile_program_matches",
        counted_compile,
    )
    runtime = PipelineDriver()
    prepared = _prepared_program("case true of | true => 1 | false => 0")

    discovery = runtime.discover_params(prepared)
    assert discovery.compiled is not None
    result = runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
    )

    assert result.ok, result.diagnostics
    assert compile_count == 1


def test_match_invalid_unreachable_case_fails_single_dry_run() -> None:
    result = PipelineDriver().run(
        "def dormant(x: bool) -> int =\n  case x of\n    | true => 1\n()",
        check_only=True,
    )

    assert not result.ok
    assert result.error is None
    assert result.bindings == {}
    assert [(diagnostic.severity, diagnostic.line) for diagnostic in result.diagnostics] == [
        ("error", 2)
    ]


def test_match_invalid_unreachable_import_fails_graph_check_only(tmp_path: Path) -> None:
    (tmp_path / "invalid.agl").write_text(
        "def dormant(x: bool) -> int =\n  case x of\n    | true => 1\n"
    )
    prepared = _prepared_program(
        "import invalid\n()",
        roots=frozenset({tmp_path}),
    )

    result = PipelineDriver().run_prepared(prepared, check_only=True)

    assert not result.ok
    assert result.error is None
    assert result.bindings == {}
    assert [
        (
            diagnostic.severity,
            Path(diagnostic.source_label).name if diagnostic.source_label is not None else None,
            diagnostic.line,
        )
        for diagnostic in result.diagnostics
    ] == [("error", "invalid.agl", 2)]
