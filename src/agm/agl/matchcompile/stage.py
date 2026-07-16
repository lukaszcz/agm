"""Whole-program match compilation and immutable stage artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from agm.agl.diagnostics import Diagnostic, diagnostic_from_span
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.syntax.nodes import Case, Program
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck.env import CheckedProgram
from agm.agl.typecheck.graph import CheckedModule, CheckedModuleGraph

from .compiler import CompiledCase, compile_case, validate_compiled_case
from .diagnostics import (
    MatchIssue,
    NonExhaustiveIssue,
    RedundantArmIssue,
    issue_sort_key,
    render_witness,
)
from .normalize import MatchCompileInvariantError, normalize_case


def _immutable_cases(cases: Mapping[int, CompiledCase]) -> Mapping[int, CompiledCase]:
    return MappingProxyType(dict(cases))


def _immutable_module_cases(
    cases_by_module: Mapping[ModuleId, Mapping[int, CompiledCase]],
) -> Mapping[ModuleId, Mapping[int, CompiledCase]]:
    return MappingProxyType(
        {module_id: _immutable_cases(cases) for module_id, cases in cases_by_module.items()}
    )


@dataclass(frozen=True, slots=True)
class MatchCompiledProgram:
    """A checked single program plus one compiled decision DAG per source case."""

    checked: CheckedProgram
    cases: Mapping[int, CompiledCase]

    @property
    def capabilities(self) -> object | None:
        """Capabilities are inseparable from the checked artifact."""
        return self.checked.capabilities

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", _immutable_cases(self.cases))
        validate_match_compiled_program(self)


@dataclass(frozen=True, slots=True)
class MatchCompiledModuleGraph:
    """A checked module graph plus total per-module compiled case mappings."""

    checked_graph: CheckedModuleGraph
    cases_by_module: Mapping[ModuleId, Mapping[int, CompiledCase]]

    @property
    def capabilities(self) -> object | None:
        """Capabilities are inseparable from the checked graph artifact."""
        return self.checked_graph.capabilities

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases_by_module", _immutable_module_cases(self.cases_by_module))
        validate_match_compiled_graph(self)


MatchCompiledArtifact: TypeAlias = MatchCompiledProgram | MatchCompiledModuleGraph


@dataclass(frozen=True, slots=True)
class MatchCompilationResult:
    """Non-raising whole-program stage result."""

    compiled: MatchCompiledArtifact | None
    issues: tuple[MatchIssue, ...]

    def __post_init__(self) -> None:
        if self.issues != tuple(sorted(self.issues, key=issue_sort_key)):
            raise ValueError("match-compilation issues must be sorted by source location")
        if (self.compiled is None) == (not self.issues):
            raise ValueError("match compilation must return exactly one of an artifact or issues")


def _source_cases(program: Program) -> tuple[Case, ...]:
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(program, collect)
    return tuple(cases)


def _expected_case_map(program: Program) -> dict[int, Case]:
    expected: dict[int, Case] = {}
    for case in _source_cases(program):
        if case.node_id in expected:
            raise MatchCompileInvariantError(
                f"duplicate source case node id {case.node_id} in one program"
            )
        expected[case.node_id] = case
    return expected


def _validate_cases(
    *,
    owner: CheckedProgram | CheckedModule,
    module_id: ModuleId,
    cases: Mapping[int, CompiledCase],
) -> None:
    program = owner.resolved.program
    expected = _expected_case_map(program)
    actual_ids = set(cases)
    expected_ids = set(expected)
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing:
        raise MatchCompileInvariantError(
            f"match-compiled artifact is missing case ids {sorted(missing)}"
        )
    if extra:
        raise MatchCompileInvariantError(
            f"match-compiled artifact has extra case ids {sorted(extra)}"
        )
    for case_id, compiled in cases.items():
        source_case = expected[case_id]
        normalized = compiled.normalized
        if compiled.case_node_id != case_id or normalized.case_node_id != source_case.node_id:
            raise MatchCompileInvariantError(
                f"compiled case mapping key {case_id} does not match its source case"
            )
        if normalized.span != source_case.span:
            raise MatchCompileInvariantError(
                f"compiled case {case_id} carries mismatched source provenance"
            )
        context = normalized.case_context
        if context.module_id != module_id:
            raise MatchCompileInvariantError(
                f"compiled case {case_id} belongs to module {context.module_id.dotted()}, "
                f"not {module_id.dotted()}"
            )
        if context.owner_program is not program:
            raise MatchCompileInvariantError(
                f"compiled case {case_id} belongs to a different checked program"
            )
        validate_compiled_case(
            compiled,
            expected_normalized=normalize_case(source_case, owner),
            require_success=True,
        )


def validate_match_compiled_program(compiled: MatchCompiledProgram) -> None:
    """Validate totality and provenance of a single-program artifact."""
    _validate_cases(
        owner=compiled.checked,
        module_id=ENTRY_ID,
        cases=compiled.cases,
    )


def validate_match_compiled_graph(compiled: MatchCompiledModuleGraph) -> None:
    """Validate totality, module ownership, and provenance of a graph artifact."""
    expected_modules = set(compiled.checked_graph.modules)
    actual_modules = set(compiled.cases_by_module)
    if expected_modules != actual_modules:
        missing = sorted((mid.dotted() for mid in expected_modules - actual_modules))
        extra = sorted((mid.dotted() for mid in actual_modules - expected_modules))
        raise MatchCompileInvariantError(
            f"match-compiled graph module mismatch; missing={missing}, extra={extra}"
        )
    for module_id, checked_module in compiled.checked_graph.modules.items():
        _validate_cases(
            owner=checked_module,
            module_id=module_id,
            cases=compiled.cases_by_module[module_id],
        )


def _compile_owner_cases(
    owner: CheckedProgram | CheckedModule,
) -> tuple[dict[int, CompiledCase], list[MatchIssue]]:
    cases: dict[int, CompiledCase] = {}
    issues: list[MatchIssue] = []
    for source_case in _source_cases(owner.resolved.program):
        if source_case.node_id in cases:
            raise MatchCompileInvariantError(
                f"duplicate source case node id {source_case.node_id} in one program"
            )
        # Clean cases are validated once at the artifact boundary
        # (``MatchCompiled*.__post_init__``); skip the redundant per-case replay
        # here.  Issue-bearing cases never reach that boundary (the stage returns
        # issues without an artifact), so keep their per-case invariant check.
        compiled = compile_case(normalize_case(source_case, owner), validate=False)
        if compiled.issues:
            validate_compiled_case(compiled)
        cases[source_case.node_id] = compiled
        issues.extend(compiled.issues)
    return cases, issues


def compile_program_matches(checked: CheckedProgram) -> MatchCompilationResult:
    """Compile every case in a checked single program without raising for source issues."""
    cases, issues = _compile_owner_cases(checked)
    sorted_issues = tuple(sorted(issues, key=issue_sort_key))
    if sorted_issues:
        return MatchCompilationResult(compiled=None, issues=sorted_issues)
    return MatchCompilationResult(
        compiled=MatchCompiledProgram(checked=checked, cases=cases), issues=()
    )


def compile_graph_matches(checked_graph: CheckedModuleGraph) -> MatchCompilationResult:
    """Compile every case in every reachable checked module without source-error raises."""
    cases_by_module: dict[ModuleId, Mapping[int, CompiledCase]] = {}
    issues: list[MatchIssue] = []
    for module_id, checked_module in checked_graph.modules.items():
        module_cases, module_issues = _compile_owner_cases(checked_module)
        cases_by_module[module_id] = module_cases
        issues.extend(module_issues)
    sorted_issues = tuple(sorted(issues, key=issue_sort_key))
    if sorted_issues:
        return MatchCompilationResult(compiled=None, issues=sorted_issues)
    return MatchCompilationResult(
        compiled=MatchCompiledModuleGraph(
            checked_graph=checked_graph, cases_by_module=cases_by_module
        ),
        issues=(),
    )


def diagnostic_from_match_issue(issue: MatchIssue) -> Diagnostic:
    """Adapt one structured compiler issue to the ordinary static diagnostic channel."""
    if isinstance(issue, NonExhaustiveIssue):
        message = f"Non-exhaustive case; missing pattern: {render_witness(issue.witness)}."
    elif isinstance(issue, RedundantArmIssue):
        message = "Redundant case arm; this pattern can never be selected."
    else:
        raise AssertionError(f"unsupported match issue: {type(issue).__name__}")
    return diagnostic_from_span(message, issue.span)


def diagnostics_from_match_issues(issues: tuple[MatchIssue, ...]) -> tuple[Diagnostic, ...]:
    """Adapt and deterministically order match issues for pipeline consumers."""
    return tuple(diagnostic_from_match_issue(issue) for issue in sorted(issues, key=issue_sort_key))


__all__ = [
    "MatchCompilationResult",
    "MatchCompiledArtifact",
    "MatchCompiledModuleGraph",
    "MatchCompiledProgram",
    "compile_graph_matches",
    "compile_program_matches",
    "diagnostic_from_match_issue",
    "diagnostics_from_match_issues",
    "validate_match_compiled_graph",
    "validate_match_compiled_program",
]
