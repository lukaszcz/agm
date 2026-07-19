"""Whole-program match compilation and immutable stage artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from agm.agl.diagnostics import Diagnostic, diagnostic_from_span
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.syntax.nodes import Case, Program
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck.env import CheckedModule
from agm.agl.typecheck.program import CheckedProgram

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
class MatchCompiledModule:
    """A checked module plus one compiled decision DAG per source case."""

    checked: CheckedModule
    cases: Mapping[int, CompiledCase]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", _immutable_cases(self.cases))
        if self_validation_enabled():
            validate_match_compiled_module(self)


@dataclass(frozen=True, slots=True)
class MatchCompiledProgram:
    """A checked program plus total per-module compiled case mappings."""

    checked: CheckedProgram
    cases_by_module: Mapping[ModuleId, Mapping[int, CompiledCase]]

    @property
    def capabilities(self) -> object | None:
        """Capabilities are inseparable from the checked program artifact."""
        return self.checked.capabilities

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases_by_module", _immutable_module_cases(self.cases_by_module))
        if self_validation_enabled():
            validate_match_compiled_program(self)


MatchCompiledArtifact: TypeAlias = MatchCompiledModule | MatchCompiledProgram


@dataclass(frozen=True, slots=True)
class MatchCompilationResult:
    """Non-raising whole-program stage result.

    Carries exactly one of a compiled artifact or a source-ordered tuple of
    issues; the stage entry points below establish both.
    """

    compiled: MatchCompiledArtifact | None
    issues: tuple[MatchIssue, ...]


def _source_cases(program: Program) -> dict[int, Case]:
    """Collect every source case of *program*, keyed by its unique node id."""
    cases: dict[int, Case] = {}

    def collect(node: object) -> None:
        if not isinstance(node, Case):
            return
        if node.node_id in cases:
            raise MatchCompileInvariantError(
                f"duplicate source case node id {node.node_id} in one program"
            )
        cases[node.node_id] = node

    walk(program, collect)
    return cases


def _compile_owner_cases(owner: CheckedModule) -> tuple[dict[int, CompiledCase], list[MatchIssue]]:
    cases: dict[int, CompiledCase] = {}
    issues: list[MatchIssue] = []
    # Every case of one checked owner shares that owner's writable enum
    # spellings, and enumerating them rescans the whole type namespace.
    owner_forms = owner.type_env.enum_owner_forms()
    for case_node_id, source_case in _source_cases(owner.resolved.program).items():
        compiled = compile_case(normalize_case(source_case, owner, enum_owner_forms=owner_forms))
        cases[case_node_id] = compiled
        issues.extend(compiled.issues)
    return cases, issues


def _rejected(
    cases_by_owner: Iterable[Mapping[int, CompiledCase]],
    issues: tuple[MatchIssue, ...],
) -> MatchCompilationResult:
    """Build the issue-carrying stage result, discarding the cases compiled so far.

    Rejected cases never reach an artifact, so this is the one boundary at which
    they can be validated; the checks run only when optional match-compilation
    validation is enabled (see :mod:`agm.agl.self_validation`).
    """
    if self_validation_enabled():
        for owner_cases in cases_by_owner:
            for compiled in owner_cases.values():
                validate_compiled_case(compiled)
    return MatchCompilationResult(compiled=None, issues=issues)


def compile_module_matches(checked: CheckedModule) -> MatchCompilationResult:
    """Compile every case in a checked module source without raising for source issues."""
    cases, issues = _compile_owner_cases(checked)
    sorted_issues = tuple(sorted(issues, key=issue_sort_key))
    if sorted_issues:
        return _rejected((cases,), sorted_issues)
    return MatchCompilationResult(
        compiled=MatchCompiledModule(checked=checked, cases=cases), issues=()
    )


def compile_program_matches(checked: CheckedProgram) -> MatchCompilationResult:
    """Compile every case in every reachable checked module without source-error raises."""
    cases_by_module: dict[ModuleId, Mapping[int, CompiledCase]] = {}
    issues: list[MatchIssue] = []
    for module_id, checked_module in checked.modules.items():
        module_cases, module_issues = _compile_owner_cases(checked_module)
        cases_by_module[module_id] = module_cases
        issues.extend(module_issues)
    sorted_issues = tuple(sorted(issues, key=issue_sort_key))
    if sorted_issues:
        return _rejected(cases_by_module.values(), sorted_issues)
    return MatchCompilationResult(
        compiled=MatchCompiledProgram(checked=checked, cases_by_module=cases_by_module),
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


# ---------------------------------------------------------------------------
# Optional self-validation
#
# Invariant self-checks that re-verify this module's own output.  They never
# change the compiler's result and run only when optional match-compilation
# validation is enabled (see ``agm.agl.self_validation``); the test harness
# turns them on so every compile in the suite is validated.
# ---------------------------------------------------------------------------


def _validate_cases(
    *,
    owner: CheckedModule,
    module_id: ModuleId,
    cases: Mapping[int, CompiledCase],
) -> None:
    program = owner.resolved.program
    expected = _source_cases(program)
    actual_ids = set(cases)
    expected_ids = set(expected)
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    # Hoisted for the same reason as in ``_compile_owner_cases``: every case of
    # one checked owner shares that owner's writable enum spellings, and
    # enumerating them rescans the whole type namespace.
    owner_forms = owner.type_env.enum_owner_forms()
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
                f"compiled case {case_id} belongs to module {context.module_id.path_str()}, "
                f"not {module_id.path_str()}"
            )
        if context.owner_program is not program:
            raise MatchCompileInvariantError(
                f"compiled case {case_id} belongs to a different checked program"
            )
        validate_compiled_case(
            compiled,
            expected_normalized=normalize_case(source_case, owner, enum_owner_forms=owner_forms),
            require_success=True,
        )


def validate_match_compiled_module(compiled: MatchCompiledModule) -> None:
    """Validate totality and provenance of a module artifact."""
    _validate_cases(
        owner=compiled.checked,
        module_id=ENTRY_ID,
        cases=compiled.cases,
    )


def validate_match_compiled_program(compiled: MatchCompiledProgram) -> None:
    """Validate totality, module ownership, and provenance of a program artifact."""
    expected_modules = set(compiled.checked.modules)
    actual_modules = set(compiled.cases_by_module)
    if expected_modules != actual_modules:
        missing = sorted((mid.path_str() for mid in expected_modules - actual_modules))
        extra = sorted((mid.path_str() for mid in actual_modules - expected_modules))
        raise MatchCompileInvariantError(
            f"match-compiled program module mismatch; missing={missing}, extra={extra}"
        )
    for module_id, checked_module in compiled.checked.modules.items():
        _validate_cases(
            owner=checked_module,
            module_id=module_id,
            cases=compiled.cases_by_module[module_id],
        )


__all__ = [
    "MatchCompilationResult",
    "MatchCompiledArtifact",
    "MatchCompiledProgram",
    "MatchCompiledModule",
    "compile_program_matches",
    "compile_module_matches",
    "diagnostic_from_match_issue",
    "diagnostics_from_match_issues",
    "validate_match_compiled_program",
    "validate_match_compiled_module",
]
