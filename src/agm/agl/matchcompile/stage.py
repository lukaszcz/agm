"""Whole-program match compilation and immutable match-site artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from agm.agl.diagnostics import Diagnostic, diagnostic_from_span
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.syntax.nodes import Case, LetDecl, Program
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck.env import CheckedModule
from agm.agl.typecheck.program import CheckedProgram

from .compiler import (
    CompiledCase,
    CompiledMatchSite,
    compile_case,
    compile_match_site,
    validate_compiled_case,
)
from .diagnostics import (
    MatchIssue,
    NonExhaustiveIssue,
    RedundantArmIssue,
    RefutableLetIssue,
    issue_sort_key,
    render_witness,
)
from .model import MatchSiteKind, NormalizedMatchSite
from .normalize import (
    MatchCompileInvariantError,
    normalize_case,
    normalize_let,
    resolve_bare_enum_constructors,
)

SourceMatchSite: TypeAlias = Case | LetDecl


def _immutable_sites(sites: Mapping[int, CompiledMatchSite]) -> Mapping[int, CompiledMatchSite]:
    return MappingProxyType(dict(sites))


def _immutable_module_sites(
    sites_by_module: Mapping[ModuleId, Mapping[int, CompiledMatchSite]],
) -> Mapping[ModuleId, Mapping[int, CompiledMatchSite]]:
    return MappingProxyType(
        {module_id: _immutable_sites(sites) for module_id, sites in sites_by_module.items()}
    )


def _case_sites(sites: Mapping[int, CompiledMatchSite]) -> Mapping[int, CompiledCase]:
    """Expose the decision view consumed by case-only lowering until TASK 009."""
    return MappingProxyType(
        {node_id: site for node_id, site in sites.items() if site.source_kind is MatchSiteKind.CASE}
    )


@dataclass(frozen=True, slots=True)
class MatchCompiledModule:
    """A checked module plus one compiled decision DAG per source match site."""

    checked: CheckedModule
    sites: Mapping[int, CompiledMatchSite]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sites", _immutable_sites(self.sites))
        if self_validation_enabled():
            validate_match_compiled_module(self)

    @property
    def cases(self) -> Mapping[int, CompiledCase]:
        """Immutable filtered view for consumers that only lower source cases."""
        return _case_sites(self.sites)


@dataclass(frozen=True, slots=True)
class MatchCompiledProgram:
    """A checked program plus total per-module compiled match-site mappings."""

    checked: CheckedProgram
    sites_by_module: Mapping[ModuleId, Mapping[int, CompiledMatchSite]]

    @property
    def capabilities(self) -> object | None:
        """Capabilities are inseparable from the checked program artifact."""
        return self.checked.capabilities

    def __post_init__(self) -> None:
        object.__setattr__(self, "sites_by_module", _immutable_module_sites(self.sites_by_module))
        if self_validation_enabled():
            validate_match_compiled_program(self)

    @property
    def cases_by_module(self) -> Mapping[ModuleId, Mapping[int, CompiledCase]]:
        """Immutable filtered per-module case view for the current lowerer."""
        return MappingProxyType(
            {module_id: _case_sites(sites) for module_id, sites in self.sites_by_module.items()}
        )


MatchCompiledArtifact: TypeAlias = MatchCompiledModule | MatchCompiledProgram


@dataclass(frozen=True, slots=True)
class MatchCompilationResult:
    """Non-raising stage result with exactly one artifact or source issue tuple."""

    compiled: MatchCompiledArtifact | None
    issues: tuple[MatchIssue, ...]


def _source_sites(program: Program) -> dict[int, SourceMatchSite]:
    """Collect every source case and immutable let, keyed by unique node id."""
    sites: dict[int, SourceMatchSite] = {}

    def collect(node: object) -> None:
        if not isinstance(node, (Case, LetDecl)):
            return
        if node.node_id in sites:
            raise MatchCompileInvariantError(
                f"duplicate source match-site node id {node.node_id} in one program"
            )
        sites[node.node_id] = node

    walk(program, collect)
    return sites


def _source_cases(program: Program) -> dict[int, Case]:
    """Compatibility helper for white-box case-only tests."""
    return {
        node_id: site for node_id, site in _source_sites(program).items() if isinstance(site, Case)
    }


def _compile_owner_sites(
    owner: CheckedModule,
) -> tuple[dict[int, CompiledMatchSite], list[MatchIssue]]:
    sites: dict[int, CompiledMatchSite] = {}
    issues: list[MatchIssue] = []
    owner_forms = owner.type_env.enum_owner_forms()
    blocked_variants = owner.type_env.blocked_enum_variants()
    bare_constructors = resolve_bare_enum_constructors(owner)
    for site_node_id, source_site in _source_sites(owner.resolved.program).items():
        if isinstance(source_site, Case):
            normalized = normalize_case(
                source_site,
                owner,
                enum_owner_forms=owner_forms,
                blocked_enum_variants=blocked_variants,
                bare_enum_constructors=bare_constructors,
            )
            compiled = compile_case(normalized)
        else:
            normalized = normalize_let(
                source_site,
                owner,
                enum_owner_forms=owner_forms,
                blocked_enum_variants=blocked_variants,
                bare_enum_constructors=bare_constructors,
            )
            compiled = compile_match_site(normalized)
        sites[site_node_id] = compiled
        issues.extend(compiled.issues)
    return sites, issues


def _rejected(
    sites_by_owner: Iterable[Mapping[int, CompiledMatchSite]],
    issues: tuple[MatchIssue, ...],
) -> MatchCompilationResult:
    if self_validation_enabled():
        for owner_sites in sites_by_owner:
            for compiled in owner_sites.values():
                validate_compiled_case(compiled)
    return MatchCompilationResult(compiled=None, issues=issues)


def compile_module_matches(checked: CheckedModule) -> MatchCompilationResult:
    """Compile every match site in a checked module without raising source issues."""
    sites, issues = _compile_owner_sites(checked)
    sorted_issues = tuple(sorted(issues, key=issue_sort_key))
    if sorted_issues:
        return _rejected((sites,), sorted_issues)
    return MatchCompilationResult(
        compiled=MatchCompiledModule(checked=checked, sites=sites), issues=()
    )


def compile_program_matches(checked: CheckedProgram) -> MatchCompilationResult:
    """Compile every match site in every reachable checked module."""
    sites_by_module: dict[ModuleId, Mapping[int, CompiledMatchSite]] = {}
    issues: list[MatchIssue] = []
    for module_id, checked_module in checked.modules.items():
        module_sites, module_issues = _compile_owner_sites(checked_module)
        sites_by_module[module_id] = module_sites
        issues.extend(module_issues)
    sorted_issues = tuple(sorted(issues, key=issue_sort_key))
    if sorted_issues:
        return _rejected(sites_by_module.values(), sorted_issues)
    return MatchCompilationResult(
        compiled=MatchCompiledProgram(checked=checked, sites_by_module=sites_by_module), issues=()
    )


def diagnostic_from_match_issue(issue: MatchIssue) -> Diagnostic:
    """Adapt one structured compiler issue to the ordinary static diagnostic channel."""
    if isinstance(issue, NonExhaustiveIssue):
        message = f"Non-exhaustive case; missing pattern: {render_witness(issue.witness)}."
    elif isinstance(issue, RefutableLetIssue):
        message = f"Refutable let pattern; missing pattern: {render_witness(issue.witness)}."
    elif isinstance(issue, RedundantArmIssue):
        message = "Redundant case arm; this pattern can never be selected."
    else:
        raise AssertionError(f"unsupported match issue: {type(issue).__name__}")
    return diagnostic_from_span(message, issue.span)


def diagnostics_from_match_issues(issues: tuple[MatchIssue, ...]) -> tuple[Diagnostic, ...]:
    """Adapt and deterministically order match issues for pipeline consumers."""
    return tuple(diagnostic_from_match_issue(issue) for issue in sorted(issues, key=issue_sort_key))


def _normalize_source_site(source: SourceMatchSite, owner: CheckedModule) -> NormalizedMatchSite:
    if isinstance(source, Case):
        return normalize_case(source, owner)
    return normalize_let(source, owner)


def _validate_sites(
    *,
    owner: CheckedModule,
    module_id: ModuleId,
    sites: Mapping[int, CompiledMatchSite],
) -> None:
    program = owner.resolved.program
    expected = _source_sites(program)
    actual_ids = set(sites)
    expected_ids = set(expected)
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing:
        raise MatchCompileInvariantError(
            f"match-compiled artifact is missing match-site ids {sorted(missing)}"
        )
    if extra:
        raise MatchCompileInvariantError(
            f"match-compiled artifact has extra match-site ids {sorted(extra)}"
        )
    for site_id, compiled in sites.items():
        source = expected[site_id]
        normalized = compiled.normalized
        expected_kind = MatchSiteKind.CASE if isinstance(source, Case) else MatchSiteKind.LET
        if compiled.site_node_id != site_id or normalized.site_node_id != source.node_id:
            raise MatchCompileInvariantError(
                f"compiled match-site mapping key {site_id} does not match its source"
            )
        if normalized.source_kind is not expected_kind:
            raise MatchCompileInvariantError(
                f"compiled match site {site_id} carries the wrong source kind"
            )
        if normalized.span != source.span:
            raise MatchCompileInvariantError(
                f"compiled match site {site_id} carries mismatched source provenance"
            )
        context = normalized.case_context
        if context.module_id != module_id:
            raise MatchCompileInvariantError(
                f"compiled match site {site_id} belongs to module {context.module_id.path_str()}, "
                f"not {module_id.path_str()}"
            )
        if context.owner_program is not program:
            raise MatchCompileInvariantError(
                f"compiled match site {site_id} belongs to a different checked program"
            )
        validate_compiled_case(
            compiled,
            expected_normalized=_normalize_source_site(source, owner),
            require_success=True,
        )


def validate_match_compiled_module(compiled: MatchCompiledModule) -> None:
    """Validate totality, ownership, provenance, and replay for a module artifact."""
    _validate_sites(owner=compiled.checked, module_id=ENTRY_ID, sites=compiled.sites)


def validate_match_compiled_program(compiled: MatchCompiledProgram) -> None:
    """Validate totality, ownership, provenance, and replay for a program artifact."""
    expected_modules = set(compiled.checked.modules)
    actual_modules = set(compiled.sites_by_module)
    if expected_modules != actual_modules:
        missing = sorted((mid.path_str() for mid in expected_modules - actual_modules))
        extra = sorted((mid.path_str() for mid in actual_modules - expected_modules))
        raise MatchCompileInvariantError(
            f"match-compiled program module mismatch; missing={missing}, extra={extra}"
        )
    for module_id, checked_module in compiled.checked.modules.items():
        _validate_sites(
            owner=checked_module,
            module_id=module_id,
            sites=compiled.sites_by_module[module_id],
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
