"""Whole-program match compilation and immutable match-site artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from agm.agl.diagnostics import Diagnostic, diagnostic_from_span
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.syntax.nodes import Case, LetDecl, Program, simple_let_pattern_name
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck.env import CheckedModule
from agm.agl.typecheck.program import CheckedProgram

from .compiler import CompiledMatchSite, compile_match_site, validate_compiled_case
from .diagnostics import (
    MatchIssue,
    NonExhaustiveIssue,
    RedundantArmIssue,
    RefutableLetIssue,
    issue_sort_key,
    render_witness,
)
from .model import CaseSite, LetSite, MatchCaseContext, NormalizedMatchSite
from .normalize import (
    MatchCompileInvariantError,
    match_case_context,
    normalize_case,
    normalize_let,
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


@dataclass(frozen=True, slots=True)
class MatchCompiledModule:
    """A checked module plus one compiled decision DAG per source match site."""

    checked: CheckedModule
    sites: Mapping[int, CompiledMatchSite]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sites", _immutable_sites(self.sites))
        if self_validation_enabled():
            validate_match_compiled_module(self)


@dataclass(frozen=True, slots=True)
class MatchCompiledProgram:
    """A checked program plus total per-module compiled match-site mappings.

    ``reused_modules`` names the modules whose sites were carried over from an
    earlier compilation of the same checked module instead of being compiled
    here. Self-validation skips them, because it already validated those very
    objects when they were first compiled.
    """

    checked: CheckedProgram
    sites_by_module: Mapping[ModuleId, Mapping[int, CompiledMatchSite]]
    reused_modules: frozenset[ModuleId] = frozenset()

    @property
    def capabilities(self) -> object | None:
        """Capabilities are inseparable from the checked program artifact."""
        return self.checked.capabilities

    def __post_init__(self) -> None:
        object.__setattr__(self, "sites_by_module", _immutable_module_sites(self.sites_by_module))
        if self_validation_enabled():
            validate_match_compiled_program(self)


@dataclass(frozen=True, slots=True)
class CachedModuleSites:
    """One module's compiled match sites and the checked module they came from.

    Compiled sites are a pure function of the checked module they were compiled
    from, so pairing the two is the whole condition for reusing them.
    """

    owner: CheckedModule
    sites: Mapping[int, CompiledMatchSite]


MatchCompiledArtifact: TypeAlias = MatchCompiledModule | MatchCompiledProgram


@dataclass(frozen=True, slots=True)
class MatchCompilationResult:
    """Non-raising stage result with exactly one artifact or source issue tuple."""

    compiled: MatchCompiledArtifact | None
    issues: tuple[MatchIssue, ...]


def _source_sites(program: Program) -> dict[int, SourceMatchSite]:
    """Collect match sites, excluding irrefutable simple-name and discard lets.

    A bare-name or ``_`` let is an irrefutable single-binder site by
    construction. Compiling it would spend normalization, matrix construction,
    occurrence allocation, and DAG derivation to describe one lowering
    instruction, so only cases and destructuring lets enter this artifact.
    """
    sites: dict[int, SourceMatchSite] = {}

    def collect(node: object) -> None:
        if isinstance(node, LetDecl) and simple_let_pattern_name(node.pattern) is not None:
            return
        if not isinstance(node, (Case, LetDecl)):
            return
        if node.node_id in sites:
            raise MatchCompileInvariantError(
                f"duplicate source match-site node id {node.node_id} in one program"
            )
        sites[node.node_id] = node

    walk(program, collect)
    return sites


def _normalize_source_site(
    source: SourceMatchSite, owner: CheckedModule, case_context: MatchCaseContext
) -> NormalizedMatchSite:
    if isinstance(source, Case):
        return normalize_case(source, owner, case_context=case_context)
    return normalize_let(source, owner, case_context=case_context)


def _compile_owner_sites(
    owner: CheckedModule,
) -> tuple[dict[int, CompiledMatchSite], list[MatchIssue]]:
    sites: dict[int, CompiledMatchSite] = {}
    issues: list[MatchIssue] = []
    source_sites = _source_sites(owner.resolved.program)
    if not source_sites:
        return sites, issues
    case_context = match_case_context(owner)
    for site_node_id, source_site in source_sites.items():
        compiled = compile_match_site(_normalize_source_site(source_site, owner, case_context))
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


def cached_module_sites(
    previous: MatchCompiledProgram | None,
) -> dict[ModuleId, CachedModuleSites]:
    """Adapt an earlier whole-program artifact to the per-module reuse seam."""
    if previous is None:
        return {}
    return {
        module_id: CachedModuleSites(previous.checked.modules[module_id], sites)
        for module_id, sites in previous.sites_by_module.items()
        if module_id in previous.checked.modules
    }


def compile_program_matches(
    checked: CheckedProgram,
    cached_sites: Mapping[ModuleId, CachedModuleSites] | None = None,
) -> MatchCompilationResult:
    """Compile every match site in every reachable checked module.

    A module carries its sites over from *cached_sites*, the artifacts of an
    earlier compilation, only while the pairing still holds the very
    :class:`~agm.agl.typecheck.env.CheckedModule` this program carries;
    compiled sites are a pure function of that module, so identity is the whole
    condition. Only a compilation that produced an artifact can be offered
    here, so a module with issues re-reports them every time.
    """
    sites_by_module: dict[ModuleId, Mapping[int, CompiledMatchSite]] = {}
    reused_modules: set[ModuleId] = set()
    issues: list[MatchIssue] = []
    for module_id, checked_module in checked.modules.items():
        cached = cached_sites.get(module_id) if cached_sites is not None else None
        if cached is not None and cached.owner is checked_module:
            sites_by_module[module_id] = cached.sites
            reused_modules.add(module_id)
            continue
        module_sites, module_issues = _compile_owner_sites(checked_module)
        sites_by_module[module_id] = module_sites
        issues.extend(module_issues)
    sorted_issues = tuple(sorted(issues, key=issue_sort_key))
    if sorted_issues:
        return _rejected(sites_by_module.values(), sorted_issues)
    return MatchCompilationResult(
        compiled=MatchCompiledProgram(
            checked=checked,
            sites_by_module=sites_by_module,
            reused_modules=frozenset(reused_modules),
        ),
        issues=(),
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


def _validate_sites(
    *,
    owner: CheckedModule,
    module_id: ModuleId,
    sites: Mapping[int, CompiledMatchSite],
) -> None:
    program = owner.resolved.program
    expected = _source_sites(program)
    case_context = match_case_context(owner)
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
        if compiled.site_node_id != site_id or normalized.site_node_id != source.node_id:
            raise MatchCompileInvariantError(
                f"compiled match-site mapping key {site_id} does not match its source"
            )
        expected_payload = CaseSite if isinstance(source, Case) else LetSite
        if not isinstance(normalized.source, expected_payload):
            raise MatchCompileInvariantError(
                f"compiled match site {site_id} carries the wrong source payload"
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
            expected_normalized=_normalize_source_site(source, owner, case_context),
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
        if module_id in compiled.reused_modules:
            continue
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
    "CachedModuleSites",
    "cached_module_sites",
    "compile_program_matches",
    "diagnostic_from_match_issue",
    "diagnostics_from_match_issues",
    "validate_match_compiled_program",
    "validate_match_compiled_module",
]
