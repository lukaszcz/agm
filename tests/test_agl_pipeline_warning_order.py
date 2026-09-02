"""Checker warnings survive later match-compilation failures on every pipeline surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ParamDiscovery, PreparedProgram, RunResult
from agm.agl.repl.session import EntryResult, ReplSession
from tests._agl_helpers import prepare_inline_command, run_inline_command

_FAILING_SOURCE = (
    'let idle = AgentCommand("idle")\n'
    'let response: text = ask("Q", on-parse-error = Abort())\n'
    "case true of\n"
    "  | true => ()\n"
)

_CACHED_SOURCE = (
    "# cached header\n"
    'let idle = AgentCommand("idle")\n'
    'let response: text = ask("Q", on-parse-error = Abort())\n'
    "case true of\n"
    "  | true => ()\n"
    "  | false => ()\n"
)


def _prepare_graph(source: str) -> PreparedProgram:
    stdlib = Path(__file__).resolve().parent.parent / "stdlib"
    return prepare_inline_command(
        source,
        entry_path=None,
        roots=RootSet(roots=frozenset({stdlib})),
    )


def _assert_warning_then_match_error(
    result: RunResult | ParamDiscovery | EntryResult,
) -> None:
    assert [(item.line, item.severity) for item in result.warnings] == [(2, "warning")]
    assert [(item.line, item.severity) for item in result.diagnostics] == [(3, "error")]


@pytest.mark.parametrize("check_only", [False, True])
def test_single_run_preserves_checker_warning_before_match_failure(check_only: bool) -> None:
    result = run_inline_command(
        PipelineDriver(agent_dispatcher=lambda _request: ""),
        _FAILING_SOURCE,
        check_only=check_only,
    )

    assert not result.ok
    _assert_warning_then_match_error(result)


def test_single_discovery_preserves_checker_warning_before_match_failure() -> None:
    runtime = PipelineDriver(agent_dispatcher=lambda _request: "")
    result = runtime.discover_params(prepare_inline_command(_FAILING_SOURCE))

    assert result.compiled is None
    _assert_warning_then_match_error(result)


@pytest.mark.parametrize("check_only", [False, True])
def test_program_run_preserves_checker_warning_before_match_failure(check_only: bool) -> None:
    result = PipelineDriver(agent_dispatcher=lambda _request: "").run_prepared(
        _prepare_graph(_FAILING_SOURCE),
        check_only=check_only,
    )

    assert not result.ok
    _assert_warning_then_match_error(result)


def test_program_discovery_preserves_checker_warning_before_match_failure() -> None:
    result = PipelineDriver(agent_dispatcher=lambda _request: "").discover_params(
        _prepare_graph(_FAILING_SOURCE)
    )

    assert result.compiled is None
    _assert_warning_then_match_error(result)


@pytest.mark.parametrize("check_only", [False, True])
def test_repl_preserves_checker_warning_before_match_failure(check_only: bool) -> None:
    result = ReplSession(agent_dispatcher=lambda _request: "").eval_entry(
        _FAILING_SOURCE,
        check_only=check_only,
    )

    assert not result.ok
    _assert_warning_then_match_error(result)
