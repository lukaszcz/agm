"""Checker warnings survive later match-compilation failures on every pipeline surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ParamDiscovery, PreparedProgram, RunResult
from agm.agl.repl.session import EntryResult, ReplSession

_FAILING_SOURCE = (
    "agent idle\n"
    'let response: text = ask("Q", on_parse_error = Abort())\n'
    "case true of\n"
    "  | true => ()\n"
)

_CACHED_SOURCE = (
    "# cached header\n"
    "agent idle\n"
    'let response: text = ask("Q", on_parse_error = Abort())\n'
    "case true of\n"
    "  | true => ()\n"
    "  | false => ()\n"
)


def _prepare_graph(source: str) -> PreparedProgram:
    stdlib = Path(__file__).resolve().parent.parent / "stdlib"
    return PipelineDriver.prepare_program(
        source,
        entry_path=None,
        roots=RootSet(roots=frozenset({stdlib})),
    )


def _assert_warning_then_match_error(
    result: RunResult | ParamDiscovery | EntryResult,
) -> None:
    assert [(item.line, item.severity) for item in result.warnings] == [
        (1, "warning"),
        (2, "warning"),
    ]
    assert [(item.line, item.severity) for item in result.diagnostics] == [(3, "error")]


@pytest.mark.parametrize("check_only", [False, True])
def test_single_run_preserves_checker_warning_before_match_failure(check_only: bool) -> None:
    result = PipelineDriver(default_agent=lambda _request: "").run(
        _FAILING_SOURCE,
        check_only=check_only,
    )

    assert not result.ok
    _assert_warning_then_match_error(result)


def test_single_discovery_preserves_checker_warning_before_match_failure() -> None:
    runtime = PipelineDriver(default_agent=lambda _request: "")
    result = runtime.discover_params(runtime.prepare_program(_FAILING_SOURCE))

    assert result.compiled is None
    _assert_warning_then_match_error(result)


@pytest.mark.parametrize("check_only", [False, True])
def test_program_run_preserves_checker_warning_before_match_failure(check_only: bool) -> None:
    result = PipelineDriver(default_agent=lambda _request: "").run_prepared(
        _prepare_graph(_FAILING_SOURCE),
        check_only=check_only,
    )

    assert not result.ok
    _assert_warning_then_match_error(result)


def test_program_discovery_preserves_checker_warning_before_match_failure() -> None:
    result = PipelineDriver(default_agent=lambda _request: "").discover_params(
        _prepare_graph(_FAILING_SOURCE)
    )

    assert result.compiled is None
    _assert_warning_then_match_error(result)


@pytest.mark.parametrize("check_only", [False, True])
def test_repl_preserves_checker_warning_before_match_failure(check_only: bool) -> None:
    result = ReplSession(default_agent=lambda _request: "").eval_entry(
        _FAILING_SOURCE,
        check_only=check_only,
    )

    assert not result.ok
    _assert_warning_then_match_error(result)


@pytest.mark.parametrize("check_only", [False, True])
def test_single_cached_run_does_not_duplicate_checker_warnings(check_only: bool) -> None:
    runtime = PipelineDriver(default_agent=lambda _request: "")
    prepared: PreparedProgram = runtime.prepare_program(_CACHED_SOURCE)
    discovery = runtime.discover_params(prepared)
    assert discovery.compiled is not None

    result = runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
        check_only=check_only,
    )

    assert result.ok
    assert [(item.line, item.severity) for item in result.warnings] == [
        (2, "warning"),
        (3, "warning"),
    ]


def test_graph_cached_consumers_do_not_duplicate_checker_warnings() -> None:
    runtime = PipelineDriver(default_agent=lambda _request: "")
    prepared = _prepare_graph(_CACHED_SOURCE)
    discovery = runtime.discover_params(prepared)
    assert discovery.compiled is not None

    rediscovery = runtime.discover_params(
        prepared,
        compiled=discovery.compiled,
    )
    run = runtime.run_prepared(
        prepared,
        compiled=discovery.compiled,
        check_only=True,
    )

    assert rediscovery.diagnostics == ()
    assert run.ok
    for result in (rediscovery, run):
        assert [(item.line, item.severity) for item in result.warnings] == [
            (2, "warning"),
            (3, "warning"),
        ]
