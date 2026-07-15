"""Checker warnings survive later match-compilation failures on every pipeline surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ParamDiscovery, PreparedGraph, PreparedProgram, RunResult
from agm.agl.repl.session import EntryResult, ReplSession

_FAILING_SOURCE = (
    "agent idle\n"
    'let response: text = ask("Q", on_parse_error = Abort())\n'
    "case true of\n"
    "  | true => ()\n"
)

_CACHED_SOURCE = (
    "config log = true\n"
    "agent idle\n"
    'let response: text = ask("Q", on_parse_error = Abort())\n'
    "case true of\n"
    "  | true => ()\n"
    "  | false => ()\n"
)


def _prepare_graph(source: str) -> PreparedGraph:
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
    result = runtime.discover_params(runtime.prepare(_FAILING_SOURCE))

    assert result.compiled is None
    _assert_warning_then_match_error(result)


@pytest.mark.parametrize("check_only", [False, True])
def test_graph_run_preserves_checker_warning_before_match_failure(check_only: bool) -> None:
    result = PipelineDriver(default_agent=lambda _request: "").run_prepared_graph(
        _prepare_graph(_FAILING_SOURCE),
        check_only=check_only,
    )

    assert not result.ok
    _assert_warning_then_match_error(result)


def test_graph_discovery_preserves_checker_warning_before_match_failure() -> None:
    result = PipelineDriver(default_agent=lambda _request: "").discover_params_graph(
        _prepare_graph(_FAILING_SOURCE)
    )

    assert result.compiled_graph is None
    _assert_warning_then_match_error(result)


def test_startup_config_preserves_checker_warning_before_match_failure() -> None:
    source = "config log = true\n" + _FAILING_SOURCE
    result = PipelineDriver(default_agent=lambda _request: "").collect_startup_config_graph(
        _prepare_graph(source),
        names={"log"},
    )

    assert not result.ok
    assert [(item.line, item.severity) for item in result.warnings] == [
        (2, "warning"),
        (3, "warning"),
    ]
    assert [(item.line, item.severity) for item in result.diagnostics] == [(4, "error")]


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
    prepared: PreparedProgram = runtime.prepare(_CACHED_SOURCE)
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
    discovery = runtime.discover_params_graph(prepared)
    assert discovery.compiled_graph is not None

    rediscovery = runtime.discover_params_graph(
        prepared,
        compiled_graph=discovery.compiled_graph,
    )
    run = runtime.run_prepared_graph(
        prepared,
        compiled_graph=discovery.compiled_graph,
        check_only=True,
    )
    startup = runtime.collect_startup_config_graph(
        prepared,
        names={"log"},
        compiled_graph=discovery.compiled_graph,
    )

    assert rediscovery.diagnostics == ()
    assert run.ok
    assert startup.ok
    for result in (rediscovery, run, startup):
        assert [(item.line, item.severity) for item in result.warnings] == [
            (2, "warning"),
            (3, "warning"),
        ]
