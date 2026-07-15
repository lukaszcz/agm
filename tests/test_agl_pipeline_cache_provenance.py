"""Cached match artifacts must belong to the exact prepared source consuming them."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl.matchcompile import MatchCompiledModuleGraph
from agm.agl.modules.ids import ENTRY_ID, STD_CORE_ID
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ParamDiscovery, PipelineDriver, PreparedGraph


def _prepare_graph(
    source: str,
    *,
    extra_roots: frozenset[Path] = frozenset(),
) -> PreparedGraph:
    stdlib = Path(__file__).resolve().parent.parent / "stdlib"
    return PipelineDriver.prepare_program(
        source,
        entry_path=None,
        roots=RootSet(roots=frozenset({stdlib, *extra_roots})),
    )


def _compiled_graph(source: str) -> tuple[PreparedGraph, ParamDiscovery]:
    prepared = _prepare_graph(source)
    discovery = PipelineDriver().discover_params_graph(prepared)
    assert discovery.compiled_graph is not None
    return prepared, discovery


def _assert_cache_mismatch(diagnostics: Iterator[object]) -> None:
    items = list(diagnostics)
    assert items
    for item in items:
        severity = getattr(item, "severity")
        line = getattr(item, "line")
        assert severity == "error"
        assert isinstance(line, int) and line >= 1


@pytest.mark.parametrize("check_only", [False, True])
def test_single_run_rejects_cached_artifact_from_different_prepared_program(
    check_only: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = PipelineDriver()
    prepared_a = runtime.prepare('param a: int = 1\nprint "stale ${a}"')
    discovery_a = runtime.discover_params(prepared_a)
    assert discovery_a.compiled is not None
    prepared_b = runtime.prepare('param b: int = 2\nprint "fresh ${b}"')

    result = runtime.run_prepared(
        prepared_b,
        check_only=check_only,
        compiled=discovery_a.compiled,
    )

    assert not result.ok
    assert result.error is None
    _assert_cache_mismatch(iter(result.diagnostics))
    assert result.bindings == {}
    assert capsys.readouterr().out == ""


def test_graph_discovery_rejects_cached_artifact_from_different_prepared_graph() -> None:
    _prepared_a, discovery_a = _compiled_graph("param a: int = 1\na")
    prepared_b = _prepare_graph("param b: text = \"b\"\nb")

    discovery_b = PipelineDriver().discover_params_graph(
        prepared_b, compiled_graph=discovery_a.compiled_graph
    )

    assert discovery_b.params == ()
    assert discovery_b.checked is None
    assert discovery_b.checked_graph is None
    assert discovery_b.compiled_graph is None
    _assert_cache_mismatch(iter(discovery_b.diagnostics))


def test_graph_discovery_rejects_cached_artifact_with_different_entry_identity() -> None:
    prepared, discovery = _compiled_graph("param value: int = 1\nvalue")
    assert discovery.compiled_graph is not None
    wrong_entry_checked = replace(
        discovery.compiled_graph.checked_graph,
        entry_id=STD_CORE_ID,
    )
    wrong_entry_compiled = MatchCompiledModuleGraph(
        checked_graph=wrong_entry_checked,
        cases_by_module=discovery.compiled_graph.cases_by_module,
    )

    rejected = PipelineDriver().discover_params_graph(
        prepared,
        compiled_graph=wrong_entry_compiled,
    )

    assert rejected.params == ()
    assert rejected.checked_graph is None
    assert rejected.compiled_graph is None
    _assert_cache_mismatch(iter(rejected.diagnostics))


def test_graph_discovery_rejects_cached_artifact_with_different_module_set(
    tmp_path: Path,
) -> None:
    (tmp_path / "helper.agl").write_text("def answer() -> int = 42\n")
    prepared_with_import = _prepare_graph(
        "import helper\nlet value = 1\nvalue",
        extra_roots=frozenset({tmp_path}),
    )
    discovery = PipelineDriver().discover_params_graph(prepared_with_import)
    assert discovery.compiled_graph is not None
    prepared_without_import = _prepare_graph("let value = 2\nvalue")

    rejected = PipelineDriver().discover_params_graph(
        prepared_without_import,
        compiled_graph=discovery.compiled_graph,
    )

    assert rejected.params == ()
    assert rejected.checked_graph is None
    assert rejected.compiled_graph is None
    _assert_cache_mismatch(iter(rejected.diagnostics))


def test_single_run_rechecks_cached_artifact_when_host_capabilities_change() -> None:
    source = 'ask("cached")'
    cached_by = PipelineDriver(default_agent=lambda _prompt: "answer")
    prepared = cached_by.prepare(source)
    discovery = cached_by.discover_params(prepared)
    assert discovery.compiled is not None

    result = PipelineDriver().run_prepared(prepared, compiled=discovery.compiled)

    assert not result.ok
    assert result.error is None
    assert result.diagnostics


def test_graph_cache_mismatch_without_prepared_entry_uses_fallback_location() -> None:
    prepared, discovery = _compiled_graph("let value = 1\nvalue")
    assert prepared.resolved_graph is not None
    assert discovery.compiled_graph is not None
    resolved_without_entry = replace(
        prepared.resolved_graph,
        modules={
            module_id: module
            for module_id, module in prepared.resolved_graph.modules.items()
            if module_id != ENTRY_ID
        },
    )
    malformed_prepared = replace(prepared, resolved_graph=resolved_without_entry)

    rejected = PipelineDriver().discover_params_graph(
        malformed_prepared,
        compiled_graph=discovery.compiled_graph,
    )

    assert rejected.params == ()
    assert rejected.checked_graph is None
    assert rejected.compiled_graph is None
    assert len(rejected.diagnostics) == 1
    assert rejected.diagnostics[0].severity == "error"
    assert rejected.diagnostics[0].line == 1


@pytest.mark.parametrize("check_only", [False, True])
def test_graph_run_rejects_cached_artifact_from_different_prepared_graph(
    check_only: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepared_a, discovery_a = _compiled_graph('print "stale"')
    prepared_b = _prepare_graph('print "fresh"')

    result = PipelineDriver().run_prepared_graph(
        prepared_b,
        check_only=check_only,
        compiled_graph=discovery_a.compiled_graph,
    )

    assert not result.ok
    assert result.error is None
    _assert_cache_mismatch(iter(result.diagnostics))
    assert result.bindings == {}
    assert capsys.readouterr().out == ""


def test_startup_config_rejects_cached_artifact_from_different_prepared_graph(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _prepared_a, discovery_a = _compiled_graph(
        'config log = true\nprint "stale startup"'
    )
    prepared_b = _prepare_graph('config log = false\nprint "fresh startup"')

    result = PipelineDriver().collect_startup_config_graph(
        prepared_b,
        names={"log"},
        compiled_graph=discovery_a.compiled_graph,
    )

    assert not result.ok
    assert result.error is None
    assert result.values == {}
    assert result.checked_graph is None
    assert result.compiled_graph is None
    _assert_cache_mismatch(iter(result.diagnostics))
    assert capsys.readouterr().out == ""


def test_startup_config_validates_cached_artifact_even_when_no_value_is_requested() -> None:
    _prepared_a, discovery_a = _compiled_graph("let a = 1\na")
    prepared_b = _prepare_graph("let b = 2\nb")

    result = PipelineDriver().collect_startup_config_graph(
        prepared_b,
        names={"log"},
        compiled_graph=discovery_a.compiled_graph,
    )

    assert not result.ok
    assert result.values == {}
    _assert_cache_mismatch(iter(result.diagnostics))
