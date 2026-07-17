"""Checked artifacts must be bound to their prepared source and host capabilities.

Source provenance is an internal invariant, re-verified only under AgL's
self-validation (enabled suite-wide) and raising when violated.  Capability
provenance is different: it is real cache invalidation on the production path,
and re-running the checker is its fallback.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ArtifactProvenanceError, PipelineDriver, PreparedGraph
from agm.agl.typecheck import check
from agm.agl.typecheck.graph import check_graph


def _prepare_graph(source: str) -> PreparedGraph:
    stdlib = Path(__file__).resolve().parent.parent / "stdlib"
    return PipelineDriver.prepare_program(
        source,
        entry_path=None,
        roots=RootSet(roots=frozenset({stdlib})),
    )


def test_single_run_rejects_checked_artifact_from_different_prepared_program(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = PipelineDriver()
    prepared_a = runtime.prepare('print "stale"')
    discovery_a = runtime.discover_params(prepared_a)
    assert discovery_a.checked is not None
    prepared_b = runtime.prepare('print "fresh"')

    with pytest.raises(ArtifactProvenanceError):
        runtime.run_prepared(prepared_b, checked=discovery_a.checked)

    assert capsys.readouterr().out == ""


def test_graph_run_rejects_checked_artifact_from_different_prepared_graph(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = PipelineDriver()
    prepared_a = _prepare_graph('print "stale"')
    discovery_a = runtime.discover_params_graph(prepared_a)
    assert discovery_a.checked_graph is not None
    prepared_b = _prepare_graph('print "fresh"')

    with pytest.raises(ArtifactProvenanceError):
        runtime.run_prepared_graph(prepared_b, checked_graph=discovery_a.checked_graph)

    assert capsys.readouterr().out == ""


def test_single_run_rechecks_checked_artifact_when_capabilities_change() -> None:
    checking_runtime = PipelineDriver(default_agent=lambda _request: "result")
    runtime = PipelineDriver()
    prepared = runtime.prepare('let value = ask "request"\nvalue')
    assert prepared.resolved is not None
    checked = check(prepared.resolved, checking_runtime.host_environment().capabilities)
    checked = replace(checked, capabilities=checking_runtime.host_environment().capabilities)

    result = runtime.run_prepared(prepared, checked=checked, check_only=True)

    assert not result.ok
    assert result.error is None
    assert result.diagnostics


def test_graph_run_rechecks_checked_artifact_when_capabilities_change() -> None:
    checking_runtime = PipelineDriver(default_agent=lambda _request: "result")
    runtime = PipelineDriver()
    prepared = _prepare_graph('let value = ask "request"\nvalue')
    assert prepared.resolved_graph is not None
    checked = check_graph(prepared.resolved_graph, checking_runtime.host_environment().capabilities)
    checked = replace(checked, capabilities=checking_runtime.host_environment().capabilities)

    result = runtime.run_prepared_graph(prepared, checked_graph=checked, check_only=True)

    assert not result.ok
    assert result.error is None
    assert result.diagnostics
