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
from agm.agl.pipeline import ArtifactProvenanceError, PipelineDriver, PreparedProgram
from agm.agl.typecheck.program import check_program


def _prepare_graph(source: str) -> PreparedProgram:
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
    prepared_a = runtime.prepare_program('print "stale"')
    discovery_a = runtime.discover_params(prepared_a)
    assert discovery_a.checked is not None
    prepared_b = runtime.prepare_program('print "fresh"')

    with pytest.raises(ArtifactProvenanceError):
        runtime.run_prepared(prepared_b, checked=discovery_a.checked)

    assert capsys.readouterr().out == ""


def test_program_run_rejects_checked_artifact_from_different_prepared_program(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = PipelineDriver()
    prepared_a = _prepare_graph('print "stale"')
    discovery_a = runtime.discover_params(prepared_a)
    assert discovery_a.checked is not None
    prepared_b = _prepare_graph('print "fresh"')

    with pytest.raises(ArtifactProvenanceError):
        runtime.run_prepared(prepared_b, checked=discovery_a.checked)

    assert capsys.readouterr().out == ""


def test_single_run_rechecks_checked_artifact_when_capabilities_change() -> None:
    checking_runtime = PipelineDriver(default_agent=lambda _request: "result")
    runtime = PipelineDriver()
    prepared = runtime.prepare_program('let value = ask "request"\nvalue')
    assert prepared.resolved is not None
    checked = check_program(prepared.resolved, checking_runtime.host_environment().capabilities)
    checked = replace(checked, capabilities=checking_runtime.host_environment().capabilities)

    result = runtime.run_prepared(prepared, checked=checked, check_only=True)

    assert not result.ok
    assert result.error is None
    assert result.diagnostics


def test_program_run_rechecks_checked_artifact_when_capabilities_change() -> None:
    checking_runtime = PipelineDriver(default_agent=lambda _request: "result")
    runtime = PipelineDriver()
    prepared = _prepare_graph('let value = ask "request"\nvalue')
    assert prepared.resolved is not None
    checked = check_program(prepared.resolved, checking_runtime.host_environment().capabilities)
    checked = replace(checked, capabilities=checking_runtime.host_environment().capabilities)

    result = runtime.run_prepared(prepared, checked=checked, check_only=True)

    assert not result.ok
    assert result.error is None
    assert result.diagnostics
