"""Cached pipeline artifacts retain prepared-source and host capability provenance.

Frontend source provenance is an internal invariant, re-verified only under
AgL's self-validation (enabled suite-wide). Lowered executable provenance is
owned by the issuing pipeline. Capability changes invalidate either cache on
the production path and rerun the affected passes.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agm.agl.lower import lower_program
from agm.agl.pipeline import (
    ArgumentPreflight,
    ArtifactProvenanceError,
    PipelineDriver,
    PreparedProgram,
)
from agm.agl.runtime.arguments import ProgramArguments
from agm.agl.runtime.codec import TextCodec
from tests._agl_helpers import agl_roots, prepare_inline_command


def _prepare_graph(source: str) -> PreparedProgram:
    return prepare_inline_command(
        source,
        entry_path=None,
        roots=agl_roots(),
    )


def _preflight(runtime: PipelineDriver, prepared: PreparedProgram) -> ArgumentPreflight:
    """Preflight the wrapped entry program's (empty) argument list.

    Every source in this file is a bare inline statement, wrapped into a
    zero-parameter synthetic ``program def main``. Only the resulting
    lowered executable matters here — this file tests artifact provenance,
    not argument binding.
    """
    discovery = runtime.discover_programs(prepared)
    (program,) = discovery.programs
    return runtime.preflight_arguments(
        prepared, program, ProgramArguments(positional=(), named={}), compiled=discovery.compiled
    )


def test_single_run_rejects_checked_artifact_from_different_prepared_program(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = PipelineDriver()
    prepared_a = prepare_inline_command('print "stale"')
    discovery_a = runtime.discover_programs(prepared_a)
    assert discovery_a.checked is not None
    prepared_b = prepare_inline_command('print "fresh"')

    with pytest.raises(ArtifactProvenanceError):
        runtime.run_prepared(prepared_b, checked=discovery_a.checked)

    assert capsys.readouterr().out == ""


def test_program_run_rejects_checked_artifact_from_different_prepared_program(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = PipelineDriver()
    prepared_a = _prepare_graph('print "stale"')
    discovery_a = runtime.discover_programs(prepared_a)
    assert discovery_a.checked is not None
    prepared_b = _prepare_graph('print "fresh"')

    with pytest.raises(ArtifactProvenanceError):
        runtime.run_prepared(prepared_b, checked=discovery_a.checked)

    assert capsys.readouterr().out == ""


def test_run_rejects_executable_from_different_prepared_program(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = PipelineDriver()
    prepared_a = prepare_inline_command('print "stale"')
    preflight_a = _preflight(runtime, prepared_a)
    assert preflight_a.result.ok
    assert preflight_a.executable is not None
    prepared_b = prepare_inline_command('print "fresh"')

    with pytest.raises(ArtifactProvenanceError):
        runtime.run_prepared(
            prepared_b,
            executable=preflight_a.executable,
            select_default_program=True,
        )

    assert capsys.readouterr().out == ""


def test_run_rejects_executable_issued_by_another_pipeline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    issuer = PipelineDriver()
    prepared = prepare_inline_command('print "stale"')
    preflight = _preflight(issuer, prepared)
    assert preflight.result.ok
    assert preflight.executable is not None

    with pytest.raises(ArtifactProvenanceError):
        PipelineDriver().run_prepared(
            prepared,
            executable=preflight.executable,
            select_default_program=True,
        )

    assert capsys.readouterr().out == ""


def test_run_resumes_the_executable_from_its_preflight(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = PipelineDriver()
    prepared = prepare_inline_command('print "fresh"')

    with patch("agm.agl.lower.lower_program", wraps=lower_program) as lower:
        preflight = _preflight(runtime, prepared)
        assert preflight.result.ok
        assert preflight.executable is not None

        result = runtime.run_prepared(
            prepared,
            executable=preflight.executable,
            select_default_program=True,
        )

    assert result.ok
    assert lower.call_count == 1
    assert capsys.readouterr().out == "fresh\n"


def test_run_relowers_a_preflight_executable_when_host_capabilities_change(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ExtraCodec(TextCodec):
        @property
        def name(self) -> str:
            return "extra"

    runtime = PipelineDriver()
    prepared = prepare_inline_command('print "fresh"')

    with patch("agm.agl.lower.lower_program", wraps=lower_program) as lower:
        preflight = _preflight(runtime, prepared)
        assert preflight.result.ok
        assert preflight.executable is not None
        runtime.register_codec(ExtraCodec())

        result = runtime.run_prepared(
            prepared,
            executable=preflight.executable,
            select_default_program=True,
        )

    assert result.ok
    assert lower.call_count == 2
    assert capsys.readouterr().out == "fresh\n"
