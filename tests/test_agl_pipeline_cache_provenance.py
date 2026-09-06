"""Cached match artifacts must belong to the exact prepared source consuming them.

Artifact provenance is an AgL self-check: the host layer that hands a cached
artifact back into the pipeline is the compiler's own caller, so a mismatch is a
wiring bug, not a user error.  The check runs only when self-validation is
enabled (as it is for the whole suite) and raises rather than producing a
diagnostic; with self-validation off — the production path — a cached artifact is
trusted and never re-verified.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl.matchcompile import MatchCompiledProgram, compile_program_matches
from agm.agl.modules.ids import STD_PRELUDE_ID
from agm.agl.pipeline import (
    ArtifactProvenanceError,
    PipelineDriver,
    PreparedProgram,
    ProgramDiscovery,
)
from agm.agl.typecheck.program import check_program
from tests._agl_helpers import agl_roots, prepare_inline_command


def _prepare_graph(
    source: str,
    *,
    extra_roots: frozenset[Path] = frozenset(),
) -> PreparedProgram:
    return prepare_inline_command(
        source,
        entry_path=None,
        roots=agl_roots(*extra_roots),
    )


def _compiled(source: str) -> tuple[PreparedProgram, ProgramDiscovery]:
    prepared = _prepare_graph(source)
    discovery = PipelineDriver().discover_programs(prepared)
    assert discovery.compiled is not None
    return prepared, discovery


def _change_capabilities(runtime: PipelineDriver) -> None:
    from agm.agl.runtime.codec import TextCodec

    class ExtraCodec(TextCodec):
        @property
        def name(self) -> str:
            return "extra"

    runtime.register_codec(ExtraCodec())


@pytest.mark.parametrize("check_only", [False, True])
def test_single_run_rejects_cached_artifact_from_different_prepared_program(
    check_only: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = PipelineDriver()
    prepared_a = prepare_inline_command('let a = 1\nprint "stale %{a}"')
    discovery_a = runtime.discover_programs(prepared_a)
    assert discovery_a.compiled is not None
    prepared_b = prepare_inline_command('let b = 2\nprint "fresh %{b}"')

    with pytest.raises(ArtifactProvenanceError):
        runtime.run_prepared(
            prepared_b,
            check_only=check_only,
            compiled=discovery_a.compiled,
        )

    assert capsys.readouterr().out == ""


def test_program_discovery_rejects_cached_artifact_from_different_prepared_program() -> None:
    _prepared_a, discovery_a = _compiled("let a = 1\na")
    prepared_b = _prepare_graph('let b = "b"\nb')

    with pytest.raises(ArtifactProvenanceError):
        PipelineDriver().discover_programs(prepared_b, compiled=discovery_a.compiled)


def test_program_discovery_rejects_cached_artifact_with_different_entry_identity() -> None:
    prepared, discovery = _compiled("let value = 1\nvalue")
    assert discovery.compiled is not None
    wrong_entry_checked = replace(
        discovery.compiled.checked,
        entry_id=STD_PRELUDE_ID,
    )
    wrong_entry_compiled = MatchCompiledProgram(
        checked=wrong_entry_checked,
        sites_by_module=discovery.compiled.sites_by_module,
    )

    with pytest.raises(ArtifactProvenanceError):
        PipelineDriver().discover_programs(
            prepared,
            compiled=wrong_entry_compiled,
        )


def test_program_discovery_rejects_cached_artifact_with_different_module_set(
    tmp_path: Path,
) -> None:
    (tmp_path / "helper.agl").write_text("def answer() -> int = 42\n")
    prepared_with_import = _prepare_graph(
        "import helper\nlet value = 1\nvalue",
        extra_roots=frozenset({tmp_path}),
    )
    discovery = PipelineDriver().discover_programs(prepared_with_import)
    assert discovery.compiled is not None
    prepared_without_import = _prepare_graph("let value = 2\nvalue")

    with pytest.raises(ArtifactProvenanceError):
        PipelineDriver().discover_programs(
            prepared_without_import,
            compiled=discovery.compiled,
        )


def test_single_run_rechecks_cached_artifact_when_host_capabilities_change() -> None:
    from agm.agl.runtime.codec import TextCodec

    class ExtraCodec(TextCodec):
        @property
        def name(self) -> str:
            return "extra"

    source = "let value = 1\nvalue"
    runtime = PipelineDriver()
    prepared = prepare_inline_command(source)
    discovery = runtime.discover_programs(prepared)
    assert discovery.compiled is not None

    runtime.register_codec(ExtraCodec())
    result = runtime.run_prepared(prepared, compiled=discovery.compiled)

    assert result.ok
    assert result.diagnostics == []


def test_graph_cache_derives_capability_provenance_from_checked() -> None:
    runtime = PipelineDriver()
    prepared = _prepare_graph("let value = 1\nvalue")
    assert prepared.resolved is not None
    compiled = compile_program_matches(
        check_program(prepared.resolved, runtime.host_environment().capabilities)
    ).compiled
    assert isinstance(compiled, MatchCompiledProgram)

    discovery = runtime.discover_programs(prepared, compiled=compiled)

    assert discovery.compiled is compiled


def test_graph_artifact_is_rechecked_when_host_capabilities_change() -> None:
    runtime = PipelineDriver()
    prepared = _prepare_graph("let value = 1\nvalue")
    assert prepared.resolved is not None
    compiled = compile_program_matches(
        check_program(prepared.resolved, runtime.host_environment().capabilities)
    ).compiled
    assert isinstance(compiled, MatchCompiledProgram)
    _change_capabilities(runtime)

    discovery = runtime.discover_programs(prepared, compiled=compiled)

    assert discovery.compiled is not None
    assert discovery.compiled is not compiled


def test_program_run_rechecks_prechecked_artifact_when_host_capabilities_change() -> None:
    runtime = PipelineDriver()
    prepared = _prepare_graph("let value = 1\nvalue")
    discovery = runtime.discover_programs(prepared)
    assert discovery.checked is not None
    _change_capabilities(runtime)

    run = runtime.run_prepared(prepared, check_only=True, checked=discovery.checked)

    assert run.ok


def test_program_run_rechecks_compiled_artifact_when_host_capabilities_change() -> None:
    runtime = PipelineDriver()
    prepared = _prepare_graph("let value = 1\nvalue")
    assert prepared.resolved is not None
    compiled = compile_program_matches(
        check_program(prepared.resolved, runtime.host_environment().capabilities)
    ).compiled
    assert isinstance(compiled, MatchCompiledProgram)
    _change_capabilities(runtime)

    run = runtime.run_prepared(prepared, check_only=True, compiled=compiled)

    assert run.ok


@pytest.mark.parametrize("check_only", [False, True])
def test_program_run_rejects_cached_artifact_from_different_prepared_program(
    check_only: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepared_a, discovery_a = _compiled('print "stale"')
    prepared_b = _prepare_graph('print "fresh"')

    with pytest.raises(ArtifactProvenanceError):
        PipelineDriver().run_prepared(
            prepared_b,
            check_only=check_only,
            compiled=discovery_a.compiled,
        )

    assert capsys.readouterr().out == ""


def test_prechecked_artifacts_compile_without_rechecking_single_and_graph_paths() -> None:
    runtime = PipelineDriver()
    single_prepared = prepare_inline_command("let value = 1\nvalue")
    single_discovery = runtime.discover_programs(single_prepared)
    assert single_discovery.checked is not None
    single_run = runtime.run_prepared(
        single_prepared, checked=single_discovery.checked, check_only=True
    )
    assert single_run.ok, single_run.diagnostics

    graph_prepared = _prepare_graph("let g = 1\ng")
    program_discovery = runtime.discover_programs(graph_prepared)
    assert program_discovery.checked is not None
    graph_run = runtime.run_prepared(
        graph_prepared, checked=program_discovery.checked, check_only=True
    )
    assert graph_run.ok, graph_run.diagnostics


def test_production_path_reuses_cached_artifacts_without_verifying_provenance(
    self_validation_disabled: None,
) -> None:
    """With the self-checks off, every cached-artifact seam trusts its input."""
    runtime = PipelineDriver()
    single_prepared = prepare_inline_command("let value = 1\nvalue")
    single_discovery = runtime.discover_programs(single_prepared)
    assert single_discovery.compiled is not None
    assert single_discovery.checked is not None
    assert runtime.run_prepared(
        single_prepared, compiled=single_discovery.compiled, check_only=True
    ).ok
    assert runtime.run_prepared(
        single_prepared, checked=single_discovery.checked, check_only=True
    ).ok

    graph_prepared = _prepare_graph("let g = 1\ng")
    program_discovery = runtime.discover_programs(graph_prepared)
    assert program_discovery.compiled is not None
    assert program_discovery.checked is not None
    reused = runtime.discover_programs(graph_prepared, compiled=program_discovery.compiled)
    assert reused.compiled is program_discovery.compiled
    assert runtime.run_prepared(
        graph_prepared, compiled=program_discovery.compiled, check_only=True
    ).ok
    assert runtime.run_prepared(
        graph_prepared, checked=program_discovery.checked, check_only=True
    ).ok
