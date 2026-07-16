"""Host-service reconfiguration for host-consumed ``builtin var`` settings.

Writing the ``runner``, ``log``, or ``log-file`` engine settings (via
``std.config::NAME := ...``) reflects into the live host services: ``runner``
rebuilds the default agent that unnamed ``ask`` calls dispatch through, and
``log``/``log-file`` repoint the trace store.  These tests drive the ``agm exec``
command with the agent runner subprocess mocked, plus a direct pipeline test with
a recording policy for the reconfiguration hooks.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.semantics.values import Value
from agm.cli_support.args import ExecArgs
from agm.commands import exec as exec_command

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _exec_args(
    agl_file: Path, *, no_log: bool = True, log: bool = False, log_file: str | None = None
) -> ExecArgs:
    """Build ExecArgs for *agl_file* with logging off unless overridden."""
    return ExecArgs(
        file=str(agl_file),
        param_tokens=[],
        strict_json=None,
        max_iters=None,
        runner=None,
        no_log=no_log,
        log=log,
        log_file=log_file,
    )


def _patch_runner(received_cmds: list[list[str]]) -> object:
    """Context manager stack that records the runner command of every dispatch."""
    from agm.agent.runner import PreparedPromptRun

    def fake_prepare(
        rendered_prompt: str, *, runner: str, temp_files: object, env: object
    ) -> object:
        received_cmds.append(shlex.split(runner))
        return PreparedPromptRun(
            command=shlex.split(runner),
            effective_file=Path("/tmp/p.md"),
            env={},
            temp_files=[],
        )

    return patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare)


def _ok_run_result() -> MagicMock:
    return MagicMock(
        returncode=0, stdout="ok", stderr="", elapsed=0.1, timed_out=False, spawn_error=None
    )


class TestRunnerReconfiguration:
    def test_runner_write_reconfigures_default_agent(self, tmp_path: Path) -> None:
        """A ``runner :=`` before an ``ask`` dispatches through the new command."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'import std.config\nstd.config::runner := "codex-runner"\nask("hi")\n'
        )

        received: list[list[str]] = []
        with (
            _patch_runner(received),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=_ok_run_result()),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            exec_command.run(_exec_args(agl_file))

        # The runner write precedes the ask, so only the new command dispatches.
        assert received == [["codex-runner"]]

    def test_no_runner_write_uses_default_runner(self, tmp_path: Path) -> None:
        """Without a ``runner :=`` the default runner floor still dispatches (regression)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('ask("hi")\n')

        received: list[list[str]] = []
        with (
            _patch_runner(received),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=_ok_run_result()),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            exec_command.run(_exec_args(agl_file))

        assert received == [["claude", "-p"]]


class TestTraceReconfiguration:
    def test_log_file_write_routes_trace_to_that_file(self, tmp_path: Path) -> None:
        """A mid-run ``log-file :=`` sends subsequent trace events to that file."""
        trace_path = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "import std.config\n"
            'print "before"\n'
            f'std.config::log-file := Some("{trace_path}")\n'
            'print "after"\n'
        )

        exec_command.run(_exec_args(agl_file))

        assert trace_path.exists()
        lines = [ln for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines, "no JSONL records were written to the routed trace file"
        import json

        rendered = [json.loads(ln).get("rendered") for ln in lines]
        assert "after" in rendered
        assert "before" not in rendered

    def test_log_false_disables_further_trace_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting ``log := false`` after logging was on stops later trace writes."""
        monkeypatch.setattr("agm.core.log.default_agent_files_dir", lambda: tmp_path)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "import std.config\n"
            "std.config::log := true\n"
            'print "first"\n'
            "std.config::log := false\n"
            'print "second"\n'
        )

        exec_command.run(_exec_args(agl_file))

        logs = list(tmp_path.glob("exec-*.log"))
        assert len(logs) == 1, f"expected exactly one auto-named log, got {logs}"
        text = logs[0].read_text(encoding="utf-8")
        import json

        rendered = [
            json.loads(ln).get("rendered")
            for ln in text.splitlines()
            if ln.strip() and json.loads(ln).get("kind") == "print"
        ]
        assert "first" in rendered
        assert "second" not in rendered

    def test_cli_log_flag_seeds_log_register_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--log`` seeds the ``log`` register so a read before any write sees True."""
        monkeypatch.setattr("agm.core.log.default_agent_files_dir", lambda: tmp_path)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("import std.config\nlet l = std.config::log\nprint l\n")

        exec_command.run(_exec_args(agl_file, no_log=False, log=True))

        assert capsys.readouterr().out == "true\n"

    def test_untouched_logging_still_writes(self, tmp_path: Path) -> None:
        """A program that never touches the settings logs to --log-file as before."""
        trace_path = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "hello"\n')

        exec_command.run(_exec_args(agl_file, no_log=False, log_file=str(trace_path)))

        assert trace_path.exists()
        import json

        rendered = [
            json.loads(ln).get("rendered")
            for ln in trace_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and json.loads(ln).get("kind") == "print"
        ]
        assert "hello" in rendered


class _RecordingPolicy:
    """A host policy whose hooks record every reconfiguration request."""

    def __init__(self) -> None:
        self.runner_commands: list[str] = []
        self.trace_calls: list[tuple[bool, str | None]] = []

    def build_runner(self, command: str) -> AgentFn:
        self.runner_commands.append(command)
        return lambda req: "ok"

    def resolve_trace_path(self, enabled: bool, log_file: str | None) -> Path | None:
        self.trace_calls.append((enabled, log_file))
        return None


def _run_graph_with_policy(
    source: str, *, policy: HostSettingsPolicy, seed: dict[str, Value] | None = None
) -> RunResult:
    rt = PipelineDriver()
    prepared = rt.prepare_program(
        source, entry_path=None, roots=RootSet(roots=frozenset({_STDLIB}))
    )
    result = rt.run_prepared_graph(
        prepared, host_settings_policy=policy, builtin_host_settings=seed
    )
    assert isinstance(result, RunResult)
    return result


class TestReconfigureHooks:
    def test_hooks_fire_on_host_consumed_writes(self) -> None:
        recorder = _RecordingPolicy()
        policy = HostSettingsPolicy(
            build_runner=recorder.build_runner, resolve_trace_path=recorder.resolve_trace_path
        )
        source = (
            "import std.config\n"
            'std.config::runner := "codex"\n'
            "std.config::log := true\n"
            'std.config::log-file := Some("out.jsonl")\n'
            "print 1\n"
        )
        result = _run_graph_with_policy(source, policy=policy)

        assert result.ok, f"expected success but got: {result.error!r}"
        assert recorder.runner_commands == ["codex"]
        # Both the ``log`` and ``log-file`` writes recompute the trace destination.
        assert (True, None) in recorder.trace_calls
        assert (True, "out.jsonl") in recorder.trace_calls

    def test_runner_build_value_error_becomes_agl_value_error(self) -> None:
        def failing_build(command: str) -> AgentFn:
            raise ValueError("bad runner command")

        policy = HostSettingsPolicy(
            build_runner=failing_build, resolve_trace_path=lambda enabled, log_file: None
        )
        source = 'import std.config\nstd.config::runner := "boom"\nprint 1\n'
        result = _run_graph_with_policy(source, policy=policy)

        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ValueError"
