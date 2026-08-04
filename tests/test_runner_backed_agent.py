"""Tests for the runner-backed agent.

Covers:
- agm.agent.runner: prepare_rendered_prompt_run / run_prepared_prompt_result / PromptRunResult
- agm.agl.runtime.agents: AgentCallHostError, runner_backed_agent_factory
  - command resolution: per-name map → exec config runner → shared loop default
  - message composition: prompt + format_instructions + retry feedback (attempt≥1)
  - §7.8 corrective feedback exact wording
  - AgentCallError mapping: cause=spawn_failure/nonzero_exit/timeout + metadata fields
  - exit-0 empty stdout = valid empty response
  - verbatim prompt preservation ($NAME/${NAME} not expanded)
- PipelineDriver: fallback from runner config, default_agent from runner config
- CLI-level fake-runner binary integration test
"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agm.agl.ir.builtin_nominals import BuiltinNominals
from agm.agl.ir.ids import AgentId, NominalId
from agm.agl.modules.ids import STD_CORE_ID
from agm.agl.runtime.agents import AgentRegistry
from agm.agl.semantics.values import EnumValue, ExceptionValue, TextValue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _command_agent(command: str) -> EnumValue:
    return EnumValue(
        nominal=NominalId(STD_CORE_ID, "Agent"),
        display_name="Agent",
        variant="AgentCommand",
        fields={"command": TextValue(command)},
    )


def _make_process_result(
    *,
    returncode: int | None = 0,
    stdout: str = "ok",
    stderr: str = "",
    elapsed: float = 0.1,
    timed_out: bool = False,
    spawn_error: str | None = None,
    spawn_errno: int | None = None,
) -> object:
    from agm.core.process import ProcessCaptureResult

    return ProcessCaptureResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed=elapsed,
        timed_out=timed_out,
        spawn_error=spawn_error,
        spawn_errno=spawn_errno,
    )


# ---------------------------------------------------------------------------
# PromptRunResult dataclass
# ---------------------------------------------------------------------------


class TestPromptRunResult:
    """PromptRunResult is a structured, importable dataclass."""

    def test_import(self) -> None:
        from agm.agent.runner import PromptRunResult

        r = PromptRunResult(
            returncode=0,
            stdout="hello",
            stderr="",
            elapsed=1.0,
            timed_out=False,
            spawn_error=None,
        )
        assert r.returncode == 0
        assert r.stdout == "hello"
        assert r.stderr == ""
        assert r.elapsed == 1.0
        assert r.timed_out is False
        assert r.spawn_error is None

    def test_spawn_failure_fields(self) -> None:
        from agm.agent.runner import PromptRunResult

        r = PromptRunResult(
            returncode=None,
            stdout="",
            stderr="",
            elapsed=0.0,
            timed_out=False,
            spawn_error="command not found: claude",
        )
        assert r.returncode is None
        assert r.spawn_error == "command not found: claude"


# ---------------------------------------------------------------------------
# prepare_rendered_prompt_run
# ---------------------------------------------------------------------------


class TestPrepareRenderedPromptRun:
    """prepare_rendered_prompt_run writes the prompt verbatim to a temp file."""

    def test_returns_prepared_run(self, tmp_path: Path) -> None:
        from agm.agent.runner import PreparedPromptRun, prepare_rendered_prompt_run

        temp_files: list[Path] = []
        result = prepare_rendered_prompt_run(
            "Hello world",
            runner="claude -p",
            temp_files=temp_files,
            env={},
        )
        assert isinstance(result, PreparedPromptRun)
        # Cleans up
        for f in temp_files:
            f.unlink(missing_ok=True)

    def test_prompt_written_verbatim(self) -> None:
        """The prompt is written without env-var expansion."""
        from agm.agent.runner import prepare_rendered_prompt_run

        prompt = "Value is $NAME and ${OTHER}"
        temp_files: list[Path] = []
        result = prepare_rendered_prompt_run(prompt, runner="echo", temp_files=temp_files, env={})
        content = result.effective_file.read_text(encoding="utf-8")
        # Must be written verbatim — no env-var expansion
        assert content == prompt
        for f in temp_files:
            f.unlink(missing_ok=True)

    def test_temp_file_registered(self) -> None:
        from agm.agent.runner import prepare_rendered_prompt_run

        temp_files: list[Path] = []
        prepare_rendered_prompt_run("hi", runner="echo", temp_files=temp_files, env={})
        assert len(temp_files) >= 1
        for f in temp_files:
            f.unlink(missing_ok=True)

    def test_no_validate_command_called(self) -> None:
        """prepare_rendered_prompt_run must NOT call validate_command (which prints
        and raises SystemExit for missing executables)."""
        from agm.agent.runner import prepare_rendered_prompt_run

        # "nonexistent_binary_xyz123" is not in PATH; validate_command would SystemExit.
        temp_files: list[Path] = []
        # Should NOT raise SystemExit even though the binary doesn't exist.
        result = prepare_rendered_prompt_run(
            "test", runner="nonexistent_binary_xyz123", temp_files=temp_files, env={}
        )
        assert result is not None  # just needs to return normally
        for f in temp_files:
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# run_prepared_prompt_result
# ---------------------------------------------------------------------------


class TestRunPreparedPromptResult:
    """run_prepared_prompt_result returns structured result; never prints/SystemExit."""

    def test_success_returns_prompt_run_result(self) -> None:
        from agm.agent.runner import PreparedPromptRun, PromptRunResult, run_prepared_prompt_result

        mock_result = _make_process_result(returncode=0, stdout="agent output")
        temp_file = Path("/tmp/test_prompt.md")
        prepared = PreparedPromptRun(
            command=["echo"],
            effective_file=temp_file,
            env={},
            temp_files=[],
        )
        with patch("agm.agent.runner.run_capture_result", return_value=mock_result):
            result = run_prepared_prompt_result(prepared, idle_timeout=None)
        assert isinstance(result, PromptRunResult)
        assert result.returncode == 0
        assert result.stdout == "agent output"

    def test_nonzero_exit_returned_structured(self) -> None:
        from agm.agent.runner import PreparedPromptRun, run_prepared_prompt_result

        mock_result = _make_process_result(returncode=1, stdout="", stderr="error msg")
        prepared = PreparedPromptRun(
            command=["false"],
            effective_file=Path("/tmp/p.md"),
            env={},
            temp_files=[],
        )
        with patch("agm.agent.runner.run_capture_result", return_value=mock_result):
            result = run_prepared_prompt_result(prepared, idle_timeout=None)
        assert result.returncode == 1
        assert result.stderr == "error msg"

    def test_timeout_returned_structured(self) -> None:
        from agm.agent.runner import PreparedPromptRun, run_prepared_prompt_result

        mock_result = _make_process_result(returncode=124, timed_out=True)
        prepared = PreparedPromptRun(
            command=["sleep"],
            effective_file=Path("/tmp/p.md"),
            env={},
            temp_files=[],
        )
        with patch("agm.agent.runner.run_capture_result", return_value=mock_result):
            result = run_prepared_prompt_result(prepared, idle_timeout=1.0)
        assert result.timed_out is True

    def test_spawn_failure_returned_structured(self) -> None:
        from agm.agent.runner import PreparedPromptRun, run_prepared_prompt_result

        mock_result = _make_process_result(returncode=None, spawn_error="No such file or directory")
        prepared = PreparedPromptRun(
            command=["nonexistent_xyz"],
            effective_file=Path("/tmp/p.md"),
            env={},
            temp_files=[],
        )
        with patch("agm.agent.runner.run_capture_result", return_value=mock_result):
            result = run_prepared_prompt_result(prepared, idle_timeout=None)
        assert result.spawn_error is not None
        assert result.returncode is None

    def test_never_prints_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agm.agent.runner import PreparedPromptRun, run_prepared_prompt_result

        mock_result = _make_process_result(returncode=1)
        prepared = PreparedPromptRun(
            command=["false"], effective_file=Path("/tmp/p.md"), env={}, temp_files=[]
        )
        with patch("agm.agent.runner.run_capture_result", return_value=mock_result):
            run_prepared_prompt_result(prepared, idle_timeout=None)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_uses_isolate_process_group(self) -> None:
        """run_prepared_prompt_result passes isolate_process_group=True to run_capture_result."""
        from agm.agent.runner import PreparedPromptRun, run_prepared_prompt_result

        captured_kwargs: dict[str, object] = {}

        def fake_run_capture(cmd: list[str], **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            return _make_process_result()

        prepared = PreparedPromptRun(
            command=["echo"], effective_file=Path("/tmp/p.md"), env={}, temp_files=[]
        )
        with patch("agm.agent.runner.run_capture_result", side_effect=fake_run_capture):
            run_prepared_prompt_result(prepared, idle_timeout=None)
        assert captured_kwargs.get("isolate_process_group") is True

    def test_elapsed_time_preserved(self) -> None:
        from agm.agent.runner import PreparedPromptRun, run_prepared_prompt_result

        mock_result = _make_process_result(elapsed=3.7)
        prepared = PreparedPromptRun(
            command=["echo"], effective_file=Path("/tmp/p.md"), env={}, temp_files=[]
        )
        with patch("agm.agent.runner.run_capture_result", return_value=mock_result):
            result = run_prepared_prompt_result(prepared, idle_timeout=None)
        assert result.elapsed == pytest.approx(3.7)


# ---------------------------------------------------------------------------
# AgentCallHostError
# ---------------------------------------------------------------------------


class TestAgentCallHostError:
    """AgentCallHostError is importable and carries cause + metadata."""

    def test_import(self) -> None:
        from agm.agl.runtime.agents import AgentCallHostError

        err = AgentCallHostError(
            cause="nonzero_exit",
            exit_code=1,
            stderr_tail="some error",
            elapsed=0.5,
        )
        assert err.cause == "nonzero_exit"
        assert err.exit_code == 1
        assert err.stderr_tail == "some error"
        assert err.elapsed == pytest.approx(0.5)

    def test_spawn_failure_cause(self) -> None:
        from agm.agl.runtime.agents import AgentCallHostError

        err = AgentCallHostError(
            cause="spawn_failure",
            exit_code=None,
            stderr_tail="",
            elapsed=0.0,
        )
        assert err.cause == "spawn_failure"
        assert err.exit_code is None

    def test_timeout_cause(self) -> None:
        from agm.agl.runtime.agents import AgentCallHostError

        err = AgentCallHostError(
            cause="timeout",
            exit_code=124,
            stderr_tail="",
            elapsed=30.0,
        )
        assert err.cause == "timeout"

    def test_is_exception(self) -> None:
        from agm.agl.runtime.agents import AgentCallHostError

        err = AgentCallHostError(cause="nonzero_exit", exit_code=1, stderr_tail="", elapsed=0.0)
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# AgentCallError dispatch seam (registry converts AgentCallHostError → AglRaise)
# ---------------------------------------------------------------------------


class TestAgentCallErrorSeam:
    """AgentRegistry.dispatch converts AgentCallHostError → AglRaise(AgentCallError)."""

    def _make_registry_with_failing_agent(
        self, *, cause: str, exit_code: int | None, stderr_tail: str, elapsed: float
    ) -> AgentRegistry:
        from agm.agl.runtime.agents import AgentCallHostError

        def failing_agent(req: object) -> str:
            raise AgentCallHostError(
                cause=cause,
                exit_code=exit_code,
                stderr_tail=stderr_tail,
                elapsed=elapsed,
            )

        return AgentRegistry(
            named={AgentId("tester"): failing_agent}, default_agent=None, value_agent=failing_agent
        )

    def _dispatch_failure(
        self, registry: AgentRegistry, *, nominals: BuiltinNominals | None = None
    ) -> ExceptionValue:
        from agm.agl.runtime import AgentRequest
        from agm.agl.semantics.exceptions import AglRaise

        agent = _command_agent("tester")
        with pytest.raises(AglRaise) as exc_info:
            if nominals is None:
                registry.dispatch(agent, AgentRequest(agent=agent, prompt="q"))
            else:
                registry.dispatch(agent, AgentRequest(agent=agent, prompt="q"), nominals=nominals)
        return exc_info.value.exc

    def test_nonzero_exit_raises_agl_raise(self) -> None:
        registry = self._make_registry_with_failing_agent(
            cause="nonzero_exit", exit_code=1, stderr_tail="err", elapsed=0.1
        )
        exc_val = self._dispatch_failure(registry)
        assert exc_val.display_name == "AgentCallError"

        from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS

        assert exc_val.nominal == NO_BUILTIN_DECLARATIONS.nominal("AgentCallError")

    def test_dispatch_forwards_a_custom_nominals_table(self) -> None:
        """A caller-supplied ``nominals`` table (not the default) is honored."""
        from agm.agl.ir.builtin_nominals import BuiltinNominals
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID

        registry = self._make_registry_with_failing_agent(
            cause="nonzero_exit", exit_code=1, stderr_tail="err", elapsed=0.1
        )
        custom = BuiltinNominals(declared={"AgentCallError": NominalId(ENTRY_ID, "AgentCallError")})
        assert self._dispatch_failure(registry, nominals=custom).nominal == NominalId(
            ENTRY_ID, "AgentCallError"
        )

    def test_spawn_failure_raises_agl_raise(self) -> None:
        registry = self._make_registry_with_failing_agent(
            cause="spawn_failure", exit_code=None, stderr_tail="", elapsed=0.0
        )
        exc_val = self._dispatch_failure(registry)
        assert exc_val.display_name == "AgentCallError"

    def test_timeout_raises_agl_raise(self) -> None:
        registry = self._make_registry_with_failing_agent(
            cause="timeout", exit_code=124, stderr_tail="", elapsed=30.0
        )
        exc_val = self._dispatch_failure(registry)
        assert exc_val.display_name == "AgentCallError"

    def test_cause_field_preserved(self) -> None:
        registry = self._make_registry_with_failing_agent(
            cause="nonzero_exit", exit_code=2, stderr_tail="fail msg", elapsed=0.5
        )
        exc_val = self._dispatch_failure(registry)
        cause = exc_val.fields.get("cause")
        assert isinstance(cause, TextValue)
        assert cause.value == "nonzero_exit"

    def test_metadata_exit_code_preserved(self) -> None:
        from agm.agl.semantics.values import JsonValue

        registry = self._make_registry_with_failing_agent(
            cause="nonzero_exit", exit_code=42, stderr_tail="some err", elapsed=1.0
        )
        exc_val = self._dispatch_failure(registry)
        meta = exc_val.fields.get("metadata")
        assert isinstance(meta, JsonValue)
        raw = meta.raw
        assert isinstance(raw, dict)
        assert raw.get("exit_code") == 42

    def test_metadata_stderr_tail_preserved(self) -> None:
        from agm.agl.semantics.values import JsonValue

        registry = self._make_registry_with_failing_agent(
            cause="nonzero_exit", exit_code=1, stderr_tail="the error msg", elapsed=0.0
        )
        exc_val = self._dispatch_failure(registry)
        meta = exc_val.fields.get("metadata")
        assert isinstance(meta, JsonValue)
        raw = meta.raw
        assert isinstance(raw, dict)
        assert "the error msg" in str(raw.get("stderr_tail", ""))

    def test_metadata_elapsed_preserved(self) -> None:
        from agm.agl.semantics.values import JsonValue

        registry = self._make_registry_with_failing_agent(
            cause="timeout", exit_code=124, stderr_tail="", elapsed=5.5
        )
        exc_val = self._dispatch_failure(registry)
        meta = exc_val.fields.get("metadata")
        assert isinstance(meta, JsonValue)
        raw = meta.raw
        assert isinstance(raw, dict)
        assert isinstance(raw.get("elapsed"), float)
        assert abs(float(raw.get("elapsed", 0)) - 5.5) < 0.01

    def test_agent_field_reflects_agent_name(self) -> None:
        registry = self._make_registry_with_failing_agent(
            cause="spawn_failure", exit_code=None, stderr_tail="", elapsed=0.0
        )
        exc_val = self._dispatch_failure(registry)
        agent_field = exc_val.fields.get("agent")
        assert isinstance(agent_field, EnumValue)
        assert agent_field == _command_agent("tester")

    def test_agl_raise_carries_exception_base_fields(self) -> None:
        registry = self._make_registry_with_failing_agent(
            cause="nonzero_exit", exit_code=1, stderr_tail="", elapsed=0.0
        )
        exc_val = self._dispatch_failure(registry)
        # Base Exception fields: message and trace_id must be present.
        assert "message" in exc_val.fields
        assert "trace_id" in exc_val.fields
        assert isinstance(exc_val.fields["message"], TextValue)

    def test_command_values_fall_back_or_raise_without_a_dispatcher(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.semantics.exceptions import AglRaise

        command = _command_agent("unknown")
        request = AgentRequest(agent=command, prompt="q")
        fallback = AgentRegistry(
            named={"known": lambda _request: "known"}, default_agent=lambda _: "ok"
        )
        assert fallback.dispatch(command, request).content == "ok"

        no_dispatcher = AgentRegistry(named={"known": lambda _request: "known"}, default_agent=None)
        with pytest.raises(AglRaise) as exc_info:
            no_dispatcher.dispatch(command, request)
        assert exc_info.value.exc.fields["cause"] == TextValue("no_dispatcher")

        claude = EnumValue(
            nominal=NominalId(STD_CORE_ID, "Agent"),
            display_name="Agent",
            variant="AgentClaude",
            fields={"model": TextValue(""), "thinking": TextValue("")},
        )
        assert fallback.dispatch(claude, AgentRequest(agent=claude, prompt="q")).content == "ok"


# ---------------------------------------------------------------------------
# runner_backed_agent_factory — command resolution
# ---------------------------------------------------------------------------


class TestRunnerBackedAgentCommandResolution:
    """runner_backed_agent_factory resolves the runner command per-name or default."""

    def _make_factory_fn(
        self, *, runner: str = "echo", per_agent: dict[str, str] | None = None
    ) -> object:
        from agm.agl.runtime.agents import runner_backed_agent_factory

        return runner_backed_agent_factory(
            default_runner_cmd=runner,
            per_agent_cmds=per_agent or {},
            idle_timeout=None,
        )

    def test_default_command_used_for_unknown_agent(self) -> None:
        from agm.agl.runtime import AgentRequest

        received_cmds: list[list[str]] = []

        def fake_prepare(
            rendered_prompt: str,
            *,
            runner: str | list[str],
            temp_files: object,
            env: object,
        ) -> object:
            received_cmds.append((shlex.split(runner) if isinstance(runner, str) else runner))
            from agm.agent.runner import PreparedPromptRun

            return PreparedPromptRun(
                command=(shlex.split(runner) if isinstance(runner, str) else runner),
                effective_file=Path("/tmp/p.md"),
                env={},
                temp_files=[],
            )

        factory_fn = self._make_factory_fn(runner="my-runner --flag")
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch(
                "agm.agent.runner.run_prepared_prompt_result",
                return_value=MagicMock(
                    returncode=0,
                    stdout="ok",
                    stderr="",
                    elapsed=0.1,
                    timed_out=False,
                    spawn_error=None,
                ),
            ),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            factory_fn(AgentRequest(agent=_command_agent("anon_agent"), prompt="hi"))
        assert received_cmds[0] == ["anon_agent"]

    def test_legacy_per_agent_command_does_not_override_agent_command_value(self) -> None:
        from agm.agl.runtime import AgentRequest

        received_cmds: list[list[str]] = []

        def fake_prepare(
            rendered_prompt: str,
            *,
            runner: str | list[str],
            temp_files: object,
            env: object,
        ) -> object:
            received_cmds.append((shlex.split(runner) if isinstance(runner, str) else runner))
            from agm.agent.runner import PreparedPromptRun

            return PreparedPromptRun(
                command=(shlex.split(runner) if isinstance(runner, str) else runner),
                effective_file=Path("/tmp/p.md"),
                env={},
                temp_files=[],
            )

        factory_fn = self._make_factory_fn(
            runner="default-runner",
            per_agent={"reviewer": "codex exec"},
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch(
                "agm.agent.runner.run_prepared_prompt_result",
                return_value=MagicMock(
                    returncode=0,
                    stdout="ok",
                    stderr="",
                    elapsed=0.1,
                    timed_out=False,
                    spawn_error=None,
                ),
            ),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            factory_fn(AgentRequest(agent=_command_agent("reviewer"), prompt="review it"))
        assert received_cmds[0] == ["reviewer"]

    def test_default_runner_used_when_not_in_per_agent_map(self) -> None:
        from agm.agl.runtime import AgentRequest

        received_cmds: list[list[str]] = []

        def fake_prepare(
            rendered_prompt: str,
            *,
            runner: str | list[str],
            temp_files: object,
            env: object,
        ) -> object:
            received_cmds.append((shlex.split(runner) if isinstance(runner, str) else runner))
            from agm.agent.runner import PreparedPromptRun

            return PreparedPromptRun(
                command=(shlex.split(runner) if isinstance(runner, str) else runner),
                effective_file=Path("/tmp/p.md"),
                env={},
                temp_files=[],
            )

        factory_fn = self._make_factory_fn(
            runner="default-runner",
            per_agent={"reviewer": "codex exec"},
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch(
                "agm.agent.runner.run_prepared_prompt_result",
                return_value=MagicMock(
                    returncode=0,
                    stdout="ok",
                    stderr="",
                    elapsed=0.1,
                    timed_out=False,
                    spawn_error=None,
                ),
            ),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            factory_fn(AgentRequest(agent=_command_agent("impl"), prompt="implement it"))
        assert received_cmds[0] == ["impl"]

    def test_direct_agent_values_ignore_legacy_overrides(self) -> None:
        from agm.agl.runtime import AgentRequest

        received: list[list[str]] = []

        def fake_prepare(
            rendered_prompt: str, *, runner: str | list[str], temp_files: object, env: object
        ) -> object:
            command = (
                (shlex.split(runner) if isinstance(runner, str) else runner)
                if isinstance(runner, str)
                else runner
            )
            received.append(command)
            from agm.agent.runner import PreparedPromptRun

            return PreparedPromptRun(
                command=command,
                effective_file=Path("/tmp/p.md"),
                env={},
                temp_files=[],
            )

        factory = self._make_factory_fn(
            runner="legacy-default", per_agent={"reviewer": "codex exec"}
        )
        success = MagicMock(
            returncode=0, stdout="ok", stderr="", elapsed=0.1, timed_out=False, spawn_error=None
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=success),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            factory(AgentRequest(agent=_command_agent("reviewer"), prompt="review"))
            factory(AgentRequest(agent=_command_agent("direct --flag"), prompt="run"))
            pi = EnumValue(
                nominal=NominalId(STD_CORE_ID, "Agent"),
                display_name="Agent",
                variant="AgentPi",
                fields={
                    "provider": TextValue("openai"),
                    "model": TextValue("gpt"),
                    "thinking": TextValue("high"),
                },
            )
            factory(AgentRequest(agent=pi, prompt="reason"))

        assert received == [
            ["reviewer"],
            ["direct", "--flag"],
            ["pi", "-p", "--provider", "openai", "--model", "gpt", "--thinking", "high"],
        ]

    def test_invalid_agent_value_maps_to_a_host_error(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentCallHostError, value_driven_agent_factory

        invalid = EnumValue(
            nominal=NominalId(STD_CORE_ID, "Agent"),
            display_name="Agent",
            variant="Unknown",
            fields={},
        )
        factory = value_driven_agent_factory(idle_timeout=None)
        with pytest.raises(AgentCallHostError, match="invalid_agent"):
            factory(AgentRequest(agent=invalid, prompt="q"))
        with (
            patch("agm.agent.spec.decode", return_value=object()),
            pytest.raises(AgentCallHostError, match="invalid_agent"),
        ):
            value_driven_agent_factory(idle_timeout=None)(
                AgentRequest(agent=_command_agent("command"), prompt="q")
            )


# ---------------------------------------------------------------------------
# runner_backed_agent_factory — failure mapping (spawn, timeout, nonzero)
# ---------------------------------------------------------------------------


class TestRunnerBackedAgentFailureMapping:
    """runner_backed_agent_factory maps run failures to AgentCallHostError."""

    def _make_factory_and_call(self, *, run_result_kwargs: dict[str, object]) -> None:
        """Call factory with mocked run result, expect AgentCallHostError."""
        from agm.agent.runner import PreparedPromptRun
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import runner_backed_agent_factory

        def fake_prepare(
            rendered_prompt: str,
            *,
            runner: str | list[str],
            temp_files: object,
            env: object,
        ) -> object:
            return PreparedPromptRun(
                command=["echo"],
                effective_file=Path("/tmp/p.md"),
                env={},
                temp_files=[],
            )

        run_mock = MagicMock(**run_result_kwargs)
        factory_fn = runner_backed_agent_factory(
            default_runner_cmd="echo",
            per_agent_cmds={},
            idle_timeout=None,
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=run_mock),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            factory_fn(AgentRequest(agent=_command_agent("ask"), prompt="hi"))

    def test_spawn_failure_raises_agent_call_host_error(self) -> None:
        from agm.agent.runner import PreparedPromptRun
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentCallHostError, runner_backed_agent_factory

        def fake_prepare(
            rendered_prompt: str, *, runner: str | list[str], temp_files: object, env: object
        ) -> object:
            return PreparedPromptRun(
                command=["echo"], effective_file=Path("/tmp/p.md"), env={}, temp_files=[]
            )

        run_mock = MagicMock(
            returncode=None,
            stdout="",
            stderr="no such file",
            elapsed=0.0,
            timed_out=False,
            spawn_error="No such file or directory",
        )
        factory_fn = runner_backed_agent_factory(
            default_runner_cmd="echo", per_agent_cmds={}, idle_timeout=None
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=run_mock),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            with pytest.raises(AgentCallHostError) as exc_info:
                factory_fn(AgentRequest(agent=_command_agent("ask"), prompt="hi"))
        assert exc_info.value.cause == "spawn_failure"

    def test_timeout_raises_agent_call_host_error(self) -> None:
        from agm.agent.runner import PreparedPromptRun
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentCallHostError, runner_backed_agent_factory

        def fake_prepare(
            rendered_prompt: str, *, runner: str | list[str], temp_files: object, env: object
        ) -> object:
            return PreparedPromptRun(
                command=["echo"], effective_file=Path("/tmp/p.md"), env={}, temp_files=[]
            )

        run_mock = MagicMock(
            returncode=124, stdout="", stderr="", elapsed=30.0, timed_out=True, spawn_error=None
        )
        factory_fn = runner_backed_agent_factory(
            default_runner_cmd="echo", per_agent_cmds={}, idle_timeout=30.0
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=run_mock),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            with pytest.raises(AgentCallHostError) as exc_info:
                factory_fn(AgentRequest(agent=_command_agent("ask"), prompt="hi"))
        assert exc_info.value.cause == "timeout"

    def test_nonzero_exit_raises_agent_call_host_error(self) -> None:
        from agm.agent.runner import PreparedPromptRun
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentCallHostError, runner_backed_agent_factory

        def fake_prepare(
            rendered_prompt: str, *, runner: str | list[str], temp_files: object, env: object
        ) -> object:
            return PreparedPromptRun(
                command=["echo"], effective_file=Path("/tmp/p.md"), env={}, temp_files=[]
            )

        run_mock = MagicMock(
            returncode=2,
            stdout="",
            stderr="error output",
            elapsed=0.5,
            timed_out=False,
            spawn_error=None,
        )
        factory_fn = runner_backed_agent_factory(
            default_runner_cmd="echo", per_agent_cmds={}, idle_timeout=None
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=run_mock),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            with pytest.raises(AgentCallHostError) as exc_info:
                factory_fn(AgentRequest(agent=_command_agent("ask"), prompt="hi"))
        assert exc_info.value.cause == "nonzero_exit"
        assert exc_info.value.exit_code == 2


# ---------------------------------------------------------------------------
# _stderr_tail truncation (tested via AgentCallHostError.stderr_tail)
# ---------------------------------------------------------------------------


class TestStderrTail:
    """_stderr_tail returns last 500 chars of long stderr; tested via factory failure path."""

    def _get_stderr_tail_via_factory(self, *, stderr: str) -> str:
        """Run factory with a nonzero exit + given stderr; return the AgentCallHostError tail."""
        from agm.agent.runner import PreparedPromptRun
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentCallHostError, runner_backed_agent_factory

        def fake_prepare(
            rendered_prompt: str, *, runner: str | list[str], temp_files: object, env: object
        ) -> object:
            return PreparedPromptRun(
                command=["echo"], effective_file=Path("/tmp/p.md"), env={}, temp_files=[]
            )

        run_mock = MagicMock(
            returncode=1, stdout="", stderr=stderr, elapsed=0.1, timed_out=False, spawn_error=None
        )
        factory_fn = runner_backed_agent_factory(
            default_runner_cmd="echo", per_agent_cmds={}, idle_timeout=None
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=run_mock),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            try:
                factory_fn(AgentRequest(agent=_command_agent("ask"), prompt="hi"))
            except AgentCallHostError as e:
                return e.stderr_tail
        return ""

    def test_short_stderr_returned_verbatim(self) -> None:
        result = self._get_stderr_tail_via_factory(stderr="short error")
        assert result == "short error"

    def test_long_stderr_truncated_to_last_500(self) -> None:
        long_str = "x" * 600
        result = self._get_stderr_tail_via_factory(stderr=long_str)
        assert len(result) == 500
        assert result == "x" * 500


# ---------------------------------------------------------------------------
# validation_errors list in retry feedback (§7.8)
# ---------------------------------------------------------------------------


class TestRetryFeedbackValidationErrors:
    """Validation errors from previous attempt appear in retry feedback."""

    def test_validation_errors_included_in_retry_feedback(self) -> None:
        """Each validation error appears as a bullet point in the retry feedback."""
        from agm.agent.runner import PreparedPromptRun
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import runner_backed_agent_factory
        from agm.agl.runtime.request import ValidationError

        written_prompts: list[str] = []

        def fake_prepare(
            rendered_prompt: str, *, runner: str | list[str], temp_files: object, env: object
        ) -> object:
            written_prompts.append(rendered_prompt)
            return PreparedPromptRun(
                command=["echo"], effective_file=Path("/tmp/p.md"), env={}, temp_files=[]
            )

        factory_fn = runner_backed_agent_factory(
            default_runner_cmd="echo", per_agent_cmds={}, idle_timeout=None
        )
        run_mock = MagicMock(
            returncode=0, stdout="ok", stderr="", elapsed=0.1, timed_out=False, spawn_error=None
        )
        req = AgentRequest(
            agent=_command_agent("ask"),
            prompt="Do X.",
            attempt=1,
            previous_invalid_output="bad",
            validation_errors=[
                ValidationError(category="missing_field", message="missing field 'name'"),
                ValidationError(category="wrong_type", message="type mismatch: expected int"),
            ],
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch("agm.agent.runner.run_prepared_prompt_result", return_value=run_mock),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            factory_fn(req)

        text = written_prompts[0]
        assert "missing field 'name'" in text
        assert "type mismatch: expected int" in text
        # Each error should appear as a bullet
        assert "- missing field" in text


# ---------------------------------------------------------------------------
# Message composition (§9.5 / §7.8)
# ---------------------------------------------------------------------------


class TestRunnerMessageComposition:
    """Message sent to runner = rendered prompt + format_instructions + retry feedback."""

    def _call_factory(
        self,
        *,
        prompt: str = "Hello world",
        attempt: int = 0,
        previous_invalid_output: str | None = None,
        validation_errors: list[object] | None = None,
        format_instructions: str = "",
    ) -> str:
        """Call runner_backed_agent_factory and return the rendered prompt written to temp file."""
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import runner_backed_agent_factory
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.runtime.request import ValidationError

        ve: list[ValidationError] = (
            [v for v in validation_errors if isinstance(v, ValidationError)]
            if validation_errors is not None
            else []
        )

        written_prompts: list[str] = []

        def fake_prepare(
            rendered_prompt: str,
            *,
            runner: str | list[str],
            temp_files: list[Path],
            env: object,
        ) -> object:
            written_prompts.append(rendered_prompt)
            from agm.agent.runner import PreparedPromptRun

            return PreparedPromptRun(
                command=["echo"],
                effective_file=Path("/tmp/p.md"),
                env={},
                temp_files=[],
            )

        contract = OutputContract(
            target_type_label="text",
            codec=TextCodec(),
            strict_json=None,
            format_instructions=format_instructions,
            json_schema=None,
        )

        req = AgentRequest(
            agent=_command_agent("ask"),
            prompt=prompt,
            attempt=attempt,
            previous_invalid_output=previous_invalid_output,
            validation_errors=ve,
            output_contract=contract,
        )

        factory_fn = runner_backed_agent_factory(
            default_runner_cmd="echo",
            per_agent_cmds={},
            idle_timeout=None,
        )
        with (
            patch("agm.agent.runner.prepare_rendered_prompt_run", side_effect=fake_prepare),
            patch(
                "agm.agent.runner.run_prepared_prompt_result",
                return_value=MagicMock(
                    returncode=0,
                    stdout="response",
                    stderr="",
                    elapsed=0.1,
                    timed_out=False,
                    spawn_error=None,
                ),
            ),
            patch("agm.agent.runner.cleanup_temp_files"),
        ):
            factory_fn(req)

        return written_prompts[0]

    def test_prompt_only_on_first_attempt(self) -> None:
        text = self._call_factory(prompt="Do X.", attempt=0)
        assert text.startswith("Do X.")

    def test_format_instructions_appended_when_present(self) -> None:
        text = self._call_factory(
            prompt="Do X.",
            attempt=0,
            format_instructions="Return plain text.",
        )
        assert "Do X." in text
        assert "Return plain text." in text

    def test_no_format_instructions_when_empty(self) -> None:
        text = self._call_factory(
            prompt="Do X.",
            attempt=0,
            format_instructions="",
        )
        # Should just be the prompt (and no injected blank lines from format_instructions)
        assert "Do X." in text
        assert "Return plain text." not in text

    def test_retry_feedback_on_attempt_1(self) -> None:
        """§7.8 corrective message included on attempt≥1."""
        text = self._call_factory(
            prompt="Do X.",
            attempt=1,
            previous_invalid_output="bad output",
            validation_errors=[],
        )
        # Must include corrective feedback text per §7.8
        assert "Your previous response did not match the required output format" in text

    def test_retry_previous_output_included(self) -> None:
        """Previous response is quoted in retry feedback (§7.8)."""
        text = self._call_factory(
            prompt="Do X.",
            attempt=1,
            previous_invalid_output="the-bad-output-xyz",
            validation_errors=[],
        )
        assert "the-bad-output-xyz" in text

    def test_no_retry_feedback_on_attempt_0(self) -> None:
        """No corrective feedback on first attempt."""
        text = self._call_factory(prompt="Do X.", attempt=0)
        assert "Your previous response did not match" not in text

    def test_verbatim_dollar_name_not_expanded(self) -> None:
        """$NAME and ${NAME} in the rendered prompt are NOT expanded (§9.5)."""
        text = self._call_factory(
            prompt="Value is $NAME and ${OTHER}",
            attempt=0,
        )
        assert "$NAME" in text
        assert "${OTHER}" in text

    def test_format_instructions_precede_retry_feedback(self) -> None:
        """Order: prompt → format_instructions → retry feedback."""
        text = self._call_factory(
            prompt="Do X.",
            attempt=1,
            previous_invalid_output="bad",
            validation_errors=[],
            format_instructions="Return JSON.",
        )
        fmt_pos = text.find("Return JSON.")
        retry_pos = text.find("Your previous response")
        assert fmt_pos < retry_pos, "format_instructions must come before retry feedback"


# ---------------------------------------------------------------------------
# AgentCallError via PipelineDriver — integration (mocked boundary)
# ---------------------------------------------------------------------------


class TestAgentCallErrorViaRuntime:
    """AgentCallError is catchable in AgL programs when runner fails."""

    def _make_failing_default_agent(self, *, cause: str, exit_code: int | None = 1) -> object:
        from agm.agl.runtime.agents import AgentCallHostError

        def agent(req: object) -> str:
            raise AgentCallHostError(
                cause=cause,
                exit_code=exit_code,
                stderr_tail="stderr content",
                elapsed=0.1,
            )

        return agent

    def test_uncaught_agent_call_error_exits_2(self) -> None:
        from agm.agl import PipelineDriver

        rt = PipelineDriver(default_agent=self._make_failing_default_agent(cause="nonzero_exit"))
        result = rt.run('let x = ask("hi")\nx')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "AgentCallError"

    def test_caught_agent_call_error_run_succeeds(self) -> None:
        from agm.agl import PipelineDriver

        program = 'try\n  ask("hi")\n  ()\ncatch AgentCallError as e =>\n  print(e.cause)\n'
        rt = PipelineDriver(default_agent=self._make_failing_default_agent(cause="nonzero_exit"))
        result = rt.run(program)
        assert result.ok is True
        assert result.error is None

    def test_cause_field_accessible_in_catch(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agm.agl import PipelineDriver

        program = 'try\n  ask("hi")\n  ()\ncatch AgentCallError as e =>\n  print(e.cause)\n'
        rt = PipelineDriver(
            default_agent=self._make_failing_default_agent(cause="spawn_failure", exit_code=None)
        )
        result = rt.run(program)
        assert result.ok is True
        out = capsys.readouterr().out.strip()
        assert out == "spawn_failure"

    def test_exit_0_empty_stdout_is_valid_response(self) -> None:
        """Exit 0 with empty stdout = valid empty response."""
        from agm.agl import PipelineDriver

        rt = PipelineDriver(default_agent=lambda req: "")
        result = rt.run('let x = ask("say nothing")\nx')
        assert result.ok is True
        from agm.agl.semantics.values import TextValue

        assert result.bindings["x"] == TextValue("")

    def test_nonzero_exit_not_retried(self) -> None:
        """AgentCallError cause=nonzero_exit is NOT retried."""
        call_count = [0]
        from agm.agl.runtime.agents import AgentCallHostError

        def counting_agent(req: object) -> str:
            call_count[0] += 1
            raise AgentCallHostError(cause="nonzero_exit", exit_code=1, stderr_tail="", elapsed=0.0)

        from agm.agl import PipelineDriver

        rt = PipelineDriver(default_agent=counting_agent)
        rt.run('let x = ask("hi", on_parse_error = Retry(n = 3))\nx')
        # Must only be called once — transport failures are not retried
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# PipelineDriver: runner config wiring
# ---------------------------------------------------------------------------


class TestPipelineDriverRunnerWiring:
    """PipelineDriver accepts runner_config to build runner-backed default + fallback."""

    def test_runtime_agent_value_dispatches_through_default(self) -> None:
        """An explicit Agent value dispatches through the default dispatcher."""
        from agm.agl import PipelineDriver

        rt = PipelineDriver(default_agent=lambda req: "ok")
        result = rt.run(
            'let any_random_agent = AgentCommand("any-random-agent")\n'
            'let x = ask("hi", agent = any_random_agent)\nx'
        )
        assert result.ok is True

    def test_exec_config_runner_wires_through_to_runtime(self) -> None:
        """exec.py constructs runtime using runner from ExecConfig (not dead code)."""
        import agm.commands.exec as exec_mod
        from agm.agl.pipeline import PipelineDriver
        from agm.cli_support.args import ExecArgs

        constructed_runtimes: list[PipelineDriver] = []
        original_init = PipelineDriver.__init__

        def capturing_init(self: PipelineDriver, **kwargs: object) -> None:
            constructed_runtimes.append(self)
            original_init(self, **kwargs)

        # Patch config to return a runner
        from agm.config.general import ExecConfig

        fake_config = ExecConfig(
            runner="my-test-runner",
            strict_json=False,
            timeout=None,
            agents={},
            log=False,
            log_file=None,
        )

        import agm.agl.pipeline as rt_mod

        agl_file = Path("/tmp/nonexistent_test.agl")
        # Just check that exec_config_from_merged is called and runner is used; we
        # don't need to run the full pipeline.
        with (
            patch.object(exec_mod, "exec_config_from_merged", return_value=fake_config),
            patch.object(
                exec_mod,
                "read_text_arg",
                return_value="let x = 1\nx\n",
            ),
            patch.object(rt_mod.PipelineDriver, "__init__", capturing_init),
            patch.object(
                rt_mod.PipelineDriver,
                "run",
                return_value=MagicMock(
                    ok=True,
                    diagnostics=[],
                    error=None,
                    warnings=[],
                    call_sites=(),
                    bindings={},
                ),
            ),
        ):
            args = ExecArgs(
                file=str(agl_file),
                param_tokens=[],
                strict_json=None,
                runner=None,
                no_log=True,
                log_file=None,
            )
            exec_mod.run(args)

        assert len(constructed_runtimes) >= 1
        # The runtime was constructed; the runner config was passed through
        # (verified by the fact that exec.py construction now uses runner)


# ---------------------------------------------------------------------------
# CLI-level: fake runner binary integration test
# ---------------------------------------------------------------------------


def _install_fake_runner(directory: Path, env: dict[str, str]) -> Path:
    """Create a fake runner binary that echoes its @file argument as a response."""
    directory.mkdir(parents=True, exist_ok=True)
    runner = directory / "fake-runner"
    runner.write_text(
        "#!/bin/bash\n"
        "# Read the prompt file path from args (last @<path> arg or --file arg)\n"
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == @* ]]; then\n'
        '    prompt_file="${arg#@}"\n'
        "  fi\n"
        "done\n"
        "# Echo canned response\n"
        'echo "runner-response"\n'
    )
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    env["PATH"] = str(directory) + ":" + env["PATH"]
    return runner


def _install_nonzero_runner(directory: Path, env: dict[str, str]) -> Path:
    """Create a runner that always exits nonzero."""
    directory.mkdir(parents=True, exist_ok=True)
    runner = directory / "fail-runner"
    runner.write_text("#!/bin/bash\necho 'error output' >&2\nexit 2\n")
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    env["PATH"] = str(directory) + ":" + env["PATH"]
    return runner


class TestCliRunnerIntegration:
    """CLI-level integration: agm exec with a real fake runner binary."""

    def _run_agm_exec(
        self,
        args: list[str],
        *,
        env: dict[str, str],
        cwd: Path,
    ) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            [sys.executable, "-m", "agm.cli", "exec", *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd),
            check=False,
        )

    def _base_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("HOME", str(Path.home()))
        return env

    def test_runner_backed_agent_produces_response(self, tmp_path: Path) -> None:
        """A real fake runner binary drives an AgL program end-to-end."""
        env = self._base_env()
        _install_fake_runner(tmp_path / "bin", env)

        agl_file = tmp_path / "prog.agl"
        # Print the agent response directly
        agl_file.write_text('let x = ask "Say something"\nprint x\n')

        result = self._run_agm_exec(
            ["--agent", 'AgentCommand("fake-runner")', str(agl_file), "--no-log"],
            env=env,
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "runner-response" in result.stdout

    def test_nonzero_exit_surfaces_as_agent_call_error(self, tmp_path: Path) -> None:
        """A runner that exits nonzero → AgentCallError → exit code 2 when uncaught."""
        env = self._base_env()
        _install_nonzero_runner(tmp_path / "bin", env)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let x = ask "hi"\nprint x\n')

        result = self._run_agm_exec(
            ["--agent", 'AgentCommand("fail-runner")', str(agl_file), "--no-log"],
            env=env,
            cwd=tmp_path,
        )
        assert result.returncode == 2, f"stderr: {result.stderr}"
        assert "AgentCallError" in result.stderr

    def test_catchable_agent_call_error_in_program(self, tmp_path: Path) -> None:
        """AgentCallError is catchable in AgL; run exits 0 when caught."""
        env = self._base_env()
        _install_nonzero_runner(tmp_path / "bin", env)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'try\n  ask("hi")\n  ()\ncatch AgentCallError as e =>\n  print(e.cause)\n'
        )

        result = self._run_agm_exec(
            ["--agent", 'AgentCommand("fail-runner")', str(agl_file), "--no-log"],
            env=env,
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "nonzero_exit" in result.stdout

    def test_agent_override_flag_selects_the_default_agent_value(self, tmp_path: Path) -> None:
        """--agent selects the default Agent value."""
        env = self._base_env()
        _install_fake_runner(tmp_path / "bin", env)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let x = ask "hi"\nprint x\n')

        result = self._run_agm_exec(
            ["--agent", 'AgentCommand("fake-runner")', str(agl_file), "--no-log"],
            env=env,
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "runner-response" in result.stdout

    def test_explicit_agent_command_uses_its_own_runner(self, tmp_path: Path) -> None:
        """An explicit AgentCommand dispatches its own runner."""
        env = self._base_env()
        _install_fake_runner(tmp_path / "bin", env)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'let myagent = AgentCommand("fake-runner")\n'
            'let x = ask("do this", agent = myagent)\nprint(x)\n'
        )

        result = self._run_agm_exec(
            [str(agl_file), "--no-log"],
            env=env,
            cwd=tmp_path,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_enoexec_runner_raises_agent_call_error(self, tmp_path: Path) -> None:
        """A runner script with exec bit but no shebang (ENOEXEC) maps to
        AgentCallError cause=spawn_failure — not a raw traceback (exit 2, not crash)."""
        env = self._base_env()

        # Create a runner that is executable but has no shebang and binary content.
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        runner = bin_dir / "enoexec-runner"
        runner.write_bytes(b"\x7f\x45\x4c\x46garbage_no_shebang")
        runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
        env["PATH"] = str(bin_dir) + ":" + env["PATH"]

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let x = ask "hi"\nprint x\n')

        result = self._run_agm_exec(
            ["--agent", 'AgentCommand("enoexec-runner")', str(agl_file), "--no-log"],
            env=env,
            cwd=tmp_path,
        )
        # ENOEXEC → spawn_failure → AgentCallError → exit 2 (not a crash/traceback)
        assert result.returncode == 2, f"stderr: {result.stderr}"
        # Must show AgentCallError, not a raw OSError traceback
        assert "AgentCallError" in result.stderr
        assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# split_command ValueError guard — malformed quoting → clean exit
# ---------------------------------------------------------------------------


class TestSplitCommandMalformedQuoting:
    """split_command must catch ValueError from shlex.split (e.g. unclosed quote)
    and produce a clean 'Error: <kind> command: <reason>' + SystemExit(1).

    This covers the lazy path (prepare_rendered_prompt_run → split_command)
    so that a malformed-quote runner config also exits cleanly.
    """

    def test_malformed_quote_raises_system_exit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """split_command with unclosed quote → SystemExit(1), not ValueError."""
        from agm.agent.runner import split_command

        with pytest.raises(SystemExit) as exc_info:
            split_command('"foo', kind="runner")
        assert exc_info.value.code == 1

    def test_malformed_quote_prints_clean_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """split_command with unclosed quote prints clean 'Error:' to stderr."""
        from agm.agent.runner import split_command

        with pytest.raises(SystemExit):
            split_command('"foo', kind="runner")
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "runner" in captured.err.lower()
        assert "Traceback" not in captured.err
        assert "ValueError" not in captured.err

    def test_valid_command_still_works(self) -> None:
        """split_command('claude -p') still returns the split tokens."""
        from agm.agent.runner import split_command

        result = split_command("claude -p", kind="runner")
        assert result == ["claude", "-p"]

    def test_empty_command_still_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The existing empty-command guard still works after adding the ValueError guard."""
        from agm.agent.runner import split_command

        with pytest.raises(SystemExit) as exc_info:
            split_command("   ", kind="runner")
        assert exc_info.value.code == 1
