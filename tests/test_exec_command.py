"""Tests for the `agm exec` CLI command (M0).

Covers:
- CLI wires FILE argument and --input, --strict-json/--no-strict-json,
  --max-iters, --runner, --log-file, --no-log flags into ExecArgs
- Missing file exits with code 1 and prints to stderr
- Unreadable file exits with code 1 and prints error to stderr
- Valid .agl file: runtime.run is called, diagnostics printed to stderr, exits 1
  (pre-execution failure per exit-code contract, since runtime is not yet implemented)
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.exec as exec_command


class RecordedArgs(Protocol):
    def __getattr__(self, name: str) -> object: ...


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner: CliRunner, argv: list[str]) -> Result:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm", catch_exceptions=False)


@pytest.fixture()
def recorded_runs(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch ``exec.run`` to record its ExecArgs instead of executing.

    Returns the list of recorded call arguments so parser-contract tests can
    assert how CLI flags map onto ``ExecArgs`` fields.
    """
    import agm.commands.exec as exec_mod

    calls: list[object] = []

    def fake_run(args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(exec_mod, "run", fake_run)
    return calls


class TestExecArgsParsing:
    """Parser-contract tests: verify CLI flags map to ExecArgs fields."""

    def test_exec_file_argument(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file)])
        assert result.exit_code == 0

        assert len(recorded_runs) == 1
        args = recorded_runs[0]
        assert getattr(args, "file") == str(agl_file)

    def test_exec_input_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", "--input", "k=v", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "inputs") == ["k=v"]

    def test_exec_multiple_inputs(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", "--input", "a=1", "--input", "b=2", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "inputs") == ["a=1", "b=2"]

    def test_exec_strict_json_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", "--strict-json", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "strict_json") is True

    def test_exec_no_strict_json_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", "--no-strict-json", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "strict_json") is False

    def test_exec_max_iters_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", "--max-iters", "10", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "max_iters") == 10

    def test_exec_runner_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", "--runner", "claude -p", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "runner") == "claude -p"

    def test_exec_log_file_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", "--log-file", "/tmp/out.log", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "log_file") == "/tmp/out.log"

    def test_exec_no_log_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", "--no-log", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "no_log") is True


class TestExecCommandBehavior:
    """Behavior tests for the exec command run() function."""

    def test_missing_file_exits_1(self, tmp_path: Path) -> None:
        from agm.commands.args import ExecArgs

        args = ExecArgs(
            file=str(tmp_path / "nonexistent.agl"),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_missing_file_prints_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.commands.args import ExecArgs

        args = ExecArgs(
            file=str(tmp_path / "nonexistent.agl"),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit):
            exec_command.run(args)
        captured = capsys.readouterr()
        assert "Error" in captured.err or "error" in captured.err.lower()

    def test_unreadable_file_exits_1_with_friendly_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A path that is a directory (not a readable file) exits 1 with a friendly error."""
        from agm.commands.args import ExecArgs

        a_dir = tmp_path / "a_directory"
        a_dir.mkdir()

        args = ExecArgs(
            file=str(a_dir),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        # The friendly message names the offending path.
        assert "a_directory" in captured.err

    def test_valid_file_exits_0_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A valid .agl file with no agent calls exits 0 (success)."""
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        from agm.commands.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        # M1: a simple valid program succeeds
        result = exec_command.run(args)
        assert result is None  # returns None on success (exit 0)

    def test_static_error_file_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A .agl file with a static error exits 1 and prints diagnostics to stderr."""
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = undefined_name\n")
        from agm.commands.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.err


class TestExecCommandEdgePaths:
    """Cover the ok=True and error!=None branches via monkeypatching."""

    def test_ok_result_returns_normally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful run returns normally (exit 0) instead of raising SystemExit."""
        from agm.agl.runtime.runtime import RunResult, WorkflowRuntime
        from agm.commands.args import ExecArgs

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        def fake_run(self: WorkflowRuntime, source: str, *, inputs: object = None) -> RunResult:
            return RunResult(ok=True, diagnostics=[], error=None)

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run", fake_run)

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        # Falls through and returns None (no SystemExit on the success path).
        assert exec_command.run(args) is None

    def test_error_result_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.agl.runtime.runtime import RunError, RunResult, WorkflowRuntime
        from agm.commands.args import ExecArgs

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        def fake_run(self: WorkflowRuntime, source: str, *, inputs: object = None) -> RunResult:
            return RunResult(
                ok=False,
                diagnostics=[],
                error=RunError(type_name="AgentParseError", fields={}),
            )

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run", fake_run)

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "AgentParseError" in captured.err

    def test_error_result_includes_message_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The exit-2 path prints the exception's ``fields['message']`` when present."""
        from agm.agl.runtime.runtime import RunError, RunResult, WorkflowRuntime
        from agm.commands.args import ExecArgs

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        def fake_run(self: WorkflowRuntime, source: str, *, inputs: object = None) -> RunResult:
            return RunResult(
                ok=False,
                diagnostics=[],
                error=RunError(
                    type_name="AgentParseError",
                    fields={"message": "could not parse agent output"},
                ),
            )

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run", fake_run)

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "AgentParseError" in captured.err
        assert "could not parse agent output" in captured.err

    def test_bad_input_format_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.commands.args import ExecArgs

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        args = ExecArgs(
            file=str(agl_file),
            inputs=["noequals"],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestExecCommandWarnings:
    """Warning-severity diagnostics are reported but never affect the exit code."""

    def test_warning_with_ok_returns_normally_and_prints_to_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.agl.runtime.runtime import Diagnostic, RunResult, WorkflowRuntime
        from agm.commands.args import ExecArgs

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        warning = Diagnostic(message="case is non-exhaustive", line=7, severity="warning")

        def fake_run(self: WorkflowRuntime, source: str, *, inputs: object = None) -> RunResult:
            return RunResult(ok=True, diagnostics=[warning], error=None)

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run", fake_run)

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        # ok=True even with a warning: returns normally (exit 0).
        assert exec_command.run(args) is None
        captured = capsys.readouterr()
        assert "case is non-exhaustive" in captured.err
        assert "line 7" in captured.err

    def test_error_diagnostic_still_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.agl.runtime.runtime import Diagnostic, RunResult, WorkflowRuntime
        from agm.commands.args import ExecArgs

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        error = Diagnostic(message="type mismatch", line=3)

        def fake_run(self: WorkflowRuntime, source: str, *, inputs: object = None) -> RunResult:
            return RunResult(ok=False, diagnostics=[error], error=None)

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run", fake_run)

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "type mismatch" in captured.err

    def test_warning_and_error_together_exits_1_and_prints_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.agl.runtime.runtime import Diagnostic, RunResult, WorkflowRuntime
        from agm.commands.args import ExecArgs

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        warning = Diagnostic(message="unused binding", line=2, severity="warning")
        error = Diagnostic(message="unknown name", line=5)

        def fake_run(self: WorkflowRuntime, source: str, *, inputs: object = None) -> RunResult:
            return RunResult(ok=False, diagnostics=[warning, error], error=None)

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run", fake_run)

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "unused binding" in captured.err
        assert "unknown name" in captured.err


class TestExecCLIPaths:
    """Cover the CLI paths for missing FILE and --no-log/--log-file conflict."""

    def test_exec_missing_file_exits_nonzero(self, runner: CliRunner) -> None:
        result = invoke(runner, ["exec"])
        assert result.exit_code != 0

    def test_exec_no_log_and_log_file_conflict(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        # recorded_runs intercepts exec.run so we don't actually run the file.
        result = invoke(
            runner,
            ["exec", "--no-log", "--log-file", "/tmp/x.log", str(agl_file)],
        )
        assert result.exit_code != 0


class TestExecCommandM1:
    """M1 exec command behavior: exit codes, inputs, agent calls."""

    def test_valid_program_exits_0(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        from agm.commands.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        result = exec_command.run(args)
        assert result is None  # no SystemExit → exit 0

    def test_program_with_inputs_exits_0(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("input msg\nprint msg\n")
        from agm.commands.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            inputs=["msg=hello"],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        result = exec_command.run(args)
        assert result is None

    def test_missing_input_exits_1(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("input msg\nprint msg\n")
        from agm.commands.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],  # missing 'msg'
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_uncaught_agl_exception_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An uncaught AgL exception during execution → exit code 2."""
        from agm.agl.runtime.runtime import RunError, RunResult, WorkflowRuntime
        from agm.commands.args import ExecArgs

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        def fake_run(self: WorkflowRuntime, source: str, *, inputs: object = None) -> RunResult:
            return RunResult(
                ok=False,
                diagnostics=[],
                error=RunError(type_name="Abort", fields={"message": "fatal"}),
            )

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run", fake_run)

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 2

    def test_static_error_exits_1_not_2(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = undefined_name\n")
        from agm.commands.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1  # static error, not AgL exception


class TestParseKeyValue:
    """Tests for the general key=value parser helper."""

    def test_parse_valid_key_value(self) -> None:
        from agm.core.cli_helpers import parse_key_value

        key, value = parse_key_value("name=Alice")
        assert key == "name"
        assert value == "Alice"

    def test_parse_key_value_with_equals_in_value(self) -> None:
        from agm.core.cli_helpers import parse_key_value

        key, value = parse_key_value("expr=a=b")
        assert key == "expr"
        assert value == "a=b"

    def test_parse_key_value_invalid_raises(self) -> None:
        from agm.core.cli_helpers import parse_key_value

        with pytest.raises(ValueError, match="="):
            parse_key_value("noequals")

    def test_parse_key_value_empty_key_raises(self) -> None:
        from agm.core.cli_helpers import parse_key_value

        with pytest.raises(ValueError):
            parse_key_value("=value")

    def test_parse_inputs_list(self) -> None:
        from agm.core.cli_helpers import parse_inputs

        result = parse_inputs(["a=1", "b=hello"])
        assert result == {"a": "1", "b": "hello"}

    def test_parse_inputs_duplicate_key_raises(self) -> None:
        from agm.core.cli_helpers import parse_inputs

        with pytest.raises(ValueError, match="a"):
            parse_inputs(["a=1", "a=2"])
