"""Tests for the `agm exec` CLI command.

Covers:
- CLI wires FILE argument and params, --strict-json/--no-strict-json,
  --max-iters, --runner, --log-file, --no-log flags into ExecArgs
- Missing file exits with code 1 and prints to stderr
- Unreadable file exits with code 1 and prints error to stderr
- Valid .agl file: runtime.run is called, diagnostics printed to stderr, exits 1
  (pre-execution failure per exit-code contract, since runtime is not yet implemented)
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.exec as exec_command
from agm.cli_support.args import ExecArgs


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

    def test_exec_param_token_after_file(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file), "--k", "v"])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "param_tokens") == ["--k", "v"]

    def test_exec_multiple_param_tokens(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file), "--a", "1", "--b", "2"])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "param_tokens") == ["--a", "1", "--b", "2"]

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


class TestExecCommandArgParsing:
    """Parser-contract tests for the -c/--command option."""

    def test_exec_command_flag_maps_to_command(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "-c", 'print "hi"'])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "command") == 'print "hi"'
        assert getattr(args, "file") is None

    def test_exec_command_long_flag_maps_to_command(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "--command", "let x = 1"])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "command") == "let x = 1"

    def test_exec_file_and_command_are_mutually_exclusive(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", "-c", "let x = 1", str(agl_file)])
        assert result.exit_code != 0
        # run() must not be reached when the CLI rejects the combination.
        assert recorded_runs == []

    def test_exec_neither_file_nor_command_exits_nonzero(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec"])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_exec_help_without_file_prints_help(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "--help"])
        assert result.exit_code == 0
        assert "agm exec" in result.output
        assert recorded_runs == []

    def test_exec_param_before_file_is_usage_error(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "--msg", "hello"])
        assert result.exit_code != 0
        assert "program parameter options must come after the FILE argument" in result.output
        assert recorded_runs == []

    def test_exec_inline_param_token_from_file_slot(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "-c", "param msg\nprint msg", "--msg", "hello"])
        assert result.exit_code == 0
        args = recorded_runs[0]
        assert getattr(args, "file") is None
        assert getattr(args, "param_tokens") == ["--msg", "hello"]


class TestExecCommandInline:
    """Behavior tests for executing an inline -c/--command program."""

    def _command_args(
        self, command: str, *, param_tokens: list[str] | None = None
    ) -> ExecArgs:
        return ExecArgs(
            file=None,
            command=command,
            param_tokens=param_tokens or [],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
        )

    def test_inline_command_runs_and_prints(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert exec_command.run(self._command_args('print "hello"')) is None
        assert capsys.readouterr().out == "hello\n"

    def test_inline_command_with_params(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = self._command_args("param msg\nprint msg", param_tokens=["--msg", "hi"])
        assert exec_command.run(args) is None
        assert capsys.readouterr().out == "hi\n"

    def test_inline_command_static_error_exits_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(self._command_args("let x = undefined_name"))
        assert exc_info.value.code == 1
        assert capsys.readouterr().err

    def test_neither_file_nor_command_exits_1(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Calling run() with neither source set fails cleanly (defensive guard)."""
        args = ExecArgs(
            file=None,
            command=None,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        assert "Error" in capsys.readouterr().err


class TestExecDynamicHelp:
    def test_exec_help_for_file_includes_discovered_params(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('program demo\nparam msg: text = "hi"\nprint msg\n')

        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=str(agl_file), command=None)

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Program parameters:" in out
        assert "--msg" in out

    def test_exec_help_for_inline_command_includes_discovered_params(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=None, command='param count: int = 1\nprint count')

        assert exc_info.value.code == 0
        assert "--count" in capsys.readouterr().out

    def test_exec_help_for_source_without_params_has_no_param_section(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=None, command='print "hi"')

        assert exc_info.value.code == 0
        assert "Program parameters:" not in capsys.readouterr().out

    def test_exec_help_for_unreadable_file_degrades(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=str(tmp_path / "missing.agl"), command=None)

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "agm exec" in out
        assert "Program parameters:" not in out


class TestExecCommandBehavior:
    """Behavior tests for the exec command run() function."""

    def test_missing_file_exits_1(self, tmp_path: Path) -> None:
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(tmp_path / "nonexistent.agl"),
            param_tokens=[],
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
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(tmp_path / "nonexistent.agl"),
            param_tokens=[],
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
        from agm.cli_support.args import ExecArgs

        a_dir = tmp_path / "a_directory"
        a_dir.mkdir()

        args = ExecArgs(
            file=str(a_dir),
            param_tokens=[],
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
        agl_file.write_text("let x = 1\nx\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        # a simple valid program succeeds
        result = exec_command.run(args)
        assert result is None  # returns None on success (exit 0)

    def test_static_error_file_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A .agl file with a static error exits 1 and prints diagnostics to stderr."""
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = undefined_name\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
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


def _exec_args(
    agl_file: Path, *, param_tokens: list[str] | None = None, log_file: str | None = None
) -> ExecArgs:
    """Build ExecArgs for *agl_file* with all optional flags defaulted."""
    return ExecArgs(
        file=str(agl_file),
        param_tokens=param_tokens or [],
        strict_json=None,
        max_iters=None,
        runner=None,
        no_log=False,
        log_file=log_file,
    )


_skip_if_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission tests are meaningless as root (root bypasses file modes)",
)


class TestExecLogFileValidatedUpFront:
    """a non-writable --log-file fails up front with a clean Error + exit 1."""

    @_skip_if_root
    def test_unwritable_log_dir_exits_1_with_clean_error_before_running(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A --log-file under a read-only directory yields ``Error: ...`` + exit 1
        BEFORE any program statement runs (no raw PermissionError traceback)."""
        agl_file = tmp_path / "test.agl"
        # If the program ran, it would print to stdout — it must NOT.
        agl_file.write_text('print "should-not-run"\n')

        ro_dir = tmp_path / "ro"
        ro_dir.mkdir()
        ro_dir.chmod(0o555)
        log_path = ro_dir / "trace.log"

        try:
            with pytest.raises(SystemExit) as exc_info:
                exec_command.run(_exec_args(agl_file, log_file=str(log_path)))
        finally:
            ro_dir.chmod(0o755)  # restore so tmp_path cleanup succeeds

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        # Up-front failure: the program never ran, so no program output.
        assert "should-not-run" not in captured.out


class TestExecCommandEdgePaths:
    """Real-program coverage of the ok=True and pre-execution-error branches.

    The exit-2 (uncaught-AgL-exception) seam is exercised separately in
    ``TestExecExitCodeMapping`` because no real source reaches it.
    """

    def test_ok_result_returns_normally(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A successful real program prints its output and returns (exit 0)."""
        agl_file = tmp_path / "test.agl"
        agl_file.write_text('print "ok"\n')

        # Real pipeline: no SystemExit on the success path.
        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert captured.out == "ok\n"

    def test_unknown_param_option_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text('param msg: text = "ok"\nprint msg\n')

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file, param_tokens=["--unknown"]))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "error:" in captured.err

    def test_non_exhaustive_case_errors_and_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-exhaustive enum ``case`` fails statically before execution."""
        agl_file = tmp_path / "test.agl"
        agl_file.write_text(
            "enum R\n"
            "  | Pass\n"
            "  | Fail\n"
            "let r: R = Pass()\n"
            "case r of\n"
            '  | Pass() => print "passed"\n'
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "error:" in captured.err
        assert "Non-exhaustive" in captured.err
        assert "Fail" in captured.err


class TestExecExitCodeMapping:
    """the exit-2 (uncaught AgL exception) seam.

    Exit 2 is unreachable through current real source. These mocked tests pin the
    CLI's RunResult-to-exit mapping for uncaught AgL exceptions.
    """

    @pytest.mark.parametrize(
        ("fields", "expected_fragments"),
        [
            ({}, ["AgentParseError"]),
            (
                {"message": "could not parse agent output"},
                ["AgentParseError", "could not parse agent output"],
            ),
        ],
    )
    def test_uncaught_exception_maps_to_exit_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fields: dict[str, object],
        expected_fragments: list[str],
    ) -> None:
        from agm.agl.pipeline import PipelineDriver, RunError, RunResult

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        def fake_run(
            self: PipelineDriver,
            prepared: object,
            *,
            param_values: object = None,
            check_only: bool = False,
            **_kwargs: object,
        ) -> RunResult:
            return RunResult(
                ok=False,
                diagnostics=[],
                error=RunError(type_name="AgentParseError", fields=fields),
            )

        import agm.agl.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod.PipelineDriver, "run_prepared_graph", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        for fragment in expected_fragments:
            assert fragment in captured.err


class TestExecCommandWarnings:
    """Warning-severity diagnostics are reported but never affect the exit code.

    These warning paths are driven through a mocked ``run_prepared`` that injects
    a warning diagnostic.
    The error→exit-1 path IS reachable through real source and is covered by
    ``test_error_diagnostic_still_exits_1`` below.
    """

    def test_warning_with_ok_returns_normally_and_prints_to_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mocked warning: this pins that a
        # warning prints to stderr and never raises SystemExit (exit 0).
        from agm.agl.diagnostics import Diagnostic
        from agm.agl.pipeline import PipelineDriver, RunResult

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        warning = Diagnostic(
            message="declared agent 'reviewer' is unused",
            line=7,
            column=3,
            end_line=7,
            end_column=8,
            severity="warning",
        )

        def fake_run(
            self: PipelineDriver,
            prepared: object,
            *,
            param_values: object = None,
            check_only: bool = False,
            **_kwargs: object,
        ) -> RunResult:
            return RunResult(ok=True, diagnostics=[], error=None, warnings=[warning])

        import agm.agl.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod.PipelineDriver, "run_prepared_graph", fake_run)

        # ok=True even with a warning: returns normally (exit 0).
        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert f"{agl_file}:7:3-7: warning: declared agent 'reviewer' is unused" in captured.err

    def test_error_diagnostic_still_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Real source: an undefined name is a static (error-severity) diagnostic.
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = undefined_name\n")

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "undefined_name" in captured.err
        assert captured.err.startswith("test.agl:1:9-22: error:")

    def test_inline_error_diagnostic_has_command_label(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Inline -c errors carry the ``<command>:`` source label (from SourceId)."""
        args = ExecArgs(
            file=None,
            command="let x = undefined_name\n",
            param_tokens=[],
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
        assert "undefined_name" in captured.err
        assert "1:9-22: error:" in captured.err
        # The graph loader stamps inline source with SourceId(label="<command>"),
        # so <command>: appears as the source label in the diagnostic output.
        assert "<command>:1:9-22: error:" in captured.err

    def test_warning_and_error_together_exits_1_and_prints_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mocked: combining a warning with an error requires an organic warning,
        # which is injected here to pin that both print and exit is 1.
        from agm.agl.diagnostics import Diagnostic
        from agm.agl.pipeline import PipelineDriver, RunResult

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        warning = Diagnostic(
            message="unused binding",
            line=2,
            column=1,
            end_line=2,
            end_column=4,
            severity="warning",
        )
        error = Diagnostic(
            message="unknown name",
            line=5,
            column=9,
            end_line=5,
            end_column=13,
        )

        def fake_run(
            self: PipelineDriver,
            prepared: object,
            *,
            param_values: object = None,
            check_only: bool = False,
            **_kwargs: object,
        ) -> RunResult:
            return RunResult(
                ok=False, diagnostics=[error], error=None, warnings=[warning]
            )

        import agm.agl.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod.PipelineDriver, "run_prepared_graph", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert f"{agl_file}:2:1-3: warning: unused binding" in captured.err
        assert f"{agl_file}:5:9-12: error: unknown name" in captured.err
        assert f"{agl_file}:5:9-12: warning: unknown name" not in captured.err


class TestExecParsesSourceOnce:
    """``agm exec`` loads and scopes the graph exactly ONCE (no double parse).

    Regression guard: ``agm exec`` learns the declared-agent inventory (to wire
    registrations) AND executes the program.  Both must come from a single
    ``prepare_program`` call so the source is never loaded or scoped twice.
    """

    def test_exec_parses_and_scopes_source_exactly_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.agl.modules.loader as loader_mod
        import agm.agl.scope.graph as scope_graph_mod
        from agm.agl.modules.loader import ModuleGraph
        from agm.agl.modules.roots import RootSet
        from agm.agl.scope.graph import ResolvedModuleGraph
        from agm.core import dry_run

        agl_file = tmp_path / "prog.agl"
        # A declared+called agent: exec must read the inventory AND run the
        # static pipeline, the exact scenario that previously parsed twice.
        agl_file.write_text('agent impl\nask("do it", agent = impl)\n')

        real_load = loader_mod.load_graph
        real_resolve_graph = scope_graph_mod.resolve_graph
        load_calls = 0
        resolve_graph_calls = 0

        def counting_load(
            entry_source: str,
            *,
            entry_path: Path | None,
            roots: RootSet,
            default_stdlib: bool = True,
        ) -> ModuleGraph:
            nonlocal load_calls
            load_calls += 1
            return real_load(
                entry_source,
                entry_path=entry_path,
                roots=roots,
                default_stdlib=default_stdlib,
            )

        def counting_resolve_graph(
            graph: ModuleGraph,
            *,
            ambient_agents: frozenset[str] = frozenset(),
        ) -> ResolvedModuleGraph:
            nonlocal resolve_graph_calls
            resolve_graph_calls += 1
            return real_resolve_graph(graph, ambient_agents=ambient_agents)

        monkeypatch.setattr(loader_mod, "load_graph", counting_load)
        monkeypatch.setattr(scope_graph_mod, "resolve_graph", counting_resolve_graph)
        # Dry-run drives the full static pipeline (parse → scope → typecheck →
        # reconcile) without executing any agent.
        monkeypatch.setattr(dry_run, "_ENABLED", True)

        assert exec_command.run(_exec_args(agl_file)) is None
        assert load_calls == 1
        assert resolve_graph_calls == 1


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


class TestExecCommandExitCodes:
    """Exec command exit codes for valid programs, params, and flag collisions."""

    def test_valid_program_exits_0(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\nx\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        result = exec_command.run(args)
        assert result is None  # no SystemExit → exit 0

    def test_program_with_params_exits_0(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("param msg\nprint msg\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=["--msg", "hello"],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        result = exec_command.run(args)
        assert result is None

    def test_startup_config_can_read_supplied_param(self, tmp_path: Path) -> None:
        """Startup config uses the same resolved parameter values as the main run."""
        agl_file = tmp_path / "test.agl"
        agl_file.write_text('param chosen: text\nconfig runner = chosen\nprint chosen\n')

        assert exec_command.run(_exec_args(agl_file, param_tokens=["--chosen", "echo"])) is None

    def test_startup_artifact_with_declared_agents_preserves_params(
        self, tmp_path: Path
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("agent worker\nconfig log = false\nparam value: int\nprint value\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=["--value", "7"],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )

        assert exec_command.run(args) is None

    def test_missing_param_exits_1(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("param msg\nprint msg\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],  # missing 'msg'
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_param_flag_collision_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # 'timeout' is an engine key name (kebab); param timeout → --timeout collides.
        agl_file = tmp_path / "test.agl"
        agl_file.write_text('param timeout: text = "30s"\nprint timeout\n')

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))

        assert exc_info.value.code == 1
        assert "collides with a built-in exec option" in capsys.readouterr().err

    def test_undeclared_param_config_warns_but_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "\n".join(["[demo]", 'typo = "ignored"'])
        )
        agl_file = tmp_path / "test.agl"
        agl_file.write_text('program demo\nparam msg: text = "ok"\nprint msg\n')
        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert captured.out == "ok\n"
        assert "typo" in captured.err

    def test_ask_program_dispatches_to_runner_backed_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``agm exec`` always wires a runner-backed default agent; prompt
        calls are dispatched at runtime (not rejected statically), producing an
        AgentCallError (exit 2) when the runner subprocess fails."""
        import agm.commands.exec as exec_mod
        from agm.agl.runtime.agents import AgentCallHostError
        from agm.cli_support.args import ExecArgs

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('ask("hi")\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
        )
        # Patch the runner factory to return an agent that raises AgentCallHostError
        # (simulating a subprocess that fails), which exec.py surfaces as exit 2.
        def failing_agent(req: object) -> str:
            raise AgentCallHostError(
                cause="spawn_failure", exit_code=None, stderr_tail="no runner", elapsed=0.0
            )

        monkeypatch.setattr(exec_mod, "runner_backed_agent_factory", lambda **_: failing_agent)
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 2

    def test_dry_run_printing_program_exits_0_no_stdout(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``agm exec --dry-run`` runs the static pipeline only — no output."""
        from agm.cli_support.args import ExecArgs
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "hello"\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        assert exec_command.run(args) is None  # exit 0
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_dry_run_static_error_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a static-error program under --dry-run still exits 1."""
        from agm.cli_support.args import ExecArgs
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = undefined_name\n")

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_dry_run_unreachable_match_error_exits_1_before_execution(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agm.cli_support.args import ExecArgs
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "def dormant(x: bool) -> int =\n"
            "  case x of\n"
            "    | true => 1\n"
            'print "unreachable"\n'
        )
        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
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
        assert captured.out == ""
        assert ": error:" in captured.err

    def test_static_error_exits_1_not_2(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = undefined_name\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1  # static error, not AgL exception


def _spy_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Patch ``exec.PipelineDriver`` with a recording subclass.

    Returns a dict that captures the constructor kwargs the command passed.
    """
    from agm.agl.pipeline import PipelineDriver as RealRuntime

    captured: dict[str, object] = {}

    class RecordingRuntime(RealRuntime):
        def __init__(
            self,
            *,
            default_loop_limit: int = 5,
            default_strict_json: bool = False,
            default_agent: Any | None = None,
            shell_exec_timeout: float | None = None,
            default_call_depth_limit: int | None = None,
        ) -> None:
            captured["default_loop_limit"] = default_loop_limit
            captured["default_strict_json"] = default_strict_json
            captured["shell_exec_timeout"] = shell_exec_timeout
            captured["default_call_depth_limit"] = default_call_depth_limit
            super().__init__(
                default_loop_limit=default_loop_limit,
                default_strict_json=default_strict_json,
                default_agent=default_agent,
                shell_exec_timeout=shell_exec_timeout,
                default_call_depth_limit=default_call_depth_limit,
            )

    monkeypatch.setattr(exec_command, "PipelineDriver", RecordingRuntime)
    return captured


class TestExecConfigWiring:
    """[exec] config (strict-json/max-iters) flows into the runtime."""

    def _config_home(self, tmp_path: Path) -> Path:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[exec]\nstrict-json = true\nmax-iters = 9\n"
        )
        return home

    def test_config_values_reach_runtime_constructor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.cli_support.args import ExecArgs
        from agm.config.context import ConfigContext

        home = self._config_home(tmp_path)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\nx\n")

        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        captured = _spy_runtime(monkeypatch)

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        assert exec_command.run(args) is None
        assert captured["default_strict_json"] is True
        assert captured["default_loop_limit"] == 9

    def test_cli_strict_json_overrides_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.cli_support.args import ExecArgs
        from agm.config.context import ConfigContext

        home = self._config_home(tmp_path)  # config sets strict_json = true
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\nx\n")

        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        captured = _spy_runtime(monkeypatch)

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=False,  # CLI --no-strict-json overrides config true
            max_iters=7,  # CLI --max-iters overrides config 9
            runner=None,
            no_log=False,
            log_file=None,
        )
        assert exec_command.run(args) is None
        assert captured["default_strict_json"] is False
        assert captured["default_loop_limit"] == 7

    def test_timeout_config_flows_to_shell_exec_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[exec] timeout config is wired to shell_exec_timeout on PipelineDriver."""
        from agm.cli_support.args import ExecArgs
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[exec]\ntimeout = 60\n")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\nx\n")

        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        captured = _spy_runtime(monkeypatch)

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        assert exec_command.run(args) is None
        assert captured["shell_exec_timeout"] == 60.0

    def test_invalid_timeout_config_exits_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: "pytest.CaptureFixture[str]",
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[exec]\ntimeout = "forever"\n')
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")

        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(
                ExecArgs(
                    file=str(agl_file),
                    param_tokens=[],
                    strict_json=None,
                    max_iters=None,
                    runner=None,
                    no_log=True,
                    log_file=None,
                )
            )

        assert exc_info.value.code == 1
        assert "Error: invalid exec configuration" in capsys.readouterr().err


def _exec_args_with_fallback_runtime(
    agl_file: Path, monkeypatch: pytest.MonkeyPatch, *, param_tokens: list[str] | None = None
) -> ExecArgs:
    """Return ExecArgs for *agl_file* and patch PipelineDriver to have a fallback agent.

    In real use the CLI wires the runner-backed default agent; in tests we
    patch the runtime to avoid the "no default agent" static error on prompt/named-agent calls.
    """
    from agm.agl.pipeline import PipelineDriver as RealRuntime
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.request import AgentRequest, AgentResponse

    def stub_agent(req: AgentRequest) -> AgentResponse:
        return AgentResponse(content="stub")

    class FallbackRuntime(RealRuntime):
        def __init__(
            self,
            *,
            default_loop_limit: int = 5,
            default_strict_json: bool = False,
            default_agent: AgentFn | None = None,
            shell_exec_timeout: float | None = None,
            default_call_depth_limit: int | None = None,
        ) -> None:
            del default_agent
            super().__init__(
                default_loop_limit=default_loop_limit,
                default_strict_json=default_strict_json,
                default_agent=stub_agent,
                shell_exec_timeout=shell_exec_timeout,
                default_call_depth_limit=default_call_depth_limit,
            )

    monkeypatch.setattr(exec_command, "PipelineDriver", FallbackRuntime)
    return _exec_args(agl_file, param_tokens=param_tokens)


class TestDryRunInventory:
    """--dry-run prints the ."""

    def test_dry_run_inventory_ask_call(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--dry-run prints one inventory entry per agent call site."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let x = ask("Hello")\nx\n')

        args = _exec_args_with_fallback_runtime(agl_file, monkeypatch)
        assert exec_command.run(args) is None
        captured = capsys.readouterr()
        # Should print the call-sites inventory header and one entry.
        assert "call-sites" in captured.out
        assert "ask" in captured.out
        assert "text" in captured.out
        # The entry surfaces both the source line and column as "line N:C:"
        # (the captured call-site column is not dead).  `ask` starts at
        # column 9 of `let x = ask("Hello")`.
        assert "line 1:9:" in captured.out

    def test_dry_run_inventory_named_agent(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Named agent call appears in the inventory."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('agent reviewer\nask("Review this", agent = reviewer)\n')

        args = _exec_args_with_fallback_runtime(agl_file, monkeypatch)
        assert exec_command.run(args) is None
        captured = capsys.readouterr()
        # Named-agent calls use ask(..., agent: name); the inventory shows "ask"
        # as the callee (the agent: arg is a routing hint, not the callee).
        assert "ask" in captured.out

    def test_dry_run_inventory_abort_policy(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit on_parse_error: abort policy surfaces in the inventory."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('ask("Hello", on_parse_error = Abort)\n')

        args = _exec_args_with_fallback_runtime(agl_file, monkeypatch)
        assert exec_command.run(args) is None
        captured = capsys.readouterr()
        assert "policy: abort" in captured.out

    def test_dry_run_inventory_no_call_sites_empty(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--dry-run with no agent calls produces no call-sites output."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "hello"\n')

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "call-sites" not in captured.out

    def test_dry_run_inventory_static_error_exits_1_no_inventory(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Static error under --dry-run exits 1; no inventory is printed."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = undefined_name\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "call-sites" not in captured.out

    def test_dry_run_inventory_nothing_executes(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--dry-run: the registered agent stub is never invoked."""
        from agm.agl.pipeline import PipelineDriver as RealRuntime
        from agm.agl.runtime.agents import AgentFn
        from agm.agl.runtime.request import AgentRequest, AgentResponse
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agent_calls: list[AgentRequest] = []

        def spy_agent(req: AgentRequest) -> AgentResponse:
            agent_calls.append(req)
            raise AssertionError("agent should not be invoked in dry-run mode")

        class SpyRuntime(RealRuntime):
            def __init__(
                self,
                *,
                default_loop_limit: int = 5,
                default_strict_json: bool = False,
                default_agent: AgentFn | None = None,
                shell_exec_timeout: float | None = None,
                default_call_depth_limit: int | None = None,
            ) -> None:
                del default_agent
                super().__init__(
                    default_loop_limit=default_loop_limit,
                    default_strict_json=default_strict_json,
                    default_agent=spy_agent,
                    shell_exec_timeout=shell_exec_timeout,
                    default_call_depth_limit=default_call_depth_limit,
                )

        monkeypatch.setattr(exec_command, "PipelineDriver", SpyRuntime)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('ask("Hi")\n')

        assert exec_command.run(_exec_args(agl_file)) is None
        assert agent_calls == []


class TestExecFFI:
    """``agm exec`` running a file-backed program that declares ``extern def``."""

    def test_exec_runs_an_extern_program_end_to_end(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("extern def add_one(x: int) -> int\nprint(add_one(41))\n")
        (tmp_path / "prog.py").write_text("def add_one(x):\n    return x + 1\n")

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert captured.out == "42\n"

    def test_dry_run_lists_the_extern_call_site_without_importing_companion(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--dry-run's inventory lists extern calls without companion side effects."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("extern def add_one(x: int) -> int\nadd_one(41)\n")
        (tmp_path / "prog.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\n"
            "def add_one(x):\n"
            f"    open({str(marker)!r}, 'a').write('called')\n"
            "    return x + 1\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "call-sites" in captured.out
        assert "add_one" in captured.out
        assert not marker.exists()

    def test_dry_run_skips_startup_config_extern_import_and_execution(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Source startup config must not import or call externs during --dry-run."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "extern def choose_runner() -> text\n"
            "config runner = choose_runner()\n"
            "choose_runner()\n"
        )
        (tmp_path / "prog.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\n"
            "def choose_runner():\n"
            f"    open({str(marker)!r}, 'a').write('called')\n"
            "    return 'echo'\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "call-sites" in captured.out
        assert "choose_runner" in captured.out
        assert not marker.exists()

    def test_dry_run_lists_extern_call_from_imported_module(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("import mylib\nmylib::run()\n")
        (tmp_path / "mylib.agl").write_text(
            "extern def from_lib(x: int) -> int\n"
            "def run() -> int = from_lib(1)\n"
        )
        (tmp_path / "mylib.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\n"
            "def from_lib(x):\n    return x\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "from_lib" in captured.out
        assert not marker.exists()

    def test_dry_run_lists_extern_returned_from_ordinary_function(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "extern def chosen(x: int) -> int\n"
            "def choose() -> int -> int = chosen\n"
            "choose()(1)\n"
        )
        (tmp_path / "prog.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\n"
            "def chosen(x):\n    return x\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "chosen" in captured.out
        assert not marker.exists()

    def test_dry_run_lists_extern_invoked_after_value_call_return(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "extern def chosen(x: int) -> int\n"
            "def get() -> int -> int = chosen\n"
            "let h = get\n"
            "h()(1)\n"
        )
        (tmp_path / "prog.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\n"
            "def chosen(x):\n    return x\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "chosen" in captured.out
        assert "int -> int" not in captured.out
        assert not marker.exists()


class TestJsonParamsCLI:
    """--param with structured (record/list/decimal) types via JsonCodec."""

    def test_record_param_parsed_from_json_string(self, tmp_path: Path) -> None:
        """A record-typed param provided as a JSON string is parsed and usable."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'record Point\n  x: int\n  y: int\n'
            'param pt: Point\n'
            'print pt.x\n'
        )
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=['--pt={"x": 1, "y": 2}'],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        import io
        import sys

        out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            result = exec_command.run(args)
        finally:
            sys.stdout = old_stdout
        assert result is None
        assert out.getvalue().strip() == "1"

    def test_decimal_param_parsed_from_json_string(self, tmp_path: Path) -> None:
        """A decimal-typed param provided as a JSON string is accepted."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'param price: decimal\n'
            'print price\n'
        )
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=["--price", "1.5"],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        import io
        import sys

        out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            result = exec_command.run(args)
        finally:
            sys.stdout = old_stdout
        assert result is None
        assert out.getvalue().strip() == "1.5"

    def test_list_param_parsed_from_json_string(self, tmp_path: Path) -> None:
        """A list-typed param provided as a JSON array string is accepted."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'param tags: list[text]\n'
            'print tags\n'
        )
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=['--tags=["a", "b"]'],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        import io
        import sys

        out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            result = exec_command.run(args)
        finally:
            sys.stdout = old_stdout
        assert result is None
        # The output should contain the rendered list.
        output = out.getvalue().strip()
        assert output  # non-empty

    def test_record_param_invalid_json_exits_1(self, tmp_path: Path) -> None:
        """A record-typed param with invalid JSON exits 1."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'record Point\n  x: int\n  y: int\n'
            'param pt: Point\n'
            'print pt.x\n'
        )
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=["--pt", "not_json"],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# uncaught-exception output includes source line/col and trace_id
# ---------------------------------------------------------------------------


class TestUncaughtExceptionOutputFormat:
    """exec.py's exit-2 stderr must include source location and trace_id.

    Design : every runtime error should include source location and
    trace id.  The exec command's error-printing region should include the
    line (and col if available) of the raise site, and the trace_id field
    from the exception when present.
    """

    def _exec_args_nolog(self, agl_file: Path) -> "ExecArgs":
        from agm.cli_support.args import ExecArgs
        return ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
        )

    def test_uncaught_exception_stderr_includes_line(
        self, tmp_path: Path, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """Exit-2 stderr must include the source line number of the raise site."""
        agl_file = tmp_path / "prog.agl"
        # Force an uncaught AgentParseError from an exec call on line 1.
        agl_file.write_text('let x: int = exec "echo not-an-int"\nx\n')
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(self._exec_args_nolog(agl_file))
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        err = captured.err
        # The output must include a line reference (line 1).
        assert "line 1" in err or "line:1" in err or ":1:" in err, (
            f"Expected line reference in stderr, got: {err!r}"
        )

    def test_uncaught_exception_stderr_includes_trace_id(
        self, tmp_path: Path, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """Exit-2 stderr must include the trace_id when it is non-empty."""
        log_file = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let x: int = exec "echo not-an-int"\nx\n')
        from agm.cli_support.args import ExecArgs
        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=False,
            log_file=str(log_file),
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        err = captured.err
        # The trace_id is a UUID hex string (32 chars); it should appear in stderr.
        assert re.search(r'[0-9a-f]{32}', err), (
            f"Expected trace_id (hex string) in stderr, got: {err!r}"
        )

    def test_uncaught_exception_line_only_no_col(
        self, tmp_path: Path, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """Exit-2 stderr includes 'line N' when only line is set (col is None)."""
        from unittest.mock import MagicMock, patch

        from agm.agl.pipeline import RunError, RunResult
        from agm.commands import exec as exec_command

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        args = self._exec_args_nolog(agl_file)
        # Synthesize a RunResult whose error has line set but col=None.
        fake_result = RunResult(
            ok=False,
            diagnostics=[],
            error=RunError(
                type_name="SomeError",
                fields={"message": "oops"},
                line=5,
                col=None,
            ),
        )
        with patch("agm.commands.exec.PipelineDriver") as mock_rt:
            # prepare_program() must return a fake PreparedGraph with no resolved
            # graph so the static-config-resolution logic does not choke on
            # MagicMock values.
            fake_prepared = MagicMock()
            fake_prepared.resolved_graph = None
            fake_prepared.declared_agents = ()
            mock_rt.prepare_program.return_value = fake_prepared
            mock_rt.return_value.discover_params_graph.return_value = MagicMock(
                diagnostics=(), warnings=(), params=(), checked=None, program_name=None
            )
            mock_rt.return_value.run_prepared_graph.return_value = fake_result
            with pytest.raises(SystemExit) as exc_info:
                exec_command.run(args)
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "line 5" in err
        assert "col" not in err


# ---------------------------------------------------------------------------
# Binary .agl file → clean error, exit 1
# ---------------------------------------------------------------------------


class TestExecBinaryFileError:
    """agm exec with a binary (non-UTF-8) .agl file exits 1 with clean error."""

    def test_binary_agl_file_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A binary file passed as the .agl source file exits 1 with a clean Error."""
        binary_file = tmp_path / "prog.agl"
        binary_file.write_bytes(b"\xff\xfe binary garbage \x00\x01\x02")

        args = ExecArgs(
            file=str(binary_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        # No raw traceback
        assert "UnicodeDecodeError" not in captured.err
        assert "Traceback" not in captured.err

    def test_binary_agl_file_no_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A binary .agl file must not produce any stdout before failing."""
        binary_file = tmp_path / "prog.agl"
        binary_file.write_bytes(b"\xff\xfe binary garbage \x00\x01\x02")

        args = ExecArgs(
            file=str(binary_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit):
            exec_command.run(args)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Whitespace-only --runner exits 1 with clean error BEFORE any run
# ---------------------------------------------------------------------------


class TestExecWhitespaceRunner:
    """--runner '  ' (whitespace-only) must exit 1 with a clean error before execution."""

    def test_whitespace_runner_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--runner '  ' exits 1 before any statement runs."""
        agl_file = tmp_path / "prog.agl"
        # If the program ran, stdout would contain "should-not-run".
        agl_file.write_text('print "should-not-run"\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner="   ",  # whitespace-only
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_whitespace_runner_prints_clean_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--runner '  ' prints a clean usage-style error."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "should-not-run"\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner="   ",
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit):
            exec_command.run(args)
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "runner" in captured.err.lower()

    def test_whitespace_runner_no_stdout_before_exit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With a whitespace-only runner, the program must NOT execute (stdout empty)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "should-not-run"\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner="   ",
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit):
            exec_command.run(args)
        captured = capsys.readouterr()
        assert "should-not-run" not in captured.out


class TestExecPerAgentRunnerValidation:
    """A malformed/empty per-agent runner command (source hint or
    [exec.agents] config) for a DECLARED agent exits 1 BEFORE any statement
    runs — the same pre-execution contract as the default runner — instead of
    failing lazily mid-execution at dispatch."""

    def _args(self, file: str) -> ExecArgs:
        return ExecArgs(
            file=file,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner="claude -p",  # valid default; the per-agent hint is the offender
            no_log=True,
            log_file=None,
        )

    def test_empty_source_hint_exits_1_before_execution(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        # 'BEFORE' would print if the empty hint were caught lazily at dispatch.
        agl_file.write_text('agent x = ""\nprint "BEFORE"\nlet r = x "go"\n')
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(self._args(str(agl_file)))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "BEFORE" not in captured.out

    def test_malformed_quote_source_hint_exits_1_no_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('agent x = "bad \'quote"\nlet r = x "go"\n')
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(self._args(str(agl_file)))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "Traceback" not in captured.err

    def test_config_only_undeclared_bad_command_is_inert(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A malformed [exec.agents] entry for an agent the program never
        # declares must NOT fail the run — it is inert (never dispatched).
        from agm.config.general import ExecConfig

        bad_config = ExecConfig(
            runner="claude -p",
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            agents={"ghost": "bad 'quote"},  # malformed, but for an undeclared agent
            log=False,
            log_file=None,
        )
        monkeypatch.setattr(exec_command, "exec_config_from_merged", lambda *_, **__: bad_config)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ran"\n')
        # Must not raise (the inert ghost command is never validated/dispatched).
        exec_command.run(self._args(str(agl_file)))
        captured = capsys.readouterr()
        assert "ran" in captured.out


# ---------------------------------------------------------------------------
# Malformed-quoting --runner exits 1 with clean Error, no traceback
# ---------------------------------------------------------------------------


class TestExecMalformedQuotingRunner:
    """--runner with malformed quoting must exit 1 with a clean Error: on stderr,
    no traceback, and no program statement executed.

    ``shlex.split('"foo')`` raises ``ValueError('No closing quotation')``.
    The old code inlined shlex.split and let the ValueError propagate as a raw
    traceback.  The fix adds a ValueError guard to ``split_command`` itself.
    """

    def test_malformed_quote_runner_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--runner '\"foo' exits 1 before any statement runs."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "should-not-run"\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner='"foo',  # unclosed quote
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_malformed_quote_runner_prints_clean_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--runner '\"foo' prints a clean 'Error:' on stderr — no raw traceback."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "should-not-run"\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner='"foo',
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit):
            exec_command.run(args)
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "Traceback" not in captured.err
        assert "ValueError" not in captured.err

    def test_malformed_quote_runner_no_stdout_before_exit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With a malformed-quoting runner, the program must NOT execute."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "should-not-run"\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner='"foo',
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit):
            exec_command.run(args)
        captured = capsys.readouterr()
        assert "should-not-run" not in captured.out

    def test_valid_runner_still_works(self, tmp_path: Path) -> None:
        """'claude -p' (valid quoting) continues to work after the fix."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ok"\n')

        # Use recorded_runs style: just validate split_command works for valid input
        from agm.agent.runner import split_command

        result = split_command("claude -p", kind="runner")
        assert result == ["claude", "-p"]


# ---------------------------------------------------------------------------
# per-declared-agent registration + runner precedence
# (config > source runner hint > default runner)
# ---------------------------------------------------------------------------


def _install_marker_runner(
    directory: Path, env: dict[str, str], *, name: str, marker: str
) -> Path:
    """Install a fake runner *name* that echoes *marker* plus the prompt-file path.

    The script prints two lines: the marker (identifying WHICH runner ran) and
    ``prompt-file=<path>`` (the prompt-file argument it received).  This lets a
    test assert both the resolved command and that ``%{PROMPT_FILE}`` / ``@file``
    substitution delivered a real path to the runner.
    """
    directory.mkdir(parents=True, exist_ok=True)
    runner = directory / name
    runner.write_text(
        "#!/bin/bash\n"
        f'echo "{marker}"\n'
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == @* ]]; then\n'
        '    echo "prompt-file=${arg#@}"\n'
        '  elif [[ -f "$arg" ]]; then\n'
        '    echo "prompt-file=$arg"\n'
        "  fi\n"
        "done\n"
    )
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    if str(directory) not in env["PATH"].split(":"):
        env["PATH"] = str(directory) + ":" + env["PATH"]
    return runner


def _install_argv_echo_runner(
    directory: Path, env: dict[str, str], *, name: str, marker: str
) -> Path:
    """Install a fake runner *name* that echoes *marker* plus every raw argument.

    Unlike ``_install_marker_runner`` (which normalizes ``@file`` / existing-file
    arguments into a ``prompt-file=<path>`` line), this runner echoes each argv
    entry verbatim as ``arg=<raw>``.  That makes the difference between the
    ``%{PROMPT_FILE}`` placeholder branch (mid-argument substitution, e.g.
    ``--file=/abs/path``) and the bare-``@file`` append fallback (a separate
    trailing ``@/abs/path`` argument) observable in stdout.
    """
    directory.mkdir(parents=True, exist_ok=True)
    runner = directory / name
    runner.write_text(
        "#!/bin/bash\n"
        f'echo "{marker}"\n'
        'for arg in "$@"; do\n'
        '  echo "arg=$arg"\n'
        "done\n"
    )
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    if str(directory) not in env["PATH"].split(":"):
        env["PATH"] = str(directory) + ":" + env["PATH"]
    return runner


class TestExecAgentPrecedence:
    """declared agents resolve via config > source hint > default runner.

    Driven through real fake-runner binaries (CLI subprocess), asserting which
    runner produced the agent response — a user-visible behavior, not an
    internal call.
    """

    def _run_agm_exec(
        self, args: list[str], *, env: dict[str, str], cwd: Path
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

    def test_missing_param_prevents_startup_config_agent_call(self, tmp_path: Path) -> None:
        """Required params are checked before startup config can invoke an agent."""
        env = self._base_env()
        marker = tmp_path / "runner-called"
        runner = tmp_path / "bin" / "bootstrap-runner"
        runner.parent.mkdir()
        runner.write_text(f"#!/bin/bash\ntouch {marker}\necho ignored\n")
        runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
        env["PATH"] = str(runner.parent) + ":" + env["PATH"]

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'config runner = ask("choose a runner")\nparam required\nprint required\n'
        )
        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\nrunner = "bootstrap-runner"\n')

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)

        assert result.returncode == 1
        assert not marker.exists()

    def test_startup_config_agent_uses_bootstrap_runner(self, tmp_path: Path) -> None:
        """An agent call in source config is real, not a placeholder response."""
        env = self._base_env()
        bootstrap = _install_marker_runner(
            tmp_path / "bin", env, name="bootstrap-runner", marker="BOOTSTRAP"
        )
        bootstrap_calls = tmp_path / "bootstrap-calls"
        bootstrap.write_text(
            "#!/bin/bash\n"
            f"echo bootstrap >> {bootstrap_calls}\n"
            "echo 'final-runner %{PROMPT_FILE}'\n"
        )
        _install_marker_runner(tmp_path / "bin", env, name="final-runner", marker="FINAL")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'config runner = ask("choose a runner")\n'
            'let answer = ask("do it")\n'
            "print answer\n"
        )
        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\nrunner = "bootstrap-runner"\n')

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "FINAL" in result.stdout
        assert "BOOTSTRAP" not in result.stdout
        assert bootstrap_calls.read_text().splitlines() == ["bootstrap"]

    def test_config_beats_source_hint(self, tmp_path: Path) -> None:
        """A config [exec.agents] entry overrides the source runner hint."""
        env = self._base_env()
        _install_marker_runner(tmp_path / "bin", env, name="source-runner", marker="FROM-SOURCE")
        _install_marker_runner(tmp_path / "bin", env, name="config-runner", marker="FROM-CONFIG")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'agent impl = "source-runner %{PROMPT_FILE}"\n'
            'let x = ask("do it", agent = impl)\n'
            "print x\n"
        )

        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[exec]\nrunner = "default-runner"\n\n'
            '[exec.agents]\nimpl = "config-runner %{PROMPT_FILE}"\n'
        )

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "FROM-CONFIG" in result.stdout
        assert "FROM-SOURCE" not in result.stdout

    def test_config_beats_default_for_bare_declaration(self, tmp_path: Path) -> None:
        """A config [exec.agents] entry overrides the default runner for a BARE
        declaration (one with no source runner hint)."""
        env = self._base_env()
        _install_marker_runner(tmp_path / "bin", env, name="config-runner", marker="FROM-CONFIG")
        _install_marker_runner(tmp_path / "bin", env, name="default-runner", marker="FROM-DEFAULT")

        agl_file = tmp_path / "prog.agl"
        # ``impl`` is declared BARE (no ``= "runner"`` hint).
        agl_file.write_text('agent impl\nlet x = ask("do it", agent = impl)\nprint x\n')

        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[exec]\nrunner = "default-runner"\n\n'
            '[exec.agents]\nimpl = "config-runner %{PROMPT_FILE}"\n'
        )

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "FROM-CONFIG" in result.stdout
        assert "FROM-DEFAULT" not in result.stdout

    def test_multiple_agents_mixed_precedence_in_one_run(self, tmp_path: Path) -> None:
        """Three declared agents in ONE program route by name through the shared
        factory: config override, source hint, and default runner respectively."""
        env = self._base_env()
        _install_marker_runner(tmp_path / "bin", env, name="config-a", marker="FROM-CONFIG-A")
        _install_marker_runner(tmp_path / "bin", env, name="source-a", marker="FROM-SOURCE-A")
        _install_marker_runner(tmp_path / "bin", env, name="source-b", marker="FROM-SOURCE-B")
        _install_marker_runner(tmp_path / "bin", env, name="default-runner", marker="FROM-DEFAULT")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'agent a = "source-a %{PROMPT_FILE}"\n'  # config override → CONFIG-A
            'agent b = "source-b %{PROMPT_FILE}"\n'  # no config entry → SOURCE-B
            "agent c\n"  # bare, no config entry → DEFAULT
            'let ra = ask("first", agent = a)\n'
            'let rb = ask("second", agent = b)\n'
            'let rc = ask("third", agent = c)\n'
            "print ra\n"
            "print rb\n"
            "print rc\n"
        )

        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[exec]\nrunner = "default-runner"\n\n'
            '[exec.agents]\na = "config-a %{PROMPT_FILE}"\n'
        )

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Each agent routed to exactly the expected runner.
        assert "FROM-CONFIG-A" in result.stdout
        assert "FROM-SOURCE-A" not in result.stdout  # config beat the source hint
        assert "FROM-SOURCE-B" in result.stdout
        assert "FROM-DEFAULT" in result.stdout

    def test_source_hint_beats_default_runner(self, tmp_path: Path) -> None:
        """With no config entry, the source runner hint wins over the default runner."""
        env = self._base_env()
        _install_marker_runner(tmp_path / "bin", env, name="source-runner", marker="FROM-SOURCE")
        _install_marker_runner(tmp_path / "bin", env, name="default-runner", marker="FROM-DEFAULT")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'agent impl = "source-runner %{PROMPT_FILE}"\n'
            'let x = ask("do it", agent = impl)\n'
            "print x\n"
        )

        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\nrunner = "default-runner"\n')

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "FROM-SOURCE" in result.stdout
        assert "FROM-DEFAULT" not in result.stdout

    def test_bare_declaration_uses_default_runner(self, tmp_path: Path) -> None:
        """A bare ``agent NAME`` with no config entry uses the resolved default runner."""
        env = self._base_env()
        _install_marker_runner(tmp_path / "bin", env, name="default-runner", marker="FROM-DEFAULT")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('agent impl\nlet x = ask("do it", agent = impl)\nprint x\n')

        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\nrunner = "default-runner"\n')

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "FROM-DEFAULT" in result.stdout

    def test_source_hint_prompt_file_substitution(self, tmp_path: Path) -> None:
        """``%{PROMPT_FILE}`` in a source runner hint is substituted IN PLACE,
        mid-argument — proving the placeholder branch ran (not the ``@file``
        append fallback, which can only add a separate trailing argument)."""
        env = self._base_env()
        # An argv-echo runner reveals each raw argument verbatim, so a
        # mid-argument substitution (``--file=/abs/path``) is distinguishable
        # from the bare-``@file`` fallback (a separate ``@/abs/path`` argument).
        _install_argv_echo_runner(tmp_path / "bin", env, name="source-runner", marker="FROM-SOURCE")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'agent impl = "source-runner --file=%{PROMPT_FILE}"\n'
            'let x = ask("do it", agent = impl)\n'
            "print x\n"
        )

        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\nrunner = "default-runner"\n')

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # The placeholder was substituted mid-argument: the runner saw
        # ``--file=/<abs path>`` as a single argv entry.  The ``@file`` fallback
        # could never produce this (it would append a separate ``@/...`` arg).
        assert re.search(r"^arg=--file=/", result.stdout, re.MULTILINE), (
            f"Expected mid-argument %{{PROMPT_FILE}} substitution, got: {result.stdout!r}"
        )
        # And the fallback form must NOT appear.
        assert "arg=@/" not in result.stdout

    def test_bare_agent_and_ask_both_resolve_via_default(self, tmp_path: Path) -> None:
        """A bare declared agent and built-in ``ask`` both resolve via the default runner."""
        env = self._base_env()
        _install_marker_runner(tmp_path / "bin", env, name="default-runner", marker="FROM-DEFAULT")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "agent impl\n"
            'let a = ask "first"\n'
            'let b = ask("second", agent = impl)\n'
            "print a\n"
            "print b\n"
        )

        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\nrunner = "default-runner"\n')

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Both calls dispatched to the default runner.
        assert result.stdout.count("FROM-DEFAULT") == 2

    def test_undeclared_agent_call_exits_1_nothing_runs(self, tmp_path: Path) -> None:
        """Calling an undeclared agent is a pre-execution scope error: exit 1, no run."""
        env = self._base_env()
        _install_marker_runner(tmp_path / "bin", env, name="default-runner", marker="FROM-DEFAULT")

        agl_file = tmp_path / "prog.agl"
        # ``ghost`` is never declared with ``agent ghost``.
        agl_file.write_text('let x = ghost "do it"\nprint x\n')

        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\nrunner = "default-runner"\n')

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)
        assert result.returncode == 1, f"stdout: {result.stdout} stderr: {result.stderr}"
        # The runner never ran: no marker on stdout.
        assert "FROM-DEFAULT" not in result.stdout


# ---------------------------------------------------------------------------
# source config declaration wiring — CLI > source > config precedence
# ---------------------------------------------------------------------------


def _exec_args_no_log(
    agl_file: Path,
    *,
    strict_json: bool | None = None,
    max_iters: int | None = None,
    max_call_depth: int | None = None,
    runner: str | None = None,
    no_log: bool = True,
    log_file: str | None = None,
    log: bool = False,
) -> ExecArgs:
    """Build a minimal ExecArgs for source-config-precedence tests."""
    return ExecArgs(
        file=str(agl_file),
        param_tokens=[],
        strict_json=strict_json,
        max_iters=max_iters,
        max_call_depth=max_call_depth,
        runner=runner,
        no_log=no_log,
        log_file=log_file,
        log=log,
    )


class TestExecStartupConfigPrepass:
    def test_startup_config_uses_configured_call_depth(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config log = false\ndef loop() -> int = loop()\nloop()\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, max_call_depth=1))

        assert exc_info.value.code == 2
        assert "RecursionError" in capsys.readouterr().err

    def test_startup_diagnostics_exit_1_and_print_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.agl.diagnostics import Diagnostic
        from agm.agl.pipeline import PipelineDriver, StartupConfigResult

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config log = true\nprint 1\n")
        warning = Diagnostic(message="startup warning", line=1, severity="warning")
        error = Diagnostic(message="startup error", line=1)

        def fake_collect(
            self: PipelineDriver,
            prepared: object,
            *,
            names: set[str],
            compiled_graph: object = None,
            checked_graph: object = None,
            param_values: object = None,
            config_cli: object = None,
            config_base: object = None,
        ) -> StartupConfigResult:
            assert compiled_graph is not None
            assert checked_graph is compiled_graph.checked_graph
            return StartupConfigResult(
                ok=False,
                diagnostics=[error],
                error=None,
                warnings=[warning],
            )

        monkeypatch.setattr(PipelineDriver, "collect_startup_config_graph", fake_collect)

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "warning: startup warning" in captured.err
        assert "error: startup error" in captured.err

    def test_startup_error_exits_2_and_prints_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.agl.diagnostics import Diagnostic
        from agm.agl.pipeline import PipelineDriver, RunError, StartupConfigResult

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config runner = \"runner\"\nprint 1\n")
        warning = Diagnostic(message="startup warning", line=1, severity="warning")
        error = RunError(type_name="Abort", fields={"message": "boom"}, line=1, col=1)

        def fake_collect(
            self: PipelineDriver,
            prepared: object,
            *,
            names: set[str],
            compiled_graph: object = None,
            checked_graph: object = None,
            param_values: object = None,
            config_cli: object = None,
            config_base: object = None,
        ) -> StartupConfigResult:
            return StartupConfigResult(
                ok=False,
                diagnostics=[],
                error=error,
                warnings=[warning],
            )

        monkeypatch.setattr(PipelineDriver, "collect_startup_config_graph", fake_collect)

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "warning: startup warning" in captured.err
        assert "AgL exception: Abort: boom" in captured.err

    def test_startup_config_extern_shares_companion_state_with_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        count_file = tmp_path / "imports.txt"
        agl_file.write_text(
            "extern def runner_name() -> text\n"
            "extern def import_count() -> int\n"
            "config runner = runner_name()\n"
            "print import_count()\n"
        )
        (tmp_path / "prog.py").write_text(
            "from pathlib import Path\n"
            f"_count_path = Path({str(count_file)!r})\n"
            "_current = int(_count_path.read_text()) if _count_path.exists() else 0\n"
            "_count_path.write_text(str(_current + 1))\n"
            "def runner_name():\n"
            "    return 'runner'\n"
            "def import_count():\n"
            "    return int(_count_path.read_text())\n"
        )

        exec_command.run(_exec_args_no_log(agl_file))

        assert capsys.readouterr().out == "1\n"
        assert count_file.read_text() == "1"

    def test_option_text_value_ignores_non_text_some_payload(self) -> None:
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import STD_CORE_ID
        from agm.agl.semantics.values import EnumValue, IntValue

        value = EnumValue(
            nominal=NominalId(STD_CORE_ID, "Option"),
            display_name="Option",
            variant="Some",
            fields={"value": IntValue(1)},
        )

        assert exec_command._option_text_value({"log-file": value}, "log-file") is None


class TestExecSourceConfigPrecedence:
    """source ``config`` declarations (CLI > source > config precedence).

    Each test uses behavioral assertions — observable exit codes and output —
    rather than internal call counts, following the testing policy.
    """

    # ------------------------------------------------------------------
    # max_iters source declaration
    # ------------------------------------------------------------------

    def test_source_max_iters_caps_loop_at_source_value(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``config max-iters = 3`` in source caps the do loop at 3 iterations.

        The loop ``until n >= 100`` cannot complete in 3 iterations (n starts at
        0 and increments by 1), so the runtime raises a LoopLimitExceeded and
        the command exits 2.
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config max-iters = 3\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 2

    def test_source_max_iters_allows_completion_when_sufficient(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``config max-iters = 100`` allows a do loop that needs exactly 100 iterations."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config max-iters = 100\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n'
        )
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None  # exit 0
        assert capsys.readouterr().out == "done\n"

    def test_source_max_iters_zero_is_rejected(self, tmp_path: Path) -> None:
        """``config max-iters = 0`` is rejected instead of becoming a live loop limit."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config max-iters = 0\nprint 1\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 2

    def test_source_max_iters_negative_expression_is_rejected(self, tmp_path: Path) -> None:
        """A computed negative ``max-iters`` value is rejected at the config binding."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let bad = 0 - 1\nconfig max-iters = bad\nprint 1\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 2

    def test_cli_max_iters_overrides_source_max_iters(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI ``--max-iters 100`` overrides ``config max-iters = 3`` in source.

        With --max-iters 100 the loop completes in 100 iterations (exits 0);
        with source config max-iters=3 it would fail (exit 2).
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config max-iters = 3\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n'
        )
        result = exec_command.run(_exec_args_no_log(agl_file, max_iters=100))
        assert result is None  # exit 0 — CLI 100 overrides source 3
        assert capsys.readouterr().out == "done\n"

    def test_source_max_iters_overrides_config_max_iters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Source ``config max-iters = 100`` overrides ``[exec] max-iters = 3`` in config.

        Config says 3 (loop would fail); source declaration says 100 (loop completes).
        """
        from agm.config.general import ExecConfig

        low_limit_config = ExecConfig(
            runner=None,
            strict_json=False,
            default_loop_limit=3,
            timeout=None,
            agents={},
            log=False,
            log_file=None,
        )
        monkeypatch.setattr(
            exec_command, "exec_config_from_merged", lambda *_, **__: low_limit_config
        )

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config max-iters = 100\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n'
        )
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None  # exit 0 — source 100 overrides config 3
        assert capsys.readouterr().out == "done\n"

    # ------------------------------------------------------------------
    # max-iters valve scope: self-bounded loops are exempt
    # ------------------------------------------------------------------

    def test_max_iters_does_not_cap_for_over_finite_collection(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--max-iters`` caps only unguarded loops; a ``for`` over a finite
        collection larger than the cap must run to completion.

        Regression: the valve applied to all loops, so ``--max-iters 3`` broke
        ``for x in [1,2,3,4]``.
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "var s = 0\n"
            "for x in [1, 2, 3, 4, 5] do s := s + x done\n"
            "print s\n"
        )
        result = exec_command.run(_exec_args_no_log(agl_file, max_iters=3))
        assert result is None  # exit 0
        assert capsys.readouterr().out == "15\n"

    def test_max_iters_does_not_cap_bounded_do_n_loop(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--max-iters`` does not cap a ``do[n]`` loop whose own bound exceeds it.

        The loop's own ``[n]`` bound is its termination machinery; the host
        safety valve must not cut it short.
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "var i = 0\n"
            "do[10]\n"
            "  i := i + 1\n"
            "until i >= 5\n"
            "print i\n"
        )
        result = exec_command.run(_exec_args_no_log(agl_file, max_iters=3))
        assert result is None  # exit 0
        assert capsys.readouterr().out == "5\n"

    def test_max_iters_caps_unbounded_do_until_loop(
        self, tmp_path: Path
    ) -> None:
        """``--max-iters`` caps an unguarded ``do…until`` loop (no [n], no for)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "var i = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 1000\n"
            "print i\n"
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, max_iters=3))
        assert exc_info.value.code == 2

    def test_max_iters_five_enables_valve(self, tmp_path: Path) -> None:
        """``--max-iters 5`` enables the valve (regression: was a silent no-op).

        The old magic-``5`` sentinel treated ``5`` as "off", so ``--max-iters 5``
        was a no-op while ``config max-iters = 5`` enabled the valve.  Both must
        now enable it consistently.
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "var i = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 1000\n"
            "print i\n"
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, max_iters=5))
        assert exc_info.value.code == 2

    # ------------------------------------------------------------------
    # strict-json source declaration
    # ------------------------------------------------------------------

    def test_source_strict_json_flows_into_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config strict-json = true`` in source does NOT pre-fold into the
        PipelineDriver constructor; it is applied when the binding executes.
        The constructor receives the config-file value (False by default)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config strict-json = true\nlet x = 1\nx\n")

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        # constructor gets the config-file default; the source declaration
        # applies the live change at the point of the IrConfigBind execution.
        assert captured["default_strict_json"] is False

    def test_cli_strict_json_overrides_source_strict_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI ``--no-strict-json`` (strict_json=False) overrides ``config strict-json = true``."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config strict-json = true\nlet x = 1\nx\n")

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file, strict_json=False))
        assert result is None
        assert captured["default_strict_json"] is False

    def test_source_strict_json_overrides_config_strict_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source ``config strict-json = false`` overrides ``[exec] strict-json = true``
        when the binding executes. The PipelineDriver constructor still receives
        the config-file value (True)."""
        from agm.config.general import ExecConfig

        strict_config = ExecConfig(
            runner=None,
            strict_json=True,
            default_loop_limit=5,
            timeout=None,
            agents={},
            log=False,
            log_file=None,
        )
        monkeypatch.setattr(exec_command, "exec_config_from_merged", lambda *_, **__: strict_config)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config strict-json = false\nlet x = 1\nx\n")

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        # constructor gets config-file value (True); the source declaration (False)
        # overrides it at runtime via IrConfigBind → _apply_config_effect.
        assert captured["default_strict_json"] is True

    # ------------------------------------------------------------------
    # timeout source declaration
    # ------------------------------------------------------------------

    def test_source_timeout_flows_into_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config timeout = "30s"`` in source does NOT pre-fold into the
        PipelineDriver constructor; it is applied when the binding executes.
        The constructor receives the config-file value (None by default)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config timeout = "30s"\nlet x = 1\nx\n')

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        # constructor gets the config-file default (None); the source
        # declaration updates shell_exec_timeout at binding time.
        assert captured["shell_exec_timeout"] is None

    def test_source_timeout_integer_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``config timeout = 60`` (integer) is a type error: timeout is Option[text]."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config timeout = 60\nlet x = 1\nx\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 1

    def test_source_timeout_overrides_config_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source ``config timeout = "30s"`` overrides ``[exec] timeout = 999``
        when the binding executes. The PipelineDriver constructor still receives
        the config-file value (999.0)."""
        from agm.config.general import ExecConfig

        config_with_timeout = ExecConfig(
            runner=None,
            strict_json=False,
            default_loop_limit=5,
            timeout=999.0,
            agents={},
            log=False,
            log_file=None,
        )
        monkeypatch.setattr(
            exec_command, "exec_config_from_merged", lambda *_, **__: config_with_timeout
        )

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config timeout = "30s"\nlet x = 1\nx\n')

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        # constructor gets config-file value (999.0); the source declaration (30s)
        # overrides it at runtime via IrConfigBind → _apply_config_effect.
        assert captured["shell_exec_timeout"] == pytest.approx(999.0)

    def test_source_timeout_invalid_string_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        """A source timeout string that type-checks but fails parse_timeout raises
        a clean AgL-level ValueError at the config decl point (exit 2).

        ``config timeout = "forever"`` is a valid ``Option[text]`` value so it
        passes scope and typecheck; the config binding handler converts the
        parse_timeout ValueError to an AglRaise (uncaught AgL exception → exit 2).
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config timeout = "forever"\nlet x = 1\nx\n')

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 2

    # ------------------------------------------------------------------
    # log source declaration
    # ------------------------------------------------------------------

    def test_source_log_true_creates_trace_file(self, tmp_path: Path) -> None:
        """``config log = true`` in source enables trace logging (creates a file)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config log = true\nprint "hi"\n')

        # Run in tmp_path so .agent-files/ is created there.
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            exec_command.run(
                ExecArgs(
                    file=str(agl_file),
                    param_tokens=[],
                    strict_json=None,
                    max_iters=None,
                    runner=None,
                    no_log=False,
                    log_file=None,
                )
            )
        finally:
            os.chdir(old_cwd)

        agent_files = tmp_path / ".agent-files"
        log_files = list(agent_files.glob("exec-*.log"))
        assert log_files, "Expected a trace log file to be created by config log = true"

    def test_source_log_non_literal_creates_trace_file(self, tmp_path: Path) -> None:
        """``config log = enabled`` is honored for startup trace resolution."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let enabled = true\nconfig log = enabled\nprint "hi"\n')

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            exec_command.run(
                ExecArgs(
                    file=str(agl_file),
                    param_tokens=[],
                    strict_json=None,
                    max_iters=None,
                    runner=None,
                    no_log=False,
                    log_file=None,
                )
            )
        finally:
            os.chdir(old_cwd)

        log_files = list((tmp_path / ".agent-files").glob("exec-*.log"))
        assert log_files, "Expected a trace log file from computed config log=true"

    def test_source_log_file_writes_to_specified_path(self, tmp_path: Path) -> None:
        """``config log-file = "path"`` in source writes the trace to that path."""
        log_path = tmp_path / "trace.log"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(f'config log-file = "{log_path}"\nprint "hi"\n')

        exec_command.run(
            ExecArgs(
                file=str(agl_file),
                param_tokens=[],
                strict_json=None,
                max_iters=None,
                runner=None,
                no_log=False,
                log_file=None,
            )
        )
        assert log_path.exists(), "Expected trace log at source-specified path"

    def test_cli_no_log_overrides_source_log_true(self, tmp_path: Path) -> None:
        """CLI ``--no-log`` overrides ``config log = true`` — no trace file created."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config log = true\nprint "hi"\n')

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            exec_command.run(_exec_args_no_log(agl_file, no_log=True))
        finally:
            os.chdir(old_cwd)

        # --no-log must prevent trace creation even when the source says log=true.
        agent_files = tmp_path / ".agent-files"
        if agent_files.exists():
            log_files = list(agent_files.glob("exec-*.log"))
            assert not log_files, "Expected no trace log when --no-log overrides source log=true"

    def test_source_log_file_overrides_config_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config log-file`` source declaration overrides ``[exec] log = false`` in config."""
        log_path = tmp_path / "source_trace.log"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(f'config log-file = "{log_path}"\nprint "hi"\n')

        from agm.config.general import ExecConfig

        no_log_config = ExecConfig(
            runner=None,
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            agents={},
            log=False,
            log_file=None,
        )
        monkeypatch.setattr(exec_command, "exec_config_from_merged", lambda *_, **__: no_log_config)

        exec_command.run(
            ExecArgs(
                file=str(agl_file),
                param_tokens=[],
                strict_json=None,
                max_iters=None,
                runner=None,
                no_log=False,
                log_file=None,
            )
        )
        assert log_path.exists(), (
            "Expected trace log at source-specified path despite config log=false"
        )

    # ------------------------------------------------------------------
    # runner source declaration
    # ------------------------------------------------------------------

    def test_source_runner_flows_into_agent_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config runner = "..."`` source declaration sets the default runner command.

        We capture the runner command passed to runner_backed_agent_factory
        because it is the single user-observable boundary between exec.py and
        the subprocess world.
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config runner = "my-runner"\nlet x = 1\nx\n')

        captured_runner: list[str] = []

        import agm.agl.runtime.agents as agents_mod
        from agm.agl.runtime.agents import AgentFn

        real_factory = agents_mod.runner_backed_agent_factory

        def spy_factory(
            *,
            default_runner_cmd: str,
            per_agent_cmds: dict[str, str],
            idle_timeout: float | None = None,
        ) -> AgentFn:
            captured_runner.append(default_runner_cmd)
            return real_factory(
                default_runner_cmd=default_runner_cmd,
                per_agent_cmds=per_agent_cmds,
                idle_timeout=idle_timeout,
            )

        monkeypatch.setattr(exec_command, "runner_backed_agent_factory", spy_factory)

        exec_command.run(_exec_args_no_log(agl_file))
        assert captured_runner[-1:] == ["my-runner"]

    def test_source_runner_non_literal_flows_into_agent_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config runner = name`` is honored for startup runner resolution."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let name = "computed-runner"\nconfig runner = name\nlet x = 1\nx\n')

        captured_runner: list[str] = []

        import agm.agl.runtime.agents as agents_mod
        from agm.agl.runtime.agents import AgentFn

        real_factory = agents_mod.runner_backed_agent_factory

        def spy_factory(
            *,
            default_runner_cmd: str,
            per_agent_cmds: dict[str, str],
            idle_timeout: float | None = None,
        ) -> AgentFn:
            captured_runner.append(default_runner_cmd)
            return real_factory(
                default_runner_cmd=default_runner_cmd,
                per_agent_cmds=per_agent_cmds,
                idle_timeout=idle_timeout,
            )

        monkeypatch.setattr(exec_command, "runner_backed_agent_factory", spy_factory)

        exec_command.run(_exec_args_no_log(agl_file))
        assert captured_runner[-1:] == ["computed-runner"]

    def test_cli_runner_overrides_source_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI ``--runner`` overrides ``config runner`` source declaration."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config runner = "source-runner"\nlet x = 1\nx\n')

        captured_runner: list[str] = []

        import agm.agl.runtime.agents as agents_mod
        from agm.agl.runtime.agents import AgentFn

        real_factory = agents_mod.runner_backed_agent_factory

        def spy_factory(
            *,
            default_runner_cmd: str,
            per_agent_cmds: dict[str, str],
            idle_timeout: float | None = None,
        ) -> AgentFn:
            captured_runner.append(default_runner_cmd)
            return real_factory(
                default_runner_cmd=default_runner_cmd,
                per_agent_cmds=per_agent_cmds,
                idle_timeout=idle_timeout,
            )

        monkeypatch.setattr(exec_command, "runner_backed_agent_factory", spy_factory)

        exec_command.run(_exec_args_no_log(agl_file, runner="cli-runner"))
        assert captured_runner[-1:] == ["cli-runner"]


def _exec_args_inline_no_log(
    command: str,
    *,
    strict_json: bool | None = None,
    max_iters: int | None = None,
    runner: str | None = None,
) -> ExecArgs:
    """Build a minimal ExecArgs for -c inline exec tests."""
    return ExecArgs(
        file=None,
        command=command,
        param_tokens=[],
        strict_json=strict_json,
        max_iters=max_iters,
        runner=runner,
        no_log=True,
        log_file=None,
        log=False,
    )


class TestExecModuleRoots:
    """``agm exec`` uses graph pipeline and module roots.

    Tests verify that:
    - ``agm exec <file>`` uses the file's directory as invocation root.
    - ``agm exec -c`` uses cwd as invocation root.
    - An error in an imported module carries that module's file path in the
      diagnostic (via source_label).
    - A multi-file program executes successfully when the library is reachable.
    """

    def test_exec_file_uses_file_directory_as_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``agm exec file.agl`` can import a sibling module in the same directory."""
        lib_dir = tmp_path
        (lib_dir / "mylib.agl").write_text(
            "def answer() -> int = 42\n"
        )
        entry = lib_dir / "entry.agl"
        entry.write_text("import mylib\nlet r = answer()\nprint r\n")

        # A successful run returns normally (no SystemExit).
        exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_exec_file_import_error_reports_source_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Error in imported module shows that module's file path in diagnostic."""
        lib_dir = tmp_path
        (lib_dir / "broken.agl").write_text("def f() -> int = undeclared_name\n")
        entry = lib_dir / "entry.agl"
        entry.write_text("import broken\nlet r = f()\nr\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(entry))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # The diagnostic should mention the broken.agl file path
        assert "broken.agl" in captured.err

    def test_exec_inline_uses_cwd_as_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``agm exec -c`` uses cwd (from config context) as invocation root."""
        # Write a lib module to a tmp dir
        lib_dir = tmp_path / "libdir"
        lib_dir.mkdir()
        (lib_dir / "util.agl").write_text("def greet() -> text = \"Hi!\"\n")
        entry_source = "import util\nlet r = greet()\nprint r\n"

        # Patch current_config_context as imported in exec_command
        from agm.config import context as ctx_mod

        original_ctx = ctx_mod.current_config_context()

        class FakeCtx:
            home = original_ctx.home
            proj_dir = original_ctx.proj_dir
            cwd = lib_dir

        monkeypatch.setattr(exec_command, "current_config_context", lambda: FakeCtx())

        # A successful run returns normally (no SystemExit).
        exec_command.run(_exec_args_inline_no_log(entry_source))
        captured = capsys.readouterr()
        assert "Hi!" in captured.out

    def test_exec_file_missing_import_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing import causes exit 1 with a diagnostic on stderr."""
        entry = tmp_path / "prog.agl"
        entry.write_text("import no_such_module\nlet x = 1\nx\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(entry))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "no_such_module" in captured.err

    def test_exec_multifile_successful_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A two-module AgL program executes successfully end-to-end."""
        lib_dir = tmp_path
        (lib_dir / "calc.agl").write_text("def square(n: int) -> int = n * n\n")
        entry = lib_dir / "prog.agl"
        entry.write_text("import calc\nlet r = square(4)\nprint r\n")

        # A successful run returns normally (no SystemExit).
        exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "16" in captured.out

    def test_invalid_module_roots_config_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ValueError from load_module_roots causes exit 1 with a config error."""
        from unittest.mock import patch

        entry = tmp_path / "prog.agl"
        entry.write_text("let x = 1\nx\n")

        with (
            patch(
                "agm.commands.exec.load_module_roots",
                side_effect=ValueError("bad config"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            exec_command.run(_exec_args_no_log(entry))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "module roots" in captured.err.lower()

    def test_configured_lib_root_is_resolved(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured lib_root in module roots config is resolved and used."""
        from agm.config.module_roots import ModuleRootsConfig

        lib_dir = tmp_path / "mylib"
        lib_dir.mkdir()
        (lib_dir / "shared.agl").write_text("def pi() -> int = 314\n")

        entry_dir = tmp_path / "work"
        entry_dir.mkdir()
        entry = entry_dir / "prog.agl"
        entry.write_text("import shared\nlet r = pi()\nprint r\n")

        # Patch load_module_roots to return a ModuleRootsConfig with lib_root set.
        with monkeypatch.context() as mp:
            mp.setattr(
                exec_command,
                "load_module_roots",
                lambda *, home, proj_dir, cwd: ModuleRootsConfig(
                    lib_root=(str(lib_dir), tmp_path),  # absolute path
                    extra=(),
                ),
            )
            exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "314" in captured.out

    def test_exec_wildcard_import_multifile(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``import pkg.*`` wildcard imports two sibling modules and both are callable.

        Verifies that the wildcard import path works end-to-end through the
        exec_command pipeline (discover_params_graph + run_prepared_graph).
        """
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "add.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        (pkg_dir / "mul.agl").write_text("def mul(a: int, b: int) -> int = a * b\n")
        entry = tmp_path / "prog.agl"
        entry.write_text(
            "import pkg.*\n"
            "let s = add(3, 4)\n"
            "let p = mul(3, 4)\n"
            "print s\n"
            "print p\n"
        )

        exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "7" in captured.out
        assert "12" in captured.out

    def test_exec_qualified_import_multifile(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``import x qualified`` + ``x::name`` qualified call works end-to-end.

        Verifies that the qualified-import path works through the command pipeline.
        """
        (tmp_path / "mathlib.agl").write_text("def square(n: int) -> int = n * n\n")
        entry = tmp_path / "prog.agl"
        entry.write_text(
            "import mathlib qualified\n"
            "let r = mathlib::square(7)\n"
            "print r\n"
        )

        exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "49" in captured.out

    def test_exec_imported_function_ask_multi_scenario(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multi-scenario: imported function calls ask() with an agent passed from entry.

        Scenario A: agent returns "Alice" → result printed is "Alice"
        Scenario B: agent returns "World" → result printed is "World"

        Each scenario drives exec_command.run end-to-end with a distinct mock
        response injected via runner_backed_agent_factory, asserting each output.
        """
        import agm.commands.exec as exec_mod
        from agm.agl.runtime.request import AgentRequest, AgentResponse

        (tmp_path / "greeter.agl").write_text(
            "def greet(prompt: text, bot: agent) -> text =\n"
            "  ask(prompt, agent = bot)\n"
        )
        entry = tmp_path / "entry.agl"
        entry.write_text(
            "import greeter\n"
            "agent mybot\n"
            'let result = greeter::greet("What is your name?", mybot)\n'
            "print result\n"
        )

        def _run_with_response(response: str) -> str:
            """Inject *response* as the mock agent answer and return stdout."""

            def mock_agent(req: AgentRequest) -> AgentResponse:
                return AgentResponse(content=response)

            # Patch runner_backed_agent_factory so both the default agent and
            # the registered 'mybot' agent use our mock (matching the pattern
            # in test_ask_program_dispatches_to_runner_backed_agent above).
            monkeypatch.setattr(
                exec_mod, "runner_backed_agent_factory", lambda **_: mock_agent
            )
            exec_command.run(_exec_args_no_log(entry))
            out, _ = capsys.readouterr()
            return out

        # Scenario A: agent returns "Alice"
        out_a = _run_with_response("Alice")
        assert "Alice" in out_a

        # Scenario B: agent returns "World" — different input/output combination
        out_b = _run_with_response("World")
        assert "World" in out_b
        assert out_a != out_b


class TestExecCliModulePaths:
    """``-I/--module-path`` roots are threaded into ``assemble_roots``.

    Tests verify that a module placed only in a ``-I DIR`` root is resolvable
    by ``agm exec`` when that root is passed via ``ExecArgs.module_paths``.
    """

    def test_module_in_cli_root_is_resolvable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A module placed only in a -I root is importable by the entry program."""
        lib_root = tmp_path / "mylibs"
        lib_root.mkdir()
        (lib_root / "helper.agl").write_text("def answer() -> int = 99\n")

        # entry.agl lives in a separate directory with no sibling modules
        entry_dir = tmp_path / "prog"
        entry_dir.mkdir()
        entry = entry_dir / "main.agl"
        entry.write_text("import helper\nlet r = answer()\nprint r\n")

        args = ExecArgs(
            file=str(entry),
            command=None,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
            log=False,
            module_paths=[str(lib_root)],
        )
        exec_command.run(args)
        captured = capsys.readouterr()
        assert "99" in captured.out

    def test_multiple_cli_roots_each_resolvable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two -I roots each contribute a distinct module, both resolvable."""
        root_a = tmp_path / "rootA"
        root_a.mkdir()
        (root_a / "mod_a.agl").write_text("def va() -> int = 10\n")

        root_b = tmp_path / "rootB"
        root_b.mkdir()
        (root_b / "mod_b.agl").write_text("def vb() -> int = 20\n")

        entry_dir = tmp_path / "entry"
        entry_dir.mkdir()
        entry = entry_dir / "prog.agl"
        entry.write_text(
            "import mod_a\nimport mod_b\nlet r = va() + vb()\nprint r\n"
        )

        args = ExecArgs(
            file=str(entry),
            command=None,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
            log=False,
            module_paths=[str(root_a), str(root_b)],
        )
        exec_command.run(args)
        captured = capsys.readouterr()
        assert "30" in captured.out

    def test_module_not_found_without_cli_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without -I, a module in an external root is not found (exit 1)."""
        lib_root = tmp_path / "mylibs"
        lib_root.mkdir()
        (lib_root / "helper.agl").write_text("def answer() -> int = 99\n")

        entry_dir = tmp_path / "prog"
        entry_dir.mkdir()
        entry = entry_dir / "main.agl"
        entry.write_text("import helper\nlet r = answer()\nprint r\n")

        args = ExecArgs(
            file=str(entry),
            command=None,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
            log=False,
            module_paths=[],  # no CLI roots
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "helper" in captured.err

    def test_inline_exec_with_cli_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``-c`` inline exec resolves imports from a -I root."""
        lib_root = tmp_path / "inlinelibs"
        lib_root.mkdir()
        (lib_root / "util.agl").write_text("def greet() -> text = \"Hello!\"\n")

        # cwd is irrelevant — helper is not reachable from cwd; only via -I
        entry_dir = tmp_path / "work"
        entry_dir.mkdir()

        from agm.config import context as ctx_mod

        original_ctx = ctx_mod.current_config_context()

        class FakeCtx:
            home = original_ctx.home
            proj_dir = original_ctx.proj_dir
            cwd = entry_dir

        monkeypatch.setattr(exec_command, "current_config_context", lambda: FakeCtx())

        args = ExecArgs(
            file=None,
            command="import util\nlet r = greet()\nprint r\n",
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
            log=False,
            module_paths=[str(lib_root)],
        )
        exec_command.run(args)
        captured = capsys.readouterr()
        assert "Hello!" in captured.out


# ---------------------------------------------------------------------------
# config_base carries [exec] timeout / log-file floor (not hardcoded none)
# ---------------------------------------------------------------------------


class TestConfigBaseOptionKeys:
    """bare ``config timeout`` / ``config log-file`` respect [exec] config."""

    def _spy_config_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, object]:
        """Patch PipelineDriver.run_prepared_graph to capture the config_base kwarg."""
        from collections.abc import Mapping

        from agm.agl.matchcompile import MatchCompiledModuleGraph
        from agm.agl.pipeline import PipelineDriver as RealRuntime
        from agm.agl.pipeline import PreparedGraph, RunResult
        from agm.agl.semantics.values import Value

        captured: dict[str, object] = {}

        class CapturingRuntime(RealRuntime):
            def run_prepared_graph(
                self,
                prepared: PreparedGraph,
                *,
                param_values: Mapping[str, object] | None = None,
                check_only: bool = False,
                log_file: Path | None = None,
                compiled_graph: MatchCompiledModuleGraph | None = None,
                config_cli: Mapping[str, Value] | None = None,
                config_base: Mapping[str, Value] | None = None,
            ) -> RunResult:
                captured["config_base"] = config_base
                # Proceed with actual execution so exec.py sees a proper result.
                return super().run_prepared_graph(
                    prepared,
                    param_values=param_values,
                    check_only=check_only,
                    log_file=log_file,
                    compiled_graph=compiled_graph,
                    config_cli=config_cli,
                    config_base=config_base,
                )

        monkeypatch.setattr(exec_command, "PipelineDriver", CapturingRuntime)
        return captured

    def test_bare_config_timeout_with_exec_config_binds_some(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bare ``config timeout`` with [exec].timeout set must bind some(...), not none."""
        from agm.agl.semantics.values import EnumValue
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[exec]\ntimeout = "60s"\n')
        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config timeout\nlet x = 1\nx\n")

        captured = self._spy_config_base(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None

        config_base = captured["config_base"]
        assert isinstance(config_base, dict)
        timeout_val = config_base["timeout"]
        assert isinstance(timeout_val, EnumValue)
        assert timeout_val.variant == "Some"

    def test_bare_config_log_file_with_exec_config_binds_some(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bare ``config log-file`` with [exec].log_file set must bind some(path), not none."""
        from agm.agl.semantics.values import EnumValue, TextValue
        from agm.config.general import ExecConfig

        expected_path = "/var/log/agent.jsonl"
        config_with_log_file = ExecConfig(
            runner=None,
            strict_json=False,
            default_loop_limit=5,
            timeout=None,
            agents={},
            log=False,
            log_file=expected_path,
        )
        monkeypatch.setattr(
            exec_command, "exec_config_from_merged", lambda *_, **__: config_with_log_file
        )

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config log-file\nlet x = 1\nx\n")

        captured = self._spy_config_base(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None

        config_base = captured["config_base"]
        assert isinstance(config_base, dict)
        log_file_val = config_base["log-file"]
        assert isinstance(log_file_val, EnumValue)
        assert log_file_val.variant == "Some"
        assert log_file_val.fields == {"value": TextValue(expected_path)}


class TestReservedFileStem:
    """: a file stem matching a reserved AGM section name must exit 1."""

    def test_reserved_stem_no_program_decl_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """File 'loop.agl' with no ``program`` decl uses stem 'loop' (reserved) → exit 1."""
        agl_file = tmp_path / "loop.agl"
        agl_file.write_text("let x = 1\nx\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 1

    def test_exec_stem_no_program_decl_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """File 'exec.agl' with no ``program`` decl uses stem 'exec' (reserved) → exit 1."""
        agl_file = tmp_path / "exec.agl"
        agl_file.write_text("let x = 1\nx\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 1

    def test_reserved_stem_with_program_decl_ok(self, tmp_path: Path) -> None:
        """A file named 'loop.agl' with an explicit ``program myapp`` decl runs fine."""
        agl_file = tmp_path / "loop.agl"
        agl_file.write_text("program myapp\nlet x = 1\nx\n")

        # Should succeed (program decl provides a non-reserved key).
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None

    def test_non_reserved_stem_runs_fine(self, tmp_path: Path) -> None:
        """A file with a non-reserved stem works normally."""
        agl_file = tmp_path / "myworkflow.agl"
        agl_file.write_text("let x = 1\nx\n")

        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None

    def test_inline_c_with_reserved_name_unaffected(self, tmp_path: Path) -> None:
        """Inline -c programs have no file stem and are never affected."""
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=None,
            command="let x = 1\nx\n",
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
            log=False,
        )
        result = exec_command.run(args)
        assert result is None


class TestRawStringRoundTrip:
    """Timeout raw string round-trips through config_base (no str(float) loss)."""

    def _spy_config_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, object]:
        from collections.abc import Mapping

        from agm.agl.matchcompile import MatchCompiledModuleGraph
        from agm.agl.pipeline import PipelineDriver as RealRuntime
        from agm.agl.pipeline import PreparedGraph, RunResult
        from agm.agl.semantics.values import Value

        captured: dict[str, object] = {}

        class CapturingRuntime(RealRuntime):
            def run_prepared_graph(
                self,
                prepared: PreparedGraph,
                *,
                param_values: Mapping[str, object] | None = None,
                check_only: bool = False,
                log_file: Path | None = None,
                compiled_graph: MatchCompiledModuleGraph | None = None,
                config_cli: Mapping[str, Value] | None = None,
                config_base: Mapping[str, Value] | None = None,
            ) -> RunResult:
                captured["config_base"] = config_base
                return super().run_prepared_graph(
                    prepared,
                    param_values=param_values,
                    check_only=check_only,
                    log_file=log_file,
                    compiled_graph=compiled_graph,
                    config_cli=config_cli,
                    config_base=config_base,
                )

        monkeypatch.setattr(exec_command, "PipelineDriver", CapturingRuntime)
        return captured

    def test_exec_timeout_string_preserved_in_config_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[exec].timeout = \"30s\" must produce config_base[\"timeout\"] = some(\"30s\"),
        not some(\"30.0\") (the str(float) approximation must be fixed)."""
        from agm.agl.semantics.values import EnumValue, TextValue
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[exec]\ntimeout = "30s"\n')

        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config timeout\nlet x = 1\nx\n")

        captured = self._spy_config_base(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None

        config_base = captured["config_base"]
        assert isinstance(config_base, dict)
        timeout_val = config_base["timeout"]
        assert isinstance(timeout_val, EnumValue)
        assert timeout_val.variant == "Some"
        # The raw string "30s" must be preserved, not "30.0".
        assert timeout_val.fields == {"value": TextValue("30s")}

    def test_program_timeout_override_string_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[<program>].timeout = \"60m\" overrides [exec].timeout and preserves string."""
        from agm.agl.semantics.values import EnumValue, TextValue
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[exec]\ntimeout = "30s"\n\n[myprog]\ntimeout = "60m"\n'
        )

        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        agl_file = tmp_path / "myprog.agl"
        agl_file.write_text("config timeout\nlet x = 1\nx\n")

        captured = self._spy_config_base(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None

        config_base = captured["config_base"]
        assert isinstance(config_base, dict)
        timeout_val = config_base["timeout"]
        assert isinstance(timeout_val, EnumValue)
        assert timeout_val.variant == "Some"
        # Program override "60m" must be preserved.
        assert timeout_val.fields == {"value": TextValue("60m")}


class TestF1StemVsProgramNameBug:
    """ regression: file stem != program NAME decl must not split engine/param key."""

    def test_program_name_decl_used_for_both_engine_config_and_config_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File foo.agl with 'program bar': [bar].timeout must win for engine AND config_base.

        With the bug, stem 'foo' is used for engine config -> [foo] (empty) -> 30s
        and 'bar' is used for params. The fix makes both use 'bar' -> 60m.
        """
        from collections.abc import Mapping

        from agm.agl.matchcompile import MatchCompiledModuleGraph
        from agm.agl.pipeline import PipelineDriver as RealRuntime
        from agm.agl.pipeline import PreparedGraph, RunResult
        from agm.agl.semantics.values import EnumValue, TextValue, Value
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        # [exec] has 30s; [bar] (the declared program name) overrides with 60m.
        (home / ".agm" / "config.toml").write_text(
            '[exec]\ntimeout = "30s"\n\n[bar]\ntimeout = "60m"\n'
        )
        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        # File stem is 'foo', but program declares 'bar'.
        agl_file = tmp_path / "foo.agl"
        agl_file.write_text("program bar\nconfig timeout\nlet x = 1\nx\n")

        captured: dict[str, object] = {}

        class CapturingRuntime(RealRuntime):
            def run_prepared_graph(
                self,
                prepared: PreparedGraph,
                *,
                param_values: Mapping[str, object] | None = None,
                check_only: bool = False,
                log_file: Path | None = None,
                compiled_graph: MatchCompiledModuleGraph | None = None,
                config_cli: Mapping[str, Value] | None = None,
                config_base: Mapping[str, Value] | None = None,
            ) -> RunResult:
                captured["config_base"] = config_base
                captured["shell_exec_timeout"] = self._shell_exec_timeout
                return super().run_prepared_graph(
                    prepared,
                    param_values=param_values,
                    check_only=check_only,
                    log_file=log_file,
                    compiled_graph=compiled_graph,
                    config_cli=config_cli,
                    config_base=config_base,
                )

        monkeypatch.setattr(exec_command, "PipelineDriver", CapturingRuntime)

        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None

        # Both config_base timeout and engine shell_exec_timeout must come from [bar] (60m),
        # not [exec] (30s).
        config_base_val = captured["config_base"]
        assert isinstance(config_base_val, dict)
        timeout_binding = config_base_val["timeout"]
        assert isinstance(timeout_binding, EnumValue)
        assert timeout_binding.variant == "Some"
        assert timeout_binding.fields == {"value": TextValue("60m")}

        # Engine shell-exec timeout: 60 minutes = 3600 seconds.
        shell_timeout = captured["shell_exec_timeout"]
        assert shell_timeout == pytest.approx(3600.0)


class TestProgramLogFilePathResolution:
    """[<program>].log-file relative path is anchored to the config directory."""

    def test_program_log_file_relative_resolved_to_config_dir(
        self, tmp_path: Path
    ) -> None:
        from agm.config.general import load_merged_config, program_config_from_merged

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[myprog]\nlog-file = "my.log"\n')

        merged = load_merged_config(home=home, proj_dir=None, cwd=tmp_path)
        prog = program_config_from_merged(merged, "myprog")

        log_file_val = prog.get("log-file")
        assert isinstance(log_file_val, str)
        # Must be absolute (anchored to ~/.agm/ where the config lives)
        assert Path(log_file_val).is_absolute()
        assert log_file_val.endswith("my.log")
