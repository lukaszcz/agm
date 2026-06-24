"""Tests for the `agm exec` CLI command (M0).

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
from typing import Protocol

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
        # M1: a simple valid program succeeds
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
    """F2a: a non-writable --log-file fails up front with a clean Error + exit 1."""

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
    ``TestExecExitCodeMapping`` because no real M1 source reaches it after F6.
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

    def test_non_exhaustive_case_warns_but_exits_0(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """F1: a non-exhaustive enum ``case`` is a warning, not an error.

        The program still runs to completion: exit 0 (``run`` returns ``None``),
        with the exhaustiveness warning printed to stderr and program output on
        stdout.
        """
        agl_file = tmp_path / "test.agl"
        agl_file.write_text(
            "enum R\n"
            "  | Pass\n"
            "  | Fail\n"
            "let r: R = Pass()\n"
            "case r of\n"
            '  | Pass() => print "passed"\n'
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        # The matched branch ran (no MatchError, since the value is Pass).
        assert captured.out == "passed\n"
        # The exhaustiveness warning names the missing variant on stderr, with a
        # ``warning:`` prefix to disambiguate from errors (F8).
        assert "warning:" in captured.err
        assert "Non-exhaustive" in captured.err
        assert "Fail" in captured.err


class TestExecExitCodeMapping:
    """F13a: the exit-2 (uncaught AgL exception) seam.

    Exit 2 is unreachable through real source in M1 — F6 removed the only path
    that produced an uncaught AgL exception from a statically-valid program (the
    fabricated exec ExecError). These mocked tests pin the CLI's RunResult→exit
    mapping until a real M2+ program can drive it.
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
        from agm.agl.runtime.runtime import RunError, RunResult, WorkflowRuntime

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        def fake_run(
            self: WorkflowRuntime,
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

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run_prepared_graph", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        for fragment in expected_fragments:
            assert fragment in captured.err


class TestExecCommandWarnings:
    """Warning-severity diagnostics are reported but never affect the exit code.

    No AgL checker warning is organically producible in M1 (warnings such as
    non-exhaustive ``case`` land with M2/M3 analysis), so the warning paths are
    driven through a mocked ``run_prepared`` that injects a warning diagnostic.
    The error→exit-1 path IS reachable through real source and is covered by
    ``test_error_diagnostic_still_exits_1`` below.
    """

    def test_warning_with_ok_returns_normally_and_prints_to_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mocked: no organic M1 warning exists (lands in M2/M3). This pins that a
        # warning prints to stderr and never raises SystemExit (exit 0).
        from agm.agl.runtime.runtime import Diagnostic, RunResult, WorkflowRuntime

        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")

        warning = Diagnostic(
            message="case is non-exhaustive",
            line=7,
            column=3,
            end_line=7,
            end_column=8,
            severity="warning",
        )

        def fake_run(
            self: WorkflowRuntime,
            prepared: object,
            *,
            param_values: object = None,
            check_only: bool = False,
            **_kwargs: object,
        ) -> RunResult:
            return RunResult(ok=True, diagnostics=[], error=None, warnings=[warning])

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run_prepared_graph", fake_run)

        # ok=True even with a warning: returns normally (exit 0).
        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert f"{agl_file}:7:3-7: warning: case is non-exhaustive" in captured.err

    def test_error_diagnostic_still_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Real source: an undefined name is a static (error-severity) diagnostic.
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = undefined_name\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "undefined_name" in captured.err
        assert f"{agl_file}:1:9-22: error:" in captured.err

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
        # which does not exist until M2/M3 — pins that both print and exit is 1.
        from agm.agl.runtime.runtime import Diagnostic, RunResult, WorkflowRuntime

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
            self: WorkflowRuntime,
            prepared: object,
            *,
            param_values: object = None,
            check_only: bool = False,
            **_kwargs: object,
        ) -> RunResult:
            return RunResult(
                ok=False, diagnostics=[error], error=None, warnings=[warning]
            )

        import agm.agl.runtime.runtime as rt_mod

        monkeypatch.setattr(rt_mod.WorkflowRuntime, "run_prepared_graph", fake_run)

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
        agl_file.write_text('agent impl\nask("do it", agent: impl)\n')

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


class TestExecCommandM1:
    """M1 exec command behavior: exit codes, params, agent calls."""

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
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("param max_iters: int = 1\nprint max_iters\n")

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
            "\n".join(["[params.demo]", 'typo = "ignored"'])
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
        """M5a: ``agm exec`` always wires a runner-backed default agent; prompt
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
        """F2: ``agm exec --dry-run`` runs the static pipeline only — no output."""
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
        """F2: a static-error program under --dry-run still exits 1."""
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
    """Patch ``exec.WorkflowRuntime`` with a recording subclass.

    Returns a dict that captures the constructor kwargs the command passed.
    """
    from agm.agl.runtime.runtime import WorkflowRuntime as RealRuntime

    captured: dict[str, object] = {}

    class RecordingRuntime(RealRuntime):
        def __init__(
            self,
            *,
            default_loop_limit: int = 5,
            default_strict_json: bool = False,
            default_agent: object | None = None,
            shell_exec_timeout: float | None = None,
        ) -> None:
            captured["default_loop_limit"] = default_loop_limit
            captured["default_strict_json"] = default_strict_json
            captured["shell_exec_timeout"] = shell_exec_timeout
            super().__init__(
                default_loop_limit=default_loop_limit,
                default_strict_json=default_strict_json,
                shell_exec_timeout=shell_exec_timeout,
            )

    monkeypatch.setattr(exec_command, "WorkflowRuntime", RecordingRuntime)
    return captured


class TestExecConfigWiring:
    """F12: [exec] config (strict_json/default_loop_limit) flows into the runtime."""

    def _config_home(self, tmp_path: Path) -> Path:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[exec]\nstrict_json = true\ndefault_loop_limit = 9\n"
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
        """[exec] timeout config is wired to shell_exec_timeout on WorkflowRuntime."""
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
    """Return ExecArgs for *agl_file* and patch WorkflowRuntime to have a fallback agent.

    In real use the CLI wires the runner-backed default agent (M5); in tests we
    patch the runtime to avoid the "no default agent" static error on prompt/named-agent calls.
    """
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.request import AgentRequest, AgentResponse
    from agm.agl.runtime.runtime import WorkflowRuntime as RealRuntime

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
        ) -> None:
            super().__init__(
                default_loop_limit=default_loop_limit,
                default_strict_json=default_strict_json,
                default_agent=stub_agent,
                shell_exec_timeout=shell_exec_timeout,
            )

    monkeypatch.setattr(exec_command, "WorkflowRuntime", FallbackRuntime)
    return _exec_args(agl_file, param_tokens=param_tokens)


class TestDryRunInventory:
    """M2: --dry-run prints the §10.1 static call-site inventory."""

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
        # (F7: the captured call-site column is not dead).  `ask` starts at
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
        agl_file.write_text('agent reviewer\nask("Review this", agent: reviewer)\n')

        args = _exec_args_with_fallback_runtime(agl_file, monkeypatch)
        assert exec_command.run(args) is None
        captured = capsys.readouterr()
        # In v2, named-agent calls use ask(..., agent: name); the inventory
        # shows "ask" as the callee (the agent: arg is a routing hint, not the callee).
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
        agl_file.write_text('ask("Hello", on_parse_error: Abort)\n')

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
        from agm.agl.runtime.agents import AgentFn
        from agm.agl.runtime.request import AgentRequest, AgentResponse
        from agm.agl.runtime.runtime import WorkflowRuntime as RealRuntime
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
            ) -> None:
                super().__init__(
                    default_loop_limit=default_loop_limit,
                    default_strict_json=default_strict_json,
                    default_agent=spy_agent,
                    shell_exec_timeout=shell_exec_timeout,
                )

        monkeypatch.setattr(exec_command, "WorkflowRuntime", SpyRuntime)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('ask("Hi")\n')

        assert exec_command.run(_exec_args(agl_file)) is None
        assert agent_calls == []


class TestJsonParamsCLI:
    """M2: --param with structured (record/list/decimal) types via JsonCodec."""

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
# F7: uncaught-exception output includes source line/col and trace_id
# ---------------------------------------------------------------------------


class TestUncaughtExceptionOutputFormat:
    """F7: exec.py's exit-2 stderr must include source location and trace_id.

    Design §12.6: every runtime error should include source location and
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

        from agm.agl.runtime.runtime import RunError, RunResult
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
        with patch("agm.commands.exec.WorkflowRuntime") as mock_rt:
            # prepare_program() must return a fake PreparedGraph with empty pragmas so
            # the pragma-resolution logic does not choke on MagicMock values.
            fake_prepared = MagicMock()
            fake_prepared.config_pragmas = {}
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
# Task 3: binary .agl file → clean error, exit 1
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
# Task 4: whitespace-only --runner exits 1 with clean error BEFORE any run
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
        monkeypatch.setattr(exec_command, "load_exec_config", lambda **_: bad_config)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ran"\n')
        # Must not raise (the inert ghost command is never validated/dispatched).
        exec_command.run(self._args(str(agl_file)))
        captured = capsys.readouterr()
        assert "ran" in captured.out


# ---------------------------------------------------------------------------
# Task 1 (MAJOR): malformed-quoting --runner exits 1 with clean Error, no traceback
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
# M5: per-declared-agent registration + runner precedence
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
    """M5: declared agents resolve via config > source hint > default runner.

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

    def test_config_beats_source_hint(self, tmp_path: Path) -> None:
        """A config [exec.agents] entry overrides the source runner hint."""
        env = self._base_env()
        _install_marker_runner(tmp_path / "bin", env, name="source-runner", marker="FROM-SOURCE")
        _install_marker_runner(tmp_path / "bin", env, name="config-runner", marker="FROM-CONFIG")

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'agent impl = "source-runner %{PROMPT_FILE}"\n'
            'let x = ask("do it", agent: impl)\n'
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
        agl_file.write_text('agent impl\nlet x = ask("do it", agent: impl)\nprint x\n')

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
            'let ra = ask("first", agent: a)\n'
            'let rb = ask("second", agent: b)\n'
            'let rc = ask("third", agent: c)\n'
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
            'let x = ask("do it", agent: impl)\n'
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
        agl_file.write_text('agent impl\nlet x = ask("do it", agent: impl)\nprint x\n')

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
            'let x = ask("do it", agent: impl)\n'
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
            'let b = ask("second", agent: impl)\n'
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
# M3: config pragma wiring — CLI > pragma > config precedence
# ---------------------------------------------------------------------------


def _exec_args_no_log(
    agl_file: Path,
    *,
    strict_json: bool | None = None,
    max_iters: int | None = None,
    runner: str | None = None,
    no_log: bool = True,
    log_file: str | None = None,
    log: bool = False,
) -> ExecArgs:
    """Build a minimal ExecArgs for M3 pragma-precedence tests."""
    return ExecArgs(
        file=str(agl_file),
        param_tokens=[],
        strict_json=strict_json,
        max_iters=max_iters,
        runner=runner,
        no_log=no_log,
        log_file=log_file,
        log=log,
    )


class TestExecPragmaPrecedence:
    """M3: ``config`` pragmas in source (CLI > pragma > config precedence).

    Each test uses behavioral assertions — observable exit codes and output —
    rather than internal call counts, following the testing policy.
    """

    # ------------------------------------------------------------------
    # max_iters pragma
    # ------------------------------------------------------------------

    def test_pragma_max_iters_caps_loop_at_pragma_value(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``config max_iters = 3`` in source caps the do loop at 3 iterations.

        The loop ``until n >= 100`` cannot complete in 3 iterations (n starts at
        0 and increments by 1), so the runtime raises a LoopLimitExceeded and
        the command exits 2.
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config max_iters = 3\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 2

    def test_pragma_max_iters_allows_completion_when_sufficient(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``config max_iters = 100`` allows a do loop that needs exactly 100 iterations."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config max_iters = 100\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n'
        )
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None  # exit 0
        assert capsys.readouterr().out == "done\n"

    def test_cli_max_iters_overrides_pragma_max_iters(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI ``--max-iters 100`` overrides ``config max_iters = 3`` in source.

        With --max-iters 100 the loop completes in 100 iterations (exits 0);
        with pragma max_iters=3 it would fail (exit 2).
        """
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config max_iters = 3\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n'
        )
        result = exec_command.run(_exec_args_no_log(agl_file, max_iters=100))
        assert result is None  # exit 0 — CLI 100 overrides pragma 3
        assert capsys.readouterr().out == "done\n"

    def test_pragma_max_iters_overrides_config_max_iters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Source pragma ``config max_iters = 100`` overrides ``[exec] default_loop_limit = 3``.

        Config says 3 (loop would fail); pragma says 100 (loop completes).
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
        monkeypatch.setattr(exec_command, "load_exec_config", lambda **_: low_limit_config)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config max_iters = 100\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n'
        )
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None  # exit 0 — pragma 100 overrides config 3
        assert capsys.readouterr().out == "done\n"

    # ------------------------------------------------------------------
    # strict_json pragma
    # ------------------------------------------------------------------

    def test_pragma_strict_json_flows_into_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config strict_json = true`` in source sets ``default_strict_json=True``
        on the WorkflowRuntime (observable via the spy pattern)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config strict_json = true\nlet x = 1\nx\n")

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        assert captured["default_strict_json"] is True

    def test_cli_strict_json_overrides_pragma_strict_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI ``--no-strict-json`` (strict_json=False) overrides ``config strict_json = true``."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config strict_json = true\nlet x = 1\nx\n")

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file, strict_json=False))
        assert result is None
        assert captured["default_strict_json"] is False

    def test_pragma_strict_json_overrides_config_strict_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source pragma ``config strict_json = false`` overrides ``[exec] strict_json = true``."""
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
        monkeypatch.setattr(exec_command, "load_exec_config", lambda **_: strict_config)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config strict_json = false\nlet x = 1\nx\n")

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        assert captured["default_strict_json"] is False

    # ------------------------------------------------------------------
    # timeout pragma
    # ------------------------------------------------------------------

    def test_pragma_timeout_flows_into_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config timeout = "30s"`` in source sets shell_exec_timeout=30.0."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config timeout = "30s"\nlet x = 1\nx\n')

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        assert captured["shell_exec_timeout"] == 30.0

    def test_pragma_timeout_integer_seconds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config timeout = 60`` (integer) in source sets shell_exec_timeout=60.0."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("config timeout = 60\nlet x = 1\nx\n")

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        assert captured["shell_exec_timeout"] == 60.0

    def test_pragma_timeout_overrides_config_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source pragma timeout overrides ``[exec] timeout`` from config."""
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
        monkeypatch.setattr(exec_command, "load_exec_config", lambda **_: config_with_timeout)

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config timeout = "30s"\nlet x = 1\nx\n')

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        assert captured["shell_exec_timeout"] == 30.0

    def test_pragma_timeout_invalid_string_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pragma timeout string that passes the scope check but fails parse_timeout exits 1.

        The pragma validator accepts any non-empty string; parse_timeout rejects
        values like "forever" that are not valid duration strings.  We inject the
        invalid value via a patched prepare_program() to bypass the scope pass.
        """
        from unittest.mock import MagicMock

        from agm.agl.modules.roots import RootSet
        from agm.agl.runtime.runtime import WorkflowRuntime as RealRuntime

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")

        real_prepare_program = RealRuntime.prepare_program

        def fake_prepare_program(
            source: str,
            *,
            entry_path: Path | None,
            roots: RootSet,
            default_stdlib: bool = True,
        ) -> object:
            real_pg = real_prepare_program(
                source,
                entry_path=entry_path,
                roots=roots,
                default_stdlib=default_stdlib,
            )
            # Wrap the real PreparedGraph with an invalid timeout in config_pragmas.
            fake_pg = MagicMock()
            fake_pg.config_pragmas = {"timeout": "forever"}
            fake_pg.declared_agents = real_pg.declared_agents
            fake_pg.resolved_graph = real_pg.resolved_graph
            fake_pg.diagnostics = real_pg.diagnostics
            fake_pg.warnings = real_pg.warnings
            return fake_pg

        monkeypatch.setattr(
            exec_command.WorkflowRuntime,
            "prepare_program",
            staticmethod(fake_prepare_program),
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "timeout" in captured.err.lower()

    # ------------------------------------------------------------------
    # log pragma
    # ------------------------------------------------------------------

    def test_pragma_log_true_creates_trace_file(self, tmp_path: Path) -> None:
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

    def test_pragma_log_file_writes_to_specified_path(self, tmp_path: Path) -> None:
        """``config log_file = "path"`` in source writes the trace to that path."""
        log_path = tmp_path / "trace.log"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(f'config log_file = "{log_path}"\nprint "hi"\n')

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
        assert log_path.exists(), "Expected trace log at pragma-specified path"

    def test_cli_no_log_overrides_pragma_log_true(self, tmp_path: Path) -> None:
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

        # --no-log must prevent trace creation even when the pragma says log=true.
        agent_files = tmp_path / ".agent-files"
        if agent_files.exists():
            log_files = list(agent_files.glob("exec-*.log"))
            assert not log_files, "Expected no trace log when --no-log overrides pragma log=true"

    def test_pragma_log_file_overrides_config_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config log_file`` pragma overrides ``[exec] log = false`` in config."""
        log_path = tmp_path / "pragma_trace.log"
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(f'config log_file = "{log_path}"\nprint "hi"\n')

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
        monkeypatch.setattr(exec_command, "load_exec_config", lambda **_: no_log_config)

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
            "Expected trace log at pragma-specified path despite config log=false"
        )

    # ------------------------------------------------------------------
    # runner pragma
    # ------------------------------------------------------------------

    def test_pragma_runner_flows_into_agent_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``config runner = "..."`` pragma sets the default runner command.

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
        assert captured_runner == ["my-runner"]

    def test_cli_runner_overrides_pragma_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI ``--runner`` overrides ``config runner`` pragma."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('config runner = "pragma-runner"\nlet x = 1\nx\n')

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
        assert captured_runner == ["cli-runner"]


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
    """M5b: ``agm exec`` uses graph pipeline and module roots.

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
            "  ask(prompt, agent: bot)\n"
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
    """M6: ``-I/--module-path`` roots are threaded into ``assemble_roots``.

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
