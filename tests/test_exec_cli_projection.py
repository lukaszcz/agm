"""Tests for Task 4: unified exec CLI projection (engine flags + config_cli wiring).

Multi-scenario coverage for:
- New engine flags: --timeout/--no-timeout, --no-log-file
- Mutual exclusivity of new flag pairs
- config_cli projection for all six engine keys (CLI > source > base)
- Verbatim collision check (no underscore↔kebab normalization)
- Graceful degradation on syntax-error source (help/completion)

Do NOT assert exact help/error text — only structural assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.exec as exec_command
from agm.agl.runtime.params import convert_config_value
from agm.agl.semantics.types import OPTION_TEXT_TYPE
from agm.agl.semantics.values import BoolValue, TextValue, Value
from agm.cli_support.args import ExecArgs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def invoke(argv: list[str]) -> Result:
    return CliRunner().invoke(
        get_command(cli.app), argv, prog_name="agm", catch_exceptions=False
    )


def _exec_args(
    file: Path | str | None = None,
    *,
    command: str | None = None,
    param_tokens: list[str] | None = None,
    timeout: str | None = None,
    no_timeout: bool = False,
    no_log_file: bool = False,
    log_file: str | None = None,
    no_log: bool = False,
    log: bool = False,
    runner: str | None = None,
    strict_json: bool | None = None,
    max_iters: int | None = None,
) -> ExecArgs:
    return ExecArgs(
        file=str(file) if file is not None else None,
        command=command,
        param_tokens=param_tokens or [],
        strict_json=strict_json,
        max_iters=max_iters,
        runner=runner,
        no_log=no_log,
        log_file=log_file,
        log=log,
        timeout=timeout,
        no_timeout=no_timeout,
        no_log_file=no_log_file,
    )


def _spy_config_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Value]:
    """Patch ``run_prepared_graph`` to capture config_cli and return ok=True."""
    from agm.agl.pipeline import PipelineDriver, RunResult

    captured: dict[str, Value] = {}

    def fake_run(self: PipelineDriver, prepared: object, **kwargs: object) -> RunResult:
        cli_map = kwargs.get("config_cli")
        if isinstance(cli_map, dict):
            captured.update(cli_map)
        return RunResult(ok=True, diagnostics=[], error=None)

    monkeypatch.setattr(PipelineDriver, "run_prepared_graph", fake_run)
    return captured


# ---------------------------------------------------------------------------
# 1. New flag → ExecArgs field mapping
# ---------------------------------------------------------------------------


class TestNewFlagParsing:
    """Parser-contract tests: new flags map correctly to ExecArgs fields."""

    def test_timeout_flag_sets_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.exec as exec_mod

        calls: list[ExecArgs] = []

        def fake_run(args: ExecArgs) -> None:
            calls.append(args)

        monkeypatch.setattr(exec_mod, "run", fake_run)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(["exec", "--timeout", "30s", str(agl_file)])
        assert result.exit_code == 0
        assert calls[0].timeout == "30s"
        assert calls[0].no_timeout is False

    def test_no_timeout_flag_sets_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.exec as exec_mod

        calls: list[ExecArgs] = []

        def fake_run(args: ExecArgs) -> None:
            calls.append(args)

        monkeypatch.setattr(exec_mod, "run", fake_run)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(["exec", "--no-timeout", str(agl_file)])
        assert result.exit_code == 0
        assert calls[0].timeout is None
        assert calls[0].no_timeout is True

    def test_no_log_file_flag_sets_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.commands.exec as exec_mod

        calls: list[ExecArgs] = []

        def fake_run(args: ExecArgs) -> None:
            calls.append(args)

        monkeypatch.setattr(exec_mod, "run", fake_run)
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")

        result = invoke(["exec", "--no-log-file", str(agl_file)])
        assert result.exit_code == 0
        assert calls[0].no_log_file is True
        assert calls[0].log_file is None


# ---------------------------------------------------------------------------
# 2. Mutual exclusivity
# ---------------------------------------------------------------------------


class TestMutualExclusivity:
    """Mutually exclusive flag pairs must exit non-zero."""

    def test_timeout_and_no_timeout_exclusive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        # Intercept exec.run so we're only testing the CLI layer
        monkeypatch.setattr(exec_command, "run", lambda _: None)
        result = invoke(["exec", "--timeout", "30s", "--no-timeout", str(agl_file)])
        assert result.exit_code != 0

    def test_log_file_and_no_log_file_exclusive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        monkeypatch.setattr(exec_command, "run", lambda _: None)
        result = invoke(
            ["exec", "--log-file", "/tmp/trace.log", "--no-log-file", str(agl_file)]
        )
        assert result.exit_code != 0

    def test_no_log_and_log_exclusive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing 3-way log flag exclusivity still works."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        monkeypatch.setattr(exec_command, "run", lambda _: None)
        result = invoke(["exec", "--log", "--no-log", str(agl_file)])
        assert result.exit_code != 0

    def test_no_timeout_alone_is_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        monkeypatch.setattr(exec_command, "run", lambda _: None)
        result = invoke(["exec", "--no-timeout", str(agl_file)])
        assert result.exit_code == 0

    def test_no_log_file_alone_is_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        monkeypatch.setattr(exec_command, "run", lambda _: None)
        result = invoke(["exec", "--no-log-file", str(agl_file)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 3. config_cli projection via spy
# ---------------------------------------------------------------------------


class TestConfigCliProjection:
    """
    Verify that exec.py feeds the correct Value into config_cli for each
    of the six engine keys, using the Option/tri-state projection.

    'Absent' means the key is NOT in config_cli (falls through to source/base).
    """

    def test_log_flag_projects_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _path: None)
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file, log=True))
        assert captured.get("log") == BoolValue(True)

    def test_no_log_flag_projects_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file, no_log=True))
        assert captured.get("log") == BoolValue(False)

    def test_log_absent_not_in_config_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file))
        # Neither --log nor --no-log: key absent from config_cli
        assert "log" not in captured

    def test_log_file_projects_some(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file, log_file="/tmp/trace.log"))
        expected = convert_config_value("log-file", "/tmp/trace.log", OPTION_TEXT_TYPE)
        assert captured.get("log-file") == expected

    def test_no_log_file_projects_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file, no_log_file=True))
        expected = convert_config_value("log-file", None, OPTION_TEXT_TYPE)
        assert captured.get("log-file") == expected

    def test_log_file_absent_not_in_config_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file))
        assert "log-file" not in captured

    def test_timeout_projects_some(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file, timeout="30s"))
        expected = convert_config_value("timeout", "30s", OPTION_TEXT_TYPE)
        assert captured.get("timeout") == expected

    def test_no_timeout_projects_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file, no_timeout=True))
        expected = convert_config_value("timeout", None, OPTION_TEXT_TYPE)
        assert captured.get("timeout") == expected

    def test_timeout_absent_not_in_config_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file))
        assert "timeout" not in captured

    def test_strict_json_already_wired_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file, strict_json=True))
        assert captured.get("strict-json") == BoolValue(True)

    def test_runner_already_wired(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        captured = _spy_config_cli(monkeypatch)
        exec_command.run(_exec_args(agl_file, runner="echo"))
        assert captured.get("runner") == TextValue("echo")


# ---------------------------------------------------------------------------
# 4. CLI > source precedence (behavior tests)
# ---------------------------------------------------------------------------


class TestCliBeatsSource:
    """CLI values override source config declarations at runtime."""

    def test_cli_log_true_beats_source_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--log sets config_cli['log']=true, which overrides source config log = false."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config log = false\n"
            "case log of\n"
            "  | true => print \"logging-on\"\n"
            "  | false => print \"logging-off\"\n"
        )
        # Without --log, source config log = false → prints "logging-off"
        exec_command.run(_exec_args(agl_file, no_log=True))
        assert capsys.readouterr().out.strip() == "logging-off"

        # With --log, CLI wins → "logging-on"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _path: None)
        exec_command.run(_exec_args(agl_file, log=True))
        assert capsys.readouterr().out.strip() == "logging-on"

    def test_cli_no_timeout_beats_source_timeout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--no-timeout sets config_cli['timeout']=none, overriding source timeout = '10s'."""
        agl_file = tmp_path / "prog.agl"
        # Some(value) uses the actual field name 'value' in the Option enum.
        agl_file.write_text(
            'config timeout = "10s"\n'
            "case timeout of\n"
            '  | Some(value) => print "got-timeout"\n'
            '  | None() => print "no-timeout"\n'
        )
        # Without --no-timeout, source config timeout = "10s" → some(...)
        exec_command.run(_exec_args(agl_file, no_log=True))
        assert capsys.readouterr().out.strip() == "got-timeout"

        # With --no-timeout, CLI wins → none
        exec_command.run(_exec_args(agl_file, no_timeout=True, no_log=True))
        assert capsys.readouterr().out.strip() == "no-timeout"

    def test_cli_timeout_some_beats_source_timeout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--timeout 30s sets config_cli['timeout']=some('30s'), overriding bare source."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config timeout\n"
            "case timeout of\n"
            '  | Some(value) => print "got-timeout"\n'
            '  | None() => print "no-timeout"\n'
        )
        # Without --timeout, config_base has none → prints "no-timeout"
        exec_command.run(_exec_args(agl_file, no_log=True))
        assert capsys.readouterr().out.strip() == "no-timeout"

        # With --timeout 30s, CLI wins → some(...)
        exec_command.run(_exec_args(agl_file, timeout="30s", no_log=True))
        assert capsys.readouterr().out.strip() == "got-timeout"

    def test_cli_no_log_file_beats_source_log_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--no-log-file sets config_cli['log-file']=none, overriding source."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            'config log-file = "/tmp/trace.log"\n'
            "case log-file of\n"
            '  | Some(value) => print "has-file"\n'
            '  | None() => print "no-file"\n'
        )
        # Without --no-log-file, source config log-file = "..." → some(...)
        exec_command.run(_exec_args(agl_file, no_log=True))
        assert capsys.readouterr().out.strip() == "has-file"

        # With --no-log-file, CLI wins → none
        exec_command.run(_exec_args(agl_file, no_log_file=True, no_log=True))
        assert capsys.readouterr().out.strip() == "no-file"


# ---------------------------------------------------------------------------
# 5. --timeout/--no-timeout fold into initial shell-exec timeout
# ---------------------------------------------------------------------------


class TestTimeoutFoldsIntoDriver:
    """--timeout/--no-timeout must also fold into the driver's initial timeout."""

    def test_timeout_cli_sets_shell_exec_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.agl.pipeline import PipelineDriver as RealRuntime

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ok"\n')

        captured: dict[str, object] = {}

        class RecordingRuntime(RealRuntime):
            def __init__(
                self,
                *,
                shell_exec_timeout: float | None = None,
                default_loop_limit: int = 5,
                default_strict_json: bool = False,
                default_agent: Any = None,
                default_call_depth_limit: int | None = None,
            ) -> None:
                captured["shell_exec_timeout"] = shell_exec_timeout
                super().__init__(
                    shell_exec_timeout=shell_exec_timeout,
                    default_loop_limit=default_loop_limit,
                    default_strict_json=default_strict_json,
                    default_agent=default_agent,
                    default_call_depth_limit=default_call_depth_limit,
                )

        monkeypatch.setattr(exec_command, "PipelineDriver", RecordingRuntime)
        exec_command.run(_exec_args(agl_file, timeout="60s", no_log=True))
        assert captured["shell_exec_timeout"] == pytest.approx(60.0)

    def test_no_timeout_cli_sets_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.agl.pipeline import PipelineDriver as RealRuntime

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        # Config sets a timeout so we can verify --no-timeout overrides it.
        (home / ".agm" / "config.toml").write_text("[exec]\ntimeout = 30\n")

        from agm.config.context import ConfigContext

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ok"\n')

        captured: dict[str, object] = {}

        class RecordingRuntime(RealRuntime):
            def __init__(
                self,
                *,
                shell_exec_timeout: float | None = None,
                default_loop_limit: int = 5,
                default_strict_json: bool = False,
                default_agent: Any = None,
                default_call_depth_limit: int | None = None,
            ) -> None:
                captured["shell_exec_timeout"] = shell_exec_timeout
                super().__init__(
                    shell_exec_timeout=shell_exec_timeout,
                    default_loop_limit=default_loop_limit,
                    default_strict_json=default_strict_json,
                    default_agent=default_agent,
                    default_call_depth_limit=default_call_depth_limit,
                )

        monkeypatch.setattr(exec_command, "PipelineDriver", RecordingRuntime)
        monkeypatch.setattr(
            exec_command,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        exec_command.run(_exec_args(agl_file, no_timeout=True, no_log=True))
        assert captured["shell_exec_timeout"] is None

    def test_invalid_timeout_value_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("let x = 1\n")
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file, timeout="not-a-timeout"))
        assert exc_info.value.code == 1
        assert "Error" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 6. Verbatim collision check (normalization removed)
# ---------------------------------------------------------------------------


class TestVerbatimCollisions:
    """check_param_collisions uses verbatim flag matching only."""

    def test_param_named_timeout_collides(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('param timeout: text = "30s"\nprint timeout\n')
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1

    def test_param_named_strict_json_collides(self, tmp_path: Path) -> None:
        """param strict-json (kebab) directly collides with --strict-json."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param strict-json: bool = false\nprint strict-json\n")
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1

    def test_param_named_max_iters_kebab_collides(self, tmp_path: Path) -> None:
        """param max-iters (kebab, matches engine key exactly) collides."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param max-iters: int = 5\nprint max-iters\n")
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1

    def test_param_named_max_iters_underscore_no_collision(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """param max_iters (underscore) does NOT collide — verbatim --max_iters ≠ --max-iters."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param max_iters: int = 5\nprint max_iters\n")
        # Should succeed: no collision, param has a default so it runs without --max_iters
        result = exec_command.run(_exec_args(agl_file, no_log=True))
        assert result is None
        assert capsys.readouterr().out.strip() == "5"

    def test_param_bool_no_timeout_polarity_collides(self, tmp_path: Path) -> None:
        """param timeout (bool) generates --no-timeout, colliding with engine --no-timeout."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param timeout: bool = false\nprint timeout\n")
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1

    def test_param_named_log_collides(self, tmp_path: Path) -> None:
        """param log collides with engine key --log."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param log: bool = false\nprint log\n")
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1

    def test_param_named_log_file_collides(self, tmp_path: Path) -> None:
        """param log-file collides with engine key --log-file."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('param log-file: text = ""\nprint log-file\n')
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# 7. Engine flags work with undeclared key programs
# ---------------------------------------------------------------------------


class TestEngineFlags:
    """Engine flags are accepted regardless of whether the program declares the key."""

    def test_timeout_flag_accepted_with_undeclared_program(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--timeout works even if the program has no config timeout declaration."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ok"\n')
        result = exec_command.run(_exec_args(agl_file, timeout="30s", no_log=True))
        assert result is None
        assert capsys.readouterr().out.strip() == "ok"

    def test_no_timeout_flag_accepted_with_undeclared_program(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ok"\n')
        result = exec_command.run(_exec_args(agl_file, no_timeout=True, no_log=True))
        assert result is None
        assert capsys.readouterr().out.strip() == "ok"

    def test_no_log_file_flag_accepted_with_undeclared_program(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ok"\n')
        result = exec_command.run(_exec_args(agl_file, no_log_file=True, no_log=True))
        assert result is None
        assert capsys.readouterr().out.strip() == "ok"

    def test_timeout_flag_with_declared_config_key(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--timeout works when the program explicitly declares config timeout."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "config timeout\n"
            "case timeout of\n"
            '  | Some(value) => print "got-timeout"\n'
            '  | None() => print "no-timeout"\n'
        )
        exec_command.run(_exec_args(agl_file, timeout="30s", no_log=True))
        assert capsys.readouterr().out.strip() == "got-timeout"


# ---------------------------------------------------------------------------
# 8. Help and completion degrade gracefully on bad source
# ---------------------------------------------------------------------------


class TestHelpDegrades:
    """Syntax-error source: help/completion don't crash; error surfaces on normal run."""

    def test_help_for_syntax_error_file_exits_0(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad_file = tmp_path / "bad.agl"
        bad_file.write_text("this is !@# not valid agl\n")
        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=str(bad_file), command=None)
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        # Help header still present
        assert "agm exec" in out
        # No param section (parse failed)
        assert "Program parameters:" not in out

    def test_normal_run_of_syntax_error_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad_file = tmp_path / "bad.agl"
        bad_file.write_text("this is !@# not valid agl\n")
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(bad_file, no_log=True))
        assert exc_info.value.code == 1
        assert capsys.readouterr().err

    def test_help_lists_engine_flags(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Engine flag options appear in help output (Typer renders them)."""
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('print "ok"\n')
        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=str(agl_file), command=None)
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        # At minimum the existing engine flag options must appear
        assert "--strict-json" in out or "strict" in out.lower()


# ---------------------------------------------------------------------------
# 9. Unknown flag → hard usage error
# ---------------------------------------------------------------------------


class TestUnknownFlagError:
    def test_unknown_param_flag_is_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('param msg: text = "ok"\nprint msg\n')
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file, param_tokens=["--totally-unknown"]))
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "error:" in err.lower() or "Error" in err


# ---------------------------------------------------------------------------
# 10. RESERVED_FLAGS derivation sanity
# ---------------------------------------------------------------------------


class TestReservedFlagsDerivation:
    """The engine-key registry drives RESERVED_FLAGS — spot-check key flags present."""

    def test_new_engine_keys_in_reserved_flags(self) -> None:
        from agm.cli_support.exec_params import RESERVED_FLAGS

        # Flags that must be present because they are engine keys
        assert "--timeout" in RESERVED_FLAGS
        assert "--no-timeout" in RESERVED_FLAGS
        assert "--no-log-file" in RESERVED_FLAGS
        assert "--log-file" in RESERVED_FLAGS
        assert "--log" in RESERVED_FLAGS
        assert "--no-log" in RESERVED_FLAGS
        assert "--strict-json" in RESERVED_FLAGS
        assert "--no-strict-json" in RESERVED_FLAGS
        assert "--max-iters" in RESERVED_FLAGS
        assert "--runner" in RESERVED_FLAGS

    def test_builtin_non_engine_flags_in_reserved(self) -> None:
        from agm.cli_support.exec_params import RESERVED_FLAGS

        assert "--command" in RESERVED_FLAGS
        assert "-c" in RESERVED_FLAGS
        assert "--module-path" in RESERVED_FLAGS
        assert "-I" in RESERVED_FLAGS
        assert "--help" in RESERVED_FLAGS
        assert "-h" in RESERVED_FLAGS
        assert "--dry-run" in RESERVED_FLAGS
        assert "--no-stdlib" in RESERVED_FLAGS

    def test_underscore_names_not_in_reserved(self) -> None:
        """Normalization removed: underscore variants are NOT reserved."""
        from agm.cli_support.exec_params import RESERVED_FLAGS

        # These underscore variants must NOT be present
        assert "--max_iters" not in RESERVED_FLAGS
        assert "--strict_json" not in RESERVED_FLAGS
        assert "--dry_run" not in RESERVED_FLAGS
        assert "--module_path" not in RESERVED_FLAGS

    def test_normalize_flag_function_removed(self) -> None:
        """_normalize_flag is removed; it must not be an attribute of the module."""
        import agm.cli_support.exec_params as ep_mod

        assert not hasattr(ep_mod, "_normalize_flag"), (
            "_normalize_flag should have been removed (normalization is gone)"
        )
        assert not hasattr(ep_mod, "_NORMALIZED_RESERVED"), (
            "_NORMALIZED_RESERVED should have been removed"
        )
