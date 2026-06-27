"""Tests for resolve_log_decision and the refactored log helpers (Part A).

Coverage:
- resolve_log_decision: full precedence table (CLI > pragma > config), default off,
  path resolution, explicit disable beats lower-layer enable.
- resolve_log_file / prepare_trace_log with enabled/log_file shape.
- --log flag parsing + mutual-exclusivity rejection in typer (cli.py) parser.
- Integration: default run writes no trace; --log writes one; [exec] log=true writes one;
  --no-log overrides config log=true.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
import agm.commands.exec as exec_command
import agm.commands.repl as repl_command
from agm.cli_support.args import ExecArgs
from agm.core.log import LogDecision, resolve_log_decision, resolve_log_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def invoke(runner: CliRunner, argv: list[str]) -> object:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm", catch_exceptions=False)


def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests: resolve_log_decision
# ---------------------------------------------------------------------------


class TestResolveLogDecisionDefaults:
    """Default (all unset/False/None) → disabled, no path."""

    def test_all_defaults_disabled(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        assert d == LogDecision(enabled=False, explicit_path=None)

    def test_returns_frozen_dataclass(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            d.enabled = True  # type: ignore[misc]


class TestResolveLogDecisionCliLayer:
    """CLI flags take highest precedence."""

    def test_cli_log_enables(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=True,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        assert d.enabled is True
        assert d.explicit_path is None

    def test_cli_no_log_disables(self) -> None:
        d = resolve_log_decision(
            cli_no_log=True,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        assert d.enabled is False
        assert d.explicit_path is None

    def test_cli_log_file_enables_with_path(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file="/tmp/trace.jsonl",
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        assert d.enabled is True
        assert d.explicit_path == "/tmp/trace.jsonl"

    def test_cli_no_log_overrides_config_log_true(self) -> None:
        """CLI --no-log beats config log=true."""
        d = resolve_log_decision(
            cli_no_log=True,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=True,
            config_log_file=None,
        )
        assert d.enabled is False

    def test_cli_no_log_overrides_config_log_file(self) -> None:
        """CLI --no-log beats config log_file setting."""
        d = resolve_log_decision(
            cli_no_log=True,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file="/tmp/config.jsonl",
        )
        assert d.enabled is False

    def test_cli_no_log_overrides_pragma_log_true(self) -> None:
        """CLI --no-log beats pragma log=true."""
        d = resolve_log_decision(
            cli_no_log=True,
            cli_log=False,
            cli_log_file=None,
            pragma_log=True,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        assert d.enabled is False

    def test_cli_log_file_path_beats_pragma_path(self) -> None:
        """CLI --log-file path takes precedence over pragma_log_file."""
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file="/cli/path.jsonl",
            pragma_log=None,
            pragma_log_file="/pragma/path.jsonl",
            config_log=False,
            config_log_file="/config/path.jsonl",
        )
        assert d.explicit_path == "/cli/path.jsonl"

    def test_cli_log_file_path_beats_config_path(self) -> None:
        """CLI --log-file path takes precedence over config log_file."""
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file="/cli/path.jsonl",
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file="/config/path.jsonl",
        )
        assert d.explicit_path == "/cli/path.jsonl"


class TestResolveLogDecisionPragmaLayer:
    """Pragma layer: between CLI and config."""

    def test_pragma_log_true_enables(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=True,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        assert d.enabled is True
        assert d.explicit_path is None

    def test_pragma_log_false_disables(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=False,
            pragma_log_file=None,
            config_log=True,
            config_log_file=None,
        )
        assert d.enabled is False

    def test_pragma_log_file_enables_with_path(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file="/pragma/trace.jsonl",
            config_log=False,
            config_log_file=None,
        )
        assert d.enabled is True
        assert d.explicit_path == "/pragma/trace.jsonl"

    def test_pragma_log_file_path_beats_config_path(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file="/pragma/trace.jsonl",
            config_log=False,
            config_log_file="/config/trace.jsonl",
        )
        assert d.explicit_path == "/pragma/trace.jsonl"

    def test_pragma_disabled_beats_config_enabled(self) -> None:
        """pragma_log=False disables even when config_log=True."""
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=False,
            pragma_log_file=None,
            config_log=True,
            config_log_file=None,
        )
        assert d.enabled is False


class TestResolveLogDecisionConfigLayer:
    """Config layer: lowest priority."""

    def test_config_log_true_enables(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=True,
            config_log_file=None,
        )
        assert d.enabled is True
        assert d.explicit_path is None

    def test_config_log_file_enables(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file="/config/trace.jsonl",
        )
        assert d.enabled is True
        assert d.explicit_path == "/config/trace.jsonl"

    def test_config_log_false_does_not_enable(self) -> None:
        """config_log=False (default) with no other flags → still disabled."""
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        assert d.enabled is False


class TestResolveLogDecisionPrecedence:
    """Verify full CLI > pragma > config precedence chain."""

    def test_cli_wins_over_pragma_and_config(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=True,
            cli_log_file=None,
            pragma_log=False,
            pragma_log_file=None,
            config_log=True,
            config_log_file=None,
        )
        # CLI says ENABLE; pragma says disable — CLI wins.
        assert d.enabled is True

    def test_pragma_wins_over_config(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=True,
            pragma_log_file=None,
            config_log=False,
            config_log_file=None,
        )
        assert d.enabled is True

    def test_path_precedence_cli_over_pragma_over_config(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file="/cli.jsonl",
            pragma_log=None,
            pragma_log_file="/pragma.jsonl",
            config_log=False,
            config_log_file="/config.jsonl",
        )
        assert d.explicit_path == "/cli.jsonl"

    def test_path_precedence_pragma_over_config(self) -> None:
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=False,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file="/pragma.jsonl",
            config_log=False,
            config_log_file="/config.jsonl",
        )
        assert d.explicit_path == "/pragma.jsonl"

    def test_path_none_when_only_config_and_enabled_by_cli(self) -> None:
        """CLI --log (no path) + config log_file → path comes from config."""
        d = resolve_log_decision(
            cli_no_log=False,
            cli_log=True,
            cli_log_file=None,
            pragma_log=None,
            pragma_log_file=None,
            config_log=False,
            config_log_file="/config/trace.jsonl",
        )
        assert d.enabled is True
        # CLI enables but provides no path; config provides the path
        assert d.explicit_path == "/config/trace.jsonl"


# ---------------------------------------------------------------------------
# Unit tests: resolve_log_file with new enabled/log_file signature
# ---------------------------------------------------------------------------


class TestResolveLogFileNewShape:
    def test_disabled_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        result = resolve_log_file(command_name="exec", enabled=False, log_file=None)
        assert result is None

    def test_enabled_no_path_returns_auto_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        result = resolve_log_file(command_name="exec", enabled=True, log_file=None)
        assert result is not None
        assert result.name.startswith("exec-")
        assert result.suffix == ".log"

    def test_enabled_with_explicit_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        explicit = str(tmp_path / "my.jsonl")
        result = resolve_log_file(command_name="exec", enabled=True, log_file=explicit)
        assert result is not None
        assert result == Path(explicit)

    def test_enabled_with_relative_path_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = resolve_log_file(command_name="exec", enabled=True, log_file="out.jsonl")
        assert result is not None
        assert result.is_absolute()
        assert result == tmp_path / "out.jsonl"

    def test_unique_flag_differentiates_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        from datetime import datetime as _dt

        fixed = _dt(2026, 1, 1, 12, 0, 0)
        with patch("agm.core.log.datetime") as mock_dt, patch("agm.core.log.os.getpid") as mock_pid:
            mock_dt.now.return_value = fixed
            mock_pid.return_value = 11111
            path_a = resolve_log_file(
                command_name="exec", enabled=True, log_file=None, unique=True
            )
            mock_pid.return_value = 22222
            path_b = resolve_log_file(
                command_name="exec", enabled=True, log_file=None, unique=True
            )
        assert path_a != path_b


# ---------------------------------------------------------------------------
# CLI parsing: --log flag (typer, cli.py)
# ---------------------------------------------------------------------------


class TestExecLogFlagParsing:
    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture()
    def recorded_runs(self, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        calls: list[object] = []

        def fake_run(args: object) -> None:
            calls.append(args)

        monkeypatch.setattr(exec_command, "run", fake_run)
        return calls

    def test_log_flag_sets_log_true(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", "--log", str(agl_file)])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "log") is True

    def test_log_and_no_log_mutually_exclusive(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", "--log", "--no-log", str(agl_file)])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_log_and_log_file_mutually_exclusive(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", "--log", "--log-file", "/tmp/x.jsonl", str(agl_file)])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_default_log_is_false(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", str(agl_file)])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "log") is False


class TestReplLogFlagParsing:
    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture()
    def recorded_runs(self, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        calls: list[object] = []

        def fake_run(args: object) -> None:
            calls.append(args)

        monkeypatch.setattr(repl_command, "run", fake_run)
        return calls

    def test_log_flag_sets_log_true(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl", "--log"])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "log") is True

    def test_log_and_no_log_mutually_exclusive(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl", "--log", "--no-log"])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_log_and_log_file_mutually_exclusive(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl", "--log", "--log-file", "/tmp/x.jsonl"])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_default_log_is_false(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl"])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "log") is False


# ---------------------------------------------------------------------------
# Integration tests: trace file presence
# ---------------------------------------------------------------------------


def _exec_args(
    command: str,
    *,
    log: bool = False,
    no_log: bool = False,
    log_file: str | None = None,
) -> ExecArgs:
    return ExecArgs(
        file=None,
        command=command,
        param_tokens=[],
        strict_json=None,
        runner=None,
        log=log,
        no_log=no_log,
        log_file=log_file,
    )


class TestIntegrationDefaultNoTrace:
    """Default run (no log flags) writes NO trace file."""

    def test_default_exec_writes_no_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        exec_command.run(_exec_args('print "hello"'))
        agent_files = tmp_path / ".agent-files"
        assert not agent_files.exists(), "No .agent-files dir should be created by default"


class TestIntegrationLogFlagWritesTrace:
    """--log flag causes a trace file to be written."""

    def test_log_flag_creates_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        exec_command.run(_exec_args('print "hello"', log=True))
        agent_files = tmp_path / ".agent-files"
        assert agent_files.exists()
        log_files = list(agent_files.glob("exec-*.log"))
        assert len(log_files) == 1

    def test_explicit_log_file_path_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        log_path = tmp_path / "my_trace.jsonl"
        exec_command.run(_exec_args('print "hi"', log_file=str(log_path)))
        assert log_path.exists()


class TestIntegrationConfigLogTrue:
    """[exec] log=true in config causes a trace file to be written."""

    def test_config_log_true_creates_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text("[exec]\nlog = true\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        exec_command.run(_exec_args('print "hello"'))
        agent_files = tmp_path / ".agent-files"
        assert agent_files.exists()
        log_files = list(agent_files.glob("exec-*.log"))
        assert len(log_files) == 1

    def test_config_log_file_creates_trace_at_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        log_path = tmp_path / "config_trace.jsonl"
        (home / ".agm" / "config.toml").write_text(
            f"[exec]\nlog_file = {str(log_path)!r}\n"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
        exec_command.run(_exec_args('print "hello"'))
        assert log_path.exists()


class TestIntegrationNoLogOverridesConfig:
    """--no-log overrides config log=true."""

    def test_no_log_overrides_config_log_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text("[exec]\nlog = true\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        exec_command.run(_exec_args('print "hello"', no_log=True))
        agent_files = tmp_path / ".agent-files"
        assert not agent_files.exists(), "--no-log must override config log=true"
