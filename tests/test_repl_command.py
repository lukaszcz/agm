"""Tests for the ``agm repl`` CLI command and its config wiring.

Covers:
- the CLI surface maps each flag onto ``ReplArgs`` (parser-contract style;
  ``repl.run`` is mocked so no real terminal is needed);
- ``--no-log`` / ``--log-file`` are mutually exclusive (usage error, exit 2);
- ``repl.run`` resolves ``[exec]`` config, builds a session, and hands off to
  ``run_console`` (mocked) with the echo flag and a history path derived from
  the config-context home.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.repl as repl_command
from agm.agl.repl import ReplSession
from agm.commands.args import ReplArgs


class RecordedArgs(Protocol):
    def __getattr__(self, name: str) -> object: ...


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner: CliRunner, argv: list[str]) -> Result:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm", catch_exceptions=False)


@pytest.fixture()
def recorded_runs(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch ``repl.run`` to record its ``ReplArgs`` instead of starting a REPL."""
    calls: list[object] = []

    def fake_run(args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(repl_command, "run", fake_run)
    return calls


# ---------------------------------------------------------------------------
# Parser-contract: flags → ReplArgs
# ---------------------------------------------------------------------------


class TestReplArgsParsing:
    def test_bare_repl_defaults(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl"])
        assert result.exit_code == 0
        args = recorded_runs[0]
        assert getattr(args, "inputs") == []
        assert getattr(args, "strict_json") is None
        assert getattr(args, "max_iters") is None
        assert getattr(args, "runner") is None
        assert getattr(args, "auto_agents") is False
        assert getattr(args, "quiet") is False
        assert getattr(args, "no_log") is False
        assert getattr(args, "log_file") is None

    def test_input_repeatable(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl", "--input", "a=1", "--input", "b=2"])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "inputs") == ["a=1", "b=2"]

    def test_strict_json_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--strict-json"]).exit_code == 0
        assert getattr(recorded_runs[0], "strict_json") is True

    def test_no_strict_json_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--no-strict-json"]).exit_code == 0
        assert getattr(recorded_runs[0], "strict_json") is False

    def test_max_iters_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--max-iters", "9"]).exit_code == 0
        assert getattr(recorded_runs[0], "max_iters") == 9

    def test_runner_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--runner", "claude -p"]).exit_code == 0
        assert getattr(recorded_runs[0], "runner") == "claude -p"

    def test_auto_agents_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--auto-agents"]).exit_code == 0
        assert getattr(recorded_runs[0], "auto_agents") is True

    def test_quiet_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--quiet"]).exit_code == 0
        assert getattr(recorded_runs[0], "quiet") is True

    def test_log_file_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--log-file", "/tmp/r.log"]).exit_code == 0
        assert getattr(recorded_runs[0], "log_file") == "/tmp/r.log"

    def test_no_log_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--no-log"]).exit_code == 0
        assert getattr(recorded_runs[0], "no_log") is True


class TestReplMutualExclusion:
    def test_no_log_and_log_file_conflict(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl", "--no-log", "--log-file", "/tmp/x.log"])
        assert result.exit_code == 1
        assert recorded_runs == []  # never dispatched


# ---------------------------------------------------------------------------
# repl.run config resolution + handoff to run_console
# ---------------------------------------------------------------------------


class _ReplConsoleCall(Protocol):
    session: ReplSession
    echo: bool
    history_path: Path


@pytest.fixture()
def fake_console(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Patch the console entry point so ``repl.run`` never opens a terminal."""
    calls: list[dict[str, object]] = []

    def fake_run_console(
        session: ReplSession,
        *,
        echo: bool = True,
        check_only: bool = False,
        history_path: Path | None = None,
        input: object = None,
        output: object = None,
    ) -> None:
        calls.append(
            {
                "session": session,
                "echo": echo,
                "check_only": check_only,
                "history_path": history_path,
            }
        )

    # The command imports ``run_console`` lazily from the console module.
    import agm.agl.repl.console as console_mod

    monkeypatch.setattr(console_mod, "run_console", fake_run_console)
    return calls


def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the config context at an isolated HOME with no project config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
    return tmp_path


class TestReplRun:
    def test_builds_session_and_runs_console(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        args = ReplArgs(
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner="echo agent",
            auto_agents=False,
            quiet=False,
            no_log=False,
            log_file=None,
        )
        repl_command.run(args)

        assert len(fake_console) == 1
        call = fake_console[0]
        assert isinstance(call["session"], ReplSession)
        assert call["echo"] is True
        assert call["check_only"] is False  # not a dry-run by default
        assert call["history_path"] == home / "repl_history"

    def test_dry_run_runs_console_in_check_only_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        # ``--dry-run`` sets the shared global flag; the REPL honours it by
        # driving the console in type-check-only mode.
        from agm.core import dry_run

        _isolated_home(monkeypatch, tmp_path)
        monkeypatch.setattr(dry_run, "enabled", lambda: True)
        args = ReplArgs(
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner="echo agent",
            auto_agents=False,
            quiet=False,
            no_log=False,
            log_file=None,
        )
        repl_command.run(args)
        assert fake_console[0]["check_only"] is True

    def test_quiet_disables_echo(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        args = ReplArgs(
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner="echo agent",
            auto_agents=False,
            quiet=True,
            no_log=False,
            log_file=None,
        )
        repl_command.run(args)
        assert fake_console[0]["echo"] is False

    def test_invalid_runner_exits_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        args = ReplArgs(
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner='broken "quote',  # unbalanced quote → split_command raises
            auto_agents=False,
            quiet=False,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            repl_command.run(args)
        assert excinfo.value.code == 1
        assert fake_console == []

    def test_invalid_config_exits_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)

        def boom(**_kwargs: object) -> object:
            raise ValueError("bad config")

        monkeypatch.setattr(repl_command, "load_exec_config", boom)
        args = ReplArgs(
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            auto_agents=False,
            quiet=False,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            repl_command.run(args)
        assert excinfo.value.code == 1
        assert fake_console == []
