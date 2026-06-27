"""Tests for the ``agm repl`` CLI command and its config wiring.

Covers:
- the CLI surface maps each flag onto ``ReplArgs`` (parser-contract style;
  ``repl.run`` is mocked so no real terminal is needed);
- ``--no-log`` / ``--log-file`` are mutually exclusive (usage error, exit 2);
- ``--input`` option has been REMOVED in M6 (params resolve eagerly from
  config/defaults; there is no pre-seed CLI option);
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
from agm.cli_support.args import ReplArgs


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
        assert getattr(args, "strict_json") is None
        assert getattr(args, "runner") is None
        assert getattr(args, "confirm_agents") is False
        assert getattr(args, "quiet") is False
        assert getattr(args, "no_log") is False
        assert getattr(args, "log_file") is None

    def test_input_option_removed(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        # M6: --input has been removed from agm repl.
        result = invoke(runner, ["repl", "--input", "a=1"])
        assert result.exit_code != 0  # unknown option

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

    def test_runner_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--runner", "claude -p"]).exit_code == 0
        assert getattr(recorded_runs[0], "runner") == "claude -p"

    def test_confirm_agents_flag(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        assert invoke(runner, ["repl", "--confirm-agents"]).exit_code == 0
        assert getattr(recorded_runs[0], "confirm_agents") is True

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
        agent_mode: object = None,
        history_path: Path | None = None,
        theme: str = "auto",
        on_theme_save: object = None,
        input: object = None,
        output: object = None,
    ) -> None:
        calls.append(
            {
                "session": session,
                "echo": echo,
                "check_only": check_only,
                "agent_mode": agent_mode,
                "history_path": history_path,
                "theme": theme,
                "on_theme_save": on_theme_save,
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


def _args(
    *,
    strict_json: bool | None = None,
    runner: str | None = "echo agent",
    confirm_agents: bool = False,
    quiet: bool = False,
    no_log: bool = False,
    log_file: str | None = None,
) -> ReplArgs:
    """Build ``ReplArgs`` with sensible defaults, overriding named fields."""
    return ReplArgs(
        strict_json=strict_json,
        runner=runner,
        confirm_agents=confirm_agents,
        quiet=quiet,
        no_log=no_log,
        log_file=log_file,
    )


class TestReplRun:
    def test_builds_session_and_runs_console(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        args = ReplArgs(
            strict_json=None,
            runner="echo agent",
            confirm_agents=False,
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
            strict_json=None,
            runner="echo agent",
            confirm_agents=False,
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
            strict_json=None,
            runner="echo agent",
            confirm_agents=False,
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
            strict_json=None,
            runner='broken "quote',  # unbalanced quote → split_command raises
            confirm_agents=False,
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
            strict_json=None,
            runner=None,
            confirm_agents=False,
            quiet=False,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            repl_command.run(args)
        assert excinfo.value.code == 1
        assert fake_console == []


# ---------------------------------------------------------------------------
# M4 wiring: agent mode and trace path resolution
# ---------------------------------------------------------------------------


class TestReplAgentMode:
    def test_default_mode_is_auto(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        repl_command.run(_args())
        mode = fake_console[0]["agent_mode"]
        from agm.agl.repl.agentmode import AgentMode

        assert isinstance(mode, AgentMode)
        assert mode.mode == "auto"

    def test_confirm_agents_starts_in_confirm(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        repl_command.run(_args(confirm_agents=True))
        mode = fake_console[0]["agent_mode"]
        from agm.agl.repl.agentmode import AgentMode

        assert isinstance(mode, AgentMode)
        assert mode.mode == "confirm"


class TestReplParamsConfigLoader:
    def test_params_config_loader_wired_from_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        # The session built by repl.run should have a params_config_loader
        # that reads from the config context.  We verify by checking the
        # session can be used normally (the loader is injected, not null).
        _isolated_home(monkeypatch, tmp_path)
        repl_command.run(_args())
        session = fake_console[0]["session"]
        assert isinstance(session, ReplSession)
        # Params with defaults resolve eagerly (no pre-seed needed).
        r = session.eval_entry("param greeting = \"hi\"")
        assert r.ok

    def test_params_config_loader_invoked_on_program_decl(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        # The loader closure must actually be called when a program decl is
        # entered, exercising the load_params_config wiring.
        _isolated_home(monkeypatch, tmp_path)
        # Patch load_params_config to track calls and return an empty table.
        loader_calls: list[str] = []

        def fake_load_params_config(
            program_name: str, *, home: Path, proj_dir: object, cwd: Path
        ) -> dict[str, object]:
            loader_calls.append(program_name)
            return {}

        monkeypatch.setattr(repl_command, "load_params_config", fake_load_params_config)
        repl_command.run(_args())
        session = fake_console[0]["session"]
        assert isinstance(session, ReplSession)
        r = session.eval_entry("program myapp")
        assert r.ok
        assert loader_calls == ["myapp"]

    def test_configured_lib_root_wired_into_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        """When [modules] lib_root is set in config, it is resolved and passed to the session."""
        home = _isolated_home(monkeypatch, tmp_path)
        # Create an AGM home config with a lib_root pointing to a local dir.
        lib_dir = tmp_path / "mylib"
        lib_dir.mkdir()
        agm_home = home / ".agm"
        agm_home.mkdir(parents=True, exist_ok=True)
        (agm_home / "config.toml").write_text(
            f"[modules]\nlib_root = {str(lib_dir)!r}\n"
        )
        repl_command.run(_args())
        session = fake_console[0]["session"]
        assert isinstance(session, ReplSession)
        # The session's _lib_root should be the resolved absolute lib dir.
        assert session._lib_root == lib_dir


class TestReplTrace:
    def test_log_file_threaded_into_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        log_file = tmp_path / "trace.log"
        repl_command.run(_args(log_file=str(log_file)))
        # The validate-up-front touch creates the (empty) file.
        assert log_file.exists()
        session = fake_console[0]["session"]
        assert isinstance(session, ReplSession)

    def test_no_log_writes_no_trace(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        repl_command.run(_args(no_log=True))
        # Nothing under .agent-files was created for a --no-log session.
        assert not (tmp_path / ".agent-files").exists()

    def test_dry_run_writes_no_trace(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        from agm.core import dry_run

        _isolated_home(monkeypatch, tmp_path)
        monkeypatch.setattr(dry_run, "enabled", lambda: True)
        log_file = tmp_path / "trace.log"
        repl_command.run(_args(log_file=str(log_file)))
        # Dry-run is side-effect-free: the trace path is never touched.
        assert not log_file.exists()

    def test_unwritable_log_file_exits_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        # A path whose parent is a regular file cannot be created (mkdir fails).
        not_a_dir = tmp_path / "afile"
        not_a_dir.write_text("x")
        log_file = not_a_dir / "trace.log"
        with pytest.raises(SystemExit) as excinfo:
            repl_command.run(_args(log_file=str(log_file)))
        assert excinfo.value.code == 1
        assert fake_console == []
