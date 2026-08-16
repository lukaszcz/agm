"""Tests for the ``agm repl`` CLI command and its config wiring.

Covers:
- the CLI surface maps each flag onto ``ReplArgs`` (parser-contract style;
  ``repl.run`` is mocked so no real terminal is needed);
- ``--no-log`` / ``--log-file`` are mutually exclusive (usage error, exit 1);
- ``--input`` option has been removed (params resolve eagerly from
  config/defaults; there is no pre-seed CLI option);
- ``repl.run`` resolves ``[exec]`` config, builds a session, and hands off to
  ``run_console`` (mocked) with the echo flag and a history path derived from
  AGM home.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.repl as repl_command
from agm.agl.repl import ReplSession
from agm.agl.runtime.sessions import SessionSnapshot
from agm.agl.semantics.values import EnumValue
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
    def test_bare_repl_defaults(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        result = invoke(runner, ["repl"])
        assert result.exit_code == 0
        args = recorded_runs[0]
        assert getattr(args, "strict_json") is None
        assert getattr(args, "confirm_agents") is False
        assert getattr(args, "quiet") is False
        assert getattr(args, "no_log") is False
        assert getattr(args, "log_file") is None

    def test_input_option_removed(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        # --input has been removed from agm repl.
        result = invoke(runner, ["repl", "--input", "a=1"])
        assert result.exit_code != 0  # unknown option

    def test_strict_json_flag(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        assert invoke(runner, ["repl", "--strict-json"]).exit_code == 0
        assert getattr(recorded_runs[0], "strict_json") is True

    def test_no_strict_json_flag(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        assert invoke(runner, ["repl", "--no-strict-json"]).exit_code == 0
        assert getattr(recorded_runs[0], "strict_json") is False

    def test_agent_flag(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        assert invoke(runner, ["repl", "--agent", 'AgentCommand("echo agent")']).exit_code == 0
        assert getattr(recorded_runs[0], "agent") == 'AgentCommand("echo agent")'

    def test_confirm_agents_flag(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        assert invoke(runner, ["repl", "--confirm-agents"]).exit_code == 0
        assert getattr(recorded_runs[0], "confirm_agents") is True

    def test_quiet_flag(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        assert invoke(runner, ["repl", "--quiet"]).exit_code == 0
        assert getattr(recorded_runs[0], "quiet") is True

    def test_log_file_flag(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        assert invoke(runner, ["repl", "--log-file", "/tmp/r.log"]).exit_code == 0
        assert getattr(recorded_runs[0], "log_file") == "/tmp/r.log"

    def test_no_log_flag(self, runner: CliRunner, recorded_runs: list[object]) -> None:
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
    confirm_agents: bool = False,
    quiet: bool = False,
    no_log: bool = False,
    log: bool = False,
    log_file: str | None = None,
    max_iters: int | None = None,
    agent: str | None = None,
    no_stdlib: bool = False,
) -> ReplArgs:
    """Build ``ReplArgs`` with sensible defaults, overriding named fields."""
    return ReplArgs(
        strict_json=strict_json,
        confirm_agents=confirm_agents,
        quiet=quiet,
        no_log=no_log,
        log=log,
        log_file=log_file,
        max_iters=max_iters,
        agent=agent,
        no_stdlib=no_stdlib,
    )


class TestReplRun:
    def test_repl_exit_closes_the_injected_session_host_and_wires_confirmation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        closed: list[None] = []
        confirmations: list[object] = []

        class SessionHost:
            def close_all(self) -> None:
                closed.append(None)

        host = SessionHost()

        def create_host(**kwargs: object) -> SessionHost:
            confirmations.append(kwargs["confirm_session"])
            return host

        monkeypatch.setattr(repl_command, "create_agl_session_host", create_host)
        repl_command.run(_args())

        assert closed == [None]
        assert len(confirmations) == 1
        assert callable(confirmations[0])

    def test_repl_evaluates_sessions_through_the_injected_host_and_cleans_up_on_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        confirmations: list[tuple[str, str]] = []

        class SessionHost:
            def __init__(self, confirm_session: Callable[[EnumValue, str], None]) -> None:
                self._confirm_session = confirm_session
                self._sessions: dict[str, tuple[EnumValue, str]] = {}
                self._default_handle: str | None = None
                self.opened: list[str] = []
                self.prompts: list[tuple[str, str]] = []
                self.close_calls = 0
                self.closed_handles: set[str] = set()

            def open(self, agent: EnumValue, transport: str, *, name: str = "") -> str:
                del name
                handle = f"session-{len(self._sessions) + 1}"
                self._sessions[handle] = (agent, transport)
                self.opened.append(agent.fields["provider"].value)
                return handle

            def default(self, agent: EnumValue, transport: str, *, name: str = "") -> str:
                if self._default_handle is None:
                    self._default_handle = self.open(agent, transport, name=name)
                return self._default_handle

            def ask(self, handle: str, prompt: str) -> str:
                agent, _transport = self._sessions[handle]
                self._confirm_session(agent, prompt)
                self.prompts.append((handle, prompt))
                return "answer"

            def snapshot(self, handle: str) -> SessionSnapshot:
                agent, transport = self._sessions[handle]
                return SessionSnapshot(agent, transport)

            def close_all(self) -> None:
                self.close_calls += 1
                self.closed_handles.update(self._sessions)

        hosts: list[SessionHost] = []

        def create_host(**kwargs: object) -> SessionHost:
            confirm_session = cast(Callable[[EnumValue, str], None], kwargs["confirm_session"])
            host = SessionHost(confirm_session)
            hosts.append(host)
            return host

        def confirm(agent: str, prompt: str) -> str:
            confirmations.append((agent, prompt))
            return "yes"

        def run_console(session: ReplSession, **_kwargs: object) -> None:
            result = session.eval_entry(
                "import std/config\n"
                'std/config::default-agent := AgentPi("default", "model", "high")\n'
                'let opened = Session::open(AgentPi("opened", "model", "high"))\n'
                "let default = Session::default()\n"
                'opened.ask("open prompt")\n'
                'default.ask("default prompt")'
            )
            assert result.ok

        import agm.agl.repl.console as console_mod

        monkeypatch.setattr(console_mod, "make_console_confirm", lambda: confirm)
        monkeypatch.setattr(console_mod, "run_console", run_console)
        monkeypatch.setattr(repl_command, "create_agl_session_host", create_host)

        repl_command.run(_args(confirm_agents=True))

        assert len(hosts) == 1
        host = hosts[0]
        assert host.opened == ["opened", "default"]
        assert host.prompts == [("session-1", "open prompt"), ("session-2", "default prompt")]
        assert [prompt for _agent, prompt in confirmations] == ["open prompt", "default prompt"]
        assert host.close_calls == 1
        assert host.closed_handles == {"session-1", "session-2"}

    def test_repl_session_ask_traverses_the_shared_confirmation_gate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        from tests._agl_helpers import agent_value

        _isolated_home(monkeypatch, tmp_path)
        confirmations: list[tuple[str, str]] = []
        session_confirmation: list[object] = []

        class SessionHost:
            def close_all(self) -> None:
                pass

        def confirm(agent: str, prompt: str) -> str:
            confirmations.append((agent, prompt))
            return "always"

        import agm.agl.repl.console as console_mod

        monkeypatch.setattr(console_mod, "make_console_confirm", lambda: confirm)

        def create_host(**kwargs: object) -> SessionHost:
            session_confirmation.append(kwargs["confirm_session"])
            return SessionHost()

        monkeypatch.setattr(repl_command, "create_agl_session_host", create_host)

        repl_command.run(_args(confirm_agents=True))

        callback = session_confirmation[0]
        assert callable(callback)
        callback(agent_value("AgentCommand", command="writer"), "continue")

        assert confirmations == [('Agent::AgentCommand(command = "writer")', "continue")]
        from agm.agl.repl.agentmode import AgentMode

        mode = fake_console[0]["agent_mode"]
        assert isinstance(mode, AgentMode)
        assert mode.mode == "auto"

    def test_repl_exit_preserves_keyboard_interrupt_when_cleanup_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)

        class SessionHost:
            def close_all(self) -> None:
                raise RuntimeError("cleanup failed")

        monkeypatch.setattr(
            repl_command, "create_agl_session_host", lambda **_kwargs: SessionHost()
        )

        def interrupt(*_args: object, **_kwargs: object) -> None:
            raise KeyboardInterrupt

        import agm.agl.repl.console as console_mod

        monkeypatch.setattr(console_mod, "run_console", interrupt)
        with pytest.raises(KeyboardInterrupt) as interrupted:
            repl_command.run(_args())
        assert any("cleanup failed" in note for note in interrupted.value.__notes__)

    def test_builds_session_and_runs_console(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        args = ReplArgs(
            strict_json=None,
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
        assert call["history_path"] == home / ".agm" / "repl_history"
        assert (home / ".agm").is_dir()

    def test_invalid_development_package_exits_before_opening_the_console(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path / "home")
        (tmp_path / "package.toml").write_text("not valid TOML")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit):
            repl_command.run(_args())

        assert fake_console == []

    def test_imported_param_resolves_from_qualified_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        from agm.agl.semantics.values import TextValue

        home = _isolated_home(monkeypatch, tmp_path)
        config_dir = home / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            f'[modules]\nroots = ["{tmp_path}"]\n\n[settings]\nregion = "configured"\n'
        )
        (tmp_path / "settings.agl").write_text("param region: text\ndef read() -> text = region\n")

        repl_command.run(_args())
        session: ReplSession = fake_console[0]["session"]
        result = session.eval_entry("import settings\nsettings::read()")

        assert result.ok, result.diagnostics
        assert result.value == TextValue("configured")
        local_result = session.eval_entry('param region: text = "local"\nregion')
        assert local_result.ok, local_result.diagnostics
        assert local_result.value == TextValue("local")

    def test_imported_param_config_value_must_match_its_type(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        config_dir = home / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            f'[modules]\nroots = ["{tmp_path}"]\n\n[settings]\ncount = "not-an-int"\n'
        )
        (tmp_path / "settings.agl").write_text("param count: int\ndef read() -> int = count\n")

        repl_command.run(_args())
        session: ReplSession = fake_console[0]["session"]
        result = session.eval_entry("import settings\nsettings::read()")

        assert not result.ok

    def test_conflicting_imported_param_config_is_an_entry_diagnostic(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        config_dir = home / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            f'[modules]\nroots = ["{tmp_path}"]\n\n[settings]\nregion = "short"\n'
            '["nested/settings"]\nregion = "exact"\n'
        )
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "settings.agl").write_text("param region: text\ndef read() -> text = region\n")

        repl_command.run(_args())
        session: ReplSession = fake_console[0]["session"]
        result = session.eval_entry("import nested/settings\nsettings::read()")

        assert not result.ok

    def test_repl_excludes_the_immutable_store_from_development_discovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        calls: list[Path] = []
        original = repl_command.discover_development_packages

        def discover(anchor: Path, *, home: Path) -> tuple[object, ...]:
            calls.append(home)
            return original(anchor, home=home)

        monkeypatch.setattr(repl_command, "discover_development_packages", discover)

        repl_command.run(_args(no_stdlib=True))

        assert calls == [home]

    def test_repl_mounts_the_current_package_and_its_path_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        from agm.agl.semantics.values import IntValue

        _isolated_home(monkeypatch, tmp_path / "home")
        bravo = tmp_path / "bravo"
        (bravo / "bravo").mkdir(parents=True)
        (bravo / "package.toml").write_text('[package]\nname = "bravo"\nversion = "1.0.0"\n')
        (bravo / "bravo" / "shared.agl").write_text("def answer() -> int = 42\n")

        alpha = tmp_path / "alpha"
        alpha.mkdir()
        (alpha / "alpha").mkdir()
        (alpha / "package.toml").write_text(
            '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
            "[dependencies]\n"
            'bravo = { version = "1", path = "../bravo" }\n'
        )
        (alpha / "alpha" / "main.agl").write_text(
            "import bravo/shared\ndef value() -> int = bravo/shared::answer()\n"
        )
        monkeypatch.chdir(alpha)

        repl_command.run(_args(no_stdlib=True))
        session: ReplSession = fake_console[0]["session"]
        result = session.eval_entry("import alpha/main\nalpha/main::value()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)

    def test_separate_import_entries_validate_the_full_param_inventory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        config_dir = home / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            f'[modules]\nroots = ["{tmp_path}"]\n\n[settings]\nregion = "configured"\n'
        )
        for parent in ("first", "second"):
            module_dir = tmp_path / parent
            module_dir.mkdir()
            (module_dir / "settings.agl").write_text("param region: text\n")

        repl_command.run(_args())
        session: ReplSession = fake_console[0]["session"]
        assert session.eval_entry("import first/settings\n()").ok

        result = session.eval_entry("import second/settings\n()")

        assert not result.ok

    def test_cli_agent_seeds_and_repl_write_persists(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        from agm.agl.semantics.values import EnumValue, TextValue
        from agm.agl.setting_overrides import SettingOverride

        _isolated_home(monkeypatch, tmp_path)
        repl_command.run(_args(agent='AgentCommand("configured")'))
        session: ReplSession = fake_console[0]["session"]
        assert session._setting_overrides["default-agent"] == SettingOverride(
            source='AgentCommand("configured")', origin="--agent"
        )
        assert "default-agent" not in session._engine_seed

        assert session.eval_entry("import std/config").ok
        seeded = session.eval_entry("std/config::default-agent")
        assert seeded.ok
        assert isinstance(seeded.value, EnumValue)
        assert seeded.value.variant == "AgentCommand"
        assert seeded.value.fields["command"] == TextValue("configured")

        assert session.eval_entry('std/config::default-agent := AgentClaude("haiku", "low")').ok
        result = session.eval_entry("std/config::default-agent")
        assert result.ok
        assert isinstance(result.value, EnumValue)
        assert result.value.variant == "AgentClaude"
        assert result.value.fields["model"] == TextValue("haiku")

    def test_cli_agent_override_still_applies_after_reset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        """``:reset`` clears the session's cached stdlib, but the override reapplies."""
        from agm.agl.semantics.values import EnumValue, TextValue

        _isolated_home(monkeypatch, tmp_path)
        repl_command.run(_args(agent='AgentCommand("configured")'))
        session: ReplSession = fake_console[0]["session"]

        assert session.eval_entry("import std/config").ok
        session.reset()

        result = session.eval_entry("import std/config\nstd/config::default-agent")
        assert result.ok
        assert isinstance(result.value, EnumValue)
        assert result.value.variant == "AgentCommand"
        assert result.value.fields["command"] == TextValue("configured")

    def test_exec_config_seeds_each_configured_engine_setting(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        agm_dir = home / ".agm"
        agm_dir.mkdir()
        (agm_dir / "config.toml").write_text(
            "[exec]\n"
            "strict-json = true\n"
            "max-iters = 3\n"
            'timeout = "2s"\n'
            "log = true\n"
            'log-file = "configured.jsonl"\n'
        )
        monkeypatch.setattr(
            repl_command, "prepare_trace_log_from_decision", lambda *args, **kwargs: None
        )

        repl_command.run(_args())

        session: ReplSession = fake_console[0]["session"]
        assert set(session._engine_seed) == {
            "strict-json",
            "max-iters",
            "timeout",
            "log",
            "log-file",
        }

    def test_cli_false_strict_json_and_blank_config_runner_seed_correctly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        from agm.agl.semantics.values import BoolValue

        home = _isolated_home(monkeypatch, tmp_path)
        agm_dir = home / ".agm"
        agm_dir.mkdir()
        (agm_dir / "config.toml").write_text("[exec]\n")

        repl_command.run(_args(strict_json=False))

        session: ReplSession = fake_console[0]["session"]
        assert session._engine_seed["strict-json"] == BoolValue(False)

    def test_history_path_uses_agm_home_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path / "home")
        agm_home = tmp_path / "relocated-agm"
        monkeypatch.setenv("AGM_HOME", str(agm_home))

        repl_command.run(_args())

        call = fake_console[0]
        assert call["history_path"] == agm_home / "repl_history"

    def test_max_iters_flows_into_session_valve(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        """``--max-iters`` resolves to the session's max-iters valve (ON at N)."""
        _isolated_home(monkeypatch, tmp_path)
        repl_command.run(_args(max_iters=10))
        session: ReplSession = fake_console[0]["session"]
        assert session._default_loop_limit == 10

    @pytest.mark.parametrize("limit", [0, -1])
    def test_max_iters_requires_positive_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        limit: int,
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args(max_iters=limit))

        assert exc_info.value.code == 1
        assert fake_console == []

    @pytest.mark.parametrize(
        ("args", "expected_log", "expected_file"),
        [
            (_args(log=True), True, None),
            (_args(no_log=True), False, None),
            (_args(log_file="custom.jsonl"), True, "custom.jsonl"),
        ],
    )
    def test_cli_logging_flags_seed_builtin_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        args: ReplArgs,
        expected_log: bool,
        expected_file: str | None,
    ) -> None:
        from agm.agl.semantics.values import BoolValue, EnumValue, TextValue

        _isolated_home(monkeypatch, tmp_path)
        monkeypatch.setattr(
            repl_command, "prepare_trace_log_from_decision", lambda *args, **kwargs: None
        )
        repl_command.run(args)
        session: ReplSession = fake_console[0]["session"]
        assert session._persisted_host_settings["log"] == BoolValue(expected_log)
        if expected_file is None:
            assert "log-file" not in session._persisted_host_settings
        else:
            log_file = session._persisted_host_settings["log-file"]
            assert isinstance(log_file, EnumValue)
            assert log_file.fields["value"] == TextValue(expected_file)

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
            confirm_agents=False,
            quiet=True,
            no_log=False,
            log_file=None,
        )
        repl_command.run(args)
        assert fake_console[0]["echo"] is False

    def test_blank_agent_literal_exits_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A blank ``--agent`` value is a host-shape error, rejected before the session builds."""
        _isolated_home(monkeypatch, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args(agent=""))

        assert exc_info.value.code == 1
        assert "default-agent" in capsys.readouterr().err
        assert fake_console == []

    def test_malformed_agent_literal_exits_1_before_the_session_builds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A syntactically valid but wrong-typed ``--agent`` literal exits 1 up front.

        Unlike a blank value, ``"true"`` is a non-empty string, so it becomes a
        ``SettingOverride`` resolved by the program's own compilation rather
        than a second throwaway one — but that compilation now runs as part of
        opening the session, before the console (and its banner) ever starts.
        """
        _isolated_home(monkeypatch, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args(agent="true"))

        assert exc_info.value.code == 1
        assert "--agent" in capsys.readouterr().err
        assert fake_console == []

    def test_unparseable_agent_literal_exits_1_before_the_session_builds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A ``--agent`` literal that fails to parse as AgL also exits before the banner."""
        _isolated_home(monkeypatch, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args(agent="("))

        assert exc_info.value.code == 1
        assert "--agent" in capsys.readouterr().err
        assert fake_console == []

    def test_non_constant_agent_literal_exits_1_before_the_session_builds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A well-typed but non-constant ``--agent`` literal also exits before the banner."""
        _isolated_home(monkeypatch, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args(agent='AgentCommand("not " + "constant")'))

        assert exc_info.value.code == 1
        assert "--agent" in capsys.readouterr().err
        assert fake_console == []

    def test_malformed_agent_literal_with_no_stdlib_fails_at_session_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--no-stdlib`` never loads ``std/config``, but ``--agent`` is still an
        explicit request the host cannot silently drop: it must still exit 1 at
        session-open time (before the console starts), not be deferred to a
        later entry that happens to import ``std/config`` (or never come)."""
        _isolated_home(monkeypatch, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args(agent="(", no_stdlib=True))

        assert exc_info.value.code == 1
        assert "--agent" in capsys.readouterr().err
        assert fake_console == []

    def test_config_default_agent_with_no_stdlib_opens_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        """A project-configured ``[exec] default-agent`` is ambient configuration,
        not a request: with ``--no-stdlib`` (``std/config`` never loads), it must
        be inert rather than block the session from opening."""
        home = _isolated_home(monkeypatch, tmp_path)
        agm_dir = home / ".agm"
        agm_dir.mkdir()
        agm_dir.joinpath("config.toml").write_text(
            "[exec]\ndefault-agent = 'AgentCommand(\"echo cfg\")'\n"
        )

        repl_command.run(_args(no_stdlib=True))

        assert len(fake_console) == 1

    def test_stdlib_version_mismatch_exits_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A stdlib version mismatch from root resolution causes exit 1."""
        from agm.config.module_roots import StdlibVersionMismatchError

        _isolated_home(monkeypatch, tmp_path)

        def fake_resolve_stdlib_root(*, home: Path) -> Path:
            raise StdlibVersionMismatchError("0.0.1", "0.1.0")

        monkeypatch.setattr(repl_command, "resolve_stdlib_root", fake_resolve_stdlib_root)

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args())

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "0.0.1" in captured.err
        assert "just install" in captured.err
        assert fake_console == []

    def test_blank_default_agent_config_literal_exits_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        home = _isolated_home(monkeypatch, tmp_path)
        config_dir = home / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\ndefault-agent = ""\n')

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args())

        assert exc_info.value.code == 1
        error = capsys.readouterr().err
        assert "default-agent" in error
        assert fake_console == []

    def test_malformed_default_agent_config_literal_exits_1_before_the_session_builds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A non-blank but malformed ``[exec] default-agent`` exits 1 up front, naming it."""
        home = _isolated_home(monkeypatch, tmp_path)
        config_dir = home / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\ndefault-agent = "not an agent"\n')

        with pytest.raises(SystemExit) as exc_info:
            repl_command.run(_args())

        assert exc_info.value.code == 1
        assert "[exec] default-agent" in capsys.readouterr().err
        assert fake_console == []

    def test_exec_runner_config_seeds_default_agent_as_agent_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        """``[exec] runner`` is a bare host command, decoded into an ``AgentCommand`` value seed."""
        from agm.agl.semantics.values import EnumValue, TextValue

        home = _isolated_home(monkeypatch, tmp_path)
        config_dir = home / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[exec]\nrunner = "claude"\n')

        repl_command.run(_args())

        session: ReplSession = fake_console[0]["session"]
        seeded = session._engine_seed["default-agent"]
        assert isinstance(seeded, EnumValue)
        assert seeded.variant == "AgentCommand"
        assert seeded.fields["command"] == TextValue("claude")
        assert session._setting_overrides == {}

    def test_invalid_config_exits_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)

        def boom(*_args: object, **_kwargs: object) -> object:
            raise ValueError("bad config")

        monkeypatch.setattr(repl_command, "exec_config_from_merged", boom)
        args = ReplArgs(
            strict_json=None,
            confirm_agents=False,
            quiet=False,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            repl_command.run(args)
        assert excinfo.value.code == 1
        assert fake_console == []

    def test_numeric_timeout_in_config_accepted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        """An int-typed [exec].timeout in the TOML config is accepted."""
        home = _isolated_home(monkeypatch, tmp_path)
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True, exist_ok=True)
        (agm_dir / "config.toml").write_text("[exec]\ntimeout = 30\n")
        repl_command.run(_args())
        assert len(fake_console) == 1

    def test_tiny_numeric_timeout_round_trips_through_builtin_setting(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        from agm.agl.semantics.values import EnumValue, TextValue

        home = _isolated_home(monkeypatch, tmp_path)
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True, exist_ok=True)
        (agm_dir / "config.toml").write_text("[exec]\ntimeout = 0.0000001\n")

        repl_command.run(_args())
        session = fake_console[0]["session"]
        assert isinstance(session, ReplSession)
        result = session.eval_entry(
            "import std/config\nstd/config::timeout := std/config::timeout\nstd/config::timeout"
        )

        assert result.ok
        assert isinstance(result.value, EnumValue)
        assert result.value.fields["value"] == TextValue("0.0000001s")

    def test_string_timeout_in_config_accepted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_console: list[dict[str, object]],
    ) -> None:
        """A string-typed [exec].timeout in the TOML config is accepted."""
        home = _isolated_home(monkeypatch, tmp_path)
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True, exist_ok=True)
        (agm_dir / "config.toml").write_text('[exec]\ntimeout = "30s"\n')
        repl_command.run(_args())
        assert len(fake_console) == 1


# ---------------------------------------------------------------------------
# Wiring: agent mode and trace path resolution
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


class TestReplModuleRoots:
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
        # Put a minimal module in lib_dir so we can verify it is importable.
        (lib_dir / "mymod.agl").write_text('def greet() -> text = "hello"\n', encoding="utf-8")
        agm_home = home / ".agm"
        agm_home.mkdir(parents=True, exist_ok=True)
        (agm_home / "config.toml").write_text(f"[modules]\nlib_root = {str(lib_dir)!r}\n")
        repl_command.run(_args())
        session = fake_console[0]["session"]
        assert isinstance(session, ReplSession)
        # The lib_root is wired: importing a module from lib_dir succeeds.
        result = session.eval_entry("import mymod")
        assert result.ok


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
