"""Tests for the trace decision and the trace helpers.

Coverage:
- EngineSeedTiers.trace_decision: initial-value precedence (CLI > config file),
  default off, path resolution, explicit disable beats lower-layer enable, and
  agreement with the ``trace`` engine seed the same tiers produce.
- resolve_log_file / prepare_trace_log with their enabled/path shapes.
- --trace flag parsing + mutual-exclusivity rejection in typer (cli.py) parser.
- Integration: default run writes no trace; --trace writes one; [exec] trace=true writes one;
  --no-trace overrides config trace=true.
"""

from __future__ import annotations

from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
import agm.commands.exec as exec_command
import agm.commands.repl as repl_command
from agm.agl.semantics.values import BoolValue
from agm.cli_support.args import ExecArgs
from agm.cli_support.engine_seeds import EngineSeedTiers, build_host_engine_seeds
from agm.config.general import exec_config_from_merged
from agm.core.log import LiveTracePathResolver, TraceDecision, resolve_log_file

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
# Unit tests: EngineSeedTiers.trace_decision
# ---------------------------------------------------------------------------


def _tiers(
    *,
    cli: dict[str, object | None] | None = None,
    exec_table: dict[str, object] | None = None,
) -> EngineSeedTiers:
    """Build the seed tiers ``agm exec``/``agm repl`` resolve their trace from."""
    table = dict(exec_table or {})
    return build_host_engine_seeds(
        config=exec_config_from_merged({"exec": table}),
        primary_table=table,
        cli_values=dict(cli or {}),
    )


def _decision(
    *,
    cli: dict[str, object | None] | None = None,
    exec_table: dict[str, object] | None = None,
) -> TraceDecision:
    return _tiers(cli=cli, exec_table=exec_table).trace_decision()


class TestTraceDecisionDefaults:
    """Default (no flags, no config) -> disabled, no path."""

    def test_all_defaults_disabled(self) -> None:
        assert _decision() == TraceDecision(enabled=False, explicit_path=None)

    def test_returns_frozen_dataclass(self) -> None:
        decision = _decision()
        with pytest.raises((AttributeError, TypeError)):
            setattr(decision, "enabled", True)


class TestTraceDecisionCliLayer:
    """CLI flags take highest precedence."""

    def test_cli_trace_enables(self) -> None:
        decision = _decision(cli={"trace": True})
        assert decision.enabled is True
        assert decision.explicit_path is None

    def test_cli_no_trace_disables(self) -> None:
        decision = _decision(cli={"trace": False})
        assert decision.enabled is False
        assert decision.explicit_path is None

    def test_cli_trace_file_enables_with_path(self) -> None:
        decision = _decision(cli={"trace-file": "/tmp/trace.jsonl"})
        assert decision.enabled is True
        assert decision.explicit_path == "/tmp/trace.jsonl"

    def test_cli_no_trace_overrides_config_trace_true(self) -> None:
        """CLI --no-trace beats config trace=true."""
        assert _decision(cli={"trace": False}, exec_table={"trace": True}).enabled is False

    def test_cli_no_trace_overrides_config_trace_file(self) -> None:
        """CLI --no-trace beats a configured trace-file."""
        decision = _decision(cli={"trace": False}, exec_table={"trace-file": "/tmp/config.jsonl"})
        assert decision.enabled is False

    def test_cli_trace_file_path_beats_config_path(self) -> None:
        decision = _decision(
            cli={"trace-file": "/cli/path.jsonl"},
            exec_table={"trace-file": "/config/path.jsonl"},
        )
        assert decision.explicit_path == "/cli/path.jsonl"

    def test_cleared_cli_trace_file_keeps_the_configured_destination(self) -> None:
        """``--no-trace-file`` clears only the CLI seed, not a configured path."""
        decision = _decision(
            cli={"trace-file": None}, exec_table={"trace-file": "/config/path.jsonl"}
        )
        assert decision.enabled is True
        assert decision.explicit_path == "/config/path.jsonl"


class TestTraceDecisionConfigLayer:
    """Config layer: lowest priority."""

    def test_config_trace_true_enables(self) -> None:
        decision = _decision(exec_table={"trace": True})
        assert decision.enabled is True
        assert decision.explicit_path is None

    def test_config_trace_file_enables(self) -> None:
        decision = _decision(exec_table={"trace-file": "/config/trace.jsonl"})
        assert decision.enabled is True
        assert decision.explicit_path == "/config/trace.jsonl"

    def test_config_trace_false_does_not_enable(self) -> None:
        assert _decision(exec_table={"trace": False}).enabled is False


class TestTraceDecisionPrecedence:
    """Verify the CLI > config-file chain where the two layers disagree."""

    def test_cli_enable_takes_the_path_from_config(self) -> None:
        """CLI --trace (no path) + config trace-file -> path comes from config."""
        decision = _decision(cli={"trace": True}, exec_table={"trace-file": "/config/trace.jsonl"})
        assert decision.enabled is True
        assert decision.explicit_path == "/config/trace.jsonl"


@pytest.mark.parametrize(
    "cli,exec_table",
    [
        ({}, {}),
        ({"trace": True}, {}),
        ({"trace": False}, {"trace": True}),
        ({"trace-file": "/cli/t.jsonl"}, {}),
        ({"trace-file": None}, {"trace-file": "/config/t.jsonl"}),
        ({}, {"trace": True}),
        ({}, {"trace-file": "/config/t.jsonl"}),
        ({}, {"trace": False, "trace-file": "/config/t.jsonl"}),
    ],
)
def test_decision_agrees_with_the_seeded_trace_setting(
    cli: dict[str, object | None], exec_table: dict[str, object]
) -> None:
    """The file a run opens and the ``trace`` a program reads come from one rule."""
    tiers = _tiers(cli=cli, exec_table=exec_table)
    seed = tiers.merged().get("trace")
    seeded = isinstance(seed, BoolValue) and seed.value
    assert tiers.trace_decision().enabled is seeded


# ---------------------------------------------------------------------------
# Unit tests: resolve_log_file with its enabled/log_file signature
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

    def test_trace_preparation_preserves_an_explicit_file_extension(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.core.log import prepare_trace_log

        monkeypatch.chdir(tmp_path)
        path = prepare_trace_log(command_name="exec", enabled=True, trace_file="trace.log")
        assert path == tmp_path / "trace.log"

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
            path_a = resolve_log_file(command_name="exec", enabled=True, log_file=None, unique=True)
            mock_pid.return_value = 22222
            path_b = resolve_log_file(command_name="exec", enabled=True, log_file=None, unique=True)
        assert path_a != path_b


# ---------------------------------------------------------------------------
# Unit tests: LiveTracePathResolver — one auto trace path per run
# ---------------------------------------------------------------------------


class _StepClock:
    """A ``datetime`` stand-in whose ``now()`` advances one second per call."""

    def __init__(self) -> None:
        self._calls = 0

    def now(self) -> _datetime:
        self._calls += 1
        return _datetime(2026, 1, 1, 12, 0, 0) + _timedelta(seconds=self._calls)


class TestLiveTracePathResolver:
    def test_disabled_returns_none(self, tmp_path: Path) -> None:
        resolver = LiveTracePathResolver(command_name="exec", auto_path=tmp_path / "seed.log")
        assert resolver(False, None) is None

    def test_seeded_auto_path_is_reused_for_auto_repoints(self, tmp_path: Path) -> None:
        seeded = tmp_path / "run" / "seed.log"
        resolver = LiveTracePathResolver(command_name="exec", auto_path=seeded)
        with patch("agm.core.log.datetime", _StepClock()):
            assert resolver(True, None) == seeded
            assert resolver(True, None) == seeded
        assert seeded.parent.is_dir()

    def test_auto_path_is_minted_once_and_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agm.core.log.default_agent_files_dir", lambda: tmp_path / "af")
        resolver = LiveTracePathResolver(command_name="exec", auto_path=None)
        with patch("agm.core.log.datetime", _StepClock()):
            first = resolver(True, None)
            # A repoint through the off state must not mint a second file.
            assert resolver(False, None) is None
            second = resolver(True, None)
        assert first is not None
        assert first == second

    def test_explicit_path_wins_over_the_auto_path(self, tmp_path: Path) -> None:
        seeded = tmp_path / "seed.log"
        explicit = tmp_path / "nested" / "explicit.jsonl"
        resolver = LiveTracePathResolver(command_name="exec", auto_path=seeded)
        assert resolver(True, str(explicit)) == explicit
        assert explicit.parent.is_dir()
        # The auto path survives an explicit detour.
        assert resolver(True, None) == seeded

    def test_relative_explicit_path_is_absolutised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        resolver = LiveTracePathResolver(command_name="exec", auto_path=None)
        assert resolver(True, "out.jsonl") == tmp_path / "out.jsonl"

    def test_repoint_does_not_truncate_the_destination(self, tmp_path: Path) -> None:
        existing = tmp_path / "existing.jsonl"
        existing.write_text("kept\n", encoding="utf-8")
        resolver = LiveTracePathResolver(command_name="exec", auto_path=None)
        assert resolver(True, str(existing)) == existing
        assert existing.read_text(encoding="utf-8") == "kept\n"


# ---------------------------------------------------------------------------
# CLI parsing: --trace flag (typer, cli.py)
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

    def test_trace_flag_sets_trace_true(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", "--trace", str(agl_file)])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "trace") is True

    def test_trace_and_no_trace_mutually_exclusive(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", "--trace", "--no-trace", str(agl_file)])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_trace_and_trace_file_mutually_exclusive(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", "--trace", "--trace-file", "/tmp/x.jsonl", str(agl_file)])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_default_trace_is_false(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        agl_file.write_text("let x = 1\n")
        result = invoke(runner, ["exec", str(agl_file)])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "trace") is False


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

    def test_trace_flag_sets_trace_true(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl", "--trace"])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "trace") is True

    def test_trace_and_no_trace_mutually_exclusive(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl", "--trace", "--no-trace"])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_trace_and_trace_file_mutually_exclusive(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["repl", "--trace", "--trace-file", "/tmp/x.jsonl"])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_default_trace_is_false(self, runner: CliRunner, recorded_runs: list[object]) -> None:
        result = invoke(runner, ["repl"])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "trace") is False


# ---------------------------------------------------------------------------
# Integration tests: trace file presence
# ---------------------------------------------------------------------------


def _exec_args(
    command: str,
    *,
    trace: bool = False,
    no_trace: bool = False,
    trace_file: str | None = None,
) -> ExecArgs:
    return ExecArgs(
        file=None,
        command=command,
        argument_tokens=[],
        strict_json=None,
        trace=trace,
        no_trace=no_trace,
        trace_file=trace_file,
    )


class TestIntegrationDefaultNoTrace:
    """Default run (no trace flags) writes NO trace file."""

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
    """--trace flag causes a trace file to be written."""

    def test_trace_flag_creates_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        exec_command.run(_exec_args('print "hello"', trace=True))
        agent_files = tmp_path / ".agent-files"
        assert agent_files.exists()
        trace_files = list(agent_files.glob("exec-*.jsonl"))
        assert len(trace_files) == 1

    def test_explicit_trace_file_path_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _isolated_home(monkeypatch, tmp_path)
        trace_path = tmp_path / "my_trace.jsonl"
        exec_command.run(_exec_args('print "hi"', trace_file=str(trace_path)))
        assert trace_path.exists()


class TestIntegrationConfigLogTrue:
    """[exec] trace=true in config causes a trace file to be written."""

    def test_config_trace_true_creates_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text("[exec]\ntrace = true\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        exec_command.run(_exec_args('print "hello"'))
        agent_files = tmp_path / ".agent-files"
        assert agent_files.exists()
        trace_files = list(agent_files.glob("exec-*.jsonl"))
        assert len(trace_files) == 1

    def test_config_trace_file_creates_trace_at_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        trace_path = tmp_path / "config_trace.jsonl"
        (home / ".agm" / "config.toml").write_text(f"[exec]\ntrace-file = {str(trace_path)!r}\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
        exec_command.run(_exec_args('print "hello"'))
        assert trace_path.exists()


class TestIntegrationNoLogOverridesConfig:
    """--no-trace overrides config trace=true."""

    def test_no_trace_overrides_config_trace_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / ".agm").mkdir()
        (home / ".agm" / "config.toml").write_text("[exec]\ntrace = true\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("agm.core.log.git_helpers.containing_root", lambda _: None)
        exec_command.run(_exec_args('print "hello"', no_trace=True))
        agent_files = tmp_path / ".agent-files"
        assert not agent_files.exists(), "--no-trace must override config trace=true"
