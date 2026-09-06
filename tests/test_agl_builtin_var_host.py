"""Host-service reconfiguration for host-consumed ``builtin var`` settings.

Writing the ``default-agent``, ``log``, or ``log-file`` engine settings (via
``std/config::NAME := ...``) updates the host-visible state: ``default-agent``
changes what unnamed ``ask`` calls dispatch through, and ``log``/``log-file``
repoint the trace store. These tests drive the ``agm exec``
command with the agent runner subprocess mocked, plus a direct pipeline test with
a recording policy for the reconfiguration hooks.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import pytest

from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.cli_support.args import ExecArgs
from agm.commands import exec as exec_command
from agm.commands import exec_program as exec_engine
from agm.config.context import ConfigContext
from tests._agl_helpers import write_file_program
from tests.conftest import FakeAgentTransport

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _exec_args(
    agl_file: Path, *, no_log: bool = True, log: bool = False, log_file: str | None = None
) -> ExecArgs:
    """Build ExecArgs for *agl_file* with logging off unless overridden."""
    return ExecArgs(
        file=str(agl_file),
        argument_tokens=[],
        strict_json=None,
        max_iters=None,
        no_log=no_log,
        log=log,
        log_file=log_file,
    )


class _StepClock:
    """A ``datetime`` stand-in whose ``now()`` advances one second per call.

    Every freshly minted auto trace path therefore gets a distinct timestamp,
    so a run that mints twice is visible as two files rather than depending on
    wall-clock granularity.
    """

    def __init__(self) -> None:
        self._calls = 0

    def now(self) -> datetime:
        self._calls += 1
        return datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=self._calls)


def _trace_kinds_and_prints(path: Path) -> tuple[list[str], list[str]]:
    """Return the record kinds and the rendered ``print`` outputs of a trace file."""
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    kinds = [rec["kind"] for rec in records]
    rendered = [rec["rendered"] for rec in records if rec["kind"] == "print"]
    return kinds, rendered


def _write_command_stdlib(root: Path, config: str) -> Path:
    """Create the minimal stdlib needed to exercise the real exec command."""
    stdlib_root = root / "stdlib"
    config_path = stdlib_root / "src" / "config.agl"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'import std/prelude::*\nbuiltin var default-agent: Agent = AgentCommand("echo")\n' + config,
        encoding="utf-8",
    )
    for source in (_STDLIB / "src").iterdir():
        if source.is_file() and source.name != "config.agl":
            copyfile(source, config_path.parent / source.name)
    return stdlib_root


class TestCommandEngineSeeding:
    """The command passes only explicit host controls into builtin registers."""

    def test_invalid_config_timeout_does_not_create_a_seed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stdlib_root = _write_command_stdlib(
            tmp_path,
            'builtin var timeout: Option[text] = Option[text]::Some("2s")\n',
        )
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\nprint std/config::timeout\n",
            encoding="utf-8",
        )
        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text("[exec]\ntimeout = 0\n")
        monkeypatch.setattr(
            "agm.cli_support.exec_roots.resolve_stdlib_root",
            lambda *, home, anchor=None: stdlib_root,
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=tmp_path, proj_dir=None, cwd=tmp_path),
        )

        exec_command.run(_exec_args(agl_file))

        assert capsys.readouterr().out == 'Option::Some(value = "2s")\n'

    def test_declared_invalid_timeout_returns_agl_run_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stdlib_root = _write_command_stdlib(
            tmp_path,
            'builtin var timeout: Option[text] = Option[text]::Some("bogus")\n',
        )
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "import std/config\n()\n", encoding="utf-8")
        monkeypatch.setattr(
            "agm.cli_support.exec_roots.resolve_stdlib_root",
            lambda *, home, anchor=None: stdlib_root,
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=tmp_path, proj_dir=None, cwd=tmp_path),
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))

        assert exc_info.value.code == 2


class TestDefaultAgentReconfiguration:
    def test_default_agent_write_reconfigures_ask(
        self, tmp_path: Path, fake_agent_transport: FakeAgentTransport
    ) -> None:
        """A ``default-agent :=`` selects the following ``ask`` dispatch."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config::*\n"
            'std/config::default-agent := AgentCommand("codex-runner \\%{SESSION_ID}")\n'
            'ask("hi")\n',
        )

        exec_command.run(_exec_args(agl_file))

        assert fake_agent_transport.calls[0][1][0] == "codex-runner"
        assert len(fake_agent_transport.calls[0][1]) == 2

    def test_no_default_agent_write_uses_the_stdlib_default_agent(
        self, tmp_path: Path, fake_agent_transport: FakeAgentTransport
    ) -> None:
        """The stdlib initializer supplies the default AgentClaude value."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'ask("hi")\n')

        exec_command.run(_exec_args(agl_file))

        argv = fake_agent_transport.calls[0][1]
        assert argv[:2] == ["claude", "-p"]
        assert "--session-id" in argv

    def test_default_agent_write_does_not_change_explicit_agent_values(
        self, tmp_path: Path, fake_agent_transport: FakeAgentTransport
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config::*\n"
            'let fixed = AgentCommand("fixed-runner \\%{SESSION_ID}")\n'
            'std/config::default-agent := AgentCommand("new-runner \\%{SESSION_ID}")\n'
            'fixed.ask("one")\n'
            'ask("two")\n',
        )

        exec_command.run(_exec_args(agl_file))

        assert [argv[0] for _, argv in fake_agent_transport.calls] == [
            "fixed-runner",
            "new-runner",
        ]


class TestTraceReconfiguration:
    def test_log_file_write_routes_trace_until_log_is_disabled(self, tmp_path: Path) -> None:
        """A path enables tracing, while a later ``log := false`` disables it."""
        trace_path = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config::*\n"
            'print "before"\n'
            f'std/config::log-file := Some("{trace_path}")\n'
            'print "after"\n'
            "std/config::log := false\n"
            'print "disabled"\n',
        )

        exec_command.run(_exec_args(agl_file))

        assert trace_path.exists()
        lines = [ln for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines, "no JSONL records were written to the routed trace file"
        import json

        rendered = [json.loads(ln).get("rendered") for ln in lines]
        assert "after" in rendered
        assert "before" not in rendered
        assert "disabled" not in rendered

    def test_log_false_disables_further_trace_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting ``log := false`` after logging was on stops later trace writes."""
        monkeypatch.setattr("agm.core.log.default_agent_files_dir", lambda: tmp_path)

        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config::*\n"
            "std/config::log := true\n"
            'print "first"\n'
            "std/config::log := false\n"
            'print "second"\n',
        )

        exec_command.run(_exec_args(agl_file))

        logs = list(tmp_path.glob("exec-*.jsonl"))
        assert len(logs) == 1, f"expected exactly one auto-named log, got {logs}"
        text = logs[0].read_text(encoding="utf-8")
        import json

        rendered = [
            json.loads(ln).get("rendered")
            for ln in text.splitlines()
            if ln.strip() and json.loads(ln).get("kind") == "print"
        ]
        assert "first" in rendered
        assert "second" not in rendered

    def test_cli_log_flag_seeds_log_register_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--log`` seeds the ``log`` register so a read before any write sees True."""
        monkeypatch.setattr("agm.core.log.default_agent_files_dir", lambda: tmp_path)
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "import std/config::*\nlet l = std/config::log\nprint l\n")

        exec_command.run(_exec_args(agl_file, no_log=False, log=True))

        assert capsys.readouterr().out == "true\n"

    def test_untouched_logging_still_writes(self, tmp_path: Path) -> None:
        """A program that never touches the settings logs to --log-file as before."""
        trace_path = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'print "hello"\n')

        exec_command.run(_exec_args(agl_file, no_log=False, log_file=str(trace_path)))

        assert trace_path.exists()
        import json

        rendered = [
            json.loads(ln).get("rendered")
            for ln in trace_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and json.loads(ln).get("kind") == "print"
        ]
        assert "hello" in rendered

    def test_log_write_keeps_one_auto_trace_file_for_the_whole_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--log`` plus a mid-run ``log := true`` keeps the run in ONE file."""
        monkeypatch.setattr("agm.core.log.default_agent_files_dir", lambda: tmp_path)
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config::*\nprint "before"\nstd/config::log := true\nprint "after"\n',
        )

        with patch("agm.core.log.datetime", _StepClock()):
            exec_command.run(_exec_args(agl_file, no_log=False, log=True))

        logs = list(tmp_path.glob("exec-*.jsonl"))
        assert len(logs) == 1, f"the run split its trace across {logs}"
        kinds, rendered = _trace_kinds_and_prints(logs[0])
        assert "run_start" in kinds
        assert "run_end" in kinds
        assert rendered == ["before", "after"]

    def test_toggling_log_off_and_on_reuses_the_same_auto_trace_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A program that turns logging on, off, then on again writes ONE file."""
        monkeypatch.setattr("agm.core.log.default_agent_files_dir", lambda: tmp_path)
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config::*\n"
            "std/config::log := true\n"
            'print "first"\n'
            "std/config::log := false\n"
            'print "second"\n'
            "std/config::log := true\n"
            'print "third"\n',
        )

        with patch("agm.core.log.datetime", _StepClock()):
            exec_command.run(_exec_args(agl_file))

        logs = list(tmp_path.glob("exec-*.jsonl"))
        assert len(logs) == 1, f"re-enabling logging minted a second file: {logs}"
        _, rendered = _trace_kinds_and_prints(logs[0])
        assert rendered == ["first", "third"]

    def test_explicit_log_file_still_wins_over_the_auto_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--log-file`` keeps the whole run in the named file, minting nothing."""
        auto_dir = tmp_path / "agent-files"
        auto_dir.mkdir()
        monkeypatch.setattr("agm.core.log.default_agent_files_dir", lambda: auto_dir)
        trace_path = tmp_path / "explicit.jsonl"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config::*\nprint "before"\nstd/config::log := true\nprint "after"\n',
        )

        with patch("agm.core.log.datetime", _StepClock()):
            exec_command.run(_exec_args(agl_file, no_log=False, log_file=str(trace_path)))

        assert list(auto_dir.glob("exec-*.jsonl")) == []
        kinds, rendered = _trace_kinds_and_prints(trace_path)
        assert "run_start" in kinds
        assert "run_end" in kinds
        assert rendered == ["before", "after"]


def test_trace_reconfiguration_disables_logging_after_a_path_failure(tmp_path: Path) -> None:
    """A trace path failure is best-effort and leaves the trace disabled."""
    from agm.agl.runtime.host_settings import HostSettingsReconfigurer
    from agm.agl.runtime.trace import TraceStore

    trace = TraceStore(path=tmp_path / "trace.jsonl")
    policy = HostSettingsPolicy(
        resolve_trace_path=lambda enabled, log_file: (_ for _ in ()).throw(OSError("unwritable"))
    )

    HostSettingsReconfigurer(trace=trace, policy=policy).reconfigure_trace(
        enabled=True, log_file=None
    )

    assert trace.path is None
