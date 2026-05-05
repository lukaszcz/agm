"""Tests for agm.tmux.session — create_tmux_session, queue_command_in_session,
focus_tmux_session, kill_tmux_session, close_tmux_session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agm.tmux.session as session_module
from agm.core import dry_run
from agm.tmux.session import (
    _filter_env,
    close_tmux_session,
    create_tmux_session,
    focus_tmux_session,
    kill_tmux_session,
    queue_command_in_session,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_dry_run() -> None:
    """Ensure dry-run state is restored after every test."""
    original = dry_run.enabled()
    yield  # type: ignore[misc]
    dry_run.set_enabled(original)


def _enable_dry_run() -> None:
    dry_run.set_enabled(True)


# ===========================================================================
# _filter_env — SKIP_PREFIXES branch (lines 34-36)
# ===========================================================================


class TestFilterEnvSkipPrefixes:
    """Ensure lines 34-36 (prefix filtering + append) are exercised."""

    def test_valid_env_var_with_skip_prefix_is_excluded(self) -> None:
        result = _filter_env({"TMUX_PANE": "%0", "MYVAR": "hello"})
        names = [n for n, _ in result]
        assert "MYVAR" in names  # line 36: filtered.append
        assert "TMUX_PANE" not in names  # line 34-35: startswith check + continue

    def test_xdg_prefix_excluded(self) -> None:
        result = _filter_env({"XDG_SESSION_ID": "1", "HOME_DIR": "/home"})
        names = [n for n, _ in result]
        assert "XDG_SESSION_ID" not in names
        assert "HOME_DIR" in names


# ===========================================================================
# create_tmux_session — dry_run with env vars (line 56)
# ===========================================================================


class TestCreateTmuxSessionWithEnvArgs:
    """Ensure line 56 (tmux_env_args.extend) is hit."""

    def test_detached_dry_run_with_filtered_env_includes_e_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        # Provide an env with a safe variable that passes _filter_env
        result = create_tmux_session(
            detach=True,
            pane_count=None,
            session_name="s",
            cwd=tmp_path,
            env={"MY_CUSTOM_VAR": "value123"},
        )
        out = capsys.readouterr().out
        # -e flag should appear in the command
        assert "-e" in out
        assert result == "s"


# ===========================================================================
# focus_tmux_session
# ===========================================================================


class TestFocusTmuxSession:
    def test_dry_run_no_tmux_prints_attach_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        rc = focus_tmux_session(session_name="mysession", cwd=tmp_path, env={})
        assert rc == 0
        out = capsys.readouterr().out
        assert "attach-session" in out
        assert "mysession" in out

    def test_dry_run_with_tmux_env_prints_switch_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        rc = focus_tmux_session(
            session_name="mysession", cwd=tmp_path, env={"TMUX": "/tmp/tmux.sock"}
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "switch-client" in out
        assert "mysession" in out

    def test_live_calls_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        class FakeResult:
            returncode = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            calls.append(cmd)
            return FakeResult()

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        rc = focus_tmux_session(session_name="s1", cwd=tmp_path, env={})
        assert rc == 0
        assert calls[0][0] == "tmux"
        assert "attach-session" in calls[0]

    def test_live_switch_client_when_tmux_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        class FakeResult:
            returncode = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            calls.append(cmd)
            return FakeResult()

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        rc = focus_tmux_session(
            session_name="s1", cwd=tmp_path, env={"TMUX": "/run/tmux.sock"}
        )
        assert rc == 0
        assert "switch-client" in calls[0]


# ===========================================================================
# kill_tmux_session
# ===========================================================================


class TestKillTmuxSession:
    def test_dry_run_prints_kill_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        rc = kill_tmux_session(session_name="to-kill", cwd=tmp_path, env={})
        assert rc == 0
        out = capsys.readouterr().out
        assert "kill-session" in out
        assert "to-kill" in out

    def test_live_calls_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        class FakeResult:
            returncode = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            calls.append(cmd)
            return FakeResult()

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        rc = kill_tmux_session(session_name="to-kill", cwd=tmp_path, env={})
        assert rc == 0
        assert "kill-session" in calls[0]
        assert "to-kill" in calls[0]

    def test_live_returns_nonzero_returncode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeResult:
            returncode = 2

        monkeypatch.setattr(session_module.subprocess, "run", lambda *a, **kw: FakeResult())
        rc = kill_tmux_session(session_name="bad", cwd=tmp_path, env={})
        assert rc == 2

    def test_uses_os_environ_when_env_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        class FakeResult:
            returncode = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            calls.append(kwargs)
            return FakeResult()

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        monkeypatch.setattr(session_module.os, "environ", {"MY": "env"})
        kill_tmux_session(session_name="s", cwd=tmp_path)
        assert calls[0]["env"] == {"MY": "env"}

    def test_uses_cwd_when_not_provided(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[dict[str, Any]] = []

        class FakeResult:
            returncode = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            calls.append(kwargs)
            return FakeResult()

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        monkeypatch.setattr(session_module.Path, "cwd", staticmethod(lambda: tmp_path))
        kill_tmux_session(session_name="s", env={})
        assert calls[0]["cwd"] == tmp_path


# ===========================================================================
# close_tmux_session
# ===========================================================================


class TestCloseTmuxSession:
    def test_prints_closed_message_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(session_module, "kill_tmux_session", lambda **kw: 0)
        close_tmux_session(session_name="myses", cwd=tmp_path, env={})
        out = capsys.readouterr().out
        assert "myses" in out

    def test_raises_system_exit_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(session_module, "kill_tmux_session", lambda **kw: 3)
        with pytest.raises(SystemExit) as exc_info:
            close_tmux_session(session_name="myses", cwd=tmp_path, env={})
        assert exc_info.value.code == 3

    def test_dry_run_prints_kill_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        close_tmux_session(session_name="dryses", cwd=tmp_path, env={})
        out = capsys.readouterr().out
        assert "kill-session" in out


# ===========================================================================
# queue_command_in_session
# ===========================================================================


class TestQueueCommandInSession:
    def test_dry_run_prints_send_keys(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        queue_command_in_session(
            session_name="myses",
            command=["echo", "hello"],
            cwd=tmp_path,
            env={},
        )
        out = capsys.readouterr().out
        assert "send-keys" in out
        assert "myses" in out

    def test_dry_run_returns_none(self, tmp_path: Path) -> None:
        _enable_dry_run()
        result = queue_command_in_session(
            session_name="myses",
            command=["echo", "hi"],
            cwd=tmp_path,
            env={},
        )
        assert result is None

    def test_live_calls_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        class FakeResult:
            returncode = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            calls.append(cmd)
            return FakeResult()

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        queue_command_in_session(
            session_name="s1",
            command=["agm", "setup"],
            cwd=tmp_path,
            env={},
        )
        assert "send-keys" in calls[0]
        assert "s1:0.0" in calls[0]

    def test_live_raises_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeResult:
            returncode = 5

        monkeypatch.setattr(session_module.subprocess, "run", lambda *a, **kw: FakeResult())
        with pytest.raises(SystemExit) as exc_info:
            queue_command_in_session(
                session_name="s1",
                command=["agm", "setup"],
                cwd=tmp_path,
                env={},
            )
        assert exc_info.value.code == 5

    def test_command_is_shell_quoted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        class FakeResult:
            returncode = 0

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            calls.append(cmd)
            return FakeResult()

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        queue_command_in_session(
            session_name="s1",
            command=["echo", "hello world"],
            cwd=tmp_path,
            env={},
        )
        # The send-keys arg should contain the quoted command
        combined = " ".join(calls[0])
        assert "hello" in combined


# ===========================================================================
# create_tmux_session — dry_run paths
# ===========================================================================


class TestCreateTmuxSessionDryRun:
    def test_detached_dry_run_returns_session_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        result = create_tmux_session(
            detach=True,
            pane_count=None,
            session_name="mysession",
            cwd=tmp_path,
            env={},
        )
        assert result == "mysession"
        out = capsys.readouterr().out
        assert "new-session" in out

    def test_detached_dry_run_no_session_name_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        result = create_tmux_session(
            detach=True,
            pane_count=None,
            session_name=None,
            cwd=tmp_path,
            env={},
        )
        assert result == "0"

    def test_detached_dry_run_prints_split_window_for_extra_panes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        create_tmux_session(
            detach=True,
            pane_count="3",
            session_name="s",
            cwd=tmp_path,
            env={},
        )
        out = capsys.readouterr().out
        assert out.count("split-window") == 2

    def test_detached_dry_run_prints_select_pane(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        create_tmux_session(
            detach=True,
            pane_count=None,
            session_name="s",
            cwd=tmp_path,
            env={},
        )
        out = capsys.readouterr().out
        assert "select-pane" in out

    def test_detached_dry_run_no_switch_client_when_detach_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        create_tmux_session(
            detach=True,
            pane_count=None,
            session_name="s",
            cwd=tmp_path,
            env={"TMUX": "/run/tmux.sock"},
        )
        out = capsys.readouterr().out
        # detach=True means create_detached_session=True but switch_to_session=False
        assert "switch-client" not in out

    def test_detached_with_tmux_env_prints_switch_when_not_detach(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When TMUX env is set and detach=False, switch_to_session=True."""
        _enable_dry_run()
        create_tmux_session(
            detach=False,
            pane_count=None,
            session_name="s",
            cwd=tmp_path,
            env={"TMUX": "/run/tmux.sock"},
        )
        out = capsys.readouterr().out
        assert "switch-client" in out

    def test_attached_dry_run_prints_new_session_returns_none(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _enable_dry_run()
        result = create_tmux_session(
            detach=False,
            pane_count=None,
            session_name=None,
            cwd=tmp_path,
            env={},
        )
        assert result is None
        out = capsys.readouterr().out
        assert "new-session" in out

    def test_detached_no_attach_when_detach_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With detach=True and no TMUX, attach-session is NOT printed."""
        _enable_dry_run()
        create_tmux_session(
            detach=True,
            pane_count=None,
            session_name="s",
            cwd=tmp_path,
            env={},
        )
        out = capsys.readouterr().out
        assert "attach-session" not in out

    def test_non_detached_no_tmux_goes_to_non_detached_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With detach=False and no TMUX env, non-detached path is used."""
        _enable_dry_run()
        create_tmux_session(
            detach=False,
            pane_count="1",
            session_name="s",
            cwd=tmp_path,
            env={},
        )
        out = capsys.readouterr().out
        assert "new-session" in out


# ===========================================================================
# create_tmux_session — live subprocess paths
# ===========================================================================


class TestCreateTmuxSessionLive:
    def _make_subprocess_mock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        returncode: int = 0,
        stdout: str = "mysession\n",
    ) -> list[list[str]]:
        calls: list[list[str]] = []

        class FakeResult:
            def __init__(self, rc: int = 0, out: str = "") -> None:
                self.returncode = rc
                self.stdout = out
                self.stderr = ""

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            calls.append(list(cmd))
            if cmd[1] == "new-session":
                return FakeResult(returncode, stdout)
            if cmd[1] == "display-message":
                fmt = cmd[-1]
                if "window_id" in fmt:
                    return FakeResult(0, "@0\n")
                if "window_width" in fmt:
                    return FakeResult(0, "200\n")
                if "window_height" in fmt:
                    return FakeResult(0, "50\n")
            return FakeResult(0, "")

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        return calls

    def test_detached_live_returns_session_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls = self._make_subprocess_mock(monkeypatch, stdout="mysession\n")
        monkeypatch.setattr(session_module, "apply_layout", lambda **kw: None)
        result = create_tmux_session(
            detach=True,
            pane_count=None,
            session_name="mysession",
            cwd=tmp_path,
            env={},
        )
        assert result == "mysession"
        assert any("new-session" in " ".join(c) for c in calls)

    def test_detached_live_raises_on_new_session_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._make_subprocess_mock(monkeypatch, returncode=1, stdout="error\n")
        monkeypatch.setattr(session_module, "apply_layout", lambda **kw: None)
        with pytest.raises(SystemExit):
            create_tmux_session(
                detach=True,
                pane_count=None,
                session_name="s",
                cwd=tmp_path,
                env={},
            )

    def test_detached_live_raises_on_split_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeResult:
            def __init__(self, rc: int = 0) -> None:
                self.returncode = rc
                self.stdout = "mysession\n"
                self.stderr = ""

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            if cmd[1] == "new-session":
                return FakeResult(0)
            if cmd[1] == "split-window":
                return FakeResult(1)
            return FakeResult(0)

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        monkeypatch.setattr(session_module, "apply_layout", lambda **kw: None)
        with pytest.raises(SystemExit):
            create_tmux_session(
                detach=True,
                pane_count="2",
                session_name="s",
                cwd=tmp_path,
                env={},
            )

    def test_detached_live_raises_on_display_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeResult:
            def __init__(self, rc: int = 0, out: str = "") -> None:
                self.returncode = rc
                self.stdout = out
                self.stderr = "err"

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            if cmd[1] == "new-session":
                return FakeResult(0, "mysession\n")
            if cmd[1] == "display-message":
                return FakeResult(1, "")
            return FakeResult(0, "")

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        monkeypatch.setattr(session_module, "apply_layout", lambda **kw: None)
        with pytest.raises(SystemExit):
            create_tmux_session(
                detach=True,
                pane_count="1",
                session_name="s",
                cwd=tmp_path,
                env={},
            )

    def test_detached_live_raises_on_select_pane_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeResult:
            def __init__(self, rc: int = 0, out: str = "") -> None:
                self.returncode = rc
                self.stdout = out
                self.stderr = ""

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            if cmd[1] == "new-session":
                return FakeResult(0, "mysession\n")
            if cmd[1] == "display-message":
                fmt = cmd[-1]
                if "window_id" in fmt:
                    return FakeResult(0, "@0\n")
                if "window_width" in fmt:
                    return FakeResult(0, "200\n")
                if "window_height" in fmt:
                    return FakeResult(0, "50\n")
            if cmd[1] == "select-pane":
                return FakeResult(1, "")
            return FakeResult(0, "")

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        monkeypatch.setattr(session_module, "apply_layout", lambda **kw: None)
        with pytest.raises(SystemExit):
            create_tmux_session(
                detach=True,
                pane_count="1",
                session_name="s",
                cwd=tmp_path,
                env={},
            )

    def test_detached_live_switch_client_when_tmux_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeResult:
            def __init__(self, rc: int = 0, out: str = "") -> None:
                self.returncode = rc
                self.stdout = out
                self.stderr = ""

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            if cmd[1] == "new-session":
                return FakeResult(0, "s1\n")
            if cmd[1] == "display-message":
                fmt = cmd[-1]
                if "window_id" in fmt:
                    return FakeResult(0, "@0\n")
                if "window_width" in fmt:
                    return FakeResult(0, "200\n")
                if "window_height" in fmt:
                    return FakeResult(0, "50\n")
            if cmd[1] in {"select-pane", "switch-client"}:
                return FakeResult(0, "")
            return FakeResult(0, "")

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        monkeypatch.setattr(session_module, "apply_layout", lambda **kw: None)
        # TMUX set + detach=False → switch_to_session=True → raises SystemExit(0)
        with pytest.raises(SystemExit) as exc_info:
            create_tmux_session(
                detach=False,
                pane_count="1",
                session_name="s1",
                cwd=tmp_path,
                env={"TMUX": "/run/tmux.sock"},
            )
        assert exc_info.value.code == 0

    def test_non_detached_live_raises_system_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeResult:
            returncode = 0

        monkeypatch.setattr(session_module.subprocess, "run", lambda *a, **kw: FakeResult())
        with pytest.raises(SystemExit):
            create_tmux_session(
                detach=False,
                pane_count=None,
                session_name=None,
                cwd=tmp_path,
                env={},
            )

    def test_detached_live_writes_stderr_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Cover sys.stderr.write when new-session fails with stderr output."""

        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = "some tmux error"

        monkeypatch.setattr(session_module.subprocess, "run", lambda *a, **kw: FakeResult())
        with pytest.raises(SystemExit):
            create_tmux_session(
                detach=True,
                pane_count=None,
                session_name="s",
                cwd=tmp_path,
                env={},
            )
        err = capsys.readouterr().err
        assert "some tmux error" in err

    def test_detached_live_writes_stdout_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Cover the sys.stdout.write branch when new-session fails with stdout."""

        class FakeResult:
            returncode = 1
            stdout = "unexpected output"
            stderr = ""

        monkeypatch.setattr(session_module.subprocess, "run", lambda *a, **kw: FakeResult())
        with pytest.raises(SystemExit):
            create_tmux_session(
                detach=True,
                pane_count=None,
                session_name="s",
                cwd=tmp_path,
                env={},
            )
        out = capsys.readouterr().out
        assert "unexpected output" in out

    def test_display_failure_with_stdout_writes_to_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Cover sys.stdout.write when display-message fails with stdout."""

        class FakeResult:
            def __init__(self, rc: int = 0, out: str = "") -> None:
                self.returncode = rc
                self.stdout = out
                self.stderr = ""

        def fake_run(cmd: list[str], **kwargs: Any) -> FakeResult:
            if cmd[1] == "new-session":
                return FakeResult(0, "mysession\n")
            if cmd[1] == "display-message":
                return FakeResult(1, "display stdout output")
            return FakeResult(0, "")

        monkeypatch.setattr(session_module.subprocess, "run", fake_run)
        monkeypatch.setattr(session_module, "apply_layout", lambda **kw: None)
        with pytest.raises(SystemExit):
            create_tmux_session(
                detach=True,
                pane_count="1",
                session_name="s",
                cwd=tmp_path,
                env={},
            )
        out = capsys.readouterr().out
        assert "display stdout output" in out
