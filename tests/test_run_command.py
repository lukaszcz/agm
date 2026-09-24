"""Tests for the agm run command: sandbox delegation, resource limits, and dry run.

`agm run` is a thin client of `sandbox/prepare.py`; these tests assert its
user-visible behaviour (the argv/env/cleanup reaching `run_foreground`, alias
splitting, pty wrapping, limits, dry-run output, exit codes, error messages)
through that seam rather than through any sandbox-internal function.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest

from agm.cli_support.args import RunArgs
from agm.commands import run as run_command
from agm.config.general import CommandSetting, RunConfig


def _make_run_config(
    *,
    aliases: dict[str, str] | None = None,
    memory_limit: str | None = None,
    swap_limit: str | None = None,
    command_memory_limits: dict[str, str] | None = None,
    command_swap_limits: dict[str, str] | None = None,
    pty: bool = True,
    command_ptys: dict[str, bool] | None = None,
) -> RunConfig:
    """Build a :class:`RunConfig` with everything the caller did not name empty."""

    return RunConfig(
        alias=CommandSetting(default=None, overrides=aliases or {}),
        memory=CommandSetting(default=memory_limit, overrides=command_memory_limits or {}),
        swap=CommandSetting(default=swap_limit, overrides=command_swap_limits or {}),
        pty=CommandSetting(default=pty, overrides=command_ptys or {}),
    )


def _write_default_sandbox_settings(home: Path) -> None:
    """Write a home-scope `default.json` so sandboxed `prepare()` calls resolve."""

    sandbox_dir = home / ".agm" / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    (sandbox_dir / "default.json").write_text("{}", encoding="utf-8")


def _capturing_run_foreground(captured: dict[str, object]) -> Callable[..., int]:
    """A fake `run_foreground` that records its call and any `--settings` file's content."""

    def fake_run_foreground(
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
        isolate_process_group: bool = False,
    ) -> int:
        captured.update(
            cmd=cmd,
            cwd=cwd,
            env=env,
            interrupt_cleanup_cmd=interrupt_cleanup_cmd,
            isolate_process_group=isolate_process_group,
        )
        if "--settings" in cmd:
            settings_path = Path(cmd[cmd.index("--settings") + 1])
            captured["settings"] = json.loads(settings_path.read_text())
        return 0

    return fake_run_foreground


def test_run_delegates_sandbox_execution_to_srt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()
    _write_default_sandbox_settings(tmp_path / "home")

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

    with pytest.raises(SystemExit) as exc_info:
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=False,
                no_swap_limit=False,
                settings_file=None,
            )
        )
    assert exc_info.value.code == 0

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:4] == ["systemd-run", "--user", "--scope", "-q"]
    assert "MemoryMax=32G" in cmd
    assert "MemorySwapMax=0" in cmd
    assert "Delegate=yes" in cmd
    assert "--unit" in cmd
    assert "bash" in cmd
    assert "srt" in cmd
    assert "--settings" in cmd
    assert cmd[-3:] == ["--", "echo", "hi"]

    assert captured["cwd"] == tmp_path / "work"

    unit = cmd[cmd.index("--unit") + 1]
    assert captured["interrupt_cleanup_cmd"] == ["systemctl", "--user", "--no-block", "stop", unit]
    assert captured["isolate_process_group"] is True

    assert captured["env"] == {**env, "NODE_USE_ENV_PROXY": "1"}

    bootstrap_script = cmd[cmd.index("-c") + 1]
    assert isinstance(bootstrap_script, str)
    assert 'mkdir -p "${CG}/init"' in bootstrap_script
    assert 'echo $$ > "${CG}/init/cgroup.procs"' in bootstrap_script
    assert 'echo "+memory" > "${CG}/cgroup.subtree_control"' in bootstrap_script
    assert 'export SANDBOX_CGROUP="$CG"' in bootstrap_script


@pytest.mark.parametrize(
    ("pty_override", "interactive", "wrapped"),
    [(None, True, True), (False, True, False), (True, False, False)],
)
def test_run_allocates_pty_only_for_enabled_interactive_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pty_override: bool | None,
    interactive: bool,
    wrapped: bool,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _write_default_sandbox_settings(home)
    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", {"HOME": str(home), "PATH": "/bin"})
    monkeypatch.setattr(run_command.os, "isatty", lambda _fd: interactive)
    monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

    with pytest.raises(SystemExit):
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
                pty=pty_override,
            )
        )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[-2:] == ["echo", "hi"]
    if wrapped:
        pty_index = cmd.index(sys.executable)
        assert cmd[pty_index : pty_index + 4] == [sys.executable, "-m", "agm.sandbox.pty", "--"]
    else:
        assert sys.executable not in cmd


def test_run_patches_sandbox_when_project_is_discovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    project = tmp_path / "project"
    cwd = project / "repo"
    (tmp_path / "home").mkdir()
    cwd.mkdir(parents=True)
    _write_default_sandbox_settings(tmp_path / "home")

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: cwd))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        "agm.config.context.discover_current_project_dir",
        lambda current, env=None: project,
    )
    monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

    with pytest.raises(SystemExit):
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

    settings = captured["settings"]
    assert isinstance(settings, dict)
    allow_write = settings["filesystem"]["allowWrite"]
    assert str(project / "notes") in allow_write
    assert str(project / "deps") in allow_write
    assert str(project / "repo" / ".git") in allow_write


def test_run_no_memory_limit_omits_memory_max_but_keeps_default_swap_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()
    _write_default_sandbox_settings(tmp_path / "home")

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(memory_limit="10G", command_memory_limits={"echo": "5G"}),
    )
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

    with pytest.raises(SystemExit):
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=False,
                settings_file=None,
            )
        )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "MemoryMax=5G" not in cmd
    assert "MemorySwapMax=0" in cmd
    assert captured["interrupt_cleanup_cmd"] is not None


def test_run_no_memory_and_swap_limit_disable_wrapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()
    _write_default_sandbox_settings(tmp_path / "home")

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(memory_limit="10G", command_memory_limits={"echo": "5G"}),
    )
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

    with pytest.raises(SystemExit):
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "systemd-run" not in cmd
    assert cmd[0] == "srt"
    assert captured["interrupt_cleanup_cmd"] is None


def test_run_no_swap_limit_omits_memory_swap_max_property(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()
    _write_default_sandbox_settings(tmp_path / "home")

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

    with pytest.raises(SystemExit):
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=False,
                no_swap_limit=True,
                settings_file=None,
            )
        )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "MemoryMax=32G" in cmd
    assert "MemorySwapMax=0" not in cmd
    assert captured["interrupt_cleanup_cmd"] is not None


def test_run_zero_memory_limit_still_wraps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()
    _write_default_sandbox_settings(tmp_path / "home")

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command, "load_run_config", lambda **_: _make_run_config(memory_limit="0")
    )
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

    with pytest.raises(SystemExit):
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=False,
                no_swap_limit=False,
                settings_file=None,
            )
        )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "MemoryMax=0" in cmd
    assert "MemorySwapMax=0" in cmd
    assert captured["interrupt_cleanup_cmd"] is not None


def test_run_alias_with_flags_splits_into_separate_args_no_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(aliases={"claude": "claude --dangerously-skip-permissions"}),
    )

    captured: dict[str, object] = {}

    def fake_run_foreground(cmd: list[str], **kw: Any) -> int:
        captured.update(cmd=cmd)
        return 42

    monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

    with pytest.raises(SystemExit) as exc_info:
        run_command.run(
            RunArgs(
                run_command=["claude", "--some-arg"],
                no_sandbox=True,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

    assert captured["cmd"] == ["claude", "--dangerously-skip-permissions", "--some-arg"]
    assert exc_info.value.code == 42


def test_run_alias_with_flags_splits_into_separate_args_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()
    _write_default_sandbox_settings(tmp_path / "home")

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(aliases={"claude": "claude --dangerously-skip-permissions"}),
    )
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

    with pytest.raises(SystemExit):
        run_command.run(
            RunArgs(
                run_command=["claude", "--some-arg"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[-4:] == ["--", "claude", "--dangerously-skip-permissions", "--some-arg"]


def test_run_simple_alias_no_flags_no_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(aliases={"myagent": "claude"}),
    )

    captured: dict[str, object] = {}

    def fake_run_foreground(cmd: list[str], **kw: Any) -> int:
        captured.update(cmd=cmd)
        return 42

    monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

    with pytest.raises(SystemExit) as exc_info:
        run_command.run(
            RunArgs(
                run_command=["myagent", "--some-arg"],
                no_sandbox=True,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

    assert captured["cmd"] == ["claude", "--some-arg"]
    assert exc_info.value.code == 42


# ===========================================================================
# run — empty command error
# ===========================================================================


class TestRunEmptyCommand:
    def test_empty_command_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["--"],  # normalizes to []
                    no_sandbox=False,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "command is required" in err


# ===========================================================================
# run — sandbox / settings errors
# ===========================================================================


class TestRunSandboxErrors:
    def test_missing_srt_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        _write_default_sandbox_settings(tmp_path / "home")
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: None)

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=False,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert err.splitlines() == [
            "Error: srt is not installed or not in PATH.",
            "Install it with: npm install -g @anthropic-ai/sandbox-runtime",
        ]

    def test_no_settings_file_found_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=False,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "no sandbox settings file found" in err.lower()
        assert "checked:" in err.lower()

    def test_explicit_settings_file_not_found_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=False,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=str(tmp_path / "missing.json"),
                )
            )
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "settings file not found" in err.lower()
        assert "missing.json" in err

    def test_invalid_memory_limit_exits_with_generic_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A malformed --memory value raises SandboxSettingsError with no path or candidates,
        so `agm run` falls back to the exception's own message."""
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=True,
                    no_patch=False,
                    memory="not-a-valid-limit",
                    swap=None,
                    no_memory_limit=False,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "invalid resource limit" in err.lower()

    def test_malformed_settings_file_exits_with_error_not_a_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A single malformed settings candidate must surface as `Error: ...` and exit 1,
        never as a raw `json.JSONDecodeError` traceback."""

        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        sandbox_dir = tmp_path / "home" / ".agm" / "sandbox"
        sandbox_dir.mkdir(parents=True)
        (sandbox_dir / "default.json").write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=False,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 1
        assert capsys.readouterr().err.startswith("Error:")

    def test_empty_memory_flag_is_treated_as_not_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--memory ""` behaves as if `--memory` were not passed at all, falling through
        to config/built-in defaults, matching the historical `args.memory or ...` chain."""

        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        _write_default_sandbox_settings(tmp_path / "home")
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

        captured: dict[str, object] = {}
        monkeypatch.setattr(run_command, "run_foreground", _capturing_run_foreground(captured))

        with pytest.raises(SystemExit):
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=False,
                    no_patch=False,
                    memory="",
                    swap=None,
                    no_memory_limit=False,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )

        cmd = captured["cmd"]
        assert isinstance(cmd, list)
        assert "MemoryMax=32G" in cmd


# ===========================================================================
# run — dry_run paths
# ===========================================================================


class TestRunDryRun:
    @pytest.fixture(autouse=True)
    def reset_dry_run(self) -> Generator[None, None, None]:
        from agm.core import dry_run

        original = dry_run.enabled()
        yield
        dry_run.set_enabled(original)

    def _enable_dry_run(self) -> None:
        from agm.core import dry_run

        dry_run.set_enabled(True)

    def _setup_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        return env

    def test_dry_run_no_sandbox_prints_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=True,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )
        out = capsys.readouterr().out
        assert "echo" in out

    def test_dry_run_with_sandbox_calls_srt_and_returns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/bin/srt")

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        assert "dry-run: sandbox configuration" in out
        assert "settings source: merged" in out
        assert "srt --settings '<dry-run-settings>' -- echo hi" in out

    def test_dry_run_pty_uses_alias_target_for_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._enable_dry_run()
        env = self._setup_env(tmp_path, monkeypatch)
        agm_home = tmp_path / "agm-home"
        (agm_home / "sandbox").mkdir(parents=True)
        (agm_home / "sandbox" / "claude.json").write_text("{}", encoding="utf-8")
        env["AGM_HOME"] = str(agm_home)
        monkeypatch.setattr(run_command.os, "isatty", lambda _fd: True)
        monkeypatch.setattr(
            run_command,
            "load_run_config",
            lambda **_: _make_run_config(aliases={"myagent": "claude"}),
        )
        monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

        run_command.run(
            RunArgs(
                run_command=["myagent", "--some-arg"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        assert "claude.json" in out
        assert "myagent.json" not in out
        assert (
            f"srt --settings '<dry-run-settings>' -- {sys.executable} -m "
            "agm.sandbox.pty -- claude --some-arg"
        ) in out

    def test_dry_run_with_invalid_memory_limit_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An invalid --memory value fails before any configuration is printed: the
        detail line itself needs the resolved (and thus validated) limit."""

        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=True,
                    no_patch=False,
                    memory="not-a-valid-limit",
                    swap=None,
                    no_memory_limit=False,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid resource limit" in captured.err.lower()
        assert captured.out == ""

    def test_dry_run_with_missing_srt_prints_configuration_before_the_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Even under `--dry-run`, a missing `srt` still fails: the run/sandbox
        configuration blocks must print before that failure, as they did before the
        sandbox library existed."""

        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: None)

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=False,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "dry-run: run configuration" in captured.out
        assert "dry-run: sandbox configuration" in captured.out
        assert "srt is not installed" in captured.err

    def test_dry_run_with_proj_dir_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin", "PROJ_DIR": str(tmp_path)}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/bin/srt")

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        assert "patch proj dir path: ." in out
        assert "srt --settings '<dry-run-settings>' -- echo hi" in out

    def test_dry_run_with_sandbox_and_no_patch_shows_patch_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin", "PROJ_DIR": str(tmp_path)}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/bin/srt")

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=True,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        assert "patch proj dir path: disabled" in out
        assert "patch proj dir: disabled" in out

    def test_dry_run_sandbox_with_memory_limit_shows_combined_command(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Both a resource limit and the sandbox enabled: the printed labeled command
        contains `systemd-run`, `srt`, and the command all on the same line."""

        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/bin/tool")

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory="4G",
                swap=None,
                no_memory_limit=False,
                no_swap_limit=True,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        command_lines = [line for line in out.splitlines() if "dry-run: command [sandbox]:" in line]
        assert len(command_lines) == 1
        command_line = command_lines[0]
        assert "systemd-run" in command_line
        assert "MemoryMax=4G" in command_line
        assert "srt" in command_line
        assert "echo hi" in command_line

    def test_dry_run_no_sandbox_with_memory_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/systemd-run")
        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=True,
                no_patch=False,
                memory="10G",
                swap=None,
                no_memory_limit=False,
                no_swap_limit=True,
                settings_file=None,
            )
        )
        out = capsys.readouterr().out
        assert "echo" in out
        assert "systemd-run" in out
        assert "MemoryMax=10G" in out
        assert "MemorySwapMax" not in out
        assert "Delegate=yes" in out

    def test_dry_run_no_sandbox_with_swap_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/systemd-run")

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=True,
                no_patch=False,
                memory=None,
                swap="2G",
                no_memory_limit=True,
                no_swap_limit=False,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        assert "systemd-run" in out
        assert "MemorySwapMax=2G" in out
        assert "MemoryMax" not in out

    def test_dry_run_no_sandbox_normalizes_unlimited_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/systemd-run")

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=True,
                no_patch=False,
                memory=" unlimited ",
                swap="UNLIMITED",
                no_memory_limit=False,
                no_swap_limit=False,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        assert "MemoryMax=infinity" in out
        assert "MemorySwapMax=infinity" in out

    def test_dry_run_memory_limit_detail_shows_raw_value_while_argv_normalizes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The detail line prints the configured value; the argv carries the normalized one."""
        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/systemd-run")

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=True,
                no_patch=False,
                memory=" unlimited ",
                swap=None,
                no_memory_limit=False,
                no_swap_limit=True,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        assert "dry-run:   memory limit:  unlimited " in out
        assert "MemoryMax=infinity" in out
        assert "MemoryMax= unlimited " not in out

    def test_dry_run_no_sandbox_returns_after_printing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Cover dry_run no_sandbox path: prints command and returns without executing."""
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        # No raises: the function should simply return (not raise SystemExit)
        run_command.run(
            RunArgs(
                run_command=["mycommand"],
                no_sandbox=True,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=None,
            )
        )
        out = capsys.readouterr().out
        assert "mycommand" in out

    def test_dry_run_no_sandbox_without_limit_flags_disables_both_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--no-sandbox with neither an explicit limit nor a --no-*-limit flag applies none."""
        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=True,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=False,
                no_swap_limit=False,
                settings_file=None,
            )
        )

        out = capsys.readouterr().out
        assert "memory limit: disabled" in out
        assert "swap limit: disabled" in out
        assert "systemd-run" not in out

    def test_dry_run_with_sandbox_and_explicit_settings_file_shows_its_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/bin/srt")
        settings_file = tmp_path / "custom.json"
        settings_file.write_text("{}", encoding="utf-8")

        run_command.run(
            RunArgs(
                run_command=["echo", "hi"],
                no_sandbox=False,
                no_patch=False,
                memory=None,
                swap=None,
                no_memory_limit=True,
                no_swap_limit=True,
                settings_file=str(settings_file),
            )
        )

        out = capsys.readouterr().out
        assert "settings source: explicit" in out
        assert "custom.json" in out


# ===========================================================================
# run — no_sandbox swap limit
# ===========================================================================


class TestRunNoSandboxSwapLimit:
    def test_no_sandbox_with_limits_exits_when_systemd_run_is_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: None)

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=True,
                    no_patch=False,
                    memory="10G",
                    swap=None,
                    no_memory_limit=False,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )

        assert exc_info.value.code == 1
        assert "systemd-run is not installed" in capsys.readouterr().err

    def test_no_sandbox_uses_swap_from_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover no_sandbox path: effective_swap_limit is taken from run_args.swap."""
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *a, **kw: "/usr/bin/systemd-run")

        foreground_calls: list[list[str]] = []
        cleanup_calls: list[list[str] | None] = []

        def fake_run_foreground(cmd: list[str], **kw: Any) -> int:
            foreground_calls.append(cmd)
            cleanup_calls.append(kw["interrupt_cleanup_cmd"])
            return 0

        monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit):
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=True,
                    no_patch=False,
                    memory=None,
                    swap="2G",  # no_swap_limit=False → no_sandbox branch → swap=run_args.swap
                    no_memory_limit=True,
                    no_swap_limit=False,
                    settings_file=None,
                )
            )

        # The command should contain swap-related systemd args
        assert len(foreground_calls) == 1
        assert "MemorySwapMax=2G" in foreground_calls[0]
        assert cleanup_calls[0] is not None
        assert cleanup_calls[0][0:4] == ["systemctl", "--user", "--no-block", "stop"]


# ===========================================================================
# run — no_sandbox live path
# ===========================================================================


class TestRunNoSandboxLive:
    def test_no_sandbox_with_no_limits_runs_foreground(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        foreground_calls: list[list[str]] = []

        def fake_run_foreground(
            cmd: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
            interrupt_cleanup_cmd: list[str] | None = None,
            isolate_process_group: bool = False,
        ) -> int:
            foreground_calls.append(cmd)
            return 0

        monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=True,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 0
        assert foreground_calls == [["echo", "hi"]]

    def test_no_sandbox_keyboard_interrupt_exits_130(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())

        def fake_run_foreground(
            cmd: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
            interrupt_cleanup_cmd: list[str] | None = None,
            isolate_process_group: bool = False,
        ) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=True,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )
        assert exc_info.value.code == 130

    def test_no_sandbox_no_patch_with_proj_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin", "PROJ_DIR": str(tmp_path)}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())

        def fake_run_foreground(cmd: list[str], **kw: Any) -> int:
            return 0

        monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit):
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=True,
                    no_patch=True,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )


class TestRunSandboxKeyboardInterrupt:
    def test_cleans_up_a_real_temp_settings_file_on_interrupt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup runs on KeyboardInterrupt even when a real merge temp file exists
        (the `--no-sandbox` interrupt test above has nothing to clean)."""

        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        home_sandbox = tmp_path / "home" / ".agm" / "sandbox"
        cwd_sandbox = tmp_path / "work" / ".sandbox"
        home_sandbox.mkdir(parents=True)
        cwd_sandbox.mkdir(parents=True)
        (home_sandbox / "default.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
        (cwd_sandbox / "default.json").write_text(json.dumps({"enabled": False}), encoding="utf-8")

        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: "/bin/tool")

        captured_settings_path: dict[str, Path] = {}

        def fake_run_foreground(cmd: list[str], **kw: Any) -> int:
            settings_path = Path(cmd[cmd.index("--settings") + 1])
            assert settings_path.exists()
            captured_settings_path["path"] = settings_path
            raise KeyboardInterrupt

        monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit) as exc_info:
            run_command.run(
                RunArgs(
                    run_command=["echo", "hi"],
                    no_sandbox=False,
                    no_patch=False,
                    memory=None,
                    swap=None,
                    no_memory_limit=True,
                    no_swap_limit=True,
                    settings_file=None,
                )
            )

        assert exc_info.value.code == 130
        assert not captured_settings_path["path"].exists()
