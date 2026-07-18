"""Additional tests for agm.commands.run — edge cases and dry_run paths."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

import agm.commands.run as run_module
from agm.cli_support.args import RunArgs
from agm.commands.run import (
    normalize_run_command,
)
from agm.config.general import RunConfig


def _make_run_config(
    *,
    aliases: dict[str, str] | None = None,
    memory_limit: str | None = None,
    swap_limit: str | None = None,
) -> RunConfig:
    return RunConfig(
        aliases=aliases or {},
        default_memory_limit=memory_limit,
        command_memory_limits={},
        default_swap_limit=swap_limit,
        command_swap_limits={},
    )


# ===========================================================================
# normalize_run_command
# ===========================================================================


class TestNormalizeRunCommand:
    def test_strips_leading_double_dash(self) -> None:
        assert normalize_run_command(["--", "echo", "hi"]) == ["echo", "hi"]

    def test_strips_only_first_double_dash(self) -> None:
        assert normalize_run_command(["--", "--", "echo"]) == ["--", "echo"]

    def test_passthrough_without_double_dash(self) -> None:
        assert normalize_run_command(["echo", "hi"]) == ["echo", "hi"]

    def test_empty_list_returned_unchanged(self) -> None:
        assert normalize_run_command([]) == []

    def test_single_double_dash_stripped(self) -> None:
        assert normalize_run_command(["--"]) == []


# ===========================================================================
# run — empty command error
# ===========================================================================


class TestRunEmptyCommand:
    def test_empty_command_exits_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        with pytest.raises(SystemExit) as exc_info:
            run_module.run(
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
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        return env

    def test_dry_run_no_sandbox_prints_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        run_module.run(
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
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr(run_module.srt.shutil, "which", lambda *a, **kw: "/bin/srt")

        run_module.run(
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
        assert "srt --settings '<dry-run-settings>' -- echo hi" in out

    def test_dry_run_with_proj_dir_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin", "PROJ_DIR": str(tmp_path)}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr(run_module.srt.shutil, "which", lambda *a, **kw: "/bin/srt")

        run_module.run(
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

    def test_dry_run_no_sandbox_with_memory_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr(run_module.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")
        run_module.run(
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
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr(run_module.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")

        run_module.run(
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
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr(run_module.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")

        run_module.run(
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

    def test_dry_run_no_sandbox_returns_after_printing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Cover dry_run no_sandbox path: prints command and returns without executing."""
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        # No raises: the function should simply return (not raise SystemExit)
        run_module.run(
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
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr(run_module.shutil, "which", lambda *a, **kw: None)

        with pytest.raises(SystemExit) as exc_info:
            run_module.run(
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
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr(run_module.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")

        foreground_calls: list[list[str]] = []
        cleanup_calls: list[list[str] | None] = []

        def fake_run_foreground(cmd: list[str], **kw: Any) -> int:
            foreground_calls.append(cmd)
            cleanup_calls.append(kw["interrupt_cleanup_cmd"])
            return 0

        monkeypatch.setattr(run_module, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit):
            run_module.run(
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
        assert cleanup_calls[0][0:3] == ["systemctl", "--user", "stop"]


# ===========================================================================
# run — no_sandbox live path
# ===========================================================================


class TestRunNoSandboxLive:
    def test_no_sandbox_with_no_limits_runs_foreground(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
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

        monkeypatch.setattr(run_module, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit) as exc_info:
            run_module.run(
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
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())

        def fake_run_foreground(
            cmd: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
            interrupt_cleanup_cmd: list[str] | None = None,
            isolate_process_group: bool = False,
        ) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(run_module, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit) as exc_info:
            run_module.run(
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

    def test_no_sandbox_return_after_resource_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover the unreachable return after _run_with_optional_resource_limits.

        _run_with_optional_resource_limits always raises SystemExit, so the
        trailing return is normally unreachable. By monkeypatching it to
        return normally, we exercise that code path.
        """
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())
        # Monkeypatch _run_with_optional_resource_limits to return normally
        # instead of always raising SystemExit, making the trailing return reachable.
        monkeypatch.setattr(run_module, "_run_with_optional_resource_limits", lambda **kw: None)

        # Should return normally (no SystemExit) with return value of None
        result = run_module.run(
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
        assert result is None

    def test_no_sandbox_no_patch_with_proj_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin", "PROJ_DIR": str(tmp_path)}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_module, "load_run_config", lambda **_: _make_run_config())

        def fake_run_foreground(cmd: list[str], **kw: Any) -> int:
            return 0

        monkeypatch.setattr(run_module, "run_foreground", fake_run_foreground)

        with pytest.raises(SystemExit):
            run_module.run(
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
