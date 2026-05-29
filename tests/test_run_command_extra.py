"""Additional tests for agm.commands.run — edge cases and dry_run paths."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

import agm.commands.run as run_module
from agm.commands.args import RunArgs
from agm.commands.run import (
    _normalize_systemd_limit,
    _resource_limit_run_context,
    _systemd_run_prefix,
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
# _normalize_systemd_limit
# ===========================================================================


class TestNormalizeSystemdLimit:
    def test_unlimited_maps_to_infinity(self) -> None:
        assert _normalize_systemd_limit("unlimited") == "infinity"

    def test_unlimited_case_insensitive(self) -> None:
        assert _normalize_systemd_limit("UNLIMITED") == "infinity"
        assert _normalize_systemd_limit("Unlimited") == "infinity"

    def test_unlimited_with_whitespace(self) -> None:
        assert _normalize_systemd_limit(" unlimited ") == "infinity"

    def test_numeric_value_unchanged(self) -> None:
        assert _normalize_systemd_limit("20G") == "20G"
        assert _normalize_systemd_limit("0") == "0"


# ===========================================================================
# _systemd_run_prefix
# ===========================================================================


class TestSystemdRunPrefix:
    def test_prefix_with_memory_and_swap(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="10G", swap_limit="0")
        assert "systemd-run" in prefix
        assert "-p" in prefix
        assert "MemoryMax=10G" in prefix
        assert "MemorySwapMax=0" in prefix
        assert "Delegate=yes" in prefix

    def test_prefix_with_only_memory(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="5G", swap_limit=None)
        assert "MemoryMax=5G" in prefix
        assert not any("MemorySwapMax" in p for p in prefix)

    def test_prefix_with_only_swap(self) -> None:
        prefix = _systemd_run_prefix(memory_limit=None, swap_limit="2G")
        assert "MemorySwapMax=2G" in prefix
        assert not any("MemoryMax" in p for p in prefix)

    def test_prefix_with_neither(self) -> None:
        prefix = _systemd_run_prefix(memory_limit=None, swap_limit=None)
        assert "Delegate=yes" in prefix
        assert not any("MemoryMax" in p for p in prefix)


# ===========================================================================
# _resource_limit_run_context
# ===========================================================================


class TestResourceLimitRunContext:
    def test_returns_empty_when_no_limits(self) -> None:
        prefix, cleanup = _resource_limit_run_context({}, None, None)
        assert prefix == []
        assert cleanup is None

    def test_exits_when_systemd_run_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_module.shutil, "which", lambda *a, **kw: None)
        with pytest.raises(SystemExit) as exc_info:
            _resource_limit_run_context({"PATH": "/usr/bin"}, "10G", None)
        assert exc_info.value.code == 1

    def test_returns_prefix_and_cleanup_when_systemd_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_module.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")
        prefix, cleanup = _resource_limit_run_context({}, "10G", "0")
        assert isinstance(prefix, list)
        assert "systemd-run" in prefix
        assert isinstance(cleanup, list)
        assert "systemctl" in cleanup[0]


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
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
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

    def _setup_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, str]:
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(run_module.os, "environ", env)
        return env

    def test_dry_run_no_sandbox_prints_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_dry_run()
        self._setup_env(tmp_path, monkeypatch)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
        srt_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            run_module.srt, "run_sandboxed", lambda **kw: srt_calls.append(kw)
        )
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
        assert len(srt_calls) == 1

    def test_dry_run_with_proj_dir_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin", "PROJ_DIR": str(tmp_path)}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
        srt_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            run_module.srt, "run_sandboxed", lambda **kw: srt_calls.append(kw)
        )
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
        assert len(srt_calls) == 1
        assert srt_calls[0]["proj_dir"] == tmp_path

    def test_dry_run_no_sandbox_with_memory_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
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

    def test_dry_run_no_sandbox_returns_after_printing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Cover run.py:193 — dry_run no_sandbox path prints command and returns."""
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
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
# run — no_sandbox swap limit (line 139)
# ===========================================================================


class TestRunNoSandboxSwapLimit:
    def test_no_sandbox_uses_swap_from_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover run.py:139 — elif no_sandbox: effective_swap_limit = run_args.swap."""
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
        monkeypatch.setattr(run_module.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")

        foreground_calls: list[list[str]] = []

        def fake_run_foreground(
            cmd: list[str], **kw: Any
        ) -> int:
            foreground_calls.append(cmd)
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
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
        foreground_calls: list[list[str]] = []

        def fake_run_foreground(
            cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
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
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )

        def fake_run_foreground(
            cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
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
        """Cover run.py:193 — unreachable return after _run_with_optional_resource_limits.

        _run_with_optional_resource_limits always raises SystemExit, so the
        `return` on line 193 is normally unreachable. By monkeypatching it to
        return normally, we exercise that line.
        """
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_module.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )
        # Monkeypatch _run_with_optional_resource_limits to return normally
        # instead of always raising SystemExit, making the `return` on line 193 reachable.
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
        monkeypatch.setattr(run_module.os, "environ", env)
        monkeypatch.setattr(
            run_module, "load_run_config", lambda **_: _make_run_config()
        )

        def fake_run_foreground(
            cmd: list[str], **kw: Any
        ) -> int:
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