"""Tests for the agm run command: sandbox delegation, resource limits, and dry run."""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from agm.cli_support.args import RunArgs
from agm.commands import run as run_command
from agm.config.general import RunConfig


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
        aliases=aliases or {},
        default_memory_limit=memory_limit,
        command_memory_limits=command_memory_limits or {},
        default_swap_limit=swap_limit,
        command_swap_limits=command_swap_limits or {},
        default_pty=pty,
        command_ptys=command_ptys or {},
    )


def test_run_delegates_sandbox_execution_to_srt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(),
    )
    monkeypatch.setattr(run_command.shutil, "which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}

    def fake_run_sandboxed(
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        home: Path,
        proj_dir: Path | None,
        command_name: str,
        alias_command_name: str | None,
        settings_file: str | None,
        patch_proj_dir: Path | None,
        process_prefix: list[str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> None:
        captured.update(
            command=command,
            cwd=cwd,
            env=env,
            home=home,
            proj_dir=proj_dir,
            command_name=command_name,
            alias_command_name=alias_command_name,
            settings_file=settings_file,
            patch_proj_dir=patch_proj_dir,
            process_prefix=process_prefix,
            interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        )

    monkeypatch.setattr(run_command.srt, "run_sandboxed", fake_run_sandboxed)

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

    assert captured["command"] == ["echo", "hi"]
    assert captured["cwd"] == tmp_path / "work"
    assert captured["env"] == dict(env)
    assert captured["home"] == tmp_path / "home"
    assert captured["proj_dir"] is None
    assert captured["command_name"] == "echo"
    assert captured["alias_command_name"] is None
    assert captured["settings_file"] is None
    assert captured["patch_proj_dir"] is None

    process_prefix = captured["process_prefix"]
    assert isinstance(process_prefix, list)
    assert process_prefix[:3] == ["systemd-run", "--user", "--scope"]
    assert "MemoryMax=32G" in process_prefix
    assert "MemorySwapMax=0" in process_prefix
    assert "Delegate=yes" in process_prefix
    assert "--unit" in process_prefix
    assert "bash" in process_prefix
    unit = process_prefix[process_prefix.index("--unit") + 1]
    assert captured["interrupt_cleanup_cmd"] == [
        "systemctl",
        "--user",
        "--no-block",
        "stop",
        unit,
    ]

    bootstrap_script = process_prefix[process_prefix.index("-c") + 1]
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
    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", {"HOME": str(home), "PATH": "/bin"})
    monkeypatch.setattr(run_command.os, "isatty", lambda _fd: interactive)
    monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        run_command.srt,
        "run_sandboxed",
        lambda **kwargs: captured.update(kwargs),
    )

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

    expected_prefix = [sys.executable, "-m", "agm.sandbox.pty", "--"] if wrapped else []
    assert captured["command"] == ["echo", "hi"]
    assert captured["process_prefix"] == expected_prefix


def test_run_patches_sandbox_when_project_is_discovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    project = tmp_path / "project"
    cwd = project / "repo"
    (tmp_path / "home").mkdir()
    cwd.mkdir(parents=True)

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: cwd))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        "agm.config.context.discover_current_project_dir",
        lambda current, env=None: project,
    )
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(),
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        run_command.srt,
        "run_sandboxed",
        lambda **kwargs: captured.update(kwargs),
    )

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

    assert captured["proj_dir"] == project
    assert captured["patch_proj_dir"] == project


def test_run_no_memory_limit_omits_memory_max_but_keeps_default_swap_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(
            memory_limit="10G",
            command_memory_limits={"echo": "5G"},
        ),
    )

    captured: dict[str, object] = {}

    def fake_run_sandboxed(
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        home: Path,
        proj_dir: Path | None,
        command_name: str,
        alias_command_name: str | None,
        settings_file: str | None,
        patch_proj_dir: Path | None,
        process_prefix: list[str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> None:
        captured.update(
            command=command,
            cwd=cwd,
            env=env,
            home=home,
            proj_dir=proj_dir,
            command_name=command_name,
            alias_command_name=alias_command_name,
            settings_file=settings_file,
            patch_proj_dir=patch_proj_dir,
            process_prefix=process_prefix,
            interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        )

    monkeypatch.setattr(run_command.srt, "run_sandboxed", fake_run_sandboxed)

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

    assert captured["command"] == ["echo", "hi"]
    process_prefix = captured["process_prefix"]
    assert isinstance(process_prefix, list)
    assert "MemoryMax=5G" not in process_prefix
    assert "MemorySwapMax=0" in process_prefix
    assert captured["interrupt_cleanup_cmd"] is not None


def test_run_no_memory_and_swap_limit_disable_wrapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(memory_limit="10G", command_memory_limits={"echo": "5G"}),
    )

    captured: dict[str, object] = {}

    def fake_run_sandboxed(
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        home: Path,
        proj_dir: Path | None,
        command_name: str,
        alias_command_name: str | None,
        settings_file: str | None,
        patch_proj_dir: Path | None,
        process_prefix: list[str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> None:
        captured.update(process_prefix=process_prefix, interrupt_cleanup_cmd=interrupt_cleanup_cmd)

    monkeypatch.setattr(run_command.srt, "run_sandboxed", fake_run_sandboxed)

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

    assert captured["process_prefix"] == []
    assert captured["interrupt_cleanup_cmd"] is None


def test_run_no_swap_limit_omits_memory_swap_max_property(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(),
    )
    monkeypatch.setattr(run_command.shutil, "which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}

    def fake_run_sandboxed(
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        home: Path,
        proj_dir: Path | None,
        command_name: str,
        alias_command_name: str | None,
        settings_file: str | None,
        patch_proj_dir: Path | None,
        process_prefix: list[str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> None:
        captured.update(process_prefix=process_prefix, interrupt_cleanup_cmd=interrupt_cleanup_cmd)

    monkeypatch.setattr(run_command.srt, "run_sandboxed", fake_run_sandboxed)

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

    process_prefix = captured["process_prefix"]
    assert isinstance(process_prefix, list)
    assert "MemoryMax=32G" in process_prefix
    assert "MemorySwapMax=0" not in process_prefix
    assert captured["interrupt_cleanup_cmd"] is not None


def test_run_zero_memory_limit_still_wraps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(memory_limit="0"),
    )
    monkeypatch.setattr(run_command.shutil, "which", lambda *_args, **_kwargs: "/bin/tool")

    captured: dict[str, object] = {}

    def fake_run_sandboxed(
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        home: Path,
        proj_dir: Path | None,
        command_name: str,
        alias_command_name: str | None,
        settings_file: str | None,
        patch_proj_dir: Path | None,
        process_prefix: list[str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> None:
        captured.update(process_prefix=process_prefix, interrupt_cleanup_cmd=interrupt_cleanup_cmd)

    monkeypatch.setattr(run_command.srt, "run_sandboxed", fake_run_sandboxed)

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

    process_prefix = captured["process_prefix"]
    assert isinstance(process_prefix, list)
    assert "MemoryMax=0" in process_prefix
    assert "MemorySwapMax=0" in process_prefix
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

    def fake_run_foreground(
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
        isolate_process_group: bool = False,
    ) -> int:
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

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: _make_run_config(aliases={"claude": "claude --dangerously-skip-permissions"}),
    )

    captured: dict[str, object] = {}

    def fake_run_sandboxed(
        *,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        home: Path,
        proj_dir: Path | None,
        command_name: str,
        alias_command_name: str | None,
        settings_file: str | None,
        patch_proj_dir: Path | None,
        process_prefix: list[str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
    ) -> None:
        captured.update(
            command=command,
            command_name=command_name,
            alias_command_name=alias_command_name,
        )

    monkeypatch.setattr(run_command.srt, "run_sandboxed", fake_run_sandboxed)

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

    assert captured["command"] == ["claude", "--dangerously-skip-permissions", "--some-arg"]
    assert captured["command_name"] == "claude"
    assert captured["alias_command_name"] == "claude"


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

    def fake_run_foreground(
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        interrupt_cleanup_cmd: list[str] | None = None,
        isolate_process_group: bool = False,
    ) -> int:
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
        monkeypatch.setattr(run_command.srt.shutil, "which", lambda *a, **kw: "/bin/srt")

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
        assert "srt --settings '<dry-run-settings>' -- echo hi" in out

    def test_dry_run_pty_keeps_the_alias_target_as_the_sandboxed_identity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A terminal relay must not stand in for the alias target nor enter the sandbox."""

        self._enable_dry_run()
        agm_home = tmp_path / "agm-home"
        (agm_home / "sandbox").mkdir(parents=True)
        (agm_home / "sandbox" / "claude.json").write_text("{}", encoding="utf-8")
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin", "AGM_HOME": str(agm_home)}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command.os, "isatty", lambda _fd: True)
        monkeypatch.setattr(
            run_command,
            "load_run_config",
            lambda **_: _make_run_config(aliases={"myagent": "claude"}),
        )
        monkeypatch.setattr(run_command.srt.shutil, "which", lambda *a, **kw: "/bin/srt")

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
        assert "agm-home/sandbox/claude.json" in out
        assert Path(sys.executable).name + ".json" not in out
        assert (
            f"{shlex.quote(sys.executable)} -m agm.sandbox.pty "
            "-- srt --settings '<dry-run-settings>' "
            "-- claude --some-arg"
        ) in out

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
        monkeypatch.setattr(run_command.srt.shutil, "which", lambda *a, **kw: "/bin/srt")

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

    def test_dry_run_no_sandbox_with_memory_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._enable_dry_run()
        env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
        (tmp_path / "home").mkdir()
        monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(os, "environ", env)
        monkeypatch.setattr(run_command, "load_run_config", lambda **_: _make_run_config())
        monkeypatch.setattr(run_command.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")
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
        monkeypatch.setattr(run_command.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")

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
        monkeypatch.setattr(run_command.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")

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
        monkeypatch.setattr(run_command.shutil, "which", lambda *a, **kw: None)

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
        monkeypatch.setattr(run_command.shutil, "which", lambda *a, **kw: "/usr/bin/systemd-run")

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
