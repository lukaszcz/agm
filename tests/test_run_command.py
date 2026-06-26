"""Tests for the run command orchestration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agm.cli_support.args import RunArgs
from agm.commands import run as run_command
from agm.config.general import RunConfig


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
        lambda **_: RunConfig(
            aliases={},
            default_memory_limit=None,
            command_memory_limits={},
            default_swap_limit=None,
            command_swap_limits={},
        ),
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
    assert captured["interrupt_cleanup_cmd"] == ["systemctl", "--user", "stop", unit]

    bootstrap_script = process_prefix[process_prefix.index("-c") + 1]
    assert isinstance(bootstrap_script, str)
    assert 'mkdir -p "${CG}/init"' in bootstrap_script
    assert 'echo $$ > "${CG}/init/cgroup.procs"' in bootstrap_script
    assert 'echo "+memory" > "${CG}/cgroup.subtree_control"' in bootstrap_script
    assert 'export SANDBOX_CGROUP="$CG"' in bootstrap_script


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
        lambda **_: RunConfig(
            aliases={},
            default_memory_limit=None,
            command_memory_limits={},
            default_swap_limit=None,
            command_swap_limits={},
        ),
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
        lambda **_: RunConfig(
            aliases={},
            default_memory_limit="10G",
            command_memory_limits={"echo": "5G"},
            default_swap_limit=None,
            command_swap_limits={},
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
        lambda **_: RunConfig(
            aliases={},
            default_memory_limit="10G",
            command_memory_limits={"echo": "5G"},
            default_swap_limit=None,
            command_swap_limits={},
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
        lambda **_: RunConfig(
            aliases={},
            default_memory_limit=None,
            command_memory_limits={},
            default_swap_limit=None,
            command_swap_limits={},
        ),
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


def test_run_zero_memory_limit_still_wraps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = {"HOME": str(tmp_path / "home"), "PATH": "/bin"}
    (tmp_path / "home").mkdir()

    monkeypatch.setattr(run_command.Path, "cwd", staticmethod(lambda: tmp_path / "work"))
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        run_command,
        "load_run_config",
        lambda **_: RunConfig(
            aliases={},
            default_memory_limit="0",
            command_memory_limits={},
            default_swap_limit=None,
            command_swap_limits={},
        ),
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
        lambda **_: RunConfig(
            aliases={"claude": "claude --dangerously-skip-permissions"},
            default_memory_limit=None,
            command_memory_limits={},
            default_swap_limit=None,
            command_swap_limits={},
        ),
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
        return 0

    monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

    with pytest.raises(SystemExit):
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
        lambda **_: RunConfig(
            aliases={"claude": "claude --dangerously-skip-permissions"},
            default_memory_limit=None,
            command_memory_limits={},
            default_swap_limit=None,
            command_swap_limits={},
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
        lambda **_: RunConfig(
            aliases={"myagent": "claude"},
            default_memory_limit=None,
            command_memory_limits={},
            default_swap_limit=None,
            command_swap_limits={},
        ),
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
        return 0

    monkeypatch.setattr(run_command, "run_foreground", fake_run_foreground)

    with pytest.raises(SystemExit):
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
