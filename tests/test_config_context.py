from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.config.context as context_module
from agm.config.context import current_config_context


def test_current_config_context_uses_explicit_proj_dir(
    tmp_path: Path, env: dict[str, str]
) -> None:
    home = Path(env["HOME"])
    project = tmp_path / "project"
    repo_dir = project / "repo"
    repo_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)

    context = current_config_context(
        cwd=tmp_path,
        env={**env, "PROJ_DIR": str(project)},
    )

    assert context.home == home
    assert context.proj_dir == project
    assert context.cwd == tmp_path


def test_current_config_context_uses_explicit_proj_dir_pointing_to_agm(
    tmp_path: Path, env: dict[str, str]
) -> None:
    """When PROJ_DIR points to .agm (embedded layout), context uses it as project dir."""
    home = Path(env["HOME"])
    project = tmp_path / "project"
    agm_dir = project / ".agm"
    agm_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    context = current_config_context(
        cwd=tmp_path,
        env={**env, "PROJ_DIR": str(agm_dir)},
    )

    assert context.home == home
    assert context.proj_dir == agm_dir
    assert context.cwd == tmp_path


def test_current_config_context_uses_discovered_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    project = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.setattr(
        context_module, "discover_current_project_dir", lambda cwd, env=None: project
    )

    context = current_config_context(cwd=cwd, env={"HOME": str(home)})

    assert context.proj_dir == project


def test_current_config_context_uses_none_when_discovery_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"

    def fail(_cwd: Path, *, env: dict[str, str] | None = None) -> Path:
        del env
        raise SystemExit(1)

    monkeypatch.setattr(context_module, "discover_current_project_dir", fail)

    context = current_config_context(cwd=tmp_path, env={"HOME": str(home)})

    assert context.proj_dir is None


def test_current_config_context_resolves_implicit_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(link)
    monkeypatch.setattr(
        context_module, "discover_current_project_dir", lambda cwd, env=None: cwd
    )

    context = current_config_context(env={"HOME": str(home)})

    assert context.cwd == target
