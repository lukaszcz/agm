"""Tests for shared project path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Never

import pytest

import agm.project.layout as project_helpers
from agm.project.layout import (
    branch_session_name,
    branch_worktree_path,
    current_project_dir,
    is_main_checkout_branch,
    main_repo_dir,
)
from agm.project.setup import load_worktree_env


def test_current_project_dir_from_project_root(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    assert current_project_dir(project) == project


def test_current_project_dir_from_repo_dir(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    repo_dir = project / "repo"
    repo_dir.mkdir()
    (project / "worktrees").mkdir()

    assert current_project_dir(repo_dir) == project


def test_current_project_dir_from_repo_subdir(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    repo_subdir = project / "repo" / "src"
    repo_subdir.mkdir(parents=True)
    (project / "worktrees").mkdir()

    assert current_project_dir(repo_subdir) == project


def test_current_project_dir_from_worktree_dir(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    branch_dir = project / ".agm" / "worktrees" / "feat" / "branch"
    branch_dir.mkdir(parents=True)

    assert current_project_dir(branch_dir) == project


def test_current_project_dir_from_embedded_project_subdir(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()
    subdir = project / "src"
    subdir.mkdir()

    assert current_project_dir(subdir) == project


def test_main_repo_dir_for_embedded_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    assert main_repo_dir(project) == project


def test_main_checkout_branch_helpers_for_repo_name(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    assert is_main_checkout_branch(project, "repo", repo_branch="main") is True
    assert branch_worktree_path(project, "repo", repo_branch="main") == project / "repo"
    assert branch_session_name(project, "repo") == "proj"


def test_main_checkout_branch_helpers_for_repo_current_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    def fake_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
        del env
        return "main"

    monkeypatch.setattr(project_helpers.git_helpers, "current_branch", fake_current_branch)

    assert is_main_checkout_branch(project, "main", repo_branch="main") is True
    assert branch_worktree_path(project, "main", repo_branch="main") == project / "repo"
    assert branch_session_name(project, "main") == "proj"


def test_branch_helpers_for_worktree_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    def fake_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
        del env
        return "main"

    monkeypatch.setattr(project_helpers.git_helpers, "current_branch", fake_current_branch)

    assert is_main_checkout_branch(project, "feat/x", repo_branch="main") is False
    assert branch_worktree_path(
        project,
        "feat/x",
        repo_branch="main",
    ) == project / "worktrees" / "feat/x"
    assert branch_session_name(project, "feat/x") == "proj/feat/x"


def test_branch_session_name_for_repo_name_does_not_need_repo_branch_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    def fail_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> Never:
        del env
        raise AssertionError("current_branch should not be called for repo alias")

    monkeypatch.setattr(project_helpers.git_helpers, "current_branch", fail_current_branch)

    assert branch_session_name(project, "repo") == "proj"


def test_load_worktree_env_exposes_repo_dir_to_sourced_scripts(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    checkout_dir = project / "worktrees" / "feat"
    checkout_dir.mkdir(parents=True)
    (config_dir / "env.sh").write_text(
        'export CAPTURE_PROJ_DIR="$PROJ_DIR"\nexport CAPTURE_REPO_DIR="$REPO_DIR"\n',
        encoding="utf-8",
    )

    loaded_env = load_worktree_env(project, "feat", checkout_dir=checkout_dir, env=env)

    assert loaded_env["PROJ_DIR"] == str(project)
    assert loaded_env["REPO_DIR"] == str(checkout_dir)
    assert loaded_env["CAPTURE_PROJ_DIR"] == str(project)
    assert loaded_env["CAPTURE_REPO_DIR"] == str(checkout_dir)
