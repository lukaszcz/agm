"""Tests for shared project path resolution helpers."""

from __future__ import annotations

from pathlib import Path

from agm.utils.project import current_project_dir


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
    branch_dir = project / ".worktrees" / "feat" / "branch"
    branch_dir.mkdir(parents=True)

    assert current_project_dir(branch_dir) == project
