"""Tests for shared project path resolution helpers."""

from __future__ import annotations

from pathlib import Path

from agm.utils.project import (
    branch_session_name,
    branch_worktree_path,
    current_project_dir,
    is_main_checkout_branch,
)


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


def test_main_checkout_branch_helpers_for_repo_name(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    assert is_main_checkout_branch(project, "repo", repo_branch="main") is True
    assert branch_worktree_path(project, "repo", repo_branch="main") == project / "repo"
    assert branch_session_name(project, "repo", repo_branch="main") == "proj"


def test_main_checkout_branch_helpers_for_repo_current_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    assert is_main_checkout_branch(project, "main", repo_branch="main") is True
    assert branch_worktree_path(project, "main", repo_branch="main") == project / "repo"
    assert branch_session_name(project, "main", repo_branch="main") == "proj"


def test_branch_helpers_for_worktree_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    (project / "worktrees").mkdir()

    assert is_main_checkout_branch(project, "feat/x", repo_branch="main") is False
    assert branch_worktree_path(
        project,
        "feat/x",
        repo_branch="main",
    ) == project / "worktrees" / "feat/x"
    assert branch_session_name(project, "feat/x", repo_branch="main") == "proj/feat/x"
