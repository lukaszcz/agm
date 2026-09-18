"""Tests for AGM workspace discovery in agm.project.layout."""

from __future__ import annotations

from pathlib import Path

from agm.project.layout import Workspace, project_workspaces
from tests._git_helpers import add_linked_worktree, git_run, init_repo


def _split_project(tmp_path: Path, env: dict[str, str]) -> tuple[Path, Path]:
    project_dir = tmp_path / "proj"
    repo_dir = init_repo(project_dir / "repo", env)
    (project_dir / "worktrees").mkdir()
    return project_dir, repo_dir


def test_lists_main_then_branch_workspaces_by_name(tmp_path: Path, env: dict[str, str]) -> None:
    project_dir, repo_dir = _split_project(tmp_path, env)
    worktrees_dir = project_dir / "worktrees"
    add_linked_worktree(repo_dir, worktrees_dir / "fix", env, branch="fix")
    add_linked_worktree(repo_dir, worktrees_dir / "feat" / "a", env, branch="feat/a")

    workspaces = project_workspaces(project_dir, env=env)

    assert [(ws.name, ws.branch, ws.main) for ws in workspaces] == [
        ("repo", "main", True),
        ("feat/a", "feat/a", False),
        ("fix", "fix", False),
    ]
    assert workspaces[1].path.resolve() == (worktrees_dir / "feat" / "a").resolve()


def test_ignores_git_worktrees_outside_the_worktrees_dir(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project_dir, repo_dir = _split_project(tmp_path, env)
    add_linked_worktree(repo_dir, tmp_path / "elsewhere", env, branch="outside")
    add_linked_worktree(repo_dir, project_dir / "deps" / "odd", env, branch="odd")

    assert [ws.name for ws in project_workspaces(project_dir, env=env)] == ["repo"]


def test_detached_workspaces_keep_their_name(tmp_path: Path, env: dict[str, str]) -> None:
    project_dir, repo_dir = _split_project(tmp_path, env)
    feat_dir = add_linked_worktree(repo_dir, project_dir / "worktrees" / "feat", env, branch="feat")
    git_run(feat_dir, ["checkout", "-q", "--detach"], env)
    git_run(repo_dir, ["checkout", "-q", "--detach"], env)

    workspaces = project_workspaces(project_dir, env=env)

    assert [(ws.name, ws.branch) for ws in workspaces] == [("repo", None), ("feat", None)]


def test_embedded_layout_uses_agm_worktrees_dir(tmp_path: Path, env: dict[str, str]) -> None:
    repo_dir = init_repo(tmp_path / "proj", env)
    project_dir = repo_dir / ".agm"
    (project_dir / "worktrees").mkdir(parents=True)
    add_linked_worktree(repo_dir, project_dir / "worktrees" / "feat", env, branch="feat")
    add_linked_worktree(repo_dir, tmp_path / "elsewhere", env, branch="outside")

    workspaces = project_workspaces(project_dir, env=env)

    assert workspaces == [
        Workspace(name="repo", path=workspaces[0].path, branch="main", main=True),
        Workspace(name="feat", path=workspaces[1].path, branch="feat", main=False),
    ]
    assert workspaces[0].path.resolve() == repo_dir.resolve()
