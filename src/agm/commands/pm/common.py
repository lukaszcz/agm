"""Shared helpers for project session commands."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.tmux.session import (
    create_tmux_session,
    focus_tmux_session,
    queue_command_in_session,
)
from agm.utils.project import current_project_dir, default_worktrees_dir, main_repo_dir
from agm.utils.worktree import ensure_worktree, load_worktree_env


def validate_pane_count(pane_count: str | None) -> None:
    if pane_count is None:
        return
    if not pane_count.isdigit() or int(pane_count) < 1:
        print("error: pane count must be a positive integer", file=sys.stderr)
        raise SystemExit(1)


def branch_path(proj_dir: Path, branch: str) -> Path:
    return default_worktrees_dir(proj_dir) / branch


def expected_branch_path(proj_dir: Path, branch: str) -> Path:
    return branch_path(proj_dir, branch).resolve(strict=False)


def resolve_parent_checkout_dir(proj_dir: Path, parent: str | None, *, env: dict[str, str]) -> Path:
    repo_dir = main_repo_dir(proj_dir)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    resolved_parent = parent or repo_branch
    if resolved_parent == repo_branch:
        return repo_dir
    return branch_path(proj_dir, resolved_parent)


def has_expected_worktree(
    proj_dir: Path, branch: str, *, env: dict[str, str] | None = None
) -> bool:
    repo_dir = main_repo_dir(proj_dir)
    expected_path = expected_branch_path(proj_dir, branch)
    for worktree in git_helpers.worktree_list(repo_dir, env=env):
        if worktree.branch == branch and worktree.path.resolve(strict=False) == expected_path:
            return True
    return False


def branch_exists(repo_dir: Path, branch: str, *, env: dict[str, str] | None = None) -> bool:
    return git_helpers.local_branch_exists(
        repo_dir, branch, env=env
    ) or git_helpers.remote_branch_exists(repo_dir, branch, env=env)


def queue_setup_and_focus_session(
    *,
    pane_count: str | None,
    session_name: str,
    repo_path: Path,
    env: dict[str, str],
) -> None:
    created_session = create_tmux_session(
        detach=True,
        pane_count=pane_count,
        session_name=session_name,
        cwd=repo_path,
        env=env,
    )
    if created_session is None:
        raise AssertionError("detached tmux session creation did not return a session name")
    queue_command_in_session(
        session_name=created_session,
        command=[sys.executable, "-m", "agm.cli", "wt", "setup"],
        cwd=repo_path,
        env=env,
    )
    raise SystemExit(focus_tmux_session(session_name=created_session, cwd=repo_path, env=env))


def open_session(*, pane_count: str | None, branch: str | None, cwd: Path | None = None) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
    proj_name = proj_dir.name

    if branch is None:
        repo_path = main_repo_dir(proj_dir)
    else:
        repo_path = branch_path(proj_dir, branch)
        if not has_expected_worktree(proj_dir, branch):
            print(
                f"error: branch '{branch}' is not checked out at {repo_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
    env = load_worktree_env(proj_dir, branch, shell_cwd=repo_path)
    session_name = f"{proj_name}/{branch}" if branch else proj_name
    create_tmux_session(
        detach=False,
        pane_count=pane_count,
        session_name=session_name,
        cwd=repo_path,
        env=env,
    )


def new_session(
    *, pane_count: str | None, parent: str | None, branch: str, cwd: Path | None = None
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
    proj_name = proj_dir.name
    repo_path = branch_path(proj_dir, branch)
    repo_path.mkdir(parents=True, exist_ok=True)
    env = load_worktree_env(proj_dir, branch, shell_cwd=repo_path)
    parent_dir = resolve_parent_checkout_dir(proj_dir, parent, env=env)
    ensure_worktree(
        new_branch=branch,
        worktrees_dir=None,
        branch=None,
        existing_ok=False,
        cwd=parent_dir,
        env=env,
    )
    queue_setup_and_focus_session(
        pane_count=pane_count,
        session_name=f"{proj_name}/{branch}",
        repo_path=repo_path,
        env=env,
    )


def checkout_session(
    *, pane_count: str | None, parent: str | None, branch: str, cwd: Path | None = None
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
    proj_name = proj_dir.name
    repo_path = branch_path(proj_dir, branch)
    repo_path.mkdir(parents=True, exist_ok=True)
    env = load_worktree_env(proj_dir, branch, shell_cwd=repo_path)
    parent_dir = resolve_parent_checkout_dir(proj_dir, parent, env=env)
    ensure_worktree(
        new_branch=None,
        worktrees_dir=None,
        branch=branch,
        existing_ok=True,
        cwd=parent_dir,
        env=env,
    )
    queue_setup_and_focus_session(
        pane_count=pane_count,
        session_name=f"{proj_name}/{branch}",
        repo_path=repo_path,
        env=env,
    )


def smart_open_session(
    *,
    pane_count: str | None,
    parent: str | None,
    branch: str,
    cwd: Path | None = None,
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)

    repo_dir = main_repo_dir(proj_dir)
    if branch in {"repo", git_helpers.current_branch(repo_dir)}:
        open_session(pane_count=pane_count, branch=None, cwd=current)
        return

    git_helpers.fetch(repo_dir)
    if has_expected_worktree(proj_dir, branch):
        open_session(pane_count=pane_count, branch=branch, cwd=current)
        return
    if branch_exists(repo_dir, branch):
        checkout_session(pane_count=pane_count, parent=parent, branch=branch, cwd=current)
        return
    new_session(pane_count=pane_count, parent=parent, branch=branch, cwd=current)
