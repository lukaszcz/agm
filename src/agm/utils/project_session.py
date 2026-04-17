"""Helpers for opening project tmux sessions."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.parser import exit_with_usage_error
from agm.tmux.session import (
    create_tmux_session,
    focus_tmux_session,
    kill_tmux_session,
    queue_command_in_session,
)
from agm.tmux.session import validate_pane_count as validate_tmux_pane_count
from agm.utils.project import (
    branch_session_name,
    branch_worktree_path,
    current_project_dir,
    is_main_checkout_branch,
    main_repo_dir,
)
from agm.utils.worktree import ensure_worktree, load_worktree_env, remove_worktree


def validate_pane_count(pane_count: str | None) -> None:
    try:
        validate_tmux_pane_count(["open"], pane_count)
    except SystemExit as exc:
        if exc.code != 1:
            raise
        exit_with_usage_error(["open"], "error: pane count must be a positive integer")


def branch_path(proj_dir: Path, branch: str) -> Path:
    return branch_worktree_path(
        proj_dir,
        branch,
        repo_branch=git_helpers.current_branch(main_repo_dir(proj_dir)),
    )


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
    detached: bool,
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
    if detached:
        return
    raise SystemExit(focus_tmux_session(session_name=created_session, cwd=repo_path, env=env))


def open_session(
    *,
    detached: bool,
    pane_count: str | None,
    branch: str | None,
    cwd: Path | None = None,
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
    repo_branch = git_helpers.current_branch(main_repo_dir(proj_dir))

    if branch is None:
        repo_path = main_repo_dir(proj_dir)
        session_name = proj_dir.name
    else:
        repo_path = branch_worktree_path(proj_dir, branch, repo_branch=repo_branch)
        if not has_expected_worktree(proj_dir, branch):
            print(
                f"error: branch '{branch}' is not checked out at {repo_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        session_name = branch_session_name(proj_dir, branch)
    env = load_worktree_env(proj_dir, branch, shell_cwd=repo_path)
    create_tmux_session(
        detach=detached,
        pane_count=pane_count,
        session_name=session_name,
        cwd=repo_path,
        env=env,
    )


def new_session(
    *,
    detached: bool,
    pane_count: str | None,
    parent: str | None,
    branch: str,
    cwd: Path | None = None,
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
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
        detached=detached,
        pane_count=pane_count,
        session_name=branch_session_name(proj_dir, branch),
        repo_path=repo_path,
        env=env,
    )


def checkout_session(
    *,
    detached: bool,
    pane_count: str | None,
    parent: str | None,
    branch: str,
    cwd: Path | None = None,
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
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
        detached=detached,
        pane_count=pane_count,
        session_name=branch_session_name(proj_dir, branch),
        repo_path=repo_path,
        env=env,
    )


def smart_open_session(
    *,
    detached: bool,
    pane_count: str | None,
    parent: str | None,
    branch: str,
    cwd: Path | None = None,
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)

    repo_dir = main_repo_dir(proj_dir)
    if is_main_checkout_branch(
        proj_dir,
        branch,
        repo_branch=git_helpers.current_branch(repo_dir),
    ):
        open_session(detached=detached, pane_count=pane_count, branch=None, cwd=current)
        return

    git_helpers.fetch(repo_dir)
    if has_expected_worktree(proj_dir, branch):
        open_session(detached=detached, pane_count=pane_count, branch=branch, cwd=current)
        return
    if branch_exists(repo_dir, branch):
        checkout_session(
            detached=detached,
            pane_count=pane_count,
            parent=parent,
            branch=branch,
            cwd=current,
        )
        return
    new_session(
        detached=detached,
        pane_count=pane_count,
        parent=parent,
        branch=branch,
        cwd=current,
    )


def close_session(*, branch: str, cwd: Path | None = None) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = current_project_dir(current)
    repo_dir = main_repo_dir(proj_dir)
    env = load_worktree_env(proj_dir, None, shell_cwd=repo_dir)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    if is_main_checkout_branch(proj_dir, branch, repo_branch=repo_branch):
        print(
            (
                f"error: '{branch}' resolves to the main repo checkout at "
                f"{repo_dir} and cannot be removed"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    remove_worktree(force=False, branch=branch, cwd=current, env=env)
    session_name = branch_session_name(proj_dir, branch)
    status = kill_tmux_session(session_name=session_name, cwd=repo_dir, env=env)
    if status != 0:
        raise SystemExit(status)
