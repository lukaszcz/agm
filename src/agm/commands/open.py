"""agm open."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.commands.args import OpenArgs
from agm.core.fs import mkdir
from agm.parser import exit_with_usage_error
from agm.project.config_git import commit_config_dir_changes
from agm.project.dependency_env import ensure_dependency_configs_for_branch
from agm.project.layout import (
    branch_session_name,
    branch_worktree_path,
    is_main_checkout_branch,
    parent_config_branch,
    project_config_dir,
    project_name,
    project_repo_dir,
    require_current_project_dir,
)
from agm.project.setup import load_worktree_env
from agm.project.worktree import (
    branch_exists,
    ensure_worktree,
    has_expected_worktree,
    resolve_parent_checkout_dir,
)
from agm.tmux.session import (
    create_tmux_session,
    focus_tmux_session,
    queue_command_in_session,
)
from agm.tmux.session import validate_pane_count as validate_tmux_pane_count


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
        repo_branch=git_helpers.current_branch(project_repo_dir(proj_dir)),
    )


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
        command=["agm", "setup"],
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
    proj_dir = require_current_project_dir(current)
    repo_branch = git_helpers.current_branch(project_repo_dir(proj_dir))

    if branch is None:
        repo_path = project_repo_dir(proj_dir)
        session_name = project_name(proj_dir)
    else:
        repo_path = branch_worktree_path(proj_dir, branch, repo_branch=repo_branch)
        if not has_expected_worktree(proj_dir, branch):
            print(
                f"error: branch '{branch}' is not checked out at {repo_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        session_name = branch_session_name(proj_dir, branch)
        ensure_dependency_configs_for_branch(project_dir=proj_dir, branch=branch)
        commit_config_dir_changes(
            proj_dir, f"chore: update config for {branch}",
            add_paths=[project_config_dir(proj_dir) / branch], env={},
        )
    env = load_worktree_env(proj_dir, branch, checkout_dir=repo_path)
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
    proj_dir = require_current_project_dir(current)
    repo_path = branch_path(proj_dir, branch)
    mkdir(repo_path, parents=True, exist_ok=True)
    ensure_dependency_configs_for_branch(
        project_dir=proj_dir,
        branch=branch,
        parent_branch=parent_config_branch(proj_dir, parent),
    )
    env = load_worktree_env(proj_dir, branch, checkout_dir=repo_path)
    parent_dir = resolve_parent_checkout_dir(proj_dir, parent, env=env)
    ensure_worktree(
        new_branch=branch,
        worktrees_dir=None,
        branch=None,
        existing_ok=False,
        cwd=parent_dir,
        env=env,
    )
    commit_config_dir_changes(
        proj_dir, f"chore: add config for {branch}",
        add_paths=[project_config_dir(proj_dir) / branch], env=env,
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
    proj_dir = require_current_project_dir(current)
    repo_path = branch_path(proj_dir, branch)
    mkdir(repo_path, parents=True, exist_ok=True)
    ensure_dependency_configs_for_branch(
        project_dir=proj_dir, branch=branch,
        parent_branch=parent_config_branch(proj_dir, parent),
    )
    env = load_worktree_env(proj_dir, branch, checkout_dir=repo_path)
    parent_dir = resolve_parent_checkout_dir(proj_dir, parent, env=env)
    ensure_worktree(
        new_branch=None,
        worktrees_dir=None,
        branch=branch,
        existing_ok=True,
        cwd=parent_dir,
        env=env,
    )
    commit_config_dir_changes(
        proj_dir, f"chore: add config for {branch}",
        add_paths=[project_config_dir(proj_dir) / branch], env=env,
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
    proj_dir = require_current_project_dir(current)

    repo_dir = project_repo_dir(proj_dir)
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


def run(args: OpenArgs) -> None:
    smart_open_session(
        detached=args.detached,
        pane_count=args.pane_count,
        parent=args.parent,
        branch=args.branch,
    )
