"""agm workspace open."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.cli_support.args import OpenArgs
from agm.core.fs import mkdir
from agm.core.path import display_path
from agm.parser import exit_with_usage_error
from agm.project.config_git import commit_config_dir_changes
from agm.project.dependency_env import ensure_dependency_configs_for_branch
from agm.project.layout import (
    branch_session_name,
    branch_worktree_path,
    is_main_workspace_branch,
    parent_config_branch,
    project_config_dir,
    project_name,
    project_repo_dir,
    require_current_project_dir,
)
from agm.project.workspace_env import load_workspace_env
from agm.project.workspace_shell import ensure_workspace_shell
from agm.project.worktree import (
    branch_exists,
    ensure_worktree,
    has_expected_worktree,
)
from agm.tmux.session import (
    create_tmux_session,
    focus_tmux_session,
    queue_command_in_session,
)
from agm.tmux.session import validate_pane_count as validate_tmux_pane_count


def validate_pane_count(pane_count: str | None) -> int:
    try:
        return validate_tmux_pane_count(["open"], pane_count)
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


def require_parent_start_point(project_dir: Path, parent: str | None) -> str | None:
    """Return the new workspace start point, validating an explicit parent branch."""

    start_point = parent_config_branch(project_dir, parent)
    if start_point is None:
        return None
    if branch_exists(project_repo_dir(project_dir), start_point):
        return start_point
    print(f"error: parent branch '{start_point}' does not exist", file=sys.stderr)
    raise SystemExit(1)


def warn_parent_ignored_for_existing_branch(branch: str) -> None:
    """Warn that --parent has no effect when the target branch already exists."""

    print(
        f"warning: branch '{branch}' already exists; ignoring --parent",
        file=sys.stderr,
    )


def create_configured_workspace_session(
    *,
    detached: bool,
    pane_count: str | None,
    session_name: str,
    repo_path: Path,
    run_setup: bool,
) -> None:
    validate_pane_count(pane_count)
    created_session = create_tmux_session(
        detach=True,
        pane_count=pane_count,
        session_name=session_name,
        cwd=repo_path,
        shell_command=str(ensure_workspace_shell(session_name)),
    )
    if run_setup:
        queue_command_in_session(
            session_name=created_session,
            command=["agm", "workspace", "setup"],
            cwd=repo_path,
        )
    if detached:
        return
    raise SystemExit(focus_tmux_session(session_name=created_session, cwd=repo_path))


def queue_setup_and_focus_workspace_session(
    *,
    detached: bool,
    pane_count: str | None,
    session_name: str,
    repo_path: Path,
) -> None:
    create_configured_workspace_session(
        detached=detached,
        pane_count=pane_count,
        session_name=session_name,
        repo_path=repo_path,
        run_setup=True,
    )


def open_workspace(
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
            path = display_path(repo_path, cwd=current)
            print(f"error: branch '{branch}' is not checked out at {path}", file=sys.stderr)
            raise SystemExit(1)
        session_name = branch_session_name(proj_dir, branch)
        ensure_dependency_configs_for_branch(project_dir=proj_dir, branch=branch)
    if branch is not None:
        env = load_workspace_env(proj_dir, branch, workspace_dir=repo_path)
        commit_config_dir_changes(
            proj_dir, f"chore: update config for {branch}",
            add_paths=[project_config_dir(proj_dir) / branch], env=env,
        )
    create_configured_workspace_session(
        detached=detached,
        pane_count=pane_count,
        session_name=session_name,
        repo_path=repo_path,
        run_setup=False,
    )


def create_workspace(
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
    start_point = require_parent_start_point(proj_dir, parent)
    repo_path = branch_path(proj_dir, branch)
    mkdir(repo_path, parents=True, exist_ok=True)
    ensure_dependency_configs_for_branch(
        project_dir=proj_dir,
        branch=branch,
        parent_branch=start_point,
    )
    env = load_workspace_env(proj_dir, branch, workspace_dir=repo_path)
    ensure_worktree(
        new_branch=branch,
        worktrees_dir=None,
        branch=None,
        existing_ok=False,
        cwd=project_repo_dir(proj_dir),
        start_point=start_point,
        env=env,
    )
    commit_config_dir_changes(
        proj_dir, f"chore: add config for {branch}",
        add_paths=[project_config_dir(proj_dir) / branch], env=env,
    )
    queue_setup_and_focus_workspace_session(
        detached=detached,
        pane_count=pane_count,
        session_name=branch_session_name(proj_dir, branch),
        repo_path=repo_path,
    )


def checkout_workspace(
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
    start_point = require_parent_start_point(proj_dir, parent)
    repo_path = branch_path(proj_dir, branch)
    mkdir(repo_path, parents=True, exist_ok=True)
    ensure_dependency_configs_for_branch(
        project_dir=proj_dir,
        branch=branch,
        parent_branch=start_point,
    )
    env = load_workspace_env(proj_dir, branch, workspace_dir=repo_path)
    ensure_worktree(
        new_branch=None,
        worktrees_dir=None,
        branch=branch,
        existing_ok=True,
        cwd=project_repo_dir(proj_dir),
        env=env,
    )
    commit_config_dir_changes(
        proj_dir, f"chore: add config for {branch}",
        add_paths=[project_config_dir(proj_dir) / branch], env=env,
    )
    queue_setup_and_focus_workspace_session(
        detached=detached,
        pane_count=pane_count,
        session_name=branch_session_name(proj_dir, branch),
        repo_path=repo_path,
    )


def open_or_create_workspace(
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
    if is_main_workspace_branch(
        proj_dir,
        branch,
        repo_branch=git_helpers.current_branch(repo_dir),
    ):
        open_workspace(detached=detached, pane_count=pane_count, branch=None, cwd=current)
        return

    git_helpers.fetch(repo_dir)
    if has_expected_worktree(proj_dir, branch):
        if parent is not None:
            print(
                f"error: workspace for branch '{branch}' already exists; cannot use --parent",
                file=sys.stderr,
            )
            raise SystemExit(1)
        open_workspace(detached=detached, pane_count=pane_count, branch=branch, cwd=current)
        return
    if branch_exists(repo_dir, branch):
        if parent is not None:
            warn_parent_ignored_for_existing_branch(branch)
        checkout_workspace(
            detached=detached,
            pane_count=pane_count,
            parent=None,
            branch=branch,
            cwd=current,
        )
        return
    create_workspace(
        detached=detached,
        pane_count=pane_count,
        parent=parent,
        branch=branch,
        cwd=current,
    )


def run(args: OpenArgs) -> None:
    open_or_create_workspace(
        detached=args.detached,
        pane_count=args.pane_count,
        parent=args.parent,
        branch=args.branch,
    )
