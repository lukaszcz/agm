"""agm workspace close."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.cli_support.args import CloseArgs
from agm.core import fs
from agm.core.path import display_path
from agm.project.config_git import commit_config_dir_changes
from agm.project.layout import (
    branch_session_name,
    is_main_workspace_branch,
    project_config_dir,
    project_repo_dir,
    require_current_project_dir,
)
from agm.project.workspace_env import load_workspace_env
from agm.project.workspace_shell import remove_workspace_shell
from agm.project.worktree import remove_worktree
from agm.tmux.session import close_tmux_session


def _remove_workspace_config(*, proj_dir: Path, branch: str, env: dict[str, str]) -> None:
    config_dir = project_config_dir(proj_dir)
    workspace_config_dir = config_dir / branch
    if not workspace_config_dir.exists():
        return

    if workspace_config_dir.is_dir():
        fs.rmtree(workspace_config_dir)
    else:
        fs.unlink(workspace_config_dir)

    commit_config_dir_changes(
        proj_dir,
        f"chore: remove config for {branch}",
        add_paths=[workspace_config_dir],
        env=env,
    )


def close_workspace(
    *,
    branch: str,
    force: bool = False,
    force_delete: bool = False,
    keep_branch: bool = False,
    keep_workspace: bool = False,
    cwd: Path | None = None,
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = require_current_project_dir(current)
    repo_dir = project_repo_dir(proj_dir)
    repo_branch = git_helpers.current_branch(repo_dir)
    if is_main_workspace_branch(proj_dir, branch, repo_branch=repo_branch):
        print(
            (
                f"error: '{branch}' resolves to the main workspace at "
                f"{display_path(repo_dir, cwd=current)} and cannot be removed"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    # --force implies force_delete as well (git branch -D semantics).
    effective_force_delete = force or force_delete
    effective_keep_branch = keep_branch or keep_workspace

    if not keep_workspace:
        # Pre-check: verify the branch can be deleted before removing the Git worktree.
        # Uses default environment; project-specific env is not needed for git checks.
        if not effective_keep_branch and not git_helpers.branch_can_delete(
            repo_dir, branch, force=effective_force_delete
        ):
            if not git_helpers.local_branch_exists(repo_dir, branch):
                print(f"error: branch '{branch}' does not exist", file=sys.stderr)
            else:
                print(
                    f"error: branch '{branch}' is not fully merged. Use -D to force delete.",
                    file=sys.stderr,
                )
            raise SystemExit(1)

        remove_worktree(
            repo_dir=repo_dir,
            force=force,
            branch=branch,
            force_delete=effective_force_delete,
            delete_branch=not effective_keep_branch,
        )
    env = load_workspace_env(proj_dir, None, workspace_dir=repo_dir)
    if not keep_workspace:
        _remove_workspace_config(proj_dir=proj_dir, branch=branch, env=env)
    session_name = branch_session_name(proj_dir, branch)
    close_tmux_session(session_name=session_name, cwd=repo_dir, env=env)
    remove_workspace_shell(session_name)


def run(args: CloseArgs) -> None:
    close_workspace(
        branch=args.branch,
        force=args.force,
        force_delete=args.force_delete,
        keep_branch=args.keep_branch,
        keep_workspace=args.keep_workspace,
    )
