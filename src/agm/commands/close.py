"""agm close."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.commands.args import CloseArgs
from agm.project.layout import (
    branch_session_name,
    current_project_dir,
    is_main_checkout_branch,
    project_repo_dir,
)
from agm.project.setup import load_worktree_env
from agm.project.worktree import remove_worktree
from agm.tmux.session import close_tmux_session


def close_session(*, branch: str, cwd: Path | None = None) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = current_project_dir(current)
    repo_dir = project_repo_dir(proj_dir)
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
    close_tmux_session(session_name=session_name, cwd=repo_dir, env=env)


def run(args: CloseArgs) -> None:
    close_session(branch=args.branch)
