"""agm close."""

from __future__ import annotations

import agm.vcs.git as git_helpers
from agm.commands.args import CloseArgs
from agm.utils.project import branch_session_name, current_project_dir, main_repo_dir
from agm.utils.project_session import close_session
from agm.utils.worktree import load_worktree_env


def run(args: CloseArgs) -> None:
    proj_dir = current_project_dir()
    repo_dir = main_repo_dir(proj_dir)
    env = load_worktree_env(proj_dir, None, shell_cwd=repo_dir)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    session_name = branch_session_name(proj_dir, args.branch, repo_branch=repo_branch)
    close_session(branch=args.branch)
    print(f"Closed session {session_name}")
