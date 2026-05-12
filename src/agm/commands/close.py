"""agm close."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.commands.args import CloseArgs
from agm.core import fs
from agm.core.process import exit_with_output, require_success, run_capture
from agm.project.layout import (
    branch_session_name,
    is_main_checkout_branch,
    project_config_dir,
    project_repo_dir,
    require_current_project_dir,
)
from agm.project.setup import load_worktree_env
from agm.project.worktree import remove_worktree
from agm.tmux.session import close_tmux_session


def _remove_branch_config(*, proj_dir: Path, branch: str, env: dict[str, str]) -> None:
    config_dir = project_config_dir(proj_dir)
    branch_config_dir = config_dir / branch
    if not branch_config_dir.exists():
        return

    config_git_root = git_helpers.containing_root(config_dir, env=env)
    relative_config_path: Path | None = None
    if config_git_root is not None:
        relative_config_path = branch_config_dir.resolve(strict=False).relative_to(
            config_git_root.resolve()
        )

    if branch_config_dir.is_dir():
        fs.rmtree(branch_config_dir)
    else:
        fs.unlink(branch_config_dir)

    if config_git_root is None or relative_config_path is None:
        return

    returncode, _stdout, stderr = run_capture(
        ["git", "-C", str(config_git_root), "add", "-A", "--", str(relative_config_path)],
        env=env,
    )
    if returncode != 0:
        # git add fails with "pathspec did not match any files" when the
        # directory was never tracked – that is harmless.
        if "did not match any files" not in stderr:
            exit_with_output(returncode, stderr=stderr)
        return
    if not git_helpers.has_staged_changes(config_git_root, [relative_config_path], env=env):
        return
    require_success(
        [
            "git",
            "-C",
            str(config_git_root),
            "commit",
            "-m",
            f"chore: remove config for {branch}",
        ],
        env=env,
    )


def close_session(
    *, branch: str, force: bool = False, force_delete: bool = False, cwd: Path | None = None
) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = require_current_project_dir(current)
    repo_dir = project_repo_dir(proj_dir)
    repo_branch = git_helpers.current_branch(repo_dir)
    if is_main_checkout_branch(proj_dir, branch, repo_branch=repo_branch):
        print(
            (
                f"error: '{branch}' resolves to the main repo checkout at "
                f"{repo_dir} and cannot be removed"
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    # --force implies force_delete as well (git branch -D semantics).
    effective_force_delete = force or force_delete

    # Pre-check: verify the branch can be deleted before removing the worktree.
    # Uses default environment; project-specific env is not needed for git checks.
    if not git_helpers.branch_can_delete(repo_dir, branch, force=effective_force_delete):
        if not git_helpers.local_branch_exists(repo_dir, branch):
            print(f"error: branch '{branch}' does not exist", file=sys.stderr)
        else:
            print(
                f"error: branch '{branch}' is not fully merged. Use -D to force delete.",
                file=sys.stderr,
            )
        raise SystemExit(1)

    remove_worktree(
        repo_dir=repo_dir, force=force, branch=branch, force_delete=effective_force_delete
    )
    env = load_worktree_env(proj_dir, None, checkout_dir=repo_dir)
    _remove_branch_config(proj_dir=proj_dir, branch=branch, env=env)
    session_name = branch_session_name(proj_dir, branch)
    close_tmux_session(session_name=session_name, cwd=repo_dir, env=env)


def run(args: CloseArgs) -> None:
    close_session(branch=args.branch, force=args.force, force_delete=args.force_delete)
