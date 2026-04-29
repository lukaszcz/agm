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
    current_project_dir,
    is_main_checkout_branch,
    project_config_dir,
    project_repo_dir,
)
from agm.project.setup import load_worktree_env
from agm.project.worktree import remove_worktree
from agm.tmux.session import close_tmux_session


def _containing_git_root(path: Path, *, env: dict[str, str]) -> Path | None:
    if not path.exists():
        return None
    returncode, stdout, _stderr = run_capture(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        env=env,
    )
    if returncode != 0:
        return None
    return Path(stdout.strip())


def _has_staged_changes(repo_dir: Path, path: Path, *, env: dict[str, str]) -> bool:
    returncode, stdout, stderr = run_capture(
        ["git", "-C", str(repo_dir), "diff", "--cached", "--quiet", "--", str(path)],
        env=env,
    )
    if returncode not in {0, 1}:
        exit_with_output(returncode, stdout, stderr)
    return returncode == 1


def _remove_branch_config(*, proj_dir: Path, branch: str, env: dict[str, str]) -> None:
    config_dir = project_config_dir(proj_dir)
    branch_config_dir = config_dir / branch
    if not branch_config_dir.exists():
        return

    config_git_root = _containing_git_root(config_dir, env=env)
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

    require_success(
        ["git", "-C", str(config_git_root), "add", "-A", "--", str(relative_config_path)],
        env=env,
    )
    if not _has_staged_changes(config_git_root, relative_config_path, env=env):
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


def close_session(*, branch: str, cwd: Path | None = None) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = current_project_dir(current)
    repo_dir = project_repo_dir(proj_dir)
    env = load_worktree_env(proj_dir, None, checkout_dir=repo_dir)
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
    _remove_branch_config(proj_dir=proj_dir, branch=branch, env=env)
    session_name = branch_session_name(proj_dir, branch)
    close_tmux_session(session_name=session_name, cwd=repo_dir, env=env)


def run(args: CloseArgs) -> None:
    close_session(branch=args.branch)
