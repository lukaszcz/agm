"""Git worktree management."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agm import git as git_helpers
from agm.project import copy_config, default_worktrees_dir, detect_project_dir
from agm.shell import require_success


_MKW_USAGE = "usage: mkwt.sh [-b branch-name] [-d dir] [branch-name]"


def branch_sync(*, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Sync remote tracking branches."""

    repo_dir = git_helpers.git_setup(cwd)
    git_helpers.fetch_prune_origin(repo_dir, env=env)
    for remote_branch in git_helpers.remote_unmerged_branches(repo_dir, base_ref="origin/main", env=env):
        if remote_branch == "origin/HEAD":
            continue
        local_branch = remote_branch.removeprefix("origin/")
        if not git_helpers.local_branch_exists(repo_dir, local_branch, env=env):
            git_helpers.create_tracking_branch(repo_dir, local_branch, remote_branch, env=env)


def worktree_checkout(
    *,
    new_branch: str | None,
    worktrees_dir: str | None,
    branch: str | None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Create or check out a worktree."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    create_branch = new_branch is not None
    branch_name = new_branch if create_branch else branch
    if branch_name is None:
        print(_MKW_USAGE)
        raise SystemExit(1)

    repo_dir = git_helpers.git_setup(current)
    git_helpers.fetch(repo_dir, env=env)

    worktrees_path = (
        default_worktrees_dir(detect_project_dir(current))
        if worktrees_dir is None
        else Path(worktrees_dir)
    )
    if not worktrees_path.is_absolute():
        worktrees_path = current / worktrees_path

    dirname = worktrees_path / branch_name
    git_helpers.worktree_add(
        repo_dir,
        dirname,
        branch_name,
        create=create_branch,
        env=env,
    )
    copy_config(target=dirname, cwd=current)

    project_dir = detect_project_dir(dirname.resolve())
    setup_paths = [
        project_dir / "config" / "setup.sh",
        dirname / ".config" / "setup.sh",
        dirname / ".setup.sh",
    ]
    for setup_path in setup_paths:
        if setup_path.is_file() and os.access(setup_path, os.X_OK):
            require_success([str(setup_path)], cwd=dirname, env=env)


def worktree_remove(
    *,
    force: bool,
    branch: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Remove a worktree and delete its branch."""

    repo_dir = git_helpers.git_setup(cwd)
    worktree_path: Path | None = None
    worktrees = git_helpers.worktree_list(repo_dir, env=env)
    for worktree in worktrees:
        if worktree.branch == branch:
            worktree_path = worktree.path
            break
    if worktree_path is None:
        print(f"Error: No worktree found for branch '{branch}'")
        print("Available worktrees:")
        require_success(["git", "-C", str(repo_dir), "worktree", "list"], env=env)
        raise SystemExit(1)

    git_helpers.worktree_remove(repo_dir, worktree_path, force=force, env=env)
    print(f"Removed worktree for branch '{branch}': {worktree_path}")
    git_helpers.branch_delete(repo_dir, branch, env=env)
