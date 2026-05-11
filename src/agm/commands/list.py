"""agm list — list all open worktrees."""

from __future__ import annotations

from pathlib import Path

import agm.vcs.git as git_helpers
from agm.project.layout import (
    current_checkout,
    project_repo_dir,
    require_current_project_dir,
)


def _is_current(worktree_path: Path, current_dir: Path | None) -> bool:
    """Return whether *worktree_path* matches the current checkout directory."""
    if current_dir is None:
        return False
    return worktree_path.resolve(strict=False) == current_dir.resolve(strict=False)


def _branch_sort_key(wt: git_helpers.WorktreeInfo) -> str:
    return wt.branch if wt.branch is not None else ""


def list_worktrees(*, cwd: Path | None = None) -> None:
    """Print all open worktrees, with the main repo first and '*' marking the current one."""
    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = require_current_project_dir(current)
    repo_dir = project_repo_dir(proj_dir)
    repo_branch = git_helpers.current_branch(repo_dir)

    worktrees = git_helpers.worktree_list(repo_dir)
    checkout = current_checkout(proj_dir, cwd=current)
    current_dir = checkout.checkout_dir if checkout is not None else None

    # Find main repo worktree info from the git worktree list
    main_worktree: git_helpers.WorktreeInfo | None = None
    for wt in worktrees:
        if wt.path.resolve(strict=False) == repo_dir.resolve(strict=False):
            main_worktree = wt
            break

    # Build output lines: main repo first, then branch worktrees sorted alphabetically
    branch_worktrees: list[git_helpers.WorktreeInfo] = sorted(
        [wt for wt in worktrees if wt is not main_worktree],
        key=_branch_sort_key,
    )

    marker = "*" if _is_current(repo_dir, current_dir) else " "
    main_branch = main_worktree.branch if main_worktree is not None else repo_branch
    print(f"{marker} {main_branch}  {repo_dir}")

    for wt in branch_worktrees:
        marker = "*" if _is_current(wt.path, current_dir) else " "
        branch = wt.branch or "(detached)"
        print(f"{marker} {branch}  {wt.path}")


def run() -> None:
    list_worktrees()
