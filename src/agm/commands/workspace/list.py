"""agm workspace list — list AGM workspaces."""

from __future__ import annotations

from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.path import display_path
from agm.project.layout import (
    current_workspace,
    project_repo_dir,
    require_current_project_dir,
)


def _is_current(worktree_path: Path, current_dir: Path | None) -> bool:
    """Return whether *worktree_path* matches the current workspace directory."""
    if current_dir is None:
        return False
    return worktree_path.resolve(strict=False) == current_dir.resolve(strict=False)


def _branch_sort_key(wt: git_helpers.WorktreeInfo) -> str:
    return wt.branch if wt.branch is not None else ""


def list_workspaces(*, cwd: Path | None = None, verbose: bool = False) -> None:
    """Print all open workspaces, with the main repo first and '*' marking the current one.

    When *verbose* is False only branch names are printed; when True the workspace
    directory path is appended after the branch name.
    """
    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = require_current_project_dir(current)
    repo_dir = project_repo_dir(proj_dir)
    repo_branch = git_helpers.current_branch(repo_dir)

    worktrees = git_helpers.worktree_list(repo_dir)
    workspace = current_workspace(proj_dir, cwd=current)
    current_dir = workspace.workspace_dir if workspace is not None else None

    # Find the main workspace from the Git worktree list.
    main_worktree: git_helpers.WorktreeInfo | None = None
    for wt in worktrees:
        if wt.path.resolve(strict=False) == repo_dir.resolve(strict=False):
            main_worktree = wt
            break

    # Build output lines: main workspace first, then branch workspaces sorted alphabetically.
    branch_worktrees: list[git_helpers.WorktreeInfo] = sorted(
        [wt for wt in worktrees if wt is not main_worktree],
        key=_branch_sort_key,
    )

    marker = "*" if _is_current(repo_dir, current_dir) else " "
    main_branch = main_worktree.branch if main_worktree is not None else repo_branch
    if verbose:
        print(f"{marker} {main_branch}  {display_path(repo_dir, cwd=current)}")
    else:
        print(f"{marker} {main_branch}")

    for wt in branch_worktrees:
        marker = "*" if _is_current(wt.path, current_dir) else " "
        branch = wt.branch or "(detached)"
        if verbose:
            print(f"{marker} {branch}  {display_path(wt.path, cwd=current)}")
        else:
            print(f"{marker} {branch}")


def run(*, verbose: bool = False) -> None:
    list_workspaces(verbose=verbose)
