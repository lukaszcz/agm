"""AGM worktree orchestration built on top of git helpers."""

from __future__ import annotations

from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.process import require_success
from agm.project.dependency_env import ensure_dependency_configs_for_branch
from agm.project.layout import (
    branch_worktree_path,
    copy_config,
    current_project_dir,
    default_worktrees_dir,
    exit_if_main_checkout_branch,
    expected_branch_worktree_path,
    project_repo_dir,
)


def sync_remote_tracking_branches(
    repo_dir: Path, *, env: dict[str, str] | None = None
) -> None:
    """Create local tracking branches not merged into origin's default branch."""

    default_branch_ref = git_helpers.default_remote_branch_ref(repo_dir, env=env)

    for remote_branch in git_helpers.remote_unmerged_branches(
        repo_dir, base_ref=default_branch_ref, env=env
    ):
        if remote_branch == "origin/HEAD":
            continue
        local_branch = remote_branch.removeprefix("origin/")
        if not git_helpers.local_branch_exists(repo_dir, local_branch, env=env):
            git_helpers.create_tracking_branch(repo_dir, local_branch, remote_branch, env=env)


def branch_sync(*, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Create local tracking branches not merged into origin's default branch."""

    repo_dir = git_helpers.checkout_root(cwd)
    git_helpers.fetch_prune_origin(repo_dir, env=env)
    sync_remote_tracking_branches(repo_dir, env=env)


def has_expected_worktree(
    project_dir: Path, branch: str, *, env: dict[str, str] | None = None
) -> bool:
    """Return whether *branch* is checked out at the expected project path."""

    repo_dir = project_repo_dir(project_dir)
    expected_path = expected_branch_worktree_path(project_dir, branch)
    for worktree in git_helpers.worktree_list(repo_dir, env=env):
        if worktree.branch == branch and worktree.path.resolve(strict=False) == expected_path:
            return True
    return False


def branch_exists(
    repo_dir: Path, branch: str, *, env: dict[str, str] | None = None
) -> bool:
    """Return whether *branch* exists locally or on origin."""

    return git_helpers.local_branch_exists(
        repo_dir, branch, env=env
    ) or git_helpers.remote_branch_exists(repo_dir, branch, env=env)


def resolve_parent_checkout_dir(
    project_dir: Path, parent: str | None, *, env: dict[str, str]
) -> Path:
    """Return the checkout directory to use as the parent for a new worktree."""

    repo_dir = project_repo_dir(project_dir)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    resolved_parent = parent or repo_branch
    if resolved_parent == repo_branch:
        return repo_dir
    return branch_worktree_path(project_dir, resolved_parent, repo_branch=repo_branch)


def ensure_worktree(
    *,
    new_branch: str | None,
    worktrees_dir: str | None,
    branch: str | None,
    existing_ok: bool = False,
    reuse_existing_branch: bool = False,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """Create a worktree if needed and return its path."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    create_branch = new_branch is not None
    branch_name = new_branch if create_branch else branch
    if branch_name is None:
        raise SystemExit(1)

    project_dir = current_project_dir(current)
    repo_dir = git_helpers.checkout_root(current)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    exit_if_main_checkout_branch(project_dir, branch_name, repo_branch=repo_branch)
    worktrees_path = (
        default_worktrees_dir(project_dir) if worktrees_dir is None else Path(worktrees_dir)
    )
    if not worktrees_path.is_absolute():
        worktrees_path = current / worktrees_path
    dirname = worktrees_path / branch_name
    resolved_dirname = dirname.resolve(strict=False)

    git_helpers.fetch(repo_dir, env=env)
    if create_branch and reuse_existing_branch:
        branch_exists = git_helpers.local_branch_exists(
            repo_dir, branch_name, env=env
        ) or git_helpers.remote_branch_exists(repo_dir, branch_name, env=env)
        if branch_exists:
            create_branch = False
            existing_ok = True
    existing_worktrees = git_helpers.worktree_list(repo_dir, env=env)
    for worktree in existing_worktrees:
        if (
            worktree.branch == branch_name
            and worktree.path.resolve(strict=False) == resolved_dirname
        ):
            if not existing_ok:
                print(f"error: worktree already exists for branch '{branch_name}'")
                raise SystemExit(1)
            return dirname

    git_helpers.worktree_add(
        repo_dir,
        dirname,
        branch_name,
        create=create_branch,
        env=env,
    )
    ensure_dependency_configs_for_branch(project_dir=project_dir, branch=branch_name)
    copy_config(project_dir=project_dir, target=dirname, branch=branch_name, cwd=current)
    return dirname


def remove_worktree(
    *,
    repo_dir: Path,
    force: bool,
    branch: str,
    force_delete: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """Remove a worktree from *repo_dir* and delete its branch."""

    project_dir = current_project_dir(repo_dir)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    exit_if_main_checkout_branch(project_dir, branch, repo_branch=repo_branch)

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
    git_helpers.branch_delete(repo_dir, branch, force=force_delete, env=env)
