"""Shared worktree operations used by multiple commands."""

from __future__ import annotations

import os
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.utils.project import copy_config, current_project_dir, default_worktrees_dir, main_repo_dir
from agm.utils.shell import require_success, source_env_files


def branch_sync(*, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Sync remote tracking branches."""

    repo_dir = git_helpers.git_setup(cwd)
    git_helpers.fetch_prune_origin(repo_dir, env=env)
    for remote_branch in git_helpers.remote_unmerged_branches(
        repo_dir, base_ref="origin/main", env=env
    ):
        if remote_branch == "origin/HEAD":
            continue
        local_branch = remote_branch.removeprefix("origin/")
        if not git_helpers.local_branch_exists(repo_dir, local_branch, env=env):
            git_helpers.create_tracking_branch(repo_dir, local_branch, remote_branch, env=env)


def load_worktree_env(
    project_dir: Path,
    branch: str | None,
    *,
    shell_cwd: Path,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the sourced environment for a repo or worktree checkout."""

    resolved_env = dict(os.environ if env is None else env)
    resolved_env["PROJ_DIR"] = str(project_dir)
    env_files = [project_dir / "config" / "env.sh"]
    if branch is not None:
        env_files.append(project_dir / "config" / branch / "env.sh")
    return source_env_files(env_files, resolved_env, cwd=shell_cwd)


def run_setup(*, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run all configured setup scripts for the current checkout."""

    checkout_dir = git_helpers.git_setup(cwd)
    project_dir = current_project_dir(checkout_dir)
    branch: str | None = None
    if checkout_dir.resolve(strict=False) != main_repo_dir(project_dir).resolve(strict=False):
        branch = git_helpers.current_branch(checkout_dir, env=env)
    target_name = (
        f"{project_dir.name}/{branch}" if branch is not None else f"{project_dir.name}/repo"
    )
    setup_env = load_worktree_env(project_dir, branch, shell_cwd=checkout_dir, env=env)

    setup_paths = [
        project_dir / "config" / "setup.sh",
        checkout_dir / ".config" / "setup.sh",
        checkout_dir / ".setup.sh",
    ]
    runnable_paths = [
        setup_path
        for setup_path in setup_paths
        if setup_path.is_file() and os.access(setup_path, os.X_OK)
    ]
    if not runnable_paths:
        print(f"No setup scripts found for {target_name}.")
        return

    print(f"Running setup for {target_name}...")
    for setup_path in runnable_paths:
        try:
            setup_label = setup_path.relative_to(checkout_dir)
        except ValueError:
            try:
                setup_label = setup_path.relative_to(project_dir)
            except ValueError:
                setup_label = setup_path
        print(f"Running {setup_label}...")
        require_success(["bash", str(setup_path)], cwd=checkout_dir, env=setup_env)
    print(f"Setup complete for {target_name}.")


def ensure_worktree(
    *,
    new_branch: str | None,
    worktrees_dir: str | None,
    branch: str | None,
    existing_ok: bool = False,
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
    worktrees_path = (
        default_worktrees_dir(project_dir) if worktrees_dir is None else Path(worktrees_dir)
    )
    if not worktrees_path.is_absolute():
        worktrees_path = current / worktrees_path
    dirname = worktrees_path / branch_name
    resolved_dirname = dirname.resolve(strict=False)

    repo_dir = git_helpers.git_setup(current)
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

    git_helpers.fetch(repo_dir, env=env)
    git_helpers.worktree_add(
        repo_dir,
        dirname,
        branch_name,
        create=create_branch,
        env=env,
    )
    copy_config(project_dir=project_dir, target=dirname, cwd=current)
    return dirname


def remove_worktree(
    *,
    force: bool,
    branch: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Remove a worktree from the current repo and delete its branch."""

    remove_worktree_from_repo(
        repo_dir=git_helpers.git_setup(cwd),
        force=force,
        branch=branch,
        env=env,
    )


def remove_worktree_from_repo(
    *,
    repo_dir: Path,
    force: bool,
    branch: str,
    env: dict[str, str] | None = None,
) -> None:
    """Remove a worktree from *repo_dir* and delete its branch."""

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
