"""Setup and environment helpers for AGM checkouts."""

from __future__ import annotations

import os
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.env import source_env_files
from agm.core.process import require_success
from agm.project.layout import branch_session_name, current_project_dir, main_repo_dir


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
    repo_dir = main_repo_dir(project_dir)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    if checkout_dir.resolve(strict=False) != repo_dir.resolve(strict=False):
        branch = git_helpers.current_branch(checkout_dir, env=env)
        target_name = branch_session_name(project_dir, branch)
    else:
        target_name = branch_session_name(project_dir, repo_branch)
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
