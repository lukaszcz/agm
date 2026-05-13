"""Setup and environment helpers for AGM checkouts."""

from __future__ import annotations

import os
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core import dry_run
from agm.core.env import load_config_dotenv_files, source_env_file
from agm.core.process import require_success
from agm.project.dependency_env import load_dependency_toml_env
from agm.project.layout import (
    branch_session_name,
    current_checkout,
    project_config_dir,
    project_repo_dir,
    require_current_project_dir,
)


def load_config_env(
    project_dir: Path,
    branch: str | None,
    *,
    checkout_dir: Path,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return *env* refreshed from project and branch config files."""

    resolved_env = dict(os.environ if env is None else env)
    resolved_env["PROJ_DIR"] = str(project_dir)
    resolved_env["REPO_DIR"] = str(checkout_dir)
    config_dir = project_config_dir(project_dir)
    config_dirs = [config_dir]
    if branch is not None:
        config_dirs.append(config_dir / branch)

    resolved_env = load_dependency_toml_env(
        project_dir=project_dir,
        config_files=[env_dir / "config.toml" for env_dir in config_dirs],
        env=resolved_env,
    )
    for env_dir in config_dirs:
        resolved_env = load_config_dotenv_files([env_dir], resolved_env)
        resolved_env = source_env_file(env_dir / "env.sh", resolved_env, cwd=checkout_dir)
    return resolved_env


def load_worktree_env(
    project_dir: Path,
    branch: str | None,
    *,
    checkout_dir: Path,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the sourced environment for a repo or worktree checkout."""

    return load_config_env(project_dir, branch, checkout_dir=checkout_dir, env=env)


def load_current_config_env(
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the config-refreshed environment for the current checkout."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    project_dir = require_current_project_dir(current)
    repo_dir = project_repo_dir(project_dir)
    result = current_checkout(project_dir, cwd=cwd, env=env)
    if result is not None:
        checkout_dir = result.checkout_dir
        branch = result.branch
    else:
        checkout_dir = repo_dir if repo_dir.is_dir() else current
        branch = None
    return load_config_env(project_dir, branch, checkout_dir=checkout_dir, env=env)


def run_setup(*, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run all configured setup scripts for the current checkout."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    project_dir = require_current_project_dir(current)
    repo_dir = project_repo_dir(project_dir)
    result = current_checkout(project_dir, cwd=cwd, env=env)
    if result is not None:
        checkout_dir = result.checkout_dir
        branch = result.branch
    else:
        checkout_dir = repo_dir if repo_dir.is_dir() else current
        branch = None
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    if branch is not None:
        target_name = branch_session_name(project_dir, branch)
    else:
        target_name = branch_session_name(project_dir, repo_branch)
    setup_env = load_worktree_env(project_dir, branch, checkout_dir=checkout_dir, env=env)
    config_dir = project_config_dir(project_dir)

    setup_paths = [
        config_dir / "setup.sh",
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
        if dry_run.enabled():
            dry_run.print_operation("run-setup", str(setup_path))
        require_success(["bash", str(setup_path)], cwd=checkout_dir, env=setup_env)
    print(f"Setup complete for {target_name}.")
