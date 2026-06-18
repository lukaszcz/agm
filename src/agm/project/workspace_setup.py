"""Setup helpers for AGM workspaces."""

from __future__ import annotations

import os
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core import dry_run
from agm.core.process import require_success
from agm.project.layout import (
    branch_session_name,
    current_workspace,
    project_config_dir,
    project_repo_dir,
    require_current_project_dir,
)
from agm.project.workspace_env import load_workspace_env


def run_setup(*, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run all configured setup scripts for the current AGM workspace."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    project_dir = require_current_project_dir(current)
    repo_dir = project_repo_dir(project_dir)
    result = current_workspace(project_dir, cwd=cwd, env=env)
    if result is not None:
        workspace_dir = result.workspace_dir
        branch = result.branch
    else:
        workspace_dir = repo_dir if repo_dir.is_dir() else current
        branch = None
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    if branch is not None:
        target_name = branch_session_name(project_dir, branch)
    else:
        target_name = branch_session_name(project_dir, repo_branch)
    setup_env = load_workspace_env(project_dir, branch, workspace_dir=workspace_dir, env=env)
    config_dir = project_config_dir(project_dir)

    setup_paths = [
        config_dir / "setup.sh",
        workspace_dir / ".config" / "setup.sh",
        workspace_dir / ".setup.sh",
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
            setup_label = setup_path.relative_to(workspace_dir)
        except ValueError:
            try:
                setup_label = setup_path.relative_to(project_dir)
            except ValueError:
                setup_label = setup_path
        print(f"Running {setup_label}...")
        if dry_run.enabled():
            dry_run.print_operation("run-setup", str(setup_path))
        require_success(["bash", str(setup_path)], cwd=workspace_dir, env=setup_env)
    print(f"Setup complete for {target_name}.")
