"""Environment helpers for AGM workspaces."""

from __future__ import annotations

import os
from pathlib import Path

from agm.core.env import load_config_dotenv_files, source_env_file
from agm.project.dependency_env import load_dependency_toml_env
from agm.project.layout import (
    current_workspace,
    project_config_dir,
    project_repo_dir,
    require_current_project_dir,
)


def load_config_env(
    project_dir: Path,
    branch: str | None,
    *,
    workspace_dir: Path,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return *env* refreshed from project and workspace config files."""

    resolved_env = dict(os.environ if env is None else env)
    resolved_env["PROJ_DIR"] = str(project_dir)
    resolved_env["REPO_DIR"] = str(workspace_dir)
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
        resolved_env = source_env_file(env_dir / "env.sh", resolved_env, cwd=workspace_dir)
    return resolved_env


def load_workspace_env(
    project_dir: Path,
    branch: str | None,
    *,
    workspace_dir: Path,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the sourced environment for an AGM workspace."""

    return load_config_env(project_dir, branch, workspace_dir=workspace_dir, env=env)


def load_current_workspace_env(
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the config-refreshed environment for the current AGM workspace."""

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
    return load_config_env(project_dir, branch, workspace_dir=workspace_dir, env=env)
