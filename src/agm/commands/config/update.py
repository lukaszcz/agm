"""agm config update."""

from __future__ import annotations

import os
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.commands.args import ConfigUpdateArgs
from agm.core.fs import rglob
from agm.core.process import require_success
from agm.project.dependency_env import update_all_project_dependency_configs
from agm.project.layout import project_config_dir, require_current_project_dir


def _commit_generated_configs(project_dir: Path, *, env: dict[str, str]) -> None:
    config_dir = project_config_dir(project_dir)
    config_git_root = git_helpers.exact_repo_root(config_dir, env=env)
    if config_git_root is None:
        return

    config_files = sorted(path for path in rglob(config_dir, "config.toml") if path.is_file())
    if not config_files:
        return
    relative_config_files = [
        path.resolve().relative_to(config_git_root.resolve()) for path in config_files
    ]
    require_success(
        ["git", "-C", str(config_git_root), "add", "--", *map(str, relative_config_files)],
        env=env,
    )
    if not git_helpers.has_staged_changes(config_git_root, relative_config_files, env=env):
        return
    require_success(
        ["git", "-C", str(config_git_root), "commit", "-m", "chore: update config"],
        env=env,
    )


def run(args: ConfigUpdateArgs) -> None:
    del args
    env = dict(os.environ)
    project_dir = require_current_project_dir()
    update_all_project_dependency_configs(project_dir, env=env)
    _commit_generated_configs(project_dir, env=env)
