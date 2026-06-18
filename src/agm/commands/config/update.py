"""agm config update."""

from __future__ import annotations

import os

from agm.cli_support.args import ConfigUpdateArgs
from agm.project.config_git import commit_config_dir_changes
from agm.project.dependency_env import update_all_project_dependency_configs
from agm.project.layout import require_current_project_dir


def run(args: ConfigUpdateArgs) -> None:
    del args
    env = dict(os.environ)
    project_dir = require_current_project_dir()
    update_all_project_dependency_configs(project_dir, env=env)
    commit_config_dir_changes(project_dir, "chore: update config", env=env)
