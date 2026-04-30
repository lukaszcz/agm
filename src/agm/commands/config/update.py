"""agm config update."""

from __future__ import annotations

import os

from agm.commands.args import ConfigUpdateArgs
from agm.project.dependency_env import update_all_project_dependency_configs
from agm.project.layout import current_project_dir


def run(args: ConfigUpdateArgs) -> None:
    del args
    update_all_project_dependency_configs(current_project_dir(), env=dict(os.environ))
