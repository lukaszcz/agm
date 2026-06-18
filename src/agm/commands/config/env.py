"""agm config env."""

from __future__ import annotations

import os
import shlex

from agm.cli_support.args import ConfigEnvArgs
from agm.core.env import is_safe_shell_env_assignment_name
from agm.project.workspace_env import load_current_workspace_env


def shell_env_delta(
    *,
    before: dict[str, str],
    after: dict[str, str],
) -> list[str]:
    """Return shell statements that transform *before* into *after*."""

    statements: list[str] = []
    for name in sorted(before.keys() - after.keys()):
        if is_safe_shell_env_assignment_name(name):
            statements.append(f"unset {name}")
    for name in sorted(after):
        if not is_safe_shell_env_assignment_name(name) or before.get(name) == after[name]:
            continue
        statements.append(f"export {name}={shlex.quote(after[name])}")
    return statements


def run(args: ConfigEnvArgs) -> None:
    del args
    before = dict(os.environ)
    after = load_current_workspace_env(env=before)
    for statement in shell_env_delta(before=before, after=after):
        print(statement)
