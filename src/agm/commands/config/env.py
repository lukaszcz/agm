"""agm config env."""

from __future__ import annotations

import os
import re
import shlex

from agm.commands.args import ConfigEnvArgs
from agm.project.setup import load_current_config_env

_SHELL_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _is_shell_identifier(name: str) -> bool:
    return _SHELL_IDENTIFIER_RE.fullmatch(name) is not None


def shell_env_delta(
    *,
    before: dict[str, str],
    after: dict[str, str],
) -> list[str]:
    """Return shell statements that transform *before* into *after*."""

    statements: list[str] = []
    for name in sorted(before.keys() - after.keys()):
        if _is_shell_identifier(name):
            statements.append(f"unset {name}")
    for name in sorted(after):
        if not _is_shell_identifier(name) or before.get(name) == after[name]:
            continue
        statements.append(f"export {name}={shlex.quote(after[name])}")
    return statements


def run(args: ConfigEnvArgs) -> None:
    del args
    before = dict(os.environ)
    after = load_current_config_env(env=before)
    for statement in shell_env_delta(before=before, after=after):
        print(statement)
