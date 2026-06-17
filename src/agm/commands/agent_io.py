"""Shared helpers for prompt-driven agent commands."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import NoReturn

from agm.config.context import current_config_context
from agm.config.general import (
    ConfigCommandNotFound,
    load_loop_config,
    loop_config_from_merged,
)
from agm.core.toml import TomlDict
from agm.parser import exit_with_usage_error

StreamCallback = Callable[[str], None]


def write_stdout(chunk: str) -> None:
    if not chunk:
        return
    sys.stdout.write(chunk)
    sys.stdout.flush()


def write_stderr(chunk: str) -> None:
    if not chunk:
        return
    sys.stderr.write(chunk)
    sys.stderr.flush()


def exit_config_command_not_found(error: ConfigCommandNotFound) -> NoReturn:
    exit_with_usage_error([error.section_name], f"error: {error}")


def default_agent_runner(*, merged: TomlDict | None = None) -> str:
    """Resolve the shared default agent runner from the ``[loop]`` config.

    When *merged* is supplied, the ``[loop]`` section is derived from it,
    avoiding a redundant config-file load for callers (e.g. ``agm exec``) that
    already hold a merged config.
    """
    if merged is not None:
        loop_config = loop_config_from_merged(merged)
    else:
        context = current_config_context()
        loop_config = load_loop_config(
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
        )
    return loop_config.runner if loop_config.runner is not None else "claude -p"
