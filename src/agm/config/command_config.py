"""Generic command-config loader that wraps the config-context/try/except boilerplate."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from agm.config.context import current_config_context
from agm.config.errors import exit_config_command_not_found
from agm.config.general import ConfigCommandNotFound

_T = TypeVar("_T")


def load_command_config(
    loader: Callable[..., _T], command_name: str | None, *, require_command: bool
) -> _T:
    """Load a per-command config via *loader*, exiting cleanly on a missing named command."""
    context = current_config_context()
    try:
        return loader(
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
            command_name=command_name,
            require_command=require_command,
        )
    except ConfigCommandNotFound as error:
        exit_config_command_not_found(error)
