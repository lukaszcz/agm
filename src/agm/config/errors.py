"""CLI-facing configuration error helpers."""

from __future__ import annotations

from typing import NoReturn

from agm.config.general import ConfigCommandNotFound
from agm.parser import exit_with_usage_error


def exit_config_command_not_found(error: ConfigCommandNotFound) -> NoReturn:
    exit_with_usage_error([error.section_name], f"error: {error}")
