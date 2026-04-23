"""Dry-run state and output helpers."""

from __future__ import annotations

import shlex
from pathlib import Path

_ENABLED = False


def set_enabled(value: bool) -> None:
    """Enable or disable dry-run mode for the current CLI invocation."""

    global _ENABLED
    _ENABLED = value


def enabled() -> bool:
    """Return whether dry-run mode is enabled."""

    return _ENABLED


def format_command(cmd: list[str]) -> str:
    """Return *cmd* formatted as a shell-safe command string."""

    return " ".join(shlex.quote(part) for part in cmd)


def format_command_with_cwd(cmd: list[str], *, cwd: Path | None = None) -> str:
    """Return *cmd* formatted with an optional working-directory prefix."""

    if cwd is None:
        return format_command(cmd)
    return f"(cd {shlex.quote(str(cwd))} && {format_command(cmd)})"


def print_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Print a command that would be executed."""

    print(f"dry-run: {format_command_with_cwd(cmd, cwd=cwd)}")


def print_operation(name: str, detail: str) -> None:
    """Print a high-level AGM operation that would be executed."""

    print(f"dry-run: agm {name} {detail}")


def print_configuration(subject: str) -> None:
    """Print the heading for a resolved dry-run configuration block."""

    print(f"dry-run: {subject} configuration")


def print_detail(label: str, value: str) -> None:
    """Print one resolved dry-run configuration value."""

    print(f"dry-run:   {label}: {value}")


def print_labeled_command(label: str, cmd: list[str], *, cwd: Path | None = None) -> None:
    """Print a labeled command that would be executed."""

    print(f"dry-run: command [{label}]: {format_command_with_cwd(cmd, cwd=cwd)}")
