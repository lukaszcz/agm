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


def print_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Print a command that would be executed."""

    if cwd is None:
        print(f"dry-run: {format_command(cmd)}")
        return
    print(f"dry-run: (cd {shlex.quote(str(cwd))} && {format_command(cmd)})")


def print_operation(name: str, detail: str) -> None:
    """Print a high-level AGM operation that would be executed."""

    print(f"dry-run: agm {name} {detail}")
