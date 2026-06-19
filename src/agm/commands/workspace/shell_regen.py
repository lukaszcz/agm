"""agm workspace shell-regen."""

from __future__ import annotations

from pathlib import Path

from agm.project.workspace_shell import regenerate_workspace_shell


def run(shell_dir: str) -> None:
    regenerate_workspace_shell(Path(shell_dir))
