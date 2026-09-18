"""agm workspace list — list AGM workspaces."""

from __future__ import annotations

from pathlib import Path

from agm.core.path import display_path
from agm.project.layout import (
    Workspace,
    current_workspace,
    project_workspaces,
    require_current_project_dir,
)

_DETACHED = "(detached)"


def _label(workspace: Workspace) -> str:
    """Name a workspace: main by its branch, branch workspaces by name plus a differing HEAD."""
    if workspace.main:
        return workspace.branch or _DETACHED
    if workspace.branch is None:
        return f"{workspace.name} {_DETACHED}"
    if workspace.branch != workspace.name:
        return f"{workspace.name} (on {workspace.branch})"
    return workspace.name


def list_workspaces(*, cwd: Path | None = None, verbose: bool = False) -> None:
    """Print the main workspace, then branch workspaces, marking the current one with '*'.

    When *verbose* is True the workspace directory is appended after each label.
    """
    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = require_current_project_dir(current)
    workspace = current_workspace(proj_dir, cwd=current)
    current_dir = workspace.workspace_dir.resolve() if workspace is not None else None

    for ws in project_workspaces(proj_dir):
        marker = "*" if ws.path.resolve() == current_dir else " "
        line = f"{marker} {_label(ws)}"
        if verbose:
            line += f"  {display_path(ws.path, cwd=current)}"
        print(line)


def run(*, verbose: bool = False) -> None:
    list_workspaces(verbose=verbose)
