"""agm pull."""

from __future__ import annotations

from pathlib import Path

import agm.commands.fetch as fetch_command
import agm.vcs.git as git_helpers
from agm.project.layout import require_current_project_dir


def _display_path(project_dir: Path, path: Path) -> str:
    try:
        relative_path = path.relative_to(project_dir)
        return "." if relative_path == Path(".") else str(relative_path)
    except ValueError:
        return str(path)


def _merge_worktree(project_dir: Path, worktree_path: Path) -> None:
    print(f"Merging {_display_path(project_dir, worktree_path)}", flush=True)
    git_helpers.merge(worktree_path)


def run(args: object) -> None:
    del args
    project_dir = require_current_project_dir()
    repos = fetch_command.project_git_repos(project_dir)
    fetch_command.fetch_project_repos(project_dir, repos)
    for repo_path in repos:
        for worktree in git_helpers.worktree_list(repo_path):
            _merge_worktree(project_dir, worktree.path)
