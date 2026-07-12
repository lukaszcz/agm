"""agm sync pull."""

from __future__ import annotations

from pathlib import Path

import agm.commands.sync.fetch as fetch_command
import agm.vcs.git as git_helpers
from agm.core.path import display_path
from agm.project.layout import require_current_project_dir


def _merge_worktree(project_dir: Path, worktree_path: Path) -> None:
    del project_dir
    print(f"Merging {display_path(worktree_path)}", flush=True)
    git_helpers.merge(worktree_path)


def run(args: object) -> None:
    del args
    project_dir = require_current_project_dir()
    repos = fetch_command.project_git_repos(project_dir)
    fetch_command.fetch_project_repos(project_dir, repos)
    for repo_path in repos:
        for worktree in git_helpers.worktree_list(repo_path):
            _merge_worktree(project_dir, worktree.path)
