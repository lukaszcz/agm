"""agm fetch."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.utils.project import current_project_dir
from agm.utils.worktree import sync_remote_tracking_branches
from agm.vcs.git import fetch_prune_all, find_first_git_repo


def _fetch_repo(project_dir: Path, repo_path: Path) -> None:
    display_path = str(repo_path)
    prefix = f"{project_dir}/"
    if display_path.startswith(prefix):
        display_path = display_path[len(prefix):]
    print(f"Fetching {display_path}")
    fetch_prune_all(repo_path)
    sync_remote_tracking_branches(repo_path)


def run(args: object) -> None:
    del args
    project_dir = current_project_dir()
    repo_dir = project_dir / "repo"
    if not repo_dir.is_dir():
        print(f"error: repo does not exist in {project_dir}", file=sys.stderr)
        raise SystemExit(1)

    _fetch_repo(project_dir, repo_dir)
    deps_dir = project_dir / "deps"
    if not deps_dir.is_dir():
        return

    for dep_dir in sorted(path for path in deps_dir.iterdir() if path.is_dir()):
        repo_path = find_first_git_repo(dep_dir)
        _fetch_repo(project_dir, repo_path)
