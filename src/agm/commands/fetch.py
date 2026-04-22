"""agm fetch."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.core.fs import is_dir, iterdir
from agm.project.layout import current_project_dir, project_deps_dir, project_repo_dir
from agm.project.worktree import sync_remote_tracking_branches
from agm.vcs.git import fetch_prune_all, find_first_git_repo, is_git_repo


def _fetch_repo(project_dir: Path, repo_path: Path) -> None:
    try:
        relative_path = repo_path.relative_to(project_dir)
        display_path = "." if relative_path == Path(".") else str(relative_path)
    except ValueError:
        display_path = str(repo_path)
    print(f"Fetching {display_path}")
    fetch_prune_all(repo_path)
    sync_remote_tracking_branches(repo_path)


def run(args: object) -> None:
    del args
    project_dir = current_project_dir()
    repo_dir = project_repo_dir(project_dir)
    if not is_dir(repo_dir) or not is_git_repo(repo_dir):
        print(f"error: repo does not exist in {project_dir}", file=sys.stderr)
        raise SystemExit(1)

    _fetch_repo(project_dir, repo_dir)
    deps_dir = project_deps_dir(project_dir)
    if not is_dir(deps_dir):
        return

    for dep_dir in sorted(path for path in iterdir(deps_dir) if is_dir(path)):
        repo_path = find_first_git_repo(dep_dir)
        _fetch_repo(project_dir, repo_path)
