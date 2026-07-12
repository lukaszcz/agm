"""agm sync fetch."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.core.fs import is_dir, iterdir
from agm.core.path import display_path
from agm.project.layout import (
    project_deps_dir,
    project_repo_dir,
    require_current_project_dir,
)
from agm.project.worktree import sync_remote_tracking_branches
from agm.vcs.git import fetch_prune_all, find_first_git_repo, is_git_repo, worktree_prune


def _fetch_repo(project_dir: Path, repo_path: Path) -> None:
    del project_dir
    print(f"Fetching {display_path(repo_path)}", flush=True)
    worktree_prune(repo_path)
    fetch_prune_all(repo_path)
    sync_remote_tracking_branches(repo_path)


def project_git_repos(project_dir: Path) -> list[Path]:
    """Return the main project repo followed by each dependency repo."""

    repo_dir = project_repo_dir(project_dir)
    if not is_dir(repo_dir) or not is_git_repo(repo_dir):
        print(f"error: repo does not exist in {display_path(project_dir)}", file=sys.stderr)
        raise SystemExit(1)

    repos = [repo_dir]
    deps_dir = project_deps_dir(project_dir)
    if not is_dir(deps_dir):
        return repos

    for dep_dir in sorted(path for path in iterdir(deps_dir) if is_dir(path)):
        repos.append(find_first_git_repo(dep_dir))
    return repos


def fetch_project_repos(project_dir: Path, repos: list[Path]) -> None:
    """Fetch every repo in *repos* relative to *project_dir*."""

    for repo_path in repos:
        _fetch_repo(project_dir, repo_path)


def run(args: object) -> None:
    del args
    project_dir = require_current_project_dir()
    fetch_project_repos(project_dir, project_git_repos(project_dir))
