"""Fetch main repo and dependencies."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.git import find_first_git_repo
from agm.shell import require_success


def _fetch_repo(project_dir: Path, repo_path: Path, *, env: dict[str, str] | None = None) -> None:
    display_path = str(repo_path)
    prefix = f"{project_dir}/"
    if display_path.startswith(prefix):
        display_path = display_path[len(prefix):]
    print(f"Fetching {display_path}")
    require_success(["git", "-C", str(repo_path), "fetch"], env=env)


def fetch_all(*, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Fetch the main repo and all dependency repos."""

    project_dir = Path.cwd() if cwd is None else cwd.resolve()
    repo_dir = project_dir / "repo"
    if not repo_dir.is_dir():
        print(f"error: repo does not exist in {project_dir}", file=sys.stderr)
        raise SystemExit(1)

    _fetch_repo(project_dir, repo_dir, env=env)
    deps_dir = project_dir / "deps"
    if not deps_dir.is_dir():
        return

    for dep_dir in sorted(path for path in deps_dir.iterdir() if path.is_dir()):
        repo_path = find_first_git_repo(dep_dir)
        _fetch_repo(project_dir, repo_path, env=env)
