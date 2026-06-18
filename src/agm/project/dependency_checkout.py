"""Dependency checkout helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.fs import is_dir, rglob


def derive_dep_name(repo_url: str) -> str:
    """Derive a dependency name from *repo_url*."""

    try:
        return git_helpers.repo_name_from_url(repo_url)
    except ValueError:
        print(
            f"error: could not derive dependency name from repo url: {repo_url}",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main_dep_repo(dep_dir: Path) -> Path:
    """Return the main checked-out dependency repo under *dep_dir*."""

    for path in sorted(candidate for candidate in rglob(dep_dir, "*") if is_dir(candidate)):
        if is_dir(path / ".git") and git_helpers.is_git_repo(path):
            return path
    print(
        f"error: {dep_dir} must contain a main checked out branch",
        file=sys.stderr,
    )
    raise SystemExit(1)
