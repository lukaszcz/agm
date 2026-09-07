"""Dependency checkout helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers


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

    return git_helpers.find_first_git_repo(dep_dir, main_only=True)


def find_main_dep_repo(dep_dir: Path) -> Path | None:
    """Return the main dependency checkout, including a repository at *dep_dir*."""

    return git_helpers.first_git_repo(dep_dir, main_only=True, include_parent=True)
