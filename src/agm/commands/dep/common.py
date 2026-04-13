"""Shared helpers for dependency commands."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.utils.shell import run_capture


def derive_dep_name(repo_url: str) -> str:
    """Derive a dependency name from *repo_url*."""

    trimmed = repo_url.rstrip("/")
    dep = Path(trimmed).name.removesuffix(".git")
    if dep in {"", ".", "/"}:
        print(
            f"error: could not derive dependency name from repo url: {repo_url}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return dep


def default_branch_from_remote(repo_url: str, *, env: dict[str, str] | None = None) -> str:
    """Return the remote default branch."""

    returncode, output, _ = run_capture(
        ["git", "ls-remote", "--symref", repo_url, "HEAD"],
        env=env,
    )
    if returncode != 0:
        print(f"error: could not determine default branch for {repo_url}", file=sys.stderr)
        raise SystemExit(1)
    for line in output.splitlines():
        if line.startswith("ref:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].removeprefix("refs/heads/")
    print(f"error: could not determine default branch for {repo_url}", file=sys.stderr)
    raise SystemExit(1)


def default_branch_from_repo(repo_path: Path, *, env: dict[str, str] | None = None) -> str:
    """Return the default branch for a cloned dependency."""

    returncode, stdout, _ = run_capture(
        [
            "git",
            "-C",
            str(repo_path),
            "symbolic-ref",
            "--quiet",
            "--short",
            "refs/remotes/origin/HEAD",
        ],
        env=env,
    )
    branch = stdout.strip().removeprefix("origin/") if returncode == 0 else ""
    if branch:
        return branch
    print(
        f"error: could not determine default branch for dependency repo at {repo_path}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def first_dep_repo(dep_dir: Path) -> Path:
    """Return the first checked-out dependency repo under *dep_dir*."""

    return git_helpers.find_first_git_repo(dep_dir)
