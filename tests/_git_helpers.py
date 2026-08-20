"""Shared git fixtures for tests that need real repositories.

Building repositories with several remotes takes enough setup that the helper
is shared instead of repeated per test module.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def init_repo(path: Path, env: dict[str, str]) -> Path:
    """Initialize a git repo at *path* with one commit on ``main``."""

    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", "-q"], cwd=path, env=env, check=True)
    (path / "README.md").write_text("# test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "initial", "-q"], cwd=path, env=env, check=True)
    return path


def clone_with_fork_remote(
    parent: Path,
    env: dict[str, str],
    *,
    branch: str,
    on_origin: bool = False,
    on_fork: bool = True,
    repo_dir: Path | None = None,
    marker: str = "fork.txt",
) -> Path:
    """Clone a repo from ``origin`` and give it a second remote named ``fork``.

    *branch* carries a commit adding *marker* and ends up published on the
    remotes selected by *on_origin* / *on_fork*, so callers can build the
    "branch lives only in a fork" and "branch lives on both remotes" cases.
    The clone lands in *repo_dir* (default ``parent/repo``) and both remotes
    are already fetched.
    """

    source = init_repo(parent / "source", env)
    subprocess.run(["git", "checkout", "-b", branch, "-q"], cwd=source, env=env, check=True)
    (source / marker).write_text("fork\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=source, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "fork commit", "-q"], cwd=source, env=env, check=True)
    subprocess.run(["git", "checkout", "main", "-q"], cwd=source, env=env, check=True)

    origin = parent / "origin.git"
    fork = parent / "fork.git"
    subprocess.run(["git", "clone", "--bare", "-q", str(source), str(origin)], env=env, check=True)
    subprocess.run(["git", "clone", "--bare", "-q", str(source), str(fork)], env=env, check=True)
    if not on_origin:
        subprocess.run(["git", "branch", "-D", branch, "-q"], cwd=origin, env=env, check=True)
    if not on_fork:
        subprocess.run(["git", "branch", "-D", branch, "-q"], cwd=fork, env=env, check=True)

    repo = parent / "repo" if repo_dir is None else repo_dir
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(repo)], env=env, check=True)
    subprocess.run(["git", "remote", "add", "fork", str(fork)], cwd=repo, env=env, check=True)
    subprocess.run(["git", "fetch", "-q", "fork"], cwd=repo, env=env, check=True)
    return repo
