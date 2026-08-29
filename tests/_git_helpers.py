"""Shared git fixtures for tests that need real repositories.

Building repositories with several remotes takes enough setup that the helper
is shared instead of repeated per test module.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path


def git_run(cwd: Path | None, args: Sequence[str], env: dict[str, str]) -> None:
    """Run a git command that is expected to succeed."""

    subprocess.run(["git", *args], cwd=cwd, env=env, check=True)


def git_output(cwd: Path, args: Sequence[str], env: dict[str, str]) -> str:
    """Return the stripped stdout of a successful git command."""

    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def init_repo(path: Path, env: dict[str, str], *, branch: str = "main") -> Path:
    """Initialize a git repo at *path* with one commit on *branch*."""

    path.mkdir(parents=True, exist_ok=True)
    git_run(path, ["init", "-b", branch, "-q"], env)
    commit_file(path, env, name="README.md", content="# test\n", message="initial")
    return path


def commit_file(
    repo: Path,
    env: dict[str, str],
    *,
    name: str,
    content: str = "content\n",
    message: str = "commit",
) -> None:
    """Write *name* in *repo* and commit it on the current branch."""

    (repo / name).write_text(content, encoding="utf-8")
    git_run(repo, ["add", name], env)
    git_run(repo, ["commit", "-m", message, "-q"], env)


def clone_local_remote(
    parent: Path,
    env: dict[str, str],
    *,
    branches: Sequence[str] = (),
    default_branch: str = "main",
    source_name: str = "source",
    clone_name: str = "repo",
) -> tuple[Path, Path]:
    """Create a source repo and a clone of it, giving the clone an ``origin``.

    Each name in *branches* becomes a branch on the source carrying one commit
    of its own, so the clone starts out with the matching remote-tracking refs.
    """

    source = init_repo(parent / source_name, env, branch=default_branch)
    for branch in branches:
        git_run(source, ["checkout", "-b", branch, "-q"], env)
        commit_file(source, env, name=f"{branch}.txt", message=f"{branch} commit")
        git_run(source, ["checkout", default_branch, "-q"], env)
    clone = parent / clone_name
    git_run(None, ["clone", "-q", str(source), str(clone)], env)
    return source, clone


def add_linked_worktree(repo: Path, worktree: Path, env: dict[str, str], *, branch: str) -> Path:
    """Create a linked worktree of *repo* at *worktree*, on a new *branch*."""

    subprocess.run(
        ["git", "worktree", "add", str(worktree), "-b", branch, "-q"],
        cwd=repo,
        env=env,
        check=True,
    )
    return worktree


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
    are already fetched; the bare remotes themselves stay at
    ``parent/origin.git`` and ``parent/fork.git``.
    """

    source = init_repo(parent / "source", env)
    git_run(source, ["checkout", "-b", branch, "-q"], env)
    commit_file(source, env, name=marker, content="fork\n", message="fork commit")
    git_run(source, ["checkout", "main", "-q"], env)

    origin = parent / "origin.git"
    fork = parent / "fork.git"
    git_run(None, ["clone", "--bare", "-q", str(source), str(origin)], env)
    git_run(None, ["clone", "--bare", "-q", str(source), str(fork)], env)
    if not on_origin:
        git_run(origin, ["branch", "-D", branch, "-q"], env)
    if not on_fork:
        git_run(fork, ["branch", "-D", branch, "-q"], env)

    repo = parent / "repo" if repo_dir is None else repo_dir
    repo.parent.mkdir(parents=True, exist_ok=True)
    git_run(None, ["clone", "-q", str(origin), str(repo)], env)
    git_run(repo, ["remote", "add", "fork", str(fork)], env)
    git_run(repo, ["fetch", "-q", "fork"], env)
    return repo
