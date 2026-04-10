"""Dependency management commands."""

from __future__ import annotations

import sys
from pathlib import Path

from agm import git as git_helpers
from agm.shell import require_success, run_capture, run_foreground


def usage() -> None:
    print("usage: pm-dep.sh new [-b branch] repo-url", file=sys.stderr)
    print("       pm-dep.sh switch dep [-b] branch", file=sys.stderr)
    raise SystemExit(1)


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


def dep_new(
    *,
    branch: str | None,
    repo_url: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Clone a new dependency under deps/."""

    project_dir = Path.cwd() if cwd is None else cwd.resolve()
    dep = derive_dep_name(repo_url)
    dep_dir = project_dir / "deps" / dep
    if dep_dir.exists():
        print(f"error: deps/{dep} already exists", file=sys.stderr)
        raise SystemExit(1)

    resolved_branch = branch or default_branch_from_remote(repo_url, env=env)
    target_dir = dep_dir / resolved_branch
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    returncode = run_foreground(
        ["git", "clone", "--branch", resolved_branch, repo_url, str(target_dir)],
        env=env,
    )
    if returncode != 0:
        try:
            dep_dir.rmdir()
        except OSError:
            pass
        raise SystemExit(returncode)


def dep_switch(
    *,
    dep: str,
    branch: str,
    create_branch: bool,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Switch an existing dependency to a different branch."""

    project_dir = Path.cwd() if cwd is None else cwd.resolve()
    dep_dir = project_dir / "deps" / dep
    if not dep_dir.is_dir():
        print(f"error: deps/{dep} does not exist", file=sys.stderr)
        raise SystemExit(1)

    repo_path = first_dep_repo(dep_dir)
    target_dir = dep_dir / branch
    if target_dir.exists():
        print(f"error: deps/{dep}/{branch} already exists", file=sys.stderr)
        raise SystemExit(1)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    git_helpers.fetch(repo_path, env=env)
    if create_branch:
        default_branch = default_branch_from_repo(repo_path, env=env)
        git_helpers.worktree_add(
            repo_path,
            target_dir,
            branch,
            create=True,
            start_point=default_branch,
            env=env,
        )
        return
    git_helpers.worktree_add(repo_path, target_dir, branch, env=env)
