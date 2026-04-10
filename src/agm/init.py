"""Project initialisation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agm.shell import require_success


def usage() -> None:
    print("usage: pm-init.sh [-b branch] [project-name] [repo-url]", file=sys.stderr)
    print("       pm-init.sh [-b branch] repo-url", file=sys.stderr)
    raise SystemExit(1)


def looks_like_repo_url(value: str) -> bool:
    """Return whether *value* looks like a git repository URL."""

    return (
        "://" in value
        or value.startswith("git@") and ":" in value
        or "github.com:" in value
        or "github.com/" in value
        or value.endswith(".git")
    )


def derive_project_name(repo_url: str) -> str:
    """Derive a project name from *repo_url*."""

    trimmed = repo_url.rstrip("/")
    name = Path(trimmed).name.removesuffix(".git")
    if name in {"", ".", "/"}:
        print(
            f"error: could not derive project name from repo url: {repo_url}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return name


def write_file_if_missing(path: Path, content: str) -> None:
    """Write *content* with a trailing newline if *path* is missing."""

    if path.exists():
        return
    path.write_text(f"{content}\n", encoding="utf-8")


def init_project(
    *,
    branch: str | None,
    positional: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Initialise a project layout, optionally cloning a repo."""

    if not positional or len(positional) > 2:
        usage()

    proj = ""
    repo_url = ""
    if len(positional) == 1:
        if looks_like_repo_url(positional[0]):
            repo_url = positional[0]
        else:
            proj = positional[0]
    else:
        proj = positional[0]
        repo_url = positional[1]

    if not proj and not repo_url:
        usage()
    if not proj:
        proj = derive_project_name(repo_url)

    base_dir = Path.cwd() if cwd is None else cwd.resolve()
    project_dir = base_dir / proj
    for dirname in ("repo", "deps", "worktrees", "notes", "config"):
        (project_dir / dirname).mkdir(parents=True, exist_ok=True)

    write_file_if_missing(
        project_dir / "config" / "env.sh",
        "# Set project-level environment variables here.",
    )
    setup_path = project_dir / "config" / "setup.sh"
    write_file_if_missing(
        setup_path,
        "# Initialize a newly created worktree here.",
    )
    setup_path.chmod(setup_path.stat().st_mode | 0o111)

    if not repo_url:
        return

    repo_dir = project_dir / "repo"
    if any(repo_dir.iterdir()):
        print(f"error: {proj}/repo already exists and is not empty", file=sys.stderr)
        raise SystemExit(1)

    args = ["git", "clone"]
    if branch is not None:
        args.extend(["--branch", branch])
    args.extend([repo_url, str(repo_dir)])
    require_success(args, env=env)
