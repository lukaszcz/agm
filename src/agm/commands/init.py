"""agm init."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import agm.vcs.git as git_helpers
from agm.commands.args import InitArgs
from agm.core.process import require_success
from agm.project.layout import (
    default_worktrees_dir,
    project_config_dir,
    project_deps_dir,
    project_notes_dir,
    project_repo_dir,
)


def looks_like_repo_url(value: str) -> bool:
    return (
        "://" in value
        or value.startswith("git@") and ":" in value
        or "github.com:" in value
        or "github.com/" in value
        or value.endswith(".git")
    )


def derive_project_name(repo_url: str) -> str:
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
    if path.exists():
        return
    path.write_text(f"{content}\n", encoding="utf-8")


def ensure_gitignore_entry(path: Path, entry: str) -> None:
    if path.exists():
        content = path.read_text(encoding="utf-8")
        existing_lines = content.splitlines()
        if entry in existing_lines:
            return
        suffix = "" if content.endswith("\n") else "\n"
        path.write_text(f"{content}{suffix}{entry}\n", encoding="utf-8")
        return
    path.write_text(f"{entry}\n", encoding="utf-8")


def configure_project_dir(project_dir: Path, *, embedded: bool) -> None:
    layout_dirs: Sequence[Path]
    if embedded:
        data_dir = project_dir / ".agm"
        data_dir.mkdir(parents=True, exist_ok=True)
        ensure_gitignore_entry(project_dir / ".gitignore", ".agm")
        config_dir = data_dir / "config"
        layout_dirs = (
            data_dir / "deps",
            data_dir / "notes",
            config_dir,
            data_dir / "worktrees",
        )
    else:
        project_dir.mkdir(parents=True, exist_ok=True)
        config_dir = project_config_dir(project_dir)
        layout_dirs = (
            project_dir / "repo",
            project_deps_dir(project_dir),
            project_notes_dir(project_dir),
            config_dir,
            default_worktrees_dir(project_dir),
        )
    for dirname in layout_dirs:
        dirname.mkdir(parents=True, exist_ok=True)

    write_file_if_missing(
        config_dir / "env.sh",
        "# Set project-level environment variables here.",
    )
    setup_path = config_dir / "setup.sh"
    write_file_if_missing(
        setup_path,
        "# Initialize a newly created worktree here.",
    )
    setup_path.chmod(setup_path.stat().st_mode | 0o111)


def use_embedded_layout(args: InitArgs, *, project_dir: Path, repo_url: str) -> bool:
    if args.embedded:
        return True
    if args.workspace:
        return False
    if repo_url:
        return False
    if project_dir.exists() and git_helpers.is_git_repo(project_dir):
        print("git repo detected, choosing embedded layout")
        return True
    return False


def run(args: InitArgs) -> None:
    positional: list[str] = args.positional
    if not positional or len(positional) > 2:
        raise SystemExit(1)

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
        raise SystemExit(1)
    if not proj:
        proj = derive_project_name(repo_url)

    base_dir = Path.cwd()
    project_dir = base_dir / proj
    embedded_layout = use_embedded_layout(args, project_dir=project_dir, repo_url=repo_url)
    if not repo_url:
        configure_project_dir(project_dir, embedded=embedded_layout)
        return

    repo_dir = project_repo_dir(project_dir) if embedded_layout else project_dir / "repo"
    if repo_dir.exists() and any(repo_dir.iterdir()):
        display_dir = proj if embedded_layout else f"{proj}/repo"
        print(f"error: {display_dir} already exists and is not empty", file=sys.stderr)
        raise SystemExit(1)

    clone_args = ["git", "clone"]
    if args.branch is not None:
        clone_args.extend(["--branch", args.branch])
    clone_args.extend([repo_url, str(repo_dir)])
    require_success(clone_args)
    configure_project_dir(project_dir, embedded=embedded_layout)
