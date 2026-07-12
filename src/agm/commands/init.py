"""agm init."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import agm.vcs.git as git_helpers
from agm.cli_support.args import InitArgs
from agm.core.fs import chmod, exists, is_empty_dir, mkdir, read_text, stat, write_text
from agm.core.path import display_path
from agm.core.process import require_success
from agm.project.config_git import commit_config_dir_changes
from agm.project.dependency_env import ensure_project_name_in_config, update_main_dependency_configs
from agm.project.layout import (
    default_worktrees_dir,
    project_config_dir,
    project_deps_dir,
    project_notes_dir,
    project_repo_dir,
    project_root,
)

EMBEDDED_PROJECT_GITIGNORE_ENTRY = ".agm"
AGENT_FILES_GITIGNORE_ENTRY = ".agent-files"


def looks_like_repo_url(value: str) -> bool:
    return (
        "://" in value
        or value.startswith("git@") and ":" in value
        or "github.com:" in value
        or "github.com/" in value
        or value.endswith(".git")
    )


def derive_project_name(repo_url: str) -> str:
    try:
        return git_helpers.repo_name_from_url(repo_url)
    except ValueError:
        print(
            f"error: could not derive project name from repo url: {repo_url}",
            file=sys.stderr,
        )
        raise SystemExit(1)


def write_file_if_missing(path: Path, content: str) -> None:
    if exists(path):
        return
    write_text(path, f"{content}\n", encoding="utf-8")


def ensure_gitignore_entry(path: Path, entry: str) -> None:
    if not entry:
        return
    if exists(path):
        content = read_text(path, encoding="utf-8")
        existing_lines = content.splitlines()
        # Treat "foo" and "foo/" as equivalent gitignore entries for directories.
        base = entry.rstrip("/")
        if base in existing_lines or f"{base}/" in existing_lines:
            return
        suffix = "" if content.endswith("\n") else "\n"
        write_text(path, f"{content}{suffix}{entry}\n", encoding="utf-8")
        return
    write_text(path, f"{entry}\n", encoding="utf-8")


def ensure_git_repo(path: Path) -> None:
    if exists(path / ".git") and git_helpers.is_git_repo(path):
        return
    require_success(["git", "init", "-q", str(path)])


def configure_project_dir(
    project_dir: Path,
    *,
    embedded: bool,
    no_config_git: bool = False,
    no_notes_git: bool = False,
    no_repo_git: bool = False,
    no_git_init: bool = False,
) -> None:
    layout_dirs: Sequence[Path]
    if embedded:
        # project_dir is the .agm data directory; the git repo is its parent.
        mkdir(project_dir, parents=True, exist_ok=True)
        repo_root = project_dir.parent
        ensure_gitignore_entry(repo_root / ".gitignore", EMBEDDED_PROJECT_GITIGNORE_ENTRY)
        ensure_gitignore_entry(repo_root / ".gitignore", AGENT_FILES_GITIGNORE_ENTRY)
        config_dir = project_dir / "config"
        layout_dirs = (
            project_dir / "deps",
            project_dir / "notes",
            config_dir,
            project_dir / "worktrees",
        )
    else:
        mkdir(project_dir, parents=True, exist_ok=True)
        config_dir = project_config_dir(project_dir)
        layout_dirs = (
            project_dir / "repo",
            project_deps_dir(project_dir),
            project_notes_dir(project_dir),
            config_dir,
            default_worktrees_dir(project_dir),
        )
    for dirname in layout_dirs:
        mkdir(dirname, parents=True, exist_ok=True)
    if not embedded:
        repo_dir = project_repo_dir(project_dir)
        skip_repo_git = no_repo_git or no_git_init
        if not skip_repo_git:
            ensure_git_repo(repo_dir)
        if git_helpers.is_git_repo(repo_dir):
            ensure_gitignore_entry(repo_dir / ".gitignore", AGENT_FILES_GITIGNORE_ENTRY)

    notes_dir = project_notes_dir(project_dir)
    skip_config_git = no_config_git or no_git_init
    skip_notes_git = no_notes_git or no_git_init
    if not skip_config_git:
        ensure_git_repo(config_dir)
    if not skip_notes_git:
        ensure_git_repo(notes_dir)
    update_main_dependency_configs(project_dir)
    ensure_project_name_in_config(project_dir=project_dir, name=project_root(project_dir).name)

    write_file_if_missing(
        config_dir / "env.sh",
        "# Set project-level environment variables here.",
    )
    setup_path = config_dir / "setup.sh"
    write_file_if_missing(
        setup_path,
        "# Initialize a newly created worktree here.",
    )
    setup_mode = 0o755 if not exists(setup_path) else stat(setup_path).st_mode | 0o111
    chmod(setup_path, setup_mode)
    if not skip_config_git:
        commit_config_dir_changes(
            project_dir, "chore: initialize config",
            add_paths=[config_dir],
        )


def use_embedded_layout(args: InitArgs, *, project_dir: Path, repo_url: str) -> bool:
    if args.embedded:
        return True
    if args.split:
        return False
    if repo_url:
        return False
    if exists(project_dir) and git_helpers.is_git_repo(project_dir):
        print("git repo detected, choosing embedded layout")
        return True
    return False


def run(args: InitArgs) -> None:
    positional: list[str] = args.positional
    if len(positional) > 2:
        raise SystemExit(1)

    proj = ""
    repo_url = ""
    if not positional:
        pass
    elif len(positional) == 1:
        if looks_like_repo_url(positional[0]):
            repo_url = positional[0]
        else:
            proj = positional[0]
    else:
        proj = positional[0]
        repo_url = positional[1]

    if args.clone and not repo_url:
        print("error: --clone requires REPO_URL", file=sys.stderr)
        raise SystemExit(1)
    if args.branch is not None and not repo_url:
        print("error: --branch requires REPO_URL", file=sys.stderr)
        raise SystemExit(1)
    if args.clone and not proj:
        proj = derive_project_name(repo_url)

    base_dir = Path.cwd() / proj if proj else Path.cwd()
    embedded_layout = use_embedded_layout(args, project_dir=base_dir, repo_url=repo_url)
    project_dir = base_dir / ".agm" if embedded_layout else base_dir
    if not repo_url:
        configure_project_dir(
            project_dir,
            embedded=embedded_layout,
            no_config_git=args.no_config_git,
            no_notes_git=args.no_notes_git,
            no_repo_git=args.no_repo_git,
            no_git_init=args.no_git_init,
        )
        return

    repo_dir = base_dir if embedded_layout else base_dir / "repo"
    if exists(repo_dir) and not is_empty_dir(repo_dir):
        display_dir = display_path(repo_dir)
        print(f"error: {display_dir} already exists and is not empty", file=sys.stderr)
        raise SystemExit(1)

    clone_args = ["git", "clone"]
    if args.branch is not None:
        clone_args.extend(["--branch", args.branch])
    clone_args.extend([repo_url, str(repo_dir)])
    require_success(clone_args)
    configure_project_dir(
        project_dir,
        embedded=embedded_layout,
        no_config_git=args.no_config_git,
        no_notes_git=args.no_notes_git,
        no_repo_git=args.no_repo_git,
        no_git_init=args.no_git_init,
    )
