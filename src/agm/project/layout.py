"""Project detection and configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.process import run_foreground

CONFIG_FILES: list[str] = [
    ".setup.sh",
    ".env",
    ".env.local",
    ".config",
    ".agents",
    ".opencode",
    ".codex",
    ".claude",
    ".pi",
    ".mcp.json",
]


def _copy_existing_config_files(source_dir: Path, target_dir: Path) -> None:
    existing_paths = [
        str(source_dir / name) for name in CONFIG_FILES if (source_dir / name).exists()
    ]
    if not existing_paths:
        return
    run_foreground(["cp", "-r", *existing_paths, str(target_dir)])


def _resolved_cwd(cwd: Path | None = None) -> Path:
    return Path.cwd() if cwd is None else cwd.resolve()


def current_project_dir(cwd: Path | None = None) -> Path:
    """Return the current project directory (``PROJ_DIR``)."""

    current = _resolved_cwd(cwd)

    for candidate in (current, *current.parents):
        if (candidate / "repo").is_dir():
            return candidate
        if candidate.name == "repo" and (
            (candidate.parent / "worktrees").is_dir()
            or (candidate.parent / ".worktrees").is_dir()
        ):
            return candidate.parent
        if candidate.parent.name == ".worktrees":
            return candidate.parent.parent
        if candidate.parent.name == "worktrees" and (candidate.parent.parent / "repo").is_dir():
            return candidate.parent.parent
    return current


def is_complex_project(project_dir: Path) -> bool:
    """Return whether *project_dir* contains a repo/ subdirectory."""

    return (project_dir / "repo").is_dir()


def main_repo_dir(project_dir: Path) -> Path:
    """Return the main repository directory for *project_dir*."""

    if is_complex_project(project_dir):
        return project_dir / "repo"
    return project_dir


def default_worktrees_dir(project_dir: Path) -> Path:
    """Return the default worktrees directory for *project_dir*."""

    if (project_dir / "worktrees").is_dir():
        return project_dir / "worktrees"
    return project_dir / ".worktrees"


def is_main_checkout_branch(project_dir: Path, branch: str, *, repo_branch: str) -> bool:
    """Return whether *branch* resolves to the main repo checkout."""

    return branch in {"repo", repo_branch}


def branch_worktree_path(project_dir: Path, branch: str, *, repo_branch: str) -> Path:
    """Return the checkout path corresponding to *branch*."""

    if is_main_checkout_branch(project_dir, branch, repo_branch=repo_branch):
        return main_repo_dir(project_dir)
    return default_worktrees_dir(project_dir) / branch


def branch_session_name(project_dir: Path, branch: str) -> str:
    """Return the tmux session name corresponding to *branch*."""

    if branch == "repo":
        return project_dir.name

    repo_branch = git_helpers.current_branch(main_repo_dir(project_dir))
    if is_main_checkout_branch(project_dir, branch, repo_branch=repo_branch):
        return project_dir.name
    return f"{project_dir.name}/{branch}"


def exit_if_main_checkout_branch(project_dir: Path, branch: str, *, repo_branch: str) -> None:
    """Exit when *branch* resolves to the main repo checkout."""

    if not is_main_checkout_branch(project_dir, branch, repo_branch=repo_branch):
        return
    print(
        (
            f"error: '{branch}' resolves to the main repo checkout at "
            f"{main_repo_dir(project_dir)} and cannot be managed as a branch worktree"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)


def copy_config(
    *,
    project_dir: Path | None = None,
    target: Path,
    cwd: Path | None = None,
) -> None:
    """Copy known config files from cwd and project config/ into *target*."""

    current = _resolved_cwd(cwd)
    proj_dir = current_project_dir(current) if project_dir is None else project_dir.resolve()
    resolved_target = target if target.is_absolute() else current / target
    if not resolved_target.is_dir():
        return

    _copy_existing_config_files(current, resolved_target)

    config_dir = proj_dir / "config"
    if config_dir.is_dir():
        _copy_existing_config_files(config_dir, resolved_target)
