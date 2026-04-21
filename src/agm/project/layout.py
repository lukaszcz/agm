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


def _project_dir_from_checkout(checkout_dir: Path) -> Path | None:
    if (checkout_dir / ".agm").is_dir():
        return checkout_dir
    if (checkout_dir / "repo").is_dir():
        return checkout_dir
    if checkout_dir.name == "repo" and (
        (checkout_dir.parent / "worktrees").is_dir()
        or (checkout_dir.parent / ".worktrees").is_dir()
    ):
        return checkout_dir.parent
    if checkout_dir.parent.name == ".worktrees":
        return checkout_dir.parent.parent
    if checkout_dir.parent.name == "worktrees" and (checkout_dir.parent.parent / "repo").is_dir():
        return checkout_dir.parent.parent
    return None


def current_project_dir(cwd: Path | None = None) -> Path:
    """Return the current project directory (``PROJ_DIR``)."""

    current = _resolved_cwd(cwd)

    for candidate in (current, *current.parents):
        project_dir = _project_dir_from_checkout(candidate)
        if project_dir is not None:
            return project_dir
    if not git_helpers.is_git_repo(current):
        return current
    try:
        checkout_dir = git_helpers.git_setup(current)
    except SystemExit:
        return current
    for candidate in (checkout_dir, *checkout_dir.parents):
        project_dir = _project_dir_from_checkout(candidate)
        if project_dir is not None:
            return project_dir
    try:
        common_dir = git_helpers.git_common_dir(current)
    except SystemExit:
        common_dir = None
    if common_dir is not None:
        common_checkout = common_dir.parent
        for candidate in (common_checkout, *common_checkout.parents):
            project_dir = _project_dir_from_checkout(candidate)
            if project_dir is not None:
                return project_dir
    if checkout_dir == current or checkout_dir in current.parents:
        return checkout_dir
    return current


def is_workspace_project(project_dir: Path) -> bool:
    """Return whether *project_dir* uses the workspace layout."""

    return (project_dir / "repo").is_dir()


def project_data_dir(project_dir: Path) -> Path:
    """Return the AGM data directory for *project_dir*."""

    if (project_dir / ".agm").is_dir():
        return project_dir / ".agm"
    return project_dir


def project_repo_dir(project_dir: Path) -> Path:
    """Return the main repository directory for *project_dir*."""

    if is_workspace_project(project_dir):
        return project_dir / "repo"
    return project_dir


def main_repo_dir(project_dir: Path) -> Path:
    """Backward-compatible alias for ``project_repo_dir``."""

    return project_repo_dir(project_dir)


def default_worktrees_dir(project_dir: Path) -> Path:
    """Return the default worktrees directory for *project_dir*."""

    return project_data_dir(project_dir) / "worktrees"


def project_config_dir(project_dir: Path) -> Path:
    """Return the shared project config directory."""

    return project_data_dir(project_dir) / "config"


def project_deps_dir(project_dir: Path) -> Path:
    """Return the dependency checkout directory."""

    return project_data_dir(project_dir) / "deps"


def project_notes_dir(project_dir: Path) -> Path:
    """Return the project notes directory."""

    return project_data_dir(project_dir) / "notes"


def is_main_checkout_branch(project_dir: Path, branch: str, *, repo_branch: str) -> bool:
    """Return whether *branch* resolves to the main repo checkout."""

    return branch in {"repo", repo_branch}


def branch_worktree_path(project_dir: Path, branch: str, *, repo_branch: str) -> Path:
    """Return the checkout path corresponding to *branch*."""

    if is_main_checkout_branch(project_dir, branch, repo_branch=repo_branch):
        return project_repo_dir(project_dir)
    return default_worktrees_dir(project_dir) / branch


def branch_session_name(project_dir: Path, branch: str) -> str:
    """Return the tmux session name corresponding to *branch*."""

    if branch == "repo":
        return project_dir.name

    repo_branch = git_helpers.current_branch(project_repo_dir(project_dir))
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
            f"{project_repo_dir(project_dir)} and cannot be managed as a branch worktree"
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
    """Copy known config files from the project config directory into *target*."""

    current = _resolved_cwd(cwd)
    proj_dir = current_project_dir(current) if project_dir is None else project_dir.resolve()
    resolved_target = target if target.is_absolute() else current / target
    if not resolved_target.is_dir():
        return

    config_dir = project_config_dir(proj_dir)
    if config_dir.is_dir():
        _copy_existing_config_files(config_dir, resolved_target)
