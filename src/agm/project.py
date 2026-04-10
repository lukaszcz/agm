"""Project detection and configuration helpers."""

from __future__ import annotations

from pathlib import Path

from agm.shell import run_foreground

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


def detect_project_dir(cwd: Path | None = None) -> Path:
    """Detect the current project root."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    current_name = current.name
    parent = current.parent
    parent_name = parent.name
    grandparent = parent.parent

    if current_name == "repo" and (parent / "worktrees").is_dir():
        return parent
    if parent_name == "worktrees" and (grandparent / "repo").is_dir():
        return grandparent
    if parent_name == ".worktrees":
        return grandparent
    if (current / "repo").is_dir():
        return current
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

def copy_config(
    *,
    project_dir: Path | None = None,
    target: Path,
    cwd: Path | None = None,
) -> None:
    """Copy known config files from cwd and project config/ into *target*."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    proj_dir = detect_project_dir(current) if project_dir is None else project_dir.resolve()
    resolved_target = target if target.is_absolute() else current / target

    run_foreground(["cp", "-r", *CONFIG_FILES, str(resolved_target)], cwd=current)

    config_dir = proj_dir / "config"
    if config_dir.is_dir():
        run_foreground(["cp", "-r", *CONFIG_FILES, str(resolved_target)], cwd=config_dir)
