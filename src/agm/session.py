"""Project session management."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agm import git as git_helpers
from agm.project import default_worktrees_dir, detect_project_dir, main_repo_dir
from agm.shell import source_env_files
from agm.tmux_session import create_tmux_session
from agm.worktree import worktree_checkout


def _usage() -> None:
    print("usage: pm.sh open [-n pane_count] [branch]")
    print("       pm.sh new [-n pane_count] [-p parent] branch")
    print("       pm.sh {co|checkout} [-n pane_count] [-p parent] branch")
    raise SystemExit(1)


def _validate_pane_count(pane_count: str | None) -> None:
    if pane_count is None:
        return
    if not pane_count.isdigit() or int(pane_count) < 1:
        print("error: pane count must be a positive integer", file=sys.stderr)
        raise SystemExit(1)


def _project_paths(cwd: Path) -> tuple[Path, str]:
    proj_dir = detect_project_dir(cwd)
    return proj_dir, proj_dir.name


def _branch_path(proj_dir: Path, branch: str) -> Path:
    return default_worktrees_dir(proj_dir) / branch


def _load_env(proj_dir: Path, branch: str | None, *, shell_cwd: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PROJ_DIR"] = str(proj_dir)
    env_files = [proj_dir / "config" / "env.sh"]
    if branch:
        env_files.append(proj_dir / "config" / branch / "env.sh")
    return source_env_files(env_files, env, cwd=shell_cwd)


def open_session(
    *,
    pane_count: str | None,
    branch: str | None,
    cwd: Path | None = None,
) -> None:
    """Open a tmux session for a project or branch."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    _validate_pane_count(pane_count)
    proj_dir, proj_name = _project_paths(current)
    repo_path = _branch_path(proj_dir, branch) if branch else main_repo_dir(proj_dir)
    session_name = f"{proj_name}/{branch}" if branch else proj_name
    repo_path.mkdir(parents=True, exist_ok=True)
    env = _load_env(proj_dir, branch, shell_cwd=repo_path)
    create_tmux_session(
        detach=False,
        pane_count=pane_count,
        session_name=session_name,
        cwd=repo_path,
        env=env,
    )


def _resolve_parent_checkout_dir(proj_dir: Path, parent: str | None, *, env: dict[str, str]) -> Path:
    repo_dir = main_repo_dir(proj_dir)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    resolved_parent = parent or repo_branch
    if resolved_parent == repo_branch:
        return repo_dir
    return _branch_path(proj_dir, resolved_parent)


def new_session(
    *,
    pane_count: str | None,
    parent: str | None,
    branch: str,
    cwd: Path | None = None,
) -> None:
    """Create a new branch worktree and open a detached session."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    _validate_pane_count(pane_count)
    proj_dir, proj_name = _project_paths(current)
    repo_path = _branch_path(proj_dir, branch)
    repo_path.mkdir(parents=True, exist_ok=True)
    env = _load_env(proj_dir, branch, shell_cwd=repo_path)
    parent_dir = _resolve_parent_checkout_dir(proj_dir, parent, env=env)
    worktree_checkout(new_branch=branch, worktrees_dir=None, branch=None, cwd=parent_dir, env=env)
    create_tmux_session(
        detach=True,
        pane_count=pane_count,
        session_name=f"{proj_name}/{branch}",
        cwd=repo_path,
        env=env,
    )


def checkout_session(
    *,
    pane_count: str | None,
    parent: str | None,
    branch: str,
    cwd: Path | None = None,
) -> None:
    """Check out an existing branch and open a session."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    _validate_pane_count(pane_count)
    proj_dir, proj_name = _project_paths(current)
    repo_path = _branch_path(proj_dir, branch)
    repo_path.mkdir(parents=True, exist_ok=True)
    env = _load_env(proj_dir, branch, shell_cwd=repo_path)
    parent_dir = _resolve_parent_checkout_dir(proj_dir, parent, env=env)
    worktree_checkout(new_branch=None, worktrees_dir=None, branch=branch, cwd=parent_dir, env=env)
    create_tmux_session(
        detach=False,
        pane_count=pane_count,
        session_name=f"{proj_name}/{branch}",
        cwd=repo_path,
        env=env,
    )
