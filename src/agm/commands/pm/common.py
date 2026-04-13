"""Shared helpers for agm pm commands."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.tmux.session import create_tmux_session
from agm.utils.project import current_project_dir, default_worktrees_dir, main_repo_dir
from agm.utils.shell import source_env_files
from agm.utils.worktree import ensure_worktree


def validate_pane_count(pane_count: str | None) -> None:
    if pane_count is None:
        return
    if not pane_count.isdigit() or int(pane_count) < 1:
        print("error: pane count must be a positive integer", file=sys.stderr)
        raise SystemExit(1)


def branch_path(proj_dir: Path, branch: str) -> Path:
    return default_worktrees_dir(proj_dir) / branch


def load_env(proj_dir: Path, branch: str | None, *, shell_cwd: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PROJ_DIR"] = str(proj_dir)
    env_files = [proj_dir / "config" / "env.sh"]
    if branch:
        env_files.append(proj_dir / "config" / branch / "env.sh")
    return source_env_files(env_files, env, cwd=shell_cwd)


def resolve_parent_checkout_dir(proj_dir: Path, parent: str | None, *, env: dict[str, str]) -> Path:
    repo_dir = main_repo_dir(proj_dir)
    repo_branch = git_helpers.current_branch(repo_dir, env=env)
    resolved_parent = parent or repo_branch
    if resolved_parent == repo_branch:
        return repo_dir
    return branch_path(proj_dir, resolved_parent)


def open_session(*, pane_count: str | None, branch: str | None, cwd: Path | None = None) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
    proj_name = proj_dir.name

    if branch is None:
        repo_path = main_repo_dir(proj_dir)
    else:
        repo_path = ensure_worktree(
            new_branch=None,
            worktrees_dir=None,
            branch=branch,
            existing_ok=True,
            cwd=main_repo_dir(proj_dir),
        )
    env = load_env(proj_dir, branch, shell_cwd=repo_path)
    session_name = f"{proj_name}/{branch}" if branch else proj_name
    create_tmux_session(
        detach=False,
        pane_count=pane_count,
        session_name=session_name,
        cwd=repo_path,
        env=env,
    )


def new_session(*, pane_count: str | None, parent: str | None, branch: str, cwd: Path | None = None) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
    proj_name = proj_dir.name
    repo_path = branch_path(proj_dir, branch)
    repo_path.mkdir(parents=True, exist_ok=True)
    env = load_env(proj_dir, branch, shell_cwd=repo_path)
    parent_dir = resolve_parent_checkout_dir(proj_dir, parent, env=env)
    ensure_worktree(
        new_branch=branch,
        worktrees_dir=None,
        branch=None,
        existing_ok=False,
        cwd=parent_dir,
        env=env,
    )
    create_tmux_session(
        detach=True,
        pane_count=pane_count,
        session_name=f"{proj_name}/{branch}",
        cwd=repo_path,
        env=env,
    )


def checkout_session(*, pane_count: str | None, parent: str | None, branch: str, cwd: Path | None = None) -> None:
    current = Path.cwd() if cwd is None else cwd.resolve()
    validate_pane_count(pane_count)
    proj_dir = current_project_dir(current)
    proj_name = proj_dir.name
    repo_path = branch_path(proj_dir, branch)
    repo_path.mkdir(parents=True, exist_ok=True)
    env = load_env(proj_dir, branch, shell_cwd=repo_path)
    parent_dir = resolve_parent_checkout_dir(proj_dir, parent, env=env)
    ensure_worktree(
        new_branch=None,
        worktrees_dir=None,
        branch=branch,
        existing_ok=True,
        cwd=parent_dir,
        env=env,
    )
    create_tmux_session(
        detach=False,
        pane_count=pane_count,
        session_name=f"{proj_name}/{branch}",
        cwd=repo_path,
        env=env,
    )
