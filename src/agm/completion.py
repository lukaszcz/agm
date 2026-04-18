"""CLI completion helpers for Typer parameters."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click

import agm.vcs.git as git_helpers
from agm.commands.dep.common import main_dep_repo
from agm.config.general import load_run_config
from agm.project.layout import current_project_dir, project_deps_dir, project_repo_dir

_COMMON_PANE_COUNTS = ["1", "2", "3", "4", "6", "8", "12", "16"]

_HELP_TREE: dict[str, list[str]] = {
    "": [
        "open",
        "close",
        "init",
        "fetch",
        "config",
        "wt",
        "worktree",
        "dep",
        "run",
        "tmux",
        "help",
    ],
    "config": ["cp", "copy"],
    "wt": ["new", "setup", "rm", "remove"],
    "worktree": ["new", "setup", "rm", "remove"],
    "dep": ["new", "switch", "rm"],
    "tmux": ["open", "close", "layout"],
}


def _match(candidates: set[str] | list[str], incomplete: str) -> list[str]:
    return sorted(candidate for candidate in candidates if candidate.startswith(incomplete))


def _resolve_project_repo_dir() -> Path | None:
    try:
        return project_repo_dir(current_project_dir())
    except SystemExit:
        return None


def _resolve_project_deps_dir() -> Path | None:
    try:
        return project_deps_dir(current_project_dir())
    except SystemExit:
        return None


def _branch_candidates(repo_dir: Path) -> set[str]:
    candidates: set[str] = set()

    try:
        candidates.add(git_helpers.current_branch(repo_dir))
    except SystemExit:
        pass

    try:
        for worktree in git_helpers.worktree_list(repo_dir):
            if worktree.branch is not None:
                candidates.add(worktree.branch)
    except SystemExit:
        pass

    for ref in ("refs/heads", "refs/remotes/origin"):
        returncode, stdout, _ = git_helpers.fetch_output(
            ["git", "-C", str(repo_dir), "for-each-ref", "--format=%(refname:short)", ref]
        )
        if returncode != 0:
            continue
        for line in stdout.splitlines():
            if ref == "refs/remotes/origin":
                if line == "origin/HEAD":
                    continue
                if line.startswith("origin/"):
                    candidates.add(line.removeprefix("origin/"))
            elif line:
                candidates.add(line)
    return candidates


def _worktree_branch_candidates(repo_dir: Path) -> set[str]:
    try:
        repo_branch = git_helpers.current_branch(repo_dir)
    except SystemExit:
        return set()

    branches: set[str] = set()
    try:
        for worktree in git_helpers.worktree_list(repo_dir):
            if worktree.branch is not None and worktree.branch != repo_branch:
                branches.add(worktree.branch)
    except SystemExit:
        return set()
    return branches


def _resolve_dep_repo(dep_name: str) -> Path | None:
    deps_dir = _resolve_project_deps_dir()
    if deps_dir is None:
        return None
    dep_dir = deps_dir / dep_name
    if not dep_dir.is_dir():
        return None
    try:
        return main_dep_repo(dep_dir)
    except SystemExit:
        return None


def _path_candidates(incomplete: str) -> list[str]:
    current = Path.cwd()
    base_dir = current
    prefix = incomplete
    if incomplete:
        incomplete_path = Path(incomplete)
        if incomplete_path.parent != Path("."):
            base_dir = (current / incomplete_path.parent).resolve(strict=False)
            prefix = incomplete_path.name

    if not base_dir.is_dir():
        return []

    candidates: set[str] = set()
    for path in base_dir.iterdir():
        if not path.name.startswith(prefix):
            continue
        try:
            relative = path.relative_to(current)
            display = str(relative)
        except ValueError:
            display = str(path)
        if path.is_dir():
            display = f"{display}/"
        candidates.add(display)
    return sorted(candidates)


def complete_help_path(args: list[str], incomplete: str) -> list[str]:
    try:
        path_key = args[0] if args else ""
        return _match(_HELP_TREE.get(path_key, []), incomplete)
    except (Exception, SystemExit):
        return []


def complete_open_target(incomplete: str) -> list[str]:
    try:
        repo_dir = _resolve_project_repo_dir()
        if repo_dir is None:
            return []
        candidates = _branch_candidates(repo_dir)
        candidates.add("repo")
        return _match(candidates, incomplete)
    except (Exception, SystemExit):
        return []


def complete_close_branch(incomplete: str) -> list[str]:
    try:
        repo_dir = _resolve_project_repo_dir()
        if repo_dir is None:
            return []
        return _match(_worktree_branch_candidates(repo_dir), incomplete)
    except (Exception, SystemExit):
        return []


def complete_worktree_branch(incomplete: str) -> list[str]:
    try:
        repo_dir = _resolve_project_repo_dir()
        if repo_dir is None:
            return []
        return _match(_branch_candidates(repo_dir), incomplete)
    except (Exception, SystemExit):
        return []


def complete_dep_name(incomplete: str) -> list[str]:
    try:
        deps_dir = _resolve_project_deps_dir()
        if deps_dir is None or not deps_dir.is_dir():
            return []
        names = {path.name for path in deps_dir.iterdir() if path.is_dir()}
        return _match(names, incomplete)
    except (Exception, SystemExit):
        return []


def complete_dep_branch(args: list[str], incomplete: str) -> list[str]:
    try:
        dep_name = next((arg for arg in args if not arg.startswith("-")), "")
        if not dep_name:
            return []
        repo_dir = _resolve_dep_repo(dep_name)
        if repo_dir is None:
            return []
        return _match(_branch_candidates(repo_dir), incomplete)
    except (Exception, SystemExit):
        return []


def complete_dep_target(incomplete: str) -> list[str]:
    try:
        deps_dir = _resolve_project_deps_dir()
        if deps_dir is None or not deps_dir.is_dir():
            return []

        candidates: set[str] = set()
        for dep_dir in (path for path in deps_dir.iterdir() if path.is_dir()):
            dep_name = dep_dir.name
            candidates.add(dep_name)
            repo_dir = _resolve_dep_repo(dep_name)
            if repo_dir is None:
                continue
            candidates.add(f"{dep_name}/repo")
            for branch in _worktree_branch_candidates(repo_dir):
                candidates.add(f"{dep_name}/{branch}")
        return _match(candidates, incomplete)
    except (Exception, SystemExit):
        return []


def complete_run_command(args: list[str], incomplete: str) -> list[str]:
    try:
        if args:
            return _path_candidates(incomplete)

        candidates: set[str] = set()
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if not directory:
                continue
            path = Path(directory)
            if not path.is_dir():
                continue
            for candidate in path.iterdir():
                if (
                    candidate.is_file()
                    and os.access(candidate, os.X_OK)
                    and candidate.name.startswith(incomplete)
                ):
                    candidates.add(candidate.name)

        home = Path(os.environ.get("HOME", "~")).expanduser()
        cwd = Path.cwd()
        try:
            proj_dir = current_project_dir(cwd)
        except (OSError, SystemExit):
            proj_dir = None
        try:
            run_config = load_run_config(home=home, proj_dir=proj_dir, cwd=cwd)
        except OSError:
            run_config = None
        if run_config is not None:
            candidates.update(
                command_name
                for command_name in run_config.aliases
                if command_name.startswith(incomplete)
            )
        return sorted(candidates)
    except (Exception, SystemExit):
        return []


def complete_tmux_session(incomplete: str) -> list[str]:
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return _match(set(result.stdout.splitlines()), incomplete)


def complete_tmux_window(incomplete: str) -> list[str]:
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-F", "#{window_id}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return _match(set(result.stdout.splitlines()), incomplete)


def complete_pane_count(incomplete: str) -> list[str]:
    try:
        return _match(_COMMON_PANE_COUNTS, incomplete)
    except (Exception, SystemExit):
        return []


def complete_path_argument(
    ctx: click.Context, args: list[str], incomplete: str
) -> list[str]:
    del ctx, args
    try:
        return _path_candidates(incomplete)
    except (Exception, SystemExit):
        return []
