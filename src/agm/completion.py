"""CLI completion helpers for Typer parameters."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import cast

import click
from click.shell_completion import CompletionItem
from typer.core import TyperCommand

import agm.vcs.git as git_helpers
from agm.config.context import current_config_context
from agm.config.general import load_merged_config, load_run_config
from agm.project.dependency_checkout import main_dep_repo
from agm.project.layout import (
    current_workspace_or_project_root,
    default_worktrees_dir,
    discover_current_project_dir,
    project_deps_dir,
    project_repo_dir,
)

_COMMON_PANE_COUNTS = ["1", "2", "3", "4", "6", "8", "12", "16"]

_HELP_TREE: dict[str, list[str]] = {
    "": [
        "open",
        "close",
        "init",
        "workspace",
        "wsp",
        "sync",
        "config",
        "wt",
        "worktree",
        "dep",
        "run",
        "loop",
        "tmux",
        "help",
    ],
    "loop": ["select", "run", "step"],
    "config": ["cp", "copy", "env", "update"],
    "workspace": ["open", "close", "setup", "list"],
    "wsp": ["open", "close", "setup", "list"],
    "sync": ["fetch", "pull"],
    "wt": ["new", "rm", "remove"],
    "worktree": ["new", "rm", "remove"],
    "dep": ["list", "new", "switch", "rm", "remove"],
    "tmux": ["open", "close", "layout"],
}


def _match(candidates: set[str] | list[str], incomplete: str) -> list[str]:
    return sorted(candidate for candidate in candidates if candidate.startswith(incomplete))


def _resolve_project_repo_dir() -> Path | None:
    try:
        project_dir = discover_current_project_dir()
    except SystemExit:
        return None
    if project_dir is None:
        return None
    return project_repo_dir(project_dir)


def _resolve_project_deps_dir() -> Path | None:
    try:
        project_dir = discover_current_project_dir()
    except SystemExit:
        return None
    if project_dir is None:
        return None
    return project_deps_dir(project_dir)


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

    project_dir = current_workspace_or_project_root(repo_dir)
    worktrees_dir = default_worktrees_dir(project_dir)
    branches: set[str] = set()
    try:
        for worktree in git_helpers.worktree_list(repo_dir):
            branch = worktree.branch
            if branch is None:
                try:
                    relative_path = worktree.path.relative_to(worktrees_dir)
                except ValueError:
                    continue
                branch = relative_path.as_posix()
            if branch and branch != repo_branch:
                branches.add(branch)
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


def complete_help_path(ctx: click.Context, incomplete: str) -> list[str]:
    try:
        params = cast(dict[str, object], ctx.params)
        raw_help = params.get("help_command")
        help_command = cast(list[str], raw_help) if isinstance(raw_help, list) else []
        path_key = help_command[-1] if help_command else ""
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


def complete_dep_branch(ctx: click.Context, incomplete: str) -> list[str]:
    try:
        params = cast(dict[str, object], ctx.params)
        raw_dep = params.get("dep")
        dep_name = raw_dep if isinstance(raw_dep, str) else ""
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


def complete_run_command(ctx: click.Context, incomplete: str) -> list[str]:
    try:
        params = cast(dict[str, object], ctx.params)
        raw_cmd = params.get("run_command_args")
        cmd_args = cast(list[str], raw_cmd) if isinstance(raw_cmd, list) else []
        if cmd_args:
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

        try:
            context = current_config_context()
            run_config = load_run_config(
                home=context.home,
                proj_dir=context.proj_dir,
                cwd=context.cwd,
            )
        except (OSError, SystemExit):
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


def _configured_command_names(
    section: str, *, home: Path, proj_dir: Path | None, cwd: Path
) -> set[str]:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    table = merged.get(section)
    if not isinstance(table, dict):
        return set()
    return {
        key
        for key, value in table.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def complete_agl_file(
    ctx: click.Context, args: list[str], incomplete: str
) -> list[str]:
    """Complete ``.agl`` file paths for the ``agm exec FILE`` argument."""
    del ctx, args
    try:
        candidates = _path_candidates(incomplete)
        return [c for c in candidates if c.endswith(".agl") or c.endswith("/")]
    except (Exception, SystemExit):
        return []


def _exec_param_completion_items(source: str, incomplete: str) -> list[CompletionItem]:
    """Return ``CompletionItem`` objects for ``--<param>`` flags discovered in *source*.

    Used by :class:`ExecCommand` to augment the standard shell_complete results.
    Degrades silently to ``[]`` on any error.
    """
    from agm.agl.typecheck.types import BoolType
    from agm.cli_support.exec_params import (
        discover_params_from_source,
        negative_param_flag,
        param_flag,
    )

    items: list[CompletionItem] = []
    for param in discover_params_from_source(source):
        flag = param_flag(param.name)
        if flag.startswith(incomplete):
            items.append(CompletionItem(flag))
        if isinstance(param.type, BoolType):
            no_flag = negative_param_flag(param.name)
            if no_flag.startswith(incomplete):
                items.append(CompletionItem(no_flag))
    return items


class ExecCommand(TyperCommand):
    """Typer Command subclass for ``agm exec`` that augments shell completion.

    When the incomplete token starts with ``--``, the standard completion (built-in
    exec options) is extended with ``--<param>`` / ``--no-<param>`` items discovered
    from the FILE or ``-c``/``--command`` source already parsed into ``ctx.params``.
    Degrades to base completion on any error (unreadable file, parse failure, etc.).
    """

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        base = super().shell_complete(ctx, incomplete)
        if not incomplete.startswith("-"):
            return base
        try:
            params = cast(dict[str, object], ctx.params)
            source: str | None = None
            raw_command = params.get("command")
            if isinstance(raw_command, str):
                source = raw_command
            else:
                raw_file = params.get("file")
                if isinstance(raw_file, str):
                    try:
                        source = Path(raw_file).read_text()
                    except OSError:
                        source = None
            if source is None:
                return base
            extra = _exec_param_completion_items(source, incomplete)
        except (Exception, SystemExit):
            return base
        return base + extra


def complete_revise_command_or_review_file(
    ctx: click.Context, args: list[str], incomplete: str
) -> list[str]:
    del ctx, args
    try:
        try:
            context = current_config_context()
            command_matches = _match(
                _configured_command_names(
                    "revise",
                    home=context.home,
                    proj_dir=context.proj_dir,
                    cwd=context.cwd,
                ),
                incomplete,
            )
        except (OSError, SystemExit):
            command_matches = []
        if command_matches:
            return command_matches
        return _path_candidates(incomplete)
    except (Exception, SystemExit):
        return []
