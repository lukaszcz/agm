"""agm dep list — list dependency checkouts."""

from __future__ import annotations

from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.fs import exists, is_dir
from agm.project.dependency_env import current_config_branch, read_deps_table
from agm.project.layout import (
    project_config_dir,
    project_deps_dir,
    require_current_project_dir,
)


def _deps_for_branch(
    project_dir: Path,
    branch: str | None,
) -> dict[str, str]:
    """Return the dependency checkouts configured for *branch* (or main)."""
    config_dir = project_config_dir(project_dir)
    if branch is None:
        config_file = config_dir / "config.toml"
    else:
        config_file = config_dir / branch / "config.toml"
    return read_deps_table(config_file)


def _list_all_dep_checkouts(deps_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """Return all dependency checkouts found on disk.

    Returns a dict mapping dep name to a list of (checkout_name, path) tuples.
    Only includes directories that are git repos, consistent with
    ``_dependency_repo_paths``.
    """
    result: dict[str, list[tuple[str, Path]]] = {}
    if not is_dir(deps_dir):
        return result
    for dep_dir in sorted(p for p in deps_dir.iterdir() if is_dir(p)):
        entries: list[tuple[str, Path]] = []
        for child in sorted(
            p
            for p in dep_dir.iterdir()
            if is_dir(p) and exists(p / ".git") and git_helpers.is_git_repo(p)
        ):
            entries.append((child.name, child))
        if entries:
            result[dep_dir.name] = entries
    return result


def list_deps(
    *,
    verbose: bool = False,
    all_checkouts: bool = False,
    cwd: Path | None = None,
) -> None:
    """Print dependency checkouts for the current project checkout (or all).

    Without --all, only the deps for the current checkout's config branch are
    listed.  With --all, every checkout under deps/ is listed grouped by
    dependency name.

    Output format is ``dep/branch``.  With -v/--verbose the checkout path is
    appended after the name.
    """
    project_dir = require_current_project_dir(cwd=cwd)
    deps_dir = project_deps_dir(project_dir)

    if all_checkouts:
        all_deps = _list_all_dep_checkouts(deps_dir)
        for dep_name, entries in all_deps.items():
            for checkout_name, checkout_path in entries:
                if verbose:
                    print(f"{dep_name}/{checkout_name}  {checkout_path}")
                else:
                    print(f"{dep_name}/{checkout_name}")
        return

    config_branch = current_config_branch(project_dir, cwd=cwd)
    deps = _deps_for_branch(project_dir, config_branch)
    for dep_name, dep_branch in sorted(deps.items()):
        dep_path = deps_dir / dep_name / dep_branch
        if verbose:
            print(f"{dep_name}/{dep_branch}  {dep_path}")
        else:
            print(f"{dep_name}/{dep_branch}")


def run(*, verbose: bool = False, all_checkouts: bool = False) -> None:
    list_deps(verbose=verbose, all_checkouts=all_checkouts)
