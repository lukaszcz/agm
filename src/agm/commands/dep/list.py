"""agm dep list — list dependency checkouts."""

from __future__ import annotations

from pathlib import Path

from agm.core.fs import is_dir
from agm.core.path import display_path
from agm.project.dependency_env import (
    config_toml_file,
    current_config_branch,
    dependency_repo_paths,
    dependency_repo_sort_key,
    read_deps_table,
)
from agm.project.layout import (
    project_deps_dir,
    require_current_project_dir,
)


def _deps_for_branch(
    project_dir: Path,
    branch: str | None,
) -> dict[str, str]:
    """Return the dependency checkouts configured for *branch* (or main)."""
    return read_deps_table(config_toml_file(project_dir, branch))


def _list_all_dep_checkouts(deps_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """Return all dependency checkouts found on disk.

    Returns a dict mapping dep name to a list of (checkout_name, path) tuples.
    Uses ``dependency_repo_paths`` so nested checkouts at any depth are
    included, consistent with how dependency repos are discovered elsewhere.
    """
    result: dict[str, list[tuple[str, Path]]] = {}
    if not is_dir(deps_dir):
        return result
    for dep_dir in sorted(p for p in deps_dir.iterdir() if is_dir(p)):

        def _sort_key(repo: Path, _dep_dir: Path = dep_dir) -> tuple[int, str]:
            return dependency_repo_sort_key(_dep_dir, repo)

        repo_paths = sorted(
            # Exclude dep_dir itself (when it's a git repo at root);
            # only nested checkouts are meaningful dep entries.
            [repo for repo in dependency_repo_paths(dep_dir) if repo != dep_dir],
            key=_sort_key,
        )
        entries: list[tuple[str, Path]] = []
        for repo_path in repo_paths:
            entries.append((repo_path.relative_to(dep_dir).as_posix(), repo_path))
        if entries:
            result[dep_dir.name] = entries
    return result


def list_deps(
    *,
    verbose: bool = False,
    all_checkouts: bool = False,
) -> None:
    """Print dependency checkouts for the current workspace (or all).

    Without --all, only the deps for the current workspace's config branch are
    listed.  With --all, every checkout under deps/ is listed grouped by
    dependency name.

    Output format is ``dep/branch``.  With -v/--verbose the checkout path is
    appended after the name.
    """
    project_dir = require_current_project_dir()
    deps_dir = project_deps_dir(project_dir)

    if all_checkouts:
        all_deps = _list_all_dep_checkouts(deps_dir)
        for dep_name, entries in all_deps.items():
            for checkout_name, checkout_path in entries:
                if verbose:
                    print(f"{dep_name}/{checkout_name}  {display_path(checkout_path)}")
                else:
                    print(f"{dep_name}/{checkout_name}")
        return

    config_branch = current_config_branch(project_dir)
    deps = _deps_for_branch(project_dir, config_branch)
    for dep_name, dep_branch in sorted(deps.items()):
        dep_path = deps_dir / dep_name / dep_branch
        if verbose:
            print(f"{dep_name}/{dep_branch}  {display_path(dep_path)}")
        else:
            print(f"{dep_name}/{dep_branch}")


def run(*, verbose: bool = False, all_checkouts: bool = False) -> None:
    list_deps(verbose=verbose, all_checkouts=all_checkouts)
