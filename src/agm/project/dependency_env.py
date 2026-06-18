"""Dependency environment variable management."""

from __future__ import annotations

import re
from pathlib import Path

import tomlkit

import agm.vcs.git as git_helpers
from agm.core.fs import exists, is_dir, is_file, iterdir, mkdir, read_text, rglob, write_text
from agm.core.toml import TomlDict, load_toml_file, set_toml_table_value, toml_dict
from agm.project.layout import (
    current_workspace,
    default_worktrees_dir,
    project_config_dir,
    project_deps_dir,
    project_repo_dir,
)


def current_config_branch(
    project_dir: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str | None:
    """Return the current branch config name, or ``None`` for the main workspace."""

    result = current_workspace(project_dir, cwd=cwd, env=env)
    return result.branch if result is not None else None


def dep_env_var_name(dep_name: str) -> str:
    """Return the environment variable name for *dep_name*."""

    name = re.sub(r"[^A-Za-z0-9]+", "_", dep_name).strip("_").upper()
    if not name:
        return "DEP"
    if name[0].isdigit():
        return f"_{name}"
    return name


def config_toml_file(project_dir: Path, branch: str | None) -> Path:
    """Return the config TOML file for the main repo or *branch*."""

    config_dir = project_config_dir(project_dir)
    if branch is None:
        return config_dir / "config.toml"
    return config_dir / branch / "config.toml"


def read_deps_table(config_file: Path) -> dict[str, str]:
    """Read the [deps] table from a config TOML file.

    Returns a dict mapping dep name to dep branch (checkout name).
    Returns an empty dict if the file does not exist or has no [deps] table.
    Raises ``OSError`` or ``tomlkit.exceptions.ParseError`` if the file is
    unreadable or malformed, consistent with ``load_dependency_toml_env``.
    """
    if not config_file.is_file():
        return {}
    deps_table = toml_dict(load_toml_file(config_file).get("deps"))
    result: dict[str, str] = {}
    for dep_name, dep_branch in deps_table.items():
        if isinstance(dep_branch, str) and dep_branch:
            result[dep_name] = dep_branch
    return result


def load_dependency_toml_env(
    *,
    project_dir: Path,
    config_files: list[Path],
    env: dict[str, str],
) -> dict[str, str]:
    """Return *env* updated from dependency branches in config TOML files."""

    resolved_env = dict(env)
    deps_dir = project_deps_dir(project_dir)
    for config_file in config_files:
        if not config_file.is_file():
            continue
        deps_table = toml_dict(load_toml_file(config_file).get("deps"))
        for dep_name, dep_branch in deps_table.items():
            if not isinstance(dep_branch, str) or not dep_branch:
                continue
            dep_path = deps_dir / dep_name / dep_branch
            resolved_env[dep_env_var_name(dep_name)] = str(dep_path)
    return resolved_env


def _set_toml_deps_value(content: str, dep_name: str, dep_branch: str) -> str:
    """Set *dep_name* = *dep_branch* in ``[deps]``, preserving formatting and comments."""

    doc = tomlkit.parse(content) if content.strip() else tomlkit.document()
    set_toml_table_value(doc, "deps", dep_name, dep_branch)
    return tomlkit.dumps(doc)


def _set_toml_project_name(content: str, name: str) -> str:
    """Set ``name`` in ``[project]``, preserving formatting and comments."""

    doc = tomlkit.parse(content) if content.strip() else tomlkit.document()
    set_toml_table_value(doc, "project", "name", name)
    return tomlkit.dumps(doc)


def update_dependency_toml_config(
    *,
    project_dir: Path,
    dep_name: str,
    dep_branch: str,
    config_branch: str | None,
) -> None:
    """Update one dependency branch in the relevant config TOML file."""

    config_file = config_toml_file(project_dir, config_branch)
    content = read_text(config_file, encoding="utf-8") if exists(config_file) else ""
    updated = _set_toml_deps_value(content, dep_name, dep_branch)
    mkdir(config_file.parent, parents=True, exist_ok=True)
    write_text(config_file, updated, encoding="utf-8")


def update_dependency_config(
    *,
    project_dir: Path,
    dep_name: str,
    dep_branch: str,
    config_branch: str | None,
) -> None:
    """Update one dependency branch in the relevant config TOML file."""

    update_dependency_toml_config(
        project_dir=project_dir,
        dep_name=dep_name,
        dep_branch=dep_branch,
        config_branch=config_branch,
    )


def dependency_repo_paths(dep_dir: Path) -> list[Path]:
    candidates = [dep_dir, *rglob(dep_dir, "*")]
    return [
        path
        for path in candidates
        if is_dir(path) and exists(path / ".git") and git_helpers.is_git_repo(path)
    ]


def dependency_repo_sort_key(dep_dir: Path, repo_path: Path) -> tuple[int, str]:
    return len(repo_path.relative_to(dep_dir).parts), str(repo_path)


def _dependency_checkout_name(dep_dir: Path, repo_path: Path) -> str:
    return repo_path.relative_to(dep_dir).as_posix()


def _main_dependency_checkout_name(dep_dir: Path) -> str | None:
    repos = [
        (*dependency_repo_sort_key(dep_dir, repo_path), repo_path)
        for repo_path in dependency_repo_paths(dep_dir)
    ]
    if not repos:
        return None
    return _dependency_checkout_name(dep_dir, sorted(repos)[0][2])


def _dependency_config_checkout_name(dep_dir: Path, config_branch: str) -> str | None:
    branch_path = dep_dir / config_branch
    if exists(branch_path / ".git") and git_helpers.is_git_repo(branch_path):
        return _dependency_checkout_name(dep_dir, branch_path)
    return _main_dependency_checkout_name(dep_dir)


def update_main_dependency_configs(project_dir: Path) -> None:
    """Create/update the main config TOML with known dependency branches."""

    deps_dir = project_deps_dir(project_dir)
    if not is_dir(deps_dir):
        return
    for dep_dir in sorted(path for path in iterdir(deps_dir) if is_dir(path)):
        checkout_name = _main_dependency_checkout_name(dep_dir)
        if checkout_name is None:
            continue
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name=dep_dir.name,
            dep_branch=checkout_name,
            config_branch=None,
        )


def _ensure_config_toml_file(project_dir: Path, branch: str | None) -> None:
    config_file = config_toml_file(project_dir, branch)
    if exists(config_file):
        return
    mkdir(config_file.parent, parents=True, exist_ok=True)
    write_text(config_file, "", encoding="utf-8")


def update_dependency_configs_for_branch(
    *,
    project_dir: Path,
    branch: str,
) -> None:
    """Update branch config TOML values for all project dependencies."""

    deps_dir = project_deps_dir(project_dir)
    if not is_dir(deps_dir):
        return
    for dep_dir in sorted(path for path in iterdir(deps_dir) if is_dir(path)):
        checkout_name = _dependency_config_checkout_name(dep_dir, branch)
        if checkout_name is None:
            continue
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name=dep_dir.name,
            dep_branch=checkout_name,
            config_branch=branch,
        )


def _checked_out_project_worktree_branches(
    project_dir: Path,
    *,
    env: dict[str, str] | None = None,
) -> list[str]:
    repo_dir = project_repo_dir(project_dir)
    worktrees_dir = default_worktrees_dir(project_dir).resolve(strict=False)
    branches: set[str] = set()
    for worktree in git_helpers.worktree_list(repo_dir, env=env):
        if worktree.branch is None:
            continue
        worktree_path = worktree.path.resolve(strict=False)
        if worktree_path == worktrees_dir or worktrees_dir not in worktree_path.parents:
            continue
        branches.add(worktree.branch)
    return sorted(branches)


def update_all_project_dependency_configs(
    project_dir: Path,
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Create/update main and branch config TOML files for the project."""

    repo_dir = project_repo_dir(project_dir)
    main_branch = git_helpers.current_branch(repo_dir, env=env)
    _ensure_config_toml_file(project_dir, None)
    update_main_dependency_configs(project_dir)
    for branch in _checked_out_project_worktree_branches(project_dir, env=env):
        if branch == main_branch:
            continue
        _ensure_config_toml_file(project_dir, branch)
        update_dependency_configs_for_branch(project_dir=project_dir, branch=branch)


def _seed_from_parent_config(
    *,
    project_dir: Path,
    parent_branch: str,
    branch: str,
) -> None:
    """Seed the new branch config directory from the parent branch config."""

    config_dir = project_config_dir(project_dir)
    parent_config = config_dir / parent_branch
    new_config = config_dir / branch

    if not is_dir(parent_config):
        return
    if is_dir(new_config) and any(iterdir(new_config)):
        return

    mkdir(new_config, parents=True, exist_ok=True)
    for item in iterdir(parent_config):
        if is_file(item):
            write_text(new_config / item.name, read_text(item), encoding="utf-8")


def ensure_dependency_configs_for_branch(
    *,
    project_dir: Path,
    branch: str,
    parent_branch: str | None = None,
) -> None:
    """Create missing branch config TOML values for project dependencies."""

    deps_dir = project_deps_dir(project_dir)
    if not is_dir(deps_dir):
        return

    if parent_branch is not None:
        _seed_from_parent_config(
            project_dir=project_dir, parent_branch=parent_branch, branch=branch
        )

    config_file = config_toml_file(project_dir, branch)
    existing_deps: TomlDict = {}
    if config_file.is_file():
        existing_deps = toml_dict(load_toml_file(config_file).get("deps"))

    for dep_dir in sorted(path for path in iterdir(deps_dir) if is_dir(path)):
        existing_branch = existing_deps.get(dep_dir.name)
        if isinstance(existing_branch, str) and existing_branch:
            continue
        checkout_name = _dependency_config_checkout_name(dep_dir, branch)
        if checkout_name is None:
            continue
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name=dep_dir.name,
            dep_branch=checkout_name,
            config_branch=branch,
        )


def ensure_project_name_in_config(
    *,
    project_dir: Path,
    name: str,
) -> None:
    """Write ``[project].name`` into the main config TOML if not already set."""

    config_file = config_toml_file(project_dir, None)
    content = read_text(config_file, encoding="utf-8") if exists(config_file) else ""
    updated = _set_toml_project_name(content, name)
    mkdir(config_file.parent, parents=True, exist_ok=True)
    write_text(config_file, updated, encoding="utf-8")
