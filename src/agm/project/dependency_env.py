"""Dependency environment variable management."""

from __future__ import annotations

import re
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.dotenv import set_dotenv_value
from agm.core.fs import exists, is_dir, iterdir, rglob, write_text
from agm.project.layout import project_config_dir, project_deps_dir, project_repo_dir


def dep_env_var_name(dep_name: str) -> str:
    """Return the environment variable name for *dep_name*."""

    name = re.sub(r"[^A-Za-z0-9]+", "_", dep_name).strip("_").upper()
    if not name:
        return "DEP"
    if name[0].isdigit():
        return f"_{name}"
    return name


def config_env_file(project_dir: Path, branch: str | None) -> Path:
    """Return the config dotenv file for the main repo or *branch*."""

    config_dir = project_config_dir(project_dir)
    if branch is None:
        return config_dir / ".env"
    return config_dir / branch / ".env"


def current_config_branch(
    project_dir: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str | None:
    """Return the current branch config name, or ``None`` for the main checkout."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    repo_dir = project_repo_dir(project_dir).resolve(strict=False)
    try:
        checkout_dir = git_helpers.git_setup(current).resolve(strict=False)
    except SystemExit:
        return None
    if checkout_dir == repo_dir:
        return None
    if repo_dir in checkout_dir.parents:
        return None
    return git_helpers.current_branch(checkout_dir, env=env)


def update_dependency_env_var(
    *,
    project_dir: Path,
    dep_name: str,
    dep_branch: str,
    config_branch: str | None,
) -> None:
    """Update one dependency environment variable in the relevant config dotenv."""

    env_file = config_env_file(project_dir, config_branch)
    dep_path = project_deps_dir(project_dir) / dep_name / dep_branch
    set_dotenv_value(env_file, dep_env_var_name(dep_name), str(dep_path))


def _dependency_repo_paths(dep_dir: Path) -> list[Path]:
    candidates = [dep_dir, *rglob(dep_dir, "*")]
    return [
        path
        for path in candidates
        if is_dir(path) and exists(path / ".git") and git_helpers.is_git_repo(path)
    ]


def _dependency_repo_sort_key(dep_dir: Path, repo_path: Path) -> tuple[int, str]:
    return len(repo_path.relative_to(dep_dir).parts), str(repo_path)


def _main_dependency_branch(dep_dir: Path) -> str | None:
    repos = [
        (*_dependency_repo_sort_key(dep_dir, repo_path), repo_path)
        for repo_path in _dependency_repo_paths(dep_dir)
    ]
    if not repos:
        return None
    return git_helpers.current_branch(sorted(repos)[0][2])


def update_main_dependency_env_vars(project_dir: Path) -> None:
    """Create/update the main config dotenv with known dependency paths."""

    env_file = config_env_file(project_dir, None)
    if not exists(env_file):
        write_text(env_file, "", encoding="utf-8")

    deps_dir = project_deps_dir(project_dir)
    if not is_dir(deps_dir):
        return
    for dep_dir in sorted(path for path in iterdir(deps_dir) if is_dir(path)):
        branch = _main_dependency_branch(dep_dir)
        if branch is None:
            continue
        update_dependency_env_var(
            project_dir=project_dir,
            dep_name=dep_dir.name,
            dep_branch=branch,
            config_branch=None,
        )


def update_dependency_env_vars_for_branch(
    *,
    project_dir: Path,
    branch: str,
) -> None:
    """Update branch config dotenv values for all project dependencies."""

    deps_dir = project_deps_dir(project_dir)
    if not is_dir(deps_dir):
        return
    for dep_dir in sorted(path for path in iterdir(deps_dir) if is_dir(path)):
        update_dependency_env_var(
            project_dir=project_dir,
            dep_name=dep_dir.name,
            dep_branch=branch,
            config_branch=branch,
        )
