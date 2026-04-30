"""Dependency environment variable management."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import cast

import agm.vcs.git as git_helpers
from agm.core.fs import exists, is_dir, iterdir, mkdir, read_text, rglob, write_text
from agm.project.layout import project_config_dir, project_deps_dir, project_repo_dir

TomlDict = dict[str, object]


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


def _toml_dict(value: object) -> TomlDict:
    if isinstance(value, dict):
        return cast(TomlDict, value)
    return {}


def _load_toml_file(path: Path) -> TomlDict:
    with path.open("rb") as handle:
        loaded = cast(object, tomllib.load(handle))
    return _toml_dict(loaded)


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
        deps_table = _toml_dict(_load_toml_file(config_file).get("deps"))
        for dep_name, dep_branch in deps_table.items():
            if not isinstance(dep_branch, str) or not dep_branch:
                continue
            dep_path = deps_dir / dep_name / dep_branch
            resolved_env[dep_env_var_name(dep_name)] = str(dep_path)
    return resolved_env


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


def _toml_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key)


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _is_table_header(line: str) -> bool:
    return re.match(r"^\s*\[[^\]]+\]\s*(?:#.*)?$", line) is not None


def _is_deps_table_header(line: str) -> bool:
    return re.match(r"^\s*\[deps\]\s*(?:#.*)?$", line) is not None


def _line_sets_toml_key(line: str, dep_name: str) -> bool:
    match: re.Match[str] | None = re.match(r'^\s*("[^"]+"|[A-Za-z0-9_-]+)\s*=', line)
    if match is None:
        return False
    raw_key: str = match.group(1)
    if not raw_key.startswith('"'):
        return raw_key == dep_name
    try:
        loaded_key = cast(object, json.loads(raw_key))
    except json.JSONDecodeError:
        return False
    return isinstance(loaded_key, str) and loaded_key == dep_name


def _set_toml_deps_value(content: str, dep_name: str, dep_branch: str) -> str:
    lines = content.splitlines(keepends=True)
    key = _toml_key(dep_name)
    setting = f"{key} = {_toml_string(dep_branch)}\n"
    deps_start: int | None = None
    deps_end = len(lines)
    for index, line in enumerate(lines):
        if not _is_deps_table_header(line):
            continue
        deps_start = index
        for end_index in range(index + 1, len(lines)):
            if _is_table_header(lines[end_index]):
                deps_end = end_index
                break
        break

    if deps_start is None:
        prefix = "" if not content or content.endswith("\n") else "\n"
        separator = "" if not content else "\n"
        return f"{content}{prefix}{separator}[deps]\n{setting}"

    for index in range(deps_start + 1, deps_end):
        if _line_sets_toml_key(lines[index], dep_name):
            lines[index] = setting
            return "".join(lines)

    lines.insert(deps_end, setting)
    return "".join(lines)


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


def update_main_dependency_configs(project_dir: Path) -> None:
    """Create/update the main config TOML with known dependency branches."""

    deps_dir = project_deps_dir(project_dir)
    if not is_dir(deps_dir):
        return
    for dep_dir in sorted(path for path in iterdir(deps_dir) if is_dir(path)):
        branch = _main_dependency_branch(dep_dir)
        if branch is None:
            continue
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name=dep_dir.name,
            dep_branch=branch,
            config_branch=None,
        )


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
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name=dep_dir.name,
            dep_branch=branch,
            config_branch=branch,
        )
