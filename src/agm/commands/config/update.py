"""agm config update."""

from __future__ import annotations

import os
from pathlib import Path

from agm.commands.args import ConfigUpdateArgs
from agm.core.fs import rglob
from agm.core.process import exit_with_output, require_success, run_capture
from agm.project.dependency_env import update_all_project_dependency_configs
from agm.project.layout import project_config_dir, require_current_project_dir


def _config_git_root(config_dir: Path, *, env: dict[str, str]) -> Path | None:
    if not config_dir.exists():
        return None
    returncode, stdout, _stderr = run_capture(
        ["git", "-C", str(config_dir), "rev-parse", "--show-toplevel"],
        env=env,
    )
    if returncode != 0:
        return None
    root = Path(stdout.strip()).resolve()
    if root != config_dir.resolve():
        return None
    return root


def _has_staged_changes(repo_dir: Path, paths: list[Path], *, env: dict[str, str]) -> bool:
    returncode, stdout, stderr = run_capture(
        ["git", "-C", str(repo_dir), "diff", "--cached", "--quiet", "--", *map(str, paths)],
        env=env,
    )
    if returncode not in {0, 1}:
        exit_with_output(returncode, stdout, stderr)
    return returncode == 1


def _commit_generated_configs(project_dir: Path, *, env: dict[str, str]) -> None:
    config_dir = project_config_dir(project_dir)
    config_git_root = _config_git_root(config_dir, env=env)
    if config_git_root is None:
        return

    config_files = sorted(path for path in rglob(config_dir, "config.toml") if path.is_file())
    if not config_files:
        return
    relative_config_files = [
        path.resolve().relative_to(config_git_root.resolve()) for path in config_files
    ]
    require_success(
        ["git", "-C", str(config_git_root), "add", "--", *map(str, relative_config_files)],
        env=env,
    )
    if not _has_staged_changes(config_git_root, relative_config_files, env=env):
        return
    require_success(
        ["git", "-C", str(config_git_root), "commit", "-m", "chore: update config"],
        env=env,
    )


def run(args: ConfigUpdateArgs) -> None:
    del args
    env = dict(os.environ)
    project_dir = require_current_project_dir()
    update_all_project_dependency_configs(project_dir, env=env)
    _commit_generated_configs(project_dir, env=env)
