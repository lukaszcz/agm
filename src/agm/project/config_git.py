"""Config directory git commit helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.fs import rglob
from agm.core.process import require_success, run_capture
from agm.project.layout import project_config_dir


def _add_paths(config_git_root: Path, paths: list[Path], *, env: dict[str, str] | None) -> None:
    """Stage changes at *paths* using ``git add -A``, ignoring missing pathspecs.

    When a path was never tracked and has been deleted, ``git add -A``
    fails with "pathspec did not match any files".  This is harmless
    and silently ignored.
    """
    relative_paths = [str(p.resolve().relative_to(config_git_root.resolve())) for p in paths]
    returncode, _stdout, stderr = run_capture(
        ["git", "-C", str(config_git_root), "add", "-A", "--", *relative_paths],
        env=env,
    )
    if returncode != 0:
        # git add fails with "pathspec did not match any files" when the
        # path was never tracked and has been deleted – that is harmless.
        if "did not match any files" not in stderr:
            require_success(
                ["git", "-C", str(config_git_root), "add", "-A", "--", *relative_paths],
                env=env,
            )


def commit_config_dir_changes(
    project_dir: Path,
    message: str,
    *,
    add_paths: Sequence[Path] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Add and commit changes in the project config directory.

    If the config directory is a git repository, stages the changes
    and creates a commit with the given *message*.  Silently does
    nothing when the config directory is not a git repository or when
    there are no changes to commit.

    When *add_paths* is provided, only changes within those paths
    (relative to the config directory git root) are staged using
    ``git add -A``.  This covers new files, modifications, and
    deletions inside the specified paths.  If a path was never
    tracked by git and has been removed, the error is silently
    ignored.

    When *add_paths* is ``None``, all modifications to tracked files
    are staged (via ``git add -u``) plus any new ``config.toml``
    files discovered in the config directory.
    """
    config_dir = project_config_dir(project_dir)
    config_git_root = git_helpers.exact_repo_root(config_dir, env=env)
    if config_git_root is None:
        return
    if add_paths is not None:
        if not add_paths:
            return
        _add_paths(config_git_root, list(add_paths), env=env)
    else:
        # Stage modifications and deletions of tracked files.
        require_success(
            ["git", "-C", str(config_git_root), "add", "-u"],
            env=env,
        )
        # Also add any new config.toml files.
        config_toml_files = sorted(
            path for path in rglob(config_dir, "config.toml") if path.is_file()
        )
        if config_toml_files:
            relative_toml = [
                str(p.resolve().relative_to(config_git_root.resolve())) for p in config_toml_files
            ]
            require_success(
                ["git", "-C", str(config_git_root), "add", "--", *relative_toml],
                env=env,
            )
    if not git_helpers.has_staged_changes(config_git_root, [Path(".")], env=env):
        return
    require_success(
        ["git", "-C", str(config_git_root), "commit", "-m", message],
        env=env,
    )
