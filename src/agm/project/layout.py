"""Project detection and configuration helpers."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.dotenv import set_dotenv_value
from agm.core.env import load_dotenv_file
from agm.core.process import require_success


@dataclass(frozen=True)
class CurrentCheckout:
    """Describes the currently active worktree checkout."""

    checkout_dir: Path
    branch: str | None
    is_main: bool

CONFIG_FILES: list[str] = [
    ".setup.sh",
    ".env",
    ".env.local",
    ".config",
    ".agents",
    ".opencode",
    ".codex",
    ".claude",
    ".pi",
    ".mcp.json",
]


def _copy_existing_config_files(source_dir: Path, target_dir: Path) -> None:
    existing_paths = [
        str(source_dir / name)
        for name in CONFIG_FILES
        if (source_dir / name).exists()
        and not (name == ".env" and (source_dir / name).stat().st_size == 0)
    ]
    if not existing_paths:
        return
    require_success(["cp", "-r", *existing_paths, str(target_dir)])


def _merge_branch_env_file(source_dir: Path, target_dir: Path) -> None:
    source_env = source_dir / ".env"
    if not source_env.is_file():
        return
    target_env = target_dir / ".env"
    for key, value in load_dotenv_file(source_env).items():
        set_dotenv_value(target_env, key, value)


def _resolved_cwd(cwd: Path | None = None) -> Path:
    return Path.cwd() if cwd is None else cwd.resolve()


def _project_dir_from_checkout(checkout_dir: Path) -> Path | None:
    if (checkout_dir / ".agm").is_dir():
        return checkout_dir
    if (checkout_dir / "repo").is_dir():
        return checkout_dir
    if checkout_dir.name == "repo" and (
        (checkout_dir.parent / "worktrees").is_dir()
        or (checkout_dir.parent / ".worktrees").is_dir()
    ):
        return checkout_dir.parent
    if checkout_dir.parent.name == ".worktrees":
        return checkout_dir.parent.parent
    if checkout_dir.parent.name == "worktrees" and (checkout_dir.parent.parent / "repo").is_dir():
        return checkout_dir.parent.parent
    return None


def _project_dir_from_env(env: Mapping[str, str] | None = None) -> Path | None:
    resolved_env = os.environ if env is None else env
    raw_project_dir = resolved_env.get("PROJ_DIR")
    if not raw_project_dir:
        return None
    return Path(raw_project_dir)


def current_checkout_or_project_root(
    cwd: Path | None = None, *, env: Mapping[str, str] | None = None
) -> Path:
    """Return the current AGM project, git checkout root, or current directory."""

    env_project_dir = _project_dir_from_env(env)
    if env_project_dir is not None:
        return env_project_dir

    current = _resolved_cwd(cwd)

    for candidate in (current, *current.parents):
        project_dir = _project_dir_from_checkout(candidate)
        if project_dir is not None:
            return project_dir
    if not git_helpers.is_git_repo(current):
        return current
    try:
        return git_helpers.checkout_root(current)
    except SystemExit:
        return current


def discover_current_project_dir(
    cwd: Path | None = None, *, env: Mapping[str, str] | None = None
) -> Path | None:
    """Return the current valid AGM project directory, if one can be discovered."""

    env_project_dir = _project_dir_from_env(env)
    if env_project_dir is not None:
        return env_project_dir

    candidate = current_checkout_or_project_root(cwd, env=env)
    return candidate if is_project_dir(candidate) else None


def is_workspace_project(project_dir: Path) -> bool:
    """Return whether *project_dir* uses the workspace layout."""

    return (project_dir / "repo").is_dir()


def is_embedded_project(project_dir: Path) -> bool:
    """Return whether *project_dir* uses the embedded layout."""

    return (project_dir / ".agm").is_dir() and git_helpers.is_git_repo(project_dir)


def is_project_dir(project_dir: Path) -> bool:
    """Return whether *project_dir* is a valid AGM project directory."""

    return is_embedded_project(project_dir) or git_helpers.is_git_repo(project_dir / "repo")


def require_project_dir(project_dir: Path) -> Path:
    """Return *project_dir* or exit when it is not a valid AGM project."""

    resolved = project_dir.resolve()
    if is_project_dir(resolved):
        return resolved
    print(
        (
            f"error: {resolved} is not a valid AGM project directory "
            "(expected embedded layout with a git repo and .agm/, or workspace layout with repo/)"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)


def require_current_project_dir(
    cwd: Path | None = None, *, env: Mapping[str, str] | None = None
) -> Path:
    """Resolve and validate the current AGM project directory."""

    project_dir = discover_current_project_dir(cwd, env=env)
    if project_dir is not None:
        return project_dir.resolve()
    return require_project_dir(current_checkout_or_project_root(cwd, env=env))


def current_checkout(
    project_dir: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> CurrentCheckout | None:
    """Return the current worktree checkout within *project_dir*.

    Prefers the ``REPO_DIR`` environment variable when it points to a git
    checkout (main or worktree) inside *project_dir*.  Falls back to
    detecting the checkout from *cwd*.  Returns ``None`` when *cwd* is not inside
    *project_dir* and no usable ``REPO_DIR`` override is available.
    """
    resolved_env = env if env is not None else os.environ
    resolved_project_dir = project_dir.resolve(strict=False)
    repo_dir = project_repo_dir(project_dir).resolve(strict=False)

    # --- Try REPO_DIR env var first ---
    checkout_dir: Path | None = None
    repo_dir_var = resolved_env.get("REPO_DIR", "").strip()
    if repo_dir_var:
        candidate = Path(repo_dir_var).resolve(strict=False)
        # Must be inside the project and must be a git repo
        if (
            candidate == resolved_project_dir or resolved_project_dir in candidate.parents
        ) and git_helpers.is_git_repo(candidate):
            checkout_dir = candidate

    # --- Fall back to cwd-based detection ---
    if checkout_dir is None:
        current = Path.cwd() if cwd is None else cwd.resolve()
        current_project = discover_current_project_dir(current, env=resolved_env)
        if (
            current_project is None
            or current_project.resolve(strict=False) != resolved_project_dir
        ):
            return None

        if not git_helpers.is_git_repo(current):
            if (
                current.resolve(strict=False) == resolved_project_dir
                and git_helpers.is_git_repo(repo_dir)
            ):
                checkout_dir = repo_dir
            else:
                return None
        else:
            try:
                checkout_dir = git_helpers.checkout_root(current).resolve(strict=False)
            except SystemExit:
                if git_helpers.is_git_repo(repo_dir):
                    checkout_dir = repo_dir
                else:
                    checkout_dir = current

    # --- Determine branch / is_main ---
    if checkout_dir == repo_dir or repo_dir in checkout_dir.parents:
        return CurrentCheckout(checkout_dir=checkout_dir, branch=None, is_main=True)

    branch = git_helpers.current_branch(checkout_dir, env=env)
    return CurrentCheckout(checkout_dir=checkout_dir, branch=branch, is_main=False)


def project_data_dir(project_dir: Path) -> Path:
    """Return the AGM data directory for *project_dir*."""

    if (project_dir / ".agm").is_dir():
        return project_dir / ".agm"
    return project_dir


def project_repo_dir(project_dir: Path) -> Path:
    """Return the main repository directory for *project_dir*."""

    if is_workspace_project(project_dir):
        return project_dir / "repo"
    return project_dir


def main_repo_dir(project_dir: Path) -> Path:
    """Backward-compatible alias for ``project_repo_dir``."""

    return project_repo_dir(project_dir)


def default_worktrees_dir(project_dir: Path) -> Path:
    """Return the default worktrees directory for *project_dir*."""

    return project_data_dir(project_dir) / "worktrees"


def project_config_dir(project_dir: Path) -> Path:
    """Return the shared project config directory."""

    return project_data_dir(project_dir) / "config"


def project_deps_dir(project_dir: Path) -> Path:
    """Return the dependency checkout directory."""

    return project_data_dir(project_dir) / "deps"


def project_notes_dir(project_dir: Path) -> Path:
    """Return the project notes directory."""

    return project_data_dir(project_dir) / "notes"


def is_main_checkout_branch(project_dir: Path, branch: str, *, repo_branch: str) -> bool:
    """Return whether *branch* resolves to the main repo checkout."""

    return branch in {"repo", repo_branch}


def branch_worktree_path(project_dir: Path, branch: str, *, repo_branch: str) -> Path:
    """Return the checkout path corresponding to *branch*."""

    if is_main_checkout_branch(project_dir, branch, repo_branch=repo_branch):
        return project_repo_dir(project_dir)
    return default_worktrees_dir(project_dir) / branch


def expected_branch_worktree_path(project_dir: Path, branch: str) -> Path:
    """Return the resolved expected worktree path for *branch*."""

    repo_branch = git_helpers.current_branch(project_repo_dir(project_dir))
    return branch_worktree_path(
        project_dir,
        branch,
        repo_branch=repo_branch,
    ).resolve(strict=False)


def parent_config_branch(project_dir: Path, parent: str | None) -> str | None:
    """Return the parent branch name for config seeding, or None for main checkout."""

    repo_dir = project_repo_dir(project_dir)
    repo_branch = git_helpers.current_branch(repo_dir)
    resolved_parent = parent or repo_branch
    if is_main_checkout_branch(project_dir, resolved_parent, repo_branch=repo_branch):
        return None
    return resolved_parent


def branch_session_name(project_dir: Path, branch: str) -> str:
    """Return the tmux session name corresponding to *branch*."""

    if branch == "repo":
        return project_dir.name

    repo_branch = git_helpers.current_branch(project_repo_dir(project_dir))
    if is_main_checkout_branch(project_dir, branch, repo_branch=repo_branch):
        return project_dir.name
    return f"{project_dir.name}/{branch}"


def exit_if_main_checkout_branch(project_dir: Path, branch: str, *, repo_branch: str) -> None:
    """Exit when *branch* resolves to the main repo checkout."""

    if not is_main_checkout_branch(project_dir, branch, repo_branch=repo_branch):
        return
    print(
        (
            f"error: '{branch}' resolves to the main repo checkout at "
            f"{project_repo_dir(project_dir)} and cannot be managed as a branch worktree"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1)


def copy_config(
    *,
    project_dir: Path | None = None,
    target: Path,
    branch: str | None = None,
    cwd: Path | None = None,
) -> None:
    """Copy known config files from the project config directory into *target*."""

    current = _resolved_cwd(cwd)
    if project_dir is None:
        proj_dir = require_current_project_dir(current)
    else:
        proj_dir = project_dir.resolve()
    resolved_target = target if target.is_absolute() else current / target
    if not resolved_target.is_dir():
        return

    config_dir = project_config_dir(proj_dir)
    if config_dir.is_dir():
        _copy_existing_config_files(config_dir, resolved_target)
        if branch is not None:
            _merge_branch_env_file(config_dir / branch, resolved_target)
