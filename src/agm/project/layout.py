"""Project detection and configuration helpers."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.core.dotenv import set_dotenv_value
from agm.core.env import load_config_dotenv_files, resolve_env
from agm.core.process import require_success
from agm.core.toml import TomlDict, load_toml_file, toml_dict


@dataclass(frozen=True)
class CurrentWorkspace:
    """Describes the currently active AGM workspace."""

    workspace_dir: Path
    branch: str | None
    is_main: bool


_DOTENV_CONFIG_FILES = frozenset({".env", ".env.local"})
_DOT_CONFIG_COPY_EXCLUDES = frozenset({".git"})


def _path_name(path: Path) -> str:
    return path.name


def _copy_existing_config_files(
    source_dir: Path,
    target_dir: Path,
    *,
    include_dotenv: bool = True,
) -> None:
    existing_paths = [
        str(path)
        for path in sorted(source_dir.iterdir(), key=_path_name)
        if path.name.startswith(".")
        and path.name not in _DOT_CONFIG_COPY_EXCLUDES
        and (include_dotenv or path.name not in _DOTENV_CONFIG_FILES)
    ]
    if not existing_paths:
        return
    require_success(["cp", "-r", *existing_paths, str(target_dir)])


def _merge_config_dotenv_files(config_dirs: list[Path], target_dir: Path) -> None:
    merged_env = load_config_dotenv_files(config_dirs, env={})
    if not merged_env:
        return
    target_env = target_dir / ".env.local"
    if target_env.exists():
        target_env.unlink()
    for key, value in sorted(merged_env.items()):
        set_dotenv_value(target_env, key, value)


def _merge_branch_env_file(source_dir: Path, target_dir: Path) -> None:
    merged_env = load_config_dotenv_files([source_dir], env={})
    if not merged_env:
        return
    target_env = target_dir / ".env"
    for key, value in sorted(merged_env.items()):
        set_dotenv_value(target_env, key, value)


def _resolved_cwd(cwd: Path | None = None) -> Path:
    return Path.cwd() if cwd is None else cwd.resolve()


def _project_dir_from_workspace(workspace_dir: Path) -> Path | None:
    if (workspace_dir / ".agm").is_dir():
        return workspace_dir / ".agm"
    if (workspace_dir / "repo").is_dir():
        return workspace_dir
    if workspace_dir.name == "repo" and (
        (workspace_dir.parent / "worktrees").is_dir()
        or (workspace_dir.parent / ".worktrees").is_dir()
    ):
        return workspace_dir.parent
    if workspace_dir.parent.name == ".worktrees":
        return workspace_dir.parent.parent
    if workspace_dir.parent.name == "worktrees" and workspace_dir.parent.parent.name == ".agm":
        return workspace_dir.parent.parent
    if workspace_dir.parent.name == "worktrees" and (workspace_dir.parent.parent / "repo").is_dir():
        return workspace_dir.parent.parent
    return None


def _project_dir_from_env(env: Mapping[str, str] | None = None) -> Path | None:
    resolved_env = resolve_env(env)
    raw_project_dir = resolved_env.get("PROJ_DIR")
    if not raw_project_dir:
        return None
    return Path(raw_project_dir)


def _valid_project_dir_from_cwd(cwd: Path) -> Path | None:
    for candidate in (cwd, *cwd.parents):
        project_dir = _project_dir_from_workspace(candidate)
        if project_dir is not None and is_project_dir(project_dir):
            return project_dir
    return None


def current_workspace_or_project_root(
    cwd: Path | None = None, *, env: Mapping[str, str] | None = None
) -> Path:
    """Return the current AGM project, Git checkout root, or current directory."""

    current = _resolved_cwd(cwd)
    cwd_project_dir = _valid_project_dir_from_cwd(current)
    if cwd_project_dir is not None:
        return cwd_project_dir

    env_project_dir = _project_dir_from_env(env)
    if env_project_dir is not None:
        return env_project_dir

    for candidate in (current, *current.parents):
        project_dir = _project_dir_from_workspace(candidate)
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

    current = _resolved_cwd(cwd)
    cwd_project_dir = _valid_project_dir_from_cwd(current)
    if cwd_project_dir is not None:
        return cwd_project_dir

    env_project_dir = _project_dir_from_env(env)
    if env_project_dir is not None:
        return env_project_dir

    candidate = current_workspace_or_project_root(current, env=env)
    return candidate if is_project_dir(candidate) else None


def is_split_project(project_dir: Path) -> bool:
    """Return whether *project_dir* uses the split layout."""

    return (project_dir / "repo").is_dir()


def is_embedded_project(project_dir: Path) -> bool:
    """Return whether *project_dir* uses the embedded layout."""

    return project_dir.name == ".agm" and git_helpers.is_git_repo(project_dir.parent)


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
            "(expected embedded layout with a git repo and .agm/, or split layout with repo/)"
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
    return require_project_dir(current_workspace_or_project_root(cwd, env=env))


def current_workspace(
    project_dir: Path,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> CurrentWorkspace | None:
    """Return the current AGM workspace within *project_dir*.

    Prefers the ``REPO_DIR`` environment variable when it points to a git
    checkout (main or branch workspace) inside *project_dir*. Falls back to
    detecting the workspace from *cwd*. Returns ``None`` when *cwd* is not inside
    *project_dir* and no usable ``REPO_DIR`` override is available.
    """
    resolved_env = resolve_env(env)
    resolved_project_dir = project_dir.resolve(strict=False)
    repo_dir = project_repo_dir(project_dir).resolve(strict=False)

    # --- Try REPO_DIR env var first ---
    workspace_dir: Path | None = None
    repo_dir_var = resolved_env.get("REPO_DIR", "").strip()
    if repo_dir_var:
        candidate = Path(repo_dir_var).resolve(strict=False)
        # For embedded layout the main repo is the parent of the .agm project
        # dir, so it is not *inside* project_dir.  Accept it when it matches the
        # known repo_dir, or when it falls inside either project_dir or repo_dir.
        if git_helpers.is_git_repo(candidate) and (
            candidate == repo_dir
            or repo_dir in candidate.parents
            or candidate == resolved_project_dir
            or resolved_project_dir in candidate.parents
        ):
            workspace_dir = candidate

    # --- Fall back to cwd-based detection ---
    if workspace_dir is None:
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
                workspace_dir = repo_dir
            else:
                return None
        else:
            try:
                workspace_dir = git_helpers.checkout_root(current).resolve(strict=False)
            except SystemExit:
                if git_helpers.is_git_repo(repo_dir):
                    workspace_dir = repo_dir
                else:
                    workspace_dir = current

    # --- Determine workspace branch / is_main ---
    if workspace_dir == repo_dir or repo_dir in workspace_dir.parents:
        return CurrentWorkspace(workspace_dir=workspace_dir, branch=None, is_main=True)

    branch = git_helpers.current_branch(workspace_dir, env=env)
    return CurrentWorkspace(workspace_dir=workspace_dir, branch=branch, is_main=False)


def project_root(project_dir: Path) -> Path:
    """Return the top-level directory of the AGM project.

    For embedded layout this is the git repository directory (``.agm``'s
    parent); for split layout it is the project root directory (same
    as *project_dir*).
    """
    if is_embedded_project(project_dir):
        return project_dir.parent
    return project_dir


def _load_project_config_toml(project_dir: Path) -> TomlDict:
    config_file = project_config_dir(project_dir) / "config.toml"
    if not config_file.is_file():
        return {}
    return load_toml_file(config_file)


def project_name(project_dir: Path) -> str:
    """Return the human-readable project name.

    Reads ``[project].name`` from ``config.toml`` when present; otherwise
    falls back to the directory name of the project root.
    """
    config = _load_project_config_toml(project_dir)
    project_table = toml_dict(config.get("project"))
    name_value = project_table.get("name")
    if isinstance(name_value, str) and name_value:
        return name_value
    return project_root(project_dir).name


def _tmux_session_name(name: str) -> str:
    """Return *name* normalized the same way tmux stores session names."""

    return name.replace(".", "_").replace(":", "_")


def project_repo_dir(project_dir: Path) -> Path:
    """Return the main repository directory for *project_dir*."""

    if is_split_project(project_dir):
        return project_dir / "repo"
    if is_embedded_project(project_dir):
        return project_dir.parent
    return project_dir


def main_repo_dir(project_dir: Path) -> Path:
    """Backward-compatible alias for ``project_repo_dir``."""

    return project_repo_dir(project_dir)


def default_worktrees_dir(project_dir: Path) -> Path:
    """Return the default worktrees directory for *project_dir*."""

    return project_dir / "worktrees"


def project_config_dir(project_dir: Path) -> Path:
    """Return the shared project config directory."""

    return project_dir / "config"


def project_deps_dir(project_dir: Path) -> Path:
    """Return the dependency checkout directory."""

    return project_dir / "deps"


def project_notes_dir(project_dir: Path) -> Path:
    """Return the project notes directory."""

    return project_dir / "notes"


def is_main_workspace_branch(project_dir: Path, branch: str, *, repo_branch: str) -> bool:
    """Return whether *branch* resolves to the main workspace."""

    return branch in {"repo", repo_branch}


def branch_worktree_path(project_dir: Path, branch: str, *, repo_branch: str) -> Path:
    """Return the workspace path corresponding to *branch*."""

    if is_main_workspace_branch(project_dir, branch, repo_branch=repo_branch):
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
    """Return the parent branch name for config seeding, or None for the main workspace."""

    repo_dir = project_repo_dir(project_dir)
    repo_branch = git_helpers.current_branch(repo_dir)
    resolved_parent = parent or repo_branch
    if is_main_workspace_branch(project_dir, resolved_parent, repo_branch=repo_branch):
        return None
    return resolved_parent


def branch_session_name(project_dir: Path, branch: str) -> str:
    """Return the tmux session name corresponding to *branch*."""

    name = _tmux_session_name(project_name(project_dir))

    if branch == "repo":
        return name

    repo_branch = git_helpers.current_branch(project_repo_dir(project_dir))
    if is_main_workspace_branch(project_dir, branch, repo_branch=repo_branch):
        return name
    return f"{name}/{_tmux_session_name(branch)}"


def exit_if_main_workspace_branch(project_dir: Path, branch: str, *, repo_branch: str) -> None:
    """Exit when *branch* resolves to the main workspace."""

    if not is_main_workspace_branch(project_dir, branch, repo_branch=repo_branch):
        return
    print(
        (
            f"error: '{branch}' resolves to the main workspace at "
            f"{project_repo_dir(project_dir)} and cannot be managed as a branch workspace"
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
        resolved_branch = branch
        if resolved_branch is None:
            workspace = current_workspace(proj_dir, cwd=current)
            if workspace is not None:
                resolved_branch = workspace.branch
        if resolved_branch is None:
            _copy_existing_config_files(config_dir, resolved_target)
            return
        workspace_config_dir = config_dir / resolved_branch
        _copy_existing_config_files(config_dir, resolved_target, include_dotenv=False)
        if (config_dir / ".env").exists():
            require_success(["cp", "-r", str(config_dir / ".env"), str(resolved_target)])
        if workspace_config_dir.is_dir():
            _copy_existing_config_files(workspace_config_dir, resolved_target, include_dotenv=False)
        _merge_config_dotenv_files([config_dir, workspace_config_dir], resolved_target)
