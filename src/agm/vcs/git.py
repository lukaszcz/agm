"""Git helpers shared across commands."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agm.core.path import display_path
from agm.core.process import (
    exit_with_output,
    require_capture,
    require_success,
    run_capture,
    run_foreground,
)


@dataclass(frozen=True)
class WorktreeInfo:
    path: Path
    branch: str | None


def _git_args(repo_dir: Path | None = None) -> list[str]:
    if repo_dir is None:
        return ["git"]
    return ["git", "-C", str(repo_dir)]


def is_git_repo(path: Path) -> bool:
    """Return whether *path* is inside a git work tree."""

    return (
        run_capture(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        )[0]
        == 0
    )


def checkout_root(cwd: Path | None = None) -> Path:
    """Return the git working-tree root for the checkout containing *cwd*."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    if is_git_repo(current):
        top_level = require_capture(
            ["git", "-C", str(current), "rev-parse", "--show-toplevel"],
        ).strip()
        return Path(top_level)
    repo_dir = current / "repo"
    if repo_dir.is_dir() and is_git_repo(repo_dir):
        top_level = require_capture(
            ["git", "-C", str(repo_dir), "rev-parse", "--show-toplevel"],
        ).strip()
        return Path(top_level)
    print(
        "Error: current directory is not a git repository and repo/ is not a git repository.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def containing_root(path: Path, *, env: dict[str, str] | None = None) -> Path | None:
    """Return the containing git root for *path*, or None when absent."""

    if not path.exists():
        return None
    returncode, stdout, _stderr = run_capture(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        env=env,
    )
    if returncode != 0:
        return None
    return Path(stdout.strip())


def exact_repo_root(path: Path, *, env: dict[str, str] | None = None) -> Path | None:
    """Return *path* when it is exactly a git repo root, otherwise None."""

    root = containing_root(path, env=env)
    if root is None:
        return None
    resolved_root = root.resolve()
    if resolved_root != path.resolve():
        return None
    return resolved_root


def has_staged_changes(
    repo_dir: Path,
    paths: Sequence[Path],
    *,
    env: dict[str, str] | None = None,
) -> bool:
    """Return whether any *paths* have staged changes in *repo_dir*."""

    returncode, stdout, stderr = run_capture(
        ["git", "-C", str(repo_dir), "diff", "--cached", "--quiet", "--", *map(str, paths)],
        env=env,
    )
    if returncode not in {0, 1}:
        exit_with_output(returncode, stdout, stderr)
    return returncode == 1


def git_common_dir(cwd: Path | None = None) -> Path:
    """Return the common git directory shared by the current worktree."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    common_dir = require_capture(
        ["git", "-C", str(current), "rev-parse", "--path-format=absolute", "--git-common-dir"],
    ).strip()
    return Path(common_dir)


def fetch(repo_dir: Path, *, env: dict[str, str] | None = None) -> None:
    """Run git fetch."""

    require_success([*_git_args(repo_dir), "fetch"], env=env)


def fetch_prune_all(repo_dir: Path, *, env: dict[str, str] | None = None) -> None:
    """Run git fetch --all --prune."""

    require_success([*_git_args(repo_dir), "fetch", "--all", "--prune"], env=env)


def fetch_prune_origin(repo_dir: Path, *, env: dict[str, str] | None = None) -> None:
    """Run git fetch --prune origin."""

    require_success([*_git_args(repo_dir), "fetch", "--prune", "origin"], env=env)


def merge(repo_dir: Path, *, env: dict[str, str] | None = None) -> None:
    """Run git merge using the current branch's configured upstream."""

    require_success([*_git_args(repo_dir), "merge"], env=env)


def current_branch(repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
    """Return the current branch name."""

    return require_capture(
        [*_git_args(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        env=env,
    ).strip()


def local_branches(repo_dir: Path, *, env: dict[str, str] | None = None) -> list[str]:
    """Return local branch names for *repo_dir*."""

    output = require_capture(
        [*_git_args(repo_dir), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        env=env,
    )
    return sorted(line for line in output.splitlines() if line)


def _worktree_create_start_point(
    repo_dir: Path,
    start_point: str,
    *,
    env: dict[str, str] | None = None,
) -> str:
    """Return the start point to pass when creating a worktree branch."""

    if local_branch_exists(repo_dir, start_point, env=env):
        return start_point
    if remote_branch_exists(repo_dir, start_point, env=env):
        return f"origin/{start_point}"
    return start_point


def worktree_add(
    repo_dir: Path,
    path: Path,
    branch: str,
    *,
    create: bool = False,
    start_point: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Add a git worktree."""

    args = [*_git_args(repo_dir), "worktree", "add"]
    if create:
        args.extend(["-b", branch, "--no-track"])
    args.append(str(path))
    if create and start_point is not None:
        args.append(_worktree_create_start_point(repo_dir, start_point, env=env))
    elif not create:
        args.append(branch)
    require_success(args, env=env)


def worktree_remove(
    repo_dir: Path,
    path: Path,
    *,
    force: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """Remove a git worktree."""

    args = [*_git_args(repo_dir), "worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    require_success(args, env=env)


def worktree_prune(repo_dir: Path, *, env: dict[str, str] | None = None) -> None:
    """Prune worktree registrations whose directories no longer exist."""

    require_success([*_git_args(repo_dir), "worktree", "prune"], env=env)


def worktree_list(repo_dir: Path, *, env: dict[str, str] | None = None) -> list[WorktreeInfo]:
    """Parse git worktree list --porcelain."""

    output = require_capture([*_git_args(repo_dir), "worktree", "list", "--porcelain"], env=env)
    worktrees: list[WorktreeInfo] = []
    path: Path | None = None
    branch: str | None = None
    prunable = False
    for line in output.splitlines():
        if not line:
            if path is not None and not prunable:
                worktrees.append(WorktreeInfo(path=path, branch=branch))
            path = None
            branch = None
            prunable = False
            continue
        if line.startswith("worktree "):
            path = Path(line.removeprefix("worktree "))
        elif line.startswith("branch "):
            ref = line.removeprefix("branch ")
            branch = ref.removeprefix("refs/heads/")
        elif line.startswith("prunable"):
            # git flags worktrees whose gitdir is missing/broken; their
            # directory no longer exists, so skip them rather than operate on it.
            prunable = True
    if path is not None and not prunable:
        worktrees.append(WorktreeInfo(path=path, branch=branch))
    return worktrees


def branch_delete(
    repo_dir: Path,
    branch: str,
    *,
    force: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    """Delete a local branch."""

    flag = "-D" if force else "-d"
    require_success([*_git_args(repo_dir), "branch", flag, branch], env=env)


def _branch_upstream(
    repo_dir: Path, branch: str, *, env: dict[str, str] | None = None
) -> str | None:
    """Return the upstream branch name for *branch*, or None if not set."""

    returncode, stdout, _ = run_capture(
        [*_git_args(repo_dir), "rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"],
        env=env,
    )
    if returncode != 0:
        return None
    return stdout.strip() or None


def _is_ancestor(
    repo_dir: Path,
    ancestor: str,
    descendant: str,
    *,
    env: dict[str, str] | None = None,
) -> bool:
    """Return whether *ancestor* is an ancestor of *descendant*."""

    return (
        run_foreground(
            [*_git_args(repo_dir), "merge-base", "--is-ancestor", ancestor, descendant],
            env=env,
        )
        == 0
    )


def branch_can_delete(
    repo_dir: Path,
    branch: str,
    *,
    force: bool = False,
    env: dict[str, str] | None = None,
) -> bool:
    """Return whether a local branch can be deleted with -d (or -D if *force*)."""

    if not local_branch_exists(repo_dir, branch, env=env):
        return False
    if force:
        return True
    upstream = _branch_upstream(repo_dir, branch, env=env)
    target = upstream if upstream is not None else "HEAD"
    return _is_ancestor(repo_dir, branch, target, env=env)


def local_branch_exists(repo_dir: Path, branch: str, *, env: dict[str, str] | None = None) -> bool:
    """Return whether a local branch exists."""

    return (
        run_foreground(
            [*_git_args(repo_dir), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            env=env,
        )
        == 0
    )


def remote_branch_exists(repo_dir: Path, branch: str, *, env: dict[str, str] | None = None) -> bool:
    """Return whether *origin/branch* exists locally."""

    return (
        run_foreground(
            [
                *_git_args(repo_dir),
                "show-ref",
                "--verify",
                "--quiet",
                f"refs/remotes/origin/{branch}",
            ],
            env=env,
        )
        == 0
    )


def default_remote_branch_ref(repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
    """Return the local remote-tracking ref for origin's default branch."""

    default_ref = symbolic_ref(repo_dir, "refs/remotes/origin/HEAD", env=env)
    if default_ref:
        return default_ref
    print(
        f"error: could not determine default branch for repo at {repo_dir}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def remote_unmerged_branches(
    repo_dir: Path,
    *,
    base_ref: str,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Return remote branches not merged into *base_ref*."""

    output = require_capture(
        [
            *_git_args(repo_dir),
            "for-each-ref",
            "--format=%(refname:short)",
            f"--no-merged={base_ref}",
            "refs/remotes/origin",
        ],
        env=env,
    )
    return [line for line in output.splitlines() if line]


def create_tracking_branch(
    repo_dir: Path,
    branch: str,
    remote_branch: str,
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Create a local tracking branch."""

    require_success(
        [*_git_args(repo_dir), "branch", "--track", branch, remote_branch],
        env=env,
    )


def symbolic_ref(repo_dir: Path, ref: str, *, env: dict[str, str] | None = None) -> str:
    """Resolve a symbolic ref."""

    return require_capture(
        [*_git_args(repo_dir), "symbolic-ref", "--quiet", "--short", ref], env=env
    ).strip()


def ls_remote_head(repo_url: str, *, env: dict[str, str] | None = None) -> str:
    """Return git ls-remote --symref output for HEAD."""

    return require_capture(["git", "ls-remote", "--symref", repo_url, "HEAD"], env=env)


def default_branch_from_remote(repo_url: str, *, env: dict[str, str] | None = None) -> str:
    """Return the default branch for a remote repository URL."""

    output = ls_remote_head(repo_url, env=env)
    for line in output.splitlines():
        if line.startswith("ref:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].removeprefix("refs/heads/")
    print(f"error: could not determine default branch for {repo_url}", file=sys.stderr)
    raise SystemExit(1)


def default_branch_from_repo(repo_path: Path, *, env: dict[str, str] | None = None) -> str:
    """Return the default branch for a local repository."""

    returncode, stdout, _ = run_capture(
        [
            "git",
            "-C",
            str(repo_path),
            "symbolic-ref",
            "--quiet",
            "--short",
            "refs/remotes/origin/HEAD",
        ],
        env=env,
    )
    branch = stdout.strip().removeprefix("origin/") if returncode == 0 else ""
    if branch:
        return branch
    path = display_path(repo_path)
    message = f"error: could not determine default branch for dependency repo at {path}"
    print(message, file=sys.stderr)
    raise SystemExit(1)


def repo_name_from_url(repo_url: str) -> str:
    """Derive a repository name from *repo_url*."""

    trimmed = repo_url.rstrip("/")
    name = Path(trimmed).name.removesuffix(".git")
    if name in {"", ".", "/"}:
        raise ValueError(f"could not derive repository name from repo url: {repo_url}")
    return name


def find_first_git_repo(parent_dir: Path) -> Path:
    """Find the first child directory that is a git work tree."""

    for path in sorted(candidate for candidate in parent_dir.rglob("*") if candidate.is_dir()):
        if is_git_repo(path):
            return path
    print(
        f"error: {parent_dir} must contain at least one checked out branch",
        file=sys.stderr,
    )
    raise SystemExit(1)


def fetch_output(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a git-related command and capture output."""

    return run_capture(args, cwd=cwd, env=env)
