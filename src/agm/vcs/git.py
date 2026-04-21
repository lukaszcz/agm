"""Git helpers shared across commands."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from agm.core.process import require_capture, require_success, run_capture, run_foreground


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

    return run_capture(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
    )[0] == 0


def git_setup(cwd: Path | None = None) -> Path:
    """Detect the relevant git repository or worktree root."""

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


def current_branch(repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
    """Return the current branch name."""

    return require_capture(
        [*_git_args(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        env=env,
    ).strip()


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
        args.extend(["-b", branch])
    args.append(str(path))
    if create and start_point is not None:
        args.append(start_point)
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


def worktree_list(repo_dir: Path, *, env: dict[str, str] | None = None) -> list[WorktreeInfo]:
    """Parse git worktree list --porcelain."""

    output = require_capture([*_git_args(repo_dir), "worktree", "list", "--porcelain"], env=env)
    worktrees: list[WorktreeInfo] = []
    path: Path | None = None
    branch: str | None = None
    for line in output.splitlines():
        if not line:
            if path is not None:
                worktrees.append(WorktreeInfo(path=path, branch=branch))
            path = None
            branch = None
            continue
        if line.startswith("worktree "):
            path = Path(line.removeprefix("worktree "))
        elif line.startswith("branch "):
            ref = line.removeprefix("branch ")
            branch = ref.removeprefix("refs/heads/")
    if path is not None:
        worktrees.append(WorktreeInfo(path=path, branch=branch))
    return worktrees


def branch_delete(repo_dir: Path, branch: str, *, env: dict[str, str] | None = None) -> None:
    """Delete a local branch."""

    require_success([*_git_args(repo_dir), "branch", "-d", branch], env=env)


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
