"""agm dep rm."""

from __future__ import annotations

import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.cli_support.args import DepRemoveArgs
from agm.core.fs import is_dir, rmtree
from agm.project.dependency_checkout import main_dep_repo
from agm.project.layout import project_deps_dir, require_current_project_dir


def _parse_target(target: str, *, remove_all: bool) -> tuple[str, str | None]:
    dep, sep, ref = target.partition("/")
    if not dep:
        print(f"error: invalid dependency target: {target}", file=sys.stderr)
        raise SystemExit(1)
    if remove_all:
        if sep:
            print("error: --all expects DEP, not DEP/BRANCH", file=sys.stderr)
            raise SystemExit(1)
        return dep, None
    if not sep or not ref:
        print("error: expected DEP/BRANCH, or use --all DEP", file=sys.stderr)
        raise SystemExit(1)
    return dep, ref


def _linked_worktrees(
    *,
    repo_path: Path,
) -> list[git_helpers.WorktreeInfo]:
    resolved_repo_path = repo_path.resolve(strict=False)
    return [
        worktree
        for worktree in git_helpers.worktree_list(repo_path)
        if worktree.path.resolve(strict=False) != resolved_repo_path
    ]


def _remove_dep_dir(dep_dir: Path) -> None:
    rmtree(dep_dir)


def _worktree_at_path(
    worktrees: list[git_helpers.WorktreeInfo],
    path: Path,
) -> git_helpers.WorktreeInfo | None:
    resolved_path = path.resolve(strict=False)
    for worktree in worktrees:
        if worktree.path.resolve(strict=False) == resolved_path:
            return worktree
    return None


def _worktree_for_branch(
    worktrees: list[git_helpers.WorktreeInfo],
    branch: str,
) -> git_helpers.WorktreeInfo | None:
    for worktree in worktrees:
        if worktree.branch == branch:
            return worktree
    return None


def _remove_dep_worktree_by_path(
    *,
    repo_path: Path,
    worktree: git_helpers.WorktreeInfo,
) -> None:
    git_helpers.worktree_remove(repo_path, worktree.path)
    print(f"Removed dependency worktree: {worktree.path}")
    if worktree.branch is not None:
        git_helpers.branch_delete(repo_path, worktree.branch)


def run(args: DepRemoveArgs) -> None:
    dep, ref = _parse_target(args.target, remove_all=args.all)
    project_dir = require_current_project_dir()
    dep_dir = project_deps_dir(project_dir) / dep
    if not is_dir(dep_dir):
        print(f"error: deps/{dep} does not exist", file=sys.stderr)
        raise SystemExit(1)

    repo_path = main_dep_repo(dep_dir)
    linked_worktrees = _linked_worktrees(repo_path=repo_path)
    if args.all:
        for worktree in linked_worktrees:
            if worktree.branch is None:
                print(
                    f"error: cannot remove detached dependency worktree at {worktree.path}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            _remove_dep_worktree_by_path(repo_path=repo_path, worktree=worktree)
        _remove_dep_dir(dep_dir)
        return

    assert ref is not None
    target_path = dep_dir / ref
    if ref == "repo" or target_path.resolve(strict=False) == repo_path.resolve(strict=False):
        if linked_worktrees:
            print(
                f"error: cannot remove deps/{dep} while other worktrees exist",
                file=sys.stderr,
            )
            raise SystemExit(1)
        _remove_dep_dir(dep_dir)
        return

    selected_worktree = _worktree_at_path(linked_worktrees, target_path)
    if selected_worktree is None:
        selected_worktree = _worktree_for_branch(linked_worktrees, ref)
    if selected_worktree is None:
        print(f"error: dependency worktree does not exist: deps/{dep}/{ref}", file=sys.stderr)
        raise SystemExit(1)
    _remove_dep_worktree_by_path(repo_path=repo_path, worktree=selected_worktree)
