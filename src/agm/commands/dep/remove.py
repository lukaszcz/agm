"""agm dep rm."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import agm.vcs.git as git_helpers
from agm.commands.args import DepRemoveArgs
from agm.commands.dep.common import main_dep_repo
from agm.project.layout import current_project_dir, is_main_checkout_branch
from agm.project.worktree import remove_worktree_from_repo


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
    shutil.rmtree(dep_dir)


def run(args: DepRemoveArgs) -> None:
    dep, ref = _parse_target(args.target, remove_all=args.all)
    project_dir = current_project_dir()
    dep_dir = project_dir / "deps" / dep
    if not dep_dir.is_dir():
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
            remove_worktree_from_repo(repo_dir=repo_path, force=False, branch=worktree.branch)
        _remove_dep_dir(dep_dir)
        return

    assert ref is not None
    main_branch = git_helpers.current_branch(repo_path)
    if is_main_checkout_branch(dep_dir, ref, repo_branch=main_branch):
        if linked_worktrees:
            print(
                f"error: cannot remove deps/{dep} while other worktrees exist",
                file=sys.stderr,
            )
            raise SystemExit(1)
        _remove_dep_dir(dep_dir)
        return

    remove_worktree_from_repo(repo_dir=repo_path, force=False, branch=ref)
