"""agm worktree checkout."""

from __future__ import annotations

import argparse

from agm.utils.worktree import ensure_worktree

_USAGE = "usage: agm worktree checkout [-b branch-name] [-d dir] [branch-name]"


def run(args: argparse.Namespace) -> None:
    branch_name = args.new_branch if args.new_branch is not None else args.branch
    if branch_name is None:
        print(_USAGE)
        raise SystemExit(1)
    ensure_worktree(
        new_branch=args.new_branch,
        worktrees_dir=args.worktrees_dir,
        branch=args.branch,
        existing_ok=args.new_branch is None,
    )
