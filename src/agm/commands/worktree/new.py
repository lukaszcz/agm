"""agm worktree new."""

from __future__ import annotations

import argparse

from agm.utils.worktree import ensure_worktree


def run(args: argparse.Namespace) -> None:
    ensure_worktree(
        new_branch=args.branch,
        worktrees_dir=args.worktrees_dir,
        branch=None,
        existing_ok=False,
    )
