"""agm worktree remove."""

from __future__ import annotations

import argparse

from agm.utils.worktree import remove_worktree


def run(args: argparse.Namespace) -> None:
    remove_worktree(force=args.force, branch=args.branch)
