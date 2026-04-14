"""agm worktree new."""

from __future__ import annotations

from agm.commands.args import WorktreeNewArgs
from agm.utils.worktree import ensure_worktree


def run(args: WorktreeNewArgs) -> None:
    ensure_worktree(
        new_branch=args.branch,
        worktrees_dir=args.worktrees_dir,
        branch=None,
        existing_ok=False,
    )
