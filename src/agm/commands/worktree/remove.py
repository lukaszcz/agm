"""agm worktree remove."""

from __future__ import annotations

from agm.commands.args import WorktreeRemoveArgs
from agm.project.worktree import remove_worktree


def run(args: WorktreeRemoveArgs) -> None:
    remove_worktree(force=args.force, branch=args.branch)
