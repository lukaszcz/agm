"""agm branch sync."""

from __future__ import annotations

from agm.utils.worktree import branch_sync


def run(args: object) -> None:
    del args
    branch_sync()
