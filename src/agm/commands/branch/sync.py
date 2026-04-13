"""agm branch sync."""

from __future__ import annotations

import argparse

from agm.utils.worktree import branch_sync


def run(args: argparse.Namespace) -> None:
    del args
    branch_sync()
