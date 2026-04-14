"""agm worktree setup."""

from __future__ import annotations

from agm.commands.args import WorktreeSetupArgs
from agm.utils.worktree import run_setup


def run(args: WorktreeSetupArgs) -> None:
    del args
    run_setup()
