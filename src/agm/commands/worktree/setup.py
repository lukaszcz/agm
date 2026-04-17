"""agm worktree setup."""

from __future__ import annotations

from agm.commands.args import WorktreeSetupArgs
from agm.project.setup import run_setup


def run(args: WorktreeSetupArgs) -> None:
    del args
    run_setup()
