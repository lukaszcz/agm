"""Revise agent command."""

from agm.agent.review import revise_once
from agm.cli_support.args import ReviseArgs


def run(args: ReviseArgs) -> None:
    try:
        revise_once(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
