"""Review agent command."""

from agm.agent.review import review_once
from agm.cli_support.args import ReviewArgs


def run(args: ReviewArgs) -> None:
    try:
        review_once(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
