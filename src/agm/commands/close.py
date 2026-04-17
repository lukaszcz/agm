"""agm close."""

from __future__ import annotations

from agm.commands.args import CloseArgs
from agm.utils.project_session import close_session


def run(args: CloseArgs) -> None:
    close_session(branch=args.branch)
