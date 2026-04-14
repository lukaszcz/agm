"""agm pm open."""

from __future__ import annotations

from agm.commands.args import OpenArgs
from agm.commands.pm.common import open_session


def run(args: OpenArgs) -> None:
    open_session(detached=args.detached, pane_count=args.pane_count, branch=args.branch)
