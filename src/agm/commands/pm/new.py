"""agm pm new."""

from __future__ import annotations

from agm.commands.args import OpenArgs
from agm.commands.pm.common import new_session


def run(args: OpenArgs) -> None:
    new_session(
        detached=args.detached,
        pane_count=args.pane_count,
        parent=args.parent,
        branch=args.branch,
    )
