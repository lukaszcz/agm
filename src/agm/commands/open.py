"""agm open."""

from __future__ import annotations

from agm.commands.args import OpenArgs
from agm.utils.project_session import smart_open_session


def run(args: OpenArgs) -> None:
    smart_open_session(
        detached=args.detached,
        pane_count=args.pane_count,
        parent=args.parent,
        branch=args.branch,
    )
