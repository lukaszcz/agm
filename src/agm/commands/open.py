"""agm open."""

from __future__ import annotations

import argparse

from agm.commands.pm.common import smart_open_session


def run(args: argparse.Namespace) -> None:
    smart_open_session(
        pane_count=args.pane_count,
        parent=args.parent,
        branch=args.branch,
    )
