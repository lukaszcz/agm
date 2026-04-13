"""agm pm new."""

from __future__ import annotations

import argparse

from agm.commands.pm.common import new_session


def run(args: argparse.Namespace) -> None:
    new_session(pane_count=args.pane_count, parent=args.parent, branch=args.branch)
