"""agm pm open."""

from __future__ import annotations

import argparse

from agm.commands.pm.common import open_session


def run(args: argparse.Namespace) -> None:
    open_session(pane_count=args.pane_count, branch=args.branch)
