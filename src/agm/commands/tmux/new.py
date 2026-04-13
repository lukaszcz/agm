"""agm tmux new."""

from __future__ import annotations

import argparse

from agm.tmux.session import create_tmux_session


def run(args: argparse.Namespace) -> None:
    create_tmux_session(
        detach=args.detach,
        pane_count=args.pane_count,
        session_name=args.session_name,
    )
