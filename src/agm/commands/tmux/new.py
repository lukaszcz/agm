"""agm tmux new."""

from __future__ import annotations

from agm.commands.args import TmuxNewArgs
from agm.tmux.session import create_tmux_session


def run(args: TmuxNewArgs) -> None:
    create_tmux_session(
        detach=args.detach,
        pane_count=args.pane_count,
        session_name=args.session_name,
    )
