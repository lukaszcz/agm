"""agm tmux close."""

from __future__ import annotations

from agm.commands.args import TmuxCloseArgs
from agm.tmux.session import kill_tmux_session


def run(args: TmuxCloseArgs) -> None:
    raise SystemExit(kill_tmux_session(session_name=args.session_name))
