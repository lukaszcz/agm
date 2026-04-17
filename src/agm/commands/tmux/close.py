"""agm tmux close."""

from __future__ import annotations

from agm.commands.args import TmuxCloseArgs
from agm.tmux.session import kill_tmux_session


def run(args: TmuxCloseArgs) -> None:
    status = kill_tmux_session(session_name=args.session_name)
    if status != 0:
        raise SystemExit(status)
    print(f"Closed session {args.session_name}")
