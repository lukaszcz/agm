"""agm tmux layout."""

from __future__ import annotations

from agm.commands.args import TmuxLayoutArgs
from agm.tmux.layout import apply_layout


def run(args: TmuxLayoutArgs) -> None:
    apply_layout(
        pane_count=int(args.pane_count),
        window_id=args.window_id,
        width=int(args.width),
        height=int(args.height),
    )
