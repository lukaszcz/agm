"""agm tmux layout."""

from __future__ import annotations

from agm.commands.args import TmuxLayoutArgs
from agm.tmux.layout import apply_layout, resolve_window_layout_target


def run(args: TmuxLayoutArgs) -> None:
    window_id, width, height = resolve_window_layout_target(args.window_id)
    apply_layout(
        pane_count=int(args.pane_count),
        window_id=window_id,
        width=width,
        height=height,
    )
