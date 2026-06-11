"""Stub for lark.tree.Meta."""

from __future__ import annotations

class Meta:
    """Position metadata attached to a Tree node by propagate_positions=True."""

    line: int
    column: int
    end_line: int
    end_column: int
    start_pos: int
    end_pos: int
    empty: bool
