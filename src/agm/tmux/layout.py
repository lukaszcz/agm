"""Tmux layout calculation and application."""

from __future__ import annotations

from agm.core.process import require_capture, require_success


def layout_checksum(layout: str) -> str:
    """Return the tmux layout checksum for *layout*."""

    checksum = 0
    for char in layout:
        checksum = ((checksum >> 1) + ((checksum & 1) << 15) + ord(char)) & 0xFFFF
    return f"{checksum:04x}"


def build_row_layout(
    width: int,
    height: int,
    x: int,
    y: int,
    start_index: int,
    pane_total: int,
) -> str:
    """Build one row of panes."""

    if pane_total == 1:
        return f"{width}x{height},{x},{y},{start_index}"

    base_width = (width - (pane_total - 1)) // pane_total
    current_x = x
    children: list[str] = []
    for pane_offset in range(pane_total):
        current_width = base_width if pane_offset + 1 < pane_total else x + width - current_x
        children.append(
            f"{current_width}x{height},{current_x},{y},{start_index + pane_offset}",
        )
        current_x += current_width + 1
    return f"{width}x{height},{x},{y}" + "{" + ",".join(children) + "}"


def build_window_layout(
    width: int,
    height: int,
    pane_total: int,
    cols: int,
    rows: int,
) -> str:
    """Build the full tmux window layout."""

    if rows == 1:
        return build_row_layout(width, height, 0, 0, 0, pane_total)

    base_height = (height - (rows - 1)) // rows
    current_y = 0
    next_index = 0
    children: list[str] = []
    for row in range(rows):
        row_panes = pane_total - row * cols
        if row_panes > cols:
            row_panes = cols
        current_height = base_height if row + 1 < rows else height - current_y
        children.append(
            build_row_layout(width, current_height, 0, current_y, next_index, row_panes)
        )
        next_index += row_panes
        current_y += current_height + 1
    return f"{width}x{height},0,0[" + ",".join(children) + "]"


def layout_for_window(pane_count: int, width: int, height: int) -> str:
    """Return the complete tmux layout string including checksum."""

    rows = 1
    while (rows + 1) * (rows + 1) <= pane_count:
        rows += 1
    cols = (pane_count + rows - 1) // rows
    layout_body = build_window_layout(width, height, pane_count, cols, rows)
    return f"{layout_checksum(layout_body)},{layout_body}"


def tmux_display(format_string: str, *, target: str | None = None) -> str:
    """Return one tmux format value."""

    cmd = ["tmux", "display-message", "-p"]
    if target is not None:
        cmd.extend(["-t", target])
    cmd.append(format_string)
    return require_capture(cmd).strip()


def resolve_window_layout_target(window_id: str | None = None) -> tuple[str, int, int]:
    """Return the target window id and current dimensions in cells."""

    resolved_window_id = window_id or tmux_display("#{window_id}")
    width = int(tmux_display("#{window_width}", target=resolved_window_id))
    height = int(tmux_display("#{window_height}", target=resolved_window_id))
    return resolved_window_id, width, height


def apply_layout(*, pane_count: int, window_id: str, width: int, height: int) -> None:
    """Apply a custom layout to *window_id*."""

    require_success(
        ["tmux", "select-layout", "-t", window_id, layout_for_window(pane_count, width, height)],
    )
