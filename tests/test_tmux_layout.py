"""Comprehensive tests for agm.tmux.layout."""

from __future__ import annotations

import pytest

from agm.tmux.layout import (
    apply_layout,
    build_row_layout,
    build_window_layout,
    layout_checksum,
    layout_for_window,
    resolve_window_layout_target,
    tmux_display,
)

# ---------------------------------------------------------------------------
# layout_checksum
# ---------------------------------------------------------------------------


def test_layout_checksum_empty_string() -> None:
    assert layout_checksum("") == "0000"


def test_layout_checksum_single_char() -> None:
    # 'A' = 65; checksum = (0 >> 1) + ((0 & 1) << 15) + 65 = 65 => "0041"
    assert layout_checksum("A") == "0041"


def test_layout_checksum_known_value() -> None:
    # Verify a known multi-character string produces the expected 4-hex-digit result.
    # Manually compute: starting checksum=0
    # 'a'=97: (0>>1) + ((0&1)<<15) + 97 = 97
    # 'b'=98: (97>>1) + ((97&1)<<15) + 98 = 48 + 32768 + 98 = 32914 = 0x8092
    result = layout_checksum("ab")
    assert len(result) == 4
    assert result == "8092"


def test_layout_checksum_returns_four_hex_digits() -> None:
    for s in ["hello", "world", "tmux", "1234", "abcdef"]:
        result = layout_checksum(s)
        assert len(result) == 4
        assert all(c in "0123456789abcdef" for c in result)


def test_layout_checksum_is_deterministic() -> None:
    s = "some-layout-body"
    assert layout_checksum(s) == layout_checksum(s)


def test_layout_checksum_differs_for_different_inputs() -> None:
    assert layout_checksum("abc") != layout_checksum("xyz")


def test_layout_checksum_wraps_at_16_bits() -> None:
    # Produce a string long enough to cause wrap-around; just verify it stays 4 hex chars.
    long_str = "x" * 1000
    result = layout_checksum(long_str)
    assert len(result) == 4
    value = int(result, 16)
    assert 0 <= value <= 0xFFFF


# ---------------------------------------------------------------------------
# build_row_layout
# ---------------------------------------------------------------------------


def test_build_row_layout_single_pane() -> None:
    result = build_row_layout(200, 50, 0, 0, 0, 1)
    assert result == "200x50,0,0,0"


def test_build_row_layout_single_pane_nonzero_origin() -> None:
    result = build_row_layout(100, 30, 10, 5, 3, 1)
    assert result == "100x30,10,5,3"


def test_build_row_layout_two_panes_equal_width() -> None:
    # width=100, pane_total=2 => base_width=(100-1)//2=49
    # pane0: 49x50,0,0,0  pane1: gets remaining = x+width-current_x = 0+100-(49+1)=50
    result = build_row_layout(100, 50, 0, 0, 0, 2)
    assert result == "100x50,0,0{49x50,0,0,0,50x50,50,0,1}"


def test_build_row_layout_three_panes() -> None:
    # width=100, pane_total=3 => base_width=(100-2)//3=32
    # pane0: 32x20,0,0,0  current_x=33
    # pane1: 32x20,33,0,1  current_x=66
    # pane2: remaining = 0+100-66=34 => 34x20,66,0,2
    result = build_row_layout(100, 20, 0, 0, 0, 3)
    assert result == "100x20,0,0{32x20,0,0,0,32x20,33,0,1,34x20,66,0,2}"


def test_build_row_layout_panes_contain_separator_prefix() -> None:
    result = build_row_layout(80, 24, 0, 0, 0, 2)
    assert result.startswith("80x24,0,0{")
    assert result.endswith("}")


def test_build_row_layout_start_index_offset() -> None:
    # start_index=5, pane_total=2, so pane indices are 5 and 6
    result = build_row_layout(100, 50, 0, 0, 5, 2)
    assert ",5}" not in result
    assert "5," in result
    assert "6}" in result


def test_build_row_layout_nonzero_xy_origin_multi_pane() -> None:
    # x=10, y=5, width=50, pane_total=2
    # base_width=(50-1)//2=24
    # pane0: 24x10,10,5,0  current_x=35
    # pane1: x+width-current_x=10+50-35=25 => 25x10,35,5,1
    result = build_row_layout(50, 10, 10, 5, 0, 2)
    assert result == "50x10,10,5{24x10,10,5,0,25x10,35,5,1}"


def test_build_row_layout_last_pane_gets_remaining_width() -> None:
    # With width=10, pane_total=3: base_width=(10-2)//3=2
    # pane0: 2x5,0,0,0 current_x=3; pane1: 2x5,3,0,1 current_x=6
    # pane2: 0+10-6=4 => 4x5,6,0,2
    result = build_row_layout(10, 5, 0, 0, 0, 3)
    assert result == "10x5,0,0{2x5,0,0,0,2x5,3,0,1,4x5,6,0,2}"


# ---------------------------------------------------------------------------
# build_window_layout
# ---------------------------------------------------------------------------


def test_build_window_layout_single_row_single_pane() -> None:
    result = build_window_layout(200, 50, 1, 1, 1)
    assert result == "200x50,0,0,0"


def test_build_window_layout_single_row_multiple_panes() -> None:
    # rows=1 => delegates directly to build_row_layout
    result = build_window_layout(100, 24, 2, 2, 1)
    expected = build_row_layout(100, 24, 0, 0, 0, 2)
    assert result == expected


def test_build_window_layout_two_rows() -> None:
    # width=100, height=50, pane_total=4, cols=2, rows=2
    # base_height=(50-1)//2=24
    # row0: 2 panes, height=24, y=0
    # row1: remaining pane_total - 2 = 2 panes, height=50-25=25, y=25
    result = build_window_layout(100, 50, 4, 2, 2)
    assert result.startswith("100x50,0,0[")
    assert result.endswith("]")
    # Should contain two children separated by comma
    inner = result[len("100x50,0,0[") : -1]
    parts = inner.split(",", 1)
    assert len(parts) == 2


def test_build_window_layout_two_rows_structure() -> None:
    # Verify the exact structure for 2 rows, 2 cols, 4 panes, 100x50
    result = build_window_layout(100, 50, 4, 2, 2)
    row0 = build_row_layout(100, 24, 0, 0, 0, 2)
    row1 = build_row_layout(100, 25, 0, 25, 2, 2)
    assert result == f"100x50,0,0[{row0},{row1}]"


def test_build_window_layout_last_row_gets_remaining_height() -> None:
    # height=10, rows=3 => base_height=(10-2)//3=2
    # row0: y=0, height=2; row1: y=3, height=2; row2: y=6, height=10-6=4
    # pane_total=3 (1 per row), cols=1
    result = build_window_layout(80, 10, 3, 1, 3)
    assert "4x" in result or "4," in result
    # Last row must have height=4
    assert "80x4,0,6" in result


def test_build_window_layout_uneven_panes_in_last_row() -> None:
    # pane_total=3, cols=2, rows=2 => row0 has 2 panes, row1 has 1 pane
    result = build_window_layout(100, 50, 3, 2, 2)
    # row0: 2 panes (indices 0,1); row1: 1 pane (index 2)
    row0 = build_row_layout(100, 24, 0, 0, 0, 2)
    row1 = build_row_layout(100, 25, 0, 25, 2, 1)
    assert result == f"100x50,0,0[{row0},{row1}]"


def test_build_window_layout_three_rows() -> None:
    # pane_total=9, cols=3, rows=3, width=120, height=60
    # base_height=(60-2)//3=19
    # row0: y=0,h=19; row1: y=20,h=19; row2: y=40,h=60-40=20
    result = build_window_layout(120, 60, 9, 3, 3)
    assert result.startswith("120x60,0,0[")
    assert result.endswith("]")
    assert "120x20,0,40" in result  # last row gets remaining height


# ---------------------------------------------------------------------------
# layout_for_window
# ---------------------------------------------------------------------------


def test_layout_for_window_single_pane() -> None:
    result = layout_for_window(1, 200, 50)
    body = build_window_layout(200, 50, 1, 1, 1)
    expected_checksum = layout_checksum(body)
    assert result == f"{expected_checksum},{body}"


def test_layout_for_window_has_checksum_prefix() -> None:
    result = layout_for_window(4, 160, 48)
    parts = result.split(",", 1)
    assert len(parts) == 2
    checksum, body = parts[0], parts[1]
    assert len(checksum) == 4
    assert all(c in "0123456789abcdef" for c in checksum)
    assert checksum == layout_checksum(body)


def test_layout_for_window_checksum_matches_body() -> None:
    for pane_count in [1, 2, 3, 4, 5, 6, 9, 12]:
        result = layout_for_window(pane_count, 200, 50)
        comma_idx = result.index(",")
        checksum = result[:comma_idx]
        body = result[comma_idx + 1 :]
        assert checksum == layout_checksum(body), f"failed for pane_count={pane_count}"


def test_layout_for_window_row_calculation_one_pane() -> None:
    # rows=1 while (1+1)^2=4 > 1 => stays at rows=1, cols=1
    result = layout_for_window(1, 80, 24)
    assert "80x24" in result


def test_layout_for_window_row_calculation_four_panes() -> None:
    # rows starts 1; (1+1)^2=4<=4 => rows=2; (2+1)^2=9>4 => stop. rows=2, cols=2
    result = layout_for_window(4, 100, 50)
    body = build_window_layout(100, 50, 4, 2, 2)
    checksum = layout_checksum(body)
    assert result == f"{checksum},{body}"


def test_layout_for_window_row_calculation_nine_panes() -> None:
    # rows=1; (2)^2=4<=9 => rows=2; (3)^2=9<=9 => rows=3; (4)^2=16>9 => stop. rows=3, cols=3
    result = layout_for_window(9, 120, 60)
    body = build_window_layout(120, 60, 9, 3, 3)
    checksum = layout_checksum(body)
    assert result == f"{checksum},{body}"


def test_layout_for_window_row_calculation_five_panes() -> None:
    # rows=1; 4<=5 => rows=2; 9>5 => stop. rows=2, cols=(5+1)//2=3
    result = layout_for_window(5, 150, 40)
    body = build_window_layout(150, 40, 5, 3, 2)
    checksum = layout_checksum(body)
    assert result == f"{checksum},{body}"


def test_layout_for_window_two_panes() -> None:
    # rows=1; (2)^2=4>2 => stays at rows=1, cols=2
    result = layout_for_window(2, 100, 30)
    body = build_window_layout(100, 30, 2, 2, 1)
    checksum = layout_checksum(body)
    assert result == f"{checksum},{body}"


# ---------------------------------------------------------------------------
# tmux_display (mocked)
# ---------------------------------------------------------------------------


def test_tmux_display_calls_require_capture_without_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmd: list[list[str]] = []

    def fake_require_capture(cmd: list[str]) -> str:
        captured_cmd.append(cmd)
        return "  @1\n"

    monkeypatch.setattr("agm.tmux.layout.require_capture", fake_require_capture)

    result = tmux_display("#{window_id}")

    assert result == "@1"
    assert captured_cmd == [["tmux", "display-message", "-p", "#{window_id}"]]


def test_tmux_display_calls_require_capture_with_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmd: list[list[str]] = []

    def fake_require_capture(cmd: list[str]) -> str:
        captured_cmd.append(cmd)
        return "160\n"

    monkeypatch.setattr("agm.tmux.layout.require_capture", fake_require_capture)

    result = tmux_display("#{window_width}", target="@1")

    assert result == "160"
    assert captured_cmd == [["tmux", "display-message", "-p", "-t", "@1", "#{window_width}"]]


def test_tmux_display_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agm.tmux.layout.require_capture", lambda cmd: "  hello  \n")
    assert tmux_display("#{pane_title}") == "hello"


# ---------------------------------------------------------------------------
# resolve_window_layout_target (mocked)
# ---------------------------------------------------------------------------


def test_resolve_window_layout_target_without_window_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_require_capture(cmd: list[str]) -> str:
        calls.append(cmd)
        if "#{window_id}" in cmd:
            return "@2\n"
        if "#{window_width}" in cmd:
            return "200\n"
        if "#{window_height}" in cmd:
            return "50\n"
        return ""

    monkeypatch.setattr("agm.tmux.layout.require_capture", fake_require_capture)

    window_id, width, height = resolve_window_layout_target()

    assert window_id == "@2"
    assert width == 200
    assert height == 50
    assert any("#{window_id}" in cmd for cmd in calls)
    assert any("#{window_width}" in cmd for cmd in calls)
    assert any("#{window_height}" in cmd for cmd in calls)


def test_resolve_window_layout_target_with_window_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_require_capture(cmd: list[str]) -> str:
        calls.append(cmd)
        if "#{window_width}" in cmd:
            return "160\n"
        if "#{window_height}" in cmd:
            return "40\n"
        return ""

    monkeypatch.setattr("agm.tmux.layout.require_capture", fake_require_capture)

    window_id, width, height = resolve_window_layout_target(window_id="@5")

    assert window_id == "@5"
    assert width == 160
    assert height == 40
    # Should NOT have queried #{window_id} since it was provided
    assert not any("#{window_id}" in cmd for cmd in calls)
    # But should have used -t @5 for width/height queries
    assert any("-t" in cmd and "@5" in cmd for cmd in calls)


def test_resolve_window_layout_target_uses_resolved_id_as_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets_used: list[str] = []

    def fake_require_capture(cmd: list[str]) -> str:
        if "-t" in cmd:
            idx = cmd.index("-t")
            targets_used.append(cmd[idx + 1])
        if "#{window_id}" in cmd:
            return "@3\n"
        if "#{window_width}" in cmd:
            return "80\n"
        if "#{window_height}" in cmd:
            return "24\n"
        return ""

    monkeypatch.setattr("agm.tmux.layout.require_capture", fake_require_capture)

    resolve_window_layout_target()

    # All target usages must reference the resolved window id
    assert all(t == "@3" for t in targets_used)


# ---------------------------------------------------------------------------
# apply_layout (mocked)
# ---------------------------------------------------------------------------


def test_apply_layout_calls_select_layout_with_correct_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_require_success(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr("agm.tmux.layout.require_success", fake_require_success)

    apply_layout(pane_count=4, window_id="@1", width=100, height=50)

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:4] == ["tmux", "select-layout", "-t", "@1"]
    expected_layout = layout_for_window(4, 100, 50)
    assert cmd[4] == expected_layout


def test_apply_layout_uses_layout_for_window(monkeypatch: pytest.MonkeyPatch) -> None:
    issued_layout: list[str] = []

    def fake_require_success(cmd: list[str]) -> None:
        issued_layout.append(cmd[-1])

    monkeypatch.setattr("agm.tmux.layout.require_success", fake_require_success)

    apply_layout(pane_count=9, window_id="@7", width=120, height=60)

    assert issued_layout == [layout_for_window(9, 120, 60)]


def test_apply_layout_different_pane_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    results: dict[int, str] = {}

    def fake_require_success(cmd: list[str]) -> None:
        results[len(results)] = cmd[-1]

    monkeypatch.setattr("agm.tmux.layout.require_success", fake_require_success)

    for count in [1, 2, 3, 6]:
        apply_layout(pane_count=count, window_id="@0", width=80, height=24)

    for i, count in enumerate([1, 2, 3, 6]):
        assert results[i] == layout_for_window(count, 80, 24)