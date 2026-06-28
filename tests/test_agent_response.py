"""Tests for agent response parsing helpers."""

from __future__ import annotations

import pytest

from agm.agent.response import is_complete_output, last_response_line


class TestLastResponseLine:
    def test_returns_stripped_last_line(self) -> None:
        assert last_response_line("progress\n COMPLETE \n") == "COMPLETE"

    def test_returns_stripped_output_when_no_lines(self) -> None:
        assert last_response_line("   ") == ""


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("COMPLETE", True),
        ("COMPLETE\n", True),
        ("  COMPLETE  ", True),
        ("progress\nCOMPLETE\n", True),
        ("complete", False),
        ("COMPLETE\nmore text\n", False),
        ("all done\n", False),
        ("", False),
    ],
    ids=[
        "bare-complete",
        "trailing-newline",
        "surrounding-whitespace",
        "complete-as-last-line",
        "lowercase-is-not-complete",
        "complete-not-on-last-line",
        "unrelated-output",
        "empty-output",
    ],
)
def test_is_complete_output(output: str, expected: bool) -> None:
    assert is_complete_output(output) is expected
