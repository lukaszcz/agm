"""Tests for agent response parsing helpers."""

from __future__ import annotations

from agm.agent.response import last_response_line


class TestLastResponseLine:
    def test_returns_stripped_last_line(self) -> None:
        assert last_response_line("progress\n COMPLETE \n") == "COMPLETE"

    def test_returns_stripped_output_when_no_lines(self) -> None:
        assert last_response_line("   ") == ""
