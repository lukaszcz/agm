"""Tests for agent response parsing helpers."""

from __future__ import annotations

from agm.agent.response import is_complete_output, last_response_line


class TestLastResponseLine:
    def test_returns_stripped_last_line(self) -> None:
        assert last_response_line("progress\n COMPLETE \n") == "COMPLETE"

    def test_returns_stripped_output_when_no_lines(self) -> None:
        assert last_response_line("   ") == ""


class TestIsCompleteOutput:
    def test_bare_complete_is_true(self) -> None:
        assert is_complete_output("COMPLETE")

    def test_trailing_newline_is_true(self) -> None:
        assert is_complete_output("COMPLETE\n")

    def test_complete_with_surrounding_whitespace_is_true(self) -> None:
        assert is_complete_output("  COMPLETE  ")

    def test_complete_as_last_line_is_true(self) -> None:
        assert is_complete_output("progress\nCOMPLETE\n")

    def test_lowercase_complete_is_false(self) -> None:
        assert not is_complete_output("complete")

    def test_complete_not_on_last_line_is_false(self) -> None:
        assert not is_complete_output("COMPLETE\nmore text\n")

    def test_unrelated_output_is_false(self) -> None:
        assert not is_complete_output("all done\n")

    def test_empty_output_is_false(self) -> None:
        assert not is_complete_output("")
