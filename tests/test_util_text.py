"""Unit tests for agm.util.text."""

from __future__ import annotations

from agm.util.text import normalize_newlines


class TestNormalizeNewlines:
    def test_crlf_converted(self) -> None:
        assert normalize_newlines("a\r\nb") == "a\nb"

    def test_lone_cr_converted(self) -> None:
        assert normalize_newlines("a\rb") == "a\nb"

    def test_mixed_crlf_and_lone_cr(self) -> None:
        assert normalize_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_already_lf_unchanged(self) -> None:
        assert normalize_newlines("x\ny") == "x\ny"

    def test_empty_string(self) -> None:
        assert normalize_newlines("") == ""

    def test_only_crlf(self) -> None:
        assert normalize_newlines("\r\n") == "\n"

    def test_only_lone_cr(self) -> None:
        assert normalize_newlines("\r") == "\n"

    def test_idempotent(self) -> None:
        text = "line1\nline2\nline3"
        assert normalize_newlines(normalize_newlines(text)) == normalize_newlines(text)
