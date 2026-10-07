"""Unit tests for agm.util.text."""

from __future__ import annotations

import click

from agm.util.text import (
    first_paragraph,
    format_description_column,
    keep_indented_paragraphs,
    normalize_newlines,
)


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


class TestFirstParagraph:
    def test_a_single_line_is_its_own_summary(self) -> None:
        assert first_paragraph("Review changes") == "Review changes"

    def test_wrapped_lines_fold_into_one(self) -> None:
        assert first_paragraph("Review the tree and\nreport findings.") == (
            "Review the tree and report findings."
        )

    def test_later_paragraphs_are_dropped(self) -> None:
        assert first_paragraph("Summary line.\n\nFurther detail.") == "Summary line."

    def test_indentation_and_surrounding_blanks_are_stripped(self) -> None:
        assert first_paragraph("\n  Summary\n  line\n\n  detail\n") == "Summary line"

    def test_prose_without_a_paragraph_is_empty(self) -> None:
        assert first_paragraph("  \n\n  ") == ""


class TestFormatDescriptionColumn:
    def test_entries_align_on_the_description_column(self) -> None:
        lines = format_description_column((("open", "Open a workspace"), ("x", "Do it")), width=40)

        assert lines == ["  open  Open a workspace", "  x     Do it"]

    def test_a_long_description_wraps_under_itself(self) -> None:
        lines = format_description_column((("review", "one two three four five"),), width=24)

        assert lines == ["  review  one two three", "          four five"]

    def test_an_empty_listing_renders_nothing(self) -> None:
        assert format_description_column((), width=40) == []

    def test_an_entry_without_a_description_lists_its_name_alone(self) -> None:
        assert format_description_column((("bare", ""),), width=40) == ["  bare"]

    def test_a_name_wider_than_the_width_keeps_one_word_per_line(self) -> None:
        lines = format_description_column((("verylongname", "alpha beta"),), width=10)

        assert lines == ["  verylongname  alpha", "                beta"]


def _wrapped(text: str, width: int = 40) -> str:
    """Render *text* as Click's help formatter lays out a paragraph block."""
    return click.formatting.wrap_text(
        keep_indented_paragraphs(text), width, preserve_paragraphs=True
    )


class TestKeepIndentedParagraphs:
    def test_an_indented_block_keeps_its_line_breaks(self) -> None:
        text = "Examples:\n  agm devel refine main\n  agm devel refine tip"

        assert _wrapped(text) == text

    def test_the_transform_is_idempotent(self) -> None:
        once = keep_indented_paragraphs("Examples:\n  agm exec main\n\nProse.")

        assert keep_indented_paragraphs(once) == once

    def test_prose_paragraphs_still_rewrap(self) -> None:
        assert _wrapped("Runs a loop\nuntil done.\n\nThen stops.") == (
            "Runs a loop until done.\n\nThen stops."
        )

    def test_the_common_indentation_of_a_docstring_is_removed(self) -> None:
        text = "\n    Run.\n\n    Examples:\n      agm exec\n      agm exec tip\n"

        assert _wrapped(text) == "Run.\n\nExamples:\n  agm exec\n  agm exec tip"

    def test_a_whitespace_only_line_separates_paragraphs(self) -> None:
        text = "Run\nthis.\n   \nExamples:\n  agm exec\n  agm exec tip"

        assert _wrapped(text) == "Run this.\n\nExamples:\n  agm exec\n  agm exec tip"
