"""Tests for shared diagnostic helpers in ``agm.agl.diagnostics``."""

from __future__ import annotations

from agm.agl.diagnostics import dollar_spacing_hint


class TestDollarSpacingHint:
    def test_name_with_stem_suggests_the_spaced_form(self) -> None:
        hint = dollar_spacing_hint("exec$")
        assert hint is not None
        assert "exec $" in hint

    def test_different_stem_is_reflected_in_the_hint(self) -> None:
        hint = dollar_spacing_hint("ask$")
        assert hint is not None
        assert "ask $" in hint

    def test_name_without_dollar_suffix_has_no_hint(self) -> None:
        assert dollar_spacing_hint("exec") is None

    def test_bare_dollar_has_no_hint(self) -> None:
        """A lone '$' is never lexable as a NAME, but the helper stays total."""
        assert dollar_spacing_hint("$") is None

    def test_empty_name_has_no_hint(self) -> None:
        assert dollar_spacing_hint("") is None
