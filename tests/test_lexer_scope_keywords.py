"""Behavioral tests for contextual scope-region lexer tokens."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.lexer import spaced_qualifier_collector, tokenize, unclosed_scope_path
from agm.agl.lexer.errors import LexError


def _tokens(source: str) -> list[tuple[str, str]]:
    return [(token.type, str(token)) for token in tokenize(source)]


def _non_layout_tokens(source: str) -> list[tuple[str, str]]:
    return [token for token in _tokens(source) if not token[0].startswith("_")]


@pytest.mark.parametrize(
    "fixture",
    ("rejections/scope/used_scope_members_do_not_escape.agl",),
)
def test_use_in_scope_fixtures_remains_an_identifier(fixture: str) -> None:
    source = (Path(__file__).parent / "agl" / fixture).read_text()

    assert ("NAME", "use") in _tokens(source)


def test_scope_at_item_start_with_a_path_is_promoted() -> None:
    tokens = _non_layout_tokens("scope Point::Member")

    assert tokens[0] == ("SCOPE", "scope")
    assert tokens[1:] == [("MODQUAL", "Point"), ("NAME", "Member")]


def test_use_in_a_scope_region_is_promoted_and_keeps_its_header_window() -> None:
    assert _non_layout_tokens("scope Outer\n  use Shared::* hiding member\nend Outer") == [
        ("SCOPE", "scope"),
        ("NAME", "Outer"),
        ("USE", "use"),
        ("MODPATH", "Shared"),
        ("DCOLON", "::"),
        ("STAR", "*"),
        ("HIDING", "hiding"),
        ("NAME", "member"),
        ("END", "end"),
        ("NAME", "Outer"),
    ]


def test_use_in_expression_position_remains_an_identifier() -> None:
    assert _non_layout_tokens("let value = use + 1") == [
        ("let", "let"),
        ("NAME", "value"),
        ("EQ", "="),
        ("NAME", "use"),
        ("PLUS", "+"),
        ("INT", "1"),
    ]


@pytest.mark.parametrize("source", ("use()", "use + 1", "use(1)"))
def test_use_at_item_start_remains_an_identifier_without_a_declaration_form(source: str) -> None:
    assert _non_layout_tokens(source)[0] == ("NAME", "use")


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("use", ("NAME", "use")),
        ("use Shared", ("USE", "use")),
        ("use /module/nested/deeper::*", ("USE", "use")),
        ("use ::Scope::*", ("USE", "use")),
        ("use ::Scope", ("NAME", "use")),
        ("use ::Scope::member", ("USE", "use")),
    ),
)
def test_use_promotion_requires_a_complete_declaration_form(
    source: str, expected: tuple[str, str]
) -> None:
    assert _non_layout_tokens(source)[0] == expected


@pytest.mark.parametrize("source", ("use /module", "use /module/"))
def test_incomplete_anchored_module_use_is_rejected_as_a_path_error(source: str) -> None:
    with pytest.raises(LexError):
        _non_layout_tokens(source)


def test_incomplete_current_module_use_is_not_promoted() -> None:
    assert _non_layout_tokens("use ::*")[0] == ("NAME", "use")


def test_use_declaration_form_is_promoted_at_item_start() -> None:
    assert _non_layout_tokens("use Tools::*")[:2] == [("USE", "use"), ("MODPATH", "Tools")]


def test_scope_without_a_complete_path_remains_an_identifier() -> None:
    assert _non_layout_tokens("scope = point") == [
        ("NAME", "scope"),
        ("EQ", "="),
        ("NAME", "point"),
    ]
    assert _non_layout_tokens("scope Point::")[0] == ("NAME", "scope")


def test_scope_and_end_remain_identifiers_outside_promotion_positions() -> None:
    assert _non_layout_tokens("let scope = end\ndef end() = scope\nrecord end") == [
        ("let", "let"),
        ("NAME", "scope"),
        ("EQ", "="),
        ("NAME", "end"),
        ("def", "def"),
        ("NAME", "end"),
        ("LPAR", "("),
        ("RPAR", ")"),
        ("EQ", "="),
        ("NAME", "scope"),
        ("record", "record"),
        ("NAME", "end"),
    ]


def test_end_is_promoted_only_while_a_scope_region_is_open() -> None:
    assert _non_layout_tokens("end Point") == [("NAME", "end"), ("NAME", "Point")]


def test_nested_scope_regions_track_depth_until_the_last_end() -> None:
    assert _non_layout_tokens(
        "scope Outer\n\n  scope Inner\n  end Inner\nend Outer\n\nend Stray"
    ) == [
        ("SCOPE", "scope"),
        ("NAME", "Outer"),
        ("SCOPE", "scope"),
        ("NAME", "Inner"),
        ("END", "end"),
        ("NAME", "Inner"),
        ("END", "end"),
        ("NAME", "Outer"),
        ("NAME", "end"),
        ("NAME", "Stray"),
    ]


def test_end_is_promoted_only_at_the_open_region_layout_level() -> None:
    assert _non_layout_tokens("scope Point\n  record R\n    end: int\nend Point") == [
        ("SCOPE", "scope"),
        ("NAME", "Point"),
        ("record", "record"),
        ("NAME", "R"),
        ("NAME", "end"),
        ("COLON", ":"),
        ("NAME", "int"),
        ("END", "end"),
        ("NAME", "Point"),
    ]


def test_end_expression_in_a_declaration_suite_remains_names() -> None:
    assert _non_layout_tokens("scope Point\n  def f() -> int\n    end Thing\nend Point") == [
        ("SCOPE", "scope"),
        ("NAME", "Point"),
        ("def", "def"),
        ("NAME", "f"),
        ("LPAR", "("),
        ("RPAR", ")"),
        ("THIN_ARROW", "->"),
        ("NAME", "int"),
        ("NAME", "end"),
        ("NAME", "Thing"),
        ("END", "end"),
        ("NAME", "Point"),
    ]


def test_end_requires_a_complete_closer_line() -> None:
    assert _non_layout_tokens("scope Point\n  end Point extra\nend Point") == [
        ("SCOPE", "scope"),
        ("NAME", "Point"),
        ("NAME", "end"),
        ("NAME", "Point"),
        ("NAME", "extra"),
        ("END", "end"),
        ("NAME", "Point"),
    ]


def test_unclosed_scope_path_accounts_for_nested_closers() -> None:
    assert unclosed_scope_path("scope Outer\nscope Inner\nend Inner") == "Outer"


def test_multi_segment_qualifier_chain_emits_modqual_sequence() -> None:
    assert _non_layout_tokens("A::B::C::member") == [
        ("MODQUAL", "A"),
        ("MODQUAL", "B"),
        ("MODQUAL", "C"),
        ("NAME", "member"),
    ]


def test_use_target_uses_ordinary_module_qualifier_tokens() -> None:
    assert _non_layout_tokens("use A::B::member as Alias") == [
        ("USE", "use"),
        ("MODPATH", "A"),
        ("DCOLON", "::"),
        ("MODQUAL", "B"),
        ("NAME", "member"),
        ("as", "as"),
        ("NAME", "Alias"),
    ]


def test_spaced_qualifier_advisories_cover_each_chain_segment() -> None:
    with spaced_qualifier_collector() as advisories:
        _non_layout_tokens("A :: B :: C :: member")

    assert [advisory.segments for advisory in advisories] == [("A",), ("B",), ("C",)]
