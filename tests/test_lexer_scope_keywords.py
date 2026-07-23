"""Behavioral tests for contextual scope-region lexer tokens."""

from __future__ import annotations

from agm.agl.lexer import spaced_qualifier_collector, tokenize, unclosed_scope_path


def _tokens(source: str) -> list[tuple[str, str]]:
    return [(token.type, str(token)) for token in tokenize(source)]


def _non_layout_tokens(source: str) -> list[tuple[str, str]]:
    return [token for token in _tokens(source) if not token[0].startswith("_")]


def test_scope_at_item_start_with_a_path_is_promoted() -> None:
    tokens = _non_layout_tokens("scope Point::Member")

    assert tokens[0] == ("SCOPE", "scope")
    assert tokens[1:] == [("MODQUAL", "Point"), ("NAME", "Member")]


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
    assert _non_layout_tokens("scope Outer\nscope Inner\nend Inner\nend Outer\nend Stray") == [
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
    assert _non_layout_tokens("scope Point\nrecord R\n  end: int\nend Point") == [
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
    assert _non_layout_tokens("scope Point\ndef f() -> int\n  end Thing\nend Point") == [
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
    assert _non_layout_tokens("scope Point\nend Point extra\nend Point") == [
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


def test_spaced_qualifier_advisories_cover_each_chain_segment() -> None:
    with spaced_qualifier_collector() as advisories:
        _non_layout_tokens("A :: B :: C :: member")

    assert [advisory.segments for advisory in advisories] == [("A",), ("B",), ("C",)]
