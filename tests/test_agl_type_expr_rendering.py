"""Tests for canonical source spellings of AgL type expressions."""

from __future__ import annotations

import pytest

from agm.agl.parser import parse_type_expr
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import DictT, IntT, TextT, render_type_expr


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("text", "text"),
        ("json", "json"),
        ("bool", "bool"),
        ("int", "int"),
        ("decimal", "decimal"),
        ("unit", "unit"),
        ("agent", "agent"),
        ("array[int]", "array[int]"),
        ("dict[text, array[bool]]", "dict[text, array[bool]]"),
        ("() -> int", "() -> int"),
        ("int -> bool", "int -> bool"),
        ("(int, text) -> bool", "(int, text) -> bool"),
        ("(int -> bool) -> text", "(int -> bool) -> text"),
        ("Thing", "Thing"),
        ("Thing[int, text]", "Thing[int, text]"),
        ("module::Thing", "module::Thing"),
        ("/module/path::Thing", "/module/path::Thing"),
        ("::Thing", "::Thing"),
        ("module::Thing[int]", "module::Thing[int]"),
        ("Outer[int]::Thing", "Outer[int]::Thing"),
        # Multi-segment qualifiers: every segment is rendered, in source order,
        # each keeping its own type arguments.
        ("alpha::beta::Thing", "alpha::beta::Thing"),
        ("alpha::beta::gamma::Thing", "alpha::beta::gamma::Thing"),
        ("::alpha::beta::Thing", "::alpha::beta::Thing"),
        ("/alpha/beta::gamma::Thing", "/alpha/beta::gamma::Thing"),
        ("Outer[int]::Inner[text]::Thing", "Outer[int]::Inner[text]::Thing"),
        ("alpha::Inner[text]::beta::Thing[bool]", "alpha::Inner[text]::beta::Thing[bool]"),
    ),
)
def test_render_type_expr_uses_a_canonical_source_spelling(source: str, expected: str) -> None:
    assert render_type_expr(parse_type_expr(source)) == expected


def _span() -> SourceSpan:
    return SourceSpan(
        start_line=1, start_col=1, end_line=1, end_col=1, start_offset=0, end_offset=0
    )


def test_render_type_expr_renders_a_non_text_dict_key() -> None:
    # The parser only ever builds a text key, so a non-text key is
    # constructed directly rather than parsed from source.
    key = IntT(span=_span(), node_id=1)
    value = TextT(span=_span(), node_id=2)
    dict_t = DictT(key=key, value=value, span=_span(), node_id=3)
    assert render_type_expr(dict_t) == "dict[int, text]"
