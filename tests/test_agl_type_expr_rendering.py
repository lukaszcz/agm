"""Tests for canonical source spellings of AgL type expressions."""

from __future__ import annotations

from typing import cast

import pytest

from agm.agl.parser import parse_type_expr
from agm.agl.syntax.types import TypeExpr, render_type_expr


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
    ),
)
def test_render_type_expr_uses_a_canonical_source_spelling(source: str, expected: str) -> None:
    assert render_type_expr(parse_type_expr(source)) == expected


def test_render_type_expr_rejects_unknown_ast_nodes() -> None:
    with pytest.raises(AssertionError, match="unexpected type expression"):
        render_type_expr(cast(TypeExpr, object()))
