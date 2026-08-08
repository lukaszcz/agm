from __future__ import annotations

import pytest

from agm.util.ident import is_identifier


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("name", True),
        ("_name", True),
        ("log-file", True),
        ("ask?", True),
        ("a+b", True),
        ("do-it!", True),
        ("é2", True),
        ("", False),
        ("1x", False),
        ("a b", False),
        ("a/b", False),
        ("a=b", False),
    ],
)
def test_identifier_matches_agl_identifier_grammar(name: str, expected: bool) -> None:
    assert is_identifier(name) is expected
