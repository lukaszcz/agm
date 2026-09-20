"""Tests for the plain-name predicate in `agm.agl.keywords`."""

from __future__ import annotations

import pytest

from agm.agl.keywords import is_plain_name


@pytest.mark.parametrize(
    "name", ("radius", "ask-prompt", "do-it-now?", "or", "not", "Circle", "exec$", "ask$")
)
def test_is_plain_name_accepts_ordinary_and_soft_keyword_spellings(name: str) -> None:
    assert is_plain_name(name)


@pytest.mark.parametrize("name", ("if", "let", "true", "false", "null", ""))
def test_is_plain_name_rejects_reserved_words(name: str) -> None:
    assert not is_plain_name(name)
