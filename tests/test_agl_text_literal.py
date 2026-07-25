"""Tests for AgL's shared text-literal surface form."""

from __future__ import annotations

import pytest

from agm.agl.lexer import tokenize
from agm.agl.matchcompile import LiteralKind, LiteralWitness, render_witness
from agm.agl.runtime.render import render_value
from agm.agl.semantics.text_literal import (
    ESCAPE_DECODE,
    ESCAPE_ENCODE,
    INTERP_OPEN,
    INTERP_TRIGGER,
    quote_text,
)
from agm.agl.semantics.values import TextValue

_TEXT_CORPUS = (
    '"',
    "\\",
    "\n",
    "\t",
    "\r",
    "\b",
    "\f",
    *(chr(codepoint) for codepoint in range(0x20)),
    "\x7f",
    "\u2028",
    "é",
    "/",
    "%",
    "%{x}",
    r"\%{x}",
)


@pytest.mark.parametrize("value", _TEXT_CORPUS)
def test_text_encoders_agree(value: str) -> None:
    """Rendered text and text match witnesses use the same AgL literal form."""
    assert render_value(TextValue(value), quote_strings=True) == render_witness(
        LiteralWitness(LiteralKind.TEXT, value)
    )


@pytest.mark.parametrize("value", _TEXT_CORPUS)
def test_quoted_text_round_trips_through_the_template_scanner(value: str) -> None:
    """The shared encoder produces a quoted literal that scans back exactly."""
    tokens = list(tokenize(quote_text(value)))
    fragments = [str(token) for token in tokens if token.type == "STRING_FRAGMENT"]

    assert fragments == [value]


def test_text_literal_surface_constants_define_escape_directions() -> None:
    assert INTERP_TRIGGER == "%"
    assert INTERP_OPEN == "%{"
    assert ESCAPE_DECODE[INTERP_TRIGGER] == INTERP_TRIGGER
    assert ESCAPE_DECODE["'"] == "'"
    assert ESCAPE_DECODE["/"] == "/"
    assert ESCAPE_ENCODE[INTERP_TRIGGER] == r"\%"
    assert "'" not in ESCAPE_ENCODE
    assert "/" not in ESCAPE_ENCODE
