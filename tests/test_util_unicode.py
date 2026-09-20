"""Scalar-text helpers: surrogate detection, escaped rendering, strict JSON."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from agm.util.unicode import (
    LoneSurrogateError,
    holds_surrogate,
    loads_json,
    require_scalar_text,
    surrogate_index,
    visible_text,
)

HIGH = chr(0xD800)
LOW = chr(0xDFFF)
# The escapes are built from parts so that no tool between here and the file
# can fold an adjacent pair into the astral character it denotes.
HIGH_ESCAPE = "\\" + "ud83d"
LOW_ESCAPE = "\\" + "ude00"


# ---------------------------------------------------------------------------
# surrogate_index / require_scalar_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "plain ascii", "café \U0001f600 中"])
def test_scalar_text_has_no_surrogate_index(text: str) -> None:
    assert surrogate_index(text) is None
    assert require_scalar_text(text) == text


@pytest.mark.parametrize(
    ("text", "index"),
    [
        (HIGH, 0),
        (LOW, 0),
        ("café" + HIGH, 4),
        (HIGH + LOW, 0),
        ("a" + LOW + "b", 1),
    ],
)
def test_surrogate_index_reports_the_first_offender(text: str, index: int) -> None:
    assert surrogate_index(text) == index


def test_require_scalar_text_names_the_code_point_without_carrying_it() -> None:
    with pytest.raises(LoneSurrogateError) as excinfo:
        require_scalar_text("ab" + LOW)
    assert excinfo.value.index == 2
    message = str(excinfo.value)
    assert "U+DFFF" in message
    assert message.isascii()


def test_lone_surrogate_error_is_a_value_error() -> None:
    assert issubclass(LoneSurrogateError, ValueError)


# ---------------------------------------------------------------------------
# holds_surrogate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        HIGH,
        [1, ["ok", HIGH]],
        ("ok", (HIGH,)),
        {"key": {"inner": HIGH}},
        {HIGH: "value"},
        {"outer": [{"k": LOW}]},
    ],
)
def test_holds_surrogate_walks_keys_values_and_sequences(obj: object) -> None:
    assert holds_surrogate(obj)


@pytest.mark.parametrize(
    "obj",
    [None, True, 7, Decimal("1.5"), "clean", [], {}, {"k": ["v", 1, None]}, ("a", 2)],
)
def test_holds_surrogate_ignores_clean_data_and_non_text_leaves(obj: object) -> None:
    assert not holds_surrogate(obj)


# ---------------------------------------------------------------------------
# visible_text
# ---------------------------------------------------------------------------


def test_visible_text_escapes_surrogates_and_keeps_scalars() -> None:
    rendered = visible_text("a" + LOW + "b")
    assert rendered == "a\\udfffb"
    assert rendered.isascii()
    assert visible_text("café") == "café"


# ---------------------------------------------------------------------------
# loads_json
# ---------------------------------------------------------------------------


def test_loads_json_parses_clean_documents() -> None:
    assert loads_json('{"a": [1, "b", null]}') == {"a": [1, "b", None]}


def test_loads_json_combines_an_escape_pair() -> None:
    parsed = loads_json(f'"{HIGH_ESCAPE}{LOW_ESCAPE}"')
    assert parsed == "\U0001f600"


@pytest.mark.parametrize(
    "doc",
    [
        '"\\ud800"',
        '{"k": "\\udfff"}',
        '{"\\ud800": 1}',
        '["ok", "\\uDC00"]',
    ],
)
def test_loads_json_rejects_a_lone_surrogate_escape(doc: str) -> None:
    with pytest.raises(json.JSONDecodeError) as excinfo:
        loads_json(doc)
    assert excinfo.value.pos == doc.index("\\u")


def test_loads_json_accepts_a_literal_backslash_before_a_surrogate_spelling() -> None:
    # ``\\uD800`` is an escaped backslash followed by text, not an escape.
    assert loads_json('"\\\\ud800"') == "\\ud800"


def test_loads_json_passes_its_hooks_through() -> None:
    assert loads_json("1.5", parse_float=Decimal) == Decimal("1.5")

    def reject(constant: str) -> object:
        raise ValueError(constant)

    with pytest.raises(ValueError, match="NaN"):
        loads_json("[NaN]", parse_constant=reject)

    def pairs(items: list[tuple[str, object]]) -> object:
        return items

    assert loads_json('{"a": 1}', object_pairs_hook=pairs) == [("a", 1)]
