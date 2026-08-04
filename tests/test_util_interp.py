from __future__ import annotations

import pytest

from agm.util.interp import (
    Hole,
    InterpolationError,
    Literal,
    interp,
    interp_preserving,
    interp_segments,
    split_template,
)


@pytest.mark.parametrize(
    ("template", "variables", "expected"),
    [
        ("plain text", {}, "plain text"),
        ("100% complete", {}, "100% complete"),
        (r"\path\file", {}, r"\path\file"),
        (r"\%{name}", {"name": "Ada"}, "%{name}"),
        (r"\% no hole", {}, r"\% no hole"),
        ("%{name}", {"name": "Ada"}, "Ada"),
        ("%{first} %{last}", {"first": "Ada", "last": "Lovelace"}, "Ada Lovelace"),
        ("%{first}%{last}", {"first": "Ada", "last": "Lovelace"}, "AdaLovelace"),
        ("%{name}!", {"name": "Ada"}, "Ada!"),
        ("hello %{name}", {"name": "Ada"}, "hello Ada"),
        (
            "%{log-file} %{ask?} %{a+b} %{do-it!}",
            {"log-file": "log", "ask?": "ask", "a+b": "sum", "do-it!": "do"},
            "log ask sum do",
        ),
        (
            "first: %{first}\nsecond: %{second}",
            {"first": "one", "second": "two"},
            "first: one\nsecond: two",
        ),
        ("", {}, ""),
        ("%{empty}", {"empty": ""}, ""),
    ],
)
def test_interp_replaces_holes_and_preserves_literal_text(
    template: str, variables: dict[str, str], expected: str
) -> None:
    assert interp(template, variables) == expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (
            "before %{first}%{second} after",
            [Literal("before "), Hole("first"), Hole("second"), Literal(" after")],
        ),
        ("no holes here", [Literal("no holes here")]),
        ("100% literal", [Literal("100% literal")]),
        ("", []),
        ("%{only}", [Hole("only")]),
        (r"\%{escaped} %{hole}", [Literal("%{escaped} "), Hole("hole")]),
    ],
)
def test_split_template_separates_literals_from_holes(
    template: str, expected: list[Literal | Hole]
) -> None:
    assert split_template(template) == expected


def test_split_template_records_the_offset_of_each_hole() -> None:
    segments = split_template("ab %{first} cd %{second}")

    holes = [segment for segment in segments if isinstance(segment, Hole)]
    assert [(hole.name, hole.offset) for hole in holes] == [("first", 3), ("second", 15)]


def test_split_segments_can_be_rendered_against_different_mappings() -> None:
    segments = split_template("before %{first}%{second} after")

    assert interp_segments(segments, {"first": "A", "second": "B"}) == "before AB after"
    assert interp_segments(segments, {"first": "x", "second": "y"}) == "before xy after"


@pytest.mark.parametrize(
    ("template", "kind", "text", "offset"),
    [
        ("before %{a b}", "invalid hole name", "a b", 7),
        ("%{1x}", "invalid hole name", "1x", 0),
        ("%{}", "invalid hole name", "", 0),
        ("before %{name", "unterminated hole", "%{name", 7),
    ],
)
def test_interp_reports_malformed_holes(template: str, kind: str, text: str, offset: int) -> None:
    with pytest.raises(InterpolationError) as raised:
        interp(template, {})

    error = raised.value
    assert error.kind == kind
    assert error.text == text
    assert error.offset == offset
    assert kind in str(error)
    assert text in str(error)
    assert str(offset) in str(error)


def test_interp_reports_a_missing_variable_with_its_name_and_offset() -> None:
    with pytest.raises(InterpolationError) as raised:
        interp("hello %{missing}", {})

    error = raised.value
    assert error.kind == "missing variable"
    assert error.text == "missing"
    assert error.offset == 6
    assert "missing variable" in str(error)
    assert "missing" in str(error)
    assert "6" in str(error)


@pytest.mark.parametrize(
    "template",
    [
        "%{a b}",
        "%{1x}",
        "%{}",
        "before %{name",
    ],
)
def test_lenient_interpolation_preserves_malformed_holes_verbatim(template: str) -> None:
    assert interp_preserving(template, {})[0] == template


def test_lenient_interpolation_mixes_resolved_unresolved_and_malformed_holes() -> None:
    template = r"%{known} %{unknown} %{bad name} \%{escaped} %{unfinished"

    assert interp_preserving(template, {"known": "value"})[0] == (
        "value %{unknown} %{bad name} %{escaped} %{unfinished"
    )


def test_lenient_interpolation_has_the_same_escape_and_bare_percent_behavior() -> None:
    template = r"a \%{name} is 100% done; keep \% too"

    assert interp_preserving(template, {"name": "Ada"})[0] == interp(template, {"name": "Ada"})
    assert interp_preserving(template, {"name": "Ada"})[0] == r"a %{name} is 100% done; keep \% too"


def test_lenient_interpolation_leaves_missing_variables_verbatim() -> None:
    assert interp_preserving("%{known}/%{unknown}", {"known": "value"})[0] == "value/%{unknown}"


@pytest.mark.parametrize(
    ("template", "expected_text", "expected_unresolved"),
    [
        ("%{known}/prompt.md", "value/prompt.md", False),
        ("plain/prompt.md", "plain/prompt.md", False),
        (r"\%{known}/prompt.md", "%{known}/prompt.md", False),
        ("%{unknown}/prompt.md", "%{unknown}/prompt.md", True),
        ("%{bad name}/prompt.md", "%{bad name}/prompt.md", True),
        ("%{unterminated", "%{unterminated", True),
        ("%{known}/%{unknown}", "value/%{unknown}", True),
    ],
)
def test_interp_preserving_reports_whether_anything_stayed_unresolved(
    template: str, expected_text: str, expected_unresolved: bool
) -> None:
    text, unresolved = interp_preserving(template, {"known": "value"})

    assert text == expected_text
    assert unresolved is expected_unresolved
