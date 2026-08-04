from __future__ import annotations

import pytest

from agm.util.interp import (
    Hole,
    InterpolationError,
    Literal,
    assemble,
    interp,
    interp_lenient,
    is_interp_name,
    split_template,
)


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
def test_interp_name_matches_agl_identifier_grammar(name: str, expected: bool) -> None:
    assert is_interp_name(name) is expected


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


def test_split_template_and_assemble_are_independent_of_mappings() -> None:
    segments = split_template("before %{first}%{second} after")

    assert segments == [Literal("before "), Hole("first"), Hole("second"), Literal(" after")]
    assert assemble(segments, lambda name: name.upper()) == "before FIRSTSECOND after"


def test_assemble_resolves_each_hole_in_order() -> None:
    calls: list[str] = []

    def resolve(name: str) -> str:
        calls.append(name)
        return str(len(calls))

    assert assemble([Hole("one"), Literal("/"), Hole("two"), Hole("one")], resolve) == "1/23"
    assert calls == ["one", "two", "one"]


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
    assert interp_lenient(template, {}) == template


def test_lenient_interpolation_mixes_resolved_unresolved_and_malformed_holes() -> None:
    template = r"%{known} %{unknown} %{bad name} \%{escaped} %{unfinished"

    assert interp_lenient(template, {"known": "value"}) == (
        "value %{unknown} %{bad name} %{escaped} %{unfinished"
    )


def test_lenient_interpolation_has_the_same_escape_and_bare_percent_behavior() -> None:
    template = r"a \%{name} is 100% done; keep \% too"

    assert interp_lenient(template, {"name": "Ada"}) == interp(template, {"name": "Ada"})
    assert interp_lenient(template, {"name": "Ada"}) == r"a %{name} is 100% done; keep \% too"


def test_lenient_interpolation_leaves_missing_variables_verbatim() -> None:
    assert interp_lenient("%{known}/%{unknown}", {"known": "value"}) == "value/%{unknown}"
