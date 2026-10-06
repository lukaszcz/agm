"""A non-exhaustive ``case`` witness uses the shortest spelling its region accepts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agm.agl.matchcompile import NonExhaustiveIssue, compile_program_matches, render_witness
from tests._agl_helpers import check_agl_graph
from tests.agl.ir_harness import make_graph_from_files
from tests.agl.qualifier_support import graph_verdict

if TYPE_CHECKING:
    from agm.agl.modules.loader import ModuleGraph

_LIB = "enum Color\n  | Red\n  | Blue\nrecord Flag\n  on: bool\n"

_RB = "record Blue\n  n: int\n"

_ARMS = "@@"
"""Placeholder for the arms of the one ``case`` in an entry template."""


def _graph(tmp_path: Path, entry: str) -> ModuleGraph:
    return make_graph_from_files(tmp_path, {"lib": _LIB, "rb": _RB, "entry": entry})


def _witness(tmp_path: Path, entry: str) -> str:
    """The rendered witness of the single non-exhaustive ``case`` in *entry*."""
    issues = compile_program_matches(check_agl_graph(_graph(tmp_path, entry))).issues
    assert len(issues) == 1
    issue = issues[0]
    assert isinstance(issue, NonExhaustiveIssue)
    return render_witness(issue.witness)


def _case(header: str, scrutinee: str, indent: str = "") -> str:
    """Entry source: *header*, then a function matching a *scrutinee* parameter."""
    return f"{header}{indent}def f(v: {scrutinee}) -> int = case v of {_ARMS}\n"


_OVERQUALIFIED = "spells the constructor through its route although the region reaches it shorter"
_SCRUTINEE_SELECTS = "a same-named imported record blocks the bare member the scrutinee selects"
_SHADOWED_HEAD = "spells a route whose head a type parameter shadows at the case"

# (entry template, covered arm, expected witness)
_ENUM_ROWS = {
    "own-enum-at-root": pytest.param(
        _case("enum Color\n  | Red\n  | Blue\n", "Color"), "Red", "Blue"
    ),
    "root-import": pytest.param(_case("import lib::*\n", "Color"), "Red", "Blue"),
    "renamed-import": pytest.param(_case("import lib::{Color as Hue}\n", "Hue"), "Red", "Blue"),
    "route-only": pytest.param(
        _case("import lib\n", "lib::Color"), "lib::Color::Red", "lib::Color::Blue"
    ),
    "use-at-root": pytest.param(
        _case("import lib\nuse lib::Color\n", "Color"),
        "Red",
        "Blue",
        marks=pytest.mark.xfail(strict=True, reason=_OVERQUALIFIED),
    ),
    "use-in-scope": pytest.param(
        _case("import lib\nscope S\n  use lib::Color\n", "Color", "  ") + "end S\n",
        "Red",
        "Blue",
        marks=pytest.mark.xfail(strict=True, reason=_OVERQUALIFIED),
    ),
    "scope-import": pytest.param(
        _case("scope S\n  import lib::*\n", "Color", "  ") + "end S\n",
        "Red",
        "Blue",
        marks=pytest.mark.xfail(strict=True, reason=_OVERQUALIFIED),
    ),
    "own-record-beside-bare-member": pytest.param(
        _case("import lib::*\nrecord Blue\n  n: int\n", "Color"), "Red", "Blue"
    ),
    "own-scope-shadows-route-head": pytest.param(
        _case(
            "import lib\n\nscope lib\n  enum Color\n    | Red\n    | Green\nend lib\n\n",
            "/lib::Color",
        ),
        "/lib::Color::Red",
        "/lib::Color::Blue",
    ),
    "imported-record-beside-bare-member": pytest.param(
        _case("import lib::*\nimport rb::*\n", "Color"),
        "Red",
        "Blue",
        marks=pytest.mark.xfail(strict=True, reason=_SCRUTINEE_SELECTS),
    ),
    "route-alias": pytest.param(
        _case("import lib as L\n", "L::Color"), "L::Color::Red", "L::Color::Blue"
    ),
    "type-parameter-keeps-bare-member": pytest.param(
        "import lib::*\ndef f[Color](v: Color, w: lib::Color) -> int = case w of @@\n",
        "Red",
        "Blue",
    ),
    "type-parameter-shadows-route-head": pytest.param(
        "import lib\ndef f[lib](v: lib, w: /lib::Color) -> int = case w of @@\n",
        "/lib::Color::Red",
        "/lib::Color::Blue",
        marks=pytest.mark.xfail(strict=True, reason=_SHADOWED_HEAD),
    ),
}

_RECORD_ROWS = {
    "root-import": pytest.param(
        _case("import lib::*\n", "Flag"), "Flag(on = true)", "Flag(on = false)"
    ),
    "scope-import": pytest.param(
        _case("scope S\n  import lib::*\n", "Flag", "  ") + "end S\n",
        "Flag(on = true)",
        "Flag(on = false)",
        marks=pytest.mark.xfail(strict=True, reason=_OVERQUALIFIED),
    ),
    "own-record-shadows-imported": pytest.param(
        _case(
            "import lib::*\nscope S\n  record Flag\n    z: int\n",
            "/lib::Flag",
            "  ",
        )
        + "end S\n",
        "/lib::Flag(on = true)",
        "Flag(on = false)",
        marks=pytest.mark.xfail(strict=True, reason=_OVERQUALIFIED),
    ),
}


def _assert_witness_completes(tmp_path: Path, entry: str, covered: str, expected: str) -> None:
    """The witness is *expected*, and as an extra arm at the same site it makes the match total."""
    assert _witness(tmp_path / "missing", entry.replace(_ARMS, f"| {covered} => 0")) == expected
    completed = entry.replace(_ARMS, f"| {covered} => 0 | {expected} => 1")
    assert graph_verdict(_graph(tmp_path / "complete", completed))[0] == "accepted"


@pytest.mark.parametrize(("entry", "covered", "expected"), _ENUM_ROWS.values(), ids=_ENUM_ROWS)
def test_enum_witness_is_the_shortest_spelling_at_the_case(
    tmp_path: Path, entry: str, covered: str, expected: str
) -> None:
    _assert_witness_completes(tmp_path, entry, covered, expected)


@pytest.mark.parametrize(("entry", "covered", "expected"), _RECORD_ROWS.values(), ids=_RECORD_ROWS)
def test_record_witness_is_the_shortest_spelling_at_the_case(
    tmp_path: Path, entry: str, covered: str, expected: str
) -> None:
    _assert_witness_completes(tmp_path, entry, covered, expected)
