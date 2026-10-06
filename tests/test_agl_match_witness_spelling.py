"""A non-exhaustive ``case`` witness uses the shortest spelling its region accepts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agm.agl.matchcompile import NonExhaustiveIssue, compile_program_matches, render_witness
from agm.agl.repl.session import ReplSession
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AmbiguousConstructorError
from agm.agl.typecheck.program import check_program
from tests._agl_helpers import AGL_TEST_CAPS
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


def _witnesses(graph: ModuleGraph) -> list[str]:
    """The rendered witness of each non-exhaustive ``case`` in *graph*, spelled where written."""
    resolved = resolve_program(graph)
    issues = compile_program_matches(check_program(resolved, AGL_TEST_CAPS)).issues
    return [
        render_witness(issue.witness, resolved.speller(issue.module_id))
        for issue in issues
        if isinstance(issue, NonExhaustiveIssue)
    ]


def _witness(tmp_path: Path, entry: str) -> str:
    """The rendered witness of the single non-exhaustive ``case`` in *entry*."""
    (witness,) = _witnesses(_graph(tmp_path, entry))
    return witness


def _case(header: str, scrutinee: str, indent: str = "") -> str:
    """Entry source: *header*, then a function matching a *scrutinee* parameter."""
    return f"{header}{indent}def f(v: {scrutinee}) -> int = case v of {_ARMS}\n"


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
    "use-at-root": pytest.param(_case("import lib\nuse lib::Color\n", "Color"), "Red", "Blue"),
    "use-in-scope": pytest.param(
        _case("import lib\nscope S\n  use lib::Color\n", "Color", "  ") + "end S\n", "Red", "Blue"
    ),
    "scope-import": pytest.param(
        _case("scope S\n  import lib::*\n", "Color", "  ") + "end S\n", "Red", "Blue"
    ),
    "own-record-beside-bare-member": pytest.param(
        _case("import lib::*\nrecord Blue\n  n: int\n", "Color"), "Red", "Blue"
    ),
    "own-scope-shadows-route-head": pytest.param(
        _case(
            "import lib\n\nscope lib\n  enum Color\n    | Red\n    | Blue\nend lib\n\n",
            "/lib::Color",
        ),
        "/lib::Color::Red",
        "/lib::Color::Blue",
    ),
    "imported-record-beside-bare-member": pytest.param(
        _case("import lib::*\nimport rb::*\n", "Color"), "Red", "Blue"
    ),
    "own-alias-of-routed-enum": pytest.param(
        _case("import lib\ntype Hue = lib::Color\n", "Hue"), "Red", "Blue"
    ),
    "use-rename-in-scope": pytest.param(
        _case("import lib\nscope S\n  use lib::Color as Hue\n", "Hue", "  ") + "end S\n",
        "Red",
        "Blue",
    ),
    "own-enum-in-scope-from-root": pytest.param(
        "scope S\n  enum Color\n    | Red\n    | Blue\nend S\n"
        "\n"
        "def f(v: S::Color) -> int = case v of @@\n",
        "S::Color::Red",
        "S::Color::Blue",
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
    ),
}

_RECORD_ROWS = {
    "referenced-member-own-path": pytest.param(
        "scope Outer\n\n  scope S\n    record Go\n      n: int\n  end S\nend Outer\n"
        "\n"
        "enum S\n  | Outer::S::Go\n  | Placeholder\n"
        "def f(v: S) -> int = case v of @@\n",
        "Placeholder",
        "Go(n = _)",
    ),
    "referenced-member-of-another-enum": pytest.param(
        "enum Review = Pass | Fail(reason: text)\nenum Verdict = ::Review::Pass | Maybe\n"
        "def f(v: Verdict) -> int = case v of @@\n",
        "Maybe",
        "Pass",
    ),
    "own-record-in-scope-from-root": pytest.param(
        "scope S\n  record Flag\n    on: bool\nend S\n\ndef f(v: S::Flag) -> int = case v of @@\n",
        "S::Flag(on = true)",
        "S::Flag(on = false)",
    ),
    "root-import": pytest.param(
        _case("import lib::*\n", "Flag"), "Flag(on = true)", "Flag(on = false)"
    ),
    "scope-import": pytest.param(
        _case("scope S\n  import lib::*\n", "Flag", "  ") + "end S\n",
        "Flag(on = true)",
        "Flag(on = false)",
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


def test_witness_ignores_a_hidden_binding_of_the_route_head(tmp_path: Path) -> None:
    """A name an import hides does not make the route head ``lib`` need its anchor."""
    entry = (
        "import lib\nimport lh::* hiding lib\n"
        "def f(v: lib::Color) -> int = case v of | lib::Color::Red => 0\n"
    )
    graph = make_graph_from_files(
        tmp_path, {"lib": _LIB, "lh": "def lib() -> int = 1\n", "entry": entry}
    )
    assert _witnesses(graph) == ["lib::Color::Blue"]


def test_imported_referenced_member_witness_spells_its_own_module_path(tmp_path: Path) -> None:
    modules = {
        "lib": "record Saved\n  id: int\nenum Status = ::Saved | Fresh(n: int)",
        "entry": (
            "import lib\ndef f(v: lib::Status) -> int = case v of | lib::Status::Fresh(n) => n\n"
        ),
    }
    assert _witnesses(make_graph_from_files(tmp_path / "missing", modules)) == [
        "lib::Saved(id = _)"
    ]
    modules["entry"] = modules["entry"].rstrip("\n") + " | lib::Saved(id = _) => 0\n"
    assert graph_verdict(make_graph_from_files(tmp_path / "complete", modules))[0] == "accepted"


def test_witness_skips_a_spelling_that_selects_another_declaration(tmp_path: Path) -> None:
    """``X`` names two modules declaring ``Color::Blue``, so only the route selects this one."""
    modules = {
        "lib": _LIB,
        "lc": "enum Color\n  | Blue\n  | Green\n",
        "entry": (
            "import lib\nimport lib as X\nimport lc as X\n"
            "def f(v: lib::Color) -> int = case v of | lib::Color::Red => 0@@\n"
        ),
    }

    def graph(root: Path, arms: str) -> ModuleGraph:
        return make_graph_from_files(
            root, {**modules, "entry": modules["entry"].replace(_ARMS, arms)}
        )

    assert _witnesses(graph(tmp_path / "missing", "")) == ["lib::Color::Blue"]
    completed = graph(tmp_path / "complete", " | lib::Color::Blue => 1")
    assert graph_verdict(completed)[0] == "accepted"


_ENTRY_SHADE = "enum Shade\n  | Red\n  | Green\n"

_REPAIR_ROWS = {
    "own-alias": ("import a\nimport b::*\ntype Hue = a::Color\n", "Hue::Red"),
    "renamed-import": ("import a::{Color as Hue}\nimport b::*\n", "Hue::Red"),
    "route-only-owner": ("import a::*\nimport b::*\n", "Color::Red"),
}


@pytest.mark.parametrize(("header", "repair"), _REPAIR_ROWS.values(), ids=_REPAIR_ROWS)
def test_ambiguity_repair_is_the_shortest_spelling_that_selects_the_member(
    tmp_path: Path, header: str, repair: str
) -> None:
    modules = {"a": _LIB, "b": _ENTRY_SHADE}
    graph = make_graph_from_files(tmp_path / "ambiguous", {**modules, "entry": f"{header}Red\n"})
    with pytest.raises(AmbiguousConstructorError) as caught:
        resolve_program(graph)
    assert caught.value.repair == repair
    repaired = f"{header}let probe = {repair}\n"
    assert (
        graph_verdict(make_graph_from_files(tmp_path / "repaired", {**modules, "entry": repaired}))[
            0
        ]
        == "accepted"
    )


def test_repl_witness_is_spelled_at_the_entry(tmp_path: Path) -> None:
    (tmp_path / "lib.agl").write_text(_LIB, encoding="utf-8")
    session = ReplSession(cwd=tmp_path, default_stdlib=False)
    assert not session.open()
    assert session.eval_entry("import lib::*").ok
    entry = "def f(v: Color) -> int = case v of | Red => 0"
    failed = session.eval_entry(entry)
    assert not failed.ok
    witness = re.search(r"missing pattern: (.*)\.$", failed.diagnostics[0].message)
    assert witness is not None
    assert witness.group(1) == "Blue"
    assert session.eval_entry(f"{entry} | {witness.group(1)} => 1").ok
