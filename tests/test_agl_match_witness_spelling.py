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


_OPT = "enum Opt[T]\n  | Nn\n  | Sm(value: T)\n"

_REMOTE = "enum Remote\n  | empty\n  | item(value: int)\n"

_OWNERS = "enum Owner\n  | block\n  | free\nenum Twin\n  | block\n  | free\n"

_MODULES = {
    "lib": _LIB,
    "rb": _RB,
    "g": _OPT,
    "la": "import lib\ntype Hue = lib::Color\n",
    "la2": "import g\ntype IntOpt = g::Opt[int]\n",
    "la3": "import lib\ntype F = lib::Flag\n",
    "l3": "scope S\n  enum Color\n    | Red\n    | Blue\nend S\n",
    "library/remote": _REMOTE + "type Alias = Remote\n",
    "lib2": _OWNERS,
    "helpers/Owner": "def block() -> int = 1\n",
    "support/Status": "enum Status | External\n",
    "support/StatusFn": "enum Status | External\ndef Missing() -> int = 1\n",
    "lh": "scope lib\n  enum Color\n    | Red\n    | Green\nend lib\n",
}


def _graph(tmp_path: Path, entry: str) -> ModuleGraph:
    return make_graph_from_files(tmp_path, {**_MODULES, "entry": entry})


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
    "imported-alias-route": pytest.param(
        _case("import la\n", "la::Hue"), "la::Hue::Red", "la::Hue::Blue"
    ),
    "imported-alias-renamed-route": pytest.param(
        _case("import la as X\n", "X::Hue"), "X::Hue::Red", "X::Hue::Blue"
    ),
    "imported-applied-alias-route": pytest.param(
        _case("import la2\n", "la2::IntOpt"), "la2::IntOpt::Nn", "la2::IntOpt::Sm(value = _)"
    ),
    "imported-applied-alias-glob": pytest.param(
        _case("import la2::*\n", "IntOpt"), "Nn", "Sm(value = _)"
    ),
    "nested-through-imported-alias": pytest.param(
        _case("import la\nrecord Box\n  c: la::Hue\n", "Box"),
        "Box(c = la::Hue::Red)",
        "Box(c = la::Hue::Blue)",
    ),
    "enum-in-imported-scope-glob": pytest.param(
        _case("import l3::*\n", "S::Color"), "S::Color::Red", "S::Color::Blue"
    ),
    "remote-alias-through-renamed-route": pytest.param(
        _case("import library/remote as r\n", "r::Alias"),
        "r::Alias::empty",
        "r::Alias::item(value = _)",
    ),
    "remote-suffix-route": pytest.param(
        _case("import library/remote\n", "remote::Remote"),
        "remote::Remote::empty",
        "remote::Alias::item(value = _)",
    ),
}

_UNWRITABLE_ROWS = {
    "glob-hiding-the-member": pytest.param(
        _case("import lib::* hiding Color::Blue\n", "Color"), "Red", "_"
    ),
    "route-hiding-the-member": pytest.param(
        _case("import lib hiding Color::Blue\n", "lib::Color"), "lib::Color::Red", "_"
    ),
    "glob-hiding-an-owner-member": pytest.param(
        _case("import lib2::* hiding Owner::block\nimport helpers/Owner\n", "Owner"),
        "lib2::Owner::free",
        "_",
    ),
}

_SHADOWED_ROWS = {
    "own-enum-beats-imported-same-name": pytest.param(
        "import support/Status\nenum Status | Ready | Missing\n"
        "def f(Missing: int, v: Status) -> int = case v of @@\n",
        "Status::Ready",
        "Missing",
    ),
    "own-enum-beats-imported-function-and-owner": pytest.param(
        "import support/StatusFn\nenum Status | Ready | Missing\n"
        "def f(Missing: int, v: Status) -> int = case v of @@\n",
        "Status::Ready",
        "Missing",
    ),
    "builtin-option-member-is-bare": pytest.param(
        "def f(v: Option[int]) -> int = case v of @@\n", "Option::Some(value)", "None"
    ),
    "lexical-binding-of-a-member-name": pytest.param(
        "enum Choice\n  | empty\n  | item(value: int)\n"
        "def f(item: int, v: Choice) -> int = case v of @@\n",
        "Choice::empty",
        "item(value = _)",
    ),
    "later-lexical-binding-of-a-member-name": pytest.param(
        "enum Choice\n  | empty\n  | item(value: int)\n"
        "def f(v: Choice) -> int =\n  let result = case v of @@\n  let item = 1\n  result\n",
        "Choice::empty",
        "item(value = _)",
    ),
    "renamed-item-import": pytest.param(
        "import library/remote::{Remote as R}\ndef f(item: int, v: R) -> int = case v of @@\n",
        "R::empty",
        "item(value = _)",
    ),
    "generic-own-alias-owner": pytest.param(
        "enum Remote[T]\n  | empty\n  | item(value: T)\ntype Alias[T] = Remote[T]\n"
        "def f(item: int, v: Remote[int]) -> int = case v of @@\n",
        "Remote::empty",
        "item(value = _)",
    ),
    "own-enum-beside-same-named-route": pytest.param(
        "import library/remote as Choice\nenum Choice\n  | empty\n  | item(value: int)\n"
        "def f(item: int, v: ::Choice) -> int = case v of @@\n",
        "::Choice::empty",
        "item(value = _)",
    ),
    "own-generic-enum-beside-same-named-route": pytest.param(
        "import library/remote as Remote\nenum Remote[T]\n  | empty\n  | item(value: T)\n"
        "def f(item: int, v: ::Remote[int]) -> int = case v of @@\n",
        "::Remote::empty",
        "item(value = _)",
    ),
    "own-owner-enum-beside-route-of-that-name": pytest.param(
        f"import helpers/Owner\n{_OWNERS}def f(v: Owner) -> int = case v of @@\n",
        "Owner::free",
        "block",
    ),
    "imported-owner-enum-beside-route-of-that-name": pytest.param(
        "import lib2::*\nimport helpers/Owner\ndef f(v: Owner) -> int = case v of @@\n",
        "lib2::Owner::free",
        "block",
    ),
    "route-head-beside-an-open-import-scope": pytest.param(
        "import lib\nimport lh::*\ndef f(v: /lib::Color) -> int = case v of @@\n",
        "/lib::Color::Red",
        "lib::Color::Blue",
    ),
    "own-enum-clashing-with-an-imported-rename": pytest.param(
        "import library/remote::{Remote as Clash}\nenum Clash\n  | local\n"
        "def f(empty: int, item: int, v: /library/remote::Remote) -> int = case v of @@\n",
        "/library/remote::Remote::empty",
        "item(value = _)",
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
    "imported-alias-route": pytest.param(
        _case("import la3\n", "la3::F"), "la3::F(on = true)", "la3::F(on = false)"
    ),
    "imported-alias-glob": pytest.param(
        _case("import la3::*\n", "F"), "F(on = true)", "F(on = false)"
    ),
    "own-alias": pytest.param(
        _case("import lib\ntype F = lib::Flag\n", "F"), "F(on = true)", "F(on = false)"
    ),
    "renamed-import": pytest.param(
        _case("import lib::{Flag as F}\n", "F"), "F(on = true)", "F(on = false)"
    ),
    "aliases-tie-in-declaration-order": pytest.param(
        _case("import lib\ntype Zz = lib::Flag\ntype Aa = lib::Flag\n", "Zz"),
        "Zz(on = true)",
        "Zz(on = false)",
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


@pytest.mark.parametrize(
    ("entry", "covered", "expected"), _SHADOWED_ROWS.values(), ids=_SHADOWED_ROWS
)
def test_witness_is_spelled_to_survive_what_shadows_the_names_it_uses(
    tmp_path: Path, entry: str, covered: str, expected: str
) -> None:
    _assert_witness_completes(tmp_path, entry, covered, expected)


@pytest.mark.parametrize(("entry", "covered", "expected"), _RECORD_ROWS.values(), ids=_RECORD_ROWS)
def test_record_witness_is_the_shortest_spelling_at_the_case(
    tmp_path: Path, entry: str, covered: str, expected: str
) -> None:
    _assert_witness_completes(tmp_path, entry, covered, expected)


@pytest.mark.parametrize(
    ("entry", "covered", "expected"), _UNWRITABLE_ROWS.values(), ids=_UNWRITABLE_ROWS
)
def test_witness_no_spelling_reaches_is_a_wildcard(
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
    "imported-alias-glob": ("import la::*\nimport b::*\n", "Hue::Red"),
    "imported-alias-item": ("import la::{Hue}\nimport b::*\n", "Hue::Red"),
}


@pytest.mark.parametrize(("header", "repair"), _REPAIR_ROWS.values(), ids=_REPAIR_ROWS)
def test_ambiguity_repair_is_the_shortest_spelling_that_selects_the_member(
    tmp_path: Path, header: str, repair: str
) -> None:
    modules = {"a": _LIB, "b": _ENTRY_SHADE, "la": _MODULES["la"].replace("lib", "a")}
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


def test_ambiguity_repair_ignores_a_type_parameter_shadowing_the_owner(tmp_path: Path) -> None:
    modules = {"a": _LIB, "b": _ENTRY_SHADE}
    header = "import a::*\nimport b::*\ndef f[Color](x: Color) -> Color =\n"
    graph = make_graph_from_files(
        tmp_path / "ambiguous", {**modules, "entry": f"{header}  let z = Red\n  x\n"}
    )
    with pytest.raises(AmbiguousConstructorError) as caught:
        resolve_program(graph)
    repaired = f"{header}  let z = {caught.value.repair}\n  x\n"
    graph = make_graph_from_files(tmp_path / "repaired", {**modules, "entry": repaired})
    assert graph_verdict(graph)[0] == "accepted"


def _repl_witness(tmp_path: Path, setup: list[str], entry: str) -> str:
    """The witness an ``entry`` of a session after *setup* is rejected with, as written back."""
    (tmp_path / "lib.agl").write_text(_LIB, encoding="utf-8")
    session = ReplSession(cwd=tmp_path, default_stdlib=False)
    assert not session.open()
    for line in setup:
        assert session.eval_entry(line).ok
    failed = session.eval_entry(entry)
    assert not failed.ok
    witness = re.search(r"missing pattern: (.*)\.$", failed.diagnostics[0].message)
    assert witness is not None
    assert session.eval_entry(f"{entry} | {witness.group(1)} => 1").ok
    return witness.group(1)


def test_repl_witness_is_spelled_through_a_session_alias(tmp_path: Path) -> None:
    entry = "def f(v: lib::Flag) -> int = case v of | lib::Flag(on = true) => 0"
    assert _repl_witness(tmp_path, ["import lib", "type F = lib::Flag"], entry) == "F(on = false)"


def test_repl_witness_of_an_enum_is_spelled_through_a_session_alias(tmp_path: Path) -> None:
    entry = "def f(v: Hue) -> int = case v of | Red => 0"
    assert _repl_witness(tmp_path, ["import lib", "type Hue = lib::Color"], entry) == "Blue"


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
