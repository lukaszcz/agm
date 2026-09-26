"""Owner-qualified enum member selection: an enum's scope holds only its inline members.

A referenced member (``enum Status = ::Saved | Fresh(n: int)``) keeps its own
declaration path and its injected bare name; every owner-qualified spelling of
it -- in a value, type, pattern, ``is`` test, or method receiver -- is one
focused :class:`ReferencedMemberError`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import ReferencedMemberError
from agm.agl.scope.symbols import AglScopeError
from tests._agl_helpers import check_agl_program

_LIBRARIES = {
    "library": (
        "record Saved\n  id: int\nenum Status = ::Saved | Fresh(n: int)\ntype Alias = Status"
    ),
    "other": "record Shared\n  id: int",
    "middle": "import other\nenum Mixed = other::Shared | Own(n: int)",
    "review": "enum Review = Pass | Fail(reason: text)",
    "scoped": (
        "scope shapes\n"
        "  record Saved\n"
        "    id: int\n"
        "  enum Status = ::shapes::Saved | Fresh(n: int)\n"
        "end shapes"
    ),
}

_LOCAL = (
    "record Saved\n"
    "  id: int\n"
    "enum Status = ::Saved | Fresh(n: int)\n"
    "type Current = Status\n"
    "enum Stored[T] = ::Saved | Kept(value: T)\n"
    "enum Review = Pass | Fail(reason: text)\n"
    "enum Verdict = ::Review::Pass | Maybe\n"
    "let status: Status = Saved(id = 1)\n"
    "let stored: Stored[int] = Saved(id = 1)\n"
    "let verdict: Verdict = Verdict::Maybe\n"
)


def _check(tmp_path: Path, entry: str) -> None:
    check_agl_program(tmp_path, {**_LIBRARIES, "entry": entry}, default_stdlib=False)


@pytest.mark.parametrize(
    ("entry", "member"),
    [
        # Values.
        pytest.param(_LOCAL + "Status::Saved(id = 1)", "Saved", id="value-owner"),
        pytest.param(_LOCAL + "Current::Saved(id = 1)", "Saved", id="value-alias"),
        pytest.param(_LOCAL + "Stored[int]::Saved(id = 1)", "Saved", id="value-applied"),
        pytest.param(_LOCAL + "::Status::Saved(id = 1)", "Saved", id="value-anchored"),
        pytest.param(_LOCAL + "Verdict::Pass", "Pass", id="value-other-enum-inline"),
        pytest.param(
            "import library::Status\nStatus::Saved(id = 1)", "Saved", id="value-imported-owner"
        ),
        pytest.param(
            "import library\nlibrary::Status::Saved(id = 1)", "Saved", id="value-routed-owner"
        ),
        pytest.param(
            "import library\nlibrary::Alias::Saved(id = 1)", "Saved", id="value-routed-alias"
        ),
        pytest.param(
            "import library\nuse library::Status as S\nS::Saved(id = 1)",
            "Saved",
            id="value-use-alias",
        ),
        pytest.param(
            "import library\nuse library::{Status}\nStatus::Saved(id = 1)",
            "Saved",
            id="value-used-owner",
        ),
        pytest.param(
            "import scoped\nscoped::shapes::Status::Saved(id = 1)", "Saved", id="value-scoped-owner"
        ),
        pytest.param(
            "record A\n  x: int\nenum E = ::A\nE::A(x = 1)", "A", id="value-all-referenced"
        ),
        # Types.
        pytest.param(_LOCAL + "def f(s: Status::Saved) -> int = s.id", "Saved", id="type-owner"),
        pytest.param(_LOCAL + "def f(s: Current::Saved) -> int = s.id", "Saved", id="type-alias"),
        pytest.param(
            _LOCAL + "def f(s: Stored[int]::Saved) -> int = s.id", "Saved", id="type-applied"
        ),
        pytest.param(
            "import library\ndef f(s: library::Status::Saved) -> int = s.id",
            "Saved",
            id="type-routed-owner",
        ),
        pytest.param(
            "import library::Status\ndef f(s: Status::Saved) -> int = s.id",
            "Saved",
            id="type-imported-owner",
        ),
        pytest.param(
            "record Holder\n  saved: Status::Saved\n" + _LOCAL,
            "Saved",
            id="type-before-enum-declaration",
        ),
        pytest.param(
            "record A\n  x: int\nenum E = ::A | Wrap(a: E::A)", "A", id="type-in-own-enum-body"
        ),
        pytest.param(
            "record Box[T]\n  v: T\nenum W[T] = ::Box[T] | Other\n"
            "def f(b: W::Box[int]) -> int = b.v",
            "Box",
            id="type-applied-member",
        ),
        # Patterns.
        pytest.param(
            _LOCAL + "case status of | Status::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-owner",
        ),
        pytest.param(
            _LOCAL + "case status of | Current::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-alias",
        ),
        pytest.param(
            _LOCAL + "case status of | ::Status::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-anchored",
        ),
        pytest.param(
            _LOCAL + "case stored of | Stored[int]::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-applied",
        ),
        pytest.param(
            _LOCAL + "case Saved(id = 2) of | Status::Saved(id) => id",
            "Saved",
            id="pattern-owner-on-member-value",
        ),
        pytest.param(
            _LOCAL + "case verdict of | Verdict::Pass => 0 | _ => 1",
            "Pass",
            id="pattern-other-enum-inline",
        ),
        pytest.param(
            _LOCAL + "case verdict of | ::Pass => 0 | _ => 1", "Pass", id="pattern-current-module"
        ),
        pytest.param(
            "import library::Status\n"
            "let s: Status = library::Saved(id = 1)\n"
            "case s of | Status::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-imported-owner",
        ),
        pytest.param(
            "import library\n"
            "let s: library::Status = library::Saved(id = 1)\n"
            "case s of | library::Status::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-routed-owner",
        ),
        pytest.param(
            "import library\n"
            "use library::Status as S\n"
            "let s: library::Status = library::Saved(id = 1)\n"
            "case s of | S::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-use-alias",
        ),
        pytest.param(
            "import middle\n"
            "import other\n"
            "let m: middle::Mixed = other::Shared(id = 1)\n"
            "case m of | middle::Shared(id) => id | _ => 0",
            "Shared",
            id="pattern-module-route",
        ),
        pytest.param(
            "import scoped\n"
            "let s: scoped::shapes::Status = scoped::shapes::Saved(id = 1)\n"
            "case s of | scoped::shapes::Status::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-scoped-owner",
        ),
        pytest.param(
            "import library\n"
            "let s: library::Status = library::Saved(id = 1)\n"
            "case s of | /library::Status::Saved(id) => id | _ => 0",
            "Saved",
            id="pattern-anchored-module-route",
        ),
        # ``is`` tests.
        pytest.param(_LOCAL + "status is Status::Saved", "Saved", id="is-owner"),
        pytest.param(_LOCAL + "status is Current::Saved", "Saved", id="is-alias"),
        pytest.param(_LOCAL + "stored is Stored[int]::Saved", "Saved", id="is-applied"),
        pytest.param(_LOCAL + "verdict is Verdict::Pass", "Pass", id="is-other-enum-inline"),
        pytest.param(_LOCAL + "verdict is ::Pass", "Pass", id="is-current-module"),
        pytest.param(
            "import middle\n"
            "import other\n"
            "let m: middle::Mixed = other::Shared(id = 1)\n"
            "m is middle::Shared",
            "Shared",
            id="is-module-route",
        ),
        pytest.param(
            "import scoped\n"
            "let s: scoped::shapes::Status = scoped::shapes::Saved(id = 1)\n"
            "s is scoped::shapes::Status::Saved",
            "Saved",
            id="is-scoped-owner",
        ),
        pytest.param(
            "import library\n"
            "let s: library::Status = library::Saved(id = 1)\n"
            "s is /library::Status::Saved",
            "Saved",
            id="is-anchored-module-route",
        ),
        # Method receivers.
        pytest.param(
            _LOCAL + "def Status::Saved::label(self) -> int = self.id", "Saved", id="method-owner"
        ),
        pytest.param(
            _LOCAL + "scope Status::Saved\n  def label(self) -> int = self.id\nend Status::Saved",
            "Saved",
            id="method-region",
        ),
        pytest.param(
            "import library::Status\ndef Status::Saved::label(self) -> int = self.id",
            "Saved",
            id="method-imported-owner",
        ),
    ],
)
def test_owner_qualified_referenced_member_is_a_focused_error(
    tmp_path: Path, entry: str, member: str
) -> None:
    with pytest.raises(ReferencedMemberError) as raised:
        _check(tmp_path, entry)

    assert raised.value.member == member


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            _LOCAL
            + "let a = Status::Fresh(n = 1)\n"
            + "let b: Status = Current::Fresh(n = 2)\n"
            + "let c = ::Saved(id = 3)\n"
            + "let d = ::Maybe\n"
            + "let e = case status of | ::Saved(id) => id | ::Status::Fresh(n) => n\n"
            + "let f = case verdict of | ::Maybe => 0 | Review::Pass => 1\n"
            + "let g = verdict is Pass\n"
            + "let h = verdict is ::Maybe\n"
            + "def label(s: Status::Fresh) -> int = s.n\n"
            + "def Status::Fresh::count(self) -> int = self.n\n"
            + "def Saved::count(self) -> int = self.id",
            id="local",
        ),
        pytest.param(
            "record Box::Item\n"
            "  n: int\n"
            "enum Box = Empty | Box::Item\n"
            "let b: Box = Box::Item(n = 1)\n"
            "def x(b: Box) -> int = case b of | Box::Item(n) => n | Empty => 0\n"
            "def y(b: Box) -> bool = b is Box::Item\n"
            "def f(i: Box::Item) -> int = i.n\n"
            "def Box::Item::g(self) -> int = self.n",
            id="referenced-member-declared-in-enum-scope",
        ),
        pytest.param(
            "import review\n"
            "let r: review::Review = review::Review::Pass\n"
            "let a = case r of | review::Pass => 0 | review::Fail(reason) => 1\n"
            "let b = r is review::Pass",
            id="module-qualified-inline",
        ),
        pytest.param(
            "import middle\n"
            "import other\n"
            "let m: middle::Mixed = other::Shared(id = 1)\n"
            "let a = case m of | other::Shared(id) => id | middle::Own(n) => n\n"
            "let b = m is other::Shared\n"
            "let c = m is middle::Own",
            id="referenced-at-own-module-path",
        ),
        pytest.param(
            "import library\n"
            "let s: library::Status = library::Status::Fresh(n = 1)\n"
            "let a = case s of | /library::Status::Fresh(n) => n | /library::Saved(id) => id\n"
            "let b = s is /library::Status::Fresh",
            id="anchored-module-route",
        ),
        pytest.param(
            "import scoped\n"
            "let s: scoped::shapes::Status = scoped::shapes::Saved(id = 1)\n"
            "let a = case s of\n"
            "  | scoped::shapes::Saved(id) => id\n"
            "  | scoped::shapes::Status::Fresh(n) => n\n"
            "let b = s is scoped::shapes::Status::Fresh\n"
            "let c: scoped::shapes::Status = scoped::shapes::Status::Fresh(n = 1)",
            id="scoped-owner",
        ),
        pytest.param(
            "import library\n"
            "use library::Status as S\n"
            "let s: library::Status = S::Fresh(n = 1)\n"
            "let a = case s of | S::Fresh(n) => n | library::Saved(id) => id\n"
            "let b = s is S::Fresh",
            id="use-alias-inline",
        ),
    ],
)
def test_owner_qualified_inline_members_and_own_paths_are_accepted(
    tmp_path: Path, entry: str
) -> None:
    _check(tmp_path, entry)


def test_scoped_import_item_binds_its_path_not_the_terminal_name(tmp_path: Path) -> None:
    """``import m::{Status::Fresh}`` makes the path ``Status::Fresh`` bare, not ``Fresh``."""
    header = "import library::{Status::Fresh}\n"
    _check(
        tmp_path, header + "let fresh = Status::Fresh(n = 1)\ndef f(s: Status::Fresh) -> int = s.n"
    )

    with pytest.raises(AglScopeError):
        _check(tmp_path, header + "Fresh(n = 1)")


def test_import_item_selecting_a_referenced_member_through_its_enum_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(AglScopeError):
        _check(tmp_path, "import library::{Status::Saved}\n()")


def test_unknown_member_of_a_used_enum_owner_is_a_scope_error(tmp_path: Path) -> None:
    with pytest.raises(AglScopeError):
        _check(tmp_path, "import library\nuse library::{Status}\nStatus::Missing(id = 1)")
