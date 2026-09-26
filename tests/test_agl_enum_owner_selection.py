"""Owner-qualified enum member selection: an enum's scope holds only its inline members.

A referenced member (``enum Status = ::Saved | Fresh(n: int)``) keeps its own
declaration path and its injected bare name; every owner-qualified spelling of
it -- in a value, type, pattern, ``is`` test, or method receiver -- is one
focused :class:`ReferencedMemberError`.

A module qualifier (``mylib::Red``, ``::Red``) selects the name its module
surface injects: its own constructor declaration, else the one inline member
of its root enums with that name. Each such spelling is checked in value,
pattern, and ``is`` position together, since all three select alike -- except
that a same-module ``def`` or ``let`` of that name claims the value spelling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from agm.agl.diagnostics import AglError, AglTypeError, ReferencedMemberError
from agm.agl.scope.symbols import AglScopeError, AmbiguousConstructorError, RouteClashError
from tests._agl_helpers import check_agl_program

_LIBRARIES = {
    "library": (
        "record Saved\n  id: int\nenum Status = ::Saved | Fresh(n: int)\ntype Alias = Status"
    ),
    "other": "record Shared\n  id: int",
    "middle": "import other\nenum Mixed = other::Shared | Own(n: int)",
    "review": "enum Review = Pass | Fail(reason: text)",
    "dup": "enum A = Red | Green\nenum B = Red | Blue",
    "claimed": "record Red\n  x: int\nenum A = Red | Green",
    "company/colors": "enum Color = Red | Blue(shade: int)",
    "funcs": "def pick(x: int) -> int = x",
    "nested": "scope inner\n  enum Deep = Violet | Indigo\nend inner",
    "gen": "enum Box[T] = Full(v: T) | Empty",
    "fnlib": "enum Color = Red | Green\ndef Red() -> int = 5",
    "letlib": "enum Color = Red | Green\nlet Red = 5",
    "rootenum": "enum Red = X\nenum Color = Red | Green",
    "structural": "type Red = int\nenum Color = Red | Green",
    "enumalias": "enum Other = X\ntype Red = Other\nenum Color = Red | Green",
    "generic": "enum Red[T] = X(v: T)\nenum Color = Red | Green",
    "kinds": "enum Color = A | B\nenum Kind = Color | Other",
    "palette": "enum P = Red | Green",
    "base": "enum Color = Red | Green",
    "mid": "import base\nexport base::{Color}\nenum Light = Red | Off",
    "boxes": "enum Box[T] = Full(v: T) | Empty\ntype IntFull = Box[int]::Full",
    "picker": (
        "enum Color = Red | Green(shade: int)\n"
        "def pick() -> Color = Green(shade = 1)\n"
        "def green() -> Color::Green = Green(shade = 2)"
    ),
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


_ALIASED = (
    "record Shared\n  id: int\ntype Alias = Shared\nenum E = ::Alias | Own\n"
    "let aliased: E = Shared(id = 1)\n"
)
"""An enum referencing a record through an alias: both names spell the referenced member."""


def _check(tmp_path: Path, entry: str, *, stdlib: bool = False) -> None:
    check_agl_program(tmp_path, {**_LIBRARIES, "entry": entry}, default_stdlib=stdlib)


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
        pytest.param(_ALIASED + "E::Shared(id = 1)", "Shared", id="value-alias-target"),
        pytest.param(_ALIASED + "E::Alias(id = 1)", "Alias", id="value-alias-spelling"),
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
        pytest.param(
            _ALIASED + "def f(s: E::Shared) -> int = s.id", "Shared", id="type-alias-target"
        ),
        pytest.param(
            _ALIASED + "def f(s: E::Alias) -> int = s.id", "Alias", id="type-alias-spelling"
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
        pytest.param(
            _ALIASED + "case aliased of | E::Shared(id) => id | _ => 0",
            "Shared",
            id="pattern-alias-target",
        ),
        pytest.param(_LOCAL + "status is Status::Saved", "Saved", id="is-owner"),
        pytest.param(_ALIASED + "aliased is E::Shared", "Shared", id="is-alias-target"),
        pytest.param(_LOCAL + "status is Current::Saved", "Saved", id="is-alias"),
        pytest.param(_LOCAL + "stored is Stored[int]::Saved", "Saved", id="is-applied"),
        pytest.param(_LOCAL + "verdict is Verdict::Pass", "Pass", id="is-other-enum-inline"),
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


@dataclass(frozen=True)
class _Spelling:
    """A qualified constructor spelling of a member of enum *subject*.

    *sample* is a *subject* value spelled independently; *arguments* and
    *binders* complete construction and destructuring of a field-bearing member.
    *stdlib* loads the standard library and its prelude.
    """

    header: str
    subject: str
    sample: str
    spelling: str
    arguments: str = ""
    binders: str = ""
    stdlib: bool = False

    def program(self, position: str) -> str:
        """Return the spelling written in constructor *position*."""
        if position == "value":
            return f"{self.header}let probe: {self.subject} = {self.spelling}{self.arguments}\n()"
        scrutinee = f"{self.header}let subject: {self.subject} = {self.sample}\n"
        if position == "pattern":
            return f"{scrutinee}case subject of | {self.spelling}{self.binders} => 0 | _ => 1"
        return f"{scrutinee}subject is {self.spelling}"


_Outcome = type[AglError] | None
"""The error a spelling raises, or ``None`` when it is accepted."""

_REVIEW = "enum Review = Pass | Fail(reason: text)\n"
_ROUTED_REVIEW = {"header": "import review\n", "subject": "review::Review"}
_COLORS = {"header": "import company/colors\n", "subject": "colors::Color"}
_MIXED = {"header": "import middle\nimport other\n", "subject": "middle::Mixed"}


@pytest.mark.parametrize("position", ["value", "pattern", "is"])
@pytest.mark.parametrize(
    ("spelling", "outcome"),
    [
        # A module route or ``::`` selects an inline member its surface injects.
        pytest.param(
            _Spelling(
                **_ROUTED_REVIEW,
                sample='review::Review::Fail(reason = "x")',
                spelling="review::Pass",
            ),
            None,
            id="route-fieldless",
        ),
        pytest.param(
            _Spelling(
                **_ROUTED_REVIEW,
                sample="review::Review::Pass",
                spelling="review::Fail",
                arguments='(reason = "late")',
                binders="(reason)",
            ),
            None,
            id="route-fields",
        ),
        pytest.param(
            _Spelling(
                **_ROUTED_REVIEW,
                sample="review::Review::Pass",
                spelling="/review::Fail",
                arguments='(reason = "late")',
                binders="(reason)",
            ),
            None,
            id="anchored-route",
        ),
        pytest.param(
            _Spelling(
                **_COLORS,
                sample="colors::Color::Red",
                spelling="colors::Blue",
                arguments="(shade = 1)",
                binders="(shade)",
            ),
            None,
            id="suffix-route",
        ),
        pytest.param(
            _Spelling(
                **_COLORS, sample="colors::Color::Blue(shade = 1)", spelling="company/colors::Red"
            ),
            None,
            id="full-route",
        ),
        pytest.param(
            _Spelling(
                header=_REVIEW,
                subject="Review",
                sample="Review::Pass",
                spelling="::Fail",
                arguments='(reason = "late")',
                binders="(reason)",
            ),
            None,
            id="current-module",
        ),
        pytest.param(
            _Spelling(
                header=_REVIEW + "enum Verdict = ::Review::Pass | Maybe\n",
                subject="Verdict",
                sample="Verdict::Maybe",
                spelling="::Pass",
            ),
            None,
            id="current-module-inline-member-another-enum-references",
        ),
        pytest.param(
            _Spelling(
                header="import dup hiding A::Red\n",
                subject="dup::B",
                sample="dup::B::Blue",
                spelling="dup::Red",
            ),
            None,
            id="same-named-member-hidden",
        ),
        # A referenced member is spelled at its own declaration path.
        pytest.param(
            _Spelling(
                **_MIXED,
                sample="middle::Mixed::Own(n = 1)",
                spelling="other::Shared",
                arguments="(id = 1)",
                binders="(id)",
            ),
            None,
            id="referenced-at-own-route",
        ),
        pytest.param(
            _Spelling(
                header="import library\n",
                subject="library::Status",
                sample="library::Status::Fresh(n = 1)",
                spelling="library::Saved",
                arguments="(id = 1)",
                binders="(id)",
            ),
            None,
            id="referenced-root-record-at-own-route",
        ),
        # A module surface injects no referenced member.
        pytest.param(
            _Spelling(
                **_MIXED,
                sample="middle::Mixed::Own(n = 1)",
                spelling="middle::Shared",
                arguments="(id = 1)",
                binders="(id)",
            ),
            ReferencedMemberError,
            id="route-referenced",
        ),
        pytest.param(
            _Spelling(
                header="import other\nenum Mixed = other::Shared | Own(n: int)\n",
                subject="Mixed",
                sample="Mixed::Own(n = 1)",
                spelling="::Shared",
                arguments="(id = 1)",
                binders="(id)",
            ),
            ReferencedMemberError,
            id="current-module-referenced-import",
        ),
        pytest.param(
            _Spelling(
                header=(
                    "scope checks\n"
                    "  enum Review = Pass | Fail(reason: text)\n"
                    "end checks\n\n"
                    "enum Verdict = ::checks::Review::Pass | Maybe\n"
                ),
                subject="Verdict",
                sample="Verdict::Maybe",
                spelling="::Pass",
            ),
            ReferencedMemberError,
            id="current-module-referenced-scoped-member",
        ),
        # Two inline members of one surface make the spelling ambiguous.
        pytest.param(
            _Spelling(
                header="import dup\n", subject="dup::A", sample="dup::A::Green", spelling="dup::Red"
            ),
            AglScopeError,
            id="route-ambiguous",
        ),
        pytest.param(
            _Spelling(
                header="enum A = Red | Green\nenum B = Red | Blue\n",
                subject="A",
                sample="A::Green",
                spelling="::Red",
            ),
            AglScopeError,
            id="current-module-ambiguous",
        ),
        # Only a declaration bearing a constructor claims the name: a module's
        # type without one leaves its inline member reachable.
        pytest.param(
            _Spelling(
                header="import rootenum\n",
                subject="rootenum::Color",
                sample="rootenum::Color::Green",
                spelling="rootenum::Red",
            ),
            None,
            id="route-member-named-like-a-root-enum",
        ),
        pytest.param(
            _Spelling(
                header="import structural\n",
                subject="structural::Color",
                sample="structural::Color::Green",
                spelling="structural::Red",
            ),
            None,
            id="route-member-named-like-a-structural-alias",
        ),
        pytest.param(
            _Spelling(
                header="import enumalias\n",
                subject="enumalias::Color",
                sample="enumalias::Color::Green",
                spelling="enumalias::Red",
            ),
            None,
            id="route-member-named-like-an-enum-alias",
        ),
        pytest.param(
            _Spelling(
                header="import generic\n",
                subject="generic::Color",
                sample="generic::Color::Green",
                spelling="generic::Red",
            ),
            None,
            id="route-member-named-like-a-generic-enum",
        ),
        pytest.param(
            _Spelling(
                header="import kinds\n",
                subject="kinds::Kind",
                sample="kinds::Kind::Other",
                spelling="kinds::Color",
            ),
            None,
            id="route-member-named-like-the-enum-of-another",
        ),
        pytest.param(
            _Spelling(
                header="enum Red = X\nenum Color = Red | Green\n",
                subject="Color",
                sample="Color::Green",
                spelling="::Red",
            ),
            None,
            id="current-module-member-named-like-a-root-enum",
        ),
        pytest.param(
            _Spelling(
                header="type Red = int\nenum Color = Red | Green\n",
                subject="Color",
                sample="Color::Green",
                spelling="::Red",
            ),
            None,
            id="current-module-member-named-like-a-structural-alias",
        ),
        pytest.param(
            _Spelling(
                header="enum Red[T] = X(v: T)\nenum Color = Red | Green\n",
                subject="Color",
                sample="Color::Green",
                spelling="Red",
            ),
            None,
            id="bare-member-named-like-a-generic-enum",
        ),
        # A route also injects the inline members of the enums it re-exports.
        pytest.param(
            _Spelling(
                header="import mid\n",
                subject="mid::Color",
                sample="mid::Color::Red",
                spelling="mid::Green",
            ),
            None,
            id="route-re-exported-enum-member",
        ),
        pytest.param(
            _Spelling(
                header="import mid\n",
                subject="mid::Color",
                sample="mid::Color::Green",
                spelling="mid::Red",
            ),
            AmbiguousConstructorError,
            id="route-re-exported-and-own-member",
        ),
        # A leading segment that is both a local scope or type and a module
        # route contributing the name is ambiguous, whether the route declares
        # the name or injects it.
        pytest.param(
            _Spelling(
                header="import palette\nenum palette = Red | Blue\n",
                subject="palette",
                sample="palette::Blue",
                spelling="palette::Red",
            ),
            RouteClashError,
            id="local-type-and-route-injecting-the-member",
        ),
        pytest.param(
            _Spelling(
                header=(
                    "import palette\n\n"
                    "scope palette\n"
                    "  record Red\n"
                    "end palette\n\n"
                    "enum W = ::palette::Red | Other\n"
                ),
                subject="W",
                sample="W::Other",
                spelling="palette::Red",
            ),
            RouteClashError,
            id="local-scope-and-route-injecting-the-member",
        ),
        pytest.param(
            _Spelling(
                header="import claimed\nenum claimed = Red | Blue\n",
                subject="claimed",
                sample="claimed::Blue",
                spelling="claimed::Red",
            ),
            RouteClashError,
            id="local-type-and-route-declaring-the-name",
        ),
        pytest.param(
            _Spelling(
                header=(
                    "import review\n\n"
                    "scope review\n"
                    "  def helper() -> int = 1\n"
                    "end review\n\n"
                    "enum Mine = Pass | Other\n"
                ),
                subject="Mine",
                sample="Mine::Other",
                spelling="review::Pass",
            ),
            RouteClashError,
            id="local-scope-lacking-the-name-and-route",
        ),
        # A record declaring the name owns the spelling; no member is injected over it.
        pytest.param(
            _Spelling(
                header="import claimed\n",
                subject="claimed::A",
                sample="claimed::A::Green",
                spelling="claimed::Red",
            ),
            AglTypeError,
            id="route-declared-record",
        ),
        pytest.param(
            _Spelling(
                header="record Red\n  x: int\nenum A = Red | Green\n",
                subject="A",
                sample="A::Green",
                spelling="::Red",
            ),
            AglTypeError,
            id="current-module-declared-record",
        ),
        # ``::`` names only this module's own declarations, never a builtin or prelude one.
        pytest.param(
            _Spelling(
                header="enum Plan = Retry(n: int) | Stop\n",
                subject="Plan",
                sample="Plan::Stop",
                spelling="::Retry",
                arguments="(n = 2)",
                binders="(n)",
                stdlib=True,
            ),
            None,
            id="current-module-member-spelled-like-a-builtin-member",
        ),
        pytest.param(
            _Spelling(
                header="enum Maybe = Some(v: int) | Nada\n",
                subject="Maybe",
                sample="Maybe::Nada",
                spelling="::Some",
                arguments="(v = 1)",
                binders="(v)",
                stdlib=True,
            ),
            None,
            id="current-module-member-spelled-like-a-prelude-member",
        ),
        pytest.param(
            _Spelling(
                header="enum Plan = Abort | Go\n",
                subject="Plan",
                sample="Plan::Go",
                spelling="::Abort",
                stdlib=True,
            ),
            None,
            id="current-module-member-spelled-like-a-builtin-exception",
        ),
        pytest.param(
            _Spelling(
                header="",
                subject="Option[int]",
                sample="Some(1)",
                spelling="::Some",
                arguments="(1)",
                binders="(_)",
                stdlib=True,
            ),
            AglScopeError,
            id="current-module-prelude-option-member",
        ),
        pytest.param(
            _Spelling(
                header="",
                subject="Result[int, text]",
                sample="Ok(1)",
                spelling="::Ok",
                arguments="(1)",
                binders="(_)",
                stdlib=True,
            ),
            AglScopeError,
            id="current-module-prelude-result-member",
        ),
        pytest.param(
            _Spelling(
                header="",
                subject="ParsePolicy",
                sample="ParsePolicy::Retry(n = 1)",
                spelling="::Retry",
                arguments="(n = 2)",
                binders="(n)",
                stdlib=True,
            ),
            AglScopeError,
            id="current-module-builtin-member",
        ),
        pytest.param(
            _Spelling(
                header="import review::{Review}\n",
                subject="Review",
                sample='Review::Fail(reason = "x")',
                spelling="::Pass",
            ),
            AglScopeError,
            id="current-module-imported-member",
        ),
        # A qualifier naming no module surface holding the member is a scope error.
        pytest.param(
            _Spelling(**_ROUTED_REVIEW, sample="review::Review::Pass", spelling="review::Missing"),
            AglScopeError,
            id="route-missing",
        ),
        pytest.param(
            _Spelling(
                header="import nested\n",
                subject="nested::inner::Deep",
                sample="nested::inner::Deep::Indigo",
                spelling="nested::Violet",
            ),
            AglScopeError,
            id="route-non-root-enum-member",
        ),
        pytest.param(
            _Spelling(
                header=_REVIEW, subject="Review", sample="Review::Pass", spelling="nowhere::Pass"
            ),
            AglScopeError,
            id="unknown-qualifier",
        ),
        # An import alias is a module qualifier; a ``use`` alias is not.
        pytest.param(
            _Spelling(
                header="import review as R\n",
                subject="R::Review",
                sample="R::Review::Pass",
                spelling="R::Fail",
                arguments='(reason = "late")',
                binders="(reason)",
            ),
            None,
            id="import-alias",
        ),
        pytest.param(
            _Spelling(
                header="import review\nuse review as R\n",
                subject="review::Review",
                sample="review::Review::Pass",
                spelling="R::Fail",
                arguments='(reason = "late")',
                binders="(reason)",
            ),
            AglScopeError,
            id="use-alias",
        ),
        pytest.param(
            _Spelling(
                header="import gen\n",
                subject="gen::Box[int]",
                sample="gen::Box[int]::Empty",
                spelling="gen::Full",
                arguments="(v = 1)",
                binders="(v)",
            ),
            None,
            id="route-generic-member",
        ),
        # A same-module ``def`` or ``let`` claims the value spelling; patterns and
        # ``is`` tests still select the member.
        pytest.param(
            _Spelling(
                header="import fnlib\n",
                subject="fnlib::Color",
                sample="fnlib::Color::Green",
                spelling="fnlib::Red",
            ),
            {"value": AglTypeError},
            id="route-def-claims-the-value",
        ),
        pytest.param(
            _Spelling(
                header="import letlib\n",
                subject="letlib::Color",
                sample="letlib::Color::Green",
                spelling="letlib::Red",
            ),
            {"value": AglTypeError},
            id="route-let-claims-the-value",
        ),
        pytest.param(
            _Spelling(
                header="enum Mine = Red | Other\ndef Red() -> int = 3\n",
                subject="Mine",
                sample="Mine::Other",
                spelling="::Red",
            ),
            {"value": AglTypeError},
            id="current-module-def-claims-the-value",
        ),
    ],
)
def test_module_qualified_spelling_is_accepted_alike_in_every_position(
    tmp_path: Path,
    spelling: _Spelling,
    outcome: _Outcome | Mapping[str, _Outcome],
    position: str,
) -> None:
    expected = outcome.get(position) if isinstance(outcome, Mapping) else outcome
    program = spelling.program(position)
    if expected is None:
        _check(tmp_path, program, stdlib=spelling.stdlib)
        return
    with pytest.raises(expected):
        _check(tmp_path, program, stdlib=spelling.stdlib)


def test_module_qualified_inline_member_is_a_first_class_constructor(tmp_path: Path) -> None:
    _check(
        tmp_path,
        "import review\n"
        "enum Opt[T] = Non | Som(value: T)\n"
        "let fail: text -> review::Review = review::Fail\n"
        "let som: int -> Opt[int] = ::Som\n"
        'let values: array[review::Review] = [fail("late"), /review::Pass]\n'
        "let opts: array[Opt[int]] = [som(1), ::Non, ::Som(2)]\n"
        "()",
    )


def test_module_qualified_declared_record_constructs_the_record(tmp_path: Path) -> None:
    _check(
        tmp_path,
        "import claimed\n"
        "let red: claimed::Red = claimed::Red(x = 1)\n"
        "case red of | claimed::Red(x) => x",
    )


@pytest.mark.parametrize(
    ("use", "error"),
    [
        pytest.param(
            "case p of | /review::Review::Missing => 0 | _ => 1", AglTypeError, id="missing-pattern"
        ),
        pytest.param("p is /review::Review::Missing", AglTypeError, id="missing-is"),
        pytest.param(
            "case p of | /dup::B::Red => 0 | _ => 1", AglTypeError, id="other-enum-pattern"
        ),
        pytest.param(
            "case p of | funcs::pick::Pass => 0 | _ => 1", AglTypeError, id="not-a-type-pattern"
        ),
        pytest.param(
            "case p of | ::Nope::Pass => 0 | _ => 1", AglScopeError, id="unknown-own-owner-pattern"
        ),
    ],
)
def test_module_routed_owner_selects_only_its_own_inline_member(
    tmp_path: Path, use: str, error: type[AglError]
) -> None:
    entry = "import review\nimport dup\nimport funcs\nlet p: review::Review = review::Pass\n" + use
    with pytest.raises(error):
        _check(tmp_path, entry)


@pytest.mark.parametrize(
    ("header", "spelling", "subject"),
    [
        pytest.param("import dup\n", "dup::Red", "dup::A", id="route"),
        pytest.param("import dup as D\n", "D::Red", "D::A", id="import-alias"),
        pytest.param("import mid\n", "mid::Red", "mid::Color", id="re-export"),
        pytest.param(
            "enum A = Red | Green\nenum B = Red | Blue\n", "::Red", "A", id="current-module"
        ),
        pytest.param("import dup::*\n", "Red", "dup::A", id="bare"),
    ],
)
def test_ambiguous_constructor_repair_resolves_where_written(
    tmp_path: Path, header: str, spelling: str, subject: str
) -> None:
    with pytest.raises(AmbiguousConstructorError) as caught:
        _check(tmp_path / "ambiguous", f"{header}let probe = {spelling}\n()")
    repair = caught.value.repair
    _check(tmp_path / "repaired", f"{header}let probe: {subject} = {repair}\nprobe is {repair}")


@pytest.mark.parametrize(
    "use",
    [
        pytest.param("case pick() of | Red => 0 | _ => 1", id="bare-pattern"),
        pytest.param("case pick() of | Red() => 0 | _ => 1", id="applied-pattern"),
        pytest.param("case pick() of | Green(shade) => shade | _ => 0", id="binding-pattern"),
        pytest.param("pick() is Red", id="is"),
        pytest.param("case green() of | Green(shade) => shade", id="member-typed-pattern"),
    ],
)
@pytest.mark.parametrize(
    ("header", "accepted"),
    [
        pytest.param("import picker::{pick, green}\n", False, id="function-only"),
        pytest.param("import picker::*\n", True, id="members-visible"),
    ],
)
def test_bare_member_spelling_needs_a_visible_constructor(
    tmp_path: Path, use: str, header: str, accepted: bool
) -> None:
    """The scrutinee's type directs a bare spelling only among visible constructors."""
    if accepted:
        _check(tmp_path, header + use)
        return
    with pytest.raises(AglError):
        _check(tmp_path, header + use)


_SCOPED_INT_FULL = (
    "enum Box[T] = Full(v: T) | Empty\n\n"
    "scope s\n"
    "  type IntFull = ::Box[int]::Full\n"
    "end s\n\n"
    'let b: Box[text] = Full(v = "x")\n'
)
_ROUTED_INT_FULL = 'import boxes\nlet b: boxes::Box[text] = boxes::Full(v = "x")\n'


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            _SCOPED_INT_FULL + "case b of | s::IntFull(v) => 0 | _ => 1", id="scope-pattern"
        ),
        pytest.param(_SCOPED_INT_FULL + "b is s::IntFull", id="scope-is"),
        pytest.param(
            _ROUTED_INT_FULL + "case b of | boxes::IntFull(v) => 0 | _ => 1", id="module-pattern"
        ),
        pytest.param(_ROUTED_INT_FULL + "b is boxes::IntFull", id="module-is"),
    ],
)
def test_qualified_member_alias_at_other_arguments_is_rejected(tmp_path: Path, entry: str) -> None:
    """A scope- or module-qualified alias selecting a member at other arguments matches nothing."""
    with pytest.raises(AglTypeError):
        _check(tmp_path, entry)
