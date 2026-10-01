"""Scope's one member selection for a qualified value, pattern, ``is`` and cast.

Every class here probes the same qualifier spelling in the four positions
that select a member through a qualifier chain at the value level -- a
value (constructor call or reference), a constructor pattern, an ``is``
test and an ``as?`` cast -- in the file part and every REPL grouping (see
:mod:`tests.agl.qualifier_support`), asserting the phase, the class and the
chain's own span, or the accepted identity:

- a scope region never hides a same-spelled imported type: the type's
  members stay reachable beside the region's own;
- an orphan method on a prelude type keeps that type's own members
  reachable through its qualifier;
- hiding is decided on the outer owner's own member table, nested records
  included, and a member no candidate selects is hidden when at least one
  candidate hides it, else unknown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError, HiddenMemberError
from agm.agl.scope.symbols import UnknownMemberError
from tests.agl.qualifier_support import (
    Part,
    Phase,
    assert_verdicts,
    probe_table,
    verdict_parts,
)

_ACCEPTED: tuple[Phase, type[BaseException] | type[None]] = ("accepted", type(None))


def _rejected(
    cls: type[BaseException],
) -> tuple[Phase, type[BaseException] | type[None]]:
    return ("scope", cls)


# ---------------------------------------------------------------------------
# A ``use``-opened region's step decides the leading segment first.
# ---------------------------------------------------------------------------

_REGION_MODULES = {
    "shapes": "scope Geo\n  record Point\n    x: int\nend Geo\n",
    "tl": "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n",
}
_REGION_HEADER = ("import tl::*", "import shapes")


def _region(*items: str, method: bool) -> str:
    lines = ["scope r", "  use shapes::*"]
    if method:
        lines.append("  def Geo::f() -> int = 1")
    lines.extend(f"  {line}" for item in items for line in item.split("\n"))
    lines.append("end r")
    return "\n".join(lines)


def _region_inner_probes(*, method: bool) -> dict[str, str]:
    return {
        "value": _region("def q() = Geo::Inner(y = 1)", method=method) + "\nr::q()",
        "pattern": _region(
            "def g(v: tl::Geo::Inner) -> int =\n  case v of\n    | Geo::Inner(y) => y",
            method=method,
        )
        + "\nr::g",
        "is": _region("def q() = 1 is Geo::Inner", method=method),
        "cast": _region("def q() = 1 as? Geo::Inner", method=method),
    }


class TestUseRegionBesideImportedType:
    """A ``use``-opened region inside ``scope r`` never hides the root's imported ``Geo``.

    The root's wildcard-imported ``record Geo`` keeps its nested ``Inner``
    reachable beside the region's ``Point``, with or without a ``def Geo::f``
    creating a same-spelled method path inside ``r``.
    """

    @pytest.mark.parametrize("method", [True, False], ids=["method", "no-method"])
    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_imported_nested_type_is_reached(
        self, tmp_path: Path, method: bool, part: Part
    ) -> None:
        probes = _region_inner_probes(method=method)
        assert_verdicts(
            tmp_path,
            _REGION_MODULES,
            _REGION_HEADER,
            probe_table(
                probes,
                {
                    "value": _ACCEPTED,
                    "pattern": _ACCEPTED,
                    "is": ("typecheck", AglTypeError),
                    "cast": ("typecheck", AglTypeError),
                },
                span_texts={"is": "1 is Geo::Inner", "cast": "1 as? Geo::Inner"},
                identities={
                    "value": "record tl::Geo::Inner\n  y: int",
                    "pattern": "tl::Geo::Inner -> int",
                },
            ),
            part=part,
        )

    @pytest.mark.parametrize("method", [True, False], ids=["method", "no-method"])
    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_region_member_is_accepted(self, tmp_path: Path, method: bool, part: Part) -> None:
        probe = _region("def q() = Geo::Point(x = 1)", method=method) + "\nr::q()"
        assert_verdicts(
            tmp_path,
            _REGION_MODULES,
            _REGION_HEADER,
            probe_table(
                {"value": probe},
                {"value": _ACCEPTED},
                identities={"value": "record shapes::Geo::Point\n  x: int"},
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_method_path_member_is_accepted(self, tmp_path: Path, part: Part) -> None:
        probe = _region("def q() = Geo::f()", method=True) + "\nr::q()"
        assert_verdicts(
            tmp_path,
            _REGION_MODULES,
            _REGION_HEADER,
            probe_table({"value": probe}, {"value": _ACCEPTED}, identities={"value": "int"}),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_region_without_method_path_lacks_it(self, tmp_path: Path, part: Part) -> None:
        probe = _region("def q() = Geo::f()", method=False)
        assert_verdicts(
            tmp_path,
            _REGION_MODULES,
            _REGION_HEADER,
            probe_table(
                {"value": probe},
                {"value": _rejected(UnknownMemberError)},
                span_texts={"value": "Geo::f"},
            ),
            part=part,
        )


# ---------------------------------------------------------------------------
# An orphan method keeps a prelude type's own members reachable.
# ---------------------------------------------------------------------------

_PRELUDE_ORPHAN_HEADER = ("def Option::m[T](self) -> int = 1",)
_PRELUDE_SUBJECT = "let o: Option[int] = None"


class TestOrphanMethodKeepsPreludeEnumMembers:
    """``def Option::m`` leaves the prelude ``Option::Some`` selectable in a pattern."""

    def test_member_is_selected(self, tmp_path: Path) -> None:
        probes = {
            "pattern": f"{_PRELUDE_SUBJECT}\n"
            "case o of\n  | Option::Some(value) => value\n  | _ => 0",
            "is": f"{_PRELUDE_SUBJECT}\no is Option::Some",
        }
        assert_verdicts(
            tmp_path,
            {},
            _PRELUDE_ORPHAN_HEADER,
            probe_table(
                probes,
                {key: _ACCEPTED for key in probes},
                identities={"pattern": "int", "is": "bool"},
            ),
        )


# ---------------------------------------------------------------------------
# Hiding is decided on the outer owner's own member table.
# ---------------------------------------------------------------------------

_NESTED_LIB = {"tl": "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"}
_NESTED_HIDDEN_HEADER = ("import tl::* hiding Geo::Inner",)
_NESTED_HIDDEN_ORPHAN_HEADER = (
    "import tl::* hiding Geo::Inner",
    "def Geo::m(self) -> int = 1",
)
_NESTED_HIDDEN_PROBES = {
    "pattern": "case 1 of\n  | Geo::Inner(y) => y",
    "is": "1 is Geo::Inner",
    "cast": "1 as? Geo::Inner",
}
_ENUM_NESTED_LIB = {"en": "enum Color\n  | Red\n  | Blue\nrecord Color::Extra\n  z: int\n"}
_ENUM_NESTED_HIDDEN_HEADER = ("import en::* hiding Color::Extra",)
_ENUM_NESTED_HIDDEN_PROBES = {
    "pattern": "case 1 of\n  | Color::Extra(z) => z",
    "is": "1 is Color::Extra",
    "cast": "1 as? Color::Extra",
}


class TestHiddenNestedRecordThroughItsOwner:
    """``hiding Geo::Inner`` hides the nested record ``Geo::Inner`` names, in every position.

    Hiding is read from the outer owner's own member table, whether or not
    an orphan method on the owner creates a same-spelled method path, and
    for an enum owner's nested record as for a record owner's.
    The value position is probed in :mod:`tests.test_agl_hidden_member_selection`.
    """

    def test_record_owner(self, tmp_path: Path) -> None:
        assert_verdicts(
            tmp_path,
            _NESTED_LIB,
            _NESTED_HIDDEN_HEADER,
            probe_table(
                _NESTED_HIDDEN_PROBES,
                {key: _rejected(HiddenMemberError) for key in _NESTED_HIDDEN_PROBES},
                span_texts={key: "Geo::Inner" for key in _NESTED_HIDDEN_PROBES},
            ),
        )

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_record_owner_with_orphan_method(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _NESTED_LIB,
            _NESTED_HIDDEN_ORPHAN_HEADER,
            probe_table(
                _NESTED_HIDDEN_PROBES,
                {key: _rejected(HiddenMemberError) for key in _NESTED_HIDDEN_PROBES},
                span_texts={key: "Geo::Inner" for key in _NESTED_HIDDEN_PROBES},
            ),
            part=part,
        )

    def test_enum_owner(self, tmp_path: Path) -> None:
        assert_verdicts(
            tmp_path,
            _ENUM_NESTED_LIB,
            _ENUM_NESTED_HIDDEN_HEADER,
            probe_table(
                _ENUM_NESTED_HIDDEN_PROBES,
                {key: _rejected(HiddenMemberError) for key in _ENUM_NESTED_HIDDEN_PROBES},
                span_texts={key: "Color::Extra" for key in _ENUM_NESTED_HIDDEN_PROBES},
            ),
        )


# ---------------------------------------------------------------------------
# One full-path-first step over several candidates at that step.
# ---------------------------------------------------------------------------

_TWO_ENUM_LIB = {
    "en": "enum Color\n  | Red\n  | Blue\n",
    "en2": "enum Color\n  | Green\n",
    "en3": "enum Color\n  | Red\n  | Green\n",
}
_HIDDEN_AND_MISSING_HEADER = ("import en::* hiding Color::Red", "import en2::*")
_HIDDEN_AND_PRESENT_HEADER = ("import en::* hiding Color::Red", "import en3::*")
_TWO_ENUM_SUBJECT = "let v: en::Color = en::Color::Blue"


def _two_enum_probes(member: str) -> dict[str, str]:
    q = f"Color::{member}"
    return {
        "value": q,
        "pattern": f"{_TWO_ENUM_SUBJECT}\ncase v of\n  | {q} => 1\n  | _ => 2",
        "is": f"{_TWO_ENUM_SUBJECT}\nv is {q}",
        "cast": f"{_TWO_ENUM_SUBJECT}\nv as? {q}",
    }


class TestSameLevelCandidatesSelectFullPathFirst:
    """Two wildcard-imported ``Color`` owners: the member alone decides between them.

    A member one candidate hides and the other lacks is hidden; a member one
    candidate hides and the other declares selects the other.
    """

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_hidden_on_one_missing_on_other(self, tmp_path: Path, part: Part) -> None:
        probes = _two_enum_probes("Red")
        assert_verdicts(
            tmp_path,
            _TWO_ENUM_LIB,
            _HIDDEN_AND_MISSING_HEADER,
            probe_table(
                probes,
                {key: _rejected(HiddenMemberError) for key in probes},
                span_texts={key: "Color::Red" for key in probes},
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_hidden_on_one_present_on_other(self, tmp_path: Path, part: Part) -> None:
        probes = {
            "value": "Color::Red",
            "pattern": (
                "let v: en3::Color = en3::Color::Green\ncase v of\n  | Color::Red => 1\n  | _ => 2"
            ),
            "is": "let v: en3::Color = en3::Color::Green\nv is Color::Red",
        }
        assert_verdicts(
            tmp_path,
            _TWO_ENUM_LIB,
            _HIDDEN_AND_PRESENT_HEADER,
            probe_table(
                probes,
                {key: _ACCEPTED for key in probes},
                identities={"value": "record en3::Color::Red", "pattern": "int", "is": "bool"},
            ),
            part=part,
        )
