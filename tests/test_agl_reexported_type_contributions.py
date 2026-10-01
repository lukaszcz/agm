"""A type an import surface exposes stays a type through every re-export.

Whether a contributed name is a scope region or a type is decided by its
declaration: a type re-exported by another module, or re-rooted under a scope
region by a scoped ``export``, is still that type through ``use`` and import
tails alike, in every position, and ``hiding`` one of its members on such a
surface hides it.

Every probe is checked in the file part and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`): both modes reach
the same verdict, error span and message, or accepted identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import HiddenMemberError
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    assert_scenario,
    option_identity,
    rejected,
    scenario_params,
)

_TL = (
    "enum Color\n  | Red\n  | Blue\n"
    "record Point\n  x: int\n"
    "record Geo\n  x: int\n"
    "record Geo::Inner\n  y: int\n"
)
_REEXPORTER = "export tl::{Color, Point, Geo}\n"
_SCOPED_REEXPORTER = "scope Shapes\n  export tl::{Color, Point, Geo}\nend Shapes\n"
_REEXPORTS = {"tl": _TL, "mid": _SCOPED_REEXPORTER, "mid2": _REEXPORTER}

_OPTION_OF_RED = option_identity("tl::Color::Red")
_COLOR = "enum tl::Color\n  | Red\n  | Blue"


def _reexported_type_probes() -> dict[str, Probe]:
    """Every position spelling the types ``tl`` declares, reached through a re-export."""
    return {
        "annotation": accepted("fn(p: Point) => p", "tl::Point -> tl::Point"),
        "alias-target": accepted("type Alias = Point\nfn(p: Alias) => p", "tl::Point -> tl::Point"),
        "type-argument": accepted(
            "fn(p: array[Point]) => p", "array[tl::Point] -> array[tl::Point]"
        ),
        "constructor": accepted("Point(x = 1)", "record tl::Point\n  x: int"),
        "enum-annotation": accepted("fn(c: Color) => c", "tl::Color -> tl::Color"),
        "member-value": accepted("Color::Red", "record tl::Color::Red"),
        "member-annotation": accepted("fn(p: Color::Red) => p", "tl::Color::Red -> tl::Color::Red"),
        "member-cast": accepted("let v: Color = Color::Blue\nv as? Color::Red", _OPTION_OF_RED),
        "member-is": accepted("let v: Color = Color::Blue\nv is Color::Red", "bool"),
        "member-pattern": accepted(
            "let v: Color = Color::Blue\ncase v of\n  | Color::Red as r => [r]\n  | _ => []",
            "array[tl::Color]",
        ),
        "receiver": accepted("def Color::m(self) = self\nColor::Blue.m()", _COLOR),
        "nested-annotation": accepted("fn(p: Geo::Inner) => p", "tl::Geo::Inner -> tl::Geo::Inner"),
        "nested-receiver": accepted(
            "def Geo::Inner::m(self) = self\nGeo::Inner(y = 1).m()",
            "record tl::Geo::Inner\n  y: int",
        ),
    }


def _in_region(use: str) -> dict[str, Probe]:
    """Re-exported types spelled inside ``scope r``, which opens them by *use*."""

    def region(*lines: str) -> str:
        return "\n".join(("scope r", f"  {use}", *(f"  {line}" for line in lines), "end r"))

    return {
        "region-annotation": accepted(
            region("def f(p: Point) = p") + "\n\nr::f", "tl::Point -> tl::Point"
        ),
        "region-constructor": accepted(
            region("let q = Point(x = 1)") + "\n\nr::q", "record tl::Point\n  x: int"
        ),
        "region-member-value": accepted(
            region("let q = Color::Red") + "\n\nr::q", "record tl::Color::Red"
        ),
        "region-receiver": accepted(
            region("def Color::m(self) = self", "def q() = Color::Blue.m()") + "\n\nr::q()", _COLOR
        ),
    }


def _hidden_red_probes() -> dict[str, Probe]:
    """``Color::Red``, hidden by the header's ``hiding``, in every position; ``Blue`` stays."""
    return {
        "hidden-value": rejected("Color::Red", HiddenMemberError, "Color::Red"),
        "hidden-annotation": rejected("fn(p: Color::Red) => p", HiddenMemberError, "Color::Red"),
        "hidden-pattern": rejected(
            "let v: Color = Color::Blue\ncase v of\n  | Color::Red => 1\n  | _ => 2",
            HiddenMemberError,
            "Color::Red",
        ),
        "hidden-cast": rejected(
            "let v: Color = Color::Blue\nv as? Color::Red", HiddenMemberError, "Color::Red"
        ),
        "visible-annotation": accepted(
            "fn(p: Color::Blue) => p", "tl::Color::Blue -> tl::Color::Blue"
        ),
    }


_SCENARIOS = {
    "use-of-a-reexporting-module": Scenario(
        modules=_REEXPORTS,
        header=("import mid2", "use mid2::*"),
        probes=_reexported_type_probes(),
    ),
    "selective-use-of-a-reexporting-module": Scenario(
        modules=_REEXPORTS,
        header=("import mid2", "use mid2::{Color, Point, Geo}"),
        probes=_reexported_type_probes(),
    ),
    "use-of-a-scoped-reexport": Scenario(
        modules=_REEXPORTS,
        header=("import mid", "use mid::Shapes::*"),
        probes=_reexported_type_probes(),
    ),
    "selective-use-of-a-scoped-reexport": Scenario(
        modules=_REEXPORTS,
        header=("import mid", "use mid::Shapes::{Color, Point, Geo}"),
        probes=_reexported_type_probes(),
    ),
    "import-tail-of-a-reexporting-module": Scenario(
        modules=_REEXPORTS,
        header=("import mid2::*",),
        probes=_reexported_type_probes(),
    ),
    "region-use-of-a-reexporting-module": Scenario(
        modules=_REEXPORTS,
        header=("import mid2",),
        probes=_in_region("use mid2::*"),
    ),
    "region-use-of-a-scoped-reexport": Scenario(
        modules=_REEXPORTS,
        header=("import mid",),
        probes=_in_region("use mid::Shapes::*"),
    ),
    "hiding-on-a-use-of-a-reexporting-module": Scenario(
        modules=_REEXPORTS,
        header=("import mid2", "use mid2::* hiding Color::Red"),
        probes=_hidden_red_probes(),
    ),
    "hiding-on-a-use-of-a-scoped-reexport": Scenario(
        modules=_REEXPORTS,
        header=("import mid", "use mid::Shapes::* hiding Color::Red"),
        probes=_hidden_red_probes(),
    ),
    "hiding-on-an-import-tail-of-a-reexporting-module": Scenario(
        modules=_REEXPORTS,
        header=("import mid2::* hiding Color::Red",),
        probes=_hidden_red_probes(),
    ),
}


class TestReexportedTypes:
    """Re-exported types, used or imported, in every position."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)
