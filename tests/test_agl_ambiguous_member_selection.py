"""Member spellings several contributions supply.

A qualified spelling selects by its full path first: a member exactly one
contribution declares is selected even when the owner's name is
ambiguous, one several declare is ambiguous, and one none declares is an
unknown member. An own declaration beats every contribution, an alias
owner selects its target's members, and an enum referencing a record
never makes it a member. An owner two routes, two wildcard imports or two
``use`` declarations supply is probed here only in the positions
:mod:`tests.test_agl_qualifier_position_matrix` leaves out.

Every probe is checked in the file part and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`): both modes reach
the same verdict, error span and message, or accepted identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError, ReferencedMemberError
from agm.agl.scope.symbols import (
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    UnknownMemberError,
)
from agm.agl.typecheck.checker import EnumOwnerMismatchError
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_scenario,
    option_identity,
    rejected,
    scenario_params,
)

_TL = "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"
_TL2 = "record Geo\n  z: int\nrecord Geo::Inner\n  w: int\n"
_M = "record Point\n  x: int\nenum Shape = Circle | Sq\n"
_N = "record Point\n  z: int\nenum Shape = Circle | Tri\n"
_ONE_TYPES = "enum Color\n  | Red\n  | Green\n"
_TWO_TYPES = "enum Color\n  | Red\n  | Blue\n"
_M_2 = "enum Color\n  | Red\n  | Green\n"
_N_2 = "enum Color\n  | Red\n  | Blue\n"
_M_3 = "enum Base\n  | Red\n  | Green\ntype Color = Base\n"
_ONE_T = "enum Base\n  | Red\n  | Green\ntype Color = Base\n"
_TWO_T = "enum Color\n  | Red\n  | Blue\n"
_M_4 = "enum Base[T]\n  | Red\n  | Green(v: T)\ntype Color = Base[int]\n"
_M_5 = "record Q\n  x: int\nenum Color\n  | Q\n  | Red\n"
_LIB = "record Point\n  x: int\nenum Shape\n  | Circle\ntype Num = int\n"

_SCENARIOS = {
    "two-imported-records": Scenario(
        modules={"tl": _TL, "tl2": _TL2},
        header=(
            "import tl::*",
            "import tl2::*",
        ),
        probes={
            "nodef-annot": rejected(
                "fn(p: Geo::Inner::Nope) => 1", UnknownMemberError, "Geo::Inner::Nope"
            ),
            "nodef-pat": rejected(
                "case 1 of\n  | Geo::Inner::Nope => 1\n  | _ => 2",
                UnknownMemberError,
                "Geo::Inner::Nope",
            ),
            "nodef-val": rejected("Geo::Inner::Nope", UnknownMemberError, "Geo::Inner::Nope"),
            "amb-val": rejected("Geo::Inner(w = 1)", AmbiguousQualificationError, "Geo::Inner"),
            "amb-annot": rejected(
                "fn(p: Geo::Inner) => 1", AmbiguousQualificationError, "Geo::Inner"
            ),
            "amb-alias": rejected(
                "type AA = Geo::Inner\n1", AmbiguousQualificationError, "Geo::Inner"
            ),
        },
    ),
    "referenced-member-of-local-enum": Scenario(
        header=(
            "record Saved\n  id: int",
            "enum Stored = ::Saved | Fresh",
        ),
        probes={
            "value": rejected("Stored::Saved(id = 1)", ReferencedMemberError, "Stored::Saved"),
            "annot": rejected("fn(p: Stored::Saved) => 1", ReferencedMemberError, "Stored::Saved"),
            "tyarg": rejected(
                "fn(p: array[Stored::Saved]) => 1", ReferencedMemberError, "Stored::Saved"
            ),
            "alias": rejected("type AA = Stored::Saved\n1", ReferencedMemberError, "Stored::Saved"),
            "pattern": rejected(
                ("let v: Stored = Fresh\ncase v of\n  | Stored::Saved(id) => id\n  | _ => 2"),
                ReferencedMemberError,
                "Stored::Saved",
            ),
            "is": rejected(
                "let v: Stored = Fresh\nv is Stored::Saved", ReferencedMemberError, "Stored::Saved"
            ),
            "fresh-annot": accepted("fn(p: Stored::Fresh) => 1", "Stored::Fresh -> int"),
            "fresh-is": accepted("let v: Stored = Fresh\nv is Stored::Fresh", "bool"),
            "fresh-narrow": accepted(
                "let v: Stored = Fresh\nv as? Stored::Fresh",
                option_identity("Stored::Fresh"),
            ),
            "length-three-annot": rejected(
                "fn(p: Stored::Saved::X) => 1", ReferencedMemberError, "Stored::Saved::X"
            ),
            "length-three-fresh-annot": rejected(
                "fn(p: Stored::Fresh::X) => 1", UnknownMemberError, "Stored::Fresh::X"
            ),
            "length-three-fresh-val": rejected(
                "Stored::Fresh::X", UnknownMemberError, "Stored::Fresh::X"
            ),
        },
    ),
    "local-record-beside-two-imports": Scenario(
        modules={"m": _M, "n": _N},
        header=(
            "import m::*",
            "import n::*",
            "record Point\n  y: int",
        ),
        probes={
            "localbare-annot": accepted("fn(p: Point) => 1", "Point -> int"),
            "localbare-value": accepted("Point(y = 1)", "record Point\n  y: int"),
            "localbare-tyarg": accepted("fn(p: array[Point]) => 1", "array[Point] -> int"),
            "localbare-reptype": accepted("Point", "int -> Point"),
        },
    ),
    "local-enum-beside-two-imports": Scenario(
        modules={"m": _M, "n": _N},
        header=(
            "import m::*",
            "import n::*",
            "enum Shape = Tri | Q",
        ),
        probes={
            "localenum-annot": accepted("fn(p: Shape) => 1", "Shape -> int"),
            "localenum-member": rejected(
                "fn(p: Shape::Circle) => 1", AmbiguousQualificationError, "Shape::Circle"
            ),
            "localenum-member-val": rejected(
                "Shape::Circle", AmbiguousQualificationError, "Shape::Circle"
            ),
        },
    ),
    "two-imported-records-and-enums": Scenario(
        modules={"m": _M, "n": _N},
        header=(
            "import m::*",
            "import n::*",
        ),
        probes={
            "bare-annot": rejected("fn(p: Point) => 1", AmbiguousQualificationError, "Point"),
            "bare-reptype": rejected("Point", AmbiguousConstructorError, "Point"),
            "bare-value": rejected("Point(x = 1)", AmbiguousConstructorError, "Point"),
            "shape-circle-val": rejected(
                "Shape::Circle", AmbiguousQualificationError, "Shape::Circle"
            ),
            "shape-circle-annot": rejected(
                "fn(p: Shape::Circle) => 1", AmbiguousQualificationError, "Shape::Circle"
            ),
            "shape-tri-annot": accepted("fn(p: Shape::Tri) => 1", "n::Shape::Tri -> int"),
            "shape-tri-val": accepted("Shape::Tri", "record n::Shape::Tri"),
        },
    ),
    "two-routes-sharing-a-suffix": Scenario(
        modules={"one/types": _ONE_TYPES, "two/types": _TWO_TYPES},
        header=(
            "import one/types",
            "import two/types",
        ),
        probes={
            "route-NoSuch-call": rejected(
                "types::Color::NoSuch()", UnknownMemberError, "types::Color::NoSuch"
            ),
            "route-Red-call": rejected(
                "types::Color::Red()", AmbiguousQualificationError, "types::Color::Red"
            ),
            "route-Green-call": accepted("types::Color::Green()", "record one/types::Color::Green"),
        },
    ),
    "two-wildcard-imports": Scenario(
        modules={"m": _M_2, "n": _N_2},
        header=(
            "import m",
            "import m::*",
            "import n::*",
        ),
        probes={
            "bare-NoSuch-call": rejected("Color::NoSuch()", UnknownMemberError, "Color::NoSuch"),
            "bare-Red-call": rejected("Color::Red()", AmbiguousQualificationError, "Color::Red"),
            "bare-Green-call": accepted("Color::Green()", "record m::Color::Green"),
            "owner-annot": rejected("fn(x: Color) => 1", AmbiguousQualificationError, "Color"),
            "owner-applied-miss": rejected(
                "fn(x: Color[int]::NoSuch) => 1", UnknownMemberError, "Color[int]::NoSuch"
            ),
        },
    ),
    "two-local-uses": Scenario(
        header=(
            "use a::*",
            "use b::*",
            "scope a\n  enum E = X | Y\nend a",
            "scope b\n  enum E = X | Z\nend b",
        ),
        probes={
            "localuse-NoSuch-call": rejected("E::NoSuch()", UnknownMemberError, "E::NoSuch"),
        },
        legal=frozenset({(4, 1), (5,)}),
    ),
    "two-module-uses": Scenario(
        modules={"m": _M_2, "n": _N_2},
        header=(
            "import m",
            "import n",
            "use m::*",
            "use n::*",
        ),
        probes={
            "moduse-NoSuch-call": rejected("Color::NoSuch()", UnknownMemberError, "Color::NoSuch"),
            "moduse-Red-call": rejected("Color::Red()", AmbiguousQualificationError, "Color::Red"),
            "moduse-Green-call": accepted("Color::Green()", "record m::Color::Green"),
        },
    ),
    "alias-and-enum-through-uses": Scenario(
        modules={"m": _M_3, "n": _N_2},
        header=(
            "import m",
            "import n",
            "use m::*",
            "use n::*",
        ),
        probes={
            "alias-use-Green-value": accepted("Color::Green", "record m::Base::Green"),
            "alias-use-Green-annot": accepted("fn(x: Color::Green) => 1", "m::Base::Green -> int"),
            "alias-use-Green-is": rejected(
                "let v: n::Color = n::Color::Red\nv is Color::Green",
                EnumOwnerMismatchError,
                "v is Color::Green",
                phase="typecheck",
            ),
            "alias-use-Green-pattern": accepted(
                ("let v: m::Base = m::Base::Green\ncase v of\n  | Color::Green => 1\n  | _ => 2"),
                "int",
            ),
        },
    ),
    "alias-and-enum-through-wildcard-imports": Scenario(
        modules={"m": _M_3, "n": _N_2},
        header=(
            "import m",
            "import n",
            "import m::*",
            "import n::*",
        ),
        probes={
            "alias-imp-Green-value": accepted("Color::Green", "record m::Base::Green"),
            "alias-imp-Green-annot": accepted("fn(x: Color::Green) => 1", "m::Base::Green -> int"),
            "alias-imp-Green-is": rejected(
                "let v: n::Color = n::Color::Red\nv is Color::Green",
                EnumOwnerMismatchError,
                "v is Color::Green",
                phase="typecheck",
            ),
            "alias-imp-Green-pattern": accepted(
                ("let v: m::Base = m::Base::Green\ncase v of\n  | Color::Green => 1\n  | _ => 2"),
                "int",
            ),
        },
    ),
    "alias-and-enum-through-routes": Scenario(
        modules={"one/t": _ONE_T, "two/t": _TWO_T},
        header=(
            "import one/t",
            "import two/t",
        ),
        probes={
            "alias-route-Green-value": accepted("t::Color::Green", "record one/t::Base::Green"),
            "alias-route-Green-annot": accepted(
                "fn(x: t::Color::Green) => 1", "one/t::Base::Green -> int"
            ),
            "alias-route-Green-is": rejected(
                "let v: two/t::Color = two/t::Color::Red\nv is t::Color::Green",
                EnumOwnerMismatchError,
                "v is t::Color::Green",
                phase="typecheck",
            ),
            "alias-route-Green-pattern": accepted(
                (
                    "let v: one/t::Base = one/t::Base::Green\n"
                    "case v of\n"
                    "  | t::Color::Green => 1\n"
                    "  | _ => 2"
                ),
                "int",
            ),
        },
    ),
    "generic-alias-and-enum-through-uses": Scenario(
        modules={"m": _M_4, "n": _N_2},
        header=(
            "import m",
            "import n",
            "use m::*",
            "use n::*",
        ),
        probes={
            "galias-use-Green": accepted(
                "Color::Green(v = 1)", "record m::Base::Green[int]\n  v: int"
            ),
        },
    ),
    "local-alias-and-enum-through-uses": Scenario(
        header=(
            "use a::*",
            "use b::*",
            (
                "scope a\n"
                "  enum Base = X | Y\n"
                "  type E = Base\n"
                "end a\n"
                "\n"
                "scope b\n"
                "  enum E = X | Z\n"
                "end b"
            ),
        ),
        probes={
            "region-alias-Y": accepted("E::Y", "record a::Base::Y"),
            "region-alias-Y-annot": accepted("fn(x: E::Y) => 1", "a::Base::Y -> int"),
        },
        legal=frozenset({(3, 1), (4,)}),
    ),
    "inline-member-named-like-a-record": Scenario(
        modules={"m": _M_5, "n": _N_2},
        header=(
            "import m",
            "import n",
            "use m::*",
            "use n::*",
        ),
        probes={
            "referenced-use-Q": rejected(
                "Color::Q(x = 1)", AglTypeError, "x = 1", phase="typecheck"
            ),
        },
    ),
    "local-declarations-beside-wildcard-import": Scenario(
        modules={"lib": _LIB},
        header=(
            "import lib::*",
            "record Point\n  y: int\nenum Shape\n  | Tri\ntype Num = text",
        ),
        probes={
            "val-point": accepted("Point(y = 1)", "record Point\n  y: int"),
            "annot-point": accepted("fn(p: Point) => 1", "Point -> int"),
            "annot-shape": accepted("fn(p: Shape) => 1", "Shape -> int"),
            "annot-num": accepted('let n: Num = "a"\nn', "text"),
            "val-tri": accepted("Tri", "record Shape::Tri"),
            "val-shape-circle": accepted("Shape::Circle", "record lib::Shape::Circle"),
            "annot-shape-circle": accepted(
                "fn(p: Shape::Circle) => 1", "lib::Shape::Circle -> int"
            ),
        },
    ),
}


class TestAmbiguousMemberSelection:
    """Full-path selection among imports, uses, routes and local declarations."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)
