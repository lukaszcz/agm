"""Member spellings several contributions supply.

A qualified spelling selects by its full path first: a member exactly one
contribution declares is selected even when the owner's name is
ambiguous, one several declare is ambiguous, and one none declares is an
unknown member. An own declaration beats every contribution, an alias
owner selects its target's members, and an enum referencing a record
never makes it a member.

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
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_scenario,
    option_identity,
    rejected,
    scenario_params,
    type_positions_rejected,
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
            "let v: one/types::Color = one/types::Color::Green",
        ),
        probes={
            "route-NoSuch-value": rejected(
                "types::Color::NoSuch", UnknownMemberError, "types::Color::NoSuch"
            ),
            "route-NoSuch-pattern": rejected(
                "case v of\n  | types::Color::NoSuch => 1\n  | _ => 2",
                UnknownMemberError,
                "types::Color::NoSuch",
            ),
            "route-NoSuch-is": rejected(
                "v is types::Color::NoSuch", UnknownMemberError, "types::Color::NoSuch"
            ),
            "route-NoSuch-annot": rejected(
                "fn(x: types::Color::NoSuch) => 1", UnknownMemberError, "types::Color::NoSuch"
            ),
            "route-NoSuch-alias": rejected(
                "type CC = types::Color::NoSuch\n1", UnknownMemberError, "types::Color::NoSuch"
            ),
            "route-NoSuch-tyarg": rejected(
                "fn(x: array[types::Color::NoSuch]) => 1",
                UnknownMemberError,
                "types::Color::NoSuch",
            ),
            "route-NoSuch-reptype": rejected(
                "types::Color::NoSuch", UnknownMemberError, "types::Color::NoSuch"
            ),
            "route-NoSuch-call": rejected(
                "types::Color::NoSuch()", UnknownMemberError, "types::Color::NoSuch"
            ),
            "route-Red-value": rejected(
                "types::Color::Red", AmbiguousQualificationError, "types::Color::Red"
            ),
            "route-Red-pattern": rejected(
                "case v of\n  | types::Color::Red => 1\n  | _ => 2",
                AmbiguousQualificationError,
                "types::Color::Red",
            ),
            "route-Red-is": rejected(
                "v is types::Color::Red", AmbiguousQualificationError, "types::Color::Red"
            ),
            "route-Red-annot": rejected(
                "fn(x: types::Color::Red) => 1", AmbiguousQualificationError, "types::Color::Red"
            ),
            "route-Red-alias": rejected(
                "type CC = types::Color::Red\n1", AmbiguousQualificationError, "types::Color::Red"
            ),
            "route-Red-tyarg": rejected(
                "fn(x: array[types::Color::Red]) => 1",
                AmbiguousQualificationError,
                "types::Color::Red",
            ),
            "route-Red-reptype": rejected(
                "types::Color::Red", AmbiguousQualificationError, "types::Color::Red"
            ),
            "route-Red-call": rejected(
                "types::Color::Red()", AmbiguousQualificationError, "types::Color::Red"
            ),
            "route-Green-value": accepted("types::Color::Green", "record one/types::Color::Green"),
            "route-Green-pattern": accepted(
                "case v of\n  | types::Color::Green => 1\n  | _ => 2", "int"
            ),
            "route-Green-is": accepted("v is types::Color::Green", "bool"),
            "route-Green-narrow": accepted(
                "v as? types::Color::Green",
                option_identity("one/types::Color::Green"),
            ),
            "route-Green-annot": accepted(
                "fn(x: types::Color::Green) => 1", "one/types::Color::Green -> int"
            ),
            "route-Green-alias": accepted(
                "type CC = types::Color::Green\nfn(p: CC) => p",
                "one/types::Color::Green -> one/types::Color::Green",
            ),
            "route-Green-tyarg": accepted(
                "fn(x: array[types::Color::Green]) => 1", "array[one/types::Color::Green] -> int"
            ),
            "route-Green-reptype": accepted(
                "types::Color::Green", "record one/types::Color::Green"
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
            "let v: m::Color = m::Color::Green",
        ),
        probes={
            "bare-NoSuch-value": rejected("Color::NoSuch", UnknownMemberError, "Color::NoSuch"),
            "bare-NoSuch-pattern": rejected(
                "case v of\n  | Color::NoSuch => 1\n  | _ => 2", UnknownMemberError, "Color::NoSuch"
            ),
            "bare-NoSuch-is": rejected("v is Color::NoSuch", UnknownMemberError, "Color::NoSuch"),
            **type_positions_rejected("bare-NoSuch", "Color::NoSuch", UnknownMemberError),
            "bare-NoSuch-reptype": rejected("Color::NoSuch", UnknownMemberError, "Color::NoSuch"),
            "bare-NoSuch-call": rejected("Color::NoSuch()", UnknownMemberError, "Color::NoSuch"),
            "bare-Red-value": rejected("Color::Red", AmbiguousQualificationError, "Color::Red"),
            "bare-Red-pattern": rejected(
                "case v of\n  | Color::Red => 1\n  | _ => 2",
                AmbiguousQualificationError,
                "Color::Red",
            ),
            "bare-Red-is": rejected("v is Color::Red", AmbiguousQualificationError, "Color::Red"),
            **type_positions_rejected("bare-Red", "Color::Red", AmbiguousQualificationError),
            "bare-Red-reptype": rejected("Color::Red", AmbiguousQualificationError, "Color::Red"),
            "bare-Red-call": rejected("Color::Red()", AmbiguousQualificationError, "Color::Red"),
            "bare-Green-value": accepted("Color::Green", "record m::Color::Green"),
            "bare-Green-pattern": accepted("case v of\n  | Color::Green => 1\n  | _ => 2", "int"),
            "bare-Green-is": accepted("v is Color::Green", "bool"),
            "bare-Green-narrow": accepted(
                "v as? Color::Green",
                option_identity("m::Color::Green"),
            ),
            "bare-Green-annot": accepted("fn(x: Color::Green) => 1", "m::Color::Green -> int"),
            "bare-Green-alias": accepted(
                "type CC = Color::Green\nfn(p: CC) => p", "m::Color::Green -> m::Color::Green"
            ),
            "bare-Green-tyarg": accepted(
                "fn(x: array[Color::Green]) => 1", "array[m::Color::Green] -> int"
            ),
            "bare-Green-reptype": accepted("Color::Green", "record m::Color::Green"),
            "bare-Green-call": accepted("Color::Green()", "record m::Color::Green"),
            "value": rejected("Color::NoSuch", UnknownMemberError, "Color::NoSuch"),
            "pattern": rejected(
                "case v of\n  | Color::NoSuch => 1\n  | _ => 2", UnknownMemberError, "Color::NoSuch"
            ),
            "is": rejected("v is Color::NoSuch", UnknownMemberError, "Color::NoSuch"),
            "annot": rejected("fn(x: Color::NoSuch) => 1", UnknownMemberError, "Color::NoSuch"),
            "alias": rejected("type CC = Color::NoSuch\n1", UnknownMemberError, "Color::NoSuch"),
            "tyarg": rejected(
                "fn(x: array[Color::NoSuch]) => 1", UnknownMemberError, "Color::NoSuch"
            ),
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
            "let v: a::E = a::E::Y",
        ),
        probes={
            "localuse-NoSuch-value": rejected("E::NoSuch", UnknownMemberError, "E::NoSuch"),
            "localuse-NoSuch-pattern": rejected(
                "case v of\n  | E::NoSuch => 1\n  | _ => 2", UnknownMemberError, "E::NoSuch"
            ),
            "localuse-NoSuch-is": rejected("v is E::NoSuch", UnknownMemberError, "E::NoSuch"),
            "localuse-NoSuch-annot": rejected(
                "fn(x: E::NoSuch) => 1", UnknownMemberError, "E::NoSuch"
            ),
            "localuse-NoSuch-alias": rejected(
                "type CC = E::NoSuch\n1", UnknownMemberError, "E::NoSuch"
            ),
            "localuse-NoSuch-tyarg": rejected(
                "fn(x: array[E::NoSuch]) => 1", UnknownMemberError, "E::NoSuch"
            ),
            "localuse-NoSuch-reptype": rejected("E::NoSuch", UnknownMemberError, "E::NoSuch"),
            "localuse-NoSuch-call": rejected("E::NoSuch()", UnknownMemberError, "E::NoSuch"),
            "localuse-Red-value": rejected("E::Red", UnknownMemberError, "E::Red"),
            "localuse-Red-pattern": rejected(
                "case v of\n  | E::Red => 1\n  | _ => 2", UnknownMemberError, "E::Red"
            ),
            "localuse-Red-is": rejected("v is E::Red", UnknownMemberError, "E::Red"),
            "localuse-Red-annot": rejected("fn(x: E::Red) => 1", UnknownMemberError, "E::Red"),
            "localuse-Red-alias": rejected("type CC = E::Red\n1", UnknownMemberError, "E::Red"),
            "localuse-Red-tyarg": rejected(
                "fn(x: array[E::Red]) => 1", UnknownMemberError, "E::Red"
            ),
            "localuse-Red-reptype": rejected("E::Red", UnknownMemberError, "E::Red"),
            "localuse-Red-call": rejected("E::Red()", UnknownMemberError, "E::Red"),
            "localuse-Green-value": rejected("E::Green", UnknownMemberError, "E::Green"),
            "localuse-Green-pattern": rejected(
                "case v of\n  | E::Green => 1\n  | _ => 2", UnknownMemberError, "E::Green"
            ),
            "localuse-Green-is": rejected("v is E::Green", UnknownMemberError, "E::Green"),
            "localuse-Green-annot": rejected(
                "fn(x: E::Green) => 1", UnknownMemberError, "E::Green"
            ),
            "localuse-Green-alias": rejected(
                "type CC = E::Green\n1", UnknownMemberError, "E::Green"
            ),
            "localuse-Green-tyarg": rejected(
                "fn(x: array[E::Green]) => 1", UnknownMemberError, "E::Green"
            ),
            "localuse-Green-reptype": rejected("E::Green", UnknownMemberError, "E::Green"),
            "localuse-Green-call": rejected("E::Green()", UnknownMemberError, "E::Green"),
        },
        legal=frozenset({(4, 1, 1), (4, 2), (5, 1), (6,)}),
    ),
    "two-module-uses": Scenario(
        modules={"m": _M_2, "n": _N_2},
        header=(
            "import m",
            "import n",
            "use m::*",
            "use n::*",
            "let v: m::Color = m::Color::Green",
        ),
        probes={
            "moduse-NoSuch-value": rejected("Color::NoSuch", UnknownMemberError, "Color::NoSuch"),
            "moduse-NoSuch-pattern": rejected(
                "case v of\n  | Color::NoSuch => 1\n  | _ => 2", UnknownMemberError, "Color::NoSuch"
            ),
            "moduse-NoSuch-is": rejected("v is Color::NoSuch", UnknownMemberError, "Color::NoSuch"),
            "moduse-NoSuch-annot": rejected(
                "fn(x: Color::NoSuch) => 1", UnknownMemberError, "Color::NoSuch"
            ),
            "moduse-NoSuch-alias": rejected(
                "type CC = Color::NoSuch\n1", UnknownMemberError, "Color::NoSuch"
            ),
            "moduse-NoSuch-tyarg": rejected(
                "fn(x: array[Color::NoSuch]) => 1", UnknownMemberError, "Color::NoSuch"
            ),
            "moduse-NoSuch-reptype": rejected("Color::NoSuch", UnknownMemberError, "Color::NoSuch"),
            "moduse-NoSuch-call": rejected("Color::NoSuch()", UnknownMemberError, "Color::NoSuch"),
            "moduse-Red-value": rejected("Color::Red", AmbiguousQualificationError, "Color::Red"),
            "moduse-Red-pattern": rejected(
                "case v of\n  | Color::Red => 1\n  | _ => 2",
                AmbiguousQualificationError,
                "Color::Red",
            ),
            "moduse-Red-is": rejected("v is Color::Red", AmbiguousQualificationError, "Color::Red"),
            "moduse-Red-annot": rejected(
                "fn(x: Color::Red) => 1", AmbiguousQualificationError, "Color::Red"
            ),
            "moduse-Red-alias": rejected(
                "type CC = Color::Red\n1", AmbiguousQualificationError, "Color::Red"
            ),
            "moduse-Red-tyarg": rejected(
                "fn(x: array[Color::Red]) => 1", AmbiguousQualificationError, "Color::Red"
            ),
            "moduse-Red-reptype": rejected("Color::Red", AmbiguousQualificationError, "Color::Red"),
            "moduse-Red-call": rejected("Color::Red()", AmbiguousQualificationError, "Color::Red"),
            "moduse-Green-value": accepted("Color::Green", "record m::Color::Green"),
            "moduse-Green-pattern": accepted("case v of\n  | Color::Green => 1\n  | _ => 2", "int"),
            "moduse-Green-is": accepted("v is Color::Green", "bool"),
            "moduse-Green-narrow": accepted(
                "v as? Color::Green",
                option_identity("m::Color::Green"),
            ),
            "moduse-Green-annot": accepted("fn(x: Color::Green) => 1", "m::Color::Green -> int"),
            "moduse-Green-alias": accepted(
                "type CC = Color::Green\nfn(p: CC) => p", "m::Color::Green -> m::Color::Green"
            ),
            "moduse-Green-tyarg": accepted(
                "fn(x: array[Color::Green]) => 1", "array[m::Color::Green] -> int"
            ),
            "moduse-Green-reptype": accepted("Color::Green", "record m::Color::Green"),
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
                AglTypeError,
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
                AglTypeError,
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
                AglTypeError,
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
