"""A ``def`` path's namespace beside the types and scopes it spells.

A method (``def Owner::m(self)``) joins its owner's namespace, so the
owner's members stay reachable through the same path; a plain function's
path (``def Geo::f()``) declares an own local namespace, which beats a
same-named imported type or scope, so a member the namespace lacks is an
unknown member. Receivers an alias or scalar cannot own are rejected.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`): both modes reach
the same verdict, error span and message, or accepted identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousQualificationError,
    RouteClashError,
    UnknownMemberError,
)
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_scenario_for_grouping,
    rejected,
    scenario_params,
)

_TL = "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"
_EN = "enum Shape\n  | Circle\n  | Square\n"
_AL = "record Base\n  x: int\nrecord Base::Inner\n  y: int\ntype Geo = Base\n"
_TL2 = "record Geo\n  z: int\nrecord Geo::Inner\n  w: int\n"
_SHAPES = "scope Geo\n  record Point\n    x: int\nend Geo\n"
_SC = "type Geo = int\n"
_AL_2 = "type Geo = int\n"
_TL_2 = "record R\n  x: int\nrecord R::Geo\n  y: int\nrecord R::Geo::X\n  z: int\n"
_PAL = "enum Color\n  | Red\n"
_SHAPES_2 = "scope Geo\n\n  scope In\n    record Point\n      x: int\n  end In\nend Geo\n"

_SCENARIOS = {
    "method-on-imported-record": Scenario(
        modules={"tl": _TL},
        header=(
            "import tl::*",
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "anch-m": accepted("::Geo::m(Geo(x = 1))", "int"),
            "anch-nope": rejected("fn(p: ::Geo::Nope) => 1", UnknownMemberError, "::Geo::Nope"),
            "anch-val": rejected("::Geo::Inner(y = 1)", UnknownMemberError, "::Geo::Inner"),
            "anch-annot": rejected("fn(p: ::Geo::Inner) => 1", UnknownMemberError, "::Geo::Inner"),
            "mref-val": accepted("Geo::m(Geo(x = 1))", "int"),
            "method-owner-val": accepted("Geo::Inner(y = 1)", "record tl::Geo::Inner\n  y: int"),
            "method-owner-annot": accepted("fn(p: Geo::Inner) => 1", "tl::Geo::Inner -> int"),
            "method-owner-miss-val": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
            "method-owner-miss-annot": rejected(
                "fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"
            ),
        },
    ),
    "method-on-imported-enum": Scenario(
        modules={"en": _EN},
        header=(
            "import en::*",
            "def Shape::area(self) -> int = 1",
        ),
        probes={
            "orph-value": accepted("Shape::Circle", "record en::Shape::Circle"),
            "orph-pattern": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "orph-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "orph-annot": accepted("fn(p: Shape::Circle) => 1", "en::Shape::Circle -> int"),
            "orph-alias": accepted("type AA = Shape::Circle\n1", "int"),
            "orph-tyarg": accepted(
                "fn(p: array[Shape::Circle]) => 1", "array[en::Shape::Circle] -> int"
            ),
            "orph-reptype": accepted("Shape::Circle", "record en::Shape::Circle"),
            "orph-call": accepted("Shape::Circle.area()", "int"),
            "orph-mref": accepted("Shape::area(Shape::Circle)", "int"),
            "orph-miss-value": rejected("Shape::Nope", UnknownMemberError, "Shape::Nope"),
            "orph-miss-annot": rejected(
                "fn(p: Shape::Nope) => 1", UnknownMemberError, "Shape::Nope"
            ),
            "orph-miss-pattern": rejected(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Nope => 1\n  | _ => 2"),
                UnknownMemberError,
                "Shape::Nope",
            ),
            "orph-miss-is": rejected(
                "let v: Shape = Shape::Square\nv is Shape::Nope", UnknownMemberError, "Shape::Nope"
            ),
            "wild-pat": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "wild-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "wild-pat-miss": rejected(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Nope => 1\n  | _ => 2"),
                UnknownMemberError,
                "Shape::Nope",
            ),
        },
    ),
    "method-on-used-record": Scenario(
        modules={"tl": _TL},
        header=(
            "import tl",
            "use tl::*",
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "use-val": accepted("Geo::Inner(y = 1)", "record tl::Geo::Inner\n  y: int"),
            "use-annot": accepted("fn(p: Geo::Inner) => 1", "tl::Geo::Inner -> int"),
            "use-miss-val": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
            "use-miss-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "muse-pat": accepted(
                "let v = tl::Geo::Inner(y = 1)\ncase v of\n  | Geo::Inner(y) => y", "int"
            ),
            "muse-m": accepted("Geo::m(tl::Geo(x = 1))", "int"),
            "muse-call": accepted("tl::Geo(x = 1).m()", "int"),
        },
    ),
    "method-on-region-record-opened-by-use": Scenario(
        header=(
            "use R::*",
            ("scope R\n  record Geo\n    x: int\n  record Geo::Inner\n    y: int\nend R"),
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "ruse-val": accepted("Geo::Inner(y = 1)", "record R::Geo::Inner\n  y: int"),
            "ruse-annot": accepted("fn(p: Geo::Inner) => 1", "R::Geo::Inner -> int"),
            "ruse-miss-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "ruse-pat": accepted(
                "let v = R::Geo::Inner(y = 1)\ncase v of\n  | Geo::Inner(y) => y", "int"
            ),
            "ruse-m": accepted("Geo::m(R::Geo(x = 1))", "int"),
        },
        legal=frozenset({(2, 1, 1), (2, 2), (3, 1), (4,)}),
    ),
    "method-on-imported-alias": Scenario(
        modules={"al": _AL},
        header=(
            "import al::*",
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "alias-val": rejected("Geo::Inner(y = 1)", AglScopeError, "self"),
            "alias-annot": rejected("fn(p: Geo::Inner) => 1", AglScopeError, "self"),
            "alias-miss-annot": rejected("fn(p: Geo::Nope) => 1", AglScopeError, "self"),
        },
        legal=frozenset({(1, 2), (3,)}),
    ),
    "method-on-ambiguous-imported-record": Scenario(
        modules={"tl": _TL, "tl2": _TL2},
        header=(
            "import tl::*",
            "import tl2::*",
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "early-amb-annot": rejected(
                "fn(p: Geo::Inner) => 1", AmbiguousQualificationError, "self"
            ),
        },
        legal=frozenset({(1, 1, 2), (1, 3), (2, 2), (4,)}),
    ),
    "scoped-method-on-imported-record": Scenario(
        modules={"tl": _TL},
        header=("import tl::*",),
        probes={
            "nest-imp": accepted(
                (
                    "scope r\n"
                    "  def Geo::m(self) -> int = 1\n"
                    "  def g(p: Geo::Inner) -> int = 1\n"
                    "  let q = Geo::Inner(y = 1)\n"
                    "end r\n"
                    "\n"
                    "r::g"
                ),
                "tl::Geo::Inner -> int",
            ),
        },
    ),
    "scoped-method-on-local-record": Scenario(
        header=(
            "record Geo\n  x: int",
            "record Geo::Inner\n  y: int",
        ),
        probes={
            "nest-loc": accepted(
                (
                    "scope r\n"
                    "  def Geo::m(self) -> int = 1\n"
                    "  def g(p: Geo::Inner) -> int = 1\n"
                    "  let q = Geo::Inner(y = 1)\n"
                    "end r\n"
                    "\n"
                    "r::g"
                ),
                "Geo::Inner -> int",
            ),
        },
    ),
    "def-path-extending-imported-scope": Scenario(
        modules={"shapes": _SHAPES},
        header=(
            "import shapes::*",
            "def Geo::In::f() -> int = 1",
        ),
        probes={
            "eprefix-reg-val": rejected("Geo::Point(x = 1)", UnknownMemberError, "Geo::Point"),
            "eprefix-reg-annot": rejected(
                "fn(p: Geo::Point) => 1", UnknownMemberError, "Geo::Point"
            ),
            "eprefix-reg-pat": rejected(
                "let p = shapes::Geo::Point(x = 1)\ncase p of\n  | Geo::Point(x) => x",
                UnknownMemberError,
                "Geo::Point",
            ),
        },
    ),
    "method-on-imported-nested-record": Scenario(
        modules={"tl": _TL},
        header=(
            "import tl::*",
            "def Geo::Inner::m(self) -> int = 1",
        ),
        probes={
            "eprefix-ty-val": accepted("Geo::Inner(y = 1)", "record tl::Geo::Inner\n  y: int"),
            "eprefix-ty-annot": accepted("fn(p: Geo::Inner) => 1", "tl::Geo::Inner -> int"),
            "eprefix-ty-miss-annot": rejected(
                "fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"
            ),
            "eprefix-ty-miss-val": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
        },
    ),
    "method-on-used-enum": Scenario(
        modules={"en": _EN},
        header=(
            "import en",
            "use en::*",
            "def Shape::area(self) -> int = 1",
        ),
        probes={
            "euse-val": accepted("Shape::Circle", "record en::Shape::Circle"),
            "euse-annot": accepted("fn(p: Shape::Circle) => 1", "en::Shape::Circle -> int"),
            "euse-pat": accepted(
                ("let v: Shape = en::Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "euse-is": accepted("let v: Shape = en::Shape::Circle\nv is Shape::Circle", "bool"),
            "euse-call": accepted("en::Shape::Circle.area()", "int"),
            "use-pat": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "use-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "use-pat-miss": rejected(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Nope => 1\n  | _ => 2"),
                UnknownMemberError,
                "Shape::Nope",
            ),
        },
    ),
    "generic-method-on-prelude-enum": Scenario(
        header=("def Option::m[T](self) -> int = 1",),
        probes={
            "opt-some-val": accepted(
                "Option::Some(value = 1)", "record std/option::Option::Some[int]\n  value: int"
            ),
            "opt-none-val": accepted(
                "let o: Option[int] = Option::None\no",
                "enum std/option::Option[int]\n  | None\n  | Some(value: int)",
            ),
            "opt-annot": accepted(
                "fn(p: Option::Some[int]) => 1", "std/option::Option::Some[int] -> int"
            ),
            "opt-nope": rejected("fn(p: Option::Nope) => 1", UnknownMemberError, "Option::Nope"),
            "opt-call": accepted("Some(1).m()", "int"),
            "opt-some": accepted(
                "Option::Some(1)", "record std/option::Option::Some[int]\n  value: int"
            ),
            "opt-nope-2": rejected("Option::Nope", UnknownMemberError, "Option::Nope"),
            "opt-pat": rejected(
                "case Option::Some(1) of\n  | Option::Some(v) => v\n  | _ => 2",
                AglTypeError,
                "v",
                phase="typecheck",
            ),
        },
    ),
    "non-generic-method-on-generic-prelude-enum": Scenario(
        header=("def Option::m(self) -> int = 1",),
        probes={
            "opt2-annot": rejected(
                "fn(p: Option::Some[int]) => 1",
                AglTypeError,
                "def Option::m(self) -> int = 1",
                phase="typecheck",
            ),
        },
        legal=frozenset({(2,)}),
    ),
    "region-record-opened-by-use": Scenario(
        header=(
            "use R::*",
            ("scope R\n  record Geo\n    x: int\n  record Geo::Inner\n    y: int\nend R"),
        ),
        probes={
            "ruse-nodef-annot": accepted("fn(p: Geo::Inner) => 1", "R::Geo::Inner -> int"),
        },
        legal=frozenset({(2, 1), (3,)}),
    ),
    "record-opened-by-use": Scenario(
        modules={"tl": _TL},
        header=(
            "import tl",
            "use tl::*",
        ),
        probes={
            "muse-nodef-annot": accepted("fn(p: Geo::Inner) => 1", "tl::Geo::Inner -> int"),
        },
    ),
    "function-in-imported-alias-namespace": Scenario(
        modules={"al": _AL},
        header=(
            "import al::*",
            "def Geo::f() -> int = 1",
        ),
        probes={
            "al-inner-val": rejected("Geo::Inner(y = 1)", UnknownMemberError, "Geo::Inner"),
            "al-inner-annot": rejected("fn(p: Geo::Inner) => 1", UnknownMemberError, "Geo::Inner"),
            "al-nope-val": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
            "al-nope-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "al-f": accepted("Geo::f()", "int"),
        },
    ),
    "function-in-imported-scalar-alias-namespace": Scenario(
        modules={"sc": _SC},
        header=(
            "import sc::*",
            "def Geo::f() -> int = 1",
        ),
        probes={
            "sc-nope-val": rejected("Geo::Nope", UnknownMemberError, "Geo::Nope"),
            "sc-nope-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "sc-nope-pat": rejected(
                "case 1 of\n  | Geo::Nope => 1\n  | _ => 2", UnknownMemberError, "Geo::Nope"
            ),
            "sc-f": accepted("Geo::f()", "int"),
        },
    ),
    "function-in-fresh-namespace": Scenario(
        header=("def Geo::f() -> int = 1",),
        probes={
            "plain-nope-val": rejected("Geo::Nope", UnknownMemberError, "Geo::Nope"),
            "plain-nope-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "plain-nope-pat": rejected(
                "case 1 of\n  | Geo::Nope => 1\n  | _ => 2", UnknownMemberError, "Geo::Nope"
            ),
            "plain-f": accepted("Geo::f()", "int"),
            "noimp-annot": rejected("fn(p: Geo::Point) => 1", UnknownMemberError, "Geo::Point"),
        },
    ),
    "method-on-imported-scalar-alias": Scenario(
        modules={"al": _AL_2},
        header=(
            "import al::*",
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "scaldef-val": rejected("Geo::Nope", AglScopeError, "self"),
            "scaldef-annot": rejected("fn(p: Geo::Nope) => 1", AglScopeError, "self"),
            "scaldef-m": rejected("Geo::m(1)", AglScopeError, "self"),
        },
        legal=frozenset({(1, 2), (3,)}),
    ),
    "use-region-beside-imported-record": Scenario(
        modules={"shapes": _SHAPES, "tl": _TL},
        header=(
            "import tl::*",
            "import shapes",
        ),
        probes={
            "nodef-inner-annot": rejected(
                "scope r\n  use shapes::*\n  def g(p: Geo::Inner) -> int = 1\nend r",
                UnknownMemberError,
                "Geo::Inner",
            ),
            "nodef-inner-val": rejected(
                "scope r\n  use shapes::*\n  let q = Geo::Inner(y = 1)\nend r",
                UnknownMemberError,
                "Geo::Inner",
            ),
            "nodef-point-annot": accepted(
                ("scope r\n  use shapes::*\n  def g(p: Geo::Point) -> int = 1\nend r\n\nr::g"),
                "shapes::Geo::Point -> int",
            ),
            "nodef-point-val": accepted(
                "scope r\n  use shapes::*\n  let q = Geo::Point(x = 1)\nend r\n\nr::q",
                "record shapes::Geo::Point\n  x: int",
            ),
            "nodef-f": rejected(
                "scope r\n  use shapes::*\n  let q = Geo::f()\nend r", UnknownMemberError, "Geo::f"
            ),
            "near-inner-annot": rejected(
                (
                    "scope r\n"
                    "  use shapes::*\n"
                    "  def Geo::f() -> int = 1\n"
                    "  def g(p: Geo::Inner) -> int = 1\n"
                    "end r"
                ),
                UnknownMemberError,
                "Geo::Inner",
            ),
            "near-inner-val": rejected(
                (
                    "scope r\n"
                    "  use shapes::*\n"
                    "  def Geo::f() -> int = 1\n"
                    "  let q = Geo::Inner(y = 1)\n"
                    "end r"
                ),
                UnknownMemberError,
                "Geo::Inner",
            ),
            "near-point-annot": accepted(
                (
                    "scope r\n"
                    "  use shapes::*\n"
                    "  def Geo::f() -> int = 1\n"
                    "  def g(p: Geo::Point) -> int = 1\n"
                    "end r\n"
                    "\n"
                    "r::g"
                ),
                "shapes::Geo::Point -> int",
            ),
            "near-point-val": accepted(
                (
                    "scope r\n"
                    "  use shapes::*\n"
                    "  def Geo::f() -> int = 1\n"
                    "  let q = Geo::Point(x = 1)\n"
                    "end r\n"
                    "\n"
                    "r::q"
                ),
                "record shapes::Geo::Point\n  x: int",
            ),
            "near-f": accepted(
                (
                    "scope r\n"
                    "  use shapes::*\n"
                    "  def Geo::f() -> int = 1\n"
                    "  let q = Geo::f()\n"
                    "end r\n"
                    "\n"
                    "r::q"
                ),
                "int",
            ),
        },
    ),
    "imported-enum-without-method": Scenario(
        modules={"en": _EN},
        header=("import en::*",),
        probes={
            "nodef-pat": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "nodef-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "nodef-pat-miss": rejected(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Nope => 1\n  | _ => 2"),
                UnknownMemberError,
                "Shape::Nope",
            ),
        },
    ),
    "method-on-local-enum": Scenario(
        modules={"en": _EN},
        header=(
            "enum Shape\n  | Circle\n  | Square",
            "def Shape::area(self) -> int = 1",
        ),
        probes={
            "loc-pat": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "loc-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "loc-pat-miss": rejected(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Nope => 1\n  | _ => 2"),
                UnknownMemberError,
                "Shape::Nope",
            ),
        },
    ),
    "function-in-region-over-imported-nested-record": Scenario(
        modules={"tl": _TL_2},
        header=(
            "import tl::*",
            "scope R\n  def Geo::f() -> int = 1\nend R",
        ),
        probes={
            "def-val": rejected("R::Geo::X(z = 1)", UnknownMemberError, "R::Geo::X"),
            "def-annot": rejected("fn(p: R::Geo::X) => 1", UnknownMemberError, "R::Geo::X"),
            "def-pat": rejected(
                "case tl::R::Geo::X(z = 1) of\n  | R::Geo::X(z) => z",
                UnknownMemberError,
                "R::Geo::X",
            ),
            "def-is": rejected("1 is R::Geo::X", UnknownMemberError, "R::Geo::X"),
            "def-reptype": rejected("R::Geo::X", UnknownMemberError, "R::Geo::X"),
            "def-alias": rejected("type AA = R::Geo::X\n1", UnknownMemberError, "R::Geo::X"),
        },
    ),
    "region-named-like-imported-record": Scenario(
        modules={"tl": _TL_2},
        header=(
            "import tl::*",
            "scope R\n  def g() -> int = 1\nend R",
        ),
        probes={
            "nodef-val": rejected("R::Geo::X(z = 1)", UnknownMemberError, "R::Geo::X"),
            "nodef-annot": rejected("fn(p: R::Geo::X) => 1", UnknownMemberError, "R::Geo::X"),
            "nodef-pat": rejected(
                "case tl::R::Geo::X(z = 1) of\n  | R::Geo::X(z) => z",
                UnknownMemberError,
                "R::Geo::X",
            ),
            "nodef-is": rejected("1 is R::Geo::X", UnknownMemberError, "R::Geo::X"),
            "nodef-reptype": rejected("R::Geo::X", UnknownMemberError, "R::Geo::X"),
            "nodef-alias": rejected("type AA = R::Geo::X\n1", UnknownMemberError, "R::Geo::X"),
        },
    ),
    "function-in-module-route-namespace": Scenario(
        modules={"pal": _PAL},
        header=(
            "import pal",
            "def pal::f() -> int = 1",
        ),
        probes={
            "defns-rc-val": rejected("pal::Color::Red", RouteClashError, "pal::Color::Red"),
            "defns-rc-annot": rejected(
                "fn(p: pal::Color::Red) => 1", RouteClashError, "pal::Color::Red"
            ),
            "defns-rc-pat": rejected(
                "case 1 of\n  | pal::Color::Red => 1\n  | _ => 2",
                RouteClashError,
                "pal::Color::Red",
            ),
            "defns-f": accepted("pal::f()", "int"),
        },
    ),
    "function-in-imported-scope-namespace": Scenario(
        modules={"shapes": _SHAPES_2},
        header=(
            "import shapes::*",
            "def Geo::f() -> int = 1",
        ),
        probes={
            "defpath3-val": rejected("Geo::In::Point(x = 1)", UnknownMemberError, "Geo::In::Point"),
            "defpath3-annot": rejected(
                "fn(p: Geo::In::Point) => 1", UnknownMemberError, "Geo::In::Point"
            ),
            "defpath3-miss-val": rejected(
                "Geo::In::Nope(x = 1)", UnknownMemberError, "Geo::In::Nope"
            ),
            "defpath3-miss-annot": rejected(
                "fn(p: Geo::In::Nope) => 1", UnknownMemberError, "Geo::In::Nope"
            ),
            "defpath2-miss-val": rejected("Geo::Nope(x = 1)", UnknownMemberError, "Geo::Nope"),
            "defpath2-miss-annot": rejected(
                "fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"
            ),
        },
    ),
    "function-in-imported-nested-scope-namespace": Scenario(
        modules={"shapes": _SHAPES_2},
        header=(
            "import shapes::*",
            "def Geo::In::f() -> int = 1",
        ),
        probes={
            "defnested-annot": rejected(
                "fn(p: Geo::In::Point) => 1", UnknownMemberError, "Geo::In::Point"
            ),
            "defnested-val": rejected(
                "Geo::In::Point(x = 1)", UnknownMemberError, "Geo::In::Point"
            ),
        },
    ),
    "function-beside-imported-scope": Scenario(
        modules={"shapes": _SHAPES},
        header=(
            "import shapes::*",
            "def Geo::f() -> int = 1",
        ),
        probes={
            "is": rejected(
                "shapes::Geo::Point(x = 1) is Geo::Point", UnknownMemberError, "Geo::Point"
            ),
            "val": rejected("Geo::Point(x = 1)", UnknownMemberError, "Geo::Point"),
            "annot": rejected("fn(p: Geo::Point) => 1", UnknownMemberError, "Geo::Point"),
            "pat": rejected(
                "let p = shapes::Geo::Point(x = 1)\ncase p of\n  | Geo::Point(x) => x",
                UnknownMemberError,
                "Geo::Point",
            ),
            "val-f": accepted("Geo::f()", "int"),
        },
    ),
    "function-namespace-beside-local-record": Scenario(
        header=(
            "def Geo::f() -> int = 1",
            "record R\n  x: int",
        ),
        probes={
            "noimp-pat": rejected(
                "case R(x = 1) of\n  | Geo::Point(x) => x", UnknownMemberError, "Geo::Point"
            ),
            "noimp-is": rejected("R(x = 1) is Geo::Point", UnknownMemberError, "Geo::Point"),
            "noimp-val": rejected("Geo::Point(x = 1)", UnknownMemberError, "Geo::Point"),
        },
    ),
}


class TestDefCreatedNamespaces:
    """Owners' members through a method path; own function namespaces shadow imports."""

    @pytest.mark.parametrize(("scenario", "sizes"), scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(
        self, tmp_path: Path, scenario: str, sizes: tuple[int, ...]
    ) -> None:
        assert_scenario_for_grouping(tmp_path, _SCENARIOS[scenario], sizes)
