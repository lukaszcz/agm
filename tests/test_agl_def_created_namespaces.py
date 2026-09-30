"""A ``def`` path's namespace beside the types and scopes it spells.

A method (``def Owner::m(self)``) joins its owner's namespace, so the
owner's members stay reachable through the same path; a plain function's
path (``def Geo::f()``) only declares ``Geo::f``: it never hides a
same-named imported type or scope, whose members stay reachable beside it.
A renaming alias's method is its target's; receivers a scalar cannot own
are rejected.

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
    UnknownMemberError,
)
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_file_resolves_like_inline_entry,
    assert_repl_verdicts,
    assert_scenario,
    file_params,
    option_identity,
    rejected,
    scenario_params,
    type_positions,
    type_positions_rejected,
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
            "anchored-m": accepted("::Geo::m(Geo(x = 1))", "int"),
            **type_positions_rejected("anchored-nope", "::Geo::Nope", UnknownMemberError),
            "anchored-val": rejected("::Geo::Inner(y = 1)", UnknownMemberError, "::Geo::Inner"),
            **type_positions_rejected("anchored", "::Geo::Inner", UnknownMemberError),
            "method-ref-val": accepted("Geo::m(Geo(x = 1))", "int"),
            "method-owner-val": accepted("Geo::Inner(y = 1)", "record tl::Geo::Inner\n  y: int"),
            **type_positions("method-owner", "Geo::Inner", "tl::Geo::Inner"),
            "method-owner-miss-val": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
            **type_positions_rejected("method-owner-miss", "Geo::Nope", UnknownMemberError),
        },
    ),
    "method-on-imported-enum": Scenario(
        modules={"en": _EN},
        header=(
            "import en::*",
            "def Shape::area(self) -> int = 1",
        ),
        probes={
            "member-value": accepted("Shape::Circle", "record en::Shape::Circle"),
            "member-pattern": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "member-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "member-narrow": accepted(
                "let v: Shape = Shape::Square\nv as? Shape::Circle",
                option_identity("en::Shape::Circle"),
            ),
            "member-annot": accepted("fn(p: Shape::Circle) => 1", "en::Shape::Circle -> int"),
            "member-alias": accepted(
                "type AA = Shape::Circle\nfn(p: AA) => p", "en::Shape::Circle -> en::Shape::Circle"
            ),
            "member-tyarg": accepted(
                "fn(p: array[Shape::Circle]) => 1", "array[en::Shape::Circle] -> int"
            ),
            "member-reptype": accepted("Shape::Circle", "record en::Shape::Circle"),
            "member-call": accepted("Shape::Circle.area()", "int"),
            "member-mref": accepted("Shape::area(Shape::Circle)", "int"),
            "missing-value": rejected("Shape::Nope", UnknownMemberError, "Shape::Nope"),
            "missing-annot": rejected("fn(p: Shape::Nope) => 1", UnknownMemberError, "Shape::Nope"),
            "missing-pattern": rejected(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Nope => 1\n  | _ => 2"),
                UnknownMemberError,
                "Shape::Nope",
            ),
            "missing-is": rejected(
                "let v: Shape = Shape::Square\nv is Shape::Nope", UnknownMemberError, "Shape::Nope"
            ),
            "wild-pat": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "wild-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "wild-narrow": accepted(
                "let v: Shape = Shape::Square\nv as? Shape::Circle",
                option_identity("en::Shape::Circle"),
            ),
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
            "use-pat": accepted(
                "let v = tl::Geo::Inner(y = 1)\ncase v of\n  | Geo::Inner(y) => y", "int"
            ),
            "use-m": accepted("Geo::m(tl::Geo(x = 1))", "int"),
            "use-call": accepted("tl::Geo(x = 1).m()", "int"),
        },
    ),
    "method-on-region-record-opened-by-use": Scenario(
        header=(
            "use R::*",
            ("scope R\n  record Geo\n    x: int\n  record Geo::Inner\n    y: int\nend R"),
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "region-use-val": accepted("Geo::Inner(y = 1)", "record R::Geo::Inner\n  y: int"),
            "region-use-annot": accepted("fn(p: Geo::Inner) => 1", "R::Geo::Inner -> int"),
            "region-use-miss-annot": rejected(
                "fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"
            ),
            "region-use-pat": accepted(
                "let v = R::Geo::Inner(y = 1)\ncase v of\n  | Geo::Inner(y) => y", "int"
            ),
            "region-use-m": accepted("Geo::m(R::Geo(x = 1))", "int"),
        },
        legal=frozenset({(2, 1, 1), (2, 2), (3, 1), (4,)}),
    ),
    "method-on-imported-alias": Scenario(
        modules={"al": _AL},
        header=("import al::*",),
        probes={
            # An alias's method is its target's.
            "receiver": accepted("def Geo::m(self) -> int = self.x\nBase(x = 2).m()", "int"),
            "nope-val": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
            "nope-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
        },
    ),
    "method-on-ambiguous-imported-record": Scenario(
        modules={"tl": _TL, "tl2": _TL2},
        header=("import tl::*", "import tl2::*"),
        probes={
            "receiver": rejected(
                "def Geo::m(self) -> int = 1", AmbiguousQualificationError, "self"
            ),
            "nested-receiver": rejected(
                "def Geo::Inner::m(self) -> int = 1", AmbiguousQualificationError, "Geo::Inner"
            ),
            "annot": rejected("fn(p: Geo::Inner) => p", AmbiguousQualificationError, "Geo::Inner"),
            # The first error in source order: the miss before the ambiguous receiver.
            "miss-before-receiver": rejected(
                "let f = fn(p: Geo::Inner::Nope) => 1\ndef Geo::Inner::m(self) -> int = 1\nf",
                UnknownMemberError,
                "Geo::Inner::Nope",
            ),
        },
    ),
    "scoped-method-on-imported-record": Scenario(
        modules={"tl": _TL},
        header=("import tl::*",),
        probes={
            "nested-imported": accepted(
                (
                    "scope r\n"
                    "  def Geo::m(p: int) -> int = p\n"
                    "  def g(p: Geo::Inner) -> int = 1\n"
                    "  let q = Geo::Inner(y = 1)\n"
                    "end r\n"
                    "\n"
                    "r::g"
                ),
                "tl::Geo::Inner -> int",
            ),
            "receiver-names-the-region-path": rejected(
                "scope r\n  def Geo::m(self) -> int = 1\nend r", AglScopeError, "self"
            ),
        },
    ),
    "scoped-method-on-local-record": Scenario(
        header=(
            "record Geo\n  x: int",
            "record Geo::Inner\n  y: int",
        ),
        probes={
            "nested-local": accepted(
                (
                    "scope r\n"
                    "  def Geo::m(p: int) -> int = p\n"
                    "  def g(p: Geo::Inner) -> int = 1\n"
                    "  let q = Geo::Inner(y = 1)\n"
                    "end r\n"
                    "\n"
                    "r::g"
                ),
                "Geo::Inner -> int",
            ),
            "receiver-names-the-region-path": rejected(
                "scope r\n  def Geo::m(self) -> int = 1\nend r", AglScopeError, "self"
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
            "region-prefix-val": accepted(
                "Geo::Point(x = 1)", "record shapes::Geo::Point\n  x: int"
            ),
            "region-prefix-annot": accepted("fn(p: Geo::Point) => 1", "shapes::Geo::Point -> int"),
            "region-prefix-pat": accepted(
                "let p = shapes::Geo::Point(x = 1)\ncase p of\n  | Geo::Point(x) => x", "int"
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
            "type-prefix-val": accepted("Geo::Inner(y = 1)", "record tl::Geo::Inner\n  y: int"),
            "type-prefix-annot": accepted("fn(p: Geo::Inner) => 1", "tl::Geo::Inner -> int"),
            "type-prefix-miss-annot": rejected(
                "fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"
            ),
            "type-prefix-miss-val": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
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
            "enum-use-val": accepted("Shape::Circle", "record en::Shape::Circle"),
            "enum-use-annot": accepted("fn(p: Shape::Circle) => 1", "en::Shape::Circle -> int"),
            "enum-use-pat": accepted(
                ("let v: Shape = en::Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "enum-use-is": accepted("let v: Shape = en::Shape::Circle\nv is Shape::Circle", "bool"),
            "enum-use-narrow": accepted(
                "let v: Shape = en::Shape::Circle\nv as? Shape::Circle",
                option_identity("en::Shape::Circle"),
            ),
            "enum-use-call": accepted("en::Shape::Circle.area()", "int"),
            "use-pat": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "use-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "use-narrow": accepted(
                "let v: Shape = Shape::Square\nv as? Shape::Circle",
                option_identity("en::Shape::Circle"),
            ),
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
                option_identity("int"),
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
            "region-use-nodef-annot": accepted("fn(p: Geo::Inner) => 1", "R::Geo::Inner -> int"),
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
            "use-nodef-annot": accepted("fn(p: Geo::Inner) => 1", "tl::Geo::Inner -> int"),
        },
    ),
    "function-in-imported-alias-namespace": Scenario(
        modules={"al": _AL},
        header=(
            "import al::*",
            "def Geo::f() -> int = 1",
        ),
        probes={
            "alias-inner-val": accepted("Geo::Inner(y = 1)", "record al::Base::Inner\n  y: int"),
            "alias-inner-annot": accepted("fn(p: Geo::Inner) => 1", "al::Base::Inner -> int"),
            "alias-nope-val": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
            "alias-nope-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "alias-f": accepted("Geo::f()", "int"),
        },
    ),
    "function-in-imported-scalar-alias-namespace": Scenario(
        modules={"sc": _SC},
        header=(
            "import sc::*",
            "def Geo::f() -> int = 1",
        ),
        probes={
            "scalar-nope-val": rejected("Geo::Nope", UnknownMemberError, "Geo::Nope"),
            "scalar-nope-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "scalar-nope-pat": rejected(
                "case 1 of\n  | Geo::Nope => 1\n  | _ => 2", UnknownMemberError, "Geo::Nope"
            ),
            "scalar-f": accepted("Geo::f()", "int"),
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
        header=("import al::*",),
        probes={
            "receiver": accepted("def Geo::m(self) -> int = 1\nGeo::m(1)", "int"),
            "nope-val": rejected("Geo::Nope", UnknownMemberError, "Geo::Nope"),
            "nope-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "method-val": rejected("Geo::m(1)", UnknownMemberError, "Geo::m"),
        },
    ),
    "use-region-beside-imported-record": Scenario(
        modules={"shapes": _SHAPES, "tl": _TL},
        header=(
            "import tl::*",
            "import shapes",
        ),
        probes={
            "nodef-inner-annot": accepted(
                "scope r\n  use shapes::*\n  def g(p: Geo::Inner) -> int = 1\nend r\n\nr::g",
                "tl::Geo::Inner -> int",
            ),
            "nodef-inner-val": accepted(
                "scope r\n  use shapes::*\n  let q = Geo::Inner(y = 1)\nend r\n\nr::q",
                "record tl::Geo::Inner\n  y: int",
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
            "near-inner-annot": accepted(
                "scope r\n"
                "  use shapes::*\n"
                "  def Geo::f() -> int = 1\n"
                "  def g(p: Geo::Inner) -> int = 1\n"
                "end r\n\nr::g",
                "tl::Geo::Inner -> int",
            ),
            "near-inner-val": accepted(
                "scope r\n"
                "  use shapes::*\n"
                "  def Geo::f() -> int = 1\n"
                "  let q = Geo::Inner(y = 1)\n"
                "end r\n\nr::q",
                "record tl::Geo::Inner\n  y: int",
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
            "nodef-narrow": accepted(
                "let v: Shape = Shape::Square\nv as? Shape::Circle",
                option_identity("en::Shape::Circle"),
            ),
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
            "local-pat": accepted(
                ("let v: Shape = Shape::Circle\ncase v of\n  | Shape::Circle => 1\n  | _ => 2"),
                "int",
            ),
            "local-is": accepted("let v: Shape = Shape::Square\nv is Shape::Circle", "bool"),
            "local-narrow": accepted(
                "let v: Shape = Shape::Square\nv as? Shape::Circle",
                option_identity("Shape::Circle"),
            ),
            "local-pat-miss": rejected(
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
            "def-val": accepted("R::Geo::X(z = 1)", "record tl::R::Geo::X\n  z: int"),
            "def-annot": accepted("fn(p: R::Geo::X) => 1", "tl::R::Geo::X -> int"),
            "def-pat": accepted("case tl::R::Geo::X(z = 1) of\n  | R::Geo::X(z) => z", "int"),
            "def-is": rejected("1 is R::Geo::X", AglTypeError, "1 is R::Geo::X", phase="typecheck"),
            "def-reptype": accepted("R::Geo::X", "int -> tl::R::Geo::X"),
            "def-alias": accepted(
                "type AA = R::Geo::X\nfn(p: AA) => p", "tl::R::Geo::X -> tl::R::Geo::X"
            ),
        },
    ),
    "region-named-like-imported-record": Scenario(
        modules={"tl": _TL_2},
        header=(
            "import tl::*",
            "scope R\n  def g() -> int = 1\nend R",
        ),
        probes={
            "nodef-val": accepted("R::Geo::X(z = 1)", "record tl::R::Geo::X\n  z: int"),
            "nodef-annot": accepted("fn(p: R::Geo::X) => 1", "tl::R::Geo::X -> int"),
            "nodef-pat": accepted("case tl::R::Geo::X(z = 1) of\n  | R::Geo::X(z) => z", "int"),
            "nodef-is": rejected(
                "1 is R::Geo::X", AglTypeError, "1 is R::Geo::X", phase="typecheck"
            ),
            "nodef-reptype": accepted("R::Geo::X", "int -> tl::R::Geo::X"),
            "nodef-alias": accepted(
                "type AA = R::Geo::X\nfn(p: AA) => p", "tl::R::Geo::X -> tl::R::Geo::X"
            ),
        },
    ),
    "function-in-module-route-namespace": Scenario(
        modules={"pal": _PAL},
        header=(
            "import pal",
            "def pal::f() -> int = 1",
        ),
        probes={
            "route-member-val": accepted("pal::Color::Red", "record pal::Color::Red"),
            "route-member-annot": accepted("fn(p: pal::Color::Red) => 1", "pal::Color::Red -> int"),
            "route-member-pat": rejected(
                "case 1 of\n  | pal::Color::Red => 1\n  | _ => 2",
                AglTypeError,
                "pal::Color::Red",
                phase="typecheck",
            ),
            "route-f": accepted("pal::f()", "int"),
        },
    ),
    "function-in-imported-scope-namespace": Scenario(
        modules={"shapes": _SHAPES_2},
        header=(
            "import shapes::*",
            "def Geo::f() -> int = 1",
        ),
        probes={
            "length-three-val": accepted(
                "Geo::In::Point(x = 1)", "record shapes::Geo::In::Point\n  x: int"
            ),
            **type_positions("length-three", "Geo::In::Point", "shapes::Geo::In::Point"),
            "length-three-miss-val": rejected(
                "Geo::In::Nope(x = 1)", UnknownMemberError, "Geo::In::Nope"
            ),
            **type_positions_rejected("length-three-miss", "Geo::In::Nope", UnknownMemberError),
            "length-two-miss-val": rejected("Geo::Nope(x = 1)", UnknownMemberError, "Geo::Nope"),
            **type_positions_rejected("length-two-miss", "Geo::Nope", UnknownMemberError),
        },
    ),
    "function-in-imported-nested-scope-namespace": Scenario(
        modules={"shapes": _SHAPES_2},
        header=(
            "import shapes::*",
            "def Geo::In::f() -> int = 1",
        ),
        probes={
            "nested-annot": accepted("fn(p: Geo::In::Point) => 1", "shapes::Geo::In::Point -> int"),
            "nested-val": accepted(
                "Geo::In::Point(x = 1)", "record shapes::Geo::In::Point\n  x: int"
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
                "shapes::Geo::Point(x = 1) is Geo::Point",
                AglTypeError,
                "shapes::Geo::Point(x = 1) is Geo::Point",
                phase="typecheck",
            ),
            "val": accepted("Geo::Point(x = 1)", "record shapes::Geo::Point\n  x: int"),
            "annot": accepted("fn(p: Geo::Point) => 1", "shapes::Geo::Point -> int"),
            "pat": accepted(
                "let p = shapes::Geo::Point(x = 1)\ncase p of\n  | Geo::Point(x) => x", "int"
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
    """Owners' members through a method path; function paths beside imported types and scopes."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", file_params(_SCENARIOS))
    def test_a_file_resolves_like_the_inline_entry(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_file_resolves_like_inline_entry(tmp_path, scenario)


class TestImportAfterAMethod:
    """An import entered after a method leaves the method's decided receiver in place.

    A file declares every import before other declarations, so only a REPL
    history enters one later: the new import joins each later lookup, while
    the method keeps the receiver its own entry selected.
    """

    _MODULES = {"tl": _TL, "tl2": _TL2}

    def test_nested_receiver(self, tmp_path: Path) -> None:
        assert_repl_verdicts(
            tmp_path,
            self._MODULES,
            ("import tl::*", "def Geo::Inner::m(self) -> Geo::Inner = self", "import tl2::*"),
            {
                "method": accepted("Geo::Inner::m", "tl::Geo::Inner -> tl::Geo::Inner"),
                "owner-annot": rejected(
                    "fn(p: Geo::Inner) => p", AmbiguousQualificationError, "Geo::Inner"
                ),
                "miss-annot": rejected(
                    "fn(p: Geo::Inner::Nope) => 1", UnknownMemberError, "Geo::Inner::Nope"
                ),
                "miss-pat": rejected(
                    "case 1 of\n  | Geo::Inner::Nope => 1\n  | _ => 2",
                    UnknownMemberError,
                    "Geo::Inner::Nope",
                ),
                "miss-is": rejected(
                    "1 is Geo::Inner::Nope", UnknownMemberError, "Geo::Inner::Nope"
                ),
                "miss-val": rejected("Geo::Inner::Nope", UnknownMemberError, "Geo::Inner::Nope"),
            },
        )

    def test_owner_receiver(self, tmp_path: Path) -> None:
        assert_repl_verdicts(
            tmp_path,
            self._MODULES,
            ("import tl::*", "def Geo::m(self) -> Geo = self", "import tl2::*"),
            {
                "method": accepted("Geo::m", "tl::Geo -> tl::Geo"),
                "member-val": rejected(
                    "Geo::Inner(y = 1)", AmbiguousQualificationError, "Geo::Inner"
                ),
                "member-annot": rejected(
                    "fn(p: Geo::Inner) => p", AmbiguousQualificationError, "Geo::Inner"
                ),
                "miss-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            },
        )
