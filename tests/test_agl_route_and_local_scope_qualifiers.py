"""A qualifier head naming both a local scope and a module route.

A leading segment that is both a local declaration (scope region, type or
``use`` alias) and a module route is a route clash, whatever the rest of
the chain names; a ``::`` anchor selects the current module and a
``/``-anchored path the module. Without a clash a local scope beats a
same-named imported scope, so a member only the import declares is
unknown.

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
    UnknownQualifierError,
)
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_scenario_for_grouping,
    rejected,
    scenario_params,
)

_ALPHA_BETA = "record X\n  v: int\n"
_PALETTE = "enum Color\n  | Red\n  | Blue\n"
_PKG_FOO = "scope E\n\n  scope A\n    record Z\n      v: int\n  end A\nend E\n"
_PKG_FOO_2 = "record Other\n  v: int\n"
_PKG_FOO_3 = "record R\n  x: int\nenum E\n  | A\n  | B\nrecord G[T]\n  v: T\n"
_PKG_FOO_4 = "record R\n  x: int\nenum E\n  | A\n  | B\n"
_SHAPES = (
    "scope Geo\n"
    "  record Point\n"
    "    x: int\n"
    "  enum Shape\n"
    "    | Circle\n"
    "    | Square\n"
    "  type Num = int\n"
    "end Geo\n"
)
_SHAPES_2 = (
    "scope Geo\n"
    "  record Point\n"
    "    x: int\n"
    "  record Box[T]\n"
    "    v: T\n"
    "  enum Shape\n"
    "    | Circle\n"
    "    | Square\n"
    "end Geo\n"
)
_PALETTE_2 = "enum Color\n  | Red\n  | Blue\nrecord Other\n  x: int\n"

_SCENARIOS = {
    "module-route-spelled-with-scope-separators": Scenario(
        modules={"alpha/beta": _ALPHA_BETA},
        header=("import alpha/beta",),
        probes={
            "ab-val": rejected("alpha::beta::X(v = 1)", UnknownQualifierError, "alpha::beta::X"),
            "ab-annot": rejected(
                "fn(p: alpha::beta::X) => 1", UnknownQualifierError, "alpha::beta::X"
            ),
            "slash-val": accepted("alpha/beta::X(v = 1)", "record alpha/beta::X\n  v: int"),
        },
    ),
    "local-scopes-spelling-a-module-route": Scenario(
        modules={"alpha/beta": _ALPHA_BETA},
        header=(
            "import alpha/beta",
            ("scope alpha\n\n  scope beta\n    record Y\n      w: int\n  end beta\nend alpha"),
        ),
        probes={
            "abloc-val": rejected("alpha::beta::X(v = 1)", UnknownMemberError, "alpha::beta::X"),
            "abloc-annot": rejected(
                "fn(p: alpha::beta::X) => 1", UnknownMemberError, "alpha::beta::X"
            ),
            "abloc-Y-val": accepted("alpha::beta::Y(w = 1)", "record alpha::beta::Y\n  w: int"),
            "abloc-Y-annot": accepted("fn(p: alpha::beta::Y) => 1", "alpha::beta::Y -> int"),
        },
    ),
    "local-scope-named-like-an-imported-module": Scenario(
        modules={"palette": _PALETTE},
        header=(
            "import palette",
            "scope palette\n  record Local\nend palette",
        ),
        probes={
            "rc3-val": rejected("palette::Color::Red", RouteClashError, "palette::Color::Red"),
            "rc3-annot": rejected(
                "fn(p: palette::Color::Red) => 1", RouteClashError, "palette::Color::Red"
            ),
            "rc3-pat": rejected(
                "case 1 of\n  | palette::Color::Red => 1\n  | _ => 2",
                RouteClashError,
                "palette::Color::Red",
            ),
            "rc3-is": rejected("1 is palette::Color::Red", RouteClashError, "palette::Color::Red"),
        },
    ),
    "local-scope-matching-a-route-at-length-four": Scenario(
        modules={"pkg/Foo": _PKG_FOO},
        header=(
            "import pkg/Foo",
            (
                "scope Foo\n"
                "\n"
                "  scope E\n"
                "\n"
                "    scope A\n"
                "      record Z\n"
                "        w: int\n"
                "    end A\n"
                "  end E\n"
                "end Foo"
            ),
        ),
        probes={
            "len4-val": rejected("Foo::E::A::Z(w = 1)", RouteClashError, "Foo::E::A::Z"),
            "len4-annot": rejected("fn(p: Foo::E::A::Z) => 1", RouteClashError, "Foo::E::A::Z"),
        },
    ),
    "local-scope-missing-a-route-member-at-length-four": Scenario(
        modules={"pkg/Foo": _PKG_FOO},
        header=(
            "import pkg/Foo",
            (
                "scope Foo\n"
                "\n"
                "  scope E\n"
                "\n"
                "    scope A\n"
                "      record Q\n"
                "        w: int\n"
                "    end A\n"
                "  end E\n"
                "end Foo"
            ),
        ),
        probes={
            "len4miss-val": rejected("Foo::E::A::Z(w = 1)", RouteClashError, "Foo::E::A::Z"),
            "len4miss-annot": rejected("fn(p: Foo::E::A::Z) => 1", RouteClashError, "Foo::E::A::Z"),
        },
    ),
    "local-scope-and-route-both-lacking-the-member": Scenario(
        modules={"pkg/Foo": _PKG_FOO_2},
        header=(
            "import pkg/Foo",
            "scope Foo\n  record Z\n    w: int\n  enum K\n    | Z\nend Foo",
        ),
        probes={
            "skip-val": rejected("Foo::E::Z(w = 1)", RouteClashError, "Foo::E::Z"),
            "skip-annot": rejected("fn(p: Foo::E::Z) => 1", RouteClashError, "Foo::E::Z"),
            "skip-alias": rejected("type AA = Foo::E::Z\n1", RouteClashError, "Foo::E::Z"),
            "skip-tyarg": rejected("fn(p: array[Foo::E::Z]) => 1", RouteClashError, "Foo::E::Z"),
            "skip-pat": rejected(
                "let z = Foo::Z(w = 1)\ncase z of\n  | Foo::E::Z(w) => w",
                RouteClashError,
                "Foo::E::Z",
            ),
            "skip-is": rejected(
                "let z = Foo::Z(w = 1)\nz is Foo::E::Z", RouteClashError, "Foo::E::Z"
            ),
            "skip-reptype": rejected("Foo::E::Z", RouteClashError, "Foo::E::Z"),
            "skip3-val": rejected("Foo::E::F::Z(w = 1)", RouteClashError, "Foo::E::F::Z"),
        },
    ),
    "local-scope-without-a-route": Scenario(
        header=("scope Foo\n  record Z\n    w: int\nend Foo",),
        probes={
            "noroute-val": rejected("Foo::E::Z(w = 1)", UnknownMemberError, "Foo::E::Z"),
            "noroute-annot": rejected("fn(p: Foo::E::Z) => 1", UnknownMemberError, "Foo::E::Z"),
            "noroute-pat": rejected(
                "let z = Foo::Z(w = 1)\ncase z of\n  | Foo::E::Z(w) => w",
                UnknownMemberError,
                "Foo::E::Z",
            ),
        },
    ),
    "populated-local-scope-named-like-a-route": Scenario(
        modules={"pkg/Foo": _PKG_FOO_3},
        header=(
            "import pkg/Foo",
            (
                "scope Foo\n"
                "  record R\n"
                "    y: int\n"
                "  enum E\n"
                "    | A\n"
                "    | C\n"
                "  record G[T]\n"
                "    w: T\n"
                "end Foo"
            ),
        ),
        probes={
            "full-R-value": rejected("Foo::R(x = 1)", RouteClashError, "Foo::R"),
            "full-R-annot": rejected("fn(p: Foo::R) => 1", RouteClashError, "Foo::R"),
            "full-R-alias": rejected("type AA = Foo::R\n1", RouteClashError, "Foo::R"),
            "full-R-tyarg": rejected("fn(p: array[Foo::R]) => 1", RouteClashError, "Foo::R"),
            "full-R-reptype": rejected("Foo::R", RouteClashError, "Foo::R"),
            "full-EA-value": rejected("Foo::E::A", RouteClashError, "Foo::E::A"),
            "full-EA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2"),
                RouteClashError,
                "Foo::E::A",
            ),
            "full-EA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A", RouteClashError, "Foo::E::A"
            ),
            "full-EA-annot": rejected("fn(p: Foo::E::A) => 1", RouteClashError, "Foo::E::A"),
            "full-EA-alias": rejected("type AA = Foo::E::A\n1", RouteClashError, "Foo::E::A"),
            "full-EA-tyarg": rejected("fn(p: array[Foo::E::A]) => 1", RouteClashError, "Foo::E::A"),
            "full-EA-reptype": rejected("Foo::E::A", RouteClashError, "Foo::E::A"),
            "full-E-value": rejected("Foo::E", RouteClashError, "Foo::E"),
            "full-E-annot": rejected("fn(p: Foo::E) => 1", RouteClashError, "Foo::E"),
            "full-E-alias": rejected("type AA = Foo::E\n1", RouteClashError, "Foo::E"),
            "full-E-tyarg": rejected("fn(p: array[Foo::E]) => 1", RouteClashError, "Foo::E"),
            "full-E-reptype": rejected("Foo::E", RouteClashError, "Foo::E"),
            "full-Gi-value": rejected("Foo::G[int]", RouteClashError, "Foo::G"),
            "full-Gi-annot": rejected("fn(p: Foo::G[int]) => 1", RouteClashError, "Foo::G"),
            "full-Gi-alias": rejected("type AA = Foo::G[int]\n1", RouteClashError, "Foo::G"),
            "full-Gi-tyarg": rejected("fn(p: array[Foo::G[int]]) => 1", RouteClashError, "Foo::G"),
            "full-Gi-reptype": rejected("Foo::G[int]", RouteClashError, "Foo::G"),
            "full-slashR-value": accepted("/pkg/Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "full-slashR-annot": accepted("fn(p: /pkg/Foo::R) => 1", "pkg/Foo::R -> int"),
            "full-slashR-alias": accepted("type AA = /pkg/Foo::R\n1", "int"),
            "full-slashR-tyarg": accepted(
                "fn(p: array[/pkg/Foo::R]) => 1", "array[pkg/Foo::R] -> int"
            ),
            "full-slashR-reptype": accepted("/pkg/Foo::R", "int -> pkg/Foo::R"),
            "full-anchR-value": rejected(
                "::Foo::R(x = 1)", AglTypeError, "x = 1", phase="typecheck"
            ),
            "full-anchR-annot": accepted("fn(p: ::Foo::R) => 1", "Foo::R -> int"),
            "full-anchR-alias": accepted("type AA = ::Foo::R\n1", "int"),
            "full-anchR-tyarg": accepted("fn(p: array[::Foo::R]) => 1", "array[Foo::R] -> int"),
            "full-anchR-reptype": accepted("::Foo::R", "int -> Foo::R"),
            "full-slashEA-value": accepted("/pkg/Foo::E::A", "record pkg/Foo::E::A"),
            "full-slashEA-pattern": accepted(
                (
                    "let v: pkg/Foo::E = pkg/Foo::E::A\n"
                    "case v of\n"
                    "  | /pkg/Foo::E::A => 1\n"
                    "  | _ => 2"
                ),
                "int",
            ),
            "full-slashEA-is": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is /pkg/Foo::E::A", "bool"
            ),
            "full-slashEA-annot": accepted("fn(p: /pkg/Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "full-slashEA-alias": accepted("type AA = /pkg/Foo::E::A\n1", "int"),
            "full-slashEA-tyarg": accepted(
                "fn(p: array[/pkg/Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "full-slashEA-reptype": accepted("/pkg/Foo::E::A", "record pkg/Foo::E::A"),
            "full-anchEA-value": accepted("::Foo::E::A", "record Foo::E::A"),
            "full-anchEA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | ::Foo::E::A => 1\n  | _ => 2"),
                AglTypeError,
                "::Foo::E::A",
                phase="typecheck",
            ),
            "full-anchEA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is ::Foo::E::A",
                AglTypeError,
                "v is ::Foo::E::A",
                phase="typecheck",
            ),
            "full-anchEA-annot": accepted("fn(p: ::Foo::E::A) => 1", "Foo::E::A -> int"),
            "full-anchEA-alias": accepted("type AA = ::Foo::E::A\n1", "int"),
            "full-anchEA-tyarg": accepted(
                "fn(p: array[::Foo::E::A]) => 1", "array[Foo::E::A] -> int"
            ),
            "full-anchEA-reptype": accepted("::Foo::E::A", "record Foo::E::A"),
        },
    ),
    "unrelated-local-scope-named-like-a-route": Scenario(
        modules={"pkg/Foo": _PKG_FOO_3},
        header=(
            "import pkg/Foo",
            "scope Foo\n  record Z\nend Foo",
        ),
        probes={
            "empty-R-value": rejected("Foo::R(x = 1)", RouteClashError, "Foo::R"),
            "empty-R-annot": rejected("fn(p: Foo::R) => 1", RouteClashError, "Foo::R"),
            "empty-R-alias": rejected("type AA = Foo::R\n1", RouteClashError, "Foo::R"),
            "empty-R-tyarg": rejected("fn(p: array[Foo::R]) => 1", RouteClashError, "Foo::R"),
            "empty-R-reptype": rejected("Foo::R", RouteClashError, "Foo::R"),
            "empty-EA-value": rejected("Foo::E::A", RouteClashError, "Foo::E::A"),
            "empty-EA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2"),
                RouteClashError,
                "Foo::E::A",
            ),
            "empty-EA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A", RouteClashError, "Foo::E::A"
            ),
            "empty-EA-annot": rejected("fn(p: Foo::E::A) => 1", RouteClashError, "Foo::E::A"),
            "empty-EA-alias": rejected("type AA = Foo::E::A\n1", RouteClashError, "Foo::E::A"),
            "empty-EA-tyarg": rejected(
                "fn(p: array[Foo::E::A]) => 1", RouteClashError, "Foo::E::A"
            ),
            "empty-EA-reptype": rejected("Foo::E::A", RouteClashError, "Foo::E::A"),
            "empty-E-value": rejected("Foo::E", RouteClashError, "Foo::E"),
            "empty-E-annot": rejected("fn(p: Foo::E) => 1", RouteClashError, "Foo::E"),
            "empty-E-alias": rejected("type AA = Foo::E\n1", RouteClashError, "Foo::E"),
            "empty-E-tyarg": rejected("fn(p: array[Foo::E]) => 1", RouteClashError, "Foo::E"),
            "empty-E-reptype": rejected("Foo::E", RouteClashError, "Foo::E"),
            "empty-Gi-value": rejected("Foo::G[int]", RouteClashError, "Foo::G"),
            "empty-Gi-annot": rejected("fn(p: Foo::G[int]) => 1", RouteClashError, "Foo::G"),
            "empty-Gi-alias": rejected("type AA = Foo::G[int]\n1", RouteClashError, "Foo::G"),
            "empty-Gi-tyarg": rejected("fn(p: array[Foo::G[int]]) => 1", RouteClashError, "Foo::G"),
            "empty-Gi-reptype": rejected("Foo::G[int]", RouteClashError, "Foo::G"),
            "empty-slashR-value": accepted("/pkg/Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "empty-slashR-annot": accepted("fn(p: /pkg/Foo::R) => 1", "pkg/Foo::R -> int"),
            "empty-slashR-alias": accepted("type AA = /pkg/Foo::R\n1", "int"),
            "empty-slashR-tyarg": accepted(
                "fn(p: array[/pkg/Foo::R]) => 1", "array[pkg/Foo::R] -> int"
            ),
            "empty-slashR-reptype": accepted("/pkg/Foo::R", "int -> pkg/Foo::R"),
            "empty-anchR-value": rejected("::Foo::R(x = 1)", UnknownMemberError, "::Foo::R"),
            "empty-anchR-annot": rejected("fn(p: ::Foo::R) => 1", UnknownMemberError, "::Foo::R"),
            "empty-anchR-alias": rejected("type AA = ::Foo::R\n1", UnknownMemberError, "::Foo::R"),
            "empty-anchR-tyarg": rejected(
                "fn(p: array[::Foo::R]) => 1", UnknownMemberError, "::Foo::R"
            ),
            "empty-anchR-reptype": rejected("::Foo::R", UnknownMemberError, "::Foo::R"),
            "empty-slashEA-value": accepted("/pkg/Foo::E::A", "record pkg/Foo::E::A"),
            "empty-slashEA-pattern": accepted(
                (
                    "let v: pkg/Foo::E = pkg/Foo::E::A\n"
                    "case v of\n"
                    "  | /pkg/Foo::E::A => 1\n"
                    "  | _ => 2"
                ),
                "int",
            ),
            "empty-slashEA-is": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is /pkg/Foo::E::A", "bool"
            ),
            "empty-slashEA-annot": accepted("fn(p: /pkg/Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "empty-slashEA-alias": accepted("type AA = /pkg/Foo::E::A\n1", "int"),
            "empty-slashEA-tyarg": accepted(
                "fn(p: array[/pkg/Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "empty-slashEA-reptype": accepted("/pkg/Foo::E::A", "record pkg/Foo::E::A"),
            "empty-anchEA-value": rejected("::Foo::E::A", UnknownMemberError, "::Foo::E::A"),
            "empty-anchEA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | ::Foo::E::A => 1\n  | _ => 2"),
                UnknownMemberError,
                "::Foo::E::A",
            ),
            "empty-anchEA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is ::Foo::E::A",
                UnknownMemberError,
                "::Foo::E::A",
            ),
            "empty-anchEA-annot": rejected(
                "fn(p: ::Foo::E::A) => 1", UnknownMemberError, "::Foo::E::A"
            ),
            "empty-anchEA-alias": rejected(
                "type AA = ::Foo::E::A\n1", UnknownMemberError, "::Foo::E::A"
            ),
            "empty-anchEA-tyarg": rejected(
                "fn(p: array[::Foo::E::A]) => 1", UnknownMemberError, "::Foo::E::A"
            ),
            "empty-anchEA-reptype": rejected("::Foo::E::A", UnknownMemberError, "::Foo::E::A"),
        },
    ),
    "local-enum-named-like-a-route": Scenario(
        modules={"pkg/Foo": _PKG_FOO_3},
        header=(
            "import pkg/Foo",
            "enum Foo\n  | R(y: int)\n  | E",
        ),
        probes={
            "enum-R-value": rejected("Foo::R(x = 1)", RouteClashError, "Foo::R"),
            "enum-R-annot": rejected("fn(p: Foo::R) => 1", RouteClashError, "Foo::R"),
            "enum-R-alias": rejected("type AA = Foo::R\n1", RouteClashError, "Foo::R"),
            "enum-R-tyarg": rejected("fn(p: array[Foo::R]) => 1", RouteClashError, "Foo::R"),
            "enum-R-reptype": rejected("Foo::R", RouteClashError, "Foo::R"),
            "enum-EA-value": rejected("Foo::E::A", RouteClashError, "Foo::E::A"),
            "enum-EA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2"),
                RouteClashError,
                "Foo::E::A",
            ),
            "enum-EA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A", RouteClashError, "Foo::E::A"
            ),
            "enum-EA-annot": rejected("fn(p: Foo::E::A) => 1", RouteClashError, "Foo::E::A"),
            "enum-EA-alias": rejected("type AA = Foo::E::A\n1", RouteClashError, "Foo::E::A"),
            "enum-EA-tyarg": rejected("fn(p: array[Foo::E::A]) => 1", RouteClashError, "Foo::E::A"),
            "enum-EA-reptype": rejected("Foo::E::A", RouteClashError, "Foo::E::A"),
            "enum-E-value": rejected("Foo::E", RouteClashError, "Foo::E"),
            "enum-E-annot": rejected("fn(p: Foo::E) => 1", RouteClashError, "Foo::E"),
            "enum-E-alias": rejected("type AA = Foo::E\n1", RouteClashError, "Foo::E"),
            "enum-E-tyarg": rejected("fn(p: array[Foo::E]) => 1", RouteClashError, "Foo::E"),
            "enum-E-reptype": rejected("Foo::E", RouteClashError, "Foo::E"),
            "enum-Gi-value": rejected("Foo::G[int]", RouteClashError, "Foo::G"),
            "enum-Gi-annot": rejected("fn(p: Foo::G[int]) => 1", RouteClashError, "Foo::G"),
            "enum-Gi-alias": rejected("type AA = Foo::G[int]\n1", RouteClashError, "Foo::G"),
            "enum-Gi-tyarg": rejected("fn(p: array[Foo::G[int]]) => 1", RouteClashError, "Foo::G"),
            "enum-Gi-reptype": rejected("Foo::G[int]", RouteClashError, "Foo::G"),
            "enum-slashR-value": accepted("/pkg/Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "enum-slashR-annot": accepted("fn(p: /pkg/Foo::R) => 1", "pkg/Foo::R -> int"),
            "enum-slashR-alias": accepted("type AA = /pkg/Foo::R\n1", "int"),
            "enum-slashR-tyarg": accepted(
                "fn(p: array[/pkg/Foo::R]) => 1", "array[pkg/Foo::R] -> int"
            ),
            "enum-slashR-reptype": accepted("/pkg/Foo::R", "int -> pkg/Foo::R"),
            "enum-anchR-value": rejected(
                "::Foo::R(x = 1)", AglTypeError, "x = 1", phase="typecheck"
            ),
            "enum-anchR-annot": accepted("fn(p: ::Foo::R) => 1", "Foo::R -> int"),
            "enum-anchR-alias": accepted("type AA = ::Foo::R\n1", "int"),
            "enum-anchR-tyarg": accepted("fn(p: array[::Foo::R]) => 1", "array[Foo::R] -> int"),
            "enum-anchR-reptype": accepted("::Foo::R", "int -> Foo::R"),
            "enum-slashEA-value": accepted("/pkg/Foo::E::A", "record pkg/Foo::E::A"),
            "enum-slashEA-pattern": accepted(
                (
                    "let v: pkg/Foo::E = pkg/Foo::E::A\n"
                    "case v of\n"
                    "  | /pkg/Foo::E::A => 1\n"
                    "  | _ => 2"
                ),
                "int",
            ),
            "enum-slashEA-is": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is /pkg/Foo::E::A", "bool"
            ),
            "enum-slashEA-annot": accepted("fn(p: /pkg/Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "enum-slashEA-alias": accepted("type AA = /pkg/Foo::E::A\n1", "int"),
            "enum-slashEA-tyarg": accepted(
                "fn(p: array[/pkg/Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "enum-slashEA-reptype": accepted("/pkg/Foo::E::A", "record pkg/Foo::E::A"),
            "enum-anchEA-value": rejected("::Foo::E::A", UnknownMemberError, "::Foo::E::A"),
            "enum-anchEA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | ::Foo::E::A => 1\n  | _ => 2"),
                UnknownMemberError,
                "::Foo::E::A",
            ),
            "enum-anchEA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is ::Foo::E::A",
                UnknownMemberError,
                "::Foo::E::A",
            ),
            "enum-anchEA-annot": rejected(
                "fn(p: ::Foo::E::A) => 1", UnknownMemberError, "::Foo::E::A"
            ),
            "enum-anchEA-alias": rejected(
                "type AA = ::Foo::E::A\n1", UnknownMemberError, "::Foo::E::A"
            ),
            "enum-anchEA-tyarg": rejected(
                "fn(p: array[::Foo::E::A]) => 1", UnknownMemberError, "::Foo::E::A"
            ),
            "enum-anchEA-reptype": rejected("::Foo::E::A", UnknownMemberError, "::Foo::E::A"),
        },
    ),
    "route-without-local-declaration": Scenario(
        modules={"pkg/Foo": _PKG_FOO_3},
        header=("import pkg/Foo",),
        probes={
            "none-R-value": accepted("Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "none-R-annot": accepted("fn(p: Foo::R) => 1", "pkg/Foo::R -> int"),
            "none-R-alias": accepted("type AA = Foo::R\n1", "int"),
            "none-R-tyarg": accepted("fn(p: array[Foo::R]) => 1", "array[pkg/Foo::R] -> int"),
            "none-R-reptype": accepted("Foo::R", "int -> pkg/Foo::R"),
            "none-EA-value": accepted("Foo::E::A", "record pkg/Foo::E::A"),
            "none-EA-pattern": accepted(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2"),
                "int",
            ),
            "none-EA-is": accepted("let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A", "bool"),
            "none-EA-annot": accepted("fn(p: Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "none-EA-alias": accepted("type AA = Foo::E::A\n1", "int"),
            "none-EA-tyarg": accepted(
                "fn(p: array[Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "none-EA-reptype": accepted("Foo::E::A", "record pkg/Foo::E::A"),
            "none-E-value": rejected(
                "Foo::E", AglTypeError, "Foo::E", type_entry="enum pkg/Foo::E\n  | A\n  | B"
            ),
            "none-E-annot": accepted("fn(p: Foo::E) => 1", "pkg/Foo::E -> int"),
            "none-E-alias": accepted("type AA = Foo::E\n1", "int"),
            "none-E-tyarg": accepted("fn(p: array[Foo::E]) => 1", "array[pkg/Foo::E] -> int"),
            "none-E-reptype": rejected(
                "Foo::E", AglTypeError, "Foo::E", type_entry="enum pkg/Foo::E\n  | A\n  | B"
            ),
            "none-Gi-value": rejected(
                "Foo::G[int]", AglScopeError, "int", type_entry="record pkg/Foo::G[int]\n  v: int"
            ),
            "none-Gi-annot": accepted("fn(p: Foo::G[int]) => 1", "pkg/Foo::G[int] -> int"),
            "none-Gi-alias": accepted("type AA = Foo::G[int]\n1", "int"),
            "none-Gi-tyarg": accepted(
                "fn(p: array[Foo::G[int]]) => 1", "array[pkg/Foo::G[int]] -> int"
            ),
            "none-Gi-reptype": rejected(
                "Foo::G[int]", AglScopeError, "int", type_entry="record pkg/Foo::G[int]\n  v: int"
            ),
            "none-slashR-value": accepted("/pkg/Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "none-slashR-annot": accepted("fn(p: /pkg/Foo::R) => 1", "pkg/Foo::R -> int"),
            "none-slashR-alias": accepted("type AA = /pkg/Foo::R\n1", "int"),
            "none-slashR-tyarg": accepted(
                "fn(p: array[/pkg/Foo::R]) => 1", "array[pkg/Foo::R] -> int"
            ),
            "none-slashR-reptype": accepted("/pkg/Foo::R", "int -> pkg/Foo::R"),
            "none-anchR-value": rejected("::Foo::R(x = 1)", UnknownQualifierError, "::Foo::R"),
            "none-anchR-annot": rejected("fn(p: ::Foo::R) => 1", UnknownQualifierError, "::Foo::R"),
            "none-anchR-alias": rejected(
                "type AA = ::Foo::R\n1", UnknownQualifierError, "::Foo::R"
            ),
            "none-anchR-tyarg": rejected(
                "fn(p: array[::Foo::R]) => 1", UnknownQualifierError, "::Foo::R"
            ),
            "none-anchR-reptype": rejected("::Foo::R", UnknownQualifierError, "::Foo::R"),
            "none-slashEA-value": accepted("/pkg/Foo::E::A", "record pkg/Foo::E::A"),
            "none-slashEA-pattern": accepted(
                (
                    "let v: pkg/Foo::E = pkg/Foo::E::A\n"
                    "case v of\n"
                    "  | /pkg/Foo::E::A => 1\n"
                    "  | _ => 2"
                ),
                "int",
            ),
            "none-slashEA-is": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is /pkg/Foo::E::A", "bool"
            ),
            "none-slashEA-annot": accepted("fn(p: /pkg/Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "none-slashEA-alias": accepted("type AA = /pkg/Foo::E::A\n1", "int"),
            "none-slashEA-tyarg": accepted(
                "fn(p: array[/pkg/Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "none-slashEA-reptype": accepted("/pkg/Foo::E::A", "record pkg/Foo::E::A"),
            "none-anchEA-value": rejected("::Foo::E::A", UnknownQualifierError, "::Foo::E::A"),
            "none-anchEA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | ::Foo::E::A => 1\n  | _ => 2"),
                UnknownQualifierError,
                "::Foo::E::A",
            ),
            "none-anchEA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is ::Foo::E::A",
                UnknownQualifierError,
                "::Foo::E::A",
            ),
            "none-anchEA-annot": rejected(
                "fn(p: ::Foo::E::A) => 1", UnknownQualifierError, "::Foo::E::A"
            ),
            "none-anchEA-alias": rejected(
                "type AA = ::Foo::E::A\n1", UnknownQualifierError, "::Foo::E::A"
            ),
            "none-anchEA-tyarg": rejected(
                "fn(p: array[::Foo::E::A]) => 1", UnknownQualifierError, "::Foo::E::A"
            ),
            "none-anchEA-reptype": rejected("::Foo::E::A", UnknownQualifierError, "::Foo::E::A"),
        },
    ),
    "use-alias-named-like-a-route": Scenario(
        modules={"pkg/Foo": _PKG_FOO_4},
        header=(
            "use s as Foo",
            "import pkg/Foo",
            "scope s\n  record R\n    y: int\n  enum E\n    | A\n    | C\nend s",
        ),
        probes={
            "use-R-value": rejected("Foo::R(y = 1)", AmbiguousQualificationError, "Foo::R"),
            "use-R-annot": rejected("fn(p: Foo::R) => 1", AmbiguousQualificationError, "Foo::R"),
            "use-R-alias": rejected("type AA = Foo::R\n1", AmbiguousQualificationError, "Foo::R"),
            "use-R-tyarg": rejected(
                "fn(p: array[Foo::R]) => 1", AmbiguousQualificationError, "Foo::R"
            ),
            "use-R-reptype": rejected("Foo::R", AmbiguousQualificationError, "Foo::R"),
            "use-EA-value": rejected("Foo::E::A", AmbiguousQualificationError, "Foo::E::A"),
            "use-EA-annot": rejected(
                "fn(p: Foo::E::A) => 1", AmbiguousQualificationError, "Foo::E::A"
            ),
            "use-EA-alias": rejected(
                "type AA = Foo::E::A\n1", AmbiguousQualificationError, "Foo::E::A"
            ),
            "use-EA-tyarg": rejected(
                "fn(p: array[Foo::E::A]) => 1", AmbiguousQualificationError, "Foo::E::A"
            ),
            "use-EA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A",
                AmbiguousQualificationError,
                "Foo::E::A",
            ),
            "use-EA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2"),
                AmbiguousQualificationError,
                "Foo::E::A",
            ),
            "use-EA-reptype": rejected("Foo::E::A", AmbiguousQualificationError, "Foo::E::A"),
            "use-E-value": rejected("Foo::E", AmbiguousQualificationError, "Foo::E"),
            "use-E-annot": rejected("fn(p: Foo::E) => 1", AmbiguousQualificationError, "Foo::E"),
            "use-E-alias": rejected("type AA = Foo::E\n1", AmbiguousQualificationError, "Foo::E"),
            "use-E-tyarg": rejected(
                "fn(p: array[Foo::E]) => 1", AmbiguousQualificationError, "Foo::E"
            ),
            "use-E-reptype": rejected("Foo::E", AmbiguousQualificationError, "Foo::E"),
        },
        legal=frozenset({(3, 1), (4,)}),
    ),
    "local-scope-redeclaring-an-imported-scope": Scenario(
        modules={"shapes": _SHAPES},
        header=(
            "import shapes::*",
            (
                "scope Geo\n"
                "  record Point\n"
                "    y: int\n"
                "  enum Shape\n"
                "    | Tri\n"
                "  type Num = text\n"
                "end Geo"
            ),
        ),
        probes={
            "annot-shape": accepted("fn(p: Geo::Shape) => 1", "Geo::Shape -> int"),
            "annot-tri": accepted("fn(p: Geo::Shape::Tri) => 1", "Geo::Shape::Tri -> int"),
            "annot-circle": rejected(
                "fn(p: Geo::Shape::Circle) => 1", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "value-tri": accepted("Geo::Shape::Tri", "record Geo::Shape::Tri"),
            "let-tri": accepted(
                "let s: Geo::Shape = Geo::Shape::Tri\ns", "enum Geo::Shape\n  | Tri"
            ),
            "is-tri": rejected(
                "let s = Geo::Shape::Tri\ns is Geo::Shape::Tri",
                AglTypeError,
                "s is Geo::Shape::Tri",
                phase="typecheck",
            ),
            "is-circle": rejected(
                "let s = Geo::Shape::Tri\ns is Geo::Shape::Circle",
                UnknownMemberError,
                "Geo::Shape::Circle",
            ),
            "annot-num": accepted("fn(p: Geo::Num) => 1", "text -> int"),
            "let-num": accepted('let n: Geo::Num = "a"\nn', "text"),
            "annot-point": accepted(
                "let p: Geo::Point = Geo::Point(y = 1)\np", "record Geo::Point\n  y: int"
            ),
            "annot-point-x": rejected(
                "let p: Geo::Point = Geo::Point(x = 1)\np", AglTypeError, "x = 1", phase="typecheck"
            ),
        },
    ),
    "local-enum-beside-imported-scope": Scenario(
        modules={"shapes": _SHAPES},
        header=(
            "import shapes::*",
            "enum Shape = Tri",
        ),
        probes={
            "bare-Shape": accepted("fn(p: Shape) => 1", "Shape -> int"),
            "bare-Geo-scope": accepted("fn(p: Geo::Shape) => 1", "shapes::Geo::Shape -> int"),
        },
    ),
    "imported-scope-only": Scenario(
        modules={"shapes": _SHAPES_2},
        header=("import shapes::*",),
        probes={
            "noloc-value": accepted("Geo::Point(x = 1)", "record shapes::Geo::Point\n  x: int"),
            "noloc-value-enum": accepted("Geo::Shape::Circle", "record shapes::Geo::Shape::Circle"),
            "noloc-pattern": rejected(
                "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2",
                AglTypeError,
                "Geo::Point(x)",
                phase="typecheck",
            ),
            "noloc-pattern-enum": rejected(
                "case 1 of\n  | Geo::Shape::Circle => 1\n  | _ => 2",
                AglTypeError,
                "Geo::Shape::Circle",
                phase="typecheck",
            ),
            "noloc-is-enum": rejected(
                "1 is Geo::Shape::Circle",
                AglTypeError,
                "1 is Geo::Shape::Circle",
                phase="typecheck",
            ),
            "noloc-annot": accepted("fn(p: Geo::Point) => 1", "shapes::Geo::Point -> int"),
            "noloc-annot-enum": accepted("fn(p: Geo::Shape) => 1", "shapes::Geo::Shape -> int"),
            "noloc-annot-member": accepted(
                "fn(p: Geo::Shape::Circle) => 1", "shapes::Geo::Shape::Circle -> int"
            ),
            "noloc-alias": accepted("type A = Geo::Point\n1", "int"),
            "noloc-tyarg": accepted(
                "fn(p: array[Geo::Point]) => 1", "array[shapes::Geo::Point] -> int"
            ),
            "noloc-applied": accepted("fn(p: Geo::Box[int]) => 1", "shapes::Geo::Box[int] -> int"),
            "noloc-applied-value": accepted(
                "Geo::Box(v = 1)", "record shapes::Geo::Box[int]\n  v: int"
            ),
            "noloc-reptype": accepted("Geo::Point", "int -> shapes::Geo::Point"),
            "noloc-reptype-applied": rejected(
                "Geo::Box[int]",
                AglScopeError,
                "int",
                type_entry="record shapes::Geo::Box[int]\n  v: int",
            ),
        },
    ),
    "local-scope-lacking-imported-members": Scenario(
        modules={"shapes": _SHAPES_2},
        header=(
            "import shapes::*",
            "scope Geo\n  enum Kind = Round | Flat\nend Geo",
        ),
        probes={
            "locnoP-value": rejected("Geo::Point(x = 1)", UnknownMemberError, "Geo::Point"),
            "locnoP-value-enum": rejected(
                "Geo::Shape::Circle", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "locnoP-pattern": rejected(
                "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2", UnknownMemberError, "Geo::Point"
            ),
            "locnoP-pattern-enum": rejected(
                "case 1 of\n  | Geo::Shape::Circle => 1\n  | _ => 2",
                UnknownMemberError,
                "Geo::Shape::Circle",
            ),
            "locnoP-is-enum": rejected(
                "1 is Geo::Shape::Circle", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "locnoP-annot": rejected("fn(p: Geo::Point) => 1", UnknownMemberError, "Geo::Point"),
            "locnoP-annot-enum": rejected(
                "fn(p: Geo::Shape) => 1", UnknownMemberError, "Geo::Shape"
            ),
            "locnoP-annot-member": rejected(
                "fn(p: Geo::Shape::Circle) => 1", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "locnoP-alias": rejected("type A = Geo::Point\n1", UnknownMemberError, "Geo::Point"),
            "locnoP-tyarg": rejected(
                "fn(p: array[Geo::Point]) => 1", UnknownMemberError, "Geo::Point"
            ),
            "locnoP-applied": rejected("fn(p: Geo::Box[int]) => 1", UnknownMemberError, "Geo::Box"),
            "locnoP-applied-value": rejected("Geo::Box(v = 1)", UnknownMemberError, "Geo::Box"),
            "locnoP-reptype": rejected("Geo::Point", UnknownMemberError, "Geo::Point"),
            "locnoP-reptype-applied": rejected("Geo::Box[int]", UnknownMemberError, "Geo::Box"),
        },
    ),
    "local-scope-redeclaring-some-imported-members": Scenario(
        modules={"shapes": _SHAPES_2},
        header=(
            "import shapes::*",
            ("scope Geo\n  record Point\n    y: int\n  enum Shape\n    | Tri\nend Geo"),
        ),
        probes={
            "locP-value": rejected("Geo::Point(x = 1)", AglTypeError, "x = 1", phase="typecheck"),
            "locP-value-enum": rejected(
                "Geo::Shape::Circle", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "locP-pattern": rejected(
                "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2",
                AglTypeError,
                "Geo::Point(x)",
                phase="typecheck",
            ),
            "locP-pattern-enum": rejected(
                "case 1 of\n  | Geo::Shape::Circle => 1\n  | _ => 2",
                UnknownMemberError,
                "Geo::Shape::Circle",
            ),
            "locP-is-enum": rejected(
                "1 is Geo::Shape::Circle", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "locP-annot": accepted("fn(p: Geo::Point) => 1", "Geo::Point -> int"),
            "locP-annot-enum": accepted("fn(p: Geo::Shape) => 1", "Geo::Shape -> int"),
            "locP-annot-member": rejected(
                "fn(p: Geo::Shape::Circle) => 1", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "locP-alias": accepted("type A = Geo::Point\n1", "int"),
            "locP-tyarg": accepted("fn(p: array[Geo::Point]) => 1", "array[Geo::Point] -> int"),
            "locP-applied": rejected("fn(p: Geo::Box[int]) => 1", UnknownMemberError, "Geo::Box"),
            "locP-applied-value": rejected("Geo::Box(v = 1)", UnknownMemberError, "Geo::Box"),
            "locP-reptype": accepted("Geo::Point", "int -> Geo::Point"),
            "locP-reptype-applied": rejected("Geo::Box[int]", UnknownMemberError, "Geo::Box"),
        },
    ),
    "local-scope-beside-routed-import": Scenario(
        modules={"shapes": _SHAPES_2},
        header=(
            "import shapes",
            "scope Geo\n  enum Kind = Round | Flat\nend Geo",
        ),
        probes={
            "route-locnoP-value": rejected("Geo::Point(x = 1)", UnknownMemberError, "Geo::Point"),
            "route-locnoP-value-enum": rejected(
                "Geo::Shape::Circle", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "route-locnoP-pattern": rejected(
                "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2", UnknownMemberError, "Geo::Point"
            ),
            "route-locnoP-pattern-enum": rejected(
                "case 1 of\n  | Geo::Shape::Circle => 1\n  | _ => 2",
                UnknownMemberError,
                "Geo::Shape::Circle",
            ),
            "route-locnoP-is-enum": rejected(
                "1 is Geo::Shape::Circle", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "route-locnoP-annot": rejected(
                "fn(p: Geo::Point) => 1", UnknownMemberError, "Geo::Point"
            ),
            "route-locnoP-annot-enum": rejected(
                "fn(p: Geo::Shape) => 1", UnknownMemberError, "Geo::Shape"
            ),
            "route-locnoP-annot-member": rejected(
                "fn(p: Geo::Shape::Circle) => 1", UnknownMemberError, "Geo::Shape::Circle"
            ),
            "route-locnoP-alias": rejected(
                "type A = Geo::Point\n1", UnknownMemberError, "Geo::Point"
            ),
            "route-locnoP-tyarg": rejected(
                "fn(p: array[Geo::Point]) => 1", UnknownMemberError, "Geo::Point"
            ),
            "route-locnoP-applied": rejected(
                "fn(p: Geo::Box[int]) => 1", UnknownMemberError, "Geo::Box"
            ),
            "route-locnoP-applied-value": rejected(
                "Geo::Box(v = 1)", UnknownMemberError, "Geo::Box"
            ),
            "route-locnoP-reptype": rejected("Geo::Point", UnknownMemberError, "Geo::Point"),
            "route-locnoP-reptype-applied": rejected(
                "Geo::Box[int]", UnknownMemberError, "Geo::Box"
            ),
        },
    ),
    "local-scope-named-like-an-imported-module-with-types": Scenario(
        modules={"palette": _PALETTE_2},
        header=(
            "import palette",
            "scope palette\n  record Local\nend palette",
        ),
        probes={
            "owner-annot": rejected(
                "fn(x: palette::Color) => 1", RouteClashError, "palette::Color"
            ),
            "member-annot": rejected(
                "fn(x: palette::Color::Red) => 1", RouteClashError, "palette::Color::Red"
            ),
            "member-value": rejected("palette::Color::Red", RouteClashError, "palette::Color::Red"),
            "member-is": rejected(
                "let v = palette::Color::Red\nv is palette::Color::Red",
                RouteClashError,
                "palette::Color::Red",
            ),
            "member-pat": rejected(
                (
                    "let v = palette::Color::Red\n"
                    "case v of\n"
                    "  | palette::Color::Red => 1\n"
                    "  | _ => 2"
                ),
                RouteClashError,
                "palette::Color::Red",
            ),
            "owner-value-rec": rejected("palette::Other(x = 1)", RouteClashError, "palette::Other"),
            "surface-value": rejected("palette::Red", RouteClashError, "palette::Red"),
        },
    ),
    "use-alias-beside-same-named-local-scope": Scenario(
        header=(
            "use S as Geo",
            (
                "scope S\n"
                "  record T\n"
                "    x: int\n"
                "  enum K\n"
                "    | A\n"
                "    | B\n"
                "end S\n"
                "\n"
                "scope Geo\n"
                "  record Other\n"
                "    y: int\n"
                "end Geo"
            ),
        ),
        probes={
            "val": rejected("Geo::T(x = 1)", UnknownMemberError, "Geo::T"),
            "annot": rejected("fn(p: Geo::T) => 1", UnknownMemberError, "Geo::T"),
            "val-enum": rejected("Geo::K::A", UnknownMemberError, "Geo::K::A"),
            "annot-enum": rejected("fn(p: Geo::K) => 1", UnknownMemberError, "Geo::K"),
            "annot-enum-member": rejected("fn(p: Geo::K::A) => 1", UnknownMemberError, "Geo::K::A"),
            "pat": rejected(
                "let t = S::T(x = 1)\ncase t of\n  | Geo::T(x) => x", UnknownMemberError, "Geo::T"
            ),
            "is-enum": rejected(
                "let k: S::K = S::K::A\nk is Geo::K::A", UnknownMemberError, "Geo::K::A"
            ),
            "val-other": accepted("Geo::Other(y = 1)", "record Geo::Other\n  y: int"),
            "annot-other": accepted("fn(p: Geo::Other) => 1", "Geo::Other -> int"),
        },
        legal=frozenset({(2, 1), (3,)}),
    ),
}


class TestRouteAndLocalScopeQualifiers:
    """Route clashes, anchors, and local scopes over imported ones."""

    @pytest.mark.parametrize(("scenario", "sizes"), scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(
        self, tmp_path: Path, scenario: Scenario, sizes: tuple[int, ...]
    ) -> None:
        assert_scenario_for_grouping(tmp_path, scenario, sizes)
