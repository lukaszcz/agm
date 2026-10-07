"""A qualifier head naming both a local declaration and a module route.

A chain is looked up as its whole path: a path the own module declares
wins over the same path through a module route; a path only the route
reaches is the route's; a ``::`` anchor reads the current module only and a
``/``-anchored path the module only. A local scope and a same-named
imported scope look combined, each supplying the members it declares.

Every probe is checked in the file part and in every legal REPL grouping of its
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
    UnknownQualifierError,
)
from agm.agl.typecheck.checker import EnumOwnerMismatchError
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_scenario,
    option_identity,
    rejected,
    scenario_params,
    type_positions_rejected,
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

# ``Foo::E::Z`` beside a plain scope ``Foo`` declaring ``Z`` but no ``E``: the
# chain never skips the missing segment to reach ``Foo::Z``.
_MISSING_SEGMENT_PROBES = {
    "value": rejected("Foo::E::Z(w = 1)", UnknownMemberError, "Foo::E::Z"),
    **type_positions_rejected("type", "Foo::E::Z", UnknownMemberError),
    "pattern": rejected(
        "let z = Foo::Z(w = 1)\ncase z of\n  | Foo::E::Z(w) => w", UnknownMemberError, "Foo::E::Z"
    ),
    "is": rejected("let z = Foo::Z(w = 1)\nz is Foo::E::Z", UnknownMemberError, "Foo::E::Z"),
    "narrow": rejected("let z = Foo::Z(w = 1)\nz as? Foo::E::Z", UnknownMemberError, "Foo::E::Z"),
    "type-entry": rejected("Foo::E::Z", UnknownMemberError, "Foo::E::Z"),
    "length-four-value": rejected("Foo::E::F::Z(w = 1)", UnknownMemberError, "Foo::E::F::Z"),
}

_SCENARIOS = {
    "module-route-spelled-with-scope-separators": Scenario(
        modules={"alpha/beta": _ALPHA_BETA},
        header=("import alpha/beta",),
        probes={
            "separators-val": rejected(
                "alpha::beta::X(v = 1)", UnknownQualifierError, "alpha::beta::X"
            ),
            "separators-annot": rejected(
                "fn(p: alpha::beta::X) => 1", UnknownQualifierError, "alpha::beta::X"
            ),
            "slash-route-val": accepted("alpha/beta::X(v = 1)", "record alpha/beta::X\n  v: int"),
        },
    ),
    "local-scopes-spelling-a-module-route": Scenario(
        modules={"alpha/beta": _ALPHA_BETA},
        header=(
            "import alpha/beta",
            ("scope alpha\n\n  scope beta\n    record Y\n      w: int\n  end beta\nend alpha"),
        ),
        probes={
            "absent-val": rejected("alpha::beta::X(v = 1)", UnknownMemberError, "alpha::beta::X"),
            "absent-annot": rejected(
                "fn(p: alpha::beta::X) => 1", UnknownMemberError, "alpha::beta::X"
            ),
            "present-val": accepted("alpha::beta::Y(w = 1)", "record alpha::beta::Y\n  w: int"),
            "present-annot": accepted("fn(p: alpha::beta::Y) => 1", "alpha::beta::Y -> int"),
        },
    ),
    "local-scope-named-like-an-imported-module": Scenario(
        modules={"palette": _PALETTE},
        header=(
            "import palette",
            "scope palette\n  record Local\nend palette",
        ),
        probes={
            "route-val": accepted("palette::Color::Red", "record palette::Color::Red"),
            "route-annot": accepted(
                "fn(p: palette::Color::Red) => 1", "palette::Color::Red -> int"
            ),
            "route-pat": rejected(
                "case 1 of\n  | palette::Color::Red => 1\n  | _ => 2",
                AglTypeError,
                "palette::Color::Red",
                phase="typecheck",
            ),
            "route-is": rejected(
                "1 is palette::Color::Red",
                AglTypeError,
                "1 is palette::Color::Red",
                phase="typecheck",
            ),
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
            "length-four-val": accepted("Foo::E::A::Z(w = 1)", "record Foo::E::A::Z\n  w: int"),
            "length-four-annot": accepted("fn(p: Foo::E::A::Z) => 1", "Foo::E::A::Z -> int"),
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
            "missing-val": rejected(
                "Foo::E::A::Z(w = 1)", AglTypeError, "w = 1", phase="typecheck"
            ),
            "missing-annot": accepted("fn(p: Foo::E::A::Z) => 1", "pkg/Foo::E::A::Z -> int"),
        },
    ),
    "local-scope-and-route-both-lacking-the-member": Scenario(
        modules={"pkg/Foo": _PKG_FOO_2},
        header=(
            "import pkg/Foo",
            "scope Foo\n  record Z\n    w: int\n  enum K\n    | Z\nend Foo",
        ),
        probes=_MISSING_SEGMENT_PROBES,
    ),
    "local-scope-without-a-route": Scenario(
        header=("scope Foo\n  record Z\n    w: int\nend Foo",),
        probes=_MISSING_SEGMENT_PROBES,
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
            "full-R-value": rejected("Foo::R(x = 1)", AglTypeError, "x = 1", phase="typecheck"),
            "full-R-annot": accepted("fn(p: Foo::R) => 1", "Foo::R -> int"),
            "full-R-alias": accepted("type AA = Foo::R\nfn(p: AA) => p", "Foo::R -> Foo::R"),
            "full-R-tyarg": accepted("fn(p: array[Foo::R]) => 1", "array[Foo::R] -> int"),
            "full-R-reptype": accepted("Foo::R", "int -> Foo::R"),
            "full-EA-value": accepted("Foo::E::A", "record Foo::E::A"),
            "full-EA-pattern": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2",
                EnumOwnerMismatchError,
                "Foo::E::A",
                phase="typecheck",
            ),
            "full-EA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A",
                EnumOwnerMismatchError,
                "v is Foo::E::A",
                phase="typecheck",
            ),
            "full-EA-annot": accepted("fn(p: Foo::E::A) => 1", "Foo::E::A -> int"),
            "full-EA-alias": accepted(
                "type AA = Foo::E::A\nfn(p: AA) => p", "Foo::E::A -> Foo::E::A"
            ),
            "full-EA-tyarg": accepted("fn(p: array[Foo::E::A]) => 1", "array[Foo::E::A] -> int"),
            "full-E-value": rejected(
                "Foo::E", AglTypeError, "Foo::E", type_entry="enum Foo::E\n  | A\n  | C"
            ),
            "full-E-annot": accepted("fn(p: Foo::E) => 1", "Foo::E -> int"),
            "full-E-alias": accepted("type AA = Foo::E\nfn(p: AA) => p", "Foo::E -> Foo::E"),
            "full-E-tyarg": accepted("fn(p: array[Foo::E]) => 1", "array[Foo::E] -> int"),
            "full-Gi-value": rejected(
                "Foo::G[int]", AglScopeError, "int", type_entry="record Foo::G[int]\n  w: int"
            ),
            "full-Gi-annot": accepted("fn(p: Foo::G[int]) => 1", "Foo::G[int] -> int"),
            "full-Gi-alias": accepted(
                "type AA = Foo::G[int]\nfn(p: AA) => p", "Foo::G[int] -> Foo::G[int]"
            ),
            "full-Gi-tyarg": accepted(
                "fn(p: array[Foo::G[int]]) => 1", "array[Foo::G[int]] -> int"
            ),
            "full-slashR-value": accepted("/pkg/Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "full-slashR-annot": accepted("fn(p: /pkg/Foo::R) => 1", "pkg/Foo::R -> int"),
            "full-slashR-alias": accepted(
                "type AA = /pkg/Foo::R\nfn(p: AA) => p", "pkg/Foo::R -> pkg/Foo::R"
            ),
            "full-slashR-tyarg": accepted(
                "fn(p: array[/pkg/Foo::R]) => 1", "array[pkg/Foo::R] -> int"
            ),
            "full-slashR-reptype": accepted("/pkg/Foo::R", "int -> pkg/Foo::R"),
            "full-anchR-value": rejected(
                "::Foo::R(x = 1)", AglTypeError, "x = 1", phase="typecheck"
            ),
            "full-anchR-annot": accepted("fn(p: ::Foo::R) => 1", "Foo::R -> int"),
            "full-anchR-alias": accepted("type AA = ::Foo::R\nfn(p: AA) => p", "Foo::R -> Foo::R"),
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
            "full-slashEA-narrow": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv as? /pkg/Foo::E::A",
                option_identity("pkg/Foo::E::A"),
            ),
            "full-slashEA-annot": accepted("fn(p: /pkg/Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "full-slashEA-alias": accepted(
                "type AA = /pkg/Foo::E::A\nfn(p: AA) => p", "pkg/Foo::E::A -> pkg/Foo::E::A"
            ),
            "full-slashEA-tyarg": accepted(
                "fn(p: array[/pkg/Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "full-anchEA-value": accepted("::Foo::E::A", "record Foo::E::A"),
            "full-anchEA-pattern": rejected(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | ::Foo::E::A => 1\n  | _ => 2"),
                EnumOwnerMismatchError,
                "::Foo::E::A",
                phase="typecheck",
            ),
            "full-anchEA-is": rejected(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv is ::Foo::E::A",
                EnumOwnerMismatchError,
                "v is ::Foo::E::A",
                phase="typecheck",
            ),
            "full-anchEA-annot": accepted("fn(p: ::Foo::E::A) => 1", "Foo::E::A -> int"),
            "full-anchEA-alias": accepted(
                "type AA = ::Foo::E::A\nfn(p: AA) => p", "Foo::E::A -> Foo::E::A"
            ),
            "full-anchEA-tyarg": accepted(
                "fn(p: array[::Foo::E::A]) => 1", "array[Foo::E::A] -> int"
            ),
        },
    ),
    "unrelated-local-scope-named-like-a-route": Scenario(
        modules={"pkg/Foo": _PKG_FOO_3},
        header=(
            "import pkg/Foo",
            "scope Foo\n  record Z\nend Foo",
        ),
        probes={
            "empty-R-value": accepted("Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "empty-R-annot": accepted("fn(p: Foo::R) => 1", "pkg/Foo::R -> int"),
            "empty-R-alias": accepted(
                "type AA = Foo::R\nfn(p: AA) => p", "pkg/Foo::R -> pkg/Foo::R"
            ),
            "empty-R-tyarg": accepted("fn(p: array[Foo::R]) => 1", "array[pkg/Foo::R] -> int"),
            "empty-R-reptype": accepted("Foo::R", "int -> pkg/Foo::R"),
            "empty-EA-value": accepted("Foo::E::A", "record pkg/Foo::E::A"),
            "empty-EA-pattern": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2",
                "int",
            ),
            "empty-EA-is": accepted("let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A", "bool"),
            "empty-EA-narrow": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv as? Foo::E::A",
                option_identity("pkg/Foo::E::A"),
            ),
            "empty-EA-annot": accepted("fn(p: Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "empty-EA-alias": accepted(
                "type AA = Foo::E::A\nfn(p: AA) => p", "pkg/Foo::E::A -> pkg/Foo::E::A"
            ),
            "empty-EA-tyarg": accepted(
                "fn(p: array[Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "empty-E-value": rejected(
                "Foo::E", AglTypeError, "Foo::E", type_entry="enum pkg/Foo::E\n  | A\n  | B"
            ),
            "empty-E-annot": accepted("fn(p: Foo::E) => 1", "pkg/Foo::E -> int"),
            "empty-E-alias": accepted(
                "type AA = Foo::E\nfn(p: AA) => p", "pkg/Foo::E -> pkg/Foo::E"
            ),
            "empty-E-tyarg": accepted("fn(p: array[Foo::E]) => 1", "array[pkg/Foo::E] -> int"),
            "empty-Gi-value": rejected(
                "Foo::G[int]", AglScopeError, "int", type_entry="record pkg/Foo::G[int]\n  v: int"
            ),
            "empty-Gi-annot": accepted("fn(p: Foo::G[int]) => 1", "pkg/Foo::G[int] -> int"),
            "empty-Gi-alias": accepted(
                "type AA = Foo::G[int]\nfn(p: AA) => p", "pkg/Foo::G[int] -> pkg/Foo::G[int]"
            ),
            "empty-Gi-tyarg": accepted(
                "fn(p: array[Foo::G[int]]) => 1", "array[pkg/Foo::G[int]] -> int"
            ),
            "empty-slashR-value": accepted("/pkg/Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "empty-slashR-annot": accepted("fn(p: /pkg/Foo::R) => 1", "pkg/Foo::R -> int"),
            "empty-slashR-alias": accepted(
                "type AA = /pkg/Foo::R\nfn(p: AA) => p", "pkg/Foo::R -> pkg/Foo::R"
            ),
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
            "empty-slashEA-narrow": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv as? /pkg/Foo::E::A",
                option_identity("pkg/Foo::E::A"),
            ),
            "empty-slashEA-annot": accepted("fn(p: /pkg/Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "empty-slashEA-alias": accepted(
                "type AA = /pkg/Foo::E::A\nfn(p: AA) => p", "pkg/Foo::E::A -> pkg/Foo::E::A"
            ),
            "empty-slashEA-tyarg": accepted(
                "fn(p: array[/pkg/Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
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
        },
    ),
    "local-enum-named-like-a-route": Scenario(
        modules={"pkg/Foo": _PKG_FOO_3},
        header=(
            "import pkg/Foo",
            "enum Foo\n  | R(y: int)\n  | E",
        ),
        probes={
            "enum-R-value": rejected("Foo::R(x = 1)", AglTypeError, "x = 1", phase="typecheck"),
            "enum-R-annot": accepted("fn(p: Foo::R) => 1", "Foo::R -> int"),
            "enum-R-alias": accepted("type AA = Foo::R\nfn(p: AA) => p", "Foo::R -> Foo::R"),
            "enum-R-tyarg": accepted("fn(p: array[Foo::R]) => 1", "array[Foo::R] -> int"),
            "enum-R-reptype": accepted("Foo::R", "int -> Foo::R"),
            "enum-EA-value": accepted("Foo::E::A", "record pkg/Foo::E::A"),
            "enum-EA-pattern": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2",
                "int",
            ),
            "enum-EA-is": accepted("let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A", "bool"),
            "enum-EA-narrow": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv as? Foo::E::A",
                option_identity("pkg/Foo::E::A"),
            ),
            "enum-EA-annot": accepted("fn(p: Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "enum-EA-alias": accepted(
                "type AA = Foo::E::A\nfn(p: AA) => p", "pkg/Foo::E::A -> pkg/Foo::E::A"
            ),
            "enum-EA-tyarg": accepted(
                "fn(p: array[Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "enum-EA-cast": accepted(
                "fn(p: text) => p as? Foo::E::A", "text -> std/option::Option[pkg/Foo::E::A]"
            ),
            "enum-E-value": accepted("Foo::E", "record Foo::E"),
            "enum-E-annot": accepted("fn(p: Foo::E) => 1", "Foo::E -> int"),
            "enum-E-alias": accepted("type AA = Foo::E\nfn(p: AA) => p", "Foo::E -> Foo::E"),
            "enum-E-tyarg": accepted("fn(p: array[Foo::E]) => 1", "array[Foo::E] -> int"),
            "enum-Gi-value": rejected(
                "Foo::G[int]", AglScopeError, "int", type_entry="record pkg/Foo::G[int]\n  v: int"
            ),
            "enum-Gi-annot": accepted("fn(p: Foo::G[int]) => 1", "pkg/Foo::G[int] -> int"),
            "enum-Gi-alias": accepted(
                "type AA = Foo::G[int]\nfn(p: AA) => p", "pkg/Foo::G[int] -> pkg/Foo::G[int]"
            ),
            "enum-Gi-tyarg": accepted(
                "fn(p: array[Foo::G[int]]) => 1", "array[pkg/Foo::G[int]] -> int"
            ),
            "enum-slashR-value": accepted("/pkg/Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "enum-slashR-annot": accepted("fn(p: /pkg/Foo::R) => 1", "pkg/Foo::R -> int"),
            "enum-slashR-alias": accepted(
                "type AA = /pkg/Foo::R\nfn(p: AA) => p", "pkg/Foo::R -> pkg/Foo::R"
            ),
            "enum-slashR-tyarg": accepted(
                "fn(p: array[/pkg/Foo::R]) => 1", "array[pkg/Foo::R] -> int"
            ),
            "enum-slashR-reptype": accepted("/pkg/Foo::R", "int -> pkg/Foo::R"),
            "enum-anchR-value": rejected(
                "::Foo::R(x = 1)", AglTypeError, "x = 1", phase="typecheck"
            ),
            "enum-anchR-annot": accepted("fn(p: ::Foo::R) => 1", "Foo::R -> int"),
            "enum-anchR-alias": accepted("type AA = ::Foo::R\nfn(p: AA) => p", "Foo::R -> Foo::R"),
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
            "enum-slashEA-narrow": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv as? /pkg/Foo::E::A",
                option_identity("pkg/Foo::E::A"),
            ),
            "enum-slashEA-annot": accepted("fn(p: /pkg/Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "enum-slashEA-alias": accepted(
                "type AA = /pkg/Foo::E::A\nfn(p: AA) => p", "pkg/Foo::E::A -> pkg/Foo::E::A"
            ),
            "enum-slashEA-tyarg": accepted(
                "fn(p: array[/pkg/Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
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
            "enum-anchEA-cast": rejected(
                "fn(p: text) => p as? ::Foo::E::A", UnknownMemberError, "::Foo::E::A"
            ),
        },
    ),
    "route-without-local-declaration": Scenario(
        modules={"pkg/Foo": _PKG_FOO_3},
        header=("import pkg/Foo",),
        probes={
            "none-R-value": accepted("Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "none-R-annot": accepted("fn(p: Foo::R) => 1", "pkg/Foo::R -> int"),
            "none-R-alias": accepted(
                "type AA = Foo::R\nfn(p: AA) => p", "pkg/Foo::R -> pkg/Foo::R"
            ),
            "none-R-tyarg": accepted("fn(p: array[Foo::R]) => 1", "array[pkg/Foo::R] -> int"),
            "none-R-reptype": accepted("Foo::R", "int -> pkg/Foo::R"),
            "none-EA-value": accepted("Foo::E::A", "record pkg/Foo::E::A"),
            "none-EA-pattern": accepted(
                ("let v: pkg/Foo::E = pkg/Foo::E::A\ncase v of\n  | Foo::E::A => 1\n  | _ => 2"),
                "int",
            ),
            "none-EA-is": accepted("let v: pkg/Foo::E = pkg/Foo::E::A\nv is Foo::E::A", "bool"),
            "none-EA-narrow": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv as? Foo::E::A",
                option_identity("pkg/Foo::E::A"),
            ),
            "none-EA-annot": accepted("fn(p: Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "none-EA-alias": accepted(
                "type AA = Foo::E::A\nfn(p: AA) => p", "pkg/Foo::E::A -> pkg/Foo::E::A"
            ),
            "none-EA-tyarg": accepted(
                "fn(p: array[Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
            "none-E-value": rejected(
                "Foo::E", AglTypeError, "Foo::E", type_entry="enum pkg/Foo::E\n  | A\n  | B"
            ),
            "none-E-annot": accepted("fn(p: Foo::E) => 1", "pkg/Foo::E -> int"),
            "none-E-alias": accepted(
                "type AA = Foo::E\nfn(p: AA) => p", "pkg/Foo::E -> pkg/Foo::E"
            ),
            "none-E-tyarg": accepted("fn(p: array[Foo::E]) => 1", "array[pkg/Foo::E] -> int"),
            "none-Gi-value": rejected(
                "Foo::G[int]", AglScopeError, "int", type_entry="record pkg/Foo::G[int]\n  v: int"
            ),
            "none-Gi-annot": accepted("fn(p: Foo::G[int]) => 1", "pkg/Foo::G[int] -> int"),
            "none-Gi-alias": accepted(
                "type AA = Foo::G[int]\nfn(p: AA) => p", "pkg/Foo::G[int] -> pkg/Foo::G[int]"
            ),
            "none-Gi-tyarg": accepted(
                "fn(p: array[Foo::G[int]]) => 1", "array[pkg/Foo::G[int]] -> int"
            ),
            "none-slashR-value": accepted("/pkg/Foo::R(x = 1)", "record pkg/Foo::R\n  x: int"),
            "none-slashR-annot": accepted("fn(p: /pkg/Foo::R) => 1", "pkg/Foo::R -> int"),
            "none-slashR-alias": accepted(
                "type AA = /pkg/Foo::R\nfn(p: AA) => p", "pkg/Foo::R -> pkg/Foo::R"
            ),
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
            "none-slashEA-narrow": accepted(
                "let v: pkg/Foo::E = pkg/Foo::E::A\nv as? /pkg/Foo::E::A",
                option_identity("pkg/Foo::E::A"),
            ),
            "none-slashEA-annot": accepted("fn(p: /pkg/Foo::E::A) => 1", "pkg/Foo::E::A -> int"),
            "none-slashEA-alias": accepted(
                "type AA = /pkg/Foo::E::A\nfn(p: AA) => p", "pkg/Foo::E::A -> pkg/Foo::E::A"
            ),
            "none-slashEA-tyarg": accepted(
                "fn(p: array[/pkg/Foo::E::A]) => 1", "array[pkg/Foo::E::A] -> int"
            ),
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
            "use-E-value": rejected("Foo::E", AmbiguousQualificationError, "Foo::E"),
            "use-E-annot": rejected("fn(p: Foo::E) => 1", AmbiguousQualificationError, "Foo::E"),
            "use-E-alias": rejected("type AA = Foo::E\n1", AmbiguousQualificationError, "Foo::E"),
            "use-E-tyarg": rejected(
                "fn(p: array[Foo::E]) => 1", AmbiguousQualificationError, "Foo::E"
            ),
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
            "annot-circle": accepted(
                "fn(p: Geo::Shape::Circle) => 1", "shapes::Geo::Shape::Circle -> int"
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
                AglTypeError,
                "s is Geo::Shape::Circle",
                phase="typecheck",
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
            "imported-value": accepted("Geo::Point(x = 1)", "record shapes::Geo::Point\n  x: int"),
            "imported-value-enum": accepted(
                "Geo::Shape::Circle", "record shapes::Geo::Shape::Circle"
            ),
            "imported-pattern": rejected(
                "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2",
                AglTypeError,
                "Geo::Point(x)",
                phase="typecheck",
            ),
            "imported-pattern-enum": rejected(
                "case 1 of\n  | Geo::Shape::Circle => 1\n  | _ => 2",
                AglTypeError,
                "Geo::Shape::Circle",
                phase="typecheck",
            ),
            "imported-is-enum": rejected(
                "1 is Geo::Shape::Circle",
                AglTypeError,
                "1 is Geo::Shape::Circle",
                phase="typecheck",
            ),
            "imported-annot": accepted("fn(p: Geo::Point) => 1", "shapes::Geo::Point -> int"),
            "imported-annot-enum": accepted("fn(p: Geo::Shape) => 1", "shapes::Geo::Shape -> int"),
            "imported-annot-member": accepted(
                "fn(p: Geo::Shape::Circle) => 1", "shapes::Geo::Shape::Circle -> int"
            ),
            "imported-alias": accepted(
                "type A = Geo::Point\nfn(p: A) => p", "shapes::Geo::Point -> shapes::Geo::Point"
            ),
            "imported-tyarg": accepted(
                "fn(p: array[Geo::Point]) => 1", "array[shapes::Geo::Point] -> int"
            ),
            "imported-applied": accepted(
                "fn(p: Geo::Box[int]) => 1", "shapes::Geo::Box[int] -> int"
            ),
            "imported-applied-value": accepted(
                "Geo::Box(v = 1)", "record shapes::Geo::Box[int]\n  v: int"
            ),
            "imported-reptype": accepted("Geo::Point", "int -> shapes::Geo::Point"),
            "imported-reptype-applied": rejected(
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
            "locnoP-value": accepted("Geo::Point(x = 1)", "record shapes::Geo::Point\n  x: int"),
            "locnoP-value-enum": accepted(
                "Geo::Shape::Circle", "record shapes::Geo::Shape::Circle"
            ),
            "locnoP-pattern": rejected(
                "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2",
                AglTypeError,
                "Geo::Point(x)",
                phase="typecheck",
            ),
            "locnoP-pattern-enum": rejected(
                "case 1 of\n  | Geo::Shape::Circle => 1\n  | _ => 2",
                AglTypeError,
                "Geo::Shape::Circle",
                phase="typecheck",
            ),
            "locnoP-is-enum": rejected(
                "1 is Geo::Shape::Circle",
                AglTypeError,
                "1 is Geo::Shape::Circle",
                phase="typecheck",
            ),
            "locnoP-annot": accepted("fn(p: Geo::Point) => 1", "shapes::Geo::Point -> int"),
            "locnoP-annot-enum": accepted("fn(p: Geo::Shape) => 1", "shapes::Geo::Shape -> int"),
            "locnoP-annot-member": accepted(
                "fn(p: Geo::Shape::Circle) => 1", "shapes::Geo::Shape::Circle -> int"
            ),
            "locnoP-alias": accepted(
                "type A = Geo::Point\nfn(p: A) => p", "shapes::Geo::Point -> shapes::Geo::Point"
            ),
            "locnoP-tyarg": accepted(
                "fn(p: array[Geo::Point]) => 1", "array[shapes::Geo::Point] -> int"
            ),
            "locnoP-applied": accepted("fn(p: Geo::Box[int]) => 1", "shapes::Geo::Box[int] -> int"),
            "locnoP-applied-value": accepted(
                "Geo::Box(v = 1)", "record shapes::Geo::Box[int]\n  v: int"
            ),
            "locnoP-reptype": accepted("Geo::Point", "int -> shapes::Geo::Point"),
            "locnoP-reptype-applied": rejected(
                "Geo::Box[int]",
                AglScopeError,
                "int",
                type_entry="record shapes::Geo::Box[int]\n  v: int",
            ),
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
            "locP-value-enum": accepted("Geo::Shape::Circle", "record shapes::Geo::Shape::Circle"),
            "locP-pattern": rejected(
                "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2",
                AglTypeError,
                "Geo::Point(x)",
                phase="typecheck",
            ),
            "locP-pattern-enum": rejected(
                "case 1 of\n  | Geo::Shape::Circle => 1\n  | _ => 2",
                AglTypeError,
                "Geo::Shape::Circle",
                phase="typecheck",
            ),
            "locP-is-enum": rejected(
                "1 is Geo::Shape::Circle",
                AglTypeError,
                "1 is Geo::Shape::Circle",
                phase="typecheck",
            ),
            "locP-annot": accepted("fn(p: Geo::Point) => 1", "Geo::Point -> int"),
            "locP-annot-enum": accepted("fn(p: Geo::Shape) => 1", "Geo::Shape -> int"),
            "locP-annot-member": accepted(
                "fn(p: Geo::Shape::Circle) => 1", "shapes::Geo::Shape::Circle -> int"
            ),
            "locP-alias": accepted(
                "type A = Geo::Point\nfn(p: A) => p", "Geo::Point -> Geo::Point"
            ),
            "locP-tyarg": accepted("fn(p: array[Geo::Point]) => 1", "array[Geo::Point] -> int"),
            "locP-applied": accepted("fn(p: Geo::Box[int]) => 1", "shapes::Geo::Box[int] -> int"),
            "locP-applied-value": accepted(
                "Geo::Box(v = 1)", "record shapes::Geo::Box[int]\n  v: int"
            ),
            "locP-reptype": accepted("Geo::Point", "int -> Geo::Point"),
            "locP-reptype-applied": rejected(
                "Geo::Box[int]",
                AglScopeError,
                "int",
                type_entry="record shapes::Geo::Box[int]\n  v: int",
            ),
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
            "owner-annot": accepted("fn(x: palette::Color) => 1", "palette::Color -> int"),
            "member-annot": accepted(
                "fn(x: palette::Color::Red) => 1", "palette::Color::Red -> int"
            ),
            "member-value": accepted("palette::Color::Red", "record palette::Color::Red"),
            "member-is": accepted(
                "let v: palette::Color = palette::Color::Red\nv is palette::Color::Red", "bool"
            ),
            "member-narrow": accepted(
                "let v: palette::Color = palette::Color::Red\nv as? palette::Color::Red",
                option_identity("palette::Color::Red"),
            ),
            "member-pat": accepted(
                "let v: palette::Color = palette::Color::Red\n"
                "case v of\n  | palette::Color::Red => 1\n  | _ => 2",
                "int",
            ),
            "owner-value-rec": accepted("palette::Other(x = 1)", "record palette::Other\n  x: int"),
            "surface-value": accepted("palette::Red", "record palette::Color::Red"),
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
            "val": accepted("Geo::T(x = 1)", "record S::T\n  x: int"),
            "annot": accepted("fn(p: Geo::T) => 1", "S::T -> int"),
            "val-enum": accepted("Geo::K::A", "record S::K::A"),
            "annot-enum": accepted("fn(p: Geo::K) => 1", "S::K -> int"),
            "annot-enum-member": accepted("fn(p: Geo::K::A) => 1", "S::K::A -> int"),
            "pat": accepted("let t = S::T(x = 1)\ncase t of\n  | Geo::T(x) => x", "int"),
            "is-enum": accepted("let k: S::K = S::K::A\nk is Geo::K::A", "bool"),
            "narrow-enum": accepted(
                "let k: S::K = S::K::A\nk as? Geo::K::A",
                option_identity("S::K::A"),
            ),
            "val-other": accepted("Geo::Other(y = 1)", "record Geo::Other\n  y: int"),
            "annot-other": accepted("fn(p: Geo::Other) => 1", "Geo::Other -> int"),
        },
        legal=frozenset({(2, 1), (3,)}),
    ),
}


class TestRouteAndLocalScopeQualifiers:
    """Own paths beside module routes, anchors, and local scopes beside imported ones."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)
