"""Qualifiers whose owner cannot own the member spelled.

An unknown leading segment is an unknown qualifier; so is a chain any of
whose prefixes names a function, binding or injected enum member, own or
imported, but no scope, type or module route. A builtin type, a scalar
alias, a record and an anchored own scope have no such member. A record
reached through a scope is never an enum member, so testing or matching an
enum value against it is a type error.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`): both modes reach
the same verdict, error span and message, or accepted identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError
from agm.agl.scope.symbols import UnknownMemberError, UnknownQualifierError
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_file_resolves_like_inline_entry,
    assert_scenario,
    file_params,
    info_rejected,
    rejected,
    scenario_params,
    type_positions_rejected,
)

_AL = "type Geo = int\n"
_LIB = (
    "record Point\n  x: int\n"
    "def f() -> int = 1\n"
    "let v = 1\n"
    "enum Col = Red | Blue\n"
    "\n"
    "scope s\n  def g() -> int = 1\nend s\n"
)
_SHAPES = (
    "scope Geo\n"
    "  record Point\n"
    "    x: int\n"
    "  exception Boom extends Exception\n"
    "end Geo\n"
    "\n"
    "enum Col\n"
    "  | Red\n"
    "  | Blue\n"
)
_LIB_2 = (
    "record Box[T]\n"
    "  value: T\n"
    "record Point\n"
    "  value: int\n"
    "def helper(x: int) -> int = x\n"
    "let konst = 1\n"
    "type N = int\n"
)

_SCENARIOS = {
    "nothing-declared": Scenario(
        header=(),
        probes={
            "nodef-annot": accepted(
                "fn(p: Option::Some[int]) => 1", "std/option::Option::Some[int] -> int"
            ),
            "annot-missing": rejected(
                "fn(p: missing::Record) => 1", UnknownQualifierError, "missing::Record"
            ),
            "enum-missing": rejected(
                "enum E = missing::Record\n1", UnknownQualifierError, "missing::Record"
            ),
            "is": rejected("1 is missing::Item", UnknownQualifierError, "missing::Item"),
            "is-nested": rejected(
                "1 is missing::A::Item", UnknownQualifierError, "missing::A::Item"
            ),
            "pattern": rejected(
                "case 1 of\n  | missing::Item => 1\n  | _ => 2",
                UnknownQualifierError,
                "missing::Item",
            ),
            "pattern-args": rejected(
                "case 1 of\n  | missing::Item(x) => 1\n  | _ => 2",
                UnknownQualifierError,
                "missing::Item",
            ),
            **type_positions_rejected("item", "missing::Item", UnknownQualifierError),
            **type_positions_rejected("nested", "missing::A::Item", UnknownQualifierError),
            "value": rejected("missing::Item", UnknownQualifierError, "missing::Item"),
        },
    ),
    "imported-scalar-alias": Scenario(
        modules={"al": _AL},
        header=("import al::*",),
        probes={
            "scalar-val": rejected("Geo::Nope", UnknownMemberError, "Geo::Nope"),
            "scalar-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "scalar-ctor": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
        },
    ),
    "local-scope-lacking-the-member": Scenario(
        header=("scope Geo\n  record P\n    x: int\nend Geo",),
        probes={
            "enum-localscope-miss": rejected("enum E = Geo::Q\n1", UnknownMemberError, "Geo::Q"),
        },
    ),
    "module-function-as-owner": Scenario(
        modules={"lib": _LIB},
        header=("import lib",),
        probes={
            "enum-fnowner": rejected(
                "enum E = lib::f::Point\n1", UnknownQualifierError, "lib::f::Point"
            ),
            "annot-fnowner": rejected(
                "fn(p: lib::f::Point) => 1", UnknownQualifierError, "lib::f::Point"
            ),
            "val-fnowner": rejected("lib::f::Point(x = 1)", UnknownQualifierError, "lib::f::Point"),
            "is-fnowner": rejected("1 is lib::f::Point", UnknownQualifierError, "lib::f::Point"),
            "pat-fnowner": rejected(
                "case 1 of\n  | lib::f::Point => 1\n  | _ => 2",
                UnknownQualifierError,
                "lib::f::Point",
            ),
            "binding-owner": rejected("lib::v::Point", UnknownQualifierError, "lib::v::Point"),
            "scoped-fn-owner": rejected(
                "fn(p: lib::s::g::x) => 1", UnknownQualifierError, "lib::s::g::x"
            ),
            "injected-member-owner": rejected("lib::Red::x", UnknownQualifierError, "lib::Red::x"),
            "anchored-fn-owner": rejected("/lib::f::x", UnknownQualifierError, "/lib::f::x"),
            "anchored-scoped-fn-owner": rejected(
                "fn(p: /lib::s::g::x) => 1", UnknownQualifierError, "/lib::s::g::x"
            ),
            "member-scope-miss": rejected("lib::s::x", UnknownMemberError, "lib::s::x"),
            "member-miss": rejected("lib::Nope::x", UnknownMemberError, "lib::Nope::x"),
            "info-fn-owner": info_rejected("lib::f::Point", UnknownQualifierError, "lib::f::Point"),
            "info-binding-owner": info_rejected(
                "lib::v::Point", UnknownQualifierError, "lib::v::Point"
            ),
            "info-injected-member-owner": info_rejected(
                "lib::Red::x", UnknownQualifierError, "lib::Red::x"
            ),
            "info-member-miss": info_rejected("lib::Nope::x", UnknownMemberError, "lib::Nope::x"),
        },
    ),
    "slash-route-function-as-owner": Scenario(
        modules={"remote/config": "def Flag() -> int = 1\n"},
        header=("import remote/config", "enum Color = Red | Blue"),
        probes={
            "pat-fnowner": rejected(
                "case Color::Red of\n  | remote/config::Flag::Red => 1\n  | _ => 2",
                UnknownQualifierError,
                "remote/config::Flag::Red",
            ),
            "annot-fnowner": rejected(
                "fn(p: config::Flag::Red) => 1", UnknownQualifierError, "config::Flag::Red"
            ),
        },
    ),
    "imported-function-as-owner": Scenario(
        modules={"lib": _LIB},
        header=("import lib::*",),
        probes={
            "fn-owner": rejected("fn(p: f::x) => 1", UnknownQualifierError, "f::x"),
            "fn-owner-value": rejected("f::x", UnknownQualifierError, "f::x"),
            "binding-owner": rejected("v::x", UnknownQualifierError, "v::x"),
            "scoped-fn-owner": rejected("s::g::x", UnknownQualifierError, "s::g::x"),
            "injected-member-owner": rejected("Red::x", UnknownQualifierError, "Red::x"),
            "scope-miss": rejected("fn(p: s::x) => 1", UnknownMemberError, "s::x"),
            "type-miss": rejected("fn(p: Point::x) => 1", UnknownMemberError, "Point::x"),
        },
    ),
    "used-function-as-owner": Scenario(
        modules={"lib": _LIB},
        header=("import lib", "use lib::{f, s}"),
        probes={
            "fn-owner": rejected("f::x", UnknownQualifierError, "f::x"),
            "scoped-fn-owner": rejected("fn(p: s::g::x) => 1", UnknownQualifierError, "s::g::x"),
        },
    ),
    "scoped-record-tested-against-opened-enum": Scenario(
        modules={"shapes": _SHAPES},
        header=(
            "import shapes::*",
            "let v: Col = Col::Red",
        ),
        probes={
            "is-opened": rejected(
                "v is Geo::Point", AglTypeError, "v is Geo::Point", phase="typecheck"
            ),
            "pat-opened": rejected(
                "case v of\n  | Geo::Point(x) => x\n  | _ => 0",
                AglTypeError,
                "Geo::Point(x)",
                phase="typecheck",
            ),
        },
    ),
    "scoped-record-tested-against-routed-enum": Scenario(
        modules={"shapes": _SHAPES},
        header=(
            "import shapes",
            "let v: shapes::Col = shapes::Col::Red",
        ),
        probes={
            "is-route": rejected(
                "v is shapes::Geo::Point",
                AglTypeError,
                "v is shapes::Geo::Point",
                phase="typecheck",
            ),
            "pat-route": rejected(
                "case v of\n  | shapes::Geo::Point(x) => x\n  | _ => 0",
                AglTypeError,
                "shapes::Geo::Point(x)",
                phase="typecheck",
            ),
        },
    ),
    "scoped-record-tested-against-exception": Scenario(
        modules={"shapes": _SHAPES},
        header=(
            "import shapes::*",
            'let e: Exception = Geo::Boom(message = "m")',
        ),
        probes={
            "is-exc-opened": rejected(
                "e is Geo::Point", AglTypeError, "e is Geo::Point", phase="typecheck"
            ),
        },
    ),
    "scoped-record-tested-against-local-enum": Scenario(
        modules={"shapes": _SHAPES},
        header=(
            "scope Geo\n  record Point\n    x: int\nend Geo\n\nenum Col = Red | Blue",
            "let v: Col = Col::Red",
        ),
        probes={
            "is-local": rejected(
                "v is Geo::Point", AglTypeError, "v is Geo::Point", phase="typecheck"
            ),
        },
    ),
    "module-members-as-owners": Scenario(
        modules={"lib": _LIB_2},
        header=("import lib",),
        probes={
            "helper-applied": rejected(
                "let x: lib::helper[int] = null\nx", AglTypeError, "lib::helper"
            ),
            "helper-name": rejected("fn(x: lib::helper) => 1", AglTypeError, "lib::helper"),
            "constant-name": rejected("fn(x: lib::konst) => 1", AglTypeError, "lib::konst"),
            "constant-applied": rejected("fn(x: lib::konst[int]) => 1", AglTypeError, "lib::konst"),
            "libN-member": rejected("fn(x: lib::N::Foo) => 1", UnknownMemberError, "lib::N::Foo"),
            "libN-member-applied": rejected(
                "fn(x: lib::N::Foo[int]) => 1", UnknownMemberError, "lib::N::Foo"
            ),
            "libN-member-value": rejected("lib::N::Foo", UnknownMemberError, "lib::N::Foo"),
            "Point-member": rejected(
                "fn(x: lib::Point::Foo) => 1", UnknownMemberError, "lib::Point::Foo"
            ),
            "Point-member-applied": rejected(
                "fn(x: lib::Point::Foo[int]) => 1", UnknownMemberError, "lib::Point::Foo"
            ),
            "Point-member-value": rejected(
                "lib::Point::Foo", UnknownMemberError, "lib::Point::Foo"
            ),
            "anchored-module": rejected(
                "fn(x: /lib::Missing) => 1", UnknownMemberError, "/lib::Missing"
            ),
            "anchored-module-value": rejected("/lib::Missing", UnknownMemberError, "/lib::Missing"),
        },
    ),
    "local-scalar-alias": Scenario(
        modules={"lib": _LIB_2},
        header=("type N = int",),
        probes={
            "N-member": rejected("fn(x: N::Foo) => 1", UnknownMemberError, "N::Foo"),
            "N-member-applied": rejected("fn(x: N::Foo[int]) => 1", UnknownMemberError, "N::Foo"),
            "N-member-value": rejected("N::Foo", UnknownMemberError, "N::Foo"),
        },
    ),
    "local-function-as-owner": Scenario(
        modules={"lib": _LIB_2},
        header=(
            "def f() -> int = 1",
            "def s::g() -> int = 1",
            "record P\n  x: int",
            "def P::m(self) -> int = 1",
            "scope t\n  let v = 1\nend t",
        ),
        probes={
            "localdef-owner": rejected("fn(x: f::Foo) => 1", UnknownQualifierError, "f::Foo"),
            "localdef-owner-value": rejected("f::Foo", UnknownQualifierError, "f::Foo"),
            "scoped-def-owner": rejected("s::g::x", UnknownQualifierError, "s::g::x"),
            "scoped-def-owner-annot": rejected(
                "fn(x: s::g::x) => 1", UnknownQualifierError, "s::g::x"
            ),
            "method-owner": rejected("P::m::x", UnknownQualifierError, "P::m::x"),
            "scoped-binding-owner": rejected(
                "fn(x: t::v::x) => 1", UnknownQualifierError, "t::v::x"
            ),
            "anchored-def-owner": rejected("::s::g::x", UnknownQualifierError, "::s::g::x"),
            "scope-miss": rejected("s::x", UnknownMemberError, "s::x"),
        },
    ),
    "local-binding-as-owner": Scenario(
        modules={"lib": _LIB_2},
        header=("let q = 1",),
        probes={
            "locallet-owner": rejected("fn(x: q::Foo) => 1", UnknownQualifierError, "q::Foo"),
        },
    ),
    "builtin-and-anchored-owners": Scenario(
        modules={"lib": _LIB_2},
        header=(),
        probes={
            "builtin-owner": rejected("fn(x: int::Foo) => 1", UnknownMemberError, "int::Foo"),
            "builtin-owner-applied": rejected(
                "fn(x: int::Foo[int]) => 1", UnknownMemberError, "int::Foo"
            ),
            "builtin-owner-value": rejected("int::Foo", UnknownMemberError, "int::Foo"),
            "anchored-current-bare": rejected(
                "fn(x: ::Missing) => 1", UnknownMemberError, "::Missing"
            ),
            "anchored-current-bare-applied": rejected(
                "fn(x: ::Missing[int]) => 1", UnknownMemberError, "::Missing"
            ),
            "anchored-current-bare-value": rejected("::Missing", UnknownMemberError, "::Missing"),
        },
    ),
    "anchored-miss-in-local-scope": Scenario(
        modules={"lib": _LIB_2},
        header=("scope A\n  record R\nend A",),
        probes={
            "anchored-current": rejected(
                "fn(x: ::A::Missing[int]) => 1", UnknownMemberError, "::A::Missing"
            ),
            "anchored-current-name": rejected(
                "fn(x: ::A::Missing) => 1", UnknownMemberError, "::A::Missing"
            ),
        },
    ),
}


class TestUnresolvableQualifierOwners:
    """Unknown qualifiers and members, and records tested as enum members."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", file_params(_SCENARIOS))
    def test_a_file_resolves_like_the_inline_entry(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_file_resolves_like_inline_entry(tmp_path, scenario)
