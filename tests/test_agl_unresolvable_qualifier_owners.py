"""Qualifiers whose owner cannot own the member spelled.

An unknown leading segment is an unknown qualifier; a function, binding or
builtin type is no owner; a scalar alias, a record and an anchored own
scope have no such member. A record reached through a scope is never an
enum member, so testing or matching an enum value against it is a type
error.

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
    assert_scenario_for_grouping,
    rejected,
    scenario_params,
)

_AL = "type Geo = int\n"
_LIB = "record Point\n  x: int\ndef f() -> int = 1\n"
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
            "is1": rejected("1 is missing::Item", UnknownQualifierError, "missing::Item"),
            "is2": rejected("1 is missing::A::Item", UnknownQualifierError, "missing::A::Item"),
            "pat1": rejected(
                "case 1 of\n  | missing::Item => 1\n  | _ => 2",
                UnknownQualifierError,
                "missing::Item",
            ),
            "pat1-args": rejected(
                "case 1 of\n  | missing::Item(x) => 1\n  | _ => 2",
                UnknownQualifierError,
                "missing::Item",
            ),
            "annot1": rejected("fn(p: missing::Item) => 1", UnknownQualifierError, "missing::Item"),
            "value1": rejected("missing::Item", UnknownQualifierError, "missing::Item"),
        },
    ),
    "imported-scalar-alias": Scenario(
        modules={"al": _AL},
        header=("import al::*",),
        probes={
            "scal-val": rejected("Geo::Nope", UnknownMemberError, "Geo::Nope"),
            "scal-annot": rejected("fn(p: Geo::Nope) => 1", UnknownMemberError, "Geo::Nope"),
            "scal-ctor": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
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
                "enum E = lib::f::Point\n1", UnknownMemberError, "lib::f::Point"
            ),
            "annot-fnowner": rejected(
                "fn(p: lib::f::Point) => 1", UnknownMemberError, "lib::f::Point"
            ),
            "val-fnowner": rejected("lib::f::Point(x = 1)", UnknownMemberError, "lib::f::Point"),
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
                "let x: lib::helper[int] = null\nx",
                AglTypeError,
                "let x: lib::helper[int] = null",
                phase="typecheck",
            ),
            "helper-name": rejected(
                "fn(x: lib::helper) => 1", AglTypeError, "x: lib::helper", phase="typecheck"
            ),
            "konst-name": rejected(
                "fn(x: lib::konst) => 1", AglTypeError, "x: lib::konst", phase="typecheck"
            ),
            "konst-applied": rejected(
                "fn(x: lib::konst[int]) => 1", AglTypeError, "x: lib::konst[int]", phase="typecheck"
            ),
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
            "anch-mod": rejected("fn(x: /lib::Missing) => 1", UnknownMemberError, "/lib::Missing"),
            "anch-mod-value": rejected("/lib::Missing", UnknownMemberError, "/lib::Missing"),
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
        header=("def f() -> int = 1",),
        probes={
            "localdef-owner": rejected("fn(x: f::Foo) => 1", UnknownQualifierError, "f::Foo"),
            "localdef-owner-value": rejected("f::Foo", UnknownQualifierError, "f::Foo"),
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
            "builtin-owner": rejected("fn(x: int::Foo) => 1", UnknownQualifierError, "int::Foo"),
            "builtin-owner-applied": rejected(
                "fn(x: int::Foo[int]) => 1", UnknownQualifierError, "int::Foo"
            ),
            "builtin-owner-value": rejected("int::Foo", UnknownQualifierError, "int::Foo"),
            "anch-cur-bare": rejected("fn(x: ::Missing) => 1", UnknownMemberError, "::Missing"),
            "anch-cur-bare-applied": rejected(
                "fn(x: ::Missing[int]) => 1", UnknownMemberError, "::Missing"
            ),
            "anch-cur-bare-value": rejected("::Missing", UnknownMemberError, "::Missing"),
        },
    ),
    "anchored-miss-in-local-scope": Scenario(
        modules={"lib": _LIB_2},
        header=("scope A\n  record R\nend A",),
        probes={
            "anch-cur": rejected(
                "fn(x: ::A::Missing[int]) => 1", UnknownMemberError, "::A::Missing"
            ),
            "anch-cur-name": rejected(
                "fn(x: ::A::Missing) => 1", UnknownMemberError, "::A::Missing"
            ),
        },
    ),
}


class TestUnresolvableQualifierOwners:
    """Unknown qualifiers and members, and records tested as enum members."""

    @pytest.mark.parametrize(("scenario", "sizes"), scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(
        self, tmp_path: Path, scenario: str, sizes: tuple[int, ...]
    ) -> None:
        assert_scenario_for_grouping(tmp_path, _SCENARIOS[scenario], sizes)
