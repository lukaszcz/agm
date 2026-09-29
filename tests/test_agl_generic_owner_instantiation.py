"""Applied generic owners and qualified generic types.

``Alias[A]::Member`` instantiates the alias's target with *A* substituted
through the alias (nested, flipped, phantom or reusing the target's
parameter names), and must fit the scrutinee. Type arguments on a
non-generic owner, or of the wrong count, are rejected where the name is
looked up; a declaration nested beneath an applied owner is a type error. A
qualified generic type names the same declaration in every type position.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`): both modes reach
the same verdict, error span and message, or accepted identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError
from agm.agl.scope.symbols import AglScopeError
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_scenario_for_grouping,
    rejected,
    scenario_params,
)

_GM = (
    "enum Slot[T]\n"
    "  | Filled(value: T)\n"
    "  | Empty\n"
    "type Rows[A] = Slot[array[A]]\n"
    "type R2[B] = Rows[dict[text, B]]\n"
    "type Flip[A, B] = Two[B, A]\n"
    "enum Two[A, B]\n"
    "  | L(l: A)\n"
    "  | R(r: B)\n"
)
_LIB = (
    "record Box[T]\n"
    "  value: T\n"
    "record Point\n"
    "  value: int\n"
    "def mk() -> Point = Point(value = 1)\n"
    "def mkb() -> Box[int] = Box(value = 1)\n"
)

_SCENARIOS = {
    "generic-record-with-nested-record": Scenario(
        header=("record Box[T]\n  v: T\nrecord Box::Inner\n  x: int",),
        probes={
            "rec-pat": rejected(
                "let i = Box::Inner(x = 1)\ncase i of\n  | Box[int]::Inner(x) => x",
                AglTypeError,
                "Box[int]::Inner(x)",
                phase="typecheck",
            ),
            "rec-pat-plain": accepted(
                "let i = Box::Inner(x = 1)\ncase i of\n  | Box::Inner(x) => x", "int"
            ),
            "rec-val": rejected(
                "Box[int]::Inner(x = 1)", AglTypeError, "Box[int]::Inner(x = 1)", phase="typecheck"
            ),
            "rec-annot": rejected(
                "fn(p: Box[int]::Inner) => 1", AglTypeError, "p: Box[int]::Inner", phase="typecheck"
            ),
        },
    ),
    "generic-enum-with-nested-record": Scenario(
        header=("enum E[T]\n  | A(a: T)\n  | B\nrecord E::Inner\n  x: int",),
        probes={
            "enum-pat": rejected(
                "let i = E::Inner(x = 1)\ncase i of\n  | E[int]::Inner(x) => x",
                AglTypeError,
                "E[int]::Inner(x)",
                phase="typecheck",
            ),
            "enum-pat-plain": accepted(
                "let i = E::Inner(x = 1)\ncase i of\n  | E::Inner(x) => x", "int"
            ),
            "enum-val": rejected(
                "E[int]::Inner(x = 1)", AglTypeError, "E[int]::Inner(x = 1)", phase="typecheck"
            ),
            "enum-annot": rejected(
                "fn(p: E[int]::Inner) => 1", AglTypeError, "p: E[int]::Inner", phase="typecheck"
            ),
            "enum-val-plain": accepted("E::Inner(x = 1)", "record E::Inner\n  x: int"),
        },
    ),
    "generic-alias-with-nested-record": Scenario(
        header=("record Box[T]\n  v: T\ntype Al[T] = Box[T]\nrecord Al::Inner\n  x: int",),
        probes={
            "alias-pat": rejected(
                "let i = Al::Inner(x = 1)\ncase i of\n  | Al[int]::Inner(x) => x",
                AglTypeError,
                "Al[int]::Inner(x)",
                phase="typecheck",
            ),
        },
    ),
    "routed-generic-aliases": Scenario(
        modules={"gm": _GM},
        header=("import gm",),
        probes={
            "x-rows-pat": accepted(
                (
                    "let row: gm::Slot[array[int]] = gm::Slot::Filled(value = [1])\n"
                    "case row of\n"
                    "  | gm::Rows[int]::Filled(value) => value.size()\n"
                    "  | _ => 0"
                ),
                "int",
            ),
            "x-rows-pat-bad": rejected(
                (
                    "let row: gm::Slot[int] = gm::Slot::Filled(value = 1)\n"
                    "case row of\n"
                    "  | gm::Rows[int]::Filled(value) => 1\n"
                    "  | _ => 0"
                ),
                AglTypeError,
                "gm::Rows[int]::Filled(value)",
                phase="typecheck",
            ),
        },
    ),
    "wildcard-generic-aliases": Scenario(
        modules={"gm": _GM},
        header=("import gm::*",),
        probes={
            "x-r2-pat-bad": rejected(
                (
                    "let row: Slot[array[int]] = Slot::Empty\n"
                    "case row of\n"
                    "  | R2[int]::Filled(value) => 1\n"
                    "  | _ => 0"
                ),
                AglTypeError,
                "R2[int]::Filled(value)",
                phase="typecheck",
            ),
            "x-r2-pat": accepted(
                (
                    "let row: Slot[array[dict[text, int]]] = Slot::Empty\n"
                    "case row of\n"
                    "  | R2[int]::Filled(value) => 1\n"
                    "  | _ => 0"
                ),
                "int",
            ),
            "x-flip-is-bad": rejected(
                "let t: Two[int, text] = Two::L(l = 1)\nt is Flip[int, text]::L",
                AglTypeError,
                "t is Flip[int, text]::L",
                phase="typecheck",
            ),
            "x-flip-is": accepted(
                'let t: Two[text, int] = Two::L(l = "a")\nt is Flip[int, text]::L', "bool"
            ),
        },
    ),
    "local-alias-against-retained-binding": Scenario(
        modules={"gm": _GM},
        header=(
            "enum Slot[T]\n  | Filled(value: T)\n  | Empty",
            "type Rows[A] = Slot[array[A]]",
            "let row: Slot[int] = Slot::Filled(value = 1)",
        ),
        probes={
            "repl-local-alias": rejected(
                "case row of\n  | Rows[int]::Filled(value) => 1\n  | _ => 0",
                AglTypeError,
                "Rows[int]::Filled(value)",
                phase="typecheck",
            ),
        },
    ),
    "local-alias-in-generic-function": Scenario(
        modules={"gm": _GM},
        header=(
            "enum Slot[T]\n  | Filled(value: T)\n  | Empty",
            "type Rows[A] = Slot[array[A]]",
        ),
        probes={
            "generic-var-arg": rejected(
                ("def f[Z](r: Slot[array[Z]]) -> bool = r is Rows[Z]::Empty\nf(Slot::Empty)"),
                AglTypeError,
                "f(Slot::Empty)",
                phase="typecheck",
            ),
            "generic-var-arg-bad": rejected(
                (
                    "def f[Z](r: Slot[Z]) -> int =\n"
                    "  case r of\n"
                    "    | Rows[Z]::Filled(value) => 1\n"
                    "    | _ => 0\n"
                    "f(Slot::Empty)"
                ),
                AglTypeError,
                "Rows[Z]::Filled(value)",
                phase="typecheck",
            ),
        },
    ),
    "row-alias": Scenario(
        header=(
            (
                "enum Slot[T]\n"
                "  | Filled(value: T)\n"
                "  | Empty\n"
                "record Pair[A, B]\n"
                "  a: A\n"
                "  b: B\n"
                "enum Two[A, B]\n"
                "  | L(l: A)\n"
                "  | R(r: B)\n"
                "type Rows[A] = Slot[array[A]]"
            ),
        ),
        probes={
            "rows-pat": accepted(
                (
                    "let row: Slot[array[int]] = Slot::Filled(value = [1])\n"
                    "case row of\n"
                    "  | Rows[int]::Filled(value) => value.size()\n"
                    "  | _ => 0"
                ),
                "int",
            ),
            "rows-pat-bad": rejected(
                (
                    "let row: Slot[int] = Slot::Filled(value = 1)\n"
                    "case row of\n"
                    "  | Rows[int]::Filled(value) => 1\n"
                    "  | _ => 0"
                ),
                AglTypeError,
                "Rows[int]::Filled(value)",
                phase="typecheck",
            ),
            "rows-is": accepted(
                "let row: Slot[array[int]] = Slot::Empty\nrow is Rows[int]::Empty", "bool"
            ),
            "rows-is-bad": accepted(
                "let row: Slot[int] = Slot::Empty\nrow is Rows[int]::Empty", "bool"
            ),
            "rows-val": accepted(
                "let r: Slot[array[int]] = Rows[int]::Filled(value = [1])\nr",
                "enum Slot[array[int]]\n  | Filled(value: array[int])\n  | Empty",
            ),
            "rows-val-bad": rejected(
                "let r = Rows[int]::Filled(value = 1)\nr", AglTypeError, "1", phase="typecheck"
            ),
            "rows-annot": accepted(
                "fn(x: Rows[int]::Filled) => 1", "Slot::Filled[array[int]] -> int"
            ),
        },
    ),
    "nested-row-alias": Scenario(
        header=(
            (
                "enum Slot[T]\n"
                "  | Filled(value: T)\n"
                "  | Empty\n"
                "record Pair[A, B]\n"
                "  a: A\n"
                "  b: B\n"
                "enum Two[A, B]\n"
                "  | L(l: A)\n"
                "  | R(r: B)\n"
                "type Rows[A] = Slot[array[A]]\n"
                "type R2[B] = Rows[dict[text, B]]"
            ),
        ),
        probes={
            "alias2": accepted(
                (
                    "let row: Slot[array[dict[text, int]]] = Slot::Empty\n"
                    "case row of\n"
                    "  | R2[int]::Filled(value) => 1\n"
                    "  | R2[int]::Empty => 0"
                ),
                "int",
            ),
            "alias2-bad": rejected(
                (
                    "let row: Slot[array[int]] = Slot::Empty\n"
                    "case row of\n"
                    "  | R2[int]::Filled(value) => 1\n"
                    "  | _ => 0"
                ),
                AglTypeError,
                "R2[int]::Filled(value)",
                phase="typecheck",
            ),
        },
    ),
    "flipped-alias": Scenario(
        header=(
            (
                "enum Slot[T]\n"
                "  | Filled(value: T)\n"
                "  | Empty\n"
                "record Pair[A, B]\n"
                "  a: A\n"
                "  b: B\n"
                "enum Two[A, B]\n"
                "  | L(l: A)\n"
                "  | R(r: B)\n"
                "type Flip[A, B] = Two[B, A]"
            ),
        ),
        probes={
            "flip": accepted(
                (
                    'let t: Two[text, int] = Two::L(l = "a")\n'
                    "case t of\n"
                    "  | Flip[int, text]::L(l) => l\n"
                    '  | _ => ""'
                ),
                "text",
            ),
            "flip-bad": rejected(
                (
                    "let t: Two[int, text] = Two::L(l = 1)\n"
                    "case t of\n"
                    "  | Flip[int, text]::L(l) => 1\n"
                    "  | _ => 0"
                ),
                AglTypeError,
                "Flip[int, text]::L(l)",
                phase="typecheck",
            ),
            "flip-is": accepted(
                'let t: Two[text, int] = Two::L(l = "a")\nt is Flip[int, text]::L', "bool"
            ),
            "flip-is-bad": rejected(
                "let t: Two[int, text] = Two::L(l = 1)\nt is Flip[int, text]::L",
                AglTypeError,
                "t is Flip[int, text]::L",
                phase="typecheck",
            ),
        },
    ),
    "alias-reusing-its-target-parameter-name": Scenario(
        header=(
            (
                "enum Slot[T]\n"
                "  | Filled(value: T)\n"
                "  | Empty\n"
                "record Pair[A, B]\n"
                "  a: A\n"
                "  b: B\n"
                "enum Two[A, B]\n"
                "  | L(l: A)\n"
                "  | R(r: B)\n"
                "type S2[T] = Slot[array[T]]"
            ),
        ),
        probes={
            "samename": accepted(
                (
                    "let row: Slot[array[int]] = Slot::Empty\n"
                    "case row of\n"
                    "  | S2[int]::Filled(value) => 1\n"
                    "  | _ => 0"
                ),
                "int",
            ),
        },
    ),
    "phantom-alias": Scenario(
        header=(
            (
                "enum Slot[T]\n"
                "  | Filled(value: T)\n"
                "  | Empty\n"
                "record Pair[A, B]\n"
                "  a: A\n"
                "  b: B\n"
                "enum Two[A, B]\n"
                "  | L(l: A)\n"
                "  | R(r: B)\n"
                "type K[A] = Slot[int]"
            ),
        ),
        probes={
            "phantom": accepted(
                (
                    "let row: Slot[int] = Slot::Empty\n"
                    "case row of\n"
                    "  | K[text]::Filled(value) => value\n"
                    "  | _ => 0"
                ),
                "int",
            ),
        },
    ),
    "wrong-type-argument-count": Scenario(
        header=(
            (
                "enum Slot[T]\n"
                "  | Filled(value: T)\n"
                "  | Empty\n"
                "record Pair[A, B]\n"
                "  a: A\n"
                "  b: B\n"
                "enum Two[A, B]\n"
                "  | L(l: A)\n"
                "  | R(r: B)\n"
            ),
        ),
        probes={
            "arity-hi": rejected(
                "let row: Slot[int] = Slot::Empty\nrow is Slot[int, int]::Empty",
                AglScopeError,
                "Slot[int, int]",
            ),
            "arity-hi-annot": rejected(
                "fn(x: Slot[int, int]::Empty) => 1", AglScopeError, "Slot[int, int]"
            ),
            "arity-lo": rejected(
                "let t: Two[int, int] = Two::L(l = 1)\nt is Two[int]::L", AglScopeError, "Two[int]"
            ),
        },
    ),
    "non-generic-enum-applied": Scenario(
        header=(
            (
                "enum Slot[T]\n"
                "  | Filled(value: T)\n"
                "  | Empty\n"
                "record Pair[A, B]\n"
                "  a: A\n"
                "  b: B\n"
                "enum Two[A, B]\n"
                "  | L(l: A)\n"
                "  | R(r: B)\n"
                "enum Col = Red | Blue"
            ),
        ),
        probes={
            "nongeneric-applied": rejected(
                "let c: Col = Col::Red\nc is Col[int]::Red", AglScopeError, "Col[int]"
            ),
            "nongeneric-applied-annot": rejected(
                "fn(x: Col[int]::Red) => 1", AglScopeError, "Col[int]"
            ),
        },
    ),
    "non-generic-alias-applied": Scenario(
        header=(
            (
                "enum Slot[T]\n"
                "  | Filled(value: T)\n"
                "  | Empty\n"
                "record Pair[A, B]\n"
                "  a: A\n"
                "  b: B\n"
                "enum Two[A, B]\n"
                "  | L(l: A)\n"
                "  | R(r: B)\n"
                "type IS = Slot[int]"
            ),
        ),
        probes={
            "nongeneric-alias-applied": rejected(
                "let r: Slot[int] = Slot::Empty\nr is IS[int]::Empty", AglScopeError, "IS[int]"
            ),
        },
    ),
    "qualified-types-in-every-type-position": Scenario(
        modules={"lib": _LIB},
        header=("import lib",),
        probes={
            "cast-A": accepted(
                "let j: json = null\nj as? lib::Box[int]",
                (
                    "enum std/option::Option[lib::Box[int]]\n"
                    "  | None\n"
                    "  | Some(value: lib::Box[int])"
                ),
            ),
            "tyargs-A": rejected(
                "def id[T](x: T) -> T = x\nid::[lib::Box[int]](null)",
                AglTypeError,
                "null",
                phase="typecheck",
            ),
            "field-A": accepted("record R\n  f: lib::Box[int]\nR", "lib::Box[int] -> R"),
            "payload-A": accepted("enum E\n  | A(f: lib::Box[int])\n  | B\nE::B", "record E::B"),
            "exc-A": accepted("exception X extends Exception\n  f: lib::Box[int]\n1", "int"),
            "ret-A": rejected(
                "def f() -> lib::Box[int] = null\nf",
                AglTypeError,
                "def f() -> lib::Box[int] = null",
                phase="typecheck",
            ),
            "var-A": rejected(
                "var v: lib::Box[int] = null\nv",
                AglTypeError,
                "var v: lib::Box[int] = null",
                phase="typecheck",
            ),
            "lambda-A": accepted(
                "fn(x: lib::Box[int]) -> lib::Box[int] => x", "lib::Box[int] -> lib::Box[int]"
            ),
            "functype-A": accepted(
                "let g: (lib::Box[int]) -> int = fn(x) => 1\ng", "lib::Box[int] -> int"
            ),
            "dict-A": accepted(
                "let d: dict[text, lib::Box[int]] = {}\nd", "dict[text, lib::Box[int]]"
            ),
            "opt-A": accepted(
                "let o: Option[lib::Box[int]] = None\no",
                (
                    "enum std/option::Option[lib::Box[int]]\n"
                    "  | None\n"
                    "  | Some(value: lib::Box[int])"
                ),
            ),
            "aliasgen-A": accepted("type AA[T] = dict[T, lib::Box[int]]\n1", "int"),
            "scoped-def-A": accepted(
                "scope S\n  def f(x: lib::Box[int]) -> int = 1\nend S\n\nS::f",
                "lib::Box[int] -> int",
            ),
            "program-A": rejected(
                "program def main(x: lib::Box[int] = null) -> unit = ()",
                AglTypeError,
                "x: lib::Box[int] = null",
                phase="typecheck",
            ),
            "param-A": accepted("@param\nlet pp: Option[lib::Box[int]] = None\n1", "int"),
            "extern-A": rejected(
                "extern def ext(x: lib::Box[int]) -> int\n1",
                AglScopeError,
                "extern def ext(x: lib::Box[int]) -> int",
            ),
            "method-A": accepted(
                "record W\n  n: int\ndef W::m(self, x: lib::Box[int]) -> int = 1\n1", "int"
            ),
            "generic-def-A": accepted("def g[T](x: T, y: lib::Box[int]) -> T = x\n1", "int"),
            "parse-A": rejected("parse::[lib::Box[int]]", AglScopeError, "parse"),
            "typeentry-A": rejected(
                "lib::Box[int]",
                AglScopeError,
                "int",
                type_entry="record lib::Box[int]\n  value: int",
            ),
            "cast-N": accepted(
                "let j: json = null\nj as? lib::Point",
                ("enum std/option::Option[lib::Point]\n  | None\n  | Some(value: lib::Point)"),
            ),
            "tyargs-N": rejected(
                "def id[T](x: T) -> T = x\nid::[lib::Point](null)",
                AglTypeError,
                "null",
                phase="typecheck",
            ),
            "field-N": accepted("record R\n  f: lib::Point\nR", "lib::Point -> R"),
            "payload-N": accepted("enum E\n  | A(f: lib::Point)\n  | B\nE::B", "record E::B"),
            "exc-N": accepted("exception X extends Exception\n  f: lib::Point\n1", "int"),
            "ret-N": rejected(
                "def f() -> lib::Point = null\nf",
                AglTypeError,
                "def f() -> lib::Point = null",
                phase="typecheck",
            ),
            "var-N": rejected(
                "var v: lib::Point = null\nv",
                AglTypeError,
                "var v: lib::Point = null",
                phase="typecheck",
            ),
            "lambda-N": accepted(
                "fn(x: lib::Point) -> lib::Point => x", "lib::Point -> lib::Point"
            ),
            "functype-N": accepted(
                "let g: (lib::Point) -> int = fn(x) => 1\ng", "lib::Point -> int"
            ),
            "dict-N": accepted("let d: dict[text, lib::Point] = {}\nd", "dict[text, lib::Point]"),
            "opt-N": accepted(
                "let o: Option[lib::Point] = None\no",
                ("enum std/option::Option[lib::Point]\n  | None\n  | Some(value: lib::Point)"),
            ),
            "aliasgen-N": accepted("type AA[T] = dict[T, lib::Point]\n1", "int"),
            "scoped-def-N": accepted(
                "scope S\n  def f(x: lib::Point) -> int = 1\nend S\n\nS::f", "lib::Point -> int"
            ),
            "program-N": rejected(
                "program def main(x: lib::Point = null) -> unit = ()",
                AglTypeError,
                "x: lib::Point = null",
                phase="typecheck",
            ),
            "param-N": accepted("@param\nlet pp: Option[lib::Point] = None\n1", "int"),
            "extern-N": rejected(
                "extern def ext(x: lib::Point) -> int\n1",
                AglScopeError,
                "extern def ext(x: lib::Point) -> int",
            ),
            "method-N": accepted(
                "record W\n  n: int\ndef W::m(self, x: lib::Point) -> int = 1\n1", "int"
            ),
            "generic-def-N": accepted("def g[T](x: T, y: lib::Point) -> T = x\n1", "int"),
            "parse-N": rejected("parse::[lib::Point]", AglScopeError, "parse"),
            "typeentry-N": accepted("lib::Point", "int -> lib::Point"),
        },
    ),
}


class TestGenericOwnerInstantiation:
    """Instantiation through applied owners and aliases, in every type position."""

    @pytest.mark.parametrize(("scenario", "sizes"), scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(
        self, tmp_path: Path, scenario: Scenario, sizes: tuple[int, ...]
    ) -> None:
        assert_scenario_for_grouping(tmp_path, scenario, sizes)
