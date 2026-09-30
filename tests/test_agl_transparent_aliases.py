"""An alias is another name for its target declaration.

A type alias, ``import m as lib``, ``use … as`` and a one-segment ``use L as
M`` add spellings, never declarations: a spelling through an alias means what
the target's spelling means at the alias's site, in every position and for
every path beneath the target. Reaching one declaration through an alias and
directly, or through two aliases, is one declaration; two distinct targets
still clash. An importer's ``hiding`` removes the declaration it names -- and
the paths beneath it -- from every spelling that import brings in, aliases
the imported module declares included. A tail or ``hiding`` item may name
a path through an alias: it names the target's declaration there, and a
path naming none is an error. An alias whose target resolves to nothing is
an error where it is declared.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError, HiddenMemberError
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    NoVisibleConstructorError,
    TypeArgumentsError,
    UnknownMemberError,
    UnknownQualifierError,
)
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    assert_file_resolves_like_inline_entry,
    assert_scenario,
    file_params,
    file_verdict,
    info,
    rejected,
    scenario_params,
    span_text,
    type_positions,
    type_positions_rejected,
)

_BASE = (
    "record Base\n  x: int\nrecord Base::Inner\n  y: int\n"
    "def Base::f() -> int = 1\nenum Base::Shape\n  | Sq\n  | Ci\n"
    "exception Oops\n  code: int\nrecord Box[T]\n  v: T\nrecord Box::In\n  w: int\n"
    "enum Color\n  | Red\n  | Green\n"
)
_EXPORTING = (
    "import base\nexport base::{Base, Color}\ntype Geo = base::Base\n"
    "type Sh = base::Base::Shape\ntype Uh = base::Oops\ntype Bx[T] = base::Box[T]\n"
    "type Hue = base::Color\n"
)
"""Exports ``base``'s ``Base`` and ``Color``, and aliases of ``base``'s declarations."""
_RENAMING = "import base\ntype Base = base::Base\ntype Geo = Base\n"
"""Declares an alias named like its target, and an alias of that alias."""
_TWIN = "import base\ntype Geo = base::Base\n"
_ELSEWHERE = "record Base\n  x: int\ntype Geo = Base\n"
_REEXPORTING = "import al\nexport al::{Geo}\n"
_EXPORTING_THROUGH = (
    "import al\nexport al::{Geo::Inner, Geo::Inner as I, Geo::Shape, Geo::f, Hue::Red}\n"
)
"""Re-exports paths written through ``al``'s aliases."""
_HIDING_THROUGH = "import al\nexport al hiding Geo::Inner\n"
_MODULES = {"base": _BASE, "al": _EXPORTING}
_ROUTES = "import ex::*\nimport ex"
"""Imports ``ex``'s exports bare and by its route, in one entry."""
_CYCLING = (
    "import base\nimport ex\nexport base::{Base, Color}\ntype Geo = base::Base\ntype J = ex::I\n"
    "def mk(p: ex::I) -> int = p.y\nrecord Holder\n  i: ex::I\n"
)
"""Aliases ``base``'s ``Base``, and reads ``ex``'s export of a path through that alias."""
_CYCLED = "import al\nexport al::{Geo::Inner as I, Geo::Shape as S}\nexport al hiding Geo::f\n"
"""Re-exports paths through ``al``'s alias, and ``al`` less a path through it."""


def _export_hiding(item: str) -> dict[str, str]:
    """The modules, with ``ex`` re-exporting ``al`` less *item*."""
    return {**_MODULES, "ex": f"import al\nexport al hiding {item}\n"}


_RECORD = "record base::Base\n  x: int"
_INNER = "record base::Base::Inner\n  y: int"
_GREEN = "record base::Color::Green"
_BASE_CASE = "fn(p: base::Base) => case p of\n    | Geo(x) => x"
_BASE_PATTERN = "fn(p: base::Base) => case p of\n  | Geo(x) => x"
"""Matches ``base``'s ``Base`` by ``Geo``."""
"""Matches ``base``'s ``Base`` by ``Geo`` in a region (see :func:`_in_region`)."""


def _in_region(use: str, value: str) -> str:
    """A region whose ``use`` exposes names that its ``let v`` reads, then ``r::v``."""
    return f"scope r\n  {use}\n  let v = {value}\nend r\n\nr::v"


def _one_declaration_probes(spelling: str, scrutinee: str = "base::Base") -> dict[str, Probe]:
    """*spelling* selecting ``base``'s ``Base`` although several sources reach it.

    *scrutinee* spells ``Base`` without them.
    """
    return {
        **type_positions(spelling, spelling, "base::Base"),
        f"{spelling}-value": accepted(f"{spelling}(x = 1)", _RECORD),
        f"{spelling}-pattern": accepted(
            f"case {scrutinee}(x = 1) of\n  | {spelling}(x) => x", "int"
        ),
        f"{spelling}-nested": accepted(f"{spelling}::Inner(y = 1)", _INNER),
        f"{spelling}-function": accepted(f"{spelling}::f", "() -> int"),
    }


def _hidden_probes(spelling: str) -> dict[str, Probe]:
    """*spelling*, a path an import's ``hiding`` removed, rejected in every position."""
    return {
        f"{spelling}-value": rejected(f"{spelling}(y = 1)", HiddenMemberError, spelling),
        **type_positions_rejected(spelling, spelling, HiddenMemberError),
    }


_SCENARIOS = {
    "an-alias-named-like-its-target-is-that-declaration": Scenario(
        modules={"base": _BASE, "al": _RENAMING},
        header=("import base::*", "import al::*"),
        probes={
            **_one_declaration_probes("Base"),
            "info": info(
                "Base", "Base is a constructor.\nSignature:\n  Base(x: int) -> base::Base"
            ),
        },
    ),
    "an-alias-chain-and-its-target-are-one-declaration": Scenario(
        modules={"base": _BASE, "al": _RENAMING, "twin": "import base::*\ntype Geo = Base\n"},
        header=("import al::*", "import twin::*"),
        probes=_one_declaration_probes("Geo", "al::Geo"),
    ),
    "two-aliases-of-one-target-are-one-declaration": Scenario(
        modules={"base": _BASE, "al": _EXPORTING, "twin": _TWIN},
        header=("import al::*", "import twin::*"),
        probes=_one_declaration_probes("Geo", "al::Geo"),
    ),
    "a-re-exported-alias-and-its-target-are-one-declaration": Scenario(
        modules={**_MODULES, "re": _REEXPORTING},
        header=("import re::*", "import base::*"),
        probes={
            **_one_declaration_probes("Geo"),
            **_one_declaration_probes("Base"),
        },
    ),
    "aliases-of-distinct-targets-clash": Scenario(
        modules={**_MODULES, "other": _ELSEWHERE},
        header=("import al::*", "import other::*"),
        probes={
            "value": rejected("Geo(x = 1)", AmbiguousConstructorError, "Geo"),
            **type_positions_rejected("type", "Geo", AmbiguousQualificationError),
            "nested-only-one-declares": accepted("Geo::Inner(y = 1)", _INNER),
        },
    ),
    "hiding-a-target-path-hides-it-through-the-imported-module-aliases": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Base::Inner",),
        probes={
            **_hidden_probes("Geo::Inner"),
            **_hidden_probes("Base::Inner"),
            "routed": rejected("al::Geo::Inner(y = 1)", HiddenMemberError, "al::Geo::Inner"),
            "alias": accepted("Geo(x = 1)", _RECORD),
            "sibling": accepted("Geo::f", "() -> int"),
        },
    ),
    "hiding-leaves-what-other-imports-bring": Scenario(
        modules=_MODULES,
        # One entry: a later REPL entry's import of ``al`` supersedes an earlier one.
        header=("import al::* hiding Base::Inner\nimport al", "import base::*"),
        probes={
            "other-import": accepted("Base::Inner(y = 1)", _INNER),
            "routed-by-another-import": accepted("al::Geo::Inner(y = 1)", _INNER),
            **_hidden_probes("Geo::Inner"),
        },
    ),
    "hiding-a-target-removes-its-aliases": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Base", "import base"),
        probes={
            "alias-value": rejected("Geo(x = 1)", AglScopeError, "Geo"),
            "alias-pattern": rejected(_BASE_PATTERN, NoVisibleConstructorError, "Geo(x)"),
            "alias-annot": rejected("fn(p: Geo) => p", AglTypeError, "Geo"),
            "alias-alias": rejected("type AA = Geo\nfn(p: AA) => p", AglTypeError, "Geo"),
            "alias-tyarg": rejected("fn(p: array[Geo]) => p", AglTypeError, "Geo"),
            "alias-cast": rejected("fn(p: text) => p as? Geo", AglTypeError, "Geo"),
            "beneath-the-alias": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
            "routed-alias": rejected("al::Geo(x = 1)", HiddenMemberError, "al::Geo"),
            "beneath-it-through-another-alias": rejected("fn(p: Sh) => p", AglTypeError, "Sh"),
            "unrelated-alias": accepted("fn(p: Uh) => p", "base::Oops -> base::Oops"),
        },
    ),
    "hiding-names-a-path-through-an-alias": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Geo::Inner",),
        probes={
            **_hidden_probes("Geo::Inner"),
            **_hidden_probes("Base::Inner"),
            "routed": rejected("al::Base::Inner(y = 1)", HiddenMemberError, "al::Base::Inner"),
            "alias": accepted("Geo(x = 1)", _RECORD),
            "target": accepted("Base(x = 1)", _RECORD),
            "sibling": accepted("Base::f", "() -> int"),
        },
    ),
    "hiding-an-alias-removes-its-target": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Geo",),
        probes={
            "target-value": rejected("Base(x = 1)", AglScopeError, "Base"),
            "beneath-the-target": rejected("Base::Inner(y = 1)", HiddenMemberError, "Base::Inner"),
            "beneath-the-alias": rejected("Geo::f", HiddenMemberError, "Geo::f"),
            "routed-beneath-the-alias": rejected("al::Geo::f", HiddenMemberError, "al::Geo::f"),
            "anchored-beneath-the-alias": rejected("/al::Geo::f", HiddenMemberError, "/al::Geo::f"),
            "beneath-it-through-another-alias": rejected("fn(p: Sh) => p", AglTypeError, "Sh"),
            "unrelated-alias": accepted("fn(p: Uh) => p", "base::Oops -> base::Oops"),
        },
    ),
    "an-item-path-through-an-alias-names-a-declaration": Scenario(
        modules=_MODULES,
        header=(),
        probes={
            "hiding": rejected(
                "import al::* hiding Geo::Nope\n1",
                UnknownMemberError,
                "import al::* hiding Geo::Nope",
            ),
            "tail": rejected(
                "import al::{Geo::Nope}\n1", UnknownMemberError, "import al::{Geo::Nope}"
            ),
        },
    ),
    "an-import-tail-names-a-path-through-an-alias": Scenario(
        modules=_MODULES,
        header=("import al::{Geo::Inner as I, Hue::Red, Geo::Shape}",),
        probes={
            "beneath-it": accepted(
                "fn(p: Geo::Shape) => p is Geo::Shape::Sq", "base::Base::Shape -> bool"
            ),
            "beneath-it-value": accepted("Geo::Shape::Sq", "record base::Base::Shape::Sq"),
            "beneath-it-unknown": rejected(
                "Geo::Shape::Nope", UnknownMemberError, "Geo::Shape::Nope"
            ),
            "path-value": accepted("Geo::Inner(y = 1)", _INNER),
            **type_positions("path", "Geo::Inner", "base::Base::Inner"),
            "rename-value": accepted("I(y = 1)", _INNER),
            "rename-pattern": accepted(
                "fn(p: Geo::Inner) => case p of\n  | I(y) => y", "base::Base::Inner -> int"
            ),
            "member": accepted("Hue::Red", "record base::Color::Red"),
            "alias-not-exposed": rejected("Geo(x = 1)", AglScopeError, "Geo"),
            "region": accepted(_in_region("import al::{Geo::Inner as J}", "J(y = 1)"), _INNER),
        },
    ),
    "a-use-hiding-removes-the-declaration-from-every-spelling-it-brings": Scenario(
        modules=_MODULES,
        header=("import al", "import base"),
        probes={
            "through-the-alias": rejected(
                _in_region("use al::* hiding Base::Inner", "Geo::Inner(y = 1)"),
                HiddenMemberError,
                "Geo::Inner",
            ),
            "named-through-an-alias": rejected(
                _in_region("use al::* hiding Geo::Inner", "Base::Inner(y = 1)"),
                HiddenMemberError,
                "Base::Inner",
            ),
            "alias-kept": accepted(
                _in_region("use al::* hiding Geo::Inner", "Geo(x = 1)"), _RECORD
            ),
            "tail-through-an-alias": accepted(
                _in_region("use al::{Geo::Inner}", "Geo::Inner(y = 1)"), _INNER
            ),
            "target-removes-its-alias": rejected(
                _in_region("use al::* hiding Base", "Geo(x = 1)"), AglScopeError, "Geo"
            ),
            "target-removes-its-alias-pattern": rejected(
                _in_region("use al::* hiding Base", _BASE_CASE), NoVisibleConstructorError, "Geo(x)"
            ),
            "target-hides-beneath-its-alias": rejected(
                _in_region("use al::* hiding Base", "Geo::Inner(y = 1)"),
                HiddenMemberError,
                "Geo::Inner",
            ),
            "member-through-an-alias": rejected(
                _in_region("use al::* hiding Hue::Red", "Red"), AglScopeError, "Red"
            ),
            "member-kept": accepted(_in_region("use al::* hiding Hue::Red", "Green"), _GREEN),
        },
    ),
    "a-region-import-hiding-removes-the-declaration": Scenario(
        modules=_MODULES,
        header=("import base",),
        probes={
            "alias-value": rejected(
                _in_region("import al::* hiding Base", "Geo(x = 1)"), AglScopeError, "Geo"
            ),
            "alias-pattern": rejected(
                _in_region("import al::* hiding Base", _BASE_CASE),
                NoVisibleConstructorError,
                "Geo(x)",
            ),
            "beneath-the-alias": rejected(
                _in_region("import al::* hiding Base", "Geo::Inner(y = 1)"),
                HiddenMemberError,
                "Geo::Inner",
            ),
            "alias-kept": accepted(
                _in_region("import al::* hiding Base::Inner", "Geo(x = 1)"), _RECORD
            ),
            "member-through-an-alias": rejected(
                _in_region("import al::* hiding Hue::Red", "Red"), AglScopeError, "Red"
            ),
            "member-kept": accepted(_in_region("import al::* hiding Hue::Red", "Green"), _GREEN),
        },
    ),
    "hiding-an-enum-member-through-an-alias": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Hue::Red",),
        probes={
            "value": rejected("Red", AglScopeError, "Red"),
            "pattern": rejected(
                "fn(p: Color) => case p of\n  | Red => 1\n  | _ => 2",
                NoVisibleConstructorError,
                "Red",
            ),
            "through-the-alias": rejected("Hue::Red", HiddenMemberError, "Hue::Red"),
            "through-the-target": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "is": rejected("fn(p: Color) => p is Hue::Red", HiddenMemberError, "Hue::Red"),
            "type": rejected("fn(p: Hue::Red) => p", HiddenMemberError, "Hue::Red"),
            "kept": accepted("Green", _GREEN),
        },
    ),
    "an-export-item-names-a-path-through-an-alias": Scenario(
        modules={**_MODULES, "ex": _EXPORTING_THROUGH},
        header=("import ex::*",),
        probes={
            "path-value": accepted("Geo::Inner(y = 1)", _INNER),
            **type_positions("path", "Geo::Inner", "base::Base::Inner"),
            "path-pattern": accepted(
                "fn(p: I) => case p of\n  | Geo::Inner(y) => y", "base::Base::Inner -> int"
            ),
            "rename-value": accepted("I(y = 1)", _INNER),
            "rename-pattern": accepted(
                "fn(p: Geo::Inner) => case p of\n  | I(y) => y", "base::Base::Inner -> int"
            ),
            "nested": accepted(
                "fn(p: Geo::Shape) => p is Geo::Shape::Sq", "base::Base::Shape -> bool"
            ),
            "nested-value": accepted("Geo::Shape::Sq", "record base::Base::Shape::Sq"),
            "function": accepted("Geo::f", "() -> int"),
            "member": accepted("Hue::Red", "record base::Color::Red"),
            "routed": accepted("ex::Geo::Inner(y = 1)", _INNER),
            "routed-rename": accepted("ex::I(y = 1)", _INNER),
            "alias-not-exposed": rejected("Geo(x = 1)", AglScopeError, "Geo"),
            "info": info("I", "I is a constructor.\nSignature:\n  I(y: int) -> base::Base::Inner"),
        },
    ),
    "an-export-hiding-names-a-path-through-an-alias": Scenario(
        modules={**_MODULES, "ex": _HIDING_THROUGH},
        header=("import ex::*",),
        probes={
            **_hidden_probes("Base::Inner"),
            **_hidden_probes("Geo::Inner"),
            "routed": rejected("ex::Base::Inner(y = 1)", HiddenMemberError, "ex::Base::Inner"),
            "target": accepted("Base(x = 1)", _RECORD),
            "sibling": accepted("Base::f", "() -> int"),
        },
    ),
    "an-export-hiding-a-target-path-hides-it-through-the-module-aliases": Scenario(
        modules=_export_hiding("Base::Inner"),
        header=(_ROUTES,),
        probes={
            **_hidden_probes("Geo::Inner"),
            **_hidden_probes("Base::Inner"),
            "routed": rejected("ex::Geo::Inner(y = 1)", HiddenMemberError, "ex::Geo::Inner"),
            "alias": accepted("Geo(x = 1)", _RECORD),
            "sibling": accepted("Geo::f", "() -> int"),
        },
    ),
    "an-export-hiding-a-function-path-hides-it-in-every-spelling": Scenario(
        modules=_export_hiding("Geo::f"),
        header=(_ROUTES,),
        probes={
            "target": rejected("Base::f", HiddenMemberError, "Base::f"),
            "alias": rejected("Geo::f", HiddenMemberError, "Geo::f"),
            "routed-target": rejected("ex::Base::f", HiddenMemberError, "ex::Base::f"),
            "routed-alias": rejected("ex::Geo::f", HiddenMemberError, "ex::Geo::f"),
            "sibling": accepted("Geo::Inner(y = 1)", _INNER),
        },
    ),
    **{
        f"an-export-hiding-{name}-removes-every-spelling-of-its-declaration": Scenario(
            modules=_export_hiding(item),
            header=(_ROUTES, "import base"),
            probes={
                "target-value": rejected("Base(x = 1)", AglScopeError, "Base"),
                "alias-value": rejected("Geo(x = 1)", AglScopeError, "Geo"),
                "alias-pattern": rejected(_BASE_PATTERN, NoVisibleConstructorError, "Geo(x)"),
                "alias-annot": rejected("fn(p: Geo) => p", AglTypeError, "Geo"),
                "beneath-the-target": rejected(
                    "Base::Inner(y = 1)", UnknownQualifierError, "Base::Inner"
                ),
                "beneath-the-alias": rejected("Geo::f", UnknownQualifierError, "Geo::f"),
                "routed-alias": rejected("ex::Geo(x = 1)", UnknownMemberError, "ex::Geo"),
                "beneath-it-through-another-alias": rejected("fn(p: Sh) => p", AglTypeError, "Sh"),
                "unrelated-alias": accepted("fn(p: Uh) => p", "base::Oops -> base::Oops"),
                "other-import": accepted("base::Base::Inner(y = 1)", _INNER),
            },
        )
        for name, item in (("a-target", "Base"), ("an-alias", "Geo"))
    },
    "a-cycle-reads-its-final-exports-through-aliases": Scenario(
        modules={"base": _BASE, "al": _CYCLING, "ex": _CYCLED},
        header=(f"{_ROUTES}\nimport al",),
        probes={
            **type_positions("through-item", "I", "base::Base::Inner"),
            "through-item-value": accepted("I(y = 1)", _INNER),
            "alias-of-a-through-item": accepted("J(y = 1)", _INNER),
            "routed-alias-of-a-through-item": accepted("al::J(y = 1)", _INNER),
            "signature-through-item": accepted("al::mk(I(y = 2))", "int"),
            "field-through-item": accepted("al::Holder(i = J(y = 1)).i.y", "int"),
            "beneath-a-through-item": accepted("S::Ci", "record base::Base::Shape::Ci"),
            "hidden-through-the-alias": rejected("Geo::f", HiddenMemberError, "Geo::f"),
            "hidden-at-the-target": rejected("Base::f", HiddenMemberError, "Base::f"),
            "other-import": accepted("al::Geo::f", "() -> int"),
        },
    ),
    "a-module-importing-itself-reads-its-final-exports-through-its-aliases": Scenario(
        modules={
            "base": _BASE,
            "me": (
                "import base\nimport me\nexport base::{Base}\nexport me::{Geo::Inner as I}\n"
                "type Geo = base::Base\ntype K = me::I\n"
            ),
        },
        header=("import me::*\nimport me",),
        probes={
            "through-item": accepted("I(y = 1)", _INNER),
            "alias-of-a-through-item": accepted("K(y = 1)", _INNER),
            "routed-alias-of-a-through-item": accepted("me::K(y = 1)", _INNER),
        },
    ),
    "an-alias-spelling-reads-as-its-target-in-every-position": Scenario(
        modules=_MODULES,
        header=("import al::*",),
        probes={
            "receiver": accepted("def Geo::m(self) -> int = self.x\nGeo(x = 2).m()", "int"),
            "receiver-like-its-target": accepted(
                "def Base::m(self) -> int = self.x\nBase(x = 2).m()", "int"
            ),
            "use-glob": accepted(_in_region("use Geo::*", "Inner(y = 1)"), _INNER),
            "is": accepted(
                "fn(p: Base::Shape) => p is Geo::Shape::Sq", "base::Base::Shape -> bool"
            ),
            "pattern": accepted(
                "fn(p: Base::Shape) => case p of\n  | Geo::Shape::Sq => 1\n  | _ => 2",
                "base::Base::Shape -> int",
            ),
            "catch": accepted("try\n  1\ncatch Uh as e =>\n  e.code", "int"),
            "extends": accepted(
                "exception Local extends Uh\nfn(p: Local) => p.code", "Local -> int"
            ),
            "bare-type-entry": rejected(
                "Sh", AglTypeError, "Sh", type_entry="enum base::Base::Shape\n  | Sq\n  | Ci"
            ),
            "generic": accepted("fn(p: Bx[int]) => p.v", "base::Box[int] -> int"),
            "generic-nested": accepted("Bx::In(w = 1)", "record base::Box::In\n  w: int"),
            "generic-nested-takes-no-arguments": rejected(
                "Bx[int]::In(w = 1)", TypeArgumentsError, "Bx[int]"
            ),
        },
    ),
}

_REFERENCING = "record Box::Item\n  n: int\nenum Box = Empty | Box::Item"
"""An enum referencing a record declared beneath the enum's own path."""


def _referenced_at_its_enum_path_probes(box: str) -> dict[str, Probe]:
    """``Box::Item`` through ``Box`` and its alias ``B``; *box* renders the enum's type."""
    return {
        f"{spelling}-{position}": probe
        for spelling in ("Box", "B")
        for position, probe in {
            "value": accepted(f"{spelling}::Item(n = 1)", f"record {box}::Item\n  n: int"),
            **type_positions("item", f"{spelling}::Item", f"{box}::Item"),
            "pattern": accepted(
                f"fn(p: Box) => case p of\n  | {spelling}::Item => 1\n  | _ => 2", f"{box} -> int"
            ),
            "is": accepted(f"fn(p: Box) => p is {spelling}::Item", f"{box} -> bool"),
        }.items()
    }


_SCENARIOS |= {
    f"an-alias-reaches-a-referenced-member-at-its-enum-path-{name}": Scenario(
        modules={"box": f"{_REFERENCING}\ntype B = Box\n"},
        header=header,
        probes=_referenced_at_its_enum_path_probes(box),
    )
    for name, header, box in (
        ("own", (_REFERENCING, "type B = Box"), "Box"),
        ("an-own-alias-of-an-imported-enum", ("import box::{Box}", "type B = Box"), "box::Box"),
        ("imported", ("import box::*",), "box::Box"),
    )
}


_GENERIC = (
    "record Box[T]\n  v: T\nrecord Box::In\n  w: int\ndef Box::g() -> int = 1\n"
    "def Box::m[T](self) -> int = 1\nenum Opt[T]\n  | Some(v: T)\n  | Non\n"
)
_APPLYING = (
    "import gen::*",
    "type IntBox = Box[int]",
    "type IB2 = IntBox",
    "type B2[T] = Box[T]",
    "type X = B2[int]",
    "type IntOpt = Opt[int]",
)
"""Applied aliases of ``Box[int]``: direct, through an alias, and through a generic alias."""
_INT_BOX = "record gen::Box[int]\n  v: int"


def _applied_probes(spelling: str) -> dict[str, Probe]:
    """*spelling* of ``Box[int]`` reads a path beneath it as ``Box[int]`` does."""
    return {
        f"{spelling}-{name}": probe
        for name, probe in {
            "own-constructor": accepted(f"{spelling}::Box(v = 1)", _INT_BOX),
            "own-constructor-argument-mismatch": rejected(
                f'{spelling}::Box(v = "s")', AglTypeError, '"s"', phase="typecheck"
            ),
            "own-constructor-pattern": accepted(
                f"case Box(v = 1) of\n  | {spelling}::Box(v) => v", "int"
            ),
            "own-constructor-pattern-on-another-application": rejected(
                f'case Box(v = "s") of\n  | {spelling}::Box(v) => 1',
                AglTypeError,
                f"{spelling}::Box(v)",
                phase="typecheck",
            ),
            "own-constructor-is-no-type": rejected(
                f"fn(p: {spelling}::Box) => p.v",
                AglTypeError,
                f"p: {spelling}::Box",
                phase="typecheck",
            ),
            "nested-record": rejected(f"{spelling}::In(w = 1)", TypeArgumentsError, spelling),
            "nested-record-annotation": rejected(
                f"fn(p: {spelling}::In) => 1", TypeArgumentsError, spelling
            ),
            "static-function": rejected(f"{spelling}::g()", TypeArgumentsError, spelling),
            "method": rejected(f"{spelling}::m", TypeArgumentsError, spelling),
        }.items()
    }


_SCENARIOS["an-applied-alias-reads-paths-as-its-applied-target"] = Scenario(
    modules={"gen": _GENERIC},
    header=_APPLYING,
    probes={
        **{
            key: probe
            for spelling in ("Box[int]", "IntBox", "IB2", "X")
            for key, probe in _applied_probes(spelling).items()
        },
        "own-alias-spelling": accepted("IntBox::IntBox(v = 1)", _INT_BOX),
        "generic-alias-nested-record": accepted("B2::In(w = 1)", "record gen::Box::In\n  w: int"),
        "generic-alias-static-function": accepted("B2::g()", "int"),
        "enum-member": accepted("IntOpt::Some(v = 1)", "record gen::Opt::Some[int]\n  v: int"),
        "enum-member-argument-mismatch": rejected(
            'IntOpt::Some(v = "s")', AglTypeError, '"s"', phase="typecheck"
        ),
        "enum-member-pattern": accepted(
            "fn(p: Opt[int]) => case p of\n  | IntOpt::Some(v) => v\n  | _ => 0",
            "gen::Opt[int] -> int",
        ),
    },
)


_DENOTING_A = (
    "import gen::*\ntype IntBox = Box[int]\ntype F = int -> bool\ntype P = Box[path]\n"
    "type IntOpt = Opt[int]\ntype Arr = array[int]\ntype G[T] = Box[array[T]]\n"
    "type Txt = text\ntype Pair = dict[text, int]\ntype Bx = Box[int]\ntype R = Box[int]\n"
)
_DENOTING_B = (
    "import gen::*\ntype B2[T] = Box[T]\ntype IntBox = B2[int]\ntype F = int -> bool\n"
    "type P = Box[text]\ntype IntOpt = Opt[int]\ntype Arr = array[int]\n"
    "type G[U] = Box[array[U]]\ntype Txt = path\ntype Pair = dict[text, bool]\n"
    "type Bx[T] = Box[T]\nrecord R\n  v: int\n"
)
"""Same-named aliases: most denote one type (through a generic alias, ``path`` = ``text``,
parameters renamed), ``Pair``, ``Bx`` and ``R`` distinct ones."""

_SCENARIOS["aliases-denoting-one-type-are-one-declaration"] = Scenario(
    modules={"gen": _GENERIC, "a": _DENOTING_A, "b": _DENOTING_B},
    header=("import gen::*", "import a::*", "import b::*"),
    probes={
        **type_positions("applied", "IntBox", "gen::Box[int]"),
        "applied-constructor": accepted("IntBox(v = 1)", _INT_BOX),
        "applied-pattern": accepted("case Box(v = 1) of\n  | IntBox(v) => v", "int"),
        "applied-own-constructor-path": accepted("IntBox::Box(v = 1)", _INT_BOX),
        "applied-alias-of-it": accepted(
            "type IB = IntBox\nfn(p: IB) => p.v", "gen::Box[int] -> int"
        ),
        "function": accepted("fn(f: F) => f(1)", "(int -> bool) -> bool"),
        "builtin-alias-argument": accepted("fn(p: P) => p.v", "gen::Box[text] -> text"),
        "enum": accepted("fn(p: IntOpt) => p", "gen::Opt[int] -> gen::Opt[int]"),
        "enum-member": accepted("IntOpt::Some(v = 1)", "record gen::Opt::Some[int]\n  v: int"),
        "array": accepted("fn(p: Arr) => p", "array[int] -> array[int]"),
        "generic": accepted("fn(p: G[int]) => p.v", "gen::Box[array[int]] -> array[int]"),
        "builtin-alias": accepted("fn(p: Txt) => p", "text -> text"),
        "distinct-arguments-clash": rejected(
            "fn(p: Pair) => p", AmbiguousQualificationError, "Pair"
        ),
        "applied-and-generic-clash": rejected("fn(p: Bx) => p", AmbiguousQualificationError, "Bx"),
        "alias-and-declaration-clash": rejected("fn(p: R) => p", AmbiguousQualificationError, "R"),
    },
)


_SCENARIOS["ill-founded-aliases-denote-no-shared-type"] = Scenario(
    modules={"c1": "type T = array[T]\n", "c2": "type T = array[T]\n"},
    header=("import c1::*", "import c2::*"),
    probes={"clash": rejected("fn(p: T) => p", AmbiguousQualificationError, "T")},
    # Either import alone is an entry whose cyclic alias typecheck rejects.
    legal=frozenset({(3,)}),
)


_DENOTED_TWICE = (
    "import gen::*\ntype F = int -> bool\ntype G = int -> bool\ntype IntBox = Box[int]\n"
    "type IB2 = Box[int]\ntype H = int -> text\n"
)
"""Aliases denoting one type twice, and one denoting another."""

_SCENARIOS |= {
    "hiding-an-alias-removes-every-alias-that-import-brings-denoting-its-type": Scenario(
        modules={"gen": _GENERIC, "hd": _DENOTED_TWICE},
        header=("import hd::* hiding F, IntBox",),
        probes={
            "hidden": rejected("fn(f: F) => f(1)", AglTypeError, "F"),
            "same-type": rejected("fn(f: G) => f(1)", AglTypeError, "G"),
            "same-type-routed": rejected("fn(f: hd::G) => f(1)", HiddenMemberError, "hd::G"),
            "same-applied-type": rejected("fn(p: IB2) => p.v", AglTypeError, "IB2"),
            "another-type": accepted("fn(f: H) => f(1)", "(int -> text) -> text"),
        },
    ),
    "hiding-an-alias-leaves-other-imports-aliases-of-its-type": Scenario(
        modules={"gen": _GENERIC, "hd": _DENOTED_TWICE, "hx": "type G = int -> bool\n"},
        header=("import hd::* hiding F\nimport hx::*",),
        probes={"other-import": accepted("fn(f: G) => f(1)", "(int -> bool) -> bool")},
    ),
    "a-use-or-export-hiding-an-alias-removes-aliases-denoting-its-type": Scenario(
        modules={"gen": _GENERIC, "hd": _DENOTED_TWICE, "ex": "import hd\nexport hd hiding F\n"},
        header=("import hd\nimport ex::*",),
        probes={
            "use": rejected(
                _in_region("use hd::* hiding F", "fn(f: G) => f(1)"), AglTypeError, "G"
            ),
            "export": rejected("fn(f: G) => f(1)", AglTypeError, "G"),
            "export-another-type": accepted("fn(f: H) => f(1)", "(int -> text) -> text"),
        },
    ),
}


_BUILTIN_TARGETS = "type U2 = text\ntype Brr[E] = array[E]\ntype IB = array[int]\n"
"""Aliases of built-in types, read through an import."""
_TEXT_SCOPE = (
    "def text::zz() -> int = 1\ndef text::yy() -> int = 2\n"
    "def text::sub::zz() -> int = 1\ndef text::sub::yy() -> int = 2\n"
)
_HIDING_SITE = "import tx::* hiding text::zz, text::sub::zz\ntype V2 = text\n"
"""An alias of ``text`` where a ``hiding`` removed ``text::zz`` and ``text::sub::zz``."""
_MAP = "fn(p: array[int]) => {q}(p, fn(x: int) => x + 1)"


def _builtin_alias_probes(text: str, array: str, applied: str) -> dict[str, Probe]:
    """Paths beneath aliases of ``text``, ``array[E]`` and ``array[int]``, read as theirs.

    *text*, *array* and *applied* spell the three aliases, or the targets themselves.
    """
    return {
        f"{text}-method-value": accepted(f"{text}::size", "text -> int"),
        f"{text}-method-call": accepted(f'{text}::size("ab")', "int"),
        f"{text}-unknown": rejected(f"{text}::nope", UnknownMemberError, f"{text}::nope"),
        f"{text}-use": accepted(_in_region(f"use {text}::*", 'upper("a")'), "text"),
        f"{array}-method": accepted(_MAP.format(q=f"{array}::map"), "array[int] -> array[int]"),
        f"{array}-uninferred": rejected(
            f"{array}::map", AglTypeError, f"{array}::map", phase="typecheck"
        ),
        f"{array}-applied": rejected(f"{array}[int]::map", TypeArgumentsError, f"{array}[int]"),
        f"{applied}-applied": rejected(f"{applied}::map", TypeArgumentsError, applied),
    }


_SCENARIOS["an-alias-of-a-builtin-type-reads-paths-as-its-target"] = Scenario(
    modules={"bi": _BUILTIN_TARGETS, "tx": _TEXT_SCOPE, "bj": _HIDING_SITE},
    header=(
        "import bi::*\nimport bj::*",
        "type T2 = text\ntype Arr[E] = array[E]\ntype IA = array[int]\ntype T3 = T2\n"
        "type A3 = Arr[int]",
    ),
    probes={
        **_builtin_alias_probes("text", "array", "array[int]"),
        **_builtin_alias_probes("T2", "Arr", "IA"),
        **_builtin_alias_probes("U2", "Brr", "IB"),
        "alias-of-an-alias": accepted('T3::size("ab")', "int"),
        "applied-alias-of-an-alias": rejected("A3::map", TypeArgumentsError, "A3"),
        "hidden-at-the-alias-site": rejected("V2::zz()", HiddenMemberError, "V2::zz"),
        "reached-at-the-alias-site": accepted("V2::yy()", "int"),
        "hidden-beneath-at-the-alias-site": rejected(
            "V2::sub::zz()", HiddenMemberError, "V2::sub::zz"
        ),
        "reached-beneath-at-the-alias-site": accepted("V2::sub::yy()", "int"),
        "info": info(
            "T2::size", "T2::size is a function.\nSignature:\n  def T2::size(self: text) -> int"
        ),
    },
)


class TestTransparentAliases:
    """Alias spellings in every position, file mode and every REPL grouping."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", file_params(_SCENARIOS))
    def test_a_file_resolves_like_the_inline_entry(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_file_resolves_like_inline_entry(tmp_path, scenario)


_UNRESOLVED = "import base\ntype Geo = Base\n"
"""Its alias target names nothing where it is declared: ``base`` is imported qualified."""


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("import al::*\n1", id="unused"),
        pytest.param("import al::*\nGeo::Inner(y = 1)", id="beneath-it"),
        pytest.param("import al::*\nGeo(x = 1)", id="constructed"),
        pytest.param("import al\nfn(p: al::Geo) => p", id="annotated"),
    ],
)
def test_an_unresolved_alias_target_is_an_error_where_the_alias_is_declared(
    tmp_path: Path, entry: str
) -> None:
    phase, error, span, _identity = file_verdict(
        tmp_path, {"entry": entry, "base": _BASE, "al": _UNRESOLVED}
    )

    assert (phase, error) == ("scope", AglTypeError)
    assert span is not None and span.source.label.endswith("al.agl")
    assert span_text(_UNRESOLVED, span) == "Base"


@pytest.mark.parametrize(
    "export",
    [
        pytest.param("import al\nexport al::{Geo::Nope}\n", id="item"),
        pytest.param("import al\nexport al hiding Geo::Nope\n", id="hiding"),
    ],
)
def test_an_export_item_through_an_alias_naming_nothing_is_an_error_where_written(
    tmp_path: Path, export: str
) -> None:
    phase, error, span, _identity = file_verdict(
        tmp_path, {"entry": "import ex\n1", "ex": export, **_MODULES}
    )

    assert (phase, error) == ("scope", UnknownMemberError)
    assert span is not None and span.source.label.endswith("ex.agl")
    assert span_text(export, span) == export.splitlines()[1]
