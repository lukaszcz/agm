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

Every probe is checked in the file part and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from collections.abc import Callable, Collection
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
    assert_scenario,
    info,
    inline_verdict,
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
_RECORD_RED = "record base::Color::Red"
_BASE_CASE = "fn(p: base::Base) => case p of\n    | Geo(x) => x"
_BASE_PATTERN = "fn(p: base::Base) => case p of\n  | Geo(x) => x"
"""Matches ``base``'s ``Base`` by ``Geo``."""
"""Matches ``base``'s ``Base`` by ``Geo`` in a region (see :func:`_in_region`)."""


def _in_region(use: str, value: str) -> str:
    """A region whose ``use`` exposes names that its ``v()`` reads, then ``r::v()``."""
    return f"scope r\n  {use}\n  def v() = {value}\nend r\n\nr::v()"


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
            "info": info(
                "I", "I is a constructor.\nSignature:\n  Base::Inner(y: int) -> base::Base::Inner"
            ),
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
        f"{hiding}-{name}-removes-every-spelling-of-its-declaration": Scenario(
            modules=_export_hiding(item),
            header=(header.format(item=item), "import base"),
            probes={
                "target-value": rejected("Base(x = 1)", AglScopeError, "Base"),
                "alias-value": rejected("Geo(x = 1)", AglScopeError, "Geo"),
                "alias-pattern": rejected(_BASE_PATTERN, NoVisibleConstructorError, "Geo(x)"),
                "alias-annot": rejected("fn(p: Geo) => p", AglTypeError, "Geo"),
                "beneath-the-target": rejected(
                    "Base::Inner(y = 1)", HiddenMemberError, "Base::Inner"
                ),
                "beneath-the-alias": rejected("Geo::f", HiddenMemberError, "Geo::f"),
                "routed-alias": rejected(
                    f"{route}::Geo(x = 1)", HiddenMemberError, f"{route}::Geo"
                ),
                "routed-beneath-the-alias": rejected(
                    f"{route}::Geo::f", HiddenMemberError, f"{route}::Geo::f"
                ),
                "beneath-it-through-another-alias": rejected("fn(p: Sh) => p", AglTypeError, "Sh"),
                "unrelated-alias": accepted("fn(p: Uh) => p", "base::Oops -> base::Oops"),
                "other-import": accepted("base::Base::Inner(y = 1)", _INNER),
            },
        )
        for name, item in (("a-target", "Base"), ("an-alias", "Geo"))
        for hiding, route, header in (
            ("an-export-hiding", "ex", _ROUTES),
            ("an-import-hiding", "al", "import al::* hiding {item}\nimport al hiding {item}"),
        )
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
    **{
        f"a-cycle-names-its-final-exports-in-an-import-{name}": Scenario(
            modules={
                "base": _BASE,
                "ex": _CYCLED,
                "al": (
                    f"import base\nimport ex\n{imports}\nexport base::{{Base, Color}}\n"
                    "type Geo = base::Base\ndef mk(p: ex::I) -> int = p.y\n"
                ),
            },
            header=("import al\nimport ex",),
            probes={"signature-through-item": accepted("al::mk(ex::I(y = 2))", "int")},
        )
        for name, imports in (("tail", "import ex::{I}"), ("hiding", "import ex::* hiding I"))
    },
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

_HUE = "enum Color\n  | Red\n  | Blue\n"
"""An enum named like ``base``'s ``Color``, sharing its member ``Red``."""
_HUE_RED = "record hue::Color::Red"

_SCENARIOS["an-own-alias-wins-the-paths-beneath-it-over-an-import"] = Scenario(
    modules={**_MODULES, "hue": _HUE},
    header=("import base::*", "import hue", "type Color = hue::Color"),
    probes={
        **type_positions("alias", "Color", "hue::Color"),
        "member": accepted("Color::Red", _HUE_RED),
        "member-only-the-target-declares": accepted("Color::Blue", "record hue::Color::Blue"),
        "pattern": accepted(
            "fn(p: hue::Color) => case p of\n  | Color::Red => 1\n  | _ => 2",
            "hue::Color -> int",
        ),
        "is": accepted("fn(p: hue::Color) => p is Color::Red", "hue::Color -> bool"),
        **type_positions("member-type", "Color::Red", "hue::Color::Red"),
    },
)

_SCOPED_HUE = "scope S\n  enum Color\n    | Red\n    | Green\n\n  type C = Color\nend S\n"
"""An enum in a scope, beside an alias of it."""
_SCOPED_RED = "record sh::S::Color::Red"

_SCENARIOS["an-item-naming-an-alias-of-an-enum-injects-its-members"] = Scenario(
    modules={**_MODULES, "sh": _SCOPED_HUE},
    header=(),
    probes={
        "tail": accepted("import al::{Hue}\nRed", _RECORD_RED),
        "tail-pattern": accepted(
            "import al::{Hue}\nfn(p: Hue) => case p of\n  | Red => 1\n  | Green => 2",
            "base::Color -> int",
        ),
        "tail-in-a-region": accepted(_in_region("import al::{Hue}", "Red"), _RECORD_RED),
        "use-tail": accepted(f"import al\n{_in_region('use al::{Hue}', 'Red')}", _RECORD_RED),
        "use-rename": accepted(f"import al\n{_in_region('use al::Hue as K', 'Red')}", _RECORD_RED),
        "scoped-tail": accepted(
            "import sh::{S::C}\n\nscope S\n  def v() = Red\nend S\n\nS::v()", _SCOPED_RED
        ),
        "scoped-use-tail": accepted(
            f"import sh\n{_in_region('use sh::S::{C}', 'Red')}", _SCOPED_RED
        ),
        "scoped-use-rename": accepted(
            f"import sh\n{_in_region('use sh::S::C as K', 'Red')}", _SCOPED_RED
        ),
    },
)

_REFERENCING_A = "scope R\n  record A\n    v: int\nend R\n\nenum E\n  | R::A\n  | B\n\ntype C = E\n"
"""An enum referencing a record declared elsewhere, beside an alias of it."""
_RECORD_A = "record rf::R::A\n  v: int"

_SCENARIOS["an-item-naming-an-alias-of-an-enum-injects-its-referenced-members"] = Scenario(
    modules={"rf": _REFERENCING_A},
    header=(),
    probes={
        "tail": accepted("import rf::{C}\nA(v = 1)", _RECORD_A),
        "tail-in-a-region": accepted(_in_region("import rf::{C}", "A(v = 1)"), _RECORD_A),
        "use-tail": accepted(f"import rf\n{_in_region('use rf::{C}', 'A(v = 1)')}", _RECORD_A),
        "use-rename": accepted(f"import rf\n{_in_region('use rf::C as K', 'A(v = 1)')}", _RECORD_A),
        "use-whole": accepted(f"import rf\n{_in_region('use rf::C', 'A(v = 1)')}", _RECORD_A),
    },
)

_DECLARING_BENEATH = (
    "import base\nimport base::*\nexport base::{Base}\ntype Geo = base::Base\n"
    "def Base::h() -> int = 3\ndef Geo::k() -> int = 4\n"
    "type T2 = text\ntype U = text\ndef text::lt() -> int = 1\ndef T2::lu() -> int = 2\n"
)
"""Declares beneath ``base``'s ``Base`` and beneath ``text``, directly and through aliases."""


def _beneath_hidden_alias_probes(route: str) -> dict[str, Probe]:
    """What ``hiding Geo, T2`` removes beneath their targets: hidden under every spelling.

    *route* is the module route the removed paths are also spelled through.
    """
    return {
        **{
            f"{name}{routed}": rejected(f"{qualifier}{path}()", HiddenMemberError, qualifier + path)
            for routed, qualifier in (("", ""), ("-routed", f"{route}::"))
            for name, path in {
                "direct": "Base::h",
                "through-the-alias": "Base::k",
                "by-the-alias": "Geo::k",
                "beneath-a-builtin": "text::lt",
                "beneath-a-builtin-through-the-alias": "text::lu",
                "beneath-a-builtin-by-the-alias": "T2::lu",
            }.items()
        },
        "target": rejected("fn(x: Base) => x", AglTypeError, "Base"),
        "builtin-alias": rejected("fn(x: U) => x", AglTypeError, "U"),
        "builtin-kept": accepted('text::size("ab")', "int"),
    }


_SCENARIOS |= {
    "hiding-an-alias-removes-what-its-module-declares-beneath-the-target": Scenario(
        modules={"base": _BASE, "ad": _DECLARING_BENEATH},
        header=("import ad::* hiding Geo, T2\nimport ad hiding Geo, T2",),
        probes=_beneath_hidden_alias_probes("ad"),
    ),
    "an-export-hiding-an-alias-removes-what-its-module-declares-beneath-the-target": Scenario(
        modules={
            "base": _BASE,
            "ad": _DECLARING_BENEATH,
            "ex": "import ad\nexport ad hiding Geo, T2\n",
        },
        header=(_ROUTES,),
        probes=_beneath_hidden_alias_probes("ex"),
    ),
}


_PLACED = {"G": ("Base", "h"), "T2": ("text", "s"), "Arr": ("array", "z")}
"""Each alias ``pl`` declares beneath, with its target and the member the items name."""
_ITEMS = {
    "an-alias": ", ".join(f"{alias}::{member}" for alias, (_, member) in _PLACED.items()),
    "the-declared-path": ", ".join(f"{target}::{member}" for target, member in _PLACED.values()),
}
"""The members ``pl`` declares beneath its aliases' targets, spelled through each."""


def _placing(target: str) -> str:
    """Module ``pl``: aliases of ``base``'s ``Base`` (spelled *target*), ``text`` and ``array``.

    It declares ``j`` and the member of :data:`_PLACED` beneath each, through the alias.
    """
    aliases = {"G": target, "T2": "text", "Arr[E]": "array[E]"}
    return "import base\nimport base::*\n" + "".join(
        f"type {alias} = {aliased}\n"
        f"def {name}::{_PLACED[name][1]}() -> int = 2\ndef {name}::j() -> int = 4\n"
        for alias, aliased in aliases.items()
        for name in (alias.partition("[")[0],)
    )


_PLACING = {"a-routed-target": _placing("base::Base"), "a-bare-target": _placing("Base")}


def _placed_probes(verdict: Callable[[str], Probe], spellings: Collection[str]) -> dict[str, Probe]:
    """*verdict* of each member of :data:`_PLACED`, through each of *spellings* of its path.

    A spelling is ``alias`` or ``target``, bare or after a route; the sibling
    ``j`` stays reached through the alias.
    """
    siblings = {f"{alias}-sibling": accepted(f"{alias}::j()", "int") for alias in _PLACED}
    return siblings | {
        path: verdict(path)
        for alias, (target, member) in _PLACED.items()
        for spelling in spellings
        for path in (f"{spelling.format(alias=alias, target=target)}::{member}",)
    }


def _hidden_call(path: str) -> Probe:
    return rejected(f"{path}()", HiddenMemberError, path)


def _accepted_call(path: str) -> Probe:
    return accepted(f"{path}()", "int")


_SCENARIOS |= {
    f"an-import-tail-through-{spelled}-names-what-its-module-declares-beneath-{target}": Scenario(
        modules={"base": _BASE, "pl": placing},
        header=(f"import pl::{{{items}, G::j, T2::j, Arr::j}}",),
        probes=_placed_probes(_accepted_call, (written,)),
    )
    for target, placing in _PLACING.items()
    for (spelled, items), written in zip(_ITEMS.items(), ("{alias}", "{target}"), strict=True)
}

_SCENARIOS |= {
    f"an-export-item-through-{spelled}-names-what-its-module-declares-beneath-{target}": Scenario(
        modules={
            "base": _BASE,
            "pl": placing,
            "ex": f"import pl\nexport pl::{{{items}, G::j, T2::j, Arr::j}}\n",
        },
        header=(_ROUTES,),
        probes=_placed_probes(_accepted_call, (written, f"ex::{written}")),
    )
    for target, placing in _PLACING.items()
    for (spelled, items), written in zip(_ITEMS.items(), ("{alias}", "{target}"), strict=True)
}

_SCENARIOS |= {
    f"hiding-through-{spelled}-removes-what-its-module-declares-beneath-{target}": Scenario(
        modules={"base": _BASE, "pl": placing},
        header=(f"import pl::* hiding {items}",),
        probes=_placed_probes(_hidden_call, ("{alias}", "{target}", "pl::{alias}", "pl::{target}")),
    )
    for target, placing in _PLACING.items()
    for spelled, items in _ITEMS.items()
}

_SCENARIOS |= {
    f"an-export-hiding-through-{spelled}-removes-what-its-module-declares-beneath-{target}": (
        Scenario(
            modules={"base": _BASE, "pl": placing, "ex": f"import pl\nexport pl hiding {items}\n"},
            header=(_ROUTES,),
            probes=_placed_probes(
                _hidden_call, ("{alias}", "{target}", "ex::{alias}", "ex::{target}")
            ),
        )
    )
    for target, placing in _PLACING.items()
    for spelled, items in _ITEMS.items()
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


_REFERENCING_THROUGH_A_USE = (
    "use Box as B\nrecord Box::Item\n  n: int\nenum Box = Empty | B::Item\n"
    "def f() -> int = case B::Item(n = 3) of\n  | B::Item(n) => n"
)
"""An enum whose member reference reads the enum's own path through a use renaming it."""


_SCENARIOS |= {
    "a-member-reference-through-a-use-of-its-own-enum-reads-the-whole-enum": Scenario(
        header=(_REFERENCING_THROUGH_A_USE,),
        probes={
            "value": accepted("B::Item(n = 1)", "record Box::Item\n  n: int"),
            "member": accepted("B::Empty", "record Box::Empty"),
            "injected": accepted("Item(n = 1)", "record Box::Item\n  n: int"),
            "pattern": accepted(
                "fn(p: Box) => case p of\n  | B::Item(n) => n\n  | Empty => 0", "Box -> int"
            ),
            "is": accepted("fn(p: Box) => p is B::Item", "Box -> bool"),
            "function": accepted("f()", "int"),
        },
    ),
    "a-member-reference-through-a-use-of-its-own-enum-reads-the-whole-imported-enum": Scenario(
        modules={"box": _REFERENCING_THROUGH_A_USE},
        header=("import box::*",),
        probes={
            "value": accepted("Box::Item(n = 1)", "record box::Box::Item\n  n: int"),
            "injected": accepted("Item(n = 1)", "record box::Box::Item\n  n: int"),
            "pattern": accepted(
                "fn(p: Box) => case p of\n  | Item(n) => n\n  | Empty => 0", "box::Box -> int"
            ),
            "function": accepted("f()", "int"),
        },
    ),
}


_GENERIC = (
    "record Box[T]\n  v: T\nrecord Box::In\n  w: int\ndef Box::g() -> int = 1\n"
    "def Box::m[T](self) -> int = 1\nenum Opt[T]\n  | Some(v: T)\n  | Non\n"
)
_IMPORT_GEN = "import gen::*"
_INT_BOX_ALIAS = "type IntBox = Box[int]"
_GENERIC_ALIAS = "type B2[T] = Box[T]"
_APPLYING: dict[str, tuple[str, ...]] = {
    "Box[int]": (_IMPORT_GEN,),
    "IntBox": (_IMPORT_GEN, _INT_BOX_ALIAS),
    "IB2": (_IMPORT_GEN, _INT_BOX_ALIAS, "type IB2 = IntBox"),
    "X": (_IMPORT_GEN, _GENERIC_ALIAS, "type X = B2[int]"),
    "Y": (_IMPORT_GEN, "type Id[T] = T", "type Y = Id[Box[int]]"),
}
"""``Box[int]`` and its applied aliases -- direct, through an alias, through a
generic alias, and through one standing for its parameter -- each with the
declarations its spelling needs."""
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
            "missing-beneath-nested-record": rejected(
                f"{spelling}::In::x()", UnknownMemberError, f"{spelling}::In::x"
            ),
            "missing-beneath-nested-record-annotation": rejected(
                f"fn(p: {spelling}::In::x) => 1", UnknownMemberError, f"{spelling}::In::x"
            ),
            "static-function": rejected(f"{spelling}::g()", TypeArgumentsError, spelling),
            "method": rejected(f"{spelling}::m", TypeArgumentsError, spelling),
        }.items()
    }


_APPLIED_EXTRAS: dict[str, dict[str, Probe]] = {
    "IntBox": {"own-alias-spelling": accepted("IntBox::IntBox(v = 1)", _INT_BOX)},
    "X": {
        "generic-alias-nested-record": accepted("B2::In(w = 1)", "record gen::Box::In\n  w: int"),
        "generic-alias-static-function": accepted("B2::g()", "int"),
    },
}
"""Paths beneath the alias declarations a spelling's header declares."""

_SCENARIOS.update(
    {
        f"an-applied-alias-reads-paths-as-its-applied-target-{spelling}": Scenario(
            modules={"gen": _GENERIC},
            header=header,
            probes={**_applied_probes(spelling), **_APPLIED_EXTRAS.get(spelling, {})},
        )
        for spelling, header in _APPLYING.items()
    }
)

_SCENARIOS["an-applied-enum-alias-reads-members-as-its-applied-target"] = Scenario(
    modules={"gen": _GENERIC},
    header=(_IMPORT_GEN, "type IntOpt = Opt[int]"),
    probes={
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

_OPT = "enum Opt[T]\n  | Som(v: T)\n  | Non\n"
_APPLIED_ENUM = "import op\ntype IntOpt = op::Opt[int]\n"
"""Declares an alias applying ``op``'s generic enum."""
_INT_SOM = "record op::Opt::Som[int]\n  v: int"
_BAD_SOM = 'Som(v = "s")'


def _applied_members_probes(prefix: str, before: str, spelled: str = "IntOpt") -> dict[str, Probe]:
    """Bare members of ``IntOpt``, spelled *spelled*, read as ``IntOpt::<member>``.

    *before* injects them.
    """
    return {
        f"{prefix}-{name}": probe
        for name, probe in {
            "member": accepted(f"{before}\nSom(v = 1)", _INT_SOM),
            "member-fixes-its-arguments": rejected(
                f"{before}\n{_BAD_SOM}", AglTypeError, '"s"', phase="typecheck"
            ),
            "nullary-member": accepted(
                f"{before}\nfn(p: {spelled}) => p is Non", "op::Opt[int] -> bool"
            ),
            "pattern": accepted(
                f"{before}\nfn(p: {spelled}) => case p of\n  | Som(v) => v\n  | Non => 0",
                "op::Opt[int] -> int",
            ),
        }.items()
    }


_SCENARIOS["an-item-naming-an-applied-enum-alias-injects-its-members-applied"] = Scenario(
    modules={"op": _OPT, "ia": _APPLIED_ENUM},
    header=(),
    probes={
        **_applied_members_probes("tail", "import ia::{IntOpt}"),
        **_applied_members_probes("use-tail", "import ia\nuse ia::{IntOpt}"),
        **_applied_members_probes("use-wildcard", "import ia\nuse ia::IntOpt::*", "ia::IntOpt"),
        "use-rename": rejected(
            f"import ia\n{_in_region('use ia::IntOpt as K', _BAD_SOM)}",
            AglTypeError,
            '"s"',
            phase="typecheck",
        ),
        "own-use": rejected(
            f"import op\ntype IntOpt = op::Opt[int]\n{_in_region('use IntOpt::*', _BAD_SOM)}",
            AglTypeError,
            '"s"',
            phase="typecheck",
        ),
        "hidden-member": rejected(
            "import ia::* hiding IntOpt::Som\nSom(v = 1)", AglScopeError, "Som"
        ),
        "beside-a-hidden-member": accepted(
            "import ia::* hiding IntOpt::Som\nfn(p: IntOpt) => p is Non", "op::Opt[int] -> bool"
        ),
        **{
            f"beside-its-generic-enum-{name}": rejected(
                f"import op::*\nimport ia::*\n{value}", AmbiguousConstructorError, spelled
            )
            for name, value, spelled in (
                ("member", "Som(v = 1)", "Som"),
                ("member-at-other-arguments", _BAD_SOM, "Som"),
                ("nullary-member", "Non", "Non"),
            )
        },
        **{
            f"beside-its-generic-enum-pattern-on-{scrutinee}": accepted(
                f"import op::*\nimport ia::*\nfn(p: {scrutinee}) => case p of\n"
                f"  | Som(v) => v\n  | Non => {default}",
                f"{selected} -> {result}",
            )
            for scrutinee, selected, result, default in (
                ("IntOpt", "op::Opt[int]", "int", "0"),
                ("Opt[text]", "op::Opt[text]", "text", '""'),
            )
        },
        "beside-its-generic-enum-pattern-on-its-member": accepted(
            "import op::*\nimport ia::*\nfn(p: Opt::Som[int]) => case p of\n  | Som(v) => v",
            "op::Opt::Som[int] -> int",
        ),
        "beside-its-generic-enum-is": accepted(
            "import op::*\nimport ia::*\nfn(p: IntOpt) => p is Som", "op::Opt[int] -> bool"
        ),
    },
)

_TEXT_OPT = "import op\ntype TextOpt = op::Opt[text]\n"


def _distinct_members_probes(prefix: str, before: str, text_opt: str) -> dict[str, Probe]:
    """Bare members two aliases applying ``Opt`` at distinct arguments inject by *before*.

    *text_opt* spells the alias applying it to ``text``.
    """
    return {
        f"{prefix}-{name}": probe
        for name, probe in {
            "member": rejected(f"{before}\nSom(v = 1)", AmbiguousConstructorError, "Som"),
            "member-at-other-arguments": rejected(
                f"{before}\n{_BAD_SOM}", AmbiguousConstructorError, "Som"
            ),
            "nullary-member": rejected(f"{before}\nNon", AmbiguousConstructorError, "Non"),
            "pattern": accepted(
                f'{before}\nfn(p: {text_opt}) => case p of\n  | Som(v) => v\n  | Non => ""',
                "op::Opt[text] -> text",
            ),
            "is": accepted(f"{before}\nfn(p: {text_opt}) => p is Som", "op::Opt[text] -> bool"),
            "qualified": accepted(
                f"{before}\n{text_opt}::{_BAD_SOM}", "record op::Opt::Som[text]\n  v: text"
            ),
        }.items()
    }


_SCENARIOS["applied-enum-aliases-inject-their-members-at-their-own-arguments"] = Scenario(
    modules={
        "op": _OPT,
        "ia": _APPLIED_ENUM,
        "it": _TEXT_OPT,
        "at": _TEXT_OPT,
        "twin": "import op\ntype OtherIntOpt = op::Opt[int]\n",
        "ren": "import op\ntype Ren[T] = op::Opt[T]\n",
    },
    header=(),
    probes={
        **_distinct_members_probes("items", "import ia::{IntOpt}\nimport it::{TextOpt}", "TextOpt"),
        **_distinct_members_probes(
            "items-whatever-the-module-names",
            "import ia::{IntOpt}\nimport at::{TextOpt}",
            "TextOpt",
        ),
        **_distinct_members_probes(
            "uses",
            "import ia\nimport it\nuse ia::IntOpt::*\nuse it::TextOpt::*",
            "it::TextOpt",
        ),
        **{
            f"beside-its-generic-enum-used-{name}": rejected(
                f"import op\nimport ia\nuse op::Opt::*\nuse ia::IntOpt::*\n{value}",
                AmbiguousConstructorError,
                "Som",
            )
            for name, value in (("member", "Som(v = 1)"), ("at-other-arguments", _BAD_SOM))
        },
        "at-the-same-arguments": accepted(
            "import ia::{IntOpt}\nimport twin::{OtherIntOpt}\nSom(v = 1)", _INT_SOM
        ),
        "at-the-same-arguments-fixes-them": rejected(
            f"import ia::{{IntOpt}}\nimport twin::{{OtherIntOpt}}\n{_BAD_SOM}",
            AglTypeError,
            '"s"',
            phase="typecheck",
        ),
        **{
            f"compatible-with-neither-{name}": rejected(
                f"import ia::{{IntOpt}}\nimport it::{{TextOpt}}\nimport op\n"
                f"fn(p: op::Opt[bool]) => {text}",
                AglTypeError,
                spelling,
                phase="typecheck",
            )
            for name, text, spelling in (
                ("pattern", "case p of\n  | Som(v) => v\n  | Non => false", "Som(v)"),
                ("is", "p is Som", "p is Som"),
            )
        },
        "renaming-beside-its-generic-enum": accepted(
            f"import op::*\nimport ren::{{Ren}}\n{_BAD_SOM}", "record op::Opt::Som[text]\n  v: text"
        ),
    },
)

_SCENARIOS["an-alias-standing-for-its-parameter-applied-denotes-its-argument"] = Scenario(
    modules={"gen": _GENERIC},
    header=(
        _IMPORT_GEN,
        "type Id[T] = T\ntype Id2[T] = Id[T]",
        "type IO = Id[Opt[int]]\ntype IO2 = Id2[Id[Opt[int]]]",
    ),
    probes={
        "enum-member": accepted("IO::Some(v = 1)", "record gen::Opt::Some[int]\n  v: int"),
        "enum-member-pattern": accepted(
            "fn(p: IO) => case p of\n  | IO::Some(v) => v\n  | _ => 0", "gen::Opt[int] -> int"
        ),
        "through-a-chain": accepted("IO2::Some(v = 1)", "record gen::Opt::Some[int]\n  v: int"),
        "argument-mismatch": rejected('IO2::Some(v = "s")', AglTypeError, '"s"', phase="typecheck"),
        "applied-in-place": accepted(
            "Id[Opt[int]]::Some(v = 1)", "record gen::Opt::Some[int]\n  v: int"
        ),
        "applied-in-place-pattern": accepted(
            "fn(p: IO) => case p of\n  | Id2[Opt[int]]::Some(v) => v\n  | _ => 0",
            "gen::Opt[int] -> int",
        ),
        "applied-in-place-argument-mismatch": rejected(
            'Id[Opt[int]]::Some(v = "s")', AglTypeError, '"s"', phase="typecheck"
        ),
        "applied-in-place-to-an-alias": accepted(
            "Id[IO]::Some(v = 1)", "record gen::Opt::Some[int]\n  v: int"
        ),
        "applied-in-place-record": accepted("Id[Box[int]]::Box(v = 1)", _INT_BOX),
        "applied-in-place-nested-record": rejected(
            "Id[Box[int]]::In(w = 1)", TypeArgumentsError, "Id[Box[int]]"
        ),
        "applied-in-place-missing-beneath-nested-record": rejected(
            "Id[Box[int]]::In::x()", UnknownMemberError, "Id[Box[int]]::In::x"
        ),
        "applied-in-place-arity": rejected(
            "Id[Opt[int], int]::Some(v = 1)", TypeArgumentsError, "Id[Opt[int], int]"
        ),
        "unapplied": rejected("Id::Some(v = 1)", UnknownMemberError, "Id::Some"),
    },
)

_SCENARIOS |= {
    name: Scenario(
        modules={
            "base": _BASE,
            "pj": f"import base::*\nexport base::{{Base}}\ntype Id[T] = {stands}\n"
            f"type PB = Id[{argument}]\n",
        },
        header=(
            "import base::*\nimport pj",
            f"type Id[T] = {stands}\ntype Id2[T] = Id[T]",
            f"type IB = Id[{argument}]\ntype J = Id2[Id[{argument}]]",
        ),
        probes={
            **{
                f"{spelling}-{position}": probe
                for spelling in ("IB", "J", "pj::PB")
                for position, probe in {
                    "static": accepted(f"{spelling}::f()", "int"),
                    "nested-value": accepted(f"{spelling}::Inner(y = 1)", _INNER),
                    "enum-member": accepted(
                        f"{spelling}::Shape::Sq", "record base::Base::Shape::Sq"
                    ),
                    "value": accepted(f"{spelling}(x = 1)", "record base::Base\n  x: int"),
                    "unknown": rejected(
                        f"{spelling}::nope()", UnknownMemberError, f"{spelling}::nope"
                    ),
                    **type_positions("alias", spelling, "base::Base"),
                    **type_positions("nested", f"{spelling}::Inner", "base::Base::Inner"),
                }.items()
            },
            "declared-later": accepted("def Base::late() -> int = 5\nIB::late()", "int"),
            "one-declaration": accepted(
                "fn(a: IB, b: J) => [a, b, Base(x = 1)]",
                "(base::Base, base::Base) -> array[base::Base]",
            ),
            "written-in-place": rejected(
                f"Id[{argument}]::f()", TypeArgumentsError, f"Id[{argument}]"
            ),
        },
    )
    for name, stands, argument in (
        (
            "an-alias-of-an-alias-standing-for-its-parameter-is-another-name-for-its-argument",
            "T",
            "Base",
        ),
        (
            "an-alias-applying-a-constant-alias-is-another-name-for-what-that-stands-for",
            "Base",
            "int",
        ),
    )
}


_SCENARIOS["an-alias-applying-a-constant-alias-reaches-what-that-alias-does"] = Scenario(
    modules={
        "base": _BASE,
        "hp": "import base::* hiding Base::f\ntype Kc[T] = Base\n",
        "mid": "import hp::{Kc}\ntype Geo = Kc[int]\n",
    },
    header=("import hp::*\nimport mid::*",),
    probes={
        "applied-alias-hidden": rejected("Kc::f()", HiddenMemberError, "Kc::f"),
        "hidden": rejected("Geo::f()", HiddenMemberError, "Geo::f"),
        "nested": accepted("Geo::Inner(y = 1)", _INNER),
        "value": accepted("Geo(x = 1)", "record base::Base\n  x: int"),
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
        "applied-and-generic-one-pattern": accepted(
            "fn(p: Box[int]) => case p of\n  | Bx(v) => v", "gen::Box[int] -> int"
        ),
        "alias-and-declaration-clash": rejected("fn(p: R) => p", AmbiguousQualificationError, "R"),
    },
)

_NORMALIZING = (
    "record Box[T]\n  v: T\ndef Box::g() -> int = 1\nenum Opt[T]\n  | Som(v: T)\n  | Non\n"
    "type Id[T] = T\n"
)
"""Generic types, and an alias standing for its parameter the aliases below apply in arguments."""
_SOM = "record nm::Opt::Som[int]\n  v: int"

_SCENARIOS["applied-arguments-are-normalized"] = Scenario(
    modules={
        "nm": _NORMALIZING,
        "a": "import nm::*\ntype B = Box[int]\ntype O = Opt[int]\n",
        "f": "import nm::*\ntype B = Box[Id[int]]\ntype O = Opt[Id[int]]\n",
    },
    header=("import nm", "import a::{B, O}\nimport f::{B, O}\nimport a\nimport f"),
    probes={
        "one-record": accepted("B(v = 1)", "record nm::Box[int]\n  v: int"),
        "one-type": accepted("(fn(b: a::B) => b.v)(f::B(v = 1))", "int"),
        "one-member": accepted("Som(v = 1)", _SOM),
        "one-member-test": accepted("fn(o: a::O) => o is f::O::Som", "nm::Opt[int] -> bool"),
        "one-member-pattern": accepted(
            "fn(o: nm::Opt[int]) => case o of\n  | Som(v) => v\n  | Non => 0",
            "nm::Opt[int] -> int",
        ),
    },
)

_SCENARIOS["a-generic-alias-normalizing-its-arguments-renames-its-target"] = Scenario(
    modules={
        "nm": _NORMALIZING,
        "g3": "import nm::*\ntype B3[T] = Box[Id[T]]\ntype O3[T] = Opt[Id[T]]\n",
    },
    header=("import nm::*", "import g3::*"),
    probes={
        "reads-the-target": accepted("B3::g()", "int"),
        "declares-at-the-target": accepted("def B3::k() -> int = 1\nBox::k()", "int"),
        "one-type": accepted("(fn(b: B3[int]) => b.v)(Box(v = 1))", "int"),
        "enum-member": accepted("O3::Som(v = 1)", _SOM),
        "bare-enum-member": accepted("Som(v = 1)", _SOM),
    },
)


_EXPANDING: dict[str, tuple[tuple[str, ...], str]] = {
    "self": (("type A = array[A]",), "A"),
    "function": (("type A = int -> A",), "A"),
    "applied": (("type L[T] = array[L[T]]",), "L[int]"),
    "nominal-argument": (("record Box[T]\n  v: T", "type A = Box[A]"), "A"),
    "through-an-alias": (("type A = array[B]", "type B = A"), "B"),
    "first-declared-of-the-cycle": (("type B = A", "type A = array[B]"), "A"),
    "read-through-a-record-first": (("record W\n  v: B", "type C = B", "type B = array[C]"), "W"),
}
"""Aliases whose targets expand to themselves through a structural type, and a
spelling reading them; the cycle is reported at its alias declared first."""

_SCENARIOS["an-alias-expanding-to-itself-is-rejected-where-declared"] = Scenario(
    header=(),
    probes={
        f"{name}-{use}": rejected(
            "\n".join((*declarations, text)),
            AglTypeError,
            next(d for d in declarations if d.startswith("type")),
            phase="typecheck",
        )
        for name, (declarations, spelling) in _EXPANDING.items()
        for use, text in (("unread", "1"), ("read", f"fn(p: {spelling}) => p"))
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
    "hiding-a-generic-type-leaves-an-alias-of-its-application": Scenario(
        modules={"bx": "record Box[T]\n  v: T\ntype IntBox = Box[int]\n"},
        header=("import bx::* hiding Box",),
        probes={
            "hidden": rejected("fn(p: Box[int]) => p.v", AglTypeError, "Box[int]"),
            "kept": accepted("fn(p: IntBox) => p.v", "bx::Box[int] -> int"),
            "kept-constructor": accepted("IntBox(v = 1)", "record bx::Box[int]\n  v: int"),
        },
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
            f"let _ = {array}::map", AglTypeError, f"{array}::map", phase="typecheck"
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
            "T2::size", "T2::size is a function.\nSignature:\n  def text::size(self: text) -> int"
        ),
    },
)

_SCENARIOS["an-item-through-an-alias-names-what-its-site-reaches-beneath-the-target"] = Scenario(
    modules={
        "tx": _TEXT_SCOPE,
        "bj": _HIDING_SITE,
        "ex": "import bj\nexport bj::{V2::yy}\n",
        "eh": "import bj\nexport bj hiding V2::yy\n",
    },
    header=(),
    probes={
        "tail": accepted("import bj::{V2::yy, V2::sub::yy}\nV2::yy() + V2::sub::yy()", "int"),
        "tail-hidden-at-the-alias-site": rejected(
            "import bj::{V2::zz}\n1", UnknownMemberError, "import bj::{V2::zz}"
        ),
        "hiding": rejected("import bj::* hiding V2::yy\nV2::yy()", HiddenMemberError, "V2::yy"),
        "hiding-keeps-another": accepted("import bj::* hiding V2::yy\nV2::sub::yy()", "int"),
        "hiding-hidden-at-the-alias-site": rejected(
            "import bj::* hiding V2::zz\n1", UnknownMemberError, "import bj::* hiding V2::zz"
        ),
        "export-item": accepted("import ex::*\nV2::yy()", "int"),
        "export-item-routed": accepted("import ex\nex::V2::yy()", "int"),
        "export-hiding": rejected("import eh::*\nV2::yy()", HiddenMemberError, "V2::yy"),
        "export-hiding-keeps-another": accepted("import eh::*\nV2::sub::yy()", "int"),
    },
)

_PLAIN = (
    "record Base\n  x: int\n\ndef Base::f() -> int = 1\n\n"
    "scope Base::Sub\n  def t() -> int = 2\nend Base::Sub\n"
)
_ALIASING = "import base\ntype Geo = base::Base\ntype U2 = text\n"
_BENEATH = (
    "import base::*\n\ndef Base::k() -> int = 1\n\nrecord Base::In2\n  a: int\n\n"
    "scope Base::S\n  def s() -> int = 4\nend Base::S\n\ndef text::g() -> int = 8\n"
)
"""Declares paths beneath ``base``'s ``Base`` and beneath ``text``."""
_BENEATH_TOO = "import base::*\n\ndef Base::k() -> int = 2\n"
_FORWARDING = "import base::*\n\nscope Base\n  export lib::{free}\nend Base\n"
"""Forwards ``lib``'s ``free`` beneath ``base``'s ``Base``."""
_FORWARDING_THROUGH = "import base::*\nimport al::*\n\nscope Geo\n  export lib::{free}\nend Geo\n"
_NAMED_ALIKE = "scope Base\n  def o() -> int = 3\nend Base\n"
"""Declares a scope of its own named like ``base``'s ``Base``."""
_READER_MODULES = {
    "base": _PLAIN,
    "al": _ALIASING,
    "lib": "def free() -> int = 6\n",
    "nb": _BENEATH,
    "nc": _BENEATH_TOO,
    "fw": _FORWARDING,
    "fg": _FORWARDING_THROUGH,
    "na": _NAMED_ALIKE,
    "nt": (
        "record Base\n  y: int\n\ndef Base::k() -> int = 9\n\n"
        "scope Base::S\n  def s() -> int = 7\nend Base::S\n"
    ),
    "a2": "import al::*\ntype G2 = Geo\n",
}


_OWN_NAMED_ALIKE = (
    "scope Q\n\n  scope Base\n    def w() -> int = 5\n  end Base\n\n"
    "  def q() -> int = {}::w()\nend Q\n\nQ::q()"
)
"""Reads ``w`` of a scope of the reader's own named like ``base``'s ``Base``."""


_OWN_ROOT_NAMED_ALIKE = "scope Base\n  def w() -> int = 5\nend Base\n\n{}"
"""Reads beside a root scope of the reader's own named like ``base``'s ``Base``."""


def _in_scope(imported: str, value: str) -> str:
    """Call a function returning *value*, in a scope importing *imported*."""
    return f"scope Q\n  import {imported}\n  def q() = {value}\nend Q\n\nQ::q()"


_SCENARIOS["an-alias-reaches-what-its-reader-imports-beneath-the-target-path"] = Scenario(
    modules=_READER_MODULES,
    header=("import base::*\nimport al::*",),
    probes={
        "forwarded": accepted("import fw::*\nGeo::free()", "int"),
        "forwarded-through-an-alias": accepted("import fg::*\nGeo::free()", "int"),
        "forwarded-by-a-route": rejected(
            "import fg\nfg::Geo::free()", UnknownMemberError, "fg::Geo::free"
        ),
        "declared": accepted("import nb::*\nGeo::k()", "int"),
        "own-alias": accepted("import nb::*\ntype OG = Base\nOG::k()", "int"),
        "own-alias-of-own-root": rejected(
            "import nb::*\ntype OG = Base\n::OG::k()", UnknownMemberError, "::OG::k"
        ),
        "annotation": accepted("import nb::*\nfn(p: Geo::In2) => p.a", "nb::Base::In2 -> int"),
        "constructor": accepted("import nb::*\nGeo::In2(a = 1)", "record nb::Base::In2\n  a: int"),
        "scope-beneath": accepted("import nb::*\nGeo::S::s()", "int"),
        "use-item": accepted("import nb::*\nuse Geo::{k}\nk()", "int"),
        "scope-named-alike": accepted("import na::*\nGeo::o()", "int"),
        "hidden": rejected("import nb::* hiding Base::k\nGeo::k()", HiddenMemberError, "Geo::k"),
        "two-distinct": rejected(
            "import nb::*\nimport nc::*\nGeo::k()", AmbiguousQualificationError, "Geo::k"
        ),
        "written-in-a-scope": accepted(
            "import nb::*\n\nscope Q\n  def q() -> int = Geo::k()\nend Q\n\nQ::q()", "int"
        ),
        "use-of-a-scope-beneath": accepted("import nb::*\nuse Geo::S::*\ns()", "int"),
        "use-of-a-scope-beneath-renamed": accepted("import nb::*\nuse Geo::S as R\nR::s()", "int"),
        "imported-in-a-scope-the-alias-is-not": accepted(_in_scope("nb::*", "Geo::k()"), "int"),
        "imported-in-a-scope-beside-the-own-root": accepted(
            'def Base::k() -> text = "own"\n' + _in_scope("nb::*", "Geo::k()"), "int"
        ),
        "the-target-path-beside-the-own-root": accepted(
            'def Base::k() -> text = "own"\n' + _in_scope("nb::*", "Base::k()"), "int"
        ),
        "alias-of-the-alias-in-a-scope": accepted(
            "type G2 = Geo\n" + _in_scope("nb::*", "G2::k()"), "int"
        ),
        "hidden-in-a-scope": rejected(
            _in_scope("nb::* hiding Base::k", "Geo::k()"), HiddenMemberError, "Geo::k"
        ),
        "alias-of-an-alias-of-the-alias-hidden": rejected(
            "import nb::* hiding Base::k\ntype G2 = Geo\ntype G3 = G2\nG3::k()",
            HiddenMemberError,
            "G3::k",
        ),
        "imported-alias-of-the-alias-hidden": rejected(
            "import a2::*\nimport nb::* hiding Base::k\nG2::k()", HiddenMemberError, "G2::k"
        ),
        "alias-of-the-alias-hidden-in-a-scope": rejected(
            "type G2 = Geo\n" + _in_scope("nb::* hiding Base::k", "G2::k()"),
            HiddenMemberError,
            "G2::k",
        ),
        "two-distinct-in-a-scope": rejected(
            _in_scope("nb::*\n  import nc::*", "Geo::k()"), AmbiguousQualificationError, "Geo::k"
        ),
        "scope-beneath-in-a-scope": accepted(_in_scope("nb::*", "Geo::S::s()"), "int"),
        "use-in-a-scope": accepted(_in_scope("nb::*\n  use Geo::S::*", "s()"), "int"),
        "annotation-in-a-scope": accepted(
            "scope Q\n  import nb::*\n  def q(p: Geo::In2) -> int = p.a\nend Q\n\nQ::q",
            "nb::Base::In2 -> int",
        ),
        "builtin-in-a-scope": accepted(_in_scope("nb::*", "U2::g()"), "int"),
        "alias-declared-in-a-scope": accepted(
            "import nb::*\n\nscope Q\n  type QG = Base\n  def q() -> int = QG::k()\nend Q\n\n"
            "Q::q()",
            "int",
        ),
        "own-scope-named-alike": accepted(_OWN_NAMED_ALIKE.format("Geo"), "int"),
        "own-scope-named-alike-by-the-target-path": accepted(
            _OWN_NAMED_ALIKE.format("Base"), "int"
        ),
        "declaring-beneath": accepted(
            "import nb::*\ndef Geo::k2() -> int = Geo::k()\nGeo::k2()", "int"
        ),
        "builtin": accepted("import nb::*\nU2::g()", "int"),
    },
)

_SCENARIOS["an-alias-names-a-type-its-reader-imports-beneath-the-target-path"] = Scenario(
    modules=_READER_MODULES,
    header=("import base::*\nimport al::*\nimport nb::*",),
    probes=type_positions("type", "Geo::In2", "nb::Base::In2"),
)

_SCENARIOS["an-alias-reaches-its-target-path-where-its-reader-names-no-type-so"] = Scenario(
    modules=_READER_MODULES,
    header=("import al::*",),
    probes={
        "target-path": accepted("import nb::*\nBase::k()", "int"),
        "alias": accepted("import nb::*\nGeo::k()", "int"),
        "forwarded": accepted("import fw::*\nGeo::free()", "int"),
        "annotation": accepted("import nb::*\nfn(p: Geo::In2) => p.a", "nb::Base::In2 -> int"),
        "scope-beneath": accepted("import nb::*\nGeo::S::s()", "int"),
        "use-of-a-scope-beneath": accepted("import nb::*\nuse Geo::S::*\ns()", "int"),
        "imported-in-a-scope": accepted(_in_scope("nb::*", "Geo::k()"), "int"),
        "own-scope-named-alike": accepted(_OWN_ROOT_NAMED_ALIKE.format("Geo::w()"), "int"),
        "own-scope-named-alike-by-the-target-path": accepted(
            _OWN_ROOT_NAMED_ALIKE.format("Base::w()"), "int"
        ),
        "own-scope-named-alike-through-an-alias-of-the-alias": accepted(
            "import a2::*\n" + _OWN_ROOT_NAMED_ALIKE.format("G2::w()"), "int"
        ),
        "own-scope-named-alike-read-in-a-scope": accepted(
            _OWN_ROOT_NAMED_ALIKE.format("scope Q\n  def q() -> int = Geo::w()\nend Q\n\nQ::q()"),
            "int",
        ),
        "own-scope-beneath-the-target-path": accepted(
            "scope Base::S2\n  def u() -> int = 7\nend Base::S2\n\nGeo::S2::u()", "int"
        ),
        "use-of-an-own-scope-beneath-the-target-path": accepted(
            "use Geo::S2::*\n\nscope Base::S2\n  def u() -> int = 7\nend Base::S2\n\nu()", "int"
        ),
        "own-scope-named-alike-beside-the-target": accepted(
            _OWN_ROOT_NAMED_ALIKE.format("Geo::f()"), "int"
        ),
        "scope-the-target-declares": accepted("Geo::Sub::t()", "int"),
        "use-of-a-scope-the-target-declares": accepted("use Geo::Sub::*\nt()", "int"),
        "use-of-a-scope-the-target-declares-renamed": accepted("use Geo::Sub as R\nR::t()", "int"),
        "use-of-an-item-of-a-scope-the-target-declares": accepted("use Geo::Sub::{t}\nt()", "int"),
        "use-through-an-alias-of-the-alias": accepted("import a2::*\nuse G2::Sub::*\nt()", "int"),
        "use-in-a-scope-of-a-scope-the-target-declares": accepted(
            "scope Q\n  use Geo::Sub::*\n  def q() -> int = t()\nend Q\n\nQ::q()", "int"
        ),
        "nothing-beneath-a-scope-the-target-declares": rejected(
            "Geo::Sub::nope()", UnknownMemberError, "Geo::Sub::nope"
        ),
        "hidden": rejected("import nb::* hiding Base::k\nGeo::k()", HiddenMemberError, "Geo::k"),
        "declared-by-the-target": accepted("import nb::*\nGeo::f()", "int"),
        "nothing": rejected("import nb::*\nGeo::nope()", UnknownMemberError, "Geo::nope"),
    },
)

_SCENARIOS["an-alias-reaches-no-qualifier-a-hiding-at-its-site-removed"] = Scenario(
    modules={**_MODULES, "ah": "import base::* hiding Base::Inner\ntype Gh = Base\n"},
    header=("import ah::*",),
    probes={
        "use": rejected("use Gh::Inner::*\n1", UnknownQualifierError, "use Gh::Inner::*"),
        "use-renamed": rejected(
            "use Gh::Inner as R\n1", UnknownQualifierError, "use Gh::Inner as R"
        ),
        "constructed": rejected("Gh::Inner(y = 1)", HiddenMemberError, "Gh::Inner"),
        "sibling": accepted("Gh::f()", "int"),
        "not-an-alias": rejected(
            "use Base::Inner::*\n1", UnknownQualifierError, "use Base::Inner::*"
        ),
    },
)

_SCENARIOS["an-alias-reaches-nothing-beneath-a-path-naming-another-type"] = Scenario(
    modules=_READER_MODULES,
    header=("import al::*\nimport nt::*",),
    probes={
        "target-path": accepted("Base::k()", "int"),
        "alias": rejected("Geo::k()", UnknownMemberError, "Geo::k"),
        "in-a-scope": rejected(_in_scope("nt::*", "Geo::k()"), UnknownMemberError, "Geo::k"),
        "use-by-the-target-path": accepted("use Base::S::*\ns()", "int"),
        "use": rejected("use Geo::S::*\ns()", UnknownQualifierError, "use Geo::S::*"),
        "declared-by-the-target": accepted("Geo::f()", "int"),
    },
)

_APPLYING_MODULES = {
    "gen": "record Box[T]\n  v: T\n\nenum Opt[T]\n  | Full(v: T)\n  | Empty\n",
    "ga": "import gen::*\ntype IntBox = Box[int]\ntype IntOpt = Opt[int]\n",
    "gb": "import gen::*\n\ndef Box::k() -> int = 1\n",
}
_INT_FULL = "record gen::Opt::Full[int]\n  v: int"

_SCENARIOS["an-applied-alias-reads-its-target-path-as-its-application-does"] = Scenario(
    modules=_APPLYING_MODULES,
    header=("import ga::*",),
    probes={
        "application": rejected(
            "import gen::*\nimport gb::*\nBox[int]::k()", TypeArgumentsError, "Box[int]"
        ),
        "alias": rejected("import gb::*\nIntBox::k()", TypeArgumentsError, "IntBox"),
        "application-in-a-scope": rejected(
            _in_scope("gb::*", "Box[int]::k()"), TypeArgumentsError, "Box[int]"
        ),
        "alias-in-a-scope": rejected(
            _in_scope("gb::*", "IntBox::k()"), TypeArgumentsError, "IntBox"
        ),
        "member-application-in-a-scope": accepted(
            _in_scope("gen::*", "Opt[int]::Full(v = 1)"), _INT_FULL
        ),
        "member-in-a-scope": accepted(_in_scope("gen::*", "IntOpt::Full(v = 1)"), _INT_FULL),
    },
)

_SCENARIOS["an-alias-named-as-its-target-path-reaches-imports-beneath-that-path"] = Scenario(
    modules=_READER_MODULES,
    header=("import base\nimport nb::*",),
    probes={
        "imported-beneath": accepted("type Base = base::Base\nBase::k()", "int"),
        "use-of-a-scope-beneath": accepted("use Base::S::*\ntype Base = base::Base\ns()", "int"),
        "declared-by-the-target": accepted("type Base = base::Base\nBase::f()", "int"),
        "nothing": rejected(
            "type Base = base::Base\nBase::nope()", UnknownMemberError, "Base::nope"
        ),
    },
)


class TestTransparentAliases:
    """Alias spellings in every position, the file part and every REPL grouping."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)


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
    phase, error, span, _identity = inline_verdict(
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
        pytest.param("import bj\nexport bj::{V2::zz}\n", id="item-hidden-at-the-alias-site"),
        pytest.param("import bj\nexport bj hiding V2::zz\n", id="hiding-hidden-at-the-alias-site"),
    ],
)
def test_an_export_item_through_an_alias_naming_nothing_is_an_error_where_written(
    tmp_path: Path, export: str
) -> None:
    modules = {**_MODULES, "tx": _TEXT_SCOPE, "bj": _HIDING_SITE}
    phase, error, span, _identity = inline_verdict(
        tmp_path, {"entry": "import ex\n1", "ex": export, **modules}
    )

    assert (phase, error) == ("scope", UnknownMemberError)
    assert span is not None and span.source.label.endswith("ex.agl")
    assert span_text(export, span) == export.splitlines()[1]


@pytest.mark.parametrize(
    "imports", ["import ex::{Nope}", "import ex::* hiding Nope"], ids=["tail", "hiding"]
)
def test_an_import_item_naming_nothing_in_a_cycle_is_an_error_where_written(
    tmp_path: Path, imports: str
) -> None:
    cycling = f"import base\n{imports}\nexport base::{{Base}}\ntype Geo = base::Base\n"
    phase, error, span, _identity = inline_verdict(
        tmp_path, {"entry": "import al\n1", "base": _BASE, "al": cycling, "ex": _CYCLED}
    )

    assert (phase, error) == ("scope", UnknownMemberError)
    assert span is not None and span.source.label.endswith("al.agl")
    assert span_text(cycling, span) == imports
