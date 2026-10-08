"""``hiding`` removes a member from one import or ``use`` alone.

A hidden member is a hidden-member error when no other contribution
supplies the spelling, and selects the other contribution's member when
one does. Hiding a nested record leaves its owner's members intact, and
hiding an enum member leaves the enum's nested records intact.

Every probe runs inline; verdict representatives also cover file and REPL
entry boundaries through :mod:`tests.agl.qualifier_support`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError, HiddenMemberError
from agm.agl.scope.symbols import UnknownMemberError, UnknownQualifierError
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    assert_scenario,
    option_identity,
    rejected,
    scenario_params,
    type_positions,
    type_positions_rejected,
)

_TL = "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"
_EN = "enum Color\n  | Red\n  | Blue\nrecord Color::Extra\n  z: int\n"
_EN2 = "enum Color\n  | Green\n"
_EN_2 = "enum Color\n  | Red\n  | Blue\n"
_EN2_2 = "enum Color\n  | Red\n  | Green\n"
_TL2 = "record Geo\n  z: int\nrecord Geo::Inner\n  w: int\n"
_ONE_TYPES = "enum Color\n  | Red\n  | Green\n"
_TWO_TYPES = "enum Color\n  | Red\n  | Blue\n"
_M = "enum Color\n  | Red\n  | Green\n"
_N = "enum Color\n  | Red\n  | Blue\n"
_M_2 = "enum Base\n  | Red\n  | Green\ntype Color = Base\n"

_SCENARIOS = {
    "hidden-nested-record": Scenario(
        modules={"tl": _TL},
        header=("import tl::* hiding Geo::Inner",),
        probes={
            "hidden-val": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
            **type_positions_rejected("hidden", "Geo::Inner", HiddenMemberError),
            "hidden-reptype": rejected("Geo::Inner", HiddenMemberError, "Geo::Inner"),
            "hidden-pat": rejected(
                "case tl::Geo::Inner(y = 1) of\n  | Geo::Inner(y) => y",
                HiddenMemberError,
                "tl::Geo::Inner",
            ),
        },
    ),
    "hidden-nested-record-under-method": Scenario(
        modules={"tl": _TL},
        header=(
            "import tl::* hiding Geo::Inner",
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "hidden-under-method-val": rejected(
                "Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"
            ),
            "hidden-under-method-annot": rejected(
                "fn(p: Geo::Inner) => 1", HiddenMemberError, "Geo::Inner"
            ),
            "hidden-under-method-reptype": rejected("Geo::Inner", HiddenMemberError, "Geo::Inner"),
        },
    ),
    "hidden-nested-record-of-enum": Scenario(
        stdlib=True,
        modules={"en": _EN},
        header=("import en::* hiding Color::Extra",),
        probes={
            "enum-nested-extra-val": rejected(
                "Color::Extra(z = 1)", HiddenMemberError, "Color::Extra"
            ),
            **type_positions_rejected("enum-nested-extra", "Color::Extra", HiddenMemberError),
            "enum-nested-red-val": accepted("Color::Red", "record en::Color::Red"),
            "enum-nested-red-pat": rejected(
                "case Color::Blue of\n  | Color::Red => 1\n  | _ => 2",
                AglTypeError,
                "Color::Red",
                phase="typecheck",
            ),
            **type_positions("enum-nested-red", "Color::Red", "en::Color::Red"),
        },
    ),
    "hidden-enum-member": Scenario(
        stdlib=True,
        modules={"en": _EN},
        header=("import en::* hiding Color::Red",),
        probes={
            "enum-member-extra-val": accepted(
                "Color::Extra(z = 1)", "record en::Color::Extra\n  z: int"
            ),
            **type_positions("enum-member-extra", "Color::Extra", "en::Color::Extra"),
            "enum-member-red-val": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "enum-member-red-pat": rejected(
                "case Color::Blue of\n  | Color::Red => 1\n  | _ => 2",
                HiddenMemberError,
                "Color::Red",
            ),
            **type_positions_rejected("enum-member-red", "Color::Red", HiddenMemberError),
        },
    ),
    "hidden-nested-record-of-enum-under-method": Scenario(
        modules={"en": _EN},
        header=(
            "import en::* hiding Color::Extra",
            "def Color::m(self) -> int = 1",
        ),
        probes={
            "enum-under-method-extra-val": rejected(
                "Color::Extra(z = 1)", HiddenMemberError, "Color::Extra"
            ),
            "enum-under-method-extra-annot": rejected(
                "fn(p: Color::Extra) => 1", HiddenMemberError, "Color::Extra"
            ),
            "enum-under-method-red-val": accepted("Color::Red", "record en::Color::Red"),
            "enum-under-method-red-pat": rejected(
                "case Color::Blue of\n  | Color::Red => 1\n  | _ => 2",
                AglTypeError,
                "Color::Red",
                phase="typecheck",
            ),
            "enum-under-method-red-annot": accepted(
                "fn(p: Color::Red) => 1", "en::Color::Red -> int"
            ),
        },
    ),
    "hidden-member-beside-other-enum": Scenario(
        modules={"en": _EN, "en2": _EN2},
        header=(
            "import en::* hiding Color::Red",
            "import en2::*",
        ),
        probes={
            "other-enum-val": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "other-enum-annot": rejected("fn(p: Color::Red) => 1", HiddenMemberError, "Color::Red"),
            "other-enum-pat": rejected(
                "case 1 of\n  | Color::Red => 1\n  | _ => 2", HiddenMemberError, "Color::Red"
            ),
        },
    ),
    "hidden-member-supplied-by-other-import": Scenario(
        stdlib=True,
        modules={"en": _EN_2, "en2": _EN2_2},
        header=(
            "import en::* hiding Color::Red",
            "import en2::*",
        ),
        probes={
            "other-import-val": accepted("Color::Red", "record en2::Color::Red"),
            "other-import-annot": accepted("fn(p: Color::Red) => 1", "en2::Color::Red -> int"),
            "other-import-pat": rejected(
                "case en2::Color::Green of\n  | Color::Red => 1\n  | _ => 2",
                AglTypeError,
                "Color::Red",
                phase="typecheck",
            ),
            "other-import-is": rejected(
                "en2::Color::Green is Color::Red",
                AglTypeError,
                "en2::Color::Green is Color::Red",
                phase="typecheck",
            ),
            "other-import-alias": accepted(
                "type AA = Color::Red\nfn(p: AA) => p", "en2::Color::Red -> en2::Color::Red"
            ),
            "other-import-pat2": accepted(
                (
                    "let v: en2::Color = en2::Color::Green\n"
                    "case v of\n"
                    "  | Color::Red => 1\n"
                    "  | _ => 2"
                ),
                "int",
            ),
            "other-import-is2": accepted(
                "let v: en2::Color = en2::Color::Green\nv is Color::Red", "bool"
            ),
            "other-import-narrow2": accepted(
                "let v: en2::Color = en2::Color::Green\nv as? Color::Red",
                option_identity("en2::Color::Red"),
            ),
        },
    ),
    "hidden-nested-record-supplied-by-other-import": Scenario(
        modules={"tl": _TL, "tl2": _TL2},
        header=(
            "import tl::* hiding Geo::Inner",
            "import tl2::*",
        ),
        probes={
            "amb-hid-val": accepted("Geo::Inner(w = 1)", "record tl2::Geo::Inner\n  w: int"),
            "amb-hid-annot": accepted("fn(p: Geo::Inner) => 1", "tl2::Geo::Inner -> int"),
            "amb-hid-pat": accepted(
                (
                    "let g: tl2::Geo::Inner = tl2::Geo::Inner(w = 1)\n"
                    "case g of\n"
                    "  | Geo::Inner(w) => w"
                ),
                "int",
            ),
            "amb-hid-alias": accepted(
                "type AA = Geo::Inner\nfn(p: AA) => p", "tl2::Geo::Inner -> tl2::Geo::Inner"
            ),
            "amb-hid-reptype": accepted("Geo::Inner", "int -> tl2::Geo::Inner"),
        },
    ),
    "route-member-hidden-by-both-owners": Scenario(
        modules={"one/types": _ONE_TYPES, "two/types": _TWO_TYPES},
        header=(
            "import one/types hiding Color::Red",
            "import two/types hiding Color::Red",
        ),
        probes={
            "route-bothhide-Red-value": rejected(
                "types::Color::Red", HiddenMemberError, "types::Color::Red"
            ),
            "route-bothhide-Red-annot": rejected(
                "fn(x: types::Color::Red) => 1", HiddenMemberError, "types::Color::Red"
            ),
            "route-bothhide-Red-pattern": rejected(
                (
                    "let v: two/types::Color = two/types::Color::Blue\n"
                    "case v of\n"
                    "  | types::Color::Red => 1\n"
                    "  | _ => 2"
                ),
                HiddenMemberError,
                "types::Color::Red",
            ),
            "route-bothhide-Red-is": rejected(
                ("let v: two/types::Color = two/types::Color::Blue\nv is types::Color::Red"),
                HiddenMemberError,
                "types::Color::Red",
            ),
        },
    ),
    "route-member-hidden-by-its-only-owner": Scenario(
        modules={"one/types": _ONE_TYPES, "two/types": _TWO_TYPES},
        header=(
            "import one/types hiding Color::Green",
            "import two/types",
        ),
        probes={
            "route-onehide-Green-value": rejected(
                "types::Color::Green", HiddenMemberError, "types::Color::Green"
            ),
            "route-onehide-Green-annot": rejected(
                "fn(x: types::Color::Green) => 1", HiddenMemberError, "types::Color::Green"
            ),
            "route-onehide-Green-pattern": rejected(
                (
                    "let v: two/types::Color = two/types::Color::Blue\n"
                    "case v of\n"
                    "  | types::Color::Green => 1\n"
                    "  | _ => 2"
                ),
                HiddenMemberError,
                "types::Color::Green",
            ),
            "route-onehide-Green-is": rejected(
                ("let v: two/types::Color = two/types::Color::Blue\nv is types::Color::Green"),
                HiddenMemberError,
                "types::Color::Green",
            ),
        },
    ),
    "used-member-hidden-by-both-owners": Scenario(
        modules={"m": _M, "n": _N},
        header=(
            "import m",
            "import n",
            "use m::* hiding Color::Red",
            "use n::* hiding Color::Red",
        ),
        probes={
            "use-bothhide-Red-value": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "use-bothhide-Red-annot": rejected(
                "fn(x: Color::Red) => 1", HiddenMemberError, "Color::Red"
            ),
            "use-bothhide-Red-pattern": rejected(
                ("let v: n::Color = n::Color::Blue\ncase v of\n  | Color::Red => 1\n  | _ => 2"),
                HiddenMemberError,
                "Color::Red",
            ),
            "use-bothhide-Red-is": rejected(
                "let v: n::Color = n::Color::Blue\nv is Color::Red", HiddenMemberError, "Color::Red"
            ),
        },
    ),
    "used-member-hidden-by-its-only-owner": Scenario(
        modules={"m": _M, "n": _N},
        header=(
            "import m",
            "import n",
            "use m::* hiding Color::Green",
            "use n::*",
        ),
        probes={
            "use-onehide-Green-value": rejected("Color::Green", HiddenMemberError, "Color::Green"),
            "use-onehide-Green-annot": rejected(
                "fn(x: Color::Green) => 1", HiddenMemberError, "Color::Green"
            ),
            "use-onehide-Green-pattern": rejected(
                ("let v: n::Color = n::Color::Blue\ncase v of\n  | Color::Green => 1\n  | _ => 2"),
                HiddenMemberError,
                "Color::Green",
            ),
            "use-onehide-Green-is": rejected(
                "let v: n::Color = n::Color::Blue\nv is Color::Green",
                HiddenMemberError,
                "Color::Green",
            ),
        },
    ),
    "imported-member-hidden-by-both-owners": Scenario(
        modules={"m": _M, "n": _N},
        header=(
            "import m",
            "import n",
            "import m::* hiding Color::Red",
            "import n::* hiding Color::Red",
        ),
        probes={
            "imp-bothhide-Red-value": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "imp-bothhide-Red-annot": rejected(
                "fn(x: Color::Red) => 1", HiddenMemberError, "Color::Red"
            ),
            "imp-bothhide-Red-pattern": rejected(
                ("let v: n::Color = n::Color::Blue\ncase v of\n  | Color::Red => 1\n  | _ => 2"),
                HiddenMemberError,
                "Color::Red",
            ),
            "imp-bothhide-Red-is": rejected(
                "let v: n::Color = n::Color::Blue\nv is Color::Red", HiddenMemberError, "Color::Red"
            ),
        },
    ),
    "hiding-through-an-alias-owner": Scenario(
        modules={"m": _M_2, "n": _N},
        header=(
            "import m",
            "import n",
            "use m::* hiding Color::Green",
            "use n::*",
        ),
        probes={
            "alias-use-hide-Green": rejected("Color::Green", HiddenMemberError, "Color::Green"),
            "alias-use-hide-Green-target": rejected(
                "Base::Green", HiddenMemberError, "Base::Green"
            ),
            "alias-use-other-Blue": accepted("Color::Blue", "record n::Color::Blue"),
        },
    ),
}

_BENEATH = (
    "scope S\n  def a() -> int = 1\n  def b() -> int = 2\nend S\n"
    "\n"
    "record R\n  x: int\n\ndef R::m() -> int = 1\ndef R::n() -> int = 2\n"
)
"""Functions beneath a scope and beneath a type."""

_SCENARIOS["a-re-export-hiding-a-path-hides-it-beneath-the-scope-or-type-above"] = Scenario(
    modules={"sc": _BENEATH, "ex": "import sc\nexport sc hiding S::a, R::m\n"},
    header=("import ex::*\nimport ex", "import sc::{S::b}"),
    probes={
        f"{route}{owner}-{verdict}": probe
        for route in ("", "ex::")
        for owner, hidden, kept in (("S", "a", "b"), ("R", "m", "n"))
        for verdict, probe in {
            "hidden": rejected(
                f"{route}{owner}::{hidden}()", HiddenMemberError, f"{route}{owner}::{hidden}"
            ),
            "kept": accepted(f"{route}{owner}::{kept}()", "int"),
        }.items()
    },
)

_BENEATH_NESTED = (
    "scope S\n  def a() -> int = 1\n\n  scope T\n    def c() -> int = 3\n  end T\nend S\n"
    "\n"
    "record R\n  x: int\n\ndef R::m() -> int = 1\ndef f() -> int = 4\ndef g() -> int = 5\n"
)
"""Functions beneath a scope, a scope within it and a type, and two beside them."""

_SCENARIOS |= {
    f"{name}-of-a-scope-or-type-hides-every-path-beneath-it": Scenario(
        modules={
            "sc": _BENEATH_NESTED,
            "ex": "import sc\nexport sc hiding S, R, f\n",
            "ex2": "import ex\nexport ex\n",
        },
        header=(header,),
        probes={
            **{
                f"{routed}{path}": rejected(
                    f"{qualifier}{path}{arguments}", HiddenMemberError, qualifier + path
                )
                for routed, qualifier in (("", ""), ("routed-", f"{route}::"))
                for path, arguments in (("S::a", "()"), ("S::T::c", "()"), ("R::m", "()"))
            },
            "routed-type": rejected(f"{route}::R(x = 1)", HiddenMemberError, f"{route}::R"),
            **type_positions_rejected("routed-type", f"{route}::R", HiddenMemberError),
            "routed-function": rejected(f"{route}::f()", HiddenMemberError, f"{route}::f"),
            "kept": accepted("g()", "int"),
            "routed-kept": accepted(f"{route}::g()", "int"),
        },
    )
    for name, route, header in (
        ("an-import-hiding", "sc", "import sc::* hiding S, R, f\nimport sc hiding S, R, f"),
        ("a-re-export-hiding", "ex", "import ex::*\nimport ex"),
        ("a-re-export-of-a-re-export-hiding", "ex2", "import ex2::*\nimport ex2"),
    )
}


_PURE_SCOPE = (
    "scope S\n  def a() -> int = 1\n  def b() -> int = 2\n\n  scope T\n    def c() -> int = 3\n"
    "  end T\nend S\n\ndef g() -> int = 5\n"
)
"""A scope declaring no type, a scope within it, and a function beside it."""

_BENEATH_PURE_SCOPE = {
    "member": ("{r}S::a()", "{r}S::a"),
    "never-declared": ("{r}S::nope()", "{r}S::nope"),
    "nested": ("{r}S::T::c()", "{r}S::T::c"),
    "beneath-never-declared": ("{r}S::nope::x()", "{r}S::nope::x"),
    "use-glob": ("use {r}S::*", "use {r}S::*"),
    "use-item": ("use {r}S::{{a}}", "use {r}S::{{a}}"),
    "use-never-declared-item": ("use {r}S::{{nope}}", "use {r}S::{{nope}}"),
    "use-rename": ("use {r}S as Q", "use {r}S as Q"),
    "use-nested": ("use {r}S::T::*", "use {r}S::T::*"),
}
"""Each read beneath ``sc``'s scope ``S`` after a route ``{r}``, with its span."""


def _beneath_pure_scope_probes(route: str, region: bool) -> dict[str, Probe]:
    """Every read beneath ``S`` once a ``hiding`` removed it, after *route*, in a *region* or not.

    Each is hidden, in every position, while ``g`` stays and a never-imported
    qualifier stays unknown.
    """
    hiding = "  import sc::* hiding S\n"

    def placed(text: str) -> str:
        if not region:
            return text
        if text.startswith("use "):
            return f"scope r\n{hiding}  {text}\nend r"
        return f"scope r\n{hiding}  def v() -> int = {text}\nend r\nr::v()"

    probes = {
        name: rejected(placed(text.format(r=route)), HiddenMemberError, span.format(r=route))
        for name, (text, span) in _BENEATH_PURE_SCOPE.items()
    } | {
        "kept": accepted(placed(f"{route}g()"), "int"),
        "unknown-qualifier": rejected(placed("Nope::x()"), UnknownQualifierError, "Nope::x"),
    }
    if not region:
        probes |= type_positions_rejected("type", f"{route}S::Nope", HiddenMemberError)
    return probes


_SCENARIOS |= {
    f"{name}-hiding-a-scope-declaring-no-type-hides-every-path-beneath-it": Scenario(
        modules={"sc": _PURE_SCOPE},
        header=header,
        probes=_beneath_pure_scope_probes(route, region=not header),
    )
    for name, route, header in (
        ("an-import", "", ("import sc::* hiding S",)),
        ("a-use", "", ("import sc\nuse sc::* hiding S",)),
        ("a-route", "sc::", ("import sc hiding S",)),
        ("a-region-import", "", ()),
    )
}
_SCENARIOS["hiding-every-member-of-a-scope-keeps-it-a-qualifier"] = Scenario(
    modules={"sc": _PURE_SCOPE},
    header=("import sc::* hiding S::a, S::b",),
    probes={
        "use-glob": accepted("use S::*\n1", "int"),
        "member": rejected("S::a()", HiddenMemberError, "S::a"),
        "never-declared": rejected("S::nope()", UnknownMemberError, "S::nope"),
    },
)


class TestHiddenMemberSelection:
    """Hidden members across imports, uses, routes and method paths."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_semantics_and_entry_boundaries(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)
