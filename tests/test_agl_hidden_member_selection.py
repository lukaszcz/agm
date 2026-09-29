"""``hiding`` removes a member from one import or ``use`` alone.

A hidden member is a hidden-member error when no other contribution
supplies the spelling, and selects the other contribution's member when
one does. Hiding a nested record leaves its owner's members intact, and
hiding an enum member leaves the enum's nested records intact.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`): both modes reach
the same verdict, error span and message, or accepted identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError, HiddenMemberError
from agm.agl.scope.symbols import UnknownMemberError
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_scenario_for_grouping,
    rejected,
    scenario_params,
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
            "hid-val": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
            "hid-annot": rejected("fn(p: Geo::Inner) => 1", HiddenMemberError, "Geo::Inner"),
            "hid-reptype": rejected("Geo::Inner", HiddenMemberError, "Geo::Inner"),
            "hidnest-val": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
            "hidnest-annot": rejected("fn(p: Geo::Inner) => 1", HiddenMemberError, "Geo::Inner"),
            "hidnest-pat": rejected(
                "case tl::Geo::Inner(y = 1) of\n  | Geo::Inner(y) => y",
                HiddenMemberError,
                "tl::Geo::Inner",
            ),
            "hidnest-alias": rejected("type AA = Geo::Inner\n1", HiddenMemberError, "Geo::Inner"),
            "hidnest-tyarg": rejected(
                "fn(p: array[Geo::Inner]) => 1", HiddenMemberError, "Geo::Inner"
            ),
            "hidnest-reptype": rejected("Geo::Inner", HiddenMemberError, "Geo::Inner"),
        },
    ),
    "hidden-nested-record-under-method": Scenario(
        modules={"tl": _TL},
        header=(
            "import tl::* hiding Geo::Inner",
            "def Geo::m(self) -> int = 1",
        ),
        probes={
            "hiddef-val": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
            "hiddef-annot": rejected("fn(p: Geo::Inner) => 1", HiddenMemberError, "Geo::Inner"),
            "hiddef-reptype": rejected("Geo::Inner", HiddenMemberError, "Geo::Inner"),
        },
    ),
    "hidden-nested-record-of-enum": Scenario(
        modules={"en": _EN},
        header=("import en::* hiding Color::Extra",),
        probes={
            "ehid-extra-val": rejected("Color::Extra(z = 1)", HiddenMemberError, "Color::Extra"),
            "ehid-extra-annot": rejected(
                "fn(p: Color::Extra) => 1", HiddenMemberError, "Color::Extra"
            ),
            "ehid-red-val": accepted("Color::Red", "record en::Color::Red"),
            "ehid-red-pat": rejected(
                "case Color::Blue of\n  | Color::Red => 1\n  | _ => 2",
                AglTypeError,
                "Color::Red",
                phase="typecheck",
            ),
            "ehid-red-annot": accepted("fn(p: Color::Red) => 1", "en::Color::Red -> int"),
        },
    ),
    "hidden-enum-member": Scenario(
        modules={"en": _EN},
        header=("import en::* hiding Color::Red",),
        probes={
            "ehidm-extra-val": accepted("Color::Extra(z = 1)", "record en::Color::Extra\n  z: int"),
            "ehidm-extra-annot": accepted("fn(p: Color::Extra) => 1", "en::Color::Extra -> int"),
            "ehidm-red-val": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "ehidm-red-pat": rejected(
                "case Color::Blue of\n  | Color::Red => 1\n  | _ => 2",
                HiddenMemberError,
                "Color::Red",
            ),
            "ehidm-red-annot": rejected("fn(p: Color::Red) => 1", HiddenMemberError, "Color::Red"),
        },
    ),
    "hidden-nested-record-of-enum-under-method": Scenario(
        modules={"en": _EN},
        header=(
            "import en::* hiding Color::Extra",
            "def Color::m(self) -> int = 1",
        ),
        probes={
            "ehiddef-extra-val": rejected("Color::Extra(z = 1)", HiddenMemberError, "Color::Extra"),
            "ehiddef-extra-annot": rejected(
                "fn(p: Color::Extra) => 1", HiddenMemberError, "Color::Extra"
            ),
            "ehiddef-red-val": accepted("Color::Red", "record en::Color::Red"),
            "ehiddef-red-pat": rejected(
                "case Color::Blue of\n  | Color::Red => 1\n  | _ => 2",
                AglTypeError,
                "Color::Red",
                phase="typecheck",
            ),
            "ehiddef-red-annot": accepted("fn(p: Color::Red) => 1", "en::Color::Red -> int"),
        },
    ),
    "hidden-member-beside-other-enum": Scenario(
        modules={"en": _EN, "en2": _EN2},
        header=(
            "import en::* hiding Color::Red",
            "import en2::*",
        ),
        probes={
            "d2-val": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "d2-annot": rejected("fn(p: Color::Red) => 1", HiddenMemberError, "Color::Red"),
            "d2-pat": rejected(
                "case 1 of\n  | Color::Red => 1\n  | _ => 2", HiddenMemberError, "Color::Red"
            ),
        },
    ),
    "hidden-member-supplied-by-other-import": Scenario(
        modules={"en": _EN_2, "en2": _EN2_2},
        header=(
            "import en::* hiding Color::Red",
            "import en2::*",
        ),
        probes={
            "d2b-val": accepted("Color::Red", "record en2::Color::Red"),
            "d2b-annot": accepted("fn(p: Color::Red) => 1", "en2::Color::Red -> int"),
            "d2b-pat": rejected(
                "case en2::Color::Green of\n  | Color::Red => 1\n  | _ => 2",
                AglTypeError,
                "Color::Red",
                phase="typecheck",
            ),
            "d2b-is": rejected(
                "en2::Color::Green is Color::Red",
                AglTypeError,
                "en2::Color::Green is Color::Red",
                phase="typecheck",
            ),
            "d2b-alias": accepted("type AA = Color::Red\n1", "int"),
            "d2b-reptype": accepted("Color::Red", "record en2::Color::Red"),
            "d2b-pat2": accepted(
                (
                    "let v: en2::Color = en2::Color::Green\n"
                    "case v of\n"
                    "  | Color::Red => 1\n"
                    "  | _ => 2"
                ),
                "int",
            ),
            "d2b-is2": accepted("let v: en2::Color = en2::Color::Green\nv is Color::Red", "bool"),
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
            "amb-hid-alias": accepted("type AA = Geo::Inner\n1", "int"),
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
            "alias-use-hide-Green": rejected(
                "Color::Green", UnknownMemberError, "use m::* hiding Color::Green"
            ),
        },
        legal=frozenset({(1, 1, 3), (1, 4), (2, 3), (5,)}),
    ),
}


class TestHiddenMemberSelection:
    """Hidden members across imports, uses, routes and method paths."""

    @pytest.mark.parametrize(("scenario", "sizes"), scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(
        self, tmp_path: Path, scenario: str, sizes: tuple[int, ...]
    ) -> None:
        assert_scenario_for_grouping(tmp_path, _SCENARIOS[scenario], sizes)
