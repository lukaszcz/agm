"""A declaration path written through an alias is its target's path.

``def Geo::m(self)``, ``def Geo::f()``, ``scope Geo`` and ``def
Geo::Inner::k()`` with ``type Geo = Base`` declare beneath ``Base``: both
spellings reach the declaration in every position, whichever spelling
declared it, the alias own or imported and its target own or imported. Two
declarations of one name beneath the two spellings are one path declared
twice, in one REPL entry or across entries.

Every probe is checked in the file part and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.scope.symbols import (
    AglScopeError,
    DuplicateDeclarationError,
    TypeArgumentsError,
    UnknownMemberError,
)
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    assert_repl_verdicts,
    assert_scenario,
    info,
    rejected,
    scenario_params,
    type_positions,
)

_BASE = "record Base\n  x: int\nrecord Base::Inner\n  y: int\ndef Base::f() -> int = 1\n"
_ALIASING = "import base::*\nexport base::{Base}\ntype Geo = Base\n"
"""Exports ``base``'s ``Base`` and an alias of it."""
_MODULES = {"base": _BASE, "al": _ALIASING}

_OWN_ALIAS = ("import base::*", "type Geo = Base")
_IMPORTED_ALIAS = ("import al::*",)
_ROUTED_ALIAS = ("import base", "import base::*", "type Geo = base::Base")
_LOCAL_TARGET = ("record Base\n  x: int\nrecord Base::Inner\n  y: int", "type Geo = Base")


def _declarations(spelling: str) -> str:
    """A method, a static function, a region's function and a nested one, beneath *spelling*."""
    return (
        f"def {spelling}::m(self) -> int = self.x\n"
        f"def {spelling}::g() -> int = 2\n"
        f"scope {spelling}\n  def h() -> int = 3\nend {spelling}\n"
        f"def {spelling}::Inner::k() -> int = 4"
    )


def _reached_probes(record: str) -> dict[str, Probe]:
    """Every declaration of :func:`_declarations`, reached through both spellings.

    *record* renders the aliased record's type.
    """
    probes = {"dot": accepted("Base(x = 1).m()", "int")}
    for spelling in ("Base", "Geo"):
        probes |= {
            f"{spelling}-call": accepted(f"{spelling}::m(Base(x = 1))", "int"),
            f"{spelling}-method-value": accepted(f"{spelling}::m", f"{record} -> int"),
            f"{spelling}-static": accepted(f"{spelling}::g()", "int"),
            f"{spelling}-region": accepted(f"{spelling}::h()", "int"),
            f"{spelling}-nested": accepted(f"{spelling}::Inner::k()", "int"),
        }
    probes |= {
        "anchored-static": accepted("::Base::g()", "int"),
        "anchored-region": accepted("::Base::h()", "int"),
        "anchored-nested": accepted("::Base::Inner::k()", "int"),
    }
    return probes


def _duplicate_probes(first: str, second: str) -> dict[str, Probe]:
    """One name declared beneath *first*, then again beneath *second*."""
    return {
        f"{first}-then-{second}-static": rejected(
            f"def {first}::d() -> int = 1\ndef {second}::d() -> int = 2\n1",
            DuplicateDeclarationError,
            f"def {second}::d() -> int = 2",
        ),
        f"{first}-then-{second}-method": rejected(
            f"def {first}::n(self) -> int = 1\ndef {second}::n(self) -> int = 2\n1",
            DuplicateDeclarationError,
            f"def {second}::n(self) -> int = 2",
        ),
        f"{first}-then-{second}-region": rejected(
            f"scope {first}\n  def d() -> int = 1\nend {first}\ndef {second}::d() -> int = 2\n1",
            DuplicateDeclarationError,
            f"def {second}::d() -> int = 2",
        ),
    }


def _beside_imported_probes(nested: str, routes: tuple[str, ...]) -> dict[str, Probe]:
    """What :func:`_beside_imported` declares, reached through both spellings.

    *nested* renders the own nested record's type. Through each of *routes*,
    module routes or anchored, only the routed module's declarations are reached.
    """
    own = {
        f"{spelling}-{position}": probe
        for spelling in ("Base", "Geo")
        for position, probe in {
            "static": accepted(f"{spelling}::f()", "text"),
            "static-value": accepted(f"{spelling}::f", "() -> text"),
            "region": accepted(f"{spelling}::h()", "text"),
            "nested-value": accepted(f"{spelling}::Inner(q = true)", f"record {nested}\n  q: bool"),
            **type_positions("nested", f"{spelling}::Inner", nested),
        }.items()
    }
    return own | {
        f"{anchor}{route}-{position}": probe
        for route in routes
        for anchor in ("", "/")
        for position, probe in {
            "static": accepted(f"{anchor}{route}::f()", "int"),
            "region": accepted(f"{anchor}{route}::h()", "int"),
            "nested-value": accepted(
                f"{anchor}{route}::Inner(y = 1)", "record base::Base::Inner\n  y: int"
            ),
        }.items()
    }


def _beside_imported(spelling: str) -> str:
    """Own declarations beneath *spelling* at paths ``base`` declares beneath ``Base`` too."""
    return (
        f'def {spelling}::f() -> text = "own"\n'
        f'scope {spelling}\n  def h() -> text = "own"\nend {spelling}\n'
        f"record {spelling}::Inner\n  q: bool"
    )


_SCENARIOS = (
    {
        f"declared-through-{spelling}-{name}": Scenario(
            modules=_MODULES,
            header=(*header, _declarations(spelling)),
            probes=_reached_probes(record),
        )
        for spelling in ("Base", "Geo")
        for name, header, record in (
            ("an-own-alias", _OWN_ALIAS, "base::Base"),
            ("an-imported-alias", _IMPORTED_ALIAS, "base::Base"),
            ("an-alias-of-an-own-type", _LOCAL_TARGET, "Base"),
        )
    }
    | {
        f"declared-through-{spelling}-beside-an-imported-declaration-{name}": Scenario(
            modules={**_MODULES, "base": _BASE + "def Base::h() -> int = 3\n"},
            header=(*header, _beside_imported(spelling)),
            probes=_beside_imported_probes(f"{spelling}::Inner", routes),
        )
        for spelling in ("Base", "Geo")
        for name, header, routes in (
            ("an-own-alias", _OWN_ALIAS, ("base::Base",)),
            ("an-imported-alias", _IMPORTED_ALIAS, ("al::Base", "al::Geo")),
        )
    }
    | {
        f"declared-beneath-an-alias-of-a-path-through-{name}": Scenario(
            modules=_MODULES,
            header=(
                *header,
                "record Base::Inner\n  q: bool",
                "type Nested = Geo::Inner",
                'def Nested::Sub::k() -> text = "k"',
            ),
            probes={
                **{
                    f"{spelling}-nested": accepted(f"{spelling}::Sub::k()", "text")
                    for spelling in ("Base::Inner", "Geo::Inner", "Nested")
                },
                "value": accepted("Nested(q = true)", "record Base::Inner\n  q: bool"),
                **type_positions("alias", "Nested", "Base::Inner"),
            },
        )
        for name, header in (
            ("an-own-alias", _OWN_ALIAS),
            ("an-imported-alias", _IMPORTED_ALIAS),
            ("an-alias-of-a-routed-target", _ROUTED_ALIAS),
        )
    }
    | {
        "declared-through-an-alias-of-a-routed-target": Scenario(
            modules=_MODULES,
            header=(*_ROUTED_ALIAS, _declarations("Geo")),
            probes=_reached_probes("base::Base"),
        )
    }
    | {
        f"duplicated-beneath-{name}": Scenario(
            modules=_MODULES,
            header=header,
            probes=_duplicate_probes("Base", "Geo") | _duplicate_probes("Geo", "Base"),
        )
        for name, header in (
            ("an-own-alias", _OWN_ALIAS),
            ("an-imported-alias", _IMPORTED_ALIAS),
            ("an-alias-of-an-own-type", _LOCAL_TARGET),
        )
    }
    | {
        f"redeclared-beneath-{second}-after-{first}-{name}": Scenario(
            modules=_MODULES,
            header=(*header, f"def {first}::g() -> int = 1\ndef {first}::m(self) -> int = 1"),
            probes={
                "static": rejected(
                    f"def {second}::g() -> int = 2\n1",
                    DuplicateDeclarationError,
                    f"def {second}::g() -> int = 2",
                ),
                "method": rejected(
                    f"def {second}::m(self) -> int = 2\n1",
                    DuplicateDeclarationError,
                    f"def {second}::m(self) -> int = 2",
                ),
            },
        )
        for first, second in (("Base", "Geo"), ("Geo", "Base"))
        for name, header in (
            ("an-own-alias", _OWN_ALIAS),
            ("an-imported-alias", _IMPORTED_ALIAS),
            ("an-alias-of-an-own-type", _LOCAL_TARGET),
        )
    }
)

_APPLIED_MODULES = {"gen": "record Box[T]\n  v: T\nrecord Box::In\n  w: int\n"}
_APPLIED = ("import gen::*", "type IntBox = Box[int]", "type IB2 = IntBox")
"""Applied aliases of ``gen``'s ``Box[int]``, directly and through an alias."""


def _applied_head_probes(spelling: str) -> dict[str, Probe]:
    """Every declaration beneath *spelling*, a type application, is rejected at *spelling*.

    A written application heads only a ``def``'s one-segment path.
    """
    written = "[" in spelling
    return {
        f"{spelling}-{name}": rejected(text, TypeArgumentsError, spelling)
        for name, text in {
            "method": f"def {spelling}::m(self) -> int = 1",
            "static": f"def {spelling}::f() -> int = 1",
            **(
                {}
                if written
                else {
                    "nested": f"def {spelling}::In::f() -> int = 1",
                    "record": f"record {spelling}::R\n  x: int",
                }
            ),
        }.items()
    }


_SCENARIOS |= {
    "declared-beneath-a-type-application": Scenario(
        modules=_APPLIED_MODULES,
        header=_APPLIED,
        probes={
            **_applied_head_probes("Box[int]"),
            **_applied_head_probes("IntBox"),
            **_applied_head_probes("IB2"),
            "region": rejected(
                "scope IntBox\n  def f() -> int = 1\nend IntBox", TypeArgumentsError, "IntBox"
            ),
            "generic-head-method": rejected(
                "def Box[T]::m(self) -> int = 1", TypeArgumentsError, "Box[T]"
            ),
            "unknown-applied-head": rejected(
                "def bytes[int]::m(self) -> int = 1", TypeArgumentsError, "bytes[int]"
            ),
            "builtin-generic-head-method": accepted(
                "def array[E]::m2(self) -> int = 1\n[1].m2()", "int"
            ),
            "builtin-text-keyed-head-method": accepted(
                'def dict[text, V]::m2(self) -> int = 1\n{"a": 1}.m2()', "int"
            ),
            "builtin-generic-head-static": rejected(
                "def array[E]::f() -> int = 1", TypeArgumentsError, "array[E]"
            ),
            "builtin-applied-head-static": rejected(
                "def array[int]::f() -> int = 1", TypeArgumentsError, "array[int]"
            ),
            "builtin-applied-head-method": rejected(
                "def array[int]::m2(self) -> int = 1", AglScopeError, "array[int]"
            ),
            "builtin-bare-generic-head-method": rejected(
                "def array::m2(self) -> int = 1", AglScopeError, "array"
            ),
        },
    )
}


_BUILTIN_TARGETS = (
    "type U2 = text\ntype Brr[E] = array[E]\ntype IB = array[int]\n"
    "type TD[V] = dict[text, V]\ntype Sw[A, B] = dict[B, A]\n"
)
"""Aliases of built-in types, declared through an import; ``Sw`` swaps its arguments."""


def _builtin_head_probes(text: str, array: str, applied: str) -> dict[str, Probe]:
    """Declarations beneath aliases of ``text``, ``array[E]`` and ``array[int]``, or those.

    Each is the declaration written through the target: the same verdict and path.
    """
    return {
        f"{text}-method": accepted(f'def {text}::m(self) -> int = 1\n"a".m()', "int"),
        f"{text}-method-by-target": accepted(
            f'def {text}::m(self) -> int = 1\ntext::m("a")', "int"
        ),
        f"{text}-static-by-target": accepted(f"def {text}::f() -> int = 1\ntext::f()", "int"),
        f"{text}-static": accepted(f"def {text}::f() -> int = 1\n{text}::f()", "int"),
        f"{text}-twice": rejected(
            f"def {text}::f() -> int = 1\ndef text::f() -> int = 2",
            DuplicateDeclarationError,
            "def text::f() -> int = 2",
        ),
        f"{text}-region": accepted(
            f'scope {text}\n  def g(self) -> int = 1\nend {text}\n"a".g()', "int"
        ),
        f"{array}-method": accepted(f"def {array}[E]::m(self) -> int = 1\n[1].m()", "int"),
        f"{array}-method-by-target": accepted(
            f"def {array}[E]::m(self) -> int = 1\narray::m([1])", "int"
        ),
        f"{array}-bare-method": rejected(f"def {array}::m(self) -> int = 1", AglScopeError, array),
        f"{array}-static": rejected(
            f"def {array}[E]::f() -> int = 1", TypeArgumentsError, f"{array}[E]"
        ),
        f"{applied}-method": rejected(f"def {applied}::m(self) -> int = 1", AglScopeError, applied),
        f"{applied}-static": rejected(
            f"def {applied}::f() -> int = 1", TypeArgumentsError, applied
        ),
    }


_SCENARIOS["declared-beneath-an-alias-of-a-builtin-type"] = Scenario(
    modules={"bi": _BUILTIN_TARGETS},
    header=("import bi::*", "type T2 = text\ntype Arr[E] = array[E]\ntype IA = array[int]"),
    probes={
        **_builtin_head_probes("text", "array", "array[int]"),
        **_builtin_head_probes("T2", "Arr", "IA"),
        **_builtin_head_probes("U2", "Brr", "IB"),
        "own-by-alias": accepted('def text::m(self) -> int = 1\nT2::m("a")', "int"),
        "renaming-alias-of-an-alias": accepted(
            "type A4[Y] = Arr[Y]\ndef A4[Z]::m(self) -> int = 1\n[1].m()", "int"
        ),
        "applied-alias-of-an-alias": rejected(
            "type A3 = Arr[int]\ndef A3::m(self) -> int = 1", AglScopeError, "A3"
        ),
        "arguments-within-a-function-type": rejected(
            "type Fn[X] = array[(bi::U2, Brr[X]) -> X]\ndef Fn[E]::m(self) -> int = 1",
            AglScopeError,
            "Fn[E]",
        ),
        "one-argument-twice": rejected(
            "type Same[E] = dict[E, E]\ndef Same[E]::m(self) -> int = 1", AglScopeError, "Same[E]"
        ),
        "applied-scalar": rejected(
            "type P[X] = text\ndef P[int]::m(self) -> int = 1", TypeArgumentsError, "P[int]"
        ),
        "no-receiver-scope": rejected("def unit::m(self) -> int = 1", AglScopeError, "self"),
        "no-receiver-scope-by-alias": rejected(
            "type Un = unit\ndef Un::m(self) -> int = 1", AglScopeError, "self"
        ),
        "too-many-arguments": rejected(
            "def Arr[E, F]::m(self) -> int = 1", TypeArgumentsError, "Arr[E, F]"
        ),
        "text-keyed": accepted('def TD[V]::m(self) -> int = 1\n{"a": 1}.m()', "int"),
        "arguments-in-target-order": accepted(
            'def Sw[V, K]::key(self, k: K) -> K = k\n{"a": 1}.key("b")', "text"
        ),
    },
)


_SCENARIOS |= {
    f"declared-beneath-an-alias-declared-through-an-alias-spelled-{name}": Scenario(
        modules={},
        header=(
            "record Base\n  x: int\nrecord Holder\n  h: int",
            f"scope {owner}\n  type Y = Holder\nend {owner}",
            f"type {owner}::Y::U = Base",
            'def Holder::U::k() -> text = "k"',
        ),
        probes={
            f"{spelling}-static": accepted(f"{spelling}::k()", "text")
            for spelling in ("Base", "Holder::U", f"{owner}::Y::U")
        },
    )
    for name, owner in (("before-the-path-beneath-it", "A"), ("after-the-path-beneath-it", "X"))
}


def _through(spelling: str) -> str:
    """A module aliasing ``base``'s ``Base`` as ``Geo``, declaring beneath *spelling*."""
    record = f"record {spelling}::R\n  z: int\n"
    return f"import base::*\ntype Geo = Base\n{_declarations(spelling)}\n{record}"


def _declared_elsewhere_probes(spelling: str) -> dict[str, Probe]:
    """What ``thr`` declares beneath ``base``'s ``Base``, reached through *spelling*."""
    return {
        f"{spelling}-call": accepted(f"{spelling}::m(Base(x = 1))", "int"),
        f"{spelling}-static": accepted(f"{spelling}::g()", "int"),
        f"{spelling}-region": accepted(f"{spelling}::h()", "int"),
        f"{spelling}-nested": accepted(f"{spelling}::Inner::k()", "int"),
        f"{spelling}-record": accepted(f"{spelling}::R(z = 1).z", "int"),
        f"{spelling}-record-type": accepted(
            f"(fn(p: {spelling}::R) => p.z)(Base::R(z = 1))", "int"
        ),
    }


_SCENARIOS |= {
    f"declared-elsewhere-through-{spelling}": Scenario(
        modules={**_MODULES, "thr": _through(spelling)},
        header=("import thr::*\nimport thr", "import base::*"),
        probes={
            "dot": accepted("Base(x = 1).m()", "int"),
            **{
                key: probe
                for reached in ("Base", "Geo", "thr::Base", "thr::Geo", "/thr::Geo")
                for key, probe in _declared_elsewhere_probes(reached).items()
            },
        },
    )
    for spelling in ("Base", "Geo")
}

_SCENARIOS["a-route-spelling-a-tail-path-reaches-own-declarations"] = Scenario(
    modules={
        "other": "record Base\n  x: int\ndef Base::h() -> int = 1\ndef f() -> int = 2\n",
        "via": "scope other\n  import other\n  export other\nend other\n",
    },
    header=("import via::*", "import other", 'type G = other::Base\ndef G::h() -> text = "own"'),
    probes={
        "value": accepted("other::f()", "int"),
        "own-wins": accepted("other::Base::h()", "text"),
    },
)

_BUILTIN_THROUGH = "type T2 = text\ndef T2::via() -> int = 1\ndef text::direct() -> int = 2\n"
"""Declares beneath ``text`` through an alias of it and directly."""

_SCENARIOS["declared-elsewhere-beneath-an-alias-of-a-builtin-type"] = Scenario(
    modules={"bt": _BUILTIN_THROUGH},
    header=("import bt::*\nimport bt", 'def text::own() -> text = "own"'),
    probes={
        **{
            f"{spelling}-{name}": accepted(f"{spelling}::{name}()", "int")
            for spelling in ("text", "T2", "bt::text", "bt::T2")
            for name in ("via", "direct")
        },
        "own-through-the-imported-alias": accepted("T2::own()", "text"),
        "own-is-not-routed": rejected("bt::T2::own()", UnknownMemberError, "bt::T2::own"),
    },
)


class TestDeclarationsThroughAliases:
    """Declarations through an alias, the file part and every REPL grouping."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)


@pytest.mark.parametrize("spelling", ["Base", "Geo"])
def test_info_describes_a_declaration_through_either_spelling(
    tmp_path: Path, spelling: str
) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        (*_OWN_ALIAS, "def Geo::g() -> int = 2"),
        {
            "info": info(
                f"{spelling}::g",
                f"{spelling}::g is a function.\nSignature:\n  def {spelling}::g() -> int\n"
                "Location: <repl>:1:1",
            )
        },
    )
