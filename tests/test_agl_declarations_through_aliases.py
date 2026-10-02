"""A declaration path written through an alias is its target's path.

``def Geo::m(self)``, ``def Geo::f()``, ``scope Geo`` and ``def
Geo::Inner::k()`` with ``type Geo = Base`` declare beneath ``Base``: both
spellings reach the declaration in every position, whichever spelling
declared it, the alias own or imported and its target own or imported. It is
keyed, displayed and retained at that path: two declarations of one name
beneath the two spellings are one path declared twice in one entry, and a
later REPL entry declaring it beneath either spelling supersedes it. A region
written through the alias opens its target's scope and no scope of the alias's
name, as a path written through it declares none.

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
    UnknownQualifierError,
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
_ALIASES = (
    ("an-own-alias", _OWN_ALIAS),
    ("an-imported-alias", _IMPORTED_ALIAS),
    ("an-alias-of-an-own-type", _LOCAL_TARGET),
)
"""Each way ``Geo`` aliases ``Base``: an own alias, an imported one, an own one of an own type."""


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

    *nested* renders the own nested record's type, at the path it is declared
    at whichever spelling declared it. Through each of *routes*, module routes
    or anchored, only the routed module's declarations are reached.
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
            probes=_beside_imported_probes("Base::Inner", routes),
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
        for name, header in _ALIASES
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
            **{
                f"own-static-read-through-{spelling}": rejected(
                    f"def Box::k() -> int = 1\n{spelling}::k()", TypeArgumentsError, spelling
                )
                for spelling in ("Box[int]", "IntBox", "IB2")
            },
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
        # A region's step reads the alias at the region's own path, where no type is.
        **{
            f"{alias}-region-method": rejected(
                f"scope S\n  def {alias}::m(self) -> int = 1\nend S", AglScopeError, "self"
            )
            for alias in ("T2", "U2")
        },
        **{
            f"own-read-by-{alias}-{name}": accepted(f"{declaration}\n{read}", identity)
            for alias in ("T2", "U2")
            for name, declaration, read, identity in (
                ("static", "def text::g() -> int = 1", f"{alias}::g()", "int"),
                ("static-value", "def text::g() -> int = 1", f"{alias}::g", "() -> int"),
                ("method", "def text::m(self) -> int = 1", f'{alias}::m("a")', "int"),
                ("method-value", "def text::m(self) -> int = 1", f"{alias}::m", "text -> int"),
                (
                    "type",
                    "record text::R\n  x: int",
                    f"fn(r: {alias}::R) => r.x",
                    "text::R -> int",
                ),
            )
        },
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

_SCENARIOS["declared-at-a-path-an-own-type-of-the-target-name-declares"] = Scenario(
    modules={"other": "record Base\n  y: int\ndef Base::k() -> int = 1\n"},
    header=("import other", "record Base\n  x: int", "type G = other::Base"),
    probes={
        "alias-then-own": rejected(
            'def G::h() -> text = "g"\ndef Base::h() -> int = 1',
            DuplicateDeclarationError,
            "def Base::h() -> int = 1",
        ),
        "own-then-alias": rejected(
            'def Base::h() -> int = 1\ndef G::h() -> text = "g"',
            DuplicateDeclarationError,
            'def G::h() -> text = "g"',
        ),
        "records": rejected(
            "record G::R\nrecord Base::R", DuplicateDeclarationError, "record Base::R"
        ),
        "through-the-alias": accepted('def G::h() -> text = "g"\nG::h()', "text"),
        "through-the-own-type": accepted('def G::h() -> text = "g"\nBase::h()', "text"),
        "target-declaration": accepted("G::k()", "int"),
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

_SCENARIOS["declared-beneath-a-structural-alias"] = Scenario(
    modules={"funcs": "type G = int -> bool\n"},
    header=("import funcs::*", "type F = int -> bool"),
    probes={
        f"{spelling}-{name}": rejected(text, AglScopeError, spelling)
        for spelling in ("F", "G")
        for name, text in {
            "static": f"def {spelling}::m() -> int = 1",
            "method": f"def {spelling}::m(self) -> int = 1",
            "nested": f"def {spelling}::In::k() -> int = 1",
            "record": f"record {spelling}::R\n  z: int",
            "region": f"scope {spelling}\n  def z() -> int = 1\nend {spelling}",
            "nested-region": f"scope {spelling}::In\n  def z() -> int = 1\nend {spelling}::In",
        }.items()
    },
)

_SCENARIOS["a-method-beneath-an-alias-dropping-its-parameter"] = Scenario(
    modules={"ph": "record Box\n  v: int\ntype G[T] = Box\n"},
    header=("import ph::*", "type F[T] = Box"),
    probes={
        spelling: rejected(f"def {spelling}::m(self) -> int = 1", AglScopeError, "self")
        for spelling in ("F", "G")
    },
)

_BENEATH_IB = {
    "static": "def IB::z() -> int = 1",
    "nested": "def IB::In::k() -> int = 1",
    "region": "scope IB\n  def z() -> int = 1\nend IB",
}
"""Declarations beneath ``IB``, written before ``IB`` is declared."""
_HOSTING_NOTHING = {
    "applied": ("type IB = gen::Box[int]", TypeArgumentsError),
    "structural": ("type IB = int -> int", AglScopeError),
}
"""Aliases ``IB`` beneath which nothing is declared, and the error a declaration there is."""

_SCENARIOS["declared-before-an-alias-hosting-nothing"] = Scenario(
    modules=_APPLIED_MODULES,
    header=("import gen",),
    probes={
        f"{name}-{kind}": rejected(f"{beneath}\n{alias}", error, "IB")
        for name, beneath in _BENEATH_IB.items()
        for kind, (alias, error) in _HOSTING_NOTHING.items()
    },
)

_SCENARIOS |= {
    "let-declared-through-an-own-alias": Scenario(
        header=("record Base\n  x: int", "type Geo = Base", "let Geo::v = 1"),
        probes={
            "base-read": accepted("Base::v", "int"),
            "geo-read": accepted("Geo::v", "int"),
        },
    ),
    "scope-regions-bare-member-through-an-own-alias-reaches-its-own-keyed-sibling": Scenario(
        header=(
            "record Base\n  x: int",
            "type Geo = Base",
            "scope Geo\n  def g() -> int = 2\n  def h() -> int = g() + 1\nend Geo",
        ),
        probes={
            f"{spelling}-h": accepted(f"{spelling}::h()", "int") for spelling in ("Base", "Geo")
        },
    ),
}

_REDECLARED = {
    "": 'def Geo::f() -> text = "own"',
    "-in-a-region": 'scope Geo\n  def f() -> text = "own"\nend Geo',
}
"""``base``'s ``Base::f`` declared again through ``Geo``, by a shorthand path and in a region."""

_SCENARIOS |= {
    f"an-anchored-qualifier-reaches-{name}-redeclared-through-it{form}": Scenario(
        modules=_MODULES,
        header=(*header, redeclared),
        probes={
            # An own alias is an own name: ``::Geo`` reads it like any other.
            "anchored-alias-accepted": accepted("::Geo::f()", "text"),
            "anchored-alias-unknown-member-rejected": rejected(
                "::Geo::zz()", UnknownMemberError, "::Geo::zz"
            ),
            "anchored-target-accepted": accepted("::Base::f()", "text"),
        },
    )
    for name, header in (("an-own-alias", _OWN_ALIAS), ("an-alias-of-an-own-type", _LOCAL_TARGET))
    for form, redeclared in _REDECLARED.items()
}

_SCENARIOS |= {
    f"an-anchored-qualifier-never-reaches-an-imported-alias-redeclared-through-it{form}": Scenario(
        modules=_MODULES,
        header=(*_IMPORTED_ALIAS, redeclared),
        probes={
            "anchored-alias-rejected": rejected("::Geo::f()", UnknownQualifierError, "::Geo::f"),
            "anchored-alias-unknown-member-rejected": rejected(
                "::Geo::zz()", UnknownQualifierError, "::Geo::zz"
            ),
            "anchored-nested-alias-rejected": rejected(
                "::Geo::Inner(y = 1)", UnknownQualifierError, "::Geo::Inner"
            ),
            "anchored-target-accepted": accepted("::Base::f()", "text"),
        },
    )
    for form, redeclared in _REDECLARED.items()
}

_SCENARIOS |= {
    f"an-anchored-qualifier-still-reaches-its-own-declaration-beside-a-routed-one{form}": Scenario(
        modules=_MODULES,
        header=(
            "import base",
            *_IMPORTED_ALIAS,
            redeclared,
            'def Geo::g() -> text = "g"',
        ),
        probes={
            "base-static": accepted("Base::f()", "text"),
            "geo-static": accepted("Geo::f()", "text"),
            "anchored-target": accepted("::Base::f()", "text"),
            "anchored-alias-rejected": rejected("::Geo::f()", UnknownQualifierError, "::Geo::f"),
            "base-g": accepted("Base::g()", "text"),
            "routed-target-unaffected": accepted("base::Base::f()", "int"),
        },
    )
    for form, redeclared in _REDECLARED.items()
}


def _beside_ordinary_probes(spelling: str) -> dict[str, Probe]:
    """A member declared beneath *spelling* of the imported alias, beside an ordinary declaration.

    Only the declared path ``Base`` is the module's own: an ordinary ``Geo``
    never clashes, an ordinary ``Base`` always does, in either order, whether
    a shorthand path or a region declares the member.
    """
    member = f"def {spelling}::g() -> int = 1"
    nested = f"def {spelling}::Inner::k() -> int = 4"
    region = f"scope {spelling}\n  def g() -> int = 1\nend {spelling}"
    empty = f"scope {spelling}\nend {spelling}"
    return {
        f"{spelling}-method-then-alias-spelling": accepted(
            f"{member}\ndef Geo() -> int = 2\n1", "int"
        ),
        f"{spelling}-alias-spelling-then-method": accepted(
            f"def Geo() -> int = 2\n{member}\n1", "int"
        ),
        f"{spelling}-method-then-let": accepted(f"{member}\nlet Geo = 2\n1", "int"),
        f"{spelling}-let-then-method": accepted(f"let Geo = 2\n{member}\n1", "int"),
        f"{spelling}-nested-then-alias-spelling": accepted(
            f"{nested}\ndef Geo() -> int = 2\n1", "int"
        ),
        f"{spelling}-method-then-target-spelling": rejected(
            f"{member}\ndef Base() -> int = 2\n1",
            DuplicateDeclarationError,
            "def Base() -> int = 2",
        ),
        f"{spelling}-target-spelling-then-method": rejected(
            f"def Base() -> int = 2\n{member}\n1", DuplicateDeclarationError, member
        ),
        f"{spelling}-nested-then-target-spelling": rejected(
            f"{nested}\ndef Base() -> int = 2\n1",
            DuplicateDeclarationError,
            "def Base() -> int = 2",
        ),
        f"{spelling}-region-then-alias-spelling": accepted(
            f"{region}\ndef Geo() -> int = 2\n1", "int"
        ),
        f"{spelling}-alias-spelling-then-region": accepted(
            f"def Geo() -> int = 2\n{region}\n1", "int"
        ),
        f"{spelling}-empty-region-then-alias-spelling": accepted(
            f"{empty}\ndef Geo() -> int = 2\n1", "int"
        ),
        f"{spelling}-region-then-target-spelling": rejected(
            f"{region}\ndef Base() -> int = 2\n1",
            DuplicateDeclarationError,
            "def Base() -> int = 2",
        ),
        f"{spelling}-target-spelling-then-region": rejected(
            f"def Base() -> int = 2\n{region}\n1", DuplicateDeclarationError, region
        ),
        f"{spelling}-empty-region-then-target-spelling": rejected(
            f"{empty}\ndef Base() -> int = 2\n1",
            DuplicateDeclarationError,
            "def Base() -> int = 2",
        ),
    }


_SCENARIOS["declared-through-an-imported-alias-beside-an-ordinary-declaration"] = Scenario(
    modules=_MODULES,
    header=_IMPORTED_ALIAS,
    probes=_beside_ordinary_probes("Geo") | _beside_ordinary_probes("Base"),
)

_SCENARIOS |= {
    f"{form}-through-an-imported-alias-{order}-an-ordinary-declaration-of-its-name": Scenario(
        modules=_MODULES,
        header=(*_IMPORTED_ALIAS, *entries),
        probes={
            "member": accepted("Base::g()", "int"),
            "ordinary": accepted("Geo()", "int"),
            "anchored-ordinary": accepted("::Geo()", "int"),
        },
    )
    for form, member in (
        ("a-path", "def Geo::g() -> int = 1"),
        ("a-region", "scope Geo\n  def g() -> int = 1\nend Geo"),
        ("an-empty-region", "scope Geo\nend Geo\n\ndef Base::g() -> int = 1"),
    )
    for order, entries in (
        ("before", (member, "def Geo() -> int = 2")),
        ("after", ("def Geo() -> int = 2", member)),
    )
}

_SCENARIOS["an-own-alias-clashes-with-an-ordinary-declaration-of-its-name"] = Scenario(
    modules=_MODULES,
    header=("import base::*",),
    probes={
        "alias-method-then-ordinary": rejected(
            "type Geo = Base\ndef Geo::g() -> int = 1\ndef Geo() -> int = 2\n1",
            DuplicateDeclarationError,
            "def Geo() -> int = 2",
        ),
        "ordinary-then-alias-method": rejected(
            "def Geo() -> int = 2\ntype Geo = Base\ndef Geo::g() -> int = 1\n1",
            DuplicateDeclarationError,
            "type Geo = Base",
        ),
        "alias-region-then-ordinary": rejected(
            "type Geo = Base\n\n"
            "scope Geo\n  def g() -> int = 1\nend Geo\n\n"
            "def Geo() -> int = 2\n1",
            DuplicateDeclarationError,
            "def Geo() -> int = 2",
        ),
        "ordinary-then-alias-region": rejected(
            "def Geo() -> int = 2\ntype Geo = Base\n\n"
            "scope Geo\n  def g() -> int = 1\nend Geo\n\n1",
            DuplicateDeclarationError,
            "type Geo = Base",
        ),
    },
)

_SHAPES = {
    "functions": (
        (
            "def Geo::g() -> int = 7\n"
            "def Geo::k() -> int = g()\n"
            "def Base::j() -> int = g()\n"
            "def Geo::fact(n: int) -> int = if n <= 1 => 1 else => n * fact(n - 1)"
        ),
        ("k()", "j()", "fact(3)"),
    ),
    "bindings": (
        (
            "let Geo::v = 2\n"
            "var Geo::w = 3\n"
            "\n"
            "scope Geo\n  let rv = 4\n  var rw = 5\n  def r() -> int = rv + rw + v + w\nend Geo"
        ),
        ("v", "w", "rv", "rw", "r()"),
    ),
    "types": (
        (
            "scope Geo\n"
            "  enum E\n"
            "    | A\n"
            "    | B\n\n"
            "  record R\n"
            "    a: int\n\n"
            "  def k(e: Base::E) -> int = case e of\n"
            "    | A => 1\n"
            "    | B => 2\n\n"
            "  def m(r: R) -> int = r.a\n\n"
            "  def n() -> int = m(R(a = 3)) + k(A)\n"
            "end Geo"
        ),
        ("k(Geo::E::B)", "k(Base::E::A)", "m(Geo::R(a = 1))", "m(Base::R(a = 2))", "n()"),
    ),
}
"""Declarations through ``Geo``, each reading its siblings bare, and the reads after them."""

_SCENARIOS |= {
    f"{shape}-declared-through-{name}-read-their-siblings-bare": Scenario(
        modules=_MODULES,
        header=(*header, declarations),
        probes={
            f"{spelling}-{read}": accepted(f"{spelling}::{read}", "int")
            for spelling in ("Base", "Geo")
            for read in reads
        },
    )
    for shape, (declarations, reads) in _SHAPES.items()
    for name, header in _ALIASES
}

_REGION_MODULES = {
    **_MODULES,
    "gen": _APPLIED_MODULES["gen"],
    "lib": "scope S\n  def top() -> int = 5\nend S\n\ndef free() -> int = 6\n",
}


def _region_probes(spelling: str) -> dict[str, Probe]:
    """A region spelled *spelling* opens ``Base``: what it writes is written there.

    Its uses and imports, its nested regions, and the heads of the
    declarations in it read as in the region the target's spelling opens.
    """

    def region(body: str, then: str) -> str:
        return f"scope {spelling}\n{body}\nend {spelling}\n{then}"

    nested = "  scope N\n    def k() -> int = 4\n  end N"
    member = "  enum E\n    | A(a: int)\n    | B\n\n  def A::m(self) -> int = self.a"
    return {
        f"{spelling}-empty": accepted(f"scope {spelling}\nend {spelling}\n1", "int"),
        f"{spelling}-empty-nested": accepted(
            f"scope {spelling}::Deep\nend {spelling}::Deep\n1", "int"
        ),
        f"{spelling}-use": accepted(
            region("  use lib::S\n  def h() -> int = S::top()", "Base::h()"), "int"
        ),
        f"{spelling}-import": accepted(
            region("  import lib::*\n  def h() -> int = free()", "Geo::h()"), "int"
        ),
        **{
            f"{spelling}-use-read-beneath-{other}": accepted(
                region("  use lib::S", f"def {other}::h() -> int = S::top()\nBase::h()"), "int"
            )
            for other in ("Base", "Geo")
        },
        **{
            f"{spelling}-nested-read-by-{name}": accepted(region(nested, read), "int")
            for name, read in (
                ("target", "Base::N::k()"),
                ("alias", "Geo::N::k()"),
                ("anchored-target", "::Base::N::k()"),
            )
        },
        f"{spelling}-method": accepted(
            region("  def m(self) -> int = self.x", "Base(x = 1).m()"), "int"
        ),
        f"{spelling}-nested-method": accepted(
            region(
                "  scope Inner\n    def m(self) -> int = self.y\n  end Inner",
                "Base::Inner(y = 1).m()",
            ),
            "int",
        ),
        f"{spelling}-member-receiver": accepted(region(member, "Base::E::A(a = 1).m()"), "int"),
        f"{spelling}-unknown-receiver": rejected(
            region("  def A::m(self) -> int = 1", "1"), AglScopeError, "self"
        ),
        f"{spelling}-unknown-receiver-path": rejected(
            region("  scope Inner\n    def A::B::m(self) -> int = 1\n  end Inner", "1"),
            UnknownMemberError,
            "A::B",
        ),
        f"{spelling}-builtin-alias-receiver": rejected(
            region("  def T2::m(self) -> int = 1", "1"), AglScopeError, "self"
        ),
    }


_SCENARIOS |= {
    f"a-region-through-{name}-opens-its-target": Scenario(
        modules=_REGION_MODULES,
        header=("import lib", *header, "type T2 = text"),
        probes=_region_probes("Base") | _region_probes("Geo"),
    )
    for name, header in _ALIASES
}

_SCENARIOS |= {
    f"declared-beneath-an-applied-alias-declared-beneath-{name}": Scenario(
        modules=_REGION_MODULES,
        header=("import gen::*", *header, "type Base::IB = Box[int]"),
        probes={
            f"{spelling}-{position}": rejected(text, TypeArgumentsError, "IB")
            for spelling in ("Base", "Geo")
            for position, text in {
                "static": f"def {spelling}::IB::k() -> int = 1",
                "method": f"def {spelling}::IB::m(self) -> int = 1",
                "record": f"record {spelling}::IB::R\n  z: int",
                "in-a-region": f"scope {spelling}\n  def IB::k() -> int = 1\nend {spelling}",
                "region": f"scope {spelling}::IB\n  def k() -> int = 1\nend {spelling}::IB",
            }.items()
        },
    )
    for name, header in _ALIASES
}

_SCENARIOS |= {
    f"an-empty-region-through-{spelling}-of-an-imported-alias-is-the-targets-own-path": Scenario(
        modules=_MODULES,
        header=(*_IMPORTED_ALIAS, f"scope {spelling}\nend {spelling}"),
        probes={
            "anchored-target-is-own": rejected("::Base::zz()", UnknownMemberError, "::Base::zz"),
            "anchored-alias-is-not": rejected("::Geo::zz()", UnknownQualifierError, "::Geo::zz"),
            "ordinary-target-spelling": rejected(
                "def Base() -> int = 2\n1", DuplicateDeclarationError, "def Base() -> int = 2"
            ),
            "ordinary-alias-spelling": accepted("def Geo() -> int = 2\nGeo()", "int"),
        },
    )
    for spelling in ("Base", "Geo")
}


def _exported_probes(member: str | None) -> dict[str, Probe]:
    """What ``thr`` exports beneath ``Base``, however it spelled the path it declared at."""
    return {
        "alias-spelling-is-no-scope": rejected("Geo::zz()", UnknownQualifierError, "Geo::zz"),
        "alias-spelling-is-not-exported": rejected(
            "use thr::Geo\n1", UnknownMemberError, "use thr::Geo"
        ),
        "target-spelling-is-exported": accepted("use thr::Base\n1", "int"),
        **({} if member is None else {"member": accepted(f"Base::{member}()", "int")}),
    }


_SCENARIOS |= {
    f"{form}-through-{spelling}-elsewhere-exports-the-targets-scope": Scenario(
        modules={**_REGION_MODULES, "thr": f"import al::*\n{declared.format(spelling)}\n"},
        header=("import thr::*\nimport thr",),
        probes=_exported_probes(member),
    )
    for spelling in ("Base", "Geo")
    for form, declared, member in (
        ("a-path", "def {0}::h() -> int = 3", "h"),
        ("a-region", "scope {0}\n  def h() -> int = 3\nend {0}", "h"),
        ("an-empty-region", "scope {0}\nend {0}", None),
        ("an-export-in-a-region", "scope {0}\n  export lib::{{free}}\nend {0}", "free"),
    )
}

_SCENARIOS["a-method-beneath-an-imported-alias-reads-no-enum-member-injected-above-it"] = Scenario(
    modules=_MODULES,
    header=(*_IMPORTED_ALIAS, "enum Color\n  | T3(a: int)\n  | Blue", "type Base::T3 = text"),
    probes={
        f"through-{spelling}": rejected(
            f"def {spelling}::T3::m(self) -> int = 1", AglScopeError, "self"
        )
        for spelling in ("Base", "Geo")
    },
)


_NESTED_ALIAS = "scope Zz\n  type Gb = Base\nend Zz"
"""An alias of ``Base`` whose scope path sorts after every path beneath ``Ga`` or ``S``."""
_BOTH_ALIASES = ("Ga", "Base", "Zz::Gb")

_SCENARIOS |= {
    "a-path-through-one-alias-reads-a-type-declared-through-a-later-spelled-one": Scenario(
        modules=_MODULES,
        header=(
            "import base::*",
            "type Ga = Base",
            f"{_NESTED_ALIAS}\nrecord Zz::Gb::T\n  y: int",
            "def Ga::T::k() -> int = 4\ndef Ga::T::U::j() -> int = 5",
        ),
        probes={
            f"{spelling}-{name}": accepted(f"{spelling}::{read}", "int")
            for spelling in _BOTH_ALIASES
            for name, read in (
                ("beneath", "T::k()"),
                ("nested", "T::U::j()"),
                ("type", "T(y = 1).y"),
            )
        },
    ),
    "paths-through-two-aliases-beneath-one-nested-type-are-one-path": Scenario(
        modules=_MODULES,
        header=(
            "import base::*",
            "type Ga = Base",
            _NESTED_ALIAS,
            "def Zz::Gb::Inner::j() -> int = 5\ndef Ga::Inner::k() -> int = 4",
        ),
        probes={
            f"{spelling}-{name}": accepted(f"{spelling}::Inner{read}", "int")
            for spelling in _BOTH_ALIASES
            for name, read in (("first", "::j()"), ("second", "::k()"), ("type", "(y = 1).y"))
        },
    ),
    "a-used-path-reads-a-type-declared-through-a-later-spelled-alias": Scenario(
        modules=_MODULES,
        header=(
            "import base::*",
            f"{_NESTED_ALIAS}\nrecord Zz::Gb::T\n  y: int",
            "scope S\n  use Base::{T}\n  def make() -> T = T(y = 3)\nend S",
            "def S::T::k() -> int = 4",
        ),
        probes={
            "target": accepted("Base::T::k()", "int"),
            "alias": accepted("Zz::Gb::T::k()", "int"),
            "inside-the-use": accepted("S::make().y", "int"),
            # The use exposes ``T`` inside ``S`` alone: no path of ``S`` spells it.
            "used-path": rejected("S::T::k()", UnknownMemberError, "S::T::k"),
            "used-constructor": rejected("S::T(y = 1)", UnknownMemberError, "S::T"),
        },
    ),
}

_MANY = 12
_MANY_ALIASES = "\n".join(
    (*(f"type A{i} = Base" for i in range(_MANY - 1)), f"type A{_MANY - 1} = A{_MANY - 2}")
)
"""Aliases of one type, the last through the one before."""
_MANY_DECLARED = "\n".join(
    (
        "def A0::f0() -> int = 0",
        *(f"def A{i}::f{i}() -> int = A{i - 1}::f{i - 1}() + 1" for i in range(1, _MANY)),
    )
)
"""One function beneath each alias, calling the one beneath the alias before it."""

_SCENARIOS["declared-through-many-aliases-of-one-type"] = Scenario(
    header=(
        "record Base\n  x: int",
        _MANY_ALIASES,
        _MANY_DECLARED,
        f"def A{_MANY - 1}::Inner::k() -> int = 1\ndef A2::Inner::j() -> int = A0::Inner::k()\n"
        "record A3::Inner::T\n  y: int",
    ),
    probes={
        "target-reaches-the-last": accepted(f"Base::f{_MANY - 1}()", "int"),
        "first-reaches-the-last": accepted(f"A0::f{_MANY - 1}()", "int"),
        "last-reaches-the-first": accepted(f"A{_MANY - 1}::f0()", "int"),
        "nested-through-the-target": accepted("Base::Inner::k()", "int"),
        "nested-through-another": accepted(f"A{_MANY - 1}::Inner::j()", "int"),
        "nested-type": accepted("A1::Inner::T(y = 1).y", "int"),
        "nested-type-position": accepted(
            "(fn(p: A2::Inner::T) => p.y)(Base::Inner::T(y = 2))", "int"
        ),
        "undeclared": rejected("A4::zz()", UnknownMemberError, "A4::zz"),
    },
)

_CHAIN = 8
_CHAIN_HALVES = tuple(
    "\n".join(f"type C{i} = {'Base' if i == 0 else f'C{i - 1}'}" for i in half)
    for half in (range(_CHAIN // 2), range(_CHAIN // 2, _CHAIN))
)
"""A chain of aliases, each of the one before, the first of ``Base``; written in two halves."""
_LAST = f"C{_CHAIN - 1}"

_SCENARIOS["declared-through-a-chain-of-aliases"] = Scenario(
    header=(
        "record Base\n  x: int",
        *_CHAIN_HALVES,
        "\n".join(f"def C{i}::g{i}() -> int = {i}" for i in range(_CHAIN)),
    ),
    probes={
        "target-reaches-the-last": accepted(f"Base::g{_CHAIN - 1}()", "int"),
        "first-reaches-the-last": accepted(f"C0::g{_CHAIN - 1}()", "int"),
        "last-reaches-the-first": accepted(f"{_LAST}::g0()", "int"),
        "declared-through-the-last": accepted(f"def {_LAST}::h() -> int = 7\nC0::h()", "int"),
        "nested-through-the-last": accepted(
            f"def {_LAST}::Inner::k() -> int = 8\nC1::Inner::k()", "int"
        ),
        "value": accepted(f"{_LAST}(x = 1).x", "int"),
        "undeclared": rejected(f"{_LAST}::zz()", UnknownMemberError, f"{_LAST}::zz"),
    },
)

_CYCLE = {
    "ca": (
        "import cb::*\nrecord Ra\n  x: int\ntype Tb = Rb\n"
        "def Tb::fa() -> int = 1\ndef Tb::Inner::ka() -> int = 2\n"
    ),
    "cb": (
        "import ca::*\nrecord Rb\n  y: int\ntype Ta = Ra\n"
        "def Ta::fb() -> int = 3\ndef Ta::Inner::kb() -> int = 4\n"
    ),
}
"""Two modules importing each other, each declaring beneath an alias of the other's type."""

_SCENARIOS["declared-through-aliases-of-types-of-modules-importing-each-other"] = Scenario(
    modules=_CYCLE,
    header=(
        "import ca::*\nimport cb::*",
        "type Xa = Ta\ntype Xb = Tb",
        "def Xa::g() -> int = 5\ndef Xb::Inner::h() -> int = 6",
    ),
    probes={
        "one-through-its-target": accepted("Rb::fa()", "int"),
        "other-through-its-target": accepted("Ra::fb()", "int"),
        "one-nested": accepted("Tb::Inner::ka()", "int"),
        "other-nested-through-an-own-alias": accepted("Xa::Inner::kb()", "int"),
        "own-through-its-target": accepted("Ra::g()", "int"),
        "own-through-an-imported-alias": accepted("Ta::g()", "int"),
        "own-nested-through-its-target": accepted("Rb::Inner::h()", "int"),
        "one-nested-through-an-own-alias": accepted("Xb::Inner::ka()", "int"),
        "beneath-the-other-type": rejected("Xa::fa()", UnknownMemberError, "Xa::fa"),
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


@pytest.mark.parametrize(("first", "second"), [("Base", "Geo"), ("Geo", "Base")])
@pytest.mark.parametrize(
    "header",
    [_OWN_ALIAS, _IMPORTED_ALIAS, _LOCAL_TARGET],
    ids=["an-own-alias", "an-imported-alias", "an-alias-of-an-own-type"],
)
def test_a_later_entry_beneath_the_other_spelling_supersedes(
    tmp_path: Path, header: tuple[str, ...], first: str, second: str
) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        (
            *header,
            f"def {first}::g() -> int = 1\ndef {first}::m(self) -> int = 1\nrecord {first}::R",
            f'def {second}::g() -> text = "2"\ndef {second}::m(self) -> text = "2"\n'
            f"record {second}::R\n  z: int",
        ),
        {
            **{
                f"{spelling}-{position}": probe
                for spelling in ("Base", "Geo")
                for position, probe in {
                    "static": accepted(f"{spelling}::g()", "text"),
                    "method": accepted(f"{spelling}::m(Base(x = 1))", "text"),
                    "record": accepted(f"{spelling}::R(z = 1)", "record Base::R\n  z: int"),
                }.items()
            },
            "info": info(
                "Geo::R",
                "Geo::R is a record type.\nType:\n  record Base::R\n    z: int\n"
                "Location: <repl>:3:1",
            ),
        },
    )


@pytest.mark.parametrize(
    "header",
    [_OWN_ALIAS, _LOCAL_TARGET],
    ids=["an-own-alias", "an-alias-of-an-own-type"],
)
def test_rebinding_the_alias_leaves_earlier_declarations_at_its_old_target(
    tmp_path: Path, header: tuple[str, ...]
) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        (
            *header,
            "def Geo::g() -> int = 1\nrecord Geo::R",
            "record Other\n  w: int",
            "type Geo = Other",
        ),
        {
            "static": accepted("Base::g()", "int"),
            "record": accepted("Base::R()", "record Base::R"),
            "through-the-new-target": rejected("Geo::g()", UnknownMemberError, "Geo::g"),
            "info": info(
                "Base::R",
                "Base::R is a record type.\nType:\n  record Base::R\nLocation: <repl>:2:1",
            ),
        },
    )


@pytest.mark.parametrize(
    "binding",
    ["let Geo::v = 2", "var Geo::v = 2", "scope Geo\n  let v = 2\nend Geo"],
    ids=["let", "var", "a-region"],
)
@pytest.mark.parametrize(
    "header",
    [_OWN_ALIAS, _LOCAL_TARGET],
    ids=["an-own-alias", "an-alias-of-an-own-type"],
)
def test_rebinding_the_alias_leaves_a_binding_declared_through_it_at_its_old_target(
    tmp_path: Path, header: tuple[str, ...], binding: str
) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        (
            *header,
            binding,
            "record Other\n  w: int",
            "type Geo = Other",
        ),
        {
            "static": accepted("Base::v", "int"),
            "through-the-new-target": rejected("Geo::v", UnknownMemberError, "Geo::v"),
        },
    )


@pytest.mark.parametrize(
    "binding",
    ["let Base::v = 2", "var Geo::v = 2", "scope Geo\n  let v = 2\nend Geo"],
    ids=["let", "var", "a-region"],
)
def test_an_earlier_entrys_alias_reaches_a_binding_its_path_was_read_hidden_before(
    tmp_path: Path, binding: str
) -> None:
    """Until an entry binds ``v``, its reads through the alias find only the path a use hides."""
    assert_repl_verdicts(
        tmp_path,
        {},
        (
            "use Sc::* hiding Base::v\n\nscope Sc\n  def Base::v() -> int = 1\nend Sc",
            "record Base\n  x: int",
            "type Geo = Base",
        ),
        {
            "through-the-alias": accepted(f"{binding}\nGeo::v", "int"),
            "through-the-target": accepted(f"{binding}\nBase::v", "int"),
            "unbound": rejected("Geo::v", UnknownMemberError, "Geo::v"),
        },
    )


@pytest.mark.parametrize("beneath", list(_BENEATH_IB.values()), ids=list(_BENEATH_IB))
def test_an_alias_hosting_nothing_rejects_earlier_entries_beneath_its_path(
    tmp_path: Path, beneath: str
) -> None:
    """Declarations an earlier entry made at its path are beneath it, as in one entry."""
    assert_repl_verdicts(
        tmp_path,
        _APPLIED_MODULES,
        ("import gen", beneath),
        {kind: rejected(alias, error, alias) for kind, (alias, error) in _HOSTING_NOTHING.items()},
    )
