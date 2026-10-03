"""A declaration is at its written path: an alias declares no scope.

Declaring beneath a name the module itself declares as an alias -- ``scope
Geo``, ``def Geo::k``, ``let Geo::v``, ``record Geo::R`` and a region's
imports with an own ``type Geo = ...`` -- is an error in either order, whatever
the alias names, rejected at whichever of the two comes later (in the REPL,
the entry completing the pair). Beneath an imported alias's name an own
declaration stands at its written path: ``Geo::x`` then reads it beside what
the alias reaches beneath its target, the own one winning, and the target's
spelling never reads it. A method receiver selected through any alias is an
error. Reading through an alias stays transparent.

Every probe is checked in the file part and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglError, AglTypeError, HiddenMemberError
from agm.agl.repl.session import ReplSession
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousQualificationError,
    DuplicateDeclarationError,
    TypeArgumentsError,
    UnknownMemberError,
    UnknownQualifierError,
)
from tests._agl_helpers import repl_session_with_root
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
_MODULES = {
    "base": _BASE,
    "al": "import base::*\nexport base::{Base}\ntype Geo = Base\n",
    "gen": "record Box[T]\n  v: T\n",
    "lib": "scope S\n  def top() -> int = 5\nend S\n\ndef free() -> int = 6\n",
}

_OWN_ALIASES = {
    "of-an-imported-type": ("import base::*\nimport lib", "type Geo = Base"),
    "of-an-own-type": ("import lib\nrecord Base\n  x: int", "type Geo = Base"),
    "of-a-routed-type": ("import base\nimport lib", "type Geo = base::Base"),
    "of-an-imported-alias": ("import al\nimport lib", "type Geo = al::Geo"),
    "through-an-own-alias": ("import base::*\nimport lib\ntype Mid = Base", "type Geo = Mid"),
    "of-a-builtin-type": ("import lib", "type Geo = text"),
    "generic-of-a-builtin-type": ("import lib", "type Geo[E] = array[E]"),
    "of-an-applied-type": ("import gen::*\nimport lib", "type Geo = Box[int]"),
    "of-a-structural-type": ("import lib", "type Geo = int -> bool"),
}
"""Each way a module declares ``Geo`` as an alias: its setup, then the alias."""

_BENEATH = {
    "static": "def Geo::k() -> int = 1",
    "method": "def Geo::m(self) -> int = 1",
    "nested": "def Geo::In::k() -> int = 1",
    "let": "let Geo::v = 1",
    "var": "var Geo::v = 1",
    "record": "record Geo::R\n  z: int",
    "enum": "enum Geo::E\n  | A\n  | B",
    "alias": "type Geo::A = int",
    "region": "scope Geo\n  def h() -> int = 1\nend Geo",
    "nested-region": "scope Geo::In\n  def h() -> int = 1\nend Geo::In",
    "region-import": "scope Geo\n  import lib::*\nend Geo",
    "region-use": "scope Geo\n  use lib::S\nend Geo",
    "region-binding": "scope Geo\n  let v = 1\nend Geo",
    "empty-region": "scope Geo\nend Geo",
}
"""Each way to declare beneath ``Geo``: every declaration kind, and regions."""

_SCENARIOS: dict[str, Scenario] = {
    f"declaring-beneath-an-own-alias-{kind}": Scenario(
        modules=_MODULES,
        header=header,
        probes={
            **{
                position: rejected(text, AglScopeError, "Geo")
                for position, text in _BENEATH.items()
            },
            "in-a-region-of-another-name": accepted(
                "scope S\n  def Geo::k() -> int = 1\nend S\n\nS::Geo::k()", "int"
            ),
            "beside-it": accepted("def Gx::k() -> int = 1\nGx::k()", "int"),
        },
    )
    for kind, header in _OWN_ALIASES.items()
} | {
    f"an-own-alias-after-a-declaration-beneath-its-name-{kind}": Scenario(
        modules=_MODULES,
        header=(setup,),
        probes={
            position: rejected(f"{text}\n{alias}", AglScopeError, alias)
            for position, text in _BENEATH.items()
        },
    )
    for kind, (setup, alias) in _OWN_ALIASES.items()
}

_SCENARIOS |= {
    f"an-own-alias-after-an-entry-declaring-beneath-its-name-{position}": Scenario(
        modules=_MODULES,
        header=(_OWN_ALIASES["of-an-imported-type"][0], text),
        probes={"alias": rejected("type Geo = Base", AglScopeError, "type Geo = Base")},
        # A method entry beneath no type is itself rejected.
        legal=frozenset({(1, 2), (3,)}) if position == "method" else "ALL",
    )
    for position, text in _BENEATH.items()
} | {
    f"an-own-alias-{kind}-after-an-entry-declaring-beneath-its-name": Scenario(
        modules=_MODULES,
        header=(setup, _BENEATH["static"]),
        probes={"alias": rejected(alias, AglScopeError, alias)},
    )
    for kind, (setup, alias) in _OWN_ALIASES.items()
}

_SCENARIOS |= {
    "the-earlier-completed-pair-is-reported": Scenario(
        modules=_MODULES,
        header=("import base::*",),
        probes={
            "declaration-then-alias-then-declaration": rejected(
                "def Geo::k() -> int = 1\ntype Geo = Base\ndef Geo::j() -> int = 2",
                AglScopeError,
                "type Geo = Base",
            ),
            "alias-then-two-declarations": rejected(
                "type Geo = Base\ndef Geo::k() -> int = 1\nlet Geo::v = 2",
                AglScopeError,
                "Geo",
            ),
            "two-aliases": rejected(
                "type Ga = Base\ndef Gb::k() -> int = 1\ntype Gb = Base\ndef Ga::k() -> int = 1",
                AglScopeError,
                "type Gb = Base",
            ),
        },
    ),
    "an-alias-declared-in-a-region": Scenario(
        modules=_MODULES,
        header=("import base::*", "scope S\n  type Geo = Base\nend S"),
        probes={
            "shorthand": rejected("def S::Geo::k() -> int = 1", AglScopeError, "Geo"),
            "in-its-region": rejected(
                "scope S\n  def Geo::k() -> int = 1\nend S", AglScopeError, "Geo"
            ),
            "nested-region": rejected(
                "scope S::Geo\n  def k() -> int = 1\nend S::Geo", AglScopeError, "Geo"
            ),
            "its-name-at-the-root": accepted("def Geo::k() -> int = 1\nGeo::k()", "int"),
            "read-through-it": accepted("S::Geo::f()", "int"),
        },
    ),
}

_OWN = (
    "def Geo::k() -> int = 1\n\nscope Geo\n  def h() -> int = 2\nend Geo\n\nlet Geo::v = 3\n"
    'record Geo::R\n  z: int\ndef Geo::f() -> text = "own"\ndef Geo::Inner::j() -> int = 4'
)
"""Own declarations beneath an imported alias ``Geo``, one at a path its target declares."""


def _beneath_imported_probes() -> dict[str, Probe]:
    """What :data:`_OWN` declares, read through the alias, its target and the own root."""
    return {
        "static": accepted("Geo::k()", "int"),
        "region": accepted("Geo::h()", "int"),
        "binding": accepted("Geo::v", "int"),
        "record": accepted("Geo::R(z = 1)", "record Geo::R\n  z: int"),
        **type_positions("record", "Geo::R", "Geo::R"),
        "nested": accepted("Geo::Inner::j()", "int"),
        "own-beats-the-reach": accepted("Geo::f()", "text"),
        "the-reach-beside-them": accepted("Geo::Inner(y = 1).y", "int"),
        "anchored-static": accepted("::Geo::k()", "int"),
        "anchored-beats-the-reach": accepted("::Geo::f()", "text"),
        "anchored-reaches-nothing-else": rejected(
            "::Geo::Inner(y = 1)", UnknownMemberError, "::Geo::Inner"
        ),
        "target-static": rejected("Base::k()", UnknownMemberError, "Base::k"),
        "target-region": rejected("Base::h()", UnknownMemberError, "Base::h"),
        "target-binding": rejected("Base::v", UnknownMemberError, "Base::v"),
        "target-record": rejected("Base::R(z = 1)", UnknownMemberError, "Base::R"),
        "target-nested": rejected("Base::Inner::j()", UnknownMemberError, "Base::Inner::j"),
        "target-keeps-its-own": accepted("Base::f()", "int"),
        "anchored-target": rejected("::Base::k()", UnknownQualifierError, "::Base::k"),
    }


_SCENARIOS |= {
    f"declared-beneath-an-imported-alias-{name}": Scenario(
        modules=_MODULES,
        header=(*header, _OWN),
        probes=_beneath_imported_probes(),
    )
    for name, header in (
        ("by-a-wildcard", ("import al::*",)),
        ("by-an-item", ("import al::{Geo, Base}",)),
        ("used", ("import al", "use al::{Geo, Base}")),
    )
}

_BUILTIN_TARGETS = (
    "type U2 = text\ntype Brr[E] = array[E]\ntype IB = array[int]\n"
    "type TD[V] = dict[text, V]\ntype F2 = int -> bool\ntype IBox = gen::Box[int]\n"
    "type UP = text\ndef text::own() -> int = 0\n"
)
"""Aliases of built-in, applied and structural types, declared through an import."""

_SCENARIOS["declared-beneath-an-imported-alias-of-no-record"] = Scenario(
    modules={**_MODULES, "bi": f"import gen\n{_BUILTIN_TARGETS}"},
    header=("import bi::*",),
    probes={
        **{
            f"{alias}-static": accepted(f"def {alias}::k() -> int = 1\n{alias}::k()", "int")
            for alias in ("U2", "Brr", "F2")
        },
        "record": accepted("record U2::R\n  z: int\nU2::R(z = 1).z", "int"),
        "region": accepted("scope F2\n  def h() -> int = 2\nend F2\n\nF2::h()", "int"),
        "target-static": rejected(
            "def U2::k() -> int = 1\ntext::k()", UnknownMemberError, "text::k"
        ),
        "own-beats-the-reach": accepted('def UP::own() -> text = "own"\nUP::own()', "text"),
    },
)

_APPLIED_TARGETS = (
    "import gen\ntype IB = array[int]\ntype TD[V] = dict[text, V]\ntype IBox = gen::Box[int]\n"
    "type IO = gen::Opt[int]\ndef array::st() -> int = 9\n"
)
"""Aliases applying a built-in or generic type to own arguments, declared through an import."""

_SCENARIOS["declared-beneath-an-imported-alias-applying-its-target"] = Scenario(
    modules={
        "gen": "record Box[T]\n  v: T\ndef Box::gb() -> int = 7\n"
        "enum Opt[T] = Som(v: T) | Non\ndef Opt::ob() -> int = 8\n",
        "bi": _APPLIED_TARGETS,
    },
    header=("import bi::*\nimport gen::*",),
    probes={
        **{
            f"{alias}-static": accepted(f"def {alias}::k() -> int = 1\n{alias}::k()", "int")
            for alias in ("IB", "TD", "IBox", "IO")
        },
        "IB-record": accepted("record IB::R\n  z: int\nIB::R(z = 1).z", "int"),
        "IBox-region": accepted("scope IBox\n  def h() -> int = 2\nend IBox\n\nIBox::h()", "int"),
        "IBox-reach": accepted("IBox::gb()", "int"),
        "IO-reach": accepted("IO::ob()", "int"),
        "IB-reach": accepted("IB::st()", "int"),
        "IBox-own-beats-the-reach": accepted('def IBox::gb() -> text = "own"\nIBox::gb()', "text"),
        "IB-own-beats-the-reach": accepted('def IB::st() -> text = "own"\nIB::st()', "text"),
        "IBox-target-reads-nothing-own": rejected(
            "def IBox::k() -> int = 1\nBox::k()", UnknownMemberError, "Box::k"
        ),
        "IBox-constructor": accepted("IBox::Box(v = 1).v", "int"),
        "IO-member": accepted("IO::Som(v = 1).v", "int"),
        "IO-member-at-its-arguments": rejected(
            'IO::Som(v = "x")', AglTypeError, '"x"', phase="typecheck"
        ),
        "written-arguments": rejected("IBox[int]::gb()", TypeArgumentsError, "IBox[int]"),
    },
)


def _receiver_probes(spelling: str, record: str) -> dict[str, Probe]:
    """Methods whose receiver *spelling*, an alias of ``Base``, selects; *record* renders Base."""
    return {
        f"{spelling}-method": rejected(
            f"def {spelling}::m(self) -> int = 1", AglScopeError, "self"
        ),
        f"{spelling}-annotated-method": rejected(
            f"def {spelling}::m(self: {record}) -> int = 1", AglScopeError, f"self: {record}"
        ),
        f"{spelling}-nested": rejected(
            f"def {spelling}::Inner::m(self) -> int = 1", AglScopeError, "self"
        ),
        f"{spelling}-in-a-region": rejected(
            f"scope {spelling}\n  def m(self) -> int = 1\nend {spelling}", AglScopeError, "self"
        ),
        f"{spelling}-in-a-nested-region": rejected(
            f"scope {spelling}::Inner\n  def m(self) -> int = 1\nend {spelling}::Inner",
            AglScopeError,
            "self",
        ),
        f"{spelling}-in-a-region-beneath": rejected(
            f"scope {spelling}\n  def Inner::m(self) -> int = 1\nend {spelling}",
            AglScopeError,
            "self",
        ),
    }


_SCENARIOS |= {
    "a-receiver-through-an-imported-alias": Scenario(
        modules=_MODULES,
        header=("import al::*",),
        probes={
            **_receiver_probes("Geo", "Base"),
            "target": accepted("def Base::m(self) -> int = self.x\nBase(x = 1).m()", "int"),
            "target-nested": accepted(
                "def Base::Inner::m(self) -> int = self.y\nBase::Inner(y = 1).m()", "int"
            ),
            "target-read-through-the-alias": accepted(
                "def Base::m(self) -> int = self.x\nGeo::m(Base(x = 1))", "int"
            ),
        },
    ),
    "a-receiver-through-an-imported-alias-of-no-record": Scenario(
        modules={**_MODULES, "bi": f"import gen\n{_BUILTIN_TARGETS}"},
        header=("import bi::*\nimport gen::*",),
        probes={
            **{
                f"{alias}-method": rejected(text, AglScopeError, "self")
                for alias, text in {
                    "U2": "def U2::m(self) -> int = 1",
                    "Brr": "def Brr[E]::m(self) -> int = 1",
                    "Brr-bare": "def Brr::m(self) -> int = 1",
                    "IB": "def IB::m(self) -> int = 1",
                    "TD": "def TD[V]::m(self) -> int = 1",
                    "F2": "def F2::m(self) -> int = 1",
                    "IBox": "def IBox::m(self) -> int = 1",
                    "U2-in-a-region": "scope S\n  def U2::m(self) -> int = 1\nend S",
                }.items()
            },
            "builtin": accepted('def text::m(self) -> int = 1\n"a".m()', "int"),
            "builtin-generic": accepted("def array[E]::m2(self) -> int = 1\n[1].m2()", "int"),
            "builtin-text-keyed": accepted(
                'def dict[text, V]::m2(self) -> int = 1\n{"a": 1}.m2()', "int"
            ),
            "builtin-generic-static": rejected(
                "def array[E]::f() -> int = 1", TypeArgumentsError, "array[E]"
            ),
            "builtin-applied-method": rejected(
                "def array[int]::m2(self) -> int = 1", AglScopeError, "array[int]"
            ),
            "builtin-bare-generic-method": rejected(
                "def array::m2(self) -> int = 1", AglScopeError, "array"
            ),
            "applied-head-static": rejected(
                "def Box[int]::f() -> int = 1", TypeArgumentsError, "Box[int]"
            ),
            "generic-head-method": rejected(
                "def Box[T]::m(self) -> int = 1", TypeArgumentsError, "Box[T]"
            ),
            "unknown-applied-head": rejected(
                "def bytes[int]::m(self) -> int = 1", TypeArgumentsError, "bytes[int]"
            ),
            "no-receiver-scope": rejected("def unit::m(self) -> int = 1", AglScopeError, "self"),
        },
    ),
    "a-receiver-through-an-alias-beneath-a-type": Scenario(
        modules={
            **_MODULES,
            "ia": "record Plain\n  p: int\nrecord Hold\n  h: int\ntype Hold::P = Plain\n",
        },
        header=("import ia::*",),
        probes={
            "method": rejected("def Hold::P::m(self) -> int = 1", AglScopeError, "self"),
            "in-a-region": rejected(
                "scope Hold\n  def P::m(self) -> int = 1\nend Hold", AglScopeError, "self"
            ),
            "static": accepted("def Hold::P::k() -> int = 1\nHold::P::k()", "int"),
            "read": accepted("Hold::P(p = 1).p", "int"),
        },
    ),
}

_OWNING = (
    "record Base\n  x: int\nrecord Base::Gen[T]\n  v: T\nrecord Base::In\n  w: int\n"
    "exception Base::Ex\nenum Base::Opt[T] = Som(v: T) | Non"
)
"""Types beneath ``Base`` a constructor is spelled beneath: generic, plain, exception and enum."""


def _owner_qualified_probes() -> dict[str, Probe]:
    """Each type's constructor qualified by its type, through both spellings, in every position."""
    return {
        f"{spelling}-{position}": probe
        for spelling in ("Base", "Geo")
        for position, probe in {
            "generic": accepted(f"{spelling}::Gen::Gen(v = 1).v", "int"),
            "applied-segment": accepted(f"{spelling}::Gen[int]::Gen(v = 1).v", "int"),
            "applied-segment-arity": rejected(
                f"{spelling}::Gen[int, int]::Gen(v = 1).v", TypeArgumentsError, "Gen[int, int]"
            ),
            "applied-enum-segment": accepted(f"{spelling}::Opt[int]::Som(v = 1).v", "int"),
            "enum-value": rejected(f"[{spelling}::Opt]", AglTypeError, f"{spelling}::Opt"),
            "plain": accepted(f"{spelling}::In::In(w = 1).w", "int"),
            "exception": accepted(f'{spelling}::Ex::Ex(message = "m").message', "text"),
            "pattern": accepted(
                f"case Base::In(w = 2) of\n  | {spelling}::In::In(w = w) => w", "int"
            ),
            "is": accepted(f'Base::Ex(message = "m") is {spelling}::Ex::Ex', "bool"),
            "type": rejected(
                f"fn(p: {spelling}::In::In) => p",
                AglTypeError,
                f"p: {spelling}::In::In",
                phase="typecheck",
            ),
        }.items()
    }


_SCENARIOS |= {
    f"an-alias-reaches-a-constructor-qualified-by-its-type-{name}": Scenario(
        modules={"oq": f"{_OWNING}\n", "oqa": "import oq::*\nexport oq::{Base}\ntype Geo = Base\n"},
        header=header,
        probes=_owner_qualified_probes(),
    )
    for name, header in (
        ("own", (_OWNING, "type Geo = Base")),
        ("imported-target", ("import oq::*", "type Geo = Base")),
        ("imported-alias", ("import oqa::*",)),
    )
}

_VALUELESS = (
    "enum Top = A | B\nrecord Base\n  x: int\nenum Base::E = C | D\n"
    "enum Base::Opt[T] = Som(v: T) | Non\ntype Base::TI = int"
)
"""Types naming no value: a root enum, and beneath ``Base`` an enum, a generic one and an alias."""


def _type_name_probes(hidden: bool) -> dict[str, Probe]:
    """Each valueless type spelled as a value, through both spellings; *hidden* by an import."""
    nested = HiddenMemberError if hidden else AglTypeError
    return {"root": rejected("[Top]", AglTypeError, "Top")} | {
        f"{spelling}-{name}": rejected(f"[{spelling}::{name}]", nested, f"{spelling}::{name}")
        for spelling in ("Base", "Geo")
        for name in ("E", "Opt", "TI")
    }


_SCENARIOS |= {
    f"a-type-spelled-as-a-value-is-a-type-name-{name}": Scenario(
        modules={
            "vt": f"{_VALUELESS}\n",
            "vta": "import vt::*\nexport vt::{Base}\ntype Geo = Base\n",
        },
        header=header,
        probes=_type_name_probes(hidden=False),
    )
    for name, header in (
        ("own", (_VALUELESS, "type Geo = Base")),
        ("imported-target", ("import vt::*", "type Geo = Base")),
        ("imported-alias", ("import vta::*\nimport vt::*",)),
    )
} | {
    "a-type-spelled-as-a-value-is-hidden-where-its-import-hides-it": Scenario(
        modules={"vt": f"{_VALUELESS}\n"},
        header=("import vt::* hiding Base::E, Base::Opt, Base::TI", "type Geo = Base"),
        probes=_type_name_probes(hidden=True),
    )
}

_TYPES_OF_THEIR_OWN = (
    "record Plain\n  x: int\ndef Plain::pm() -> int = 1\ntype P[T] = Plain\ntype Q[T] = P[T]"
)
"""``P``, a generic alias that is a type of its own, and ``Q``, another name for it."""


def _renamed_type_of_its_own_probes() -> dict[str, Probe]:
    """What ``pz`` declares beneath ``P``, and the target's path, read through ``P`` and ``Q``."""
    probes: dict[str, Probe] = {}
    for read in ("P", "Q"):
        probes |= {
            f"{read}-declared-elsewhere": accepted(f"{read}::pz()", "int"),
            f"{read}-target-path": accepted(f"{read}::pm()", "int"),
            f"{read}-type": accepted(f"(fn(q: {read}[int]) => q.x)(Plain(x = 1))", "int"),
            f"{read}-undeclared": rejected(f"{read}::zz()", UnknownMemberError, f"{read}::zz"),
            f"{read}-own": accepted(f"def {read}::d() -> int = 2\n{read}::d()", "int"),
        }
    probes["no-target-path"] = rejected(
        "def P::d() -> int = 2\nPlain::d()", UnknownMemberError, "Plain::d"
    )
    return probes


_SCENARIOS["a-generic-alias-renaming-an-imported-type-of-its-own-is-that-type"] = Scenario(
    modules={"pl": f"{_TYPES_OF_THEIR_OWN}\n", "pz": "import pl::*\ndef P::pz() -> int = 4\n"},
    header=("import pl::*\nimport pz::*",),
    probes=_renamed_type_of_its_own_probes(),
)

_INNER_TYPES_OF_THEIR_OWN = (
    "record Plain\n  x: int\ndef Plain::pm() -> int = 0\nrecord Base\n  y: int\n"
    "record Mid\n  z: int\ntype Base::P[T] = Plain\ntype Base::Q[T] = Base::P[T]\n"
    "type Base::K[T] = int\ntype Mid::P[T] = Plain\ntype Base::M = Mid\n"
)
"""Aliases beneath ``Base`` that are types of their own, one renamed, and one renaming ``Mid``."""
_BENEATH_INNER = (
    "import inn::*\ndef Base::P::m() -> int = 1\ndef Base::Q::q() -> int = 2\n"
    "def Base::K::k() -> int = 3\ndef Base::P::S::d() -> int = 4\n\n"
    "scope Base::P\n  def z() -> int = 5\nend Base::P\n\n"
    "record Base::P::R\n  b: int\ndef Mid::P::n() -> int = 6\n"
)
"""Declarations beneath the imported aliases of :data:`_INNER_TYPES_OF_THEIR_OWN`."""


def _inner_type_of_its_own_probes() -> dict[str, Probe]:
    """Each declaration beneath an alias of its own beneath ``Base``, through both spellings."""
    return {
        f"{spelling}-{position}": probe
        for spelling in ("Base", "Geo")
        for position, probe in {
            "own-path": accepted(f"{spelling}::P::m()", "int"),
            "declared-through-a-renaming-name": accepted(f"{spelling}::Q::q()", "int"),
            "of-a-builtin-type": accepted(f"{spelling}::K::k()", "int"),
            "nested": accepted(f"{spelling}::P::S::d()", "int"),
            "region": accepted(f"{spelling}::P::z()", "int"),
            "record": accepted(f"{spelling}::P::R(b = 1).b", "int"),
            "target-path": accepted(f"{spelling}::P::pm()", "int"),
            "undeclared": rejected(
                f"{spelling}::P::zz()", UnknownMemberError, f"{spelling}::P::zz"
            ),
            "renamed-own-path": accepted(f"{spelling}::Q::m()", "int"),
            "renamed-nested": accepted(f"{spelling}::Q::S::d()", "int"),
            "through-a-renaming-alias": accepted(f"{spelling}::M::P::n()", "int"),
            "own-beneath-the-renamed-path": accepted(
                f"def Base::P::w() -> int = 7\n{spelling}::Q::w()", "int"
            ),
            "through-a-renaming-alias-undeclared": rejected(
                f"{spelling}::M::P::zz()", UnknownMemberError, f"{spelling}::M::P::zz"
            ),
        }.items()
    }


_SCENARIOS |= {
    f"an-outer-alias-reaches-an-inner-alias-of-its-own-path-{name}": Scenario(
        modules={"inn": f"{_INNER_TYPES_OF_THEIR_OWN}\n", "innd": _BENEATH_INNER, **modules},
        header=header,
        probes=_inner_type_of_its_own_probes(),
    )
    for name, modules, header in (
        ("own-alias", {}, ("import inn::*\nimport innd::*", "type Geo = Base")),
        (
            "imported-alias",
            {"ia": "import inn::*\nexport inn::{Base}\ntype Geo = Base\n"},
            ("import ia::*\nimport inn::*\nimport innd::*",),
        ),
    )
}

_CAPTURING = "record X\n  a: int\ntype A[T] = X\ntype C[X] = A[int]\ntype D = C[text]"
"""``C`` applies ``A``, which stands for ``X``, with a parameter spelled ``X`` of its own."""

_SCENARIOS |= {
    "applying-an-alias-never-captures-its-names-own": Scenario(
        header=(_CAPTURING,),
        probes={
            "renaming-declares-nothing": rejected("def D::m() -> int = 1", AglScopeError, "D"),
            "own-type-declares-nothing": rejected("def C::m() -> int = 1", AglScopeError, "C"),
            "renaming-reads-the-target": accepted("def X::n() -> int = 2\nD::n()", "int"),
            "renaming-value": accepted("D(a = 1).a", "int"),
            "renaming-type": accepted("(fn(d: D) => d.a)(X(a = 1))", "int"),
            "own-type-reads-the-target": accepted("def X::n() -> int = 2\nC::n()", "int"),
            "own-type-type": accepted("(fn(c: C[bool]) => c.a)(X(a = 1))", "int"),
        },
    ),
    "applying-an-alias-never-captures-its-names-imported": Scenario(
        modules={"cap": f"{_CAPTURING}\n"},
        header=("import cap::*",),
        probes={
            "renaming-declares-at-its-path": accepted("def D::m() -> int = 1\nD::m()", "int"),
            "renaming-target-reads-nothing": rejected(
                "def D::m() -> int = 1\nX::m()", UnknownMemberError, "X::m"
            ),
            "own-type-declares-at-its-path": accepted("def C::m() -> int = 1\nC::m()", "int"),
            "renaming-reads-the-target": accepted("def X::n() -> int = 2\nD::n()", "int"),
            "renaming-value": accepted("D(a = 1).a", "int"),
            "own-type-reads-the-target": accepted("def X::n() -> int = 2\nC::n()", "int"),
            "own-type-type": accepted("(fn(c: C[bool]) => c.a)(X(a = 1))", "int"),
        },
    ),
}

_THROUGH = "import base::*\ntype Geo = Base\ndef Base::g() -> int = 2\nrecord Base::R\n  z: int\n"
"""A module aliasing ``base``'s ``Base`` as ``Geo`` and declaring beneath ``Base`` itself."""

_SCENARIOS |= {
    "an-alias-reaches-what-its-module-declares-beneath-its-target": Scenario(
        modules={**_MODULES, "thr": _THROUGH},
        header=("import thr::*\nimport thr", "import base::*"),
        probes={
            f"{reached}-{name}": probe
            for reached in ("Base", "Geo", "thr::Base", "thr::Geo", "/thr::Geo")
            for name, probe in {
                "static": accepted(f"{reached}::g()", "int"),
                "record": accepted(f"{reached}::R(z = 1).z", "int"),
                "record-type": accepted(f"(fn(p: {reached}::R) => p.z)(Base::R(z = 1))", "int"),
                # A module route reaches only what the module declares or exports.
                "target": rejected(f"{reached}::f()", UnknownMemberError, f"{reached}::f")
                if reached == "thr::Base"
                else accepted(f"{reached}::f()", "int"),
            }.items()
        },
    ),
    "an-alias-reaches-what-its-module-declares-beneath-its-target-through-a-chain": Scenario(
        modules={
            **_MODULES,
            "thr": _THROUGH,
            "rex": "import thr::*\nexport thr::{Geo}\n",
        },
        header=("import rex::*",),
        probes={
            "static": accepted("Geo::g()", "int"),
            "record": accepted("Geo::R(z = 1).z", "int"),
            "target": accepted("Geo::f()", "int"),
        },
    ),
}

_ALIASING_ANOTHER_BASE = (
    "import tb\n\nrecord Base\n  y: int\n\ndef Base::q() -> int = 7\ntype Geo = tb::Base"
)
"""Aliases ``tb``'s ``Base`` as ``Geo`` beside an own ``Base`` declaring ``q``."""


def _another_base_probes(route: str, *, hidden: bool) -> dict[str, Probe]:
    """What *route* reaches beneath the own ``Base`` and through ``Geo``; ``q`` *hidden*."""
    return {
        "own-type": rejected(f"{route}Base::q()", HiddenMemberError, f"{route}Base::q")
        if hidden
        else accepted(f"{route}Base::q()", "int"),
        "through-the-alias": rejected(f"{route}Geo::q()", UnknownMemberError, f"{route}Geo::q"),
        "the-target": accepted(f"{route}Geo::t()", "int"),
    }


_SCENARIOS |= {
    f"a-route-reads-nothing-beneath-another-type-its-target-path-names-{reader}{suffix}": Scenario(
        modules={
            "tb": "record Base\n  x: int\ndef Base::t() -> int = 1",
            "sa": _ALIASING_ANOTHER_BASE,
        },
        header=(f"{header}{hiding}",),
        probes=_another_base_probes(route, hidden=bool(hiding)),
    )
    for reader, header, route in (("route", "import sa", "sa::"), ("importer", "import sa::*", ""))
    for suffix, hiding in (("", ""), ("-hiding-its-path", " hiding Base::q"))
}

_REGION_EXPORTS = {
    "items": ("util::{u}", {"u": "int", "v": UnknownMemberError, "w": UnknownMemberError}),
    "renamed": ("util::{u as v}", {"u": UnknownMemberError, "v": "int", "w": UnknownMemberError}),
    "whole": ("util", {"u": "int", "v": UnknownMemberError, "w": "int"}),
    "hiding": ("util hiding w", {"u": "int", "v": UnknownMemberError, "w": HiddenMemberError}),
    "wildcard": ("util/*", {"u": "int", "v": UnknownMemberError, "w": "int"}),
}
"""Each export a ``scope Base`` region writes of ``util``, and what it forwards there."""


def _region_export_probes(
    route: str, forwarded: dict[str, str | type[AglError]]
) -> dict[str, Probe]:
    """What a ``scope Base`` region's export *forwarded*, and its own ``h``, read after *route*."""
    return {
        f"{spelling}-{name}": accepted(f"{route}{spelling}::{name}()", verdict)
        if isinstance(verdict, str)
        else rejected(f"{route}{spelling}::{name}()", verdict, f"{route}{spelling}::{name}")
        for spelling in ("Base", "Geo")
        for name, verdict in {**forwarded, "h": "int"}.items()
    }


_SCENARIOS |= {
    f"an-alias-reaches-a-region-re-export-beneath-its-target-{form}-{target}-{reader}": Scenario(
        modules={
            "base": "record Base\n  x: int",
            "util": "def u() -> int = 5\ndef w() -> int = 6",
            "al": f"{declaring}\ntype Geo = Base\n\nscope Base\n  export {export}\n"
            "  def h() -> int = 7\nend Base",
        },
        header=(header,),
        probes=_region_export_probes(route, forwarded),
    )
    for form, (export, forwarded) in _REGION_EXPORTS.items()
    for target, declaring in (("imported", "import base::*"), ("own", "record Base\n  x: int"))
    for reader, header, route in (("route", "import al", "al::"), ("importer", "import al::*", ""))
}

_HIDDEN_PATH = (
    "use Sc::* hiding Base::v\n\nscope Sc\n  def Base::v() -> int = 1\nend Sc",
    "record Base\n  x: int",
    "type Geo = Base",
)
"""A use hiding the path ``Base::v``, which nothing else declares, and an alias of ``Base``."""

_SCENARIOS["an-alias-reaches-a-binding-declared-at-a-path-a-use-hides"] = Scenario(
    header=_HIDDEN_PATH,
    probes={
        **{
            f"{form}-through-{spelling}": accepted(f"{binding}\n{spelling}::v", "int")
            for form, binding in {
                "let": "let Base::v = 2",
                "var": "var Base::v = 2",
                "region": "scope Base\n  let v = 2\nend Base",
            }.items()
            for spelling in ("Base", "Geo")
        },
        "through-an-alias-of-the-alias": accepted("type G2 = Geo\nlet Base::v = 2\nG2::v", "int"),
        "through-an-alias-declared-after": accepted("let Base::v = 2\ntype G2 = Geo\nG2::v", "int"),
        "unbound-through-Base": rejected("Base::v", HiddenMemberError, "Base::v"),
        "unbound-through-Geo": rejected("Geo::v", HiddenMemberError, "Geo::v"),
        "unbound-through-an-alias-of-the-alias": rejected(
            "type G2 = Geo\nG2::v", HiddenMemberError, "G2::v"
        ),
    },
)

_CYCLE = {
    "ca": "import cb::*\nrecord Ra\n  x: int\ntype Ta = Ra\ndef Tb::fa() -> int = 1\n",
    "cb": "import ca::*\nrecord Rb\n  y: int\ntype Tb = Rb\ndef Ta::fb() -> int = 3\n",
}
"""Two modules importing each other, each declaring beneath the other's alias."""

_SCENARIOS["declared-beneath-aliases-of-modules-importing-each-other"] = Scenario(
    modules=_CYCLE,
    header=("import ca::*\nimport cb::*",),
    probes={
        "one": accepted("Tb::fa()", "int"),
        "other": accepted("Ta::fb()", "int"),
        "one-not-at-the-target": rejected("Rb::fa()", UnknownMemberError, "Rb::fa"),
        "other-not-at-the-target": rejected("Ra::fb()", UnknownMemberError, "Ra::fb"),
        "own-beside": accepted("def Ta::g() -> int = 5\nTa::g() + Ta::fb()", "int"),
        "own-receiver": rejected("def Ta::m(self) -> int = 1", AglScopeError, "self"),
    },
)

_SCENARIOS["what-a-cycle-member-declares-beneath-an-imported-type-is-re-exported"] = Scenario(
    modules={
        "base": _BASE,
        "dg": (
            "import xi\nimport xw\nimport base::*\ndef Base::m() -> int = 1\n"
            "record Base::R\n  y: int\n"
        ),
        "xi": "import dg\nexport dg::{Base::m, Base::R}\n",
        "xw": "import dg\nexport dg::{Base}\n",
        "dr": "import xr\nimport base::*\n\nscope Base::S\n  def s() -> int = 2\nend Base::S\n",
        "xr": "import dr\nexport dr::{Base::S}\n",
    },
    header=("import xi\nimport xw\nimport xr",),
    probes={
        "item-function": accepted("xi::Base::m()", "int"),
        "item-type": accepted("xi::Base::R(y = 1).y", "int"),
        "whole-function": accepted("xw::Base::m()", "int"),
        "whole-type": accepted("xw::Base::R(y = 1).y", "int"),
        "region": accepted("use xr::Base::S\nS::s()", "int"),
    },
)

_RENAMED_SITE = (
    "import base",
    "use base::{Base as B2}",
    "type RGeo = B2",
    "def B2::g() -> int = 2",
)
"""A site spelling ``Base`` as a ``use`` renames it, declaring beneath that spelling."""

_NESTED_SITE = ("import mid::*", "type NGeo = Mid", "def Mid::k() -> int = 3")
"""A site spelling ``Base`` through an imported alias, declaring beneath that spelling."""

_SITES = {
    "base": "record Base\n  x: int\ndef Base::f() -> int = 1\n",
    "renamed": "\n".join(_RENAMED_SITE) + "\n",
    "mid": "import base::*\ntype Mid = Base\n",
    "nested": "\n".join(_NESTED_SITE) + "\n",
    "regional": (
        "import base\n\nscope r\n  use base::{Base as B2}\n  type SGeo = B2\nend r\n\n"
        "def r::B2::g() -> int = 2\n"
    ),
    "routed": (
        "import base\n\nscope Base\n  def g() -> int = 2\nend Base\n\ntype BGeo = base::Base\n"
    ),
    "hid": (
        "import base::*\nuse base::{Base as B2}\ntype HGeo = Base\ndef B2::k() -> int = 1\n"
        "def Base::j() -> int = 2\n"
    ),
}
"""Modules declaring an alias of ``Base`` beside declarations beneath their spellings of names."""

_SCENARIOS["an-alias-reaches-what-its-site-declares-beneath-its-spelling-of-the-target"] = Scenario(
    modules=_SITES,
    header=("import renamed::{RGeo}\nimport routed::{BGeo}", "import nested::{NGeo}"),
    probes={
        "a-use-renaming-the-target": accepted("RGeo::g()", "int"),
        "an-imported-alias-of-the-target": accepted("NGeo::k()", "int"),
        "a-use-in-a-region": accepted("import regional::*\nr::SGeo::g()", "int"),
        "the-target-itself": accepted("RGeo::f() + NGeo::f() + BGeo::f()", "int"),
        "a-scope-named-like-the-target-naming-nothing": rejected(
            "BGeo::g()", UnknownMemberError, "BGeo::g"
        ),
    },
)

_SCENARIOS["a-module-route-reads-what-an-alias-site-declares-beneath-the-target"] = Scenario(
    modules=_SITES,
    header=("import renamed\nimport nested", "import regional"),
    probes={
        "a-use-renaming-the-target": accepted("renamed::RGeo::g()", "int"),
        "an-imported-alias-of-the-target": accepted("nested::NGeo::k()", "int"),
        "a-use-in-a-region": accepted("regional::r::SGeo::g()", "int"),
    },
)

_SCENARIOS["an-own-alias-reaches-the-declarations-beneath-a-use-renaming-its-target"] = Scenario(
    modules=_SITES,
    header=_RENAMED_SITE,
    probes={"declared": accepted("RGeo::g()", "int"), "target": accepted("RGeo::f()", "int")},
)

_SCENARIOS["an-own-alias-reaches-the-declarations-beneath-an-imported-alias-it-names"] = Scenario(
    modules=_SITES,
    header=_NESTED_SITE,
    probes={"declared": accepted("NGeo::k()", "int"), "target": accepted("NGeo::f()", "int")},
)

_SCENARIOS["hiding-an-alias-removes-what-its-module-declares-beneath-any-spelling"] = Scenario(
    modules=_SITES,
    header=("import hid::* hiding HGeo\nimport routed::* hiding BGeo",),
    probes={
        "a-use-renaming-the-target": rejected("B2::k()", HiddenMemberError, "B2::k"),
        "the-target-spelling": rejected("Base::j()", HiddenMemberError, "Base::j"),
        "a-scope-named-like-the-target-naming-nothing": accepted("Base::g()", "int"),
    },
)


class TestDeclarationsThroughAliases:
    """Declarations beneath alias names, the file part and every REPL grouping."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)


def test_info_describes_a_declaration_beneath_an_imported_alias_at_its_written_path(
    tmp_path: Path,
) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        (
            "import al::*",
            "def Geo::g() -> int = 2",
            "scope Geo\n  def h() -> int = 4\nend Geo",
            "let Geo::v = 5",
            "record Geo::Gen[T]\n  v: T\ntype Geo::A[T] = array[T]",
        ),
        {
            "function": info(
                "Geo::g",
                "Geo::g is a function.\nSignature:\n  def Geo::g() -> int\nLocation: <repl>:1:1",
            ),
            "region-function": info(
                "Geo::h",
                "Geo::h is a function.\nSignature:\n  def Geo::h() -> int\nLocation: <repl>:2:3",
            ),
            "binding": info(
                "Geo::v",
                "Geo::v is a binding.\nBinding:\n  let Geo::v\nType:\n  int\nValue:\n  5\n"
                "Location: <repl>:1:1",
            ),
            "generic-type": info(
                "Geo::Gen",
                "Geo::Gen is a generic record type.\nType:\n  record Geo::Gen[T]\n"
                "    v: T\nLocation: <repl>:1:1",
            ),
            "alias": info("Geo::A", "Geo::A is a type alias.\nType:\n  type Geo::A[T] = array[T]"),
        },
    )


def _session(tmp_path: Path, modules: dict[str, str], entries: tuple[str, ...]) -> ReplSession:
    """A REPL session rooted at *tmp_path*, holding *modules*, after *entries*."""
    for name, source in modules.items():
        (tmp_path / f"{name}.agl").write_text(f"{source}\n")
    session = repl_session_with_root(tmp_path)
    session.open()
    for entry in entries:
        assert session.eval_entry(entry).ok
    return session


_DECLARED = (
    "record Base\n  x: int\ndef Base::g() -> int = 2\nlet Base::v = 5\nrecord Base::Gen[T]\n"
    "  v: T\ntype Base::A[T] = array[T]\nenum Base::E\n  | P\n  | Q"
)
"""A declaration of each kind beneath ``Base``."""


@pytest.mark.parametrize("member", ["g", "v", "Gen", "A", "E", "E::P", "Gen::Gen"])
@pytest.mark.parametrize("spelling", ["Base", "Geo"])
def test_info_describes_an_imported_declaration_by_its_path_in_its_module(
    tmp_path: Path, spelling: str, member: str
) -> None:
    """As the same declaration in the session, but unlocated and constructing an imported type."""
    imported = _session(tmp_path, {"decl": _DECLARED}, ("import decl::*", "type Geo = Base"))
    local = _session(tmp_path, {}, (_DECLARED, "type Geo = Base"))

    described = imported.info_of(f"{spelling}::{member}")
    declared = local.info_of(f"{spelling}::{member}")

    assert described is not None
    assert declared is not None
    assert described.split(" -> ")[0] == declared.split("\nLocation:")[0].split(" -> ")[0]


@pytest.mark.parametrize("member", ["Gen::Gen", "In::In", "Ex::Ex", "Opt::Som"])
@pytest.mark.parametrize(
    "header",
    [(_OWNING, "type Geo = Base"), ("import oq::*", "type Geo = Base"), ("import oqa::*",)],
    ids=["own", "imported-target", "imported-alias"],
)
def test_info_describes_a_constructor_qualified_through_the_alias_as_through_its_target(
    tmp_path: Path, header: tuple[str, ...], member: str
) -> None:
    session = _session(
        tmp_path,
        {"oq": _OWNING, "oqa": "import oq::*\nexport oq::{Base}\ntype Geo = Base"},
        header,
    )

    described = session.info_of(f"Geo::{member}")
    declared = session.info_of(f"Base::{member}")

    assert described is not None
    assert declared is not None
    assert described.replace(f"Geo::{member}", "N", 1) == declared.replace(
        f"Base::{member}", "N", 1
    )


def test_rebinding_an_own_alias_moves_nothing(tmp_path: Path) -> None:
    assert_repl_verdicts(
        tmp_path,
        {},
        (
            "record Base\n  x: int\ndef Base::k() -> int = 1",
            "type G = Base",
            "record Other\n  y: int",
            "type G = Other",
        ),
        {
            "the-old-target": accepted("Base::k()", "int"),
            "through-the-alias": rejected("G::k()", UnknownMemberError, "G::k"),
            "the-new-target": accepted("G(y = 1).y", "int"),
            "still-declares-nothing": rejected("def G::k() -> int = 2", AglScopeError, "G"),
        },
    )


def test_a_type_redeclaring_an_own_alias_hosts_declarations(tmp_path: Path) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        ("import base::*", "type Geo = Base", "record Geo\n  y: int"),
        {"beneath": accepted("def Geo::k() -> int = 1\nGeo::k()", "int")},
    )


@pytest.mark.parametrize(
    "earlier",
    ["enum Geo\n  | A\n  | B", "record Geo\n  y: int", "type Geo = int"],
    ids=["an-enum", "a-record", "an-alias"],
)
def test_an_own_alias_redeclaring_a_type_hosting_nothing_is_accepted(
    tmp_path: Path, earlier: str
) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        ("import base::*", earlier),
        {"alias": accepted("type Geo = Base\nGeo(x = 1).x", "int")},
    )


@pytest.mark.parametrize(
    "beneath",
    [
        "def Geo::k() -> int = 1",
        "record Geo::R\n  z: int",
        "scope Geo::In\nend Geo::In",
        "enum Geo::E\n  | A\n  | B",
    ],
    ids=["a-function", "a-record", "an-empty-region", "an-enum"],
)
def test_an_own_alias_redeclaring_a_type_hosting_declarations_is_rejected(
    tmp_path: Path, beneath: str
) -> None:
    """What an earlier entry declared beneath the type's path is beneath the alias."""
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        ("import base::*", "enum Geo\n  | A\n  | B", beneath),
        {"alias": rejected("type Geo = Base", AglScopeError, "type Geo = Base")},
    )


@pytest.mark.parametrize(
    "imported",
    ["import lib::{free}", "import lib::*"],
    ids=["an-import-tail", "an-import-wildcard"],
)
def test_a_later_region_import_beneath_an_imported_alias_replaces_an_earlier_one(
    tmp_path: Path, imported: str
) -> None:
    """Both regions are at their written path: the later import of the module there replaces."""
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        (
            "import al::*",
            f"scope Geo\n  {imported}\nend Geo",
            "scope Geo\n  import lib::{S}\nend Geo",
        ),
        {
            "replaced": rejected("def Geo::k() -> int = free()\nGeo::k()", AglScopeError, "free"),
            "replacing": accepted("def Geo::k() -> int = S::top()\nGeo::k()", "int"),
            "not-at-the-target": rejected(
                "def Base::k() -> int = S::top()\nBase::k()", UnknownQualifierError, "S::top"
            ),
        },
    )


def test_a_declaration_beneath_an_imported_alias_before_its_import_stays_at_its_path(
    tmp_path: Path,
) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        ("def Geo::k() -> int = 1", "import al::*"),
        {
            "own": accepted("Geo::k()", "int"),
            "reach": accepted("Geo::f()", "int"),
            "target": rejected("Base::k()", UnknownMemberError, "Base::k"),
        },
    )


def test_an_alias_redeclared_over_an_earlier_entrys_declarations_beneath_its_imported_name(
    tmp_path: Path,
) -> None:
    """Beneath an imported alias they stood at their path; an own alias of that name rejects."""
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        ("import al::*", "def Geo::k() -> int = 1", "record Other\n  y: int"),
        {
            "alias": rejected("type Geo = Other", AglScopeError, "type Geo = Other"),
            "read": accepted("Geo::k() + Geo::f()", "int"),
        },
    )


@pytest.mark.parametrize(
    "binding",
    ["let Base::v = 2", "var Base::v = 2", "scope Base\n  let v = 2\nend Base"],
    ids=["let", "var", "a-region"],
)
def test_an_earlier_entrys_alias_reaches_a_binding_its_path_was_read_hidden_before(
    tmp_path: Path, binding: str
) -> None:
    """Until an entry binds ``v``, its reads through the alias find only the path a use hides."""
    assert_repl_verdicts(
        tmp_path,
        {},
        _HIDDEN_PATH,
        {
            "through-the-alias": accepted(f"{binding}\nGeo::v", "int"),
            "through-the-target": accepted(f"{binding}\nBase::v", "int"),
            "unbound": rejected("Geo::v", HiddenMemberError, "Geo::v"),
            "unbound-through-the-target": rejected("Base::v", HiddenMemberError, "Base::v"),
        },
    )


def test_a_duplicate_beneath_an_imported_alias_is_one_path_declared_twice(tmp_path: Path) -> None:
    assert_repl_verdicts(
        tmp_path,
        _MODULES,
        ("import al::*",),
        {
            "twice": rejected(
                "def Geo::d() -> int = 1\ndef Geo::d() -> int = 2",
                DuplicateDeclarationError,
                "def Geo::d() -> int = 2",
            ),
            "beside-the-target": accepted(
                'def Geo::d() -> int = 1\ndef Base::d() -> text = "b"\nBase::d()', "text"
            ),
            "both-own-through-the-alias": rejected(
                'def Geo::d() -> int = 1\ndef Base::d() -> text = "b"\nGeo::d()',
                AmbiguousQualificationError,
                "Geo::d",
            ),
        },
    )
