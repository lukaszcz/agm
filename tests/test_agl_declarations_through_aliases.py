"""A declaration path written through an alias is its target's path.

``def Geo::m(self)``, ``def Geo::f()``, ``scope Geo`` and ``def
Geo::Inner::k()`` with ``type Geo = Base`` declare beneath ``Base``: both
spellings reach the declaration in every position, whichever spelling
declared it, the alias own or imported and its target own or imported. Two
declarations of one name beneath the two spellings are one path declared
twice, in one REPL entry or across entries.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.scope.symbols import AglScopeError, DuplicateDeclarationError, TypeArgumentsError
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    assert_file_resolves_like_inline_entry,
    assert_repl_verdicts,
    assert_scenario,
    file_params,
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


def _beside_imported_probes(nested: str) -> dict[str, Probe]:
    """What :func:`_beside_imported` declares, reached through both spellings.

    *nested* renders the own nested record's type.
    """
    return {
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
            probes=_beside_imported_probes(f"{spelling}::Inner"),
        )
        for spelling in ("Base", "Geo")
        for name, header in (("an-own-alias", _OWN_ALIAS), ("an-imported-alias", _IMPORTED_ALIAS))
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


class TestDeclarationsThroughAliases:
    """Declarations through an alias, file mode and every REPL grouping."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", file_params(_SCENARIOS))
    def test_a_file_resolves_like_the_inline_entry(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_file_resolves_like_inline_entry(tmp_path, scenario)


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
