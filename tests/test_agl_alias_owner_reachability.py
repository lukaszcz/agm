"""An alias's owner-qualified members must reach through every alias on its chain.

A type alias's members and hidden set are filtered against what its own
declaration site's imports or ``use`` allow. An alias of another alias
inherits that filtering already applied at the inner alias's own site, unfiltered,
so a chain of aliases still reaches every member the innermost one does,
rather than losing them to a second, redundant filter. A local
``use ... hiding`` is honoured through an alias exactly as an import's
``hiding`` is, whether the alias's target is declared locally or reached
through an imported module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError
from tests._agl_helpers import check_agl_program

_LIBRARIES = {
    "base": "enum Color = Red | Green",
    "m2": "import base\ntype K = base::Color",
    "opt": "enum Opt[T] = Som(v: T) | Non",
}


def _check(tmp_path: Path, entry: str) -> None:
    check_agl_program(tmp_path, {**_LIBRARIES, "entry": entry})


class TestAliasOfAliasReachesEveryMember:
    """An alias of an imported alias of an enum must reach the enum's members."""

    def test_alias_of_an_imported_alias_selected_by_tail_reaches_members(
        self, tmp_path: Path
    ) -> None:
        _check(tmp_path, "import m2::{K}\ntype C = K\nC::Green")

    def test_alias_of_an_imported_alias_selected_by_route_reaches_members(
        self, tmp_path: Path
    ) -> None:
        _check(tmp_path, "import m2\ntype C = m2::K\nC::Green")

    def test_alias_of_an_own_alias_of_an_own_enum_reaches_members(self, tmp_path: Path) -> None:
        _check(tmp_path, "enum E = A | B\ntype K = E\ntype C = K\nC::A")

    def test_alias_of_an_imported_alias_reaches_members_in_type_position(
        self, tmp_path: Path
    ) -> None:
        _check(tmp_path, "import m2::{K}\ntype C = K\ndef f(g: C::Green) -> int = 1")

    def test_alias_of_an_imported_alias_reaches_members_in_pattern_position(
        self, tmp_path: Path
    ) -> None:
        _check(
            tmp_path,
            "import m2::{K}\ntype C = K\nlet g: C = C::Green\n"
            "case g of\n  | C::Green => 1\n  | _ => 2",
        )

    def test_alias_of_a_generic_alias_reaches_members(self, tmp_path: Path) -> None:
        _check(
            tmp_path,
            "import opt\ntype K[T] = opt::Opt[T]\ntype C = K[int]\nC::Non",
        )


class TestUseHidingThroughAliasIsHonouredLocally:
    """``use ... hiding`` must be honoured through an alias, local target included."""

    _SOURCE = (
        "use s::* hiding E::A\n"
        "\n"
        "scope s\n  enum E = A | B\nend s\n"
        "\n"
        "type C = E\n"
        "\n"
        "program def main() -> unit\n  print({expr})\n"
    )

    def test_hidden_member_through_alias_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(AglTypeError):
            check_agl_program(tmp_path, {"entry": self._SOURCE.format(expr="C::A")})

    def test_sibling_member_through_the_same_alias_still_works(self, tmp_path: Path) -> None:
        check_agl_program(tmp_path, {"entry": self._SOURCE.format(expr="C::B")})

    def test_hidden_member_is_rejected_directly_too(self, tmp_path: Path) -> None:
        """A direct spelling of a hidden member raises the same error class as an alias's.

        ``E::A`` and its alias spelling ``C::A`` name the same hidden fact, so
        both must raise through the one shared diagnostic helper
        (:func:`~agm.agl.diagnostics.hidden_member`), not a use-route-specific
        ambiguity error.
        """
        source = (
            "use s::* hiding E::A\n"
            "\n"
            "scope s\n  enum E = A | B\nend s\n"
            "\n"
            "program def main() -> unit\n  print(E::A)\n"
        )
        with pytest.raises(AglTypeError):
            check_agl_program(tmp_path, {"entry": source})

    def test_hidden_member_is_rejected_directly_too_in_type_position(self, tmp_path: Path) -> None:
        """The same direct hidden member, spelled in type position, raises the same class."""
        source = (
            "use s::* hiding E::A\n"
            "\n"
            "scope s\n  enum E = A | B\nend s\n"
            "\n"
            "def f(x: E::A) -> int = 1\n"
            "\n"
            "program def main() -> unit\n  print(1)\n"
        )
        with pytest.raises(AglTypeError):
            check_agl_program(tmp_path, {"entry": source})

    def test_unknown_member_in_type_position_stays_unknown(self, tmp_path: Path) -> None:
        """A genuinely undeclared member is reported as unknown, not as hidden.

        ``E`` is reached the same way as the hidden-member case above (opened by
        ``use``, unreachable as a module route), but no ``hiding`` clause and no
        inline declaration excludes ``Zzz``: the owner-fallback that checks for a
        hidden member must leave this as the plain unknown-qualifier error.
        """
        source = (
            "use s::* hiding E::A\n"
            "\n"
            "scope s\n  enum E = A | B\nend s\n"
            "\n"
            "def f(x: E::Zzz) -> int = 1\n"
            "\n"
            "program def main() -> unit\n  print(1)\n"
        )
        with pytest.raises(AglTypeError):
            check_agl_program(tmp_path, {"entry": source})


class TestUseHidingThroughAliasStaysHonouredWhenImported:
    """Cross-module ``use ... hiding`` through an alias must keep being rejected."""

    def test_hidden_member_through_alias_of_an_imported_use_is_rejected(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(AglTypeError):
            check_agl_program(
                tmp_path,
                {
                    "m": "enum E = A | B",
                    "entry": ("import m\nuse m::* hiding E::A\ntype C = E\nC::A"),
                },
            )
