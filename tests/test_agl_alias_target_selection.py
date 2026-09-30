"""An alias denotes what its target spelling selects where the alias is declared.

The alias's target is the declaration scope selects for that spelling in
type position there -- a member reached through another alias's owner, or a
scope region a ``use`` opens in the alias's own region, included -- so every
position the alias is used in (annotation, constructor, owner-qualified
constructor, pattern, ``is``) agrees with spelling the target directly, in
file mode and every REPL grouping (see :mod:`tests.agl.qualifier_support`).
The alias qualifies every member of its target, whatever other import also
spells that member's full path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.scope.symbols import AmbiguousQualificationError
from tests.agl.qualifier_support import (
    FilePhase,
    Groupings,
    accepted,
    assert_repl_verdicts,
    assert_verdicts,
    grouping_batches,
    probe_table,
)

_ACCEPTED: tuple[FilePhase, type[BaseException] | type[None]] = ("accepted", type(None))

_SLOT_LIB = {"lib": "enum Slot\n  | Filled(value: int)\n  | Empty\n\ntype O = Slot\n"}
_FILLED = "record lib::Slot::Filled\n  value: int"

_MEMBER_PROBES = {
    "annotation": "fn(t: T) => t.value",
    "constructor": "T(value = 1)",
    "owner-qualified": "T::Filled(value = 1)",
    "alias-qualified": "T::T(value = 1)",
    "pattern": "fn(s: lib::Slot) => case s of\n  | T::Filled(value) => value\n  | _ => 0",
    "is": "fn(s: lib::Slot) => s is T",
}
_MEMBER_EXPECTED = dict.fromkeys(_MEMBER_PROBES, _ACCEPTED)
_MEMBER_IDENTITIES = {
    "annotation": "lib::Slot::Filled -> int",
    "constructor": _FILLED,
    "owner-qualified": _FILLED,
    "alias-qualified": _FILLED,
    "pattern": "lib::Slot -> int",
    "is": "lib::Slot -> bool",
}


class TestAliasOfAMemberThroughAnImportedAliasOwner:
    """``type T = O::Filled``, ``O`` another module's alias of an enum, is that member."""

    @pytest.mark.parametrize("groupings", grouping_batches(3))
    def test_routed_owner(self, tmp_path: Path, groupings: Groupings) -> None:
        assert_verdicts(
            tmp_path,
            _SLOT_LIB,
            ("import lib", "type T = lib::O::Filled"),
            probe_table(_MEMBER_PROBES, _MEMBER_EXPECTED, identities=_MEMBER_IDENTITIES),
            groupings=groupings,
        )

    @pytest.mark.parametrize("groupings", grouping_batches(4))
    def test_used_owner(self, tmp_path: Path, groupings: Groupings) -> None:
        assert_verdicts(
            tmp_path,
            _SLOT_LIB,
            ("import lib", "use lib::{O}", "type T = O::Filled"),
            probe_table(_MEMBER_PROBES, _MEMBER_EXPECTED, identities=_MEMBER_IDENTITIES),
            groupings=groupings,
        )


_GEO_LIB = {"tl": "enum Geo = Point(z: text)\n"}
_SHAPES = "scope shapes\n\n  scope Geo\n    record Point\n      x: int\n  end Geo\nend shapes"
_POINT = "record shapes::Geo::Point\n  x: int"


class TestUseOpenedRegionDecidesTheAliasTarget:
    """A ``use`` inside a region opens ``Geo`` there, over a root import tail's ``Geo``.

    The alias, a direct annotation, a constructor and a pattern in that
    region all select the region's ``Geo::Point``; a bare ``Geo`` still
    selects the root import's type, which no region hides.
    """

    @pytest.mark.parametrize("groupings", grouping_batches(3))
    def test_every_position(self, tmp_path: Path, groupings: Groupings) -> None:
        def region(body: str) -> str:
            return f"scope r\n  use shapes::*\n  {body}\nend r\nr::v"

        assert_verdicts(
            tmp_path,
            _GEO_LIB,
            ("import tl::*", _SHAPES),
            probe_table(
                {
                    "value": region("let v = Geo::Point(x = 1)"),
                    "annotation": region("let v = fn(p: Geo::Point) => p.x"),
                    "alias": region("type X = Geo::Point\n  let v = fn(p: X) => p.x"),
                    "alias-constructor": region("type X = Geo::Point\n  let v = X(x = 1)"),
                    "pattern": region(
                        "let v = fn(p: Geo::Point) => case p of\n    | Geo::Point(x) => x"
                    ),
                    "bare-type-past-region": region("let v = fn(p: Geo) => p"),
                },
                {
                    "value": _ACCEPTED,
                    "annotation": _ACCEPTED,
                    "alias": _ACCEPTED,
                    "alias-constructor": _ACCEPTED,
                    "pattern": _ACCEPTED,
                    "bare-type-past-region": _ACCEPTED,
                },
                identities={
                    "value": _POINT,
                    "annotation": "shapes::Geo::Point -> int",
                    "alias": "shapes::Geo::Point -> int",
                    "alias-constructor": _POINT,
                    "pattern": "shapes::Geo::Point -> int",
                    "bare-type-past-region": "tl::Geo -> tl::Geo",
                },
            ),
            groupings=groupings,
        )

    @pytest.mark.parametrize("groupings", grouping_batches(4))
    def test_retained_alias(self, tmp_path: Path, groupings: Groupings) -> None:
        assert_verdicts(
            tmp_path,
            _GEO_LIB,
            ("import tl::*", _SHAPES, "scope r\n  use shapes::*\n  type X = Geo::Point\nend r"),
            probe_table(
                {"constructor": "r::X(x = 1)", "annotation": "fn(p: r::X) => p.x"},
                {"constructor": _ACCEPTED, "annotation": _ACCEPTED},
                identities={
                    "constructor": _POINT,
                    "annotation": "shapes::Geo::Point -> int",
                },
            ),
            groupings=groupings,
        )


class TestAliasKeepsItsTargetPastALaterDeclaration:
    """A REPL alias keeps the target its entry selected when a later entry declares its path.

    Only a REPL history declares the path after the alias; each later lookup
    selects the own declaration.
    """

    def test_retained_alias(self, tmp_path: Path) -> None:
        assert_repl_verdicts(
            tmp_path,
            {"shapes": "scope Geo\n  record Point\n    x: int\nend Geo\n"},
            (
                "import shapes",
                "use shapes::*",
                "type A = Geo::Point",
                "record Geo::Point\n  y: text",
            ),
            {
                "alias-annotation": accepted(
                    "fn(p: A) => p", "shapes::Geo::Point -> shapes::Geo::Point"
                ),
                "alias-constructor": accepted("A(x = 1)", "record shapes::Geo::Point\n  x: int"),
                "annotation": accepted("fn(p: Geo::Point) => p", "Geo::Point -> Geo::Point"),
                "constructor": accepted('Geo::Point(y = "a")', "record Geo::Point\n  y: text"),
            },
        )


class TestBareAliasOfAMemberInAPattern:
    """A bare alias of an enum member matches that member, alone or as a field's pattern."""

    @pytest.mark.parametrize("groupings", grouping_batches(4))
    def test_alias_pattern(self, tmp_path: Path, groupings: Groupings) -> None:
        assert_verdicts(
            tmp_path,
            {},
            (
                "enum Color\n  | Red\n  | Blue",
                "type X = Color::Red",
                "record Box\n  color: Color",
            ),
            {
                "pattern": accepted(
                    "fn(c: Color) => case c of\n  | X => 1\n  | _ => 0", "Color -> int"
                ),
                "as-pattern": accepted(
                    "fn(c: Color) => case c of\n  | X as r => [r]\n  | _ => []",
                    "Color -> array[Color]",
                ),
                "field-pattern": accepted(
                    "fn(b: Box) => case b of\n  | Box(color = X) => 1\n  | _ => 0", "Box -> int"
                ),
            },
            groupings=groupings,
        )


_ROUTE_SLOT = {
    "lib": "enum Slot\n  | Filled(value: int)\n  | Empty\n",
    "o": "scope lib\n\n  enum Slot\n    | Filled(value: int)\n    | Empty\nend lib\n",
    "p": "scope lib\n\n  enum Slot\n    | Filled(value: int)\n    | Empty\nend lib\n",
}
_TWO_COLORS = {
    "one/types": "enum Color\n  | Red\n  | Green\n",
    "two/types": "scope Color\n\n  record Red\nend Color\n",
}


class TestAliasMembersBesideSameSpelledImports:
    """An alias qualifies every member of its target, whatever else spells that member's path.

    Another import exposing a declaration at the member's own full path
    makes that path ambiguous spelled directly, but hides nothing: the
    alias still reaches its target's member.
    """

    @pytest.mark.parametrize("groupings", grouping_batches(4))
    def test_route_owner_beside_a_bare_compound(self, tmp_path: Path, groupings: Groupings) -> None:
        assert_verdicts(
            tmp_path,
            _ROUTE_SLOT,
            ("import lib", "import o::{lib::Slot::Filled}", "type T = lib::Slot"),
            probe_table(
                {"member": "T::Filled(value = 1)", "direct": "lib::Slot::Filled(value = 1)"},
                {"member": _ACCEPTED, "direct": ("scope", AmbiguousQualificationError)},
                span_texts={"direct": "lib::Slot::Filled"},
                identities={"member": "record lib::Slot::Filled\n  value: int"},
            ),
            groupings=groupings,
        )

    @pytest.mark.parametrize("groupings", grouping_batches(4))
    def test_bare_compound_owner_beside_another(self, tmp_path: Path, groupings: Groupings) -> None:
        assert_verdicts(
            tmp_path,
            _ROUTE_SLOT,
            ("import o::*", "import p::{lib::Slot::Filled}", "type T = lib::Slot"),
            probe_table(
                {"member": "T::Filled(value = 1)", "direct": "lib::Slot::Filled(value = 1)"},
                {"member": _ACCEPTED, "direct": ("scope", AmbiguousQualificationError)},
                span_texts={"direct": "lib::Slot::Filled"},
                identities={"member": "record o::lib::Slot::Filled\n  value: int"},
            ),
            groupings=groupings,
        )

    @pytest.mark.parametrize("groupings", grouping_batches(4))
    def test_route_owner_beside_another_route(self, tmp_path: Path, groupings: Groupings) -> None:
        assert_verdicts(
            tmp_path,
            _TWO_COLORS,
            ("import one/types", "import two/types", "type T = types::Color"),
            probe_table(
                {"member": "T::Red", "direct": "types::Color::Red"},
                {"member": _ACCEPTED, "direct": ("scope", AmbiguousQualificationError)},
                span_texts={"direct": "types::Color::Red"},
                identities={"member": "record one/types::Color::Red"},
            ),
            groupings=groupings,
        )
