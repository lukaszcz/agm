"""An alias denotes what its target spelling selects where the alias is declared.

The alias's target is the declaration scope selects for that spelling in
type position there -- a member reached through another alias's owner, or a
scope region a ``use`` opens in the alias's own region, included -- so every
position the alias is used in (annotation, constructor, owner-qualified
constructor, pattern, ``is``) agrees with spelling the target directly, in
inline and across representative file/REPL entry boundaries
(see :mod:`tests.agl.qualifier_support`).
The alias qualifies every member of its target, whatever other import also
spells that member's full path.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agm.agl.diagnostics import HiddenMemberError
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousQualificationError,
    TypeArgumentsError,
    UnknownMemberError,
    UnknownQualifierError,
)
from tests.agl.module_graph import resolve_entry
from tests.agl.qualifier_support import (
    Part,
    Phase,
    Probe,
    Scenario,
    accepted,
    assert_repl_verdicts,
    assert_scenario,
    assert_verdicts,
    info,
    inline_verdict,
    probe_table,
    rejected,
    scenario_params,
    verdict_parts,
)

_ACCEPTED: tuple[Phase, type[BaseException] | type[None]] = ("accepted", type(None))

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

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_routed_owner(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _SLOT_LIB,
            ("import lib", "type T = lib::O::Filled"),
            probe_table(_MEMBER_PROBES, _MEMBER_EXPECTED, identities=_MEMBER_IDENTITIES),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_used_owner(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _SLOT_LIB,
            ("import lib", "use lib::{O}", "type T = O::Filled"),
            probe_table(_MEMBER_PROBES, _MEMBER_EXPECTED, identities=_MEMBER_IDENTITIES),
            part=part,
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

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_every_position(self, tmp_path: Path, part: Part) -> None:
        def region(body: str) -> str:
            return f"scope r\n  use shapes::*\n  {body}\nend r\nr::v()"

        assert_verdicts(
            tmp_path,
            _GEO_LIB,
            ("import tl::*", _SHAPES),
            probe_table(
                {
                    "value": region("def v() = Geo::Point(x = 1)"),
                    "annotation": region("def v() = fn(p: Geo::Point) => p.x"),
                    "alias": region("type X = Geo::Point\n  def v() = fn(p: X) => p.x"),
                    "alias-constructor": region("type X = Geo::Point\n  def v() = X(x = 1)"),
                    "pattern": region(
                        "def v() = fn(p: Geo::Point) => case p of\n    | Geo::Point(x) => x"
                    ),
                    "bare-type-past-region": region("def v() = fn(p: Geo) => p"),
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
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_retained_alias(self, tmp_path: Path, part: Part) -> None:
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
            part=part,
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


_COLOR = "enum Color\n  | Red\n  | Green"
_SCOPED_COLOR = {
    "lib": "scope S\n  enum Color\n    | Red\n    | Green\nend S\n\ntype C = S::Color\n"
}


class TestUseSpelledThroughARetainedAlias:
    """A retained ``use`` spelled through an alias exposes what the alias's target does.

    A retained alias's target is read with only the uses written before the
    alias, so a use spelled through the alias never reads itself.
    """

    @pytest.mark.parametrize(
        ("modules", "entries", "red"),
        [
            pytest.param({}, (_COLOR, "type C = Color", "use C::*"), "Color", id="use-last"),
            pytest.param(
                _SCOPED_COLOR,
                ("import lib::*", "use C::*", "type C = lib::S::Color"),
                "lib::S::Color",
                id="alias-declared-after-the-use",
            ),
        ],
    )
    def test_entries(
        self, tmp_path: Path, modules: dict[str, str], entries: tuple[str, ...], red: str
    ) -> None:
        assert_repl_verdicts(
            tmp_path,
            modules,
            entries,
            {
                "member": accepted("Red", f"record {red}::Red"),
                "unrelated": accepted("1 + 1", "int"),
                "alias": info("C", f"C is a type alias.\nType:\n  type C = {red}"),
            },
        )

    def test_use_before_the_alias_in_one_entry(self, tmp_path: Path) -> None:
        assert_verdicts(
            tmp_path,
            {},
            (f"use C::*\n{_COLOR}\ntype C = Color",),
            {
                "member": accepted("Red", "record Color::Red"),
                "unrelated": accepted("1 + 1", "int"),
                "alias": info("C", "C is a type alias.\nType:\n  type C = Color"),
            },
        )


class TestUseThroughARetainedAliasOfABuiltinType:
    """A use through a retained alias of a built-in type reads the uses written before it.

    As a use spelling the built-in's name does: ``text`` names a scope a later
    use opens only for what follows that use.
    """

    def test_entries(self, tmp_path: Path) -> None:
        assert_repl_verdicts(
            tmp_path,
            {"lib": "scope text\n  def lf() -> int = 1\nend text\n"},
            ("import lib", "type T = text"),
            {
                "before": rejected(
                    "use T::{lf}\nuse lib::{text}\nlf()", UnknownMemberError, "use T::{lf}"
                ),
                "builtin-before": rejected(
                    "use text::{lf}\nuse lib::{text}\nlf()", UnknownMemberError, "use text::{lf}"
                ),
                "after": accepted("use lib::{text}\nuse T::{lf}\nlf()", "int"),
            },
        )


class TestBareAliasOfAMemberInAPattern:
    """A bare alias of an enum member matches that member, alone or as a field's pattern."""

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_alias_pattern(self, tmp_path: Path, part: Part) -> None:
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
            part=part,
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

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_route_owner_beside_a_bare_compound(self, tmp_path: Path, part: Part) -> None:
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
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_bare_compound_owner_beside_another(self, tmp_path: Path, part: Part) -> None:
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
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_route_owner_beside_another_route(self, tmp_path: Path, part: Part) -> None:
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
            part=part,
        )


_DECOY = 'scope S\n  record T\n    x: int\n  def T::f() -> text = ""\nend S\n'
_INNER = (
    "  scope S\n    record T\n      x: int\n    def T::f() -> int = 1\n  end S\n  type A = S::T\n"
)
_REGION = f"scope R\n{_INNER}end R\n"
_NESTED_LIB = f"{_DECOY}\n{_REGION}"
"""A root ``S::T`` beside ``R``'s ``S::T``, which ``R``'s alias ``A`` names as ``S::T``."""


def _reads(prefix: str, tag: str) -> dict[str, Probe]:
    """What ``prefix`` then ``R::A`` reads: ``R``'s ``S::T``, never the root's."""
    return {
        f"{tag}-static": accepted(f"{prefix}R::A::f()", "int"),
        f"{tag}-constructor": accepted(f"{prefix}R::A(x = 1).x", "int"),
    }


_SCENARIOS: dict[str, Scenario] = {
    f"alias-in-a-region-reached-through-{name}": Scenario(
        modules={"lib": _NESTED_LIB}, header=(header,), probes=_reads(prefix, "read")
    )
    for name, (header, prefix) in {
        "a-route": ("import lib", "lib::"),
        "an-anchored-route": ("import lib", "/lib::"),
        "a-wildcard": ("import lib::*", ""),
        "an-item": ("import lib::{R}", ""),
    }.items()
} | {
    "alias-in-a-region-reached-through-the-current-module": Scenario(
        header=(_DECOY, _REGION), probes=_reads("::", "anchored") | _reads("", "plain")
    ),
    "alias-reached-through-region-steps": Scenario(
        header=(_DECOY, f"scope Q\n{_INNER}  scope W\n    def p() = A::f()\n  end W\nend Q"),
        probes={
            "from-the-region": accepted("scope Q\n  def p() = A::f()\nend Q\n\nQ::p()", "int"),
            "from-a-nested-region": accepted("Q::W::p()", "int"),
            "qualified": accepted("Q::A::f()", "int"),
            "anchored": accepted("::Q::A::f()", "int"),
        },
    ),
}


_TOP = "record Top\n  x: int\ndef Top::f() -> int = 1\n"
_ROUTED = {
    "lib": _TOP,
    "dec/lib": 'record Top\n  x: int\ndef Top::f() -> text = ""\n',
    "rt": "import lib\ntype A = lib::Top\n\nscope P\n  type B = lib::Top\nend P\n",
}

_SCENARIOS |= {
    "alias-led-by-a-module-route": Scenario(
        modules={"lib": _TOP},
        header=("import lib",),
        probes={
            "own": accepted("type A = lib::Top\nA::f()", "int"),
            "own-in-a-region": accepted("scope P\n  type B = lib::Top\nend P\n\nP::B::f()", "int"),
        },
    ),
    "alias-led-by-a-module-route-reads-it-where-declared": Scenario(
        modules=_ROUTED,
        header=("import rt", "import dec/lib"),
        probes={
            "route": accepted("rt::A::f()", "int"),
            "anchored-route": accepted("/rt::A::f()", "int"),
            "region": accepted("rt::P::B::f()", "int"),
            "constructor": accepted("rt::A(x = 1).x", "int"),
            "decoy": accepted("lib::Top::f()", "text"),
        },
    ),
    "alias-led-by-a-module-route-reached-through-an-import": Scenario(
        modules=_ROUTED,
        header=("import rt::*", "import dec/lib"),
        probes={"wildcard": accepted("A::f()", "int"), "region": accepted("P::B::f()", "int")},
    ),
}

_BOTH = {
    "geo": "record Pt\n  x: int\ndef Pt::f() -> int = 1\n",
    "al": (
        'scope geo\n  record Pt\n    x: int\n  def Pt::f() -> text = ""\n'
        "end geo\n\ntype A = geo::Pt\n"
    ),
}
"""A module ``geo`` imported as ``geo`` and a scope ``geo``, which ``A``'s target names."""

_SCENARIOS |= {
    "alias-whose-head-is-both-a-route-and-a-scope": Scenario(
        modules=_BOTH,
        header=("import geo\nimport al", "import al::*"),
        probes={
            "own-scope-wins-in-the-module": accepted("al::A::f()", "text"),
            "through-an-import": accepted("A::f()", "text"),
        },
    ),
    "own-alias-whose-head-is-both-a-route-and-a-scope": Scenario(
        modules={"geo": _BOTH["geo"]},
        header=("import geo", _BOTH["al"]),
        probes={
            "alias": accepted("A::f()", "text"),
            "own-scope-beats-the-route": accepted("geo::Pt::f()", "text"),
            "the-scope": accepted('def Pt::f() -> text = ""\nPt::f()', "text"),
        },
    ),
}


_ROUTE_BEHIND_SCOPE = {
    "geo": _BOTH["geo"],
    "al": "import geo\n\nscope geo\n  def other() -> int = 0\nend geo\n\ntype A = geo::Pt\n",
}
"""``A``'s target head ``geo`` is a route and a scope, which holds no ``Pt``."""

_SCENARIOS |= {
    "alias-whose-head-is-a-route-and-a-scope-holding-nothing-of-the-target": Scenario(
        modules=_ROUTE_BEHIND_SCOPE,
        header=(
            'import al\n\nscope geo\n  record Pt\n    x: int\n  def Pt::f() -> text = ""\nend geo',
        ),
        probes={
            "route": accepted("al::A::f()", "int"),
            "constructor": accepted("al::A(x = 1).x", "int"),
            "entry-scope": accepted("geo::Pt::f()", "text"),
        },
    ),
    "own-alias-whose-head-is-a-route-and-a-scope-holding-nothing-of-the-target": Scenario(
        modules={"geo": _BOTH["geo"]},
        header=("import geo", "scope geo\n  def other() -> int = 0\nend geo", "type A = geo::Pt"),
        probes={
            "alias": accepted("A::f()", "int"),
            "scope-member": accepted("geo::other()", "int"),
            "route-member": accepted("geo::Pt::f()", "int"),
        },
    ),
}

_REEXPORTS = {
    "m": "record T\n  x: int\ndef T::f() -> int = 1\ndef T::g() -> int = 2\ntype A = T\n",
    "rx": "import m::*\nexport m::{A, T::f}\n",
    "ry": "import m::*\nexport m::{A}\n",
}
"""``m`` declares ``A`` of ``T``; ``rx`` re-exports ``A`` and ``T::f``, ``ry`` only ``A``."""

_SCENARIOS |= {
    "alias-reexported-is-read-through-a-route-as-far-as-its-module-exports": Scenario(
        modules=_REEXPORTS,
        header=("import rx\nimport ry",),
        probes={
            "route": accepted("rx::A::f()", "int"),
            "anchored-route": accepted("/rx::A::f()", "int"),
            "not-exported": rejected("rx::A::g()", UnknownMemberError, "rx::A::g"),
            "alias-only": rejected("ry::A::f()", UnknownMemberError, "ry::A::f"),
            "constructor": accepted("rx::A(x = 1).x", "int"),
        },
    ),
    "alias-reexported-is-read-through-an-import-as-the-importer-reads-its-target": Scenario(
        modules=_REEXPORTS,
        header=("import rx::*",),
        probes={
            "exported": accepted("A::f()", "int"),
            "beyond-the-exports": accepted("A::g()", "int"),
            "type": accepted("fn(a: A) => a.x", "m::T -> int"),
        },
    ),
    "item-beneath-an-alias-imports-that-path": Scenario(
        modules=_REEXPORTS,
        header=("import m::{A::f}",),
        probes={
            "item": accepted("A::f()", "int"),
            "other": rejected("A::g()", UnknownQualifierError, "A::g"),
            "target": rejected("T::f()", UnknownQualifierError, "T::f"),
        },
    ),
    "item-beneath-a-reexported-alias-imports-what-its-module-exports": Scenario(
        modules=_REEXPORTS,
        header=("import rx::{A::f}", "import ry::{A}"),
        probes={
            "exported": accepted("A::f()", "int"),
            "alias": accepted("fn(a: A) => a.x", "m::T -> int"),
        },
    ),
    "item-beneath-an-alias-its-module-does-not-export-is-rejected": Scenario(
        modules=_REEXPORTS,
        header=("import ry::{A::f}",),
        probes={"item": rejected("A::f()", UnknownMemberError, "import ry::{A::f}")},
        legal=frozenset({(2,)}),
    ),
}


_OWN_SC = "scope Sc\n" + textwrap.indent(f"{_DECOY}\n{_REGION}", "  ") + "end Sc"

_SCENARIOS |= (
    {
        f"hidden-{what}-beneath-an-alias-reached-through-a-wildcard": Scenario(
            modules={"lib": _NESTED_LIB},
            header=(f"import lib::* hiding {hidden}",),
            probes={"static": rejected("R::A::f()", HiddenMemberError, "R::A::f")},
        )
        for what, hidden in {
            "target-member": "R::S::T::f",
            "path": "R::A::f",
            "target-type": "R::S::T",
        }.items()
    }
    | {
        "hidden-target-type-beneath-an-alias-hides-its-constructor": Scenario(
            modules={"lib": _NESTED_LIB},
            header=("import lib::* hiding R::S::T",),
            probes={"constructor": rejected("R::A(x = 1).x", HiddenMemberError, "R::A")},
        ),
    }
    | {
        f"hidden-{what}-beneath-an-alias-reached-through-a-use": Scenario(
            header=(f"use Sc::* hiding {hidden}", _OWN_SC),
            legal=frozenset({(3,), (2, 1)}),
            probes={
                "use": rejected("R::A::f()", HiddenMemberError, "R::A::f"),
                "scope": accepted("Sc::R::A::f()", "int"),
                "anchored": accepted("::Sc::R::A::f()", "int"),
            },
        )
        for what, hidden in {"target-member": "R::S::T::f", "path": "R::A::f"}.items()
    }
)

_CYCLIC = {
    "ca": "import cb::*\ntype Ta = Tb\n",
    "cb": "import ca::*\ntype Tb = Ta\n",
    "ra": "import rb\ntype Ta = rb::Tb\n",
    "rb": "import ra\ntype Tb = ra::Ta\n",
}
"""Aliases of each other across modules, which reading beneath either reaches nothing of."""

_SCENARIOS |= {
    "alias-leading-back-to-itself-in-a-module-is-a-cycle": Scenario(
        header=("scope P\n  type B = Q::C\nend P\n\nscope Q\n  type C = P::B\nend Q",),
        probes={
            "path": rejected("P::B::f()", AglScopeError, "type B = Q::C"),
            "other-member": rejected("Q::C::f()", AglScopeError, "type B = Q::C"),
        },
        legal=frozenset({(2,)}),
    ),
}

_GROWING = {
    "ga": "import gb\ntype Ta = gb::Tb::X\n",
    "gb": "import ga\ntype Tb = ga::Ta::Y\n",
}
"""Aliases whose targets each pass beneath the other, never terminating."""

_SELF_NESTED = "record T\n  x: int\ndef T::f() -> int = 1\ntype A = T\ntype T::M = A\n"
"""``A`` is ``T``, and ``T`` owns a member ``M`` that is ``A`` again."""
_SELF_NESTED_REGION = (
    "scope P\n  record T\n    x: int\n  def T::f() -> int = 1\n  def T::g() -> int = 2\n"
    "  type A = T\n  type T::B = A\nend P\n"
)
_SIBLING_NESTED = (
    "record T\n  x: int\nrecord T::N\n  y: int\ndef T::N::f() -> int = 1\ntype A = T\n"
    "type T::M = A::N\n"
)
_PING_PONG = {
    "pa": "import pb::*\nrecord T\n  x: int\ndef T::f() -> int = 1\ntype A = T\ntype T::Up = U\n",
    "pb": "import pa::*\ntype U = A\n",
}
"""Aliases of each other across modules, one reading the other beneath a different path."""

_SCENARIOS |= {
    "alias-reentered-beneath-a-different-path-is-not-a-cycle": Scenario(
        header=(_SELF_NESTED,),
        probes={
            "once": accepted("A::M::f()", "int"),
            "twice": accepted("A::M::M::f()", "int"),
            "target-twice": accepted("T::M::M::f()", "int"),
            "anchored": accepted("::A::M::f()", "int"),
            "constructor": accepted("A::M::M(x = 1).x", "int"),
        },
    ),
    "alias-reentered-beneath-a-different-path-through-a-route": Scenario(
        modules={"lib": _SELF_NESTED},
        header=("import lib",),
        probes={
            "once": accepted("lib::A::M::f()", "int"),
            "twice": accepted("lib::A::M::M::f()", "int"),
        },
    ),
    "alias-reentered-beneath-a-different-path-through-a-wildcard": Scenario(
        modules={"lib": _SELF_NESTED},
        header=("import lib::*",),
        probes={"once": accepted("A::M::f()", "int"), "twice": accepted("A::M::M::f()", "int")},
    ),
    "alias-reentered-beneath-a-different-path-in-a-region": Scenario(
        header=(_SELF_NESTED_REGION,),
        probes={
            "once": accepted("P::A::B::f()", "int"),
            "twice": accepted("P::A::B::B::g()", "int"),
            "anchored": accepted("::P::A::B::f()", "int"),
        },
    ),
    "alias-reentered-beneath-a-different-path-across-modules": Scenario(
        modules=_PING_PONG,
        header=("import pa::*",),
        probes={"once": accepted("A::Up::f()", "int"), "twice": accepted("A::Up::Up::f()", "int")},
    ),
    "alias-whose-nested-member-is-another-member-of-its-target": Scenario(
        header=(_SIBLING_NESTED,),
        probes={
            "alias": accepted("A::M::f()", "int"),
            "anchored": accepted("::A::M::f()", "int"),
            "target": accepted("T::M::f()", "int"),
        },
    ),
    "alias-whose-nested-member-is-another-member-of-its-target-through-a-route": Scenario(
        modules={"lib": _SIBLING_NESTED},
        header=("import lib",),
        probes={
            "alias": accepted("lib::A::M::f()", "int"),
            "target": accepted("lib::T::M::f()", "int"),
        },
    ),
    "applied-alias-of-a-target-beneath-its-own-alias-reads-like-the-bare-alias": Scenario(
        header=(
            "type Id[X] = X\nrecord A\n  x: int\nrecord A::B\n  y: int\n"
            "def A::B::f() -> int = 1\n\nscope P\n  type A = Id[A::B]\nend P\n",
        ),
        probes={
            "static": accepted("P::A::f()", "int"),
            "applied": rejected("Id[P::A]::f()", TypeArgumentsError, "Id[P::A]"),
            "constructor": accepted("P::A(y = 1).y", "int"),
        },
    ),
}

_ROOT_METHOD = "record T\n  x: int\ndef T::f() -> int = 1\n"
_TEXT_SCOPE = 'scope Sc\n\n  scope T\n    def f() -> text = ""\n  end T\nend Sc\n'
_HIDING_USE = (
    f"{_ROOT_METHOD}{_TEXT_SCOPE}scope R\n  use Sc::* hiding T::f\n  type A = T\n"
    "  def direct() -> int = T::f()\nend R\n"
)
_APPLIED = (
    f"{_ROOT_METHOD}{_TEXT_SCOPE}type Id[X] = X\nscope R\n  use Sc::* hiding T::f\n"
    "  def applied() -> int = Id[T]::f()\nend R\n"
)
_HIDING_IMPORT = (
    f"{_ROOT_METHOD}scope R\n  import m::* hiding T::f\n  type A = T\n"
    "  def direct() -> int = T::f()\nend R\n"
)

_SCENARIOS |= {
    "alias-target-selected-past-a-scope-whose-member-is-hidden": Scenario(
        header=(_HIDING_USE,),
        probes={
            "alias": accepted("R::A::f()", "int"),
            "direct": accepted("R::direct()", "int"),
            "anchored": accepted("::R::A::f()", "int"),
            "constructor": accepted("R::A(x = 1).x", "int"),
        },
    ),
    "type-arguments-applied-past-a-scope-whose-member-is-hidden": Scenario(
        header=(_APPLIED,),
        probes={"applied": rejected("R::applied()", TypeArgumentsError, "Id[T]")},
        legal=frozenset({(2,)}),
    ),
    "alias-target-selected-past-an-import-whose-member-is-hidden": Scenario(
        modules={"m": 'scope T\n  def f() -> text = ""\nend T\n'},
        header=(_HIDING_IMPORT,),
        probes={
            "alias": accepted("R::A::f()", "int"),
            "direct": accepted("R::direct()", "int"),
            "anchored": accepted("::R::A::f()", "int"),
        },
    ),
}

_ALIAS_PARALLEL = (
    "scope lib\n\n  scope R\n\n    scope S\n      record T\n        x: int\n"
    '      def T::f() -> text = ""\n    end S\n  end R\nend lib'
)

_SCENARIOS |= {
    "alias-reached-through-a-route-ignores-the-readers-scope-of-that-name": Scenario(
        modules={"lib": _REGION},
        header=("import lib", _ALIAS_PARALLEL),
        probes={
            "route": accepted("lib::R::A::f()", "int"),
            "constructor": accepted("lib::R::A(x = 1).x", "int"),
            "own-scope": accepted("::lib::R::S::T::f()", "text"),
        },
    ),
    "alias-reached-through-a-route-ignores-a-wildcards-scope-of-that-name": Scenario(
        modules={"lib": _REGION, "o": f"{_ALIAS_PARALLEL}\n"},
        header=("import lib\nimport o::*",),
        probes={"route": accepted("lib::R::A::f()", "int")},
    ),
}

_HIDDEN_TARGET_MEMBER = "import lib hiding R::S::T::f"
_SCENARIOS |= {
    f"hidden-target-member-beneath-an-alias-reached-through-{name}": Scenario(
        modules={"lib": _NESTED_LIB},
        header=(_HIDDEN_TARGET_MEMBER, use),
        probes={"static": rejected(probe, HiddenMemberError, probe[:-2])},
    )
    for name, (use, probe) in {
        "a-use-of-the-region": ("use lib::R", "R::A::f()"),
        "a-use-of-the-region-as": ("use lib::R as Q", "Q::A::f()"),
        "a-use-of-the-alias-as": ("use lib::R::A as B", "B::f()"),
        "a-wildcard-use-of-the-region": ("use lib::R::*", "A::f()"),
        "a-use-of-the-module-as": ("use lib as L", "L::R::A::f()"),
    }.items()
}


class TestPathBeneathAnAliasIsReadThroughItsTarget:
    """A path beneath an alias reads its target as written where the alias is declared.

    However the alias was reached -- ``::``, a module route, an import or the
    enclosing regions -- the target's own spelling decides, not the reader's.
    """

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_semantics_and_entry_boundaries(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize(
        ("header", "probes"),
        [
            pytest.param("import ca::*", {"wildcard": "Ta::f()"}, id="wildcard"),
            pytest.param("import ra", {"route": "ra::Ta::f()"}, id="route"),
            pytest.param("import ra", {"anchored-route": "/ra::Ta::f()"}, id="anchored-route"),
        ],
    )
    def test_alias_leading_back_to_itself_across_modules_reaches_nothing(
        self, tmp_path: Path, header: str, probes: dict[str, str]
    ) -> None:
        """Aliases of each other across modules end the read at the first revisit."""
        assert_verdicts(
            tmp_path,
            _CYCLIC,
            (header,),
            {key: rejected(text, UnknownMemberError, text[:-2]) for key, text in probes.items()},
            part="file",
        )

    @pytest.mark.parametrize("hidden", ["R::S::T::f", "R::A::f"])
    def test_hidden_path_beneath_an_alias_is_not_read_from_the_decoy(
        self, tmp_path: Path, hidden: str
    ) -> None:
        decoy = 'scope S\n  record T\n    x: int\nend S\n\ndef S::T::f() -> text = ""\n'
        assert_verdicts(
            tmp_path,
            {"lib": decoy + _REGION},
            (f"import lib hiding {hidden}",),
            {
                key: rejected(text, HiddenMemberError, text[:-2])
                for key, text in {
                    "route": "lib::R::A::f()",
                    "anchored-route": "/lib::R::A::f()",
                }.items()
            },
            part="file",
        )

    @pytest.mark.parametrize(
        "source",
        [
            "type A = A::B\nA::f()",
            "scope P\n  type B = Q::C::D\nend P\n\nscope Q\n  type C = P::B\nend Q",
        ],
        ids=["root", "across-scopes"],
    )
    def test_re_entrant_alias_target_is_rejected(self, source: str) -> None:
        """A scope error, not a crash."""
        with pytest.raises(AglScopeError):
            resolve_entry(source)

    def test_target_beneath_its_own_alias_is_read_at_an_outer_step(self, tmp_path: Path) -> None:
        """Beneath the alias itself the target reaches nothing, so an outer step selects it."""
        assert_verdicts(
            tmp_path,
            {},
            (
                "record A\n  x: int\nrecord A::B\n  y: int\ndef A::B::f() -> int = 1\n\n"
                "scope P\n  type A = A::B\nend P\n",
            ),
            {
                "outer": accepted("P::A::f()", "int"),
                "type": accepted("def g(a: P::A) -> int = 1\ng(A::B(y = 2))", "int"),
            },
        )

    @pytest.mark.parametrize(
        ("header", "probe"),
        [
            pytest.param("import ga", "ga::Ta::f()", id="route"),
            pytest.param("import ga", "/ga::Ta::f()", id="anchored-route"),
            pytest.param("import ga::*", "Ta::f()", id="wildcard"),
        ],
    )
    def test_aliases_growing_each_others_targets_across_modules_reach_nothing(
        self, tmp_path: Path, header: str, probe: str
    ) -> None:
        """Each alias's target passes beneath the other, never terminating: no crash."""
        phase, error, _, _ = inline_verdict(tmp_path, {**_GROWING, "entry": f"{header}\n{probe}"})
        assert (phase, error) == ("scope", UnknownMemberError)

    def test_use_hiding_a_path_beneath_a_hidden_alias_is_rejected(self, tmp_path: Path) -> None:
        assert_verdicts(
            tmp_path,
            {"lib": _NESTED_LIB},
            (_HIDDEN_TARGET_MEMBER, "use lib::R::* hiding A::f"),
            {"use": rejected("R::A::g()", HiddenMemberError, "use lib::R::* hiding A::f")},
            part="file",
        )
