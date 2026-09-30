"""The scope steps a bare name tries, in every namespace and position.

Inside ``scope S1::S2`` a bare ``p`` tries ``S1::S2::p``, then ``S1::p``,
then ``p``; the first step finding a declaration decides. At one step this
module's own declaration beats every contribution anchored at or above it.
So a ``use`` in a nearer region beats a same-named declaration of an
enclosing scope, and an own declaration beats a ``use`` of its own step, for
a constructor call or reference, an enum member's terminal name, a type name
(annotation, alias target, type argument, applied type), a qualifier's
leading segment, a value binding and a method receiver alike -- in the file part
and every REPL grouping (see :mod:`tests.agl.qualifier_support`). A pattern
or ``is`` test's scrutinee selects among every candidate its spelling reaches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agl.qualifier_support import (
    Part,
    Phase,
    all_groupings,
    assert_verdicts,
    nonconstant_in_file,
    probe_table,
    verdict_parts,
)

_ACCEPTED: tuple[Phase, type[BaseException] | type[None]] = ("accepted", type(None))

# ``s`` contributes a same-spelled record, enum (with a member ``A``), generic
# enum and binding for each of the own declarations below.
_CONTRIBUTING_SCOPE = (
    "scope s\n"
    "  record R\n"
    "    y: text\n"
    "  enum E = A(y: text) | B\n"
    "  enum G[T] = GA(y: T)\n"
    '  let x = "s"\n'
    "end s"
)
_OWN = "record R\n  x: int\nenum E = A(x: int) | C\nenum G[T] = GB(x: T)\nlet x = 1"


def _indented(text: str, depth: int) -> str:
    return "\n".join(" " * depth + line for line in text.split("\n"))


def _in_region(use: str, body: str, outer: str | None = None) -> str:
    """Write *use* and *body* inside region ``r``, itself inside scope *outer* when set."""
    region = f"scope r\n{_indented(f'{use}\n{body}', 2)}\nend r"
    return region if outer is None else f"scope {outer}\n{_indented(region, 2)}\nend {outer}"


# Probes: a ``use`` and a body binding ``v`` written inside region ``r``. ``@``
# is the qualifier reaching the own declarations.
_NEARER_CONSTRUCTOR_PROBES: dict[str, tuple[str, str]] = {
    "call": ("use s::*", 'let v = R(y = "a")'),
    "reference": ("use s::*", "let v = R"),
    "member-call": ("use s::E::*", 'let v = A(y = "a")'),
    "member-reference": ("use s::E::*", "let v = A"),
    # A pattern's scrutinee selects among every candidate its spelling
    # reaches, so an own scrutinee still selects the own constructor.
    "pattern": ("use s::*", "let v = case @R(x = 1) of\n  | R(x) => x"),
    "member-pattern": (
        "use s::E::*",
        "def pick(e: @E) -> int =\n  case e of\n    | A(x) => x\n    | C => 0\nlet v = pick(@E::C)",
    ),
    "member-is": ("use s::E::*", "let e: @E = @E::C\nlet v = e is A"),
    "contributed-pattern": ("use s::*", 'let v = case s::R(y = "a") of\n  | R(y) => y'),
    "contributed-member-is": ("use s::E::*", "let e: s::E = s::E::B\nlet v = e is A"),
}
_NEARER_OTHER_PROBES: dict[str, tuple[str, str]] = {
    "annotation": ("use s::*", "let v = fn(e: R) => e.y"),
    "alias": ("use s::*", "type T = R\nlet v = fn(t: T) => t.y"),
    "type-argument": ("use s::*", "let v = fn(o: Option[R]) => 1"),
    "applied-type": ("use s::*", "let v = fn(g: G[int]) => 1"),
    "qualifier-head": ("use s::*", 'let v = E::A(y = "a")'),
    "qualifier-member": ("use s::*", "let v = E::B"),
    "value-binding": ("use s::*", "let v = x"),
    "receiver": ("use s::*", 'def R::m(self) -> text = self.y\nlet v = R(y = "a").m()'),
    "member-receiver": (
        "use s::*",
        'def E::A::m(self) -> text = self.y\nlet v = E::A(y = "a").m()',
    ),
}
_NEARER_IDENTITIES = {
    "call": "record s::R\n  y: text",
    "reference": "text -> s::R",
    "member-call": "record s::E::A\n  y: text",
    "member-reference": "text -> s::E::A",
    "pattern": "int",
    "member-pattern": "int",
    "member-is": "bool",
    "contributed-pattern": "text",
    "contributed-member-is": "bool",
    "annotation": "s::R -> text",
    "alias": "s::R -> text",
    "type-argument": "std/option::Option[s::R] -> int",
    "applied-type": "s::G[int] -> int",
    "qualifier-head": "record s::E::A\n  y: text",
    "qualifier-member": "record s::E::B",
    "value-binding": "text",
    "receiver": "text",
    "member-receiver": "text",
}

# The probes whose region binding ``v`` a file rejects: its initializer is
# not constant.
_NONCONSTANT = frozenset(
    {
        "pattern",
        "member-pattern",
        "member-is",
        "contributed-pattern",
        "contributed-member-is",
        "annotation",
        "alias",
        "type-argument",
        "applied-type",
        "receiver",
        "member-receiver",
    }
)

# Where the own declarations enclosing region ``r`` sit: the header declaring
# them, the scope ``r`` is written in, and the qualifier reaching them.
_ENCLOSING = {
    "module-root": (_OWN, None, "::"),
    "enclosing-scope": (f"scope q\n{_indented(_OWN, 2)}\nend q", "q", "q::"),
}


def _nearer_probes(probes: dict[str, tuple[str, str]], placement: str) -> dict[str, str]:
    _header, outer, own = _ENCLOSING[placement]
    path = "r::" if outer is None else f"{outer}::r::"
    return {
        key: f"{_in_region(use, body.replace('@', own), outer)}\n{path}v"
        for key, (use, body) in probes.items()
    }


class TestNearerUseBeatsAnEnclosingDeclaration:
    """A ``use`` in region ``r`` beats an own declaration of an enclosing scope."""

    @pytest.mark.parametrize("placement", _ENCLOSING)
    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_constructors(self, tmp_path: Path, placement: str, part: Part) -> None:
        self._assert_accepted(
            tmp_path,
            placement,
            _NEARER_CONSTRUCTOR_PROBES,
            part=part,
        )

    @pytest.mark.parametrize("placement", _ENCLOSING)
    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_other_positions(self, tmp_path: Path, placement: str, part: Part) -> None:
        self._assert_accepted(
            tmp_path,
            placement,
            _NEARER_OTHER_PROBES,
            part=part,
        )

    @staticmethod
    def _assert_accepted(
        tmp_path: Path,
        placement: str,
        written: dict[str, tuple[str, str]],
        part: Part,
    ) -> None:
        probes = _nearer_probes(written, placement)
        assert_verdicts(
            tmp_path,
            {},
            (_CONTRIBUTING_SCOPE, _ENCLOSING[placement][0]),
            nonconstant_in_file(
                probe_table(
                    probes,
                    dict.fromkeys(probes, _ACCEPTED),
                    identities={key: _NEARER_IDENTITIES[key] for key in probes},
                ),
                _NONCONSTANT,
            ),
            part=part,
        )


def _own_identities(own: str) -> dict[str, str]:
    """Each accepted probe's identity, *own* being the own declarations' qualifier."""
    return {
        "call": f"record {own}R\n  x: int",
        "reference": f"int -> {own}R",
        "pattern": "int",
        "member-call": f"record {own}E::A\n  x: int",
        "member-reference": f"int -> {own}E::A",
        "member-pattern": "int",
        "member-is": "bool",
        "annotation": f"{own}R -> int",
        "alias": f"{own}R -> int",
        "type-argument": f"std/option::Option[{own}R] -> int",
        "applied-type": f"{own}G[int] -> int",
        "qualifier-head": f"record {own}E::A\n  x: int",
        "value-binding": "int",
        "receiver": "int",
        "member-receiver": "int",
    }


# Region ``r`` declares the own declarations beside its ``use``. An enum
# injects its members' terminal names at its own step, so inside ``r`` the own
# member beats the ``use``'s; a member only the ``use`` reaches is the
# contributed member.
_SAME_REGION_PROBES: dict[str, tuple[str, str]] = {
    "call": ("use s::*", "let v = R(x = 1)"),
    "reference": ("use s::*", "let v = R"),
    "pattern": ("use s::*", "let v = case r::R(x = 1) of\n  | R(x) => x"),
    "member-call": ("use s::E::*", "let v = A(x = 1)"),
    "member-reference": ("use s::E::*", "let v = A"),
    "contributed-member": ("use s::E::*", "let v = B"),
    "annotation": ("use s::*", "let v = fn(e: R) => e.x"),
    "alias": ("use s::*", "type T = R\nlet v = fn(t: T) => t.x"),
    "type-argument": ("use s::*", "let v = fn(o: Option[R]) => 1"),
    "applied-type": ("use s::*", "let v = fn(g: G[int]) => 1"),
    "qualifier-head": ("use s::*", "let v = E::A(x = 1)"),
    "qualifier-member": ("use s::*", "let v = E::B"),
    "value-binding": ("use s::*", "let v = x"),
    "receiver": ("use s::*", "def R::m(self) -> int = self.x\nlet v = R(x = 3).m()"),
    "member-receiver": ("use s::*", "def E::A::m(self) -> int = self.x\nlet v = E::A(x = 3).m()"),
}
_SAME_REGION_IDENTITIES = {
    **_own_identities("r::"),
    "contributed-member": "record s::E::B",
    "qualifier-member": "record s::E::B",
}


class TestOwnDeclarationBeatsAUseOfItsStep:
    """In region ``r``, an own declaration of ``r`` beats the region's ``use``."""

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_every_position(self, tmp_path: Path, part: Part) -> None:
        probes = {
            key: f"{_in_region(use, body)}\nr::v"
            for key, (use, body) in _SAME_REGION_PROBES.items()
        }
        assert_verdicts(
            tmp_path,
            {},
            (_CONTRIBUTING_SCOPE, f"scope r\n{_indented(_OWN, 2)}\nend r"),
            nonconstant_in_file(
                probe_table(
                    probes,
                    dict.fromkeys(probes, _ACCEPTED),
                    identities={key: _SAME_REGION_IDENTITIES[key] for key in probes},
                ),
                _NONCONSTANT,
            ),
            part=part,
        )


# At the module root, the ``use``s precede the scope they open in a file; in
# the REPL a ``use`` naming a scope a later entry declares is rejected, so the
# ``use``s share the scope's entry.
_ROOT_USES_HEADER = ("use s::*", "use s::E::*", _CONTRIBUTING_SCOPE, _OWN)
_ROOT_USES_LEGAL = frozenset(
    sizes for sizes in all_groupings(len(_ROOT_USES_HEADER) + 1) if sizes[0] >= 3
)
_ROOT_PROBES = {
    "call": "R(x = 1)",
    "reference": "R",
    "pattern": "case R(x = 1) of\n  | R(x) => x",
    "member-call": "A(x = 1)",
    "member-reference": "A",
    "member-pattern": (
        "def pick(e: E) -> int =\n  case e of\n    | A(x) => x\n    | C => 0\npick(E::C)"
    ),
    "member-is": "let e: E = E::C\ne is A",
    "annotation": "fn(e: R) => e.x",
    "alias": "type T = R\nfn(t: T) => t.x",
    "type-argument": "fn(o: Option[R]) => 1",
    "applied-type": "fn(g: G[int]) => 1",
    "qualifier-head": "E::A(x = 1)",
    "value-binding": "x",
    "receiver": "def R::m(self) -> int = self.x\nR(x = 3).m()",
    "member-receiver": "def E::A::m(self) -> int = self.x\nE::A(x = 3).m()",
}


class TestRootDeclarationBeatsARootUse:
    """At the module root, an own declaration beats a ``use`` of the same step."""

    @pytest.mark.parametrize("part", verdict_parts(5))
    def test_every_position(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            {},
            _ROOT_USES_HEADER,
            probe_table(
                _ROOT_PROBES, dict.fromkeys(_ROOT_PROBES, _ACCEPTED), identities=_own_identities("")
            ),
            legal=_ROOT_USES_LEGAL,
            part=part,
        )


_TAIL_LIB = {"lib": "record R\n  z: bool\nenum E = A(z: bool) | D\n"}
_NEAREST_PROBES = {
    "call": 'let v = R(y = "a")',
    "annotation": "let v = fn(e: R) => e.y",
    "pattern": 'let v = case s::R(y = "a") of\n  | R(y) => y',
    "qualifier-head": 'let v = E::A(y = "a")',
}


class TestNearestContributionDecides:
    """With no own declaration, a region's ``use`` beats a farther root import tail."""

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_every_position(self, tmp_path: Path, part: Part) -> None:
        probes = {
            key: f"scope r\n  use s::*\n{_indented(body, 2)}\nend r\nr::v"
            for key, body in _NEAREST_PROBES.items()
        }
        assert_verdicts(
            tmp_path,
            _TAIL_LIB,
            ("import lib::*", _CONTRIBUTING_SCOPE),
            nonconstant_in_file(
                probe_table(
                    probes,
                    dict.fromkeys(probes, _ACCEPTED),
                    identities={
                        "call": "record s::R\n  y: text",
                        "annotation": "s::R -> text",
                        "pattern": "text",
                        "qualifier-head": "record s::E::A\n  y: text",
                    },
                ),
                _NONCONSTANT,
            ),
            part=part,
        )
