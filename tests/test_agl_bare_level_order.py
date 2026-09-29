"""One lookup order for a bare name, in every namespace and position.

This module's own declaration at any enclosing scope -- the module root
included -- beats every contribution (a ``use``, an import tail, the
prelude); among contributions the nearest layer decides. So a ``use`` in a
nearer scope region never reaches past a same-named declaration of an
enclosing scope, for a constructor call or reference, a pattern, an ``is``
test, an enum member's terminal name, a type name (annotation, alias target,
type argument, applied type), a qualifier's leading segment, a value binding
and a method receiver alike -- in file mode and every REPL grouping (see
:mod:`tests.agl.qualifier_support`). A pattern or ``is`` test's scrutinee
selects among the nearest own declaration and the nearest contribution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.scope.symbols import UnknownMemberError
from tests.agl.qualifier_support import (
    FilePhase,
    all_groupings,
    assert_verdicts_for_grouping,
    grouping_params,
)

_ACCEPTED: tuple[FilePhase, type[BaseException] | type[None]] = ("accepted", type(None))
_UNKNOWN_MEMBER: tuple[FilePhase, type[BaseException] | type[None]] = (
    "scope",
    UnknownMemberError,
)

# ``s`` contributes a same-spelled record, enum (with a member ``A``), generic
# enum and binding for each of the enclosing declarations below.
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


# Probes: a ``use`` and a body written inside region ``r``, then the entry's
# final expression. ``@`` is the qualifier reaching the enclosing
# declarations, ``%`` region ``r``'s path from the module root.
_CONSTRUCTOR_PROBES: dict[str, tuple[str, str, str]] = {
    "call": ("use s::*", "let v = R(x = 1)", "%v"),
    "reference": ("use s::*", "let v = R", "%v"),
    "pattern": ("use s::*", "let v = case @R(x = 1) of\n  | R(x) => x", "%v"),
    "member-pattern": (
        "use s::E::*",
        "def pick(e: @E) -> int =\n  case e of\n    | A(x) => x\n    | C => 0",
        "%pick(@E::C)",
    ),
    "member-is": ("use s::E::*", "let e: @E = @E::C\nlet v = e is A", "%v"),
    # A pattern's scrutinee selects among the enclosing declaration and the
    # nearest contribution, so the ``use`` still reaches a contributed scrutinee.
    "contributed-pattern": ("use s::*", 'let v = case s::R(y = "a") of\n  | R(y) => y', "%v"),
    "contributed-member-pattern": (
        "use s::E::*",
        'let v = case s::E::A(y = "a") of\n  | A(y) => y',
        "%v",
    ),
    "contributed-member-is": ("use s::E::*", "let e: s::E = s::E::B\nlet v = e is A", "%v"),
}
_OTHER_PROBES: dict[str, tuple[str, str, str]] = {
    "annotation": ("use s::*", "let v = fn(e: R) => e.x", "%v"),
    "alias": ("use s::*", "type T = R\nlet v = fn(t: T) => t.x", "%v"),
    "type-argument": ("use s::*", "let v = fn(o: Option[R]) => 1", "%v"),
    "applied-type": ("use s::*", "let v = fn(g: G[int]) => 1", "%v"),
    "qualifier-head": ("use s::*", "let v = E::A(x = 1)", "%v"),
    "qualifier-head-miss": ("use s::*", "let v = E::B", "%v"),
    "value-binding": ("use s::*", "let v = x", "%v"),
    "receiver": ("use s::*", "def R::m(self) -> int = self.x", "@R(x = 3).m()"),
    "member-receiver": ("use s::*", "def E::A::m(self) -> int = self.x", "@E::A(x = 3).m()"),
}
# Only the module root makes an enum member's terminal name a bare value; in a
# named scope the member stays at its enum's own path, so the ``use`` supplies
# the bare value there.
_MEMBER_VALUE_PROBES: dict[str, tuple[str, str, str]] = {
    "member-call": ("use s::E::*", "let v = A(x = 1)", "%v"),
    "member-reference": ("use s::E::*", "let v = A", "%v"),
}
_SCOPED_MEMBER_VALUE_PROBES: dict[str, tuple[str, str, str]] = {
    "member-call": ("use s::E::*", 'let v = A(y = "a")', "%v"),
    "member-reference": ("use s::E::*", "let v = A", "%v"),
}


def _identities(own: str) -> dict[str, str]:
    """Each accepted probe's identity, *own* being the enclosing declarations' qualifier."""
    return {
        "call": f"record {own}R\n  x: int",
        "reference": f"int -> {own}R",
        "pattern": "int",
        "member-call": f"record {own}E::A\n  x: int",
        "member-reference": f"int -> {own}E::A",
        "member-pattern": "int",
        "member-is": "bool",
        "contributed-pattern": "text",
        "contributed-member-pattern": "text",
        "contributed-member-is": "bool",
        "annotation": f"{own}R -> int",
        "alias": f"{own}R -> int",
        "type-argument": f"std/option::Option[{own}R] -> int",
        "applied-type": f"{own}G[int] -> int",
        "qualifier-head": f"record {own}E::A\n  x: int",
        "value-binding": "int",
        "receiver": "int",
        "member-receiver": "int",
    }


_SCOPED_MEMBER_IDENTITIES = {
    "member-call": "record s::E::A\n  y: text",
    "member-reference": "text -> s::E::A",
}


class _Placement:
    """Where the enclosing declarations sit, relative to region ``r`` the probe opens.

    *header* declares them. Region ``r`` is written at the module root, or
    inside scope *outer* when set. *own* is the qualifier reaching the
    enclosing declarations from ``r``, as an identity renders it too
    (empty at the module root).
    """

    def __init__(self, header: str, outer: str | None, own: str) -> None:
        self.header = (_CONTRIBUTING_SCOPE, header)
        self.outer = outer
        self.own = own

    def probes(self, probes: dict[str, tuple[str, str, str]]) -> dict[str, str]:
        """Write each ``(use, body, tail)`` of *probes* at this placement."""
        spelled = self.own or "::"
        path = "r::" if self.outer is None else f"{self.outer}::r::"
        written = {}
        for key, (use, body, tail) in probes.items():
            region = f"scope r\n{_indented(f'{use}\n{body}', 2)}\nend r"
            if self.outer is not None:
                region = f"scope {self.outer}\n{_indented(region, 2)}\nend {self.outer}"
            written[key] = f"{region}\n{tail.replace('%', path)}".replace("@", spelled)
        return written


_PLACEMENTS = {
    "module-root": _Placement(_OWN, None, ""),
    "enclosing-scope": _Placement(f"scope q\n{_indented(_OWN, 2)}\nend q", "q", "q::"),
    "same-region": _Placement(f"scope r\n{_indented(_OWN, 2)}\nend r", None, "r::"),
}


class TestEnclosingDeclarationBeatsANearerUse:
    """An own declaration of an enclosing scope beats a ``use`` in the nearer region ``r``."""

    @pytest.mark.parametrize("placement", _PLACEMENTS)
    @pytest.mark.parametrize("sizes", grouping_params(3))
    def test_constructors(self, tmp_path: Path, placement: str, sizes: tuple[int, ...]) -> None:
        where = _PLACEMENTS[placement]
        identities = _identities(where.own)
        members = _MEMBER_VALUE_PROBES if not where.own else _SCOPED_MEMBER_VALUE_PROBES
        if where.own:
            identities.update(_SCOPED_MEMBER_IDENTITIES)
        probes = where.probes({**_CONSTRUCTOR_PROBES, **members})
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            where.header,
            sizes,
            probes,
            dict.fromkeys(probes, _ACCEPTED),
            expected_identities={key: identities[key] for key in probes},
        )

    @pytest.mark.parametrize("placement", _PLACEMENTS)
    @pytest.mark.parametrize("sizes", grouping_params(3))
    def test_other_positions(self, tmp_path: Path, placement: str, sizes: tuple[int, ...]) -> None:
        where = _PLACEMENTS[placement]
        identities = _identities(where.own)
        probes = where.probes(_OTHER_PROBES)
        expected = dict.fromkeys(probes, _ACCEPTED)
        expected["qualifier-head-miss"] = _UNKNOWN_MEMBER
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            where.header,
            sizes,
            probes,
            expected,
            span_texts={"qualifier-head-miss": "E::B"},
            expected_identities={key: identities[key] for key in probes if key in identities},
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
    """At the module root, an own declaration beats a ``use`` of the same level."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_ROOT_USES_HEADER) + 1))
    def test_every_position(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _ROOT_USES_HEADER,
            sizes,
            _ROOT_PROBES,
            dict.fromkeys(_ROOT_PROBES, _ACCEPTED),
            expected_identities=_identities(""),
            expected_legal_groupings=_ROOT_USES_LEGAL,
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

    @pytest.mark.parametrize("sizes", grouping_params(3))
    def test_every_position(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {
            key: f"scope r\n  use s::*\n{_indented(body, 2)}\nend r\nr::v"
            for key, body in _NEAREST_PROBES.items()
        }
        assert_verdicts_for_grouping(
            tmp_path,
            _TAIL_LIB,
            ("import lib::*", _CONTRIBUTING_SCOPE),
            sizes,
            probes,
            dict.fromkeys(probes, _ACCEPTED),
            expected_identities={
                "call": "record s::R\n  y: text",
                "annotation": "s::R -> text",
                "pattern": "text",
                "qualifier-head": "record s::E::A\n  y: text",
            },
        )
