"""An applied type owner selects only its inline members.

``Owner[A]::member`` selects an inline member of an enum owner (directly or
through an alias of it). Any other declaration beneath the owner's path -- a
nested ``record E::Inner`` or a member the enum references from its own path --
is reached only through the owner's bare path: applying the owner to type
arguments first is a scope error on the applied segment, alike for a
constructor call or reference, a pattern, an annotation and an alias target --
in the file part and every REPL grouping (see :mod:`tests.agl.qualifier_support`).
A pattern spelling the nested declaration through the owner's bare path
matches that declaration alone, never one of the enum's members.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError
from agm.agl.scope.symbols import TypeArgumentsError
from tests.agl.qualifier_support import (
    Part,
    Phase,
    assert_verdicts,
    probe_table,
    verdict_parts,
)

_ACCEPTED: tuple[Phase, type[BaseException] | type[None]] = ("accepted", type(None))
_REJECTED: tuple[Phase, type[BaseException] | type[None]] = ("scope", TypeArgumentsError)

_OWNERS = {
    "enum": "enum E[T]\n  | A(a: T)\n  | B\nrecord E::Inner\n  x: int",
    "record": "record E[T]\n  v: T\nrecord E::Inner\n  x: int",
    "record-alias": "import al::*\nrecord E::Inner\n  x: int",
    "enum-alias": "import fl::*\nrecord E::Inner\n  x: int",
}
_ALIASES = {
    "al": "record Box[T]\n  v: T\ntype E[T] = Box[T]",
    "fl": "enum F[T]\n  | A(a: T)\n  | B\ntype E[T] = F[T]",
}
"""Imported aliases ``E`` an owner declares its ``E::Inner`` beneath."""

_APPLIED_PROBES = {
    "call": "E[int]::Inner(x = 1)",
    "reference": "E[int]::Inner",
    "pattern": "let i = E::Inner(x = 1)\ncase i of\n  | E[int]::Inner(x) => x",
    "annotation": "fn(p: E[int]::Inner) => 1",
    "alias": "type T = E[int]::Inner\nfn(p: T) => 1",
}
_APPLIED_SPANS = dict.fromkeys(_APPLIED_PROBES, "E[int]")
# The nested declaration's own bare path selects it, whatever its owner is.
_PLAIN_PROBES = {
    "call": "E::Inner(x = 1)",
    "pattern": "let i = E::Inner(x = 1)\ncase i of\n  | E::Inner(x) => x",
    "annotation": "fn(p: E::Inner) => p.x",
}
_DECLARED_OWNERS = {"enum": "E", "record": "E", "record-alias": "E", "enum-alias": "E"}
"""The path each owner's ``E::Inner`` is declared beneath."""


def _plain_identities(owner: str) -> dict[str, str]:
    declared = _DECLARED_OWNERS[owner]
    return {
        "call": f"record {declared}::Inner\n  x: int",
        "pattern": "int",
        "annotation": f"{declared}::Inner -> int",
    }


class TestNestedDeclarationBeneathAnAppliedOwner:
    """``E[A]::Inner`` never reaches a declaration merely nested beneath ``E``."""

    @pytest.mark.parametrize("owner", _OWNERS)
    def test_applied_owner_is_rejected(self, tmp_path: Path, owner: str) -> None:
        assert_verdicts(
            tmp_path,
            _ALIASES,
            (_OWNERS[owner],),
            probe_table(
                _APPLIED_PROBES,
                dict.fromkeys(_APPLIED_PROBES, _REJECTED),
                span_texts=_APPLIED_SPANS,
            ),
        )

    @pytest.mark.parametrize("owner", _OWNERS)
    def test_bare_owner_path_is_accepted(self, tmp_path: Path, owner: str) -> None:
        assert_verdicts(
            tmp_path,
            _ALIASES,
            (_OWNERS[owner],),
            probe_table(
                _PLAIN_PROBES,
                dict.fromkeys(_PLAIN_PROBES, _ACCEPTED),
                identities=_plain_identities(owner),
            ),
        )

    def test_nested_record_and_member_patterns_never_match_each_other(self, tmp_path: Path) -> None:
        probes = {
            "nested-against-member": "let i = E::A(a = 1)\ncase i of\n  | E::Inner(x) => x",
            "member-against-nested": "let i = E::Inner(x = 1)\ncase i of\n  | E::A(a) => 1",
        }
        assert_verdicts(
            tmp_path,
            {},
            (_OWNERS["enum"],),
            probe_table(
                probes,
                dict.fromkeys(probes, ("typecheck", AglTypeError)),
                span_texts={
                    "nested-against-member": "E::Inner(x)",
                    "member-against-nested": "E::A(a)",
                },
            ),
        )


_MEMBER_OWNERS = (
    "enum E[T]\n  | A(a: T)\n  | B\n  | E::Item\nrecord E::Item\n  y: int\ntype Al[T] = E[T]"
)
_MEMBER_PROBES = {
    "inline-call": "E[int]::A(a = 1)",
    "inline-pattern": "let e: E[int] = E::B\ncase e of\n  | E[int]::A(a) => a\n  | _ => 0",
    "inline-is": "let e: E[int] = E::B\ne is E[int]::B",
    "inline-annotation": "fn(p: E[int]::A) => p.a",
    "alias-call": "Al[int]::A(a = 1)",
    "alias-annotation": "fn(p: Al[int]::A) => p.a",
}


class TestAppliedOwnerSelectsItsOwnMembers:
    """An inline member, through the enum or its alias."""

    def test_every_position(self, tmp_path: Path) -> None:
        assert_verdicts(
            tmp_path,
            {},
            (_MEMBER_OWNERS,),
            probe_table(
                _MEMBER_PROBES,
                dict.fromkeys(_MEMBER_PROBES, _ACCEPTED),
                identities={
                    "inline-call": "record E::A[int]\n  a: int",
                    "inline-pattern": "int",
                    "inline-is": "bool",
                    "inline-annotation": "E::A[int] -> int",
                    "alias-call": "record E::A[int]\n  a: int",
                    "alias-annotation": "E::A[int] -> int",
                },
            ),
        )

    def test_a_member_referenced_from_the_owner_path_is_not_inline(self, tmp_path: Path) -> None:
        probes = {"call": "E[int]::Item(y = 1)", "annotation": "fn(p: E[int]::Item) => p.y"}
        assert_verdicts(
            tmp_path,
            {},
            (_MEMBER_OWNERS,),
            probe_table(
                probes, dict.fromkeys(probes, _REJECTED), span_texts=dict.fromkeys(probes, "E[int]")
            ),
        )


_LIBRARY = {"lib": "enum E[T]\n  | A(a: T)\n  | B\nrecord E::Inner\n  x: int"}
_ROUTED_PROBES = {
    "call": "lib::E[int]::Inner(x = 1)",
    "pattern": "let i = lib::E::Inner(x = 1)\ncase i of\n  | lib::E[int]::Inner(x) => x",
    "annotation": "fn(p: lib::E[int]::Inner) => 1",
    "used-call": "E[int]::Inner(x = 1)",
}


class TestImportedAppliedOwner:
    """An imported owner, spelled through its route or a ``use``, selects alike."""

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_nested_declaration_is_rejected(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _LIBRARY,
            ("import lib", "use lib::*"),
            probe_table(
                _ROUTED_PROBES,
                dict.fromkeys(_ROUTED_PROBES, _REJECTED),
                span_texts=dict.fromkeys(_ROUTED_PROBES, "E[int]"),
            ),
            part=part,
        )
