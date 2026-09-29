"""An applied type owner selects only its own members.

``Owner[A]::member`` selects an inline member of an enum owner (directly or
through an alias of it), or a name the enum references from its own path. A
declaration merely nested beneath the owner's path (``record E::Inner``) is
reached only through the owner's bare path: applying the owner to type
arguments first is a static type error, alike for a constructor call or
reference, a pattern, an annotation and an alias target --
in file mode and every REPL grouping (see :mod:`tests.agl.qualifier_support`).
A pattern spelling the nested declaration through the owner's bare path
matches that declaration alone, never one of the enum's members.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError
from tests.agl.qualifier_support import FilePhase, assert_verdicts_for_grouping, grouping_params

_ACCEPTED: tuple[FilePhase, type[BaseException] | type[None]] = ("accepted", type(None))
_REJECTED: tuple[FilePhase, type[BaseException] | type[None]] = ("typecheck", AglTypeError)

_OWNERS = {
    "enum": "enum E[T]\n  | A(a: T)\n  | B\nrecord E::Inner\n  x: int",
    "record": "record E[T]\n  v: T\nrecord E::Inner\n  x: int",
    "record-alias": "record Box[T]\n  v: T\ntype E[T] = Box[T]\nrecord E::Inner\n  x: int",
    "enum-alias": "enum F[T]\n  | A(a: T)\n  | B\ntype E[T] = F[T]\nrecord E::Inner\n  x: int",
}

_APPLIED_PROBES = {
    "call": "E[int]::Inner(x = 1)",
    "reference": "E[int]::Inner",
    "pattern": "let i = E::Inner(x = 1)\ncase i of\n  | E[int]::Inner(x) => x",
    "annotation": "fn(p: E[int]::Inner) => 1",
    "alias": "type T = E[int]::Inner\nfn(p: T) => 1",
}
_APPLIED_SPANS = {
    "call": "E[int]::Inner(x = 1)",
    "reference": "E[int]::Inner",
    "pattern": "E[int]::Inner(x)",
    "annotation": "p: E[int]::Inner",
    "alias": "type T = E[int]::Inner",
}
# The nested declaration's own bare path selects it, whatever its owner is.
_PLAIN_PROBES = {
    "call": "E::Inner(x = 1)",
    "pattern": "let i = E::Inner(x = 1)\ncase i of\n  | E::Inner(x) => x",
    "annotation": "fn(p: E::Inner) => p.x",
}
_PLAIN_IDENTITIES = {
    "call": "record E::Inner\n  x: int",
    "pattern": "int",
    "annotation": "E::Inner -> int",
}


class TestNestedDeclarationBeneathAnAppliedOwner:
    """``E[A]::Inner`` never reaches a declaration merely nested beneath ``E``."""

    @pytest.mark.parametrize("owner", _OWNERS)
    @pytest.mark.parametrize("sizes", grouping_params(2))
    def test_applied_owner_is_rejected(
        self, tmp_path: Path, owner: str, sizes: tuple[int, ...]
    ) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            (_OWNERS[owner],),
            sizes,
            _APPLIED_PROBES,
            dict.fromkeys(_APPLIED_PROBES, _REJECTED),
            span_texts=_APPLIED_SPANS,
        )

    @pytest.mark.parametrize("owner", _OWNERS)
    @pytest.mark.parametrize("sizes", grouping_params(2))
    def test_bare_owner_path_is_accepted(
        self, tmp_path: Path, owner: str, sizes: tuple[int, ...]
    ) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            (_OWNERS[owner],),
            sizes,
            _PLAIN_PROBES,
            dict.fromkeys(_PLAIN_PROBES, _ACCEPTED),
            expected_identities=_PLAIN_IDENTITIES,
        )

    @pytest.mark.parametrize("sizes", grouping_params(2))
    def test_nested_record_and_member_patterns_never_match_each_other(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        probes = {
            "nested-against-member": "let i = E::A(a = 1)\ncase i of\n  | E::Inner(x) => x",
            "member-against-nested": "let i = E::Inner(x = 1)\ncase i of\n  | E::A(a) => 1",
        }
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            (_OWNERS["enum"],),
            sizes,
            probes,
            dict.fromkeys(probes, _REJECTED),
            span_texts={"nested-against-member": "E::Inner(x)", "member-against-nested": "E::A(a)"},
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
    "own-path-referenced": "E[int]::Item(y = 1)",
}


class TestAppliedOwnerSelectsItsOwnMembers:
    """An inline member, through the enum or its alias, and an own-path reference."""

    @pytest.mark.parametrize("sizes", grouping_params(2))
    def test_every_position(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            (_MEMBER_OWNERS,),
            sizes,
            _MEMBER_PROBES,
            dict.fromkeys(_MEMBER_PROBES, _ACCEPTED),
            expected_identities={
                "inline-call": "record E::A[int]\n  a: int",
                "inline-pattern": "int",
                "inline-is": "bool",
                "inline-annotation": "E::A[int] -> int",
                "alias-call": "record E::A[int]\n  a: int",
                "alias-annotation": "E::A[int] -> int",
                "own-path-referenced": "record E::Item\n  y: int",
            },
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

    @pytest.mark.parametrize("sizes", grouping_params(3))
    def test_nested_declaration_is_rejected(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            _LIBRARY,
            ("import lib", "use lib::*"),
            sizes,
            _ROUTED_PROBES,
            dict.fromkeys(_ROUTED_PROBES, _REJECTED),
            span_texts={
                "call": "lib::E[int]::Inner(x = 1)",
                "pattern": "lib::E[int]::Inner(x)",
                "annotation": "p: lib::E[int]::Inner",
                "used-call": "E[int]::Inner(x = 1)",
            },
        )
