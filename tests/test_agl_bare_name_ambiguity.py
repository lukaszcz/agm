"""One verdict, with one origin per contributing layer, for an ambiguous bare name.

A bare spelling several equally near candidates provide is ambiguous. What
the position asks decides the class: a value reference or constructor call
asks which constructor the name denotes (:class:`AmbiguousConstructorError`),
while a type position -- an annotation, a method receiver, a bare REPL type
entry, or a value naming a type with no constructor -- asks which type
(:class:`AmbiguousQualificationError`). A pattern selects by its scrutinee
and is never ambiguous. Each origin names the selected member declaration,
with the kind of the layer that contributed it: an import tail
(:class:`ImportedModuleOrigin`), a ``use`` declaration
(:class:`UseDeclarationOrigin`), or this module's own declaration
(:class:`DeclaredOrigin`). The module root is one level however REPL entries
group its imports and ``use`` declarations, so every grouping agrees with
file mode (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.scope.symbols import (
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    DeclaredOrigin,
    ImportedModuleOrigin,
    UseDeclarationOrigin,
)
from tests.agl.qualifier_support import (
    FilePhase,
    all_groupings,
    assert_verdicts_for_grouping,
    grouping_params,
)

_ACCEPTED: tuple[FilePhase, type[BaseException] | type[None]] = ("accepted", type(None))
_CONSTRUCTOR: tuple[FilePhase, type[BaseException] | type[None]] = (
    "scope",
    AmbiguousConstructorError,
)
_QUALIFICATION: tuple[FilePhase, type[BaseException] | type[None]] = (
    "scope",
    AmbiguousQualificationError,
)

_POINTS = {"a/lib": "record Point\n  x: int\n", "b/lib": "record Point\n  x: int\n"}
_COLORS = {"a/lib": "enum Color = Red | Green\n", "b/lib": "enum Color = Red | Blue\n"}

_TWO_TAILS = ("import a/lib::*", "import b/lib::*")
_TWO_USES = ("import a/lib", "import b/lib", "use a/lib::*", "use b/lib::*")
_TAIL_AND_USE = ("import a/lib::*", "import b/lib", "use b/lib::*")

_POINT_PROBES = {
    "call": "Point(x = 1)",
    "reference": "Point",
    "annotation": "fn(p: Point) => 1",
    "receiver": "def Point::m(self) -> int = 1",
    "pattern": "case a/lib::Point(x = 1) of\n  | Point(x) => x",
}
_POINT_EXPECTED = {
    "call": _CONSTRUCTOR,
    "reference": _CONSTRUCTOR,
    "annotation": _QUALIFICATION,
    "receiver": _QUALIFICATION,
    "pattern": _ACCEPTED,
}
_POINT_SPANS = {
    "call": "Point",
    "reference": "Point",
    "annotation": "Point",
    "receiver": "self",
}


def _origins(kind: type, *spellings: str) -> frozenset[tuple[type, str]]:
    return frozenset((kind, spelling) for spelling in spellings)


def _point_origins(a_kind: type, b_kind: type) -> dict[str, frozenset[tuple[type, str]]]:
    origins = _origins(a_kind, "a/lib::Point") | _origins(b_kind, "b/lib::Point")
    return {key: origins for key in _POINT_SPANS}


class TestAmbiguousBareRecordName:
    """``Point`` from two modules: one class per position, origins from each layer."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_TWO_TAILS) + 1))
    def test_two_import_tails(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            _POINTS,
            _TWO_TAILS,
            sizes,
            _POINT_PROBES,
            _POINT_EXPECTED,
            span_texts=_POINT_SPANS,
            expected_identities={"pattern": "int"},
            expected_origins=_point_origins(ImportedModuleOrigin, ImportedModuleOrigin),
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_TWO_USES) + 1))
    def test_two_uses(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            _POINTS,
            _TWO_USES,
            sizes,
            _POINT_PROBES,
            _POINT_EXPECTED,
            span_texts=_POINT_SPANS,
            expected_identities={"pattern": "int"},
            expected_origins=_point_origins(UseDeclarationOrigin, UseDeclarationOrigin),
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_TAIL_AND_USE) + 1))
    def test_import_tail_and_use(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            _POINTS,
            _TAIL_AND_USE,
            sizes,
            _POINT_PROBES,
            _POINT_EXPECTED,
            span_texts=_POINT_SPANS,
            expected_identities={"pattern": "int"},
            expected_origins=_point_origins(ImportedModuleOrigin, UseDeclarationOrigin),
        )


_RED_PROBES = {"value": "Red", "receiver": "def Red::m(self) -> int = 1"}
_RED_EXPECTED = {"value": _CONSTRUCTOR, "receiver": _QUALIFICATION}
_RED_SPANS = {"value": "Red", "receiver": "self"}


def _red_origins(kind: type) -> dict[str, frozenset[tuple[type, str]]]:
    origins = _origins(kind, "a/lib::Color::Red", "b/lib::Color::Red")
    return {key: origins for key in _RED_PROBES}


class TestAmbiguousBareEnumMember:
    """Two enums injecting ``Red``: the members are the origins, of the contributing layer."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_TWO_TAILS) + 1))
    def test_two_import_tails(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            _COLORS,
            _TWO_TAILS,
            sizes,
            _RED_PROBES,
            _RED_EXPECTED,
            span_texts=_RED_SPANS,
            expected_origins=_red_origins(ImportedModuleOrigin),
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_TWO_USES) + 1))
    def test_two_uses(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            _COLORS,
            _TWO_USES,
            sizes,
            _RED_PROBES,
            _RED_EXPECTED,
            span_texts=_RED_SPANS,
            expected_origins=_red_origins(UseDeclarationOrigin),
        )


# ``use`` declarations precede this module's own scopes in a file; in the REPL
# a ``use`` naming a scope declared by a later entry is rejected, so only the
# groupings with every setup item in the first entry are legal.
_LOCAL_SCOPES = (
    "scope s\n  record P\n    x: int\n  enum G[T] = A(x: T)\nend s",
    "scope t\n  record P\n    x: int\n  enum G[T] = B(x: T)\nend t",
)
_LOCAL_USES_HEADER = ("use s::*", "use t::*", *_LOCAL_SCOPES)
_LOCAL_USES_LEGAL = frozenset(
    sizes for sizes in all_groupings(len(_LOCAL_USES_HEADER) + 1) if sizes[0] >= 4
)


class TestAmbiguousNamesLocalUsesContribute:
    """Two ``use``s of this module's own scopes contribute ``use`` origins, never declared ones.

    A bare generic enum ``G`` names a type with no constructor, so as a value
    it asks the type question and gets the annotation's verdict.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_LOCAL_USES_HEADER) + 1))
    def test_every_position(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {
            "call": "P(x = 1)",
            "annotation": "fn(p: P) => 1",
            "generic-value": "G",
            "generic-applied-value": "G[int]",
            "generic-annotation": "fn(g: G[int]) => 1",
        }
        point = _origins(UseDeclarationOrigin, "s::P", "t::P")
        generic = _origins(UseDeclarationOrigin, "s::G", "t::G")
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _LOCAL_USES_HEADER,
            sizes,
            probes,
            {
                "call": _CONSTRUCTOR,
                "annotation": _QUALIFICATION,
                "generic-value": _QUALIFICATION,
                "generic-applied-value": _QUALIFICATION,
                "generic-annotation": _QUALIFICATION,
            },
            span_texts={
                "call": "P",
                "annotation": "P",
                "generic-value": "G",
                "generic-applied-value": "G",
                "generic-annotation": "G[int]",
            },
            expected_origins={
                "call": point,
                "annotation": point,
                "generic-value": generic,
                "generic-applied-value": generic,
                "generic-annotation": generic,
            },
            expected_legal_groupings=_LOCAL_USES_LEGAL,
        )


_LOCAL_ENUMS_HEADER = ("enum A = Some | Other", "enum B = Some | Third")


class TestAmbiguousLocalMembers:
    """Two local enums injecting ``Some``: declared origins, however entries group them."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_LOCAL_ENUMS_HEADER) + 1))
    def test_value_is_an_ambiguous_constructor(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _LOCAL_ENUMS_HEADER,
            sizes,
            {"value": "Some"},
            {"value": _CONSTRUCTOR},
            span_texts={"value": "Some"},
            expected_origins={"value": _origins(DeclaredOrigin, "A::Some", "B::Some")},
        )


_BOOMS = {"m/a": "exception Boom extends Exception\n", "m/b": "exception Boom extends Exception\n"}
_BOOM_USES = ("import m/a", "import m/b", "use m/a::*", "use m/b::*")
_BOOM_CATCH = "let _ = try\n  ()\ncatch Boom as e =>\n  ()"
_BOOM_EXTENDS = "exception Local extends Boom"


class TestAmbiguousBareExceptionName:
    """``Boom`` from two ``use``s: a call asks for the constructor, the rest for the type."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_BOOM_USES) + 1))
    def test_every_position(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        origins = _origins(UseDeclarationOrigin, "m/a::Boom", "m/b::Boom")
        probes = {
            "call": 'Boom(message = "m")',
            "annotation": "fn(x: Boom) => 1",
            "catch": _BOOM_CATCH,
            "extends": _BOOM_EXTENDS,
        }
        assert_verdicts_for_grouping(
            tmp_path,
            _BOOMS,
            _BOOM_USES,
            sizes,
            probes,
            {
                "call": _CONSTRUCTOR,
                "annotation": _QUALIFICATION,
                "catch": _QUALIFICATION,
                "extends": _QUALIFICATION,
            },
            span_texts={
                "call": "Boom",
                "annotation": "Boom",
                "catch": _BOOM_CATCH.split("\n", 2)[2],
                "extends": _BOOM_EXTENDS,
            },
            expected_origins=dict.fromkeys(probes, origins),
        )


_LOCAL_RED_PROBES = {
    "value": "Red",
    "receiver": "def Red::m(self) -> int = 1\nA::Red.m()",
}


class TestLocalMemberWinsItsLevel:
    """A local enum's ``Red`` beats an import tail's or a ``use``'s at the module root.

    A locally declared record, enum, enum member, or exception wins; the
    other ``Red`` is not a second candidate, for a value and a receiver alike.
    """

    @pytest.mark.parametrize("sizes", grouping_params(3))
    def test_over_an_import_tail(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            _COLORS,
            ("import a/lib::*", "enum A = Red | Blue"),
            sizes,
            _LOCAL_RED_PROBES,
            dict.fromkeys(_LOCAL_RED_PROBES, _ACCEPTED),
            expected_identities={"value": "record A::Red", "receiver": "int"},
        )

    @pytest.mark.parametrize("sizes", grouping_params(4))
    def test_over_a_use(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            _COLORS,
            ("import b/lib", "use b/lib::*", "enum A = Red | Blue"),
            sizes,
            _LOCAL_RED_PROBES,
            dict.fromkeys(_LOCAL_RED_PROBES, _ACCEPTED),
            expected_identities={"value": "record A::Red", "receiver": "int"},
        )
