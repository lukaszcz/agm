"""One verdict, with one origin per contributing layer, for an ambiguous bare name.

A bare spelling several candidates at one step provide is ambiguous. What
the position asks decides the class: a value reference or constructor call
asks which constructor the name denotes (:class:`AmbiguousConstructorError`),
while a type position -- an annotation, a method receiver, a bare REPL type
entry, or a value naming a type with no constructor -- asks which type
(:class:`AmbiguousQualificationError`). A pattern selects by its scrutinee
and is never ambiguous. Each origin names the selected member declaration,
with the kind of the layer that contributed it: an import tail
(:class:`ImportedModuleOrigin`), a ``use`` declaration
(:class:`UseDeclarationOrigin`), or this module's own declaration
(:class:`DeclaredOrigin`). The module root is one step however REPL entries
group its imports and ``use`` declarations, so every grouping agrees with
the file part (see :mod:`tests.agl.qualifier_support`).
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
    Part,
    Phase,
    all_groupings,
    assert_verdicts,
    probe_table,
    verdict_parts,
)

_ACCEPTED: tuple[Phase, type[BaseException] | type[None]] = ("accepted", type(None))
_CONSTRUCTOR: tuple[Phase, type[BaseException] | type[None]] = (
    "scope",
    AmbiguousConstructorError,
)
_QUALIFICATION: tuple[Phase, type[BaseException] | type[None]] = (
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

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_two_import_tails(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _POINTS,
            _TWO_TAILS,
            probe_table(
                _POINT_PROBES,
                _POINT_EXPECTED,
                span_texts=_POINT_SPANS,
                identities={"pattern": "int"},
                origins=_point_origins(ImportedModuleOrigin, ImportedModuleOrigin),
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(5))
    def test_two_uses(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _POINTS,
            _TWO_USES,
            probe_table(
                _POINT_PROBES,
                _POINT_EXPECTED,
                span_texts=_POINT_SPANS,
                identities={"pattern": "int"},
                origins=_point_origins(UseDeclarationOrigin, UseDeclarationOrigin),
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_import_tail_and_use(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _POINTS,
            _TAIL_AND_USE,
            probe_table(
                _POINT_PROBES,
                _POINT_EXPECTED,
                span_texts=_POINT_SPANS,
                identities={"pattern": "int"},
                origins=_point_origins(ImportedModuleOrigin, UseDeclarationOrigin),
            ),
            part=part,
        )


_RED_PROBES = {"value": "Red", "receiver": "def Red::m(self) -> int = 1"}
_RED_EXPECTED = {"value": _CONSTRUCTOR, "receiver": _QUALIFICATION}
_RED_SPANS = {"value": "Red", "receiver": "self"}


def _red_origins(kind: type) -> dict[str, frozenset[tuple[type, str]]]:
    origins = _origins(kind, "a/lib::Color::Red", "b/lib::Color::Red")
    return {key: origins for key in _RED_PROBES}


class TestAmbiguousBareEnumMember:
    """Two enums injecting ``Red``: the members are the origins, of the contributing layer."""

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_two_import_tails(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _COLORS,
            _TWO_TAILS,
            probe_table(
                _RED_PROBES,
                _RED_EXPECTED,
                span_texts=_RED_SPANS,
                origins=_red_origins(ImportedModuleOrigin),
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(5))
    def test_two_uses(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _COLORS,
            _TWO_USES,
            probe_table(
                _RED_PROBES,
                _RED_EXPECTED,
                span_texts=_RED_SPANS,
                origins=_red_origins(UseDeclarationOrigin),
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_alias_use_and_import_tail(self, tmp_path: Path, part: Part) -> None:
        """A ``use`` of an enum alias contributes its target's members, as that target."""
        assert_verdicts(
            tmp_path,
            {"one": "enum E\n  | Red\ntype C = E\n", "two": "enum F\n  | Red\n"},
            ("import one::{C}", "import two::*", "use C::*"),
            probe_table(
                {"value": "[Red]"},
                {"value": _CONSTRUCTOR},
                span_texts={"value": "Red"},
                origins={
                    "value": _origins(ImportedModuleOrigin, "two::F::Red")
                    | _origins(UseDeclarationOrigin, "one::E::Red")
                },
            ),
            part=part,
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

    @pytest.mark.parametrize("part", verdict_parts(5))
    def test_every_position(self, tmp_path: Path, part: Part) -> None:
        probes = {
            "call": "P(x = 1)",
            "annotation": "fn(p: P) => 1",
            "generic-value": "G",
            "generic-applied-value": "G[int]",
            "generic-annotation": "fn(g: G[int]) => 1",
        }
        point = _origins(UseDeclarationOrigin, "s::P", "t::P")
        generic = _origins(UseDeclarationOrigin, "s::G", "t::G")
        assert_verdicts(
            tmp_path,
            {},
            _LOCAL_USES_HEADER,
            probe_table(
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
                origins={
                    "call": point,
                    "annotation": point,
                    "generic-value": generic,
                    "generic-applied-value": generic,
                    "generic-annotation": generic,
                },
            ),
            legal=_LOCAL_USES_LEGAL,
            part=part,
        )


_LOCAL_ENUMS_HEADER = ("enum A = Some | Other", "enum B = Some | Third")


class TestAmbiguousLocalMembers:
    """Two local enums injecting ``Some``: declared origins, however entries group them."""

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_value_is_an_ambiguous_constructor(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            {},
            _LOCAL_ENUMS_HEADER,
            probe_table(
                {"value": "Some"},
                {"value": _CONSTRUCTOR},
                span_texts={"value": "Some"},
                origins={"value": _origins(DeclaredOrigin, "A::Some", "B::Some")},
            ),
            part=part,
        )


_BOOMS = {"m/a": "exception Boom extends Exception\n", "m/b": "exception Boom extends Exception\n"}
_BOOM_USES = ("import m/a", "import m/b", "use m/a::*", "use m/b::*")
_BOOM_CATCH = "let _ = try\n  ()\ncatch Boom as e =>\n  ()"
_BOOM_EXTENDS = "exception Local extends Boom"


class TestAmbiguousBareExceptionName:
    """``Boom`` from two ``use``s: a call asks for the constructor, the rest for the type."""

    @pytest.mark.parametrize("part", verdict_parts(5))
    def test_every_position(self, tmp_path: Path, part: Part) -> None:
        origins = _origins(UseDeclarationOrigin, "m/a::Boom", "m/b::Boom")
        probes = {
            "call": 'Boom(message = "m")',
            "annotation": "fn(x: Boom) => 1",
            "catch": _BOOM_CATCH,
            "extends": _BOOM_EXTENDS,
        }
        assert_verdicts(
            tmp_path,
            _BOOMS,
            _BOOM_USES,
            probe_table(
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
                origins=dict.fromkeys(probes, origins),
            ),
            part=part,
        )


class TestOwnScopeWinsOnlyInsideTheUse:
    """A ``use`` of an own scope and a module picks the own member; an import tail adds its."""

    def test_beside_a_region_import_tail(self, tmp_path: Path) -> None:
        assert_verdicts(
            tmp_path,
            {"lib": "def both() -> int = 10\n"},
            ("scope lib\n  def both() -> bool = true\nend lib",),
            probe_table(
                {"value": "scope r\n  import lib::*\n  use lib::*\n  let v = both()\nend r"},
                {"value": _QUALIFICATION},
                span_texts={"value": "both"},
                origins={
                    "value": _origins(ImportedModuleOrigin, "lib::both")
                    | _origins(UseDeclarationOrigin, "lib::both")
                },
            ),
        )


_LOCAL_RED_PROBES = {
    "value": "Red",
    "receiver": "def Red::m(self) -> int = 1\nA::Red.m()",
}


class TestLocalMemberWinsItsStep:
    """A local enum's ``Red`` beats an import tail's or a ``use``'s at the module root.

    A locally declared record, enum, enum member, or exception wins; the
    other ``Red`` is not a second candidate, for a value and a receiver alike.
    """

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_over_an_import_tail(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _COLORS,
            ("import a/lib::*", "enum A = Red | Blue"),
            probe_table(
                _LOCAL_RED_PROBES,
                dict.fromkeys(_LOCAL_RED_PROBES, _ACCEPTED),
                identities={"value": "record A::Red", "receiver": "int"},
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_over_a_use(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _COLORS,
            ("import b/lib", "use b/lib::*", "enum A = Red | Blue"),
            probe_table(
                _LOCAL_RED_PROBES,
                dict.fromkeys(_LOCAL_RED_PROBES, _ACCEPTED),
                identities={"value": "record A::Red", "receiver": "int"},
            ),
            part=part,
        )


_EVERYTHING = {
    "a/lib": "record Point\n  x: int\nenum Color = Red | Green\ndef f() -> int = 1\n",
    "b/lib": "record Point\n  x: int\nenum Color = Red | Blue\ndef f() -> int = 2\n",
}
_TAIL_USE_AND_TAIL = ("import a/lib::*", "import b/lib::*", "use a/lib::*")
_REACHED_TWICE_PROBES = {
    "call": "Point(x = 1)",
    "annotation": "fn(p: Point) => 1",
    "receiver": "def Point::m(self) -> int = 1",
    "member": "Red",
    "function": "f()",
}


def _reached_twice(path: str) -> frozenset[tuple[type, str]]:
    return _origins(ImportedModuleOrigin, f"a/lib::{path}", f"b/lib::{path}") | _origins(
        UseDeclarationOrigin, f"a/lib::{path}"
    )


class TestDeclarationReachedSeveralWays:
    """One declaration an import tail and a ``use`` both reach is one candidate.

    In an ambiguity it takes part in, its origins name both ways it was reached.
    """

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_at_the_module_root(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            _EVERYTHING,
            _TAIL_USE_AND_TAIL,
            probe_table(
                _REACHED_TWICE_PROBES,
                {
                    "call": _CONSTRUCTOR,
                    "annotation": _QUALIFICATION,
                    "receiver": _QUALIFICATION,
                    "member": _CONSTRUCTOR,
                    "function": _QUALIFICATION,
                },
                span_texts={
                    "call": "Point",
                    "annotation": "Point",
                    "receiver": "self",
                    "member": "Red",
                    "function": "f",
                },
                origins={
                    "call": _reached_twice("Point"),
                    "annotation": _reached_twice("Point"),
                    "receiver": _reached_twice("Point"),
                    "member": _reached_twice("Color::Red"),
                    "function": _reached_twice("f"),
                },
            ),
            part=part,
        )

    def test_in_a_scope_region(self, tmp_path: Path) -> None:
        region = (
            "scope S\n  import a/lib::*\n  import b/lib::*\n  use a/lib::*\n  let y = {}\nend S"
        )
        probes = {
            "call": region.format("Point(x = 1)"),
            "member": region.format("Red"),
            "function": region.format("f()"),
        }
        assert_verdicts(
            tmp_path,
            _EVERYTHING,
            (),
            probe_table(
                probes,
                {"call": _CONSTRUCTOR, "member": _CONSTRUCTOR, "function": _QUALIFICATION},
                span_texts={"call": "Point", "member": "Red", "function": "f"},
                origins={
                    "call": _reached_twice("Point"),
                    "member": _reached_twice("Color::Red"),
                    "function": _reached_twice("f"),
                },
            ),
        )


_STATUSES = dict.fromkeys(
    ("a/lib", "b/lib"), "record Saved\n  x: int\nenum Status = ::Saved | Idle\n"
)


def test_a_region_import_of_an_enum_contributes_its_members_as_imported(tmp_path: Path) -> None:
    """A defined member and a referenced record both keep the import's kind."""
    region = "scope S\n  import a/lib::{{Status}}\n  import b/lib::{{Status}}\n  let y = {}\nend S"
    probes = {"defined": region.format("Idle"), "referenced": region.format("Saved(x = 1)")}
    assert_verdicts(
        tmp_path,
        _STATUSES,
        (),
        probe_table(
            probes,
            dict.fromkeys(probes, _CONSTRUCTOR),
            span_texts={"defined": "Idle", "referenced": "Saved"},
            origins={
                "defined": _origins(
                    ImportedModuleOrigin, "a/lib::Status::Idle", "b/lib::Status::Idle"
                ),
                "referenced": _origins(ImportedModuleOrigin, "a/lib::Saved", "b/lib::Saved"),
            },
        ),
    )


_MEMBER_AND_FUNCTION = {"a/lib": "enum Color = Red | Green\n", "b/lib": "def Red() -> int = 1\n"}
_MEMBER_AND_FUNCTION_PROBES = {"value": "Red", "call": "Red()"}


class TestEnumMemberBesideAContributedFunction:
    """An import tail's enum member and another module's function clash as values.

    The prelude is one more import tail.
    """

    def _assert(
        self,
        tmp_path: Path,
        header: tuple[str, ...],
        function_kind: type,
        *,
        member_use: bool = False,
        part: Part | None = None,
    ) -> None:
        origins = _origins(ImportedModuleOrigin, "a/lib::Color::Red") | _origins(
            function_kind, "b/lib::Red"
        )
        if member_use:
            origins |= _origins(UseDeclarationOrigin, "a/lib::Color::Red")
        assert_verdicts(
            tmp_path,
            _MEMBER_AND_FUNCTION,
            header,
            probe_table(
                _MEMBER_AND_FUNCTION_PROBES,
                dict.fromkeys(_MEMBER_AND_FUNCTION_PROBES, _QUALIFICATION),
                span_texts=dict.fromkeys(_MEMBER_AND_FUNCTION_PROBES, "Red"),
                origins=dict.fromkeys(_MEMBER_AND_FUNCTION_PROBES, origins),
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_two_import_tails(self, tmp_path: Path, part: Part) -> None:
        self._assert(
            tmp_path,
            ("import a/lib::*", "import b/lib::*"),
            ImportedModuleOrigin,
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_import_tail_and_use(self, tmp_path: Path, part: Part) -> None:
        header = ("import a/lib::*", "import b/lib", "use b/lib::*")
        self._assert(
            tmp_path,
            header,
            UseDeclarationOrigin,
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_member_reached_twice(self, tmp_path: Path, part: Part) -> None:
        header = ("import a/lib::*", "import b/lib::*", "use a/lib::*")
        self._assert(
            tmp_path,
            header,
            ImportedModuleOrigin,
            member_use=True,
            part=part,
        )

    def test_beside_the_prelude(self, tmp_path: Path) -> None:
        probes = {"value": "Some", "call": "Some()"}
        origins = _origins(ImportedModuleOrigin, "lib::Some", "std/option::Option::Some")
        assert_verdicts(
            tmp_path,
            {"lib": "def Some() -> int = 1\n"},
            ("import lib::*",),
            probe_table(
                probes,
                dict.fromkeys(probes, _QUALIFICATION),
                span_texts=dict.fromkeys(probes, "Some"),
                origins=dict.fromkeys(probes, origins),
            ),
        )

    def test_in_a_scope_region(self, tmp_path: Path) -> None:
        region = "scope S\n  import a/lib::*\n  import b/lib::*\n  let y = {}\nend S"
        probes = {key: region.format(probe) for key, probe in _MEMBER_AND_FUNCTION_PROBES.items()}
        origins = _origins(ImportedModuleOrigin, "a/lib::Color::Red", "b/lib::Red")
        assert_verdicts(
            tmp_path,
            _MEMBER_AND_FUNCTION,
            (),
            probe_table(
                probes,
                dict.fromkeys(probes, _QUALIFICATION),
                span_texts=dict.fromkeys(probes, "Red"),
                origins=dict.fromkeys(probes, origins),
            ),
        )
