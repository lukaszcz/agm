"""Scope's one selection for a bare or ``::``-anchored type name, in every type position.

A bare type name is the length-zero case of the qualifier decision: the one
nearest-level lookup across the type/scope namespace reads it, and scope
records the one declaration it selects (or rejects it) before typecheck runs.
Every class here probes one spelling in the type positions -- an annotation,
an alias target, a type argument, an applied type, a caught exception type and
an ``extends`` base -- in file mode and every REPL grouping (see
:mod:`tests.agl.qualifier_support`), asserting the phase, the class and the
span, or the accepted identity:

- ``::Name`` reads this module's root alone, exactly like the value
  ``::Name``, and names no member there when the root lacks it;
- a name several equally near ``use`` declarations contribute is ambiguous in
  every position, a caught exception and an ``extends`` base included;
- a nearer scope region stops the lookup, so a farther same-spelled type --
  imported or this module's own -- is never selected past it;
- the nearest type declaration is the one every position selects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.scope import AglScopeError
from agm.agl.scope.symbols import AmbiguousQualificationError, UnknownMemberError
from agm.agl.typecheck import AglTypeError
from tests.agl.qualifier_support import (
    FilePhase,
    assert_verdicts_for_grouping,
    grouping_params,
)

_ACCEPTED: tuple[FilePhase, type[BaseException] | type[None]] = ("accepted", type(None))


def _rejected(
    cls: type[BaseException],
) -> tuple[FilePhase, type[BaseException] | type[None]]:
    return ("scope", cls)


def _type_probes(spelling: str, applied: str | None = None) -> dict[str, str]:
    """Probe *spelling* as an annotation, alias target and type argument (and *applied*)."""
    probes = {
        "annotation": f"def g(x: {spelling}) -> int = 1",
        "alias": f"type A = {spelling}",
        "type-argument": f"def g(x: array[{spelling}]) -> int = 1",
    }
    if applied is not None:
        probes["applied"] = f"def g(x: {applied}) -> int = 1"
    return probes


# ---------------------------------------------------------------------------
# ``::Name`` reads this module's root alone.
# ---------------------------------------------------------------------------

_ROOT_HEADER = ("record R\n  x: int",)


class TestAnchoredTypeNameReadsTheModuleRoot:
    """``::Missing`` names no member of this module's root in any position."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_ROOT_HEADER) + 1))
    def test_missing_name_is_unknown_member(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {**_type_probes("::Missing", "::Missing[int]"), "value": "::Missing"}
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _ROOT_HEADER,
            sizes,
            probes,
            {key: _rejected(UnknownMemberError) for key in probes},
            span_texts={key: "::Missing" for key in probes},
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_ROOT_HEADER) + 1))
    def test_root_type_is_selected(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {
            "annotation": "let v: ::R = R(x = 1)\nv",
            "alias": "type A = ::R\nlet v: A = R(x = 1)\nv",
            "type-argument": "let v: array[::R] = [R(x = 1)]\nv[0]",
            "value": "::R(x = 1)",
        }
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _ROOT_HEADER,
            sizes,
            probes,
            {key: _ACCEPTED for key in probes},
            expected_identities={key: "record R\n  x: int" for key in probes},
        )


# ---------------------------------------------------------------------------
# Several equally near ``use`` contributions are ambiguous everywhere.
# ---------------------------------------------------------------------------

_BOOM_MODULES = {
    "m/a": "exception Boom extends Exception\n",
    "m/b": "exception Boom extends Exception\n",
}
_BOOM_HEADER = ("import m/a", "import m/b", "use m/a::*", "use m/b::*")
_CATCH_BOOM = "let _ = try\n  ()\ncatch Boom as e =>\n  ()"
_EXTENDS_BOOM = "exception Local extends Boom"


class TestAmbiguousBareTypeNameInEveryPosition:
    """Two ``use`` declarations contributing ``Boom`` leave it ambiguous in every position.

    A caught exception type and an ``extends`` base read the same selection
    as an annotation, however REPL entries group the ``use`` declarations.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_BOOM_HEADER) + 1))
    def test_every_position_is_ambiguous(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {**_type_probes("Boom"), "catch": _CATCH_BOOM, "extends": _EXTENDS_BOOM}
        spans = {key: "Boom" for key in probes}
        spans["catch"] = "catch Boom as e =>\n  ()"
        spans["extends"] = _EXTENDS_BOOM
        assert_verdicts_for_grouping(
            tmp_path,
            _BOOM_MODULES,
            _BOOM_HEADER,
            sizes,
            probes,
            {key: _rejected(AmbiguousQualificationError) for key in probes},
            span_texts=spans,
        )


# ---------------------------------------------------------------------------
# A nearer scope region stops the lookup.
# ---------------------------------------------------------------------------

_GEO_MODULES = {
    "shapes": "scope Geo\n  record Point\n    x: int\nend Geo\n",
    "tl": "record Geo\n  x: int\n",
    "tl2": "record Geo\n  z: int\n",
}
_GEO_HEADER = ("import tl::*", "import tl2::*", "import shapes")


def _in_region(opening: str, item: str) -> str:
    """Wrap *item* in ``scope r`` after the region's own *opening* lines."""
    body = "\n".join(f"  {line}" for line in (*opening.split("\n"), *item.split("\n")))
    return f"scope r\n{body}\nend r"


_OWN_POINT_HEADER = ("record Point\n  y: int",)
_OWN_POINT_REGION = "scope Point\n  record Q\nend Point"
_BOOM_REGION_MODULES = {
    "a": "exception Boom extends Exception\n",
    "b": "scope Boom\n  record Q\nend Boom\n",
}
_BOOM_REGION_HEADER = ("import a::*", "import b")
_LOCAL_REGION_HEADER = ("scope Geo\n  record Q\nend Geo",)


class TestNearerRegionStopsABareTypeName:
    """A scope region nearer than a same-spelled type names no type in any position.

    The farther type is never selected past the region -- whether it is one
    of several root import tails, this module's own root declaration, or a
    root import tail a ``use``-opened region shadows for a caught exception
    and an ``extends`` base.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_GEO_HEADER) + 1))
    def test_use_opened_region_over_import_tails(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        probes = {
            key: _in_region("use shapes::*", probe) for key, probe in _type_probes("Geo").items()
        }
        assert_verdicts_for_grouping(
            tmp_path,
            _GEO_MODULES,
            _GEO_HEADER,
            sizes,
            probes,
            {key: _rejected(AglScopeError) for key in probes},
            span_texts={key: "Geo" for key in probes},
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_OWN_POINT_HEADER) + 1))
    def test_own_region_over_own_root_type(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {
            key: _in_region(_OWN_POINT_REGION, probe)
            for key, probe in _type_probes("Point").items()
        }
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _OWN_POINT_HEADER,
            sizes,
            probes,
            {key: _rejected(AglScopeError) for key in probes},
            span_texts={key: "Point" for key in probes},
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_BOOM_REGION_HEADER) + 1))
    def test_use_opened_region_over_caught_and_base_exception(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        probes = {
            "annotation": _in_region("use b::*", "def g(x: Boom) -> int = 1"),
            "catch": _in_region("use b::*", _CATCH_BOOM),
            "extends": _in_region("use b::*", _EXTENDS_BOOM),
        }
        spans = {
            "annotation": "Boom",
            "catch": "catch Boom as e =>\n    ()\n",
            "extends": _EXTENDS_BOOM,
        }
        assert_verdicts_for_grouping(
            tmp_path,
            _BOOM_REGION_MODULES,
            _BOOM_REGION_HEADER,
            sizes,
            probes,
            {key: _rejected(AglScopeError) for key in probes},
            span_texts=spans,
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_LOCAL_REGION_HEADER) + 1))
    def test_region_alone_is_no_type(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = _type_probes("Geo", "Geo[int]")
        spans = {key: "Geo" for key in probes}
        spans["applied"] = "Geo[int]"
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _LOCAL_REGION_HEADER,
            sizes,
            probes,
            {key: _rejected(AglScopeError) for key in probes},
            span_texts=spans,
        )


# ---------------------------------------------------------------------------
# The nearest type declaration is selected.
# ---------------------------------------------------------------------------


class TestNearestTypeDeclarationIsSelected:
    """A type declared in the enclosing region wins over farther same-spelled types."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_GEO_HEADER) + 1))
    def test_region_type_over_import_tails(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        region_geo = "record Geo\n  w: int"
        probes = {
            "annotation": _in_region(region_geo, "let v: Geo = Geo(w = 1)") + "\nr::v",
            "alias": _in_region(region_geo, "type A = Geo\nlet v: A = Geo(w = 1)") + "\nr::v",
            "type-argument": _in_region(region_geo, "let v: array[Geo] = [Geo(w = 1)]")
            + "\nr::v[0]",
        }
        assert_verdicts_for_grouping(
            tmp_path,
            _GEO_MODULES,
            _GEO_HEADER,
            sizes,
            probes,
            {key: _ACCEPTED for key in probes},
            expected_identities={key: "record r::Geo\n  w: int" for key in probes},
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_BOOM_REGION_HEADER) + 1))
    def test_imported_exception_is_caught_and_extended(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        probes = {
            "catch": "let v = try\n  1\ncatch Boom as e =>\n  2\nv",
            "extends": 'exception Local extends Boom\nLocal(message = "m") is Boom',
        }
        assert_verdicts_for_grouping(
            tmp_path,
            _BOOM_REGION_MODULES,
            _BOOM_REGION_HEADER,
            sizes,
            probes,
            {key: _ACCEPTED for key in probes},
            expected_identities={"catch": "int", "extends": "bool"},
        )


# ---------------------------------------------------------------------------
# The selected declaration decides how a type name may be applied.
# ---------------------------------------------------------------------------

_ARITY_HEADER = (
    "record Box[T]\n  value: T",
    "type Wrapper[T] = array[T]",
    "record Plain\n  x: int",
)


class TestSelectedTypeIsAppliedByItsDeclaration:
    """A selected type takes exactly its declaration's type arguments, in every position.

    A parameterized alias or generic record needs its arguments, a plain
    record takes none, and only an exception type can be caught -- whether the
    declaration is in the same REPL entry or an earlier one.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_ARITY_HEADER) + 1))
    def test_misapplied_selection_is_rejected(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {
            "bare-alias": "def g(x: Wrapper) -> int = 1",
            "alias-arity": "def g(x: Wrapper[int, int]) -> int = 1",
            "generic-arity": "def g(x: Box[int, int]) -> int = 1",
            "plain-applied": "def g(x: Plain[int]) -> int = 1",
            "caught-record": "let _ = try\n  ()\ncatch Plain as e =>\n  ()",
        }
        spans = {
            "bare-alias": "x: Wrapper",
            "alias-arity": "x: Wrapper[int, int]",
            "generic-arity": "x: Box[int, int]",
            "plain-applied": "x: Plain[int]",
            "caught-record": "catch Plain as e =>\n  ()",
        }
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _ARITY_HEADER,
            sizes,
            probes,
            {key: ("typecheck", AglTypeError) for key in probes},
            span_texts=spans,
        )
