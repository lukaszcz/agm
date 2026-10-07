"""One verdict for an ambiguous qualified spelling: class, origins, and the spelling as written.

The route of a qualified spelling -- a slash path, a ``/``-anchored path, an
import alias, or the ``::`` current-module anchor -- is part of the spelling
the error reports. Origins are listed in one order: imported, then
``use``-contributed, then locally declared, each by declaration spelling; a
declaration reached twice the same way is listed once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.ids import ModuleId, Reader
from agm.agl.repl import ReplSession
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import (
    Ambiguity,
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    DeclaredOrigin,
    ImportedModuleOrigin,
    UseDeclarationOrigin,
    to_bare_atom,
)
from agm.agl.syntax.spans import SourceSpan
from tests.agl.ir_harness import make_graph_from_files
from tests.agl.qualifier_support import (
    Part,
    Phase,
    assert_verdicts,
    probe_table,
    verdict_parts,
)

_CONSTRUCTOR: tuple[Phase, type[BaseException] | type[None]] = (
    "scope",
    AmbiguousConstructorError,
)
_QUALIFICATION: tuple[Phase, type[BaseException] | type[None]] = (
    "scope",
    AmbiguousQualificationError,
)

_SPAN = SourceSpan(start_line=1, start_col=1, end_line=1, end_col=1, start_offset=0, end_offset=0)

_TWO_REDS = "enum A\n  | Red\nenum B\n  | Red\n"


def _origins(kind: type, *spellings: str) -> frozenset[tuple[type, str]]:
    return frozenset((kind, spelling) for spelling in spellings)


class TestAmbiguousModuleMember:
    """A module surface injecting ``Red`` from two of its enums, read through every route form."""

    _ORIGINS = _origins(ImportedModuleOrigin, "x/lib::A::Red", "x/lib::B::Red")
    _PROBES = {
        "slash": "x/lib::Red",
        "anchored": "/x/lib::Red",
        "anchored-in-scope": "scope S\n  let y = /x/lib::Red\nend S",
    }
    _SPELLINGS = {
        "slash": "x/lib::Red",
        "anchored": "/x/lib::Red",
        "anchored-in-scope": "/x/lib::Red",
    }

    def test_every_route_form(self, tmp_path: Path) -> None:
        assert_verdicts(
            tmp_path,
            {"x/lib": _TWO_REDS},
            ("import x/lib",),
            probe_table(
                self._PROBES,
                dict.fromkeys(self._PROBES, _CONSTRUCTOR),
                span_texts=self._SPELLINGS,
                origins=dict.fromkeys(self._PROBES, self._ORIGINS),
                spellings=self._SPELLINGS,
            ),
        )

    def test_import_alias(self, tmp_path: Path) -> None:
        assert_verdicts(
            tmp_path,
            {"x/lib": _TWO_REDS},
            ("import x/lib as L",),
            probe_table(
                {"alias": "L::Red"},
                {"alias": _CONSTRUCTOR},
                span_texts={"alias": "L::Red"},
                origins={"alias": self._ORIGINS},
                spellings={"alias": "L::Red"},
            ),
        )


class TestAmbiguousCurrentModuleMember:
    """``::Red`` over two of this module's own enums: declared origins, spelled anchored."""

    _HEADER = ("enum A\n  | Red", "enum B\n  | Red")

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_value(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            {},
            self._HEADER,
            probe_table(
                {"value": "::Red"},
                {"value": _CONSTRUCTOR},
                span_texts={"value": "::Red"},
                origins={"value": _origins(DeclaredOrigin, "A::Red", "B::Red")},
                spellings={"value": "::Red"},
            ),
            part=part,
        )


_MANY_REDS = "\n".join(f"enum {owner}\n  | Red" for owner in "LKJIHGFEDCBA")


@pytest.mark.parametrize("entries", [(_MANY_REDS, "::Red"), (f"{_MANY_REDS}\n::Red",)])
def test_current_module_member_repair_is_one_spelling_however_entries_group(
    tmp_path: Path, entries: tuple[str, ...]
) -> None:
    """The repair names the candidate first by declaration identity, never by set order."""
    session = ReplSession(cwd=tmp_path, default_stdlib=False)
    session.open()
    *setup, probe = entries
    for entry in setup:
        assert session.eval_entry(entry).ok
    failure = session.eval_entry(probe).failure
    assert type(failure) is AmbiguousConstructorError
    assert failure.repair == "::A::Red"


class TestAmbiguousModuleRoute:
    """A route suffix two imported modules share: the spelling keeps the written suffix."""

    _HEADER = ("import p/x/lib", "import q/x/lib")
    _PROBES = {"short": "lib::f()", "long": "x/lib::f()"}

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_every_suffix(self, tmp_path: Path, part: Part) -> None:
        origins = _origins(ImportedModuleOrigin, "p/x/lib::f", "q/x/lib::f")
        assert_verdicts(
            tmp_path,
            {"p/x/lib": "def f() -> int = 1\n", "q/x/lib": "def f() -> int = 2\n"},
            self._HEADER,
            probe_table(
                self._PROBES,
                dict.fromkeys(self._PROBES, _QUALIFICATION),
                span_texts={"short": "lib::f", "long": "x/lib::f"},
                origins=dict.fromkeys(self._PROBES, origins),
                spellings={"short": "lib::f", "long": "x/lib::f"},
            ),
            part=part,
        )


def test_origins_are_ordered_by_kind_then_spelling_and_listed_once(tmp_path: Path) -> None:
    entry = (
        "import lib::*\n"
        "import lib::{shared}\n"
        "use T::*\n"
        "use S::*\n"
        "\n"
        "scope T\n"
        "  def shared() -> int = 3\n"
        "end T\n"
        "\n"
        "scope S\n"
        "  def shared() -> int = 2\n"
        "end S\n"
        "\n"
        "shared()\n"
    )
    graph = make_graph_from_files(tmp_path, {"entry": entry, "lib": "def shared() -> int = 1\n"})
    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(graph)
    error = excinfo.value
    assert type(error) is AmbiguousQualificationError
    assert error.spelling == "shared"
    entry_id = graph.entry_id
    assert error.origins == (
        ImportedModuleOrigin((ModuleId.from_path("lib"), "shared")),
        UseDeclarationOrigin((entry_id, to_bare_atom(("S", "shared")))),
        UseDeclarationOrigin((entry_id, to_bare_atom(("T", "shared")))),
    )


def test_constructor_ambiguity_without_a_repair_stays_a_constructor_ambiguity() -> None:
    """No spelling selecting a candidate leaves the repair out, however the verdict is kept."""
    module = ModuleId.from_path("lib")
    origins = (ImportedModuleOrigin((module, to_bare_atom(("A", "Red")))),)
    error = AmbiguousConstructorError.for_constructor_origins(
        "Red", origins, repair=None, span=_SPAN, reader=Reader(module)
    )
    assert error.repair is None
    kept = Ambiguity.of(error).error("Red", _SPAN, reader=Reader(module))
    assert isinstance(kept, AmbiguousConstructorError)
    assert kept.repair is None


class TestAmbiguousOwnerProjection:
    """``Point::Point`` through a bare owner: each origin keeps the kind that reached its owner."""

    _POINTS = {"a/lib": "record Point\n  x: int\n", "b/lib": "record Point\n  x: int\n"}
    _PROBES = {"call": "Point::Point(x = 1)", "annotation": "fn(p: Point::Point) => 1"}

    def _assert(
        self,
        tmp_path: Path,
        header: tuple[str, ...],
        origins: frozenset[tuple[type, str]],
        part: Part | None = None,
    ) -> None:
        assert_verdicts(
            tmp_path,
            self._POINTS,
            header,
            probe_table(
                self._PROBES,
                dict.fromkeys(self._PROBES, _QUALIFICATION),
                span_texts=dict.fromkeys(self._PROBES, "Point::Point"),
                origins=dict.fromkeys(self._PROBES, origins),
                spellings=dict.fromkeys(self._PROBES, "Point::Point"),
            ),
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_two_import_tails(self, tmp_path: Path, part: Part) -> None:
        origins = _origins(ImportedModuleOrigin, "a/lib::Point::Point", "b/lib::Point::Point")
        self._assert(
            tmp_path,
            ("import a/lib::*", "import b/lib::*"),
            origins,
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(4))
    def test_import_tail_and_use(self, tmp_path: Path, part: Part) -> None:
        origins = _origins(ImportedModuleOrigin, "a/lib::Point::Point") | _origins(
            UseDeclarationOrigin, "b/lib::Point::Point"
        )
        self._assert(
            tmp_path,
            ("import a/lib::*", "import b/lib", "use b/lib::*"),
            origins,
            part=part,
        )

    @pytest.mark.parametrize("part", verdict_parts(3))
    def test_own_owner_wins(self, tmp_path: Path, part: Part) -> None:
        assert_verdicts(
            tmp_path,
            self._POINTS,
            ("import a/lib::*", "record Point\n  x: int"),
            probe_table(
                {"call": self._PROBES["call"]},
                {"call": ("accepted", type(None))},
                identities={"call": "record Point\n  x: int"},
            ),
            part=part,
        )
