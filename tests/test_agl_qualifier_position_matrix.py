"""Position-matrix regression tests for scope's qualifier decisions.

Covers items A and C of the dict-keys qualifier refactor: scope, not
typecheck, decides which declaration a qualified type name selects and
whether a qualifier selects none at all, identically across the value,
pattern, ``is``, annotation, alias, type-argument and applied positions, and
identically in file mode and the REPL (independent of how a REPL session
groups its declarations into entries).

``_matrix_params`` reuses one probe program per outcome across five owner
spellings (a module route, a wildcard-imported bare name, a
wildcard-imported generic owner applied to a type argument -- also proving
item B, since the owner is instantiated from scope's recorded key rather
than re-resolved by name -- a locally ``use``d scope region, and a
module-level ``use`` alias) and six syntactic positions; the seventh
position (an owner itself carrying type arguments, ``Owner[T]::Member``) is
exercised by the generic-owner spelling, whose member is *always* written
applied. ``TestLocalScopeShadowsSameNamedImport`` and
``TestImportedScopeRegionMemberIsAccepted`` cover the local-scope-vs-import
case singled out by item A, for a record owner (so ``is`` and applying type
arguments, meaningless for a non-generic record, are left to the matrix
above); the accepted case is the mutation-(b) regression (scope's new
recording for a scope-region prefix), and the rejected case is the
mutation-(a) regression (a recorded decision beating a name lookup).

The "applied" form's ``is``/pattern positions are item B's own regression:
before the fix, an ``AppliedT`` owner (``Box[int]``) was re-resolved by its
bare spelling (``Box``), ambiguous once both ``gm`` and ``gn`` are
wildcard-imported, so those positions raised an ambiguous-type error whose
span covered only the bare owner name instead of a genuine, full-span
mismatch against the recorded ``gm::Box[int]`` key --
``test_qualifier_decision_matrix_file``/``..._repl_grouped`` assert the
latter (span text equal to the full qualifier chain).
``test_nested_generic_alias_owner_is_instantiated_by_substitution`` is item
B's second, independently discovered bug: an alias target nested inside
another generic type must not be instantiated by a bare type-argument swap.

``TestUnknownQualifierRouteAcrossPositions`` is item C's other half: a
qualifier whose leading route names no module, ``use`` alias or local scope
at all (so scope's owner/full-path selection is empty from the start, never
ambiguous and never a real owner missing one member) -- the ``_FORMS``
matrix's "missing" outcome always names a real, resolvable owner, so this is
the mutation-(c) regression, scope's own selects-none raise for the type
positions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.agl.typecheck.program as tcp
from agm.agl.diagnostics import AglError
from agm.agl.repl import ReplSession
from agm.agl.repl import entry_pipeline as ep
from agm.agl.scope import AglScopeError
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import (
    AmbiguousQualificationError,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck import AglTypeError
from tests.agl.ir_harness import base_caps, make_graph_from_files

Verdict = tuple[str, type[BaseException] | type[None], SourceSpan | None]


def _file_verdict(tmp_path: Path, modules: dict[str, str]) -> Verdict:
    """Resolve and type-check *modules* as files; report which phase, if any, raised."""
    graph = make_graph_from_files(tmp_path, modules)
    try:
        resolved = resolve_program(graph)
    except AglScopeError as exc:
        return "scope", type(exc), exc.span
    try:
        tcp.check_program(resolved, base_caps())
    except AglTypeError as exc:
        return "typecheck", type(exc), exc.span
    return "accepted", type(None), None


def _repl_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, modules: dict[str, str], entries: list[str]
) -> Verdict:
    """Evaluate *entries* on a fresh REPL session; report which phase, if any, raised.

    A pre-execution diagnostic carries no exception object of its own
    (``EntryResult.error`` is set only for an entry that raises while
    *running*), so the underlying ``resolve_program``/``check_program`` calls
    are wrapped to capture the exception each phase actually raised, for the
    same class/span assertions the file-mode verdict supports. The captured
    exception's span is cross-checked against the entry's own diagnostic, so
    a wrapper drifting from the diagnostic it explains would fail loudly.
    """
    captured: list[tuple[str, AglError]] = []
    orig_resolve = ep.EntryPipeline._resolve_program
    orig_check = tcp.check_program

    def wrapped_resolve(self: ep.EntryPipeline, *args: object, **kwargs: object) -> object:
        try:
            return orig_resolve(self, *args, **kwargs)
        except AglError as exc:
            captured.append(("scope", exc))
            raise

    def wrapped_check(*args: object, **kwargs: object) -> object:
        try:
            return orig_check(*args, **kwargs)
        except AglError as exc:
            captured.append(("typecheck", exc))
            raise

    monkeypatch.setattr(ep.EntryPipeline, "_resolve_program", wrapped_resolve)
    monkeypatch.setattr(tcp, "check_program", wrapped_check)

    for name, source in modules.items():
        if name == "entry":
            continue
        path = tmp_path / f"{name}.agl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    session = ReplSession(cwd=tmp_path, default_stdlib=False)
    session.open()

    result = None
    for index, source in enumerate(entries):
        captured.clear()
        result = session.eval_entry(source)
        if index < len(entries) - 1:
            assert result.ok, result.diagnostics
    assert result is not None
    if result.ok:
        return "accepted", type(None), None
    assert captured, "a rejected entry must have raised during resolve or check"
    phase, exc = captured[-1]
    diagnostic = result.diagnostics[0]
    span = exc.span
    assert span is not None
    assert diagnostic.line == span.start_line
    assert diagnostic.column == span.start_col
    assert diagnostic.end_column == span.end_col
    return phase, type(exc), span


# ---------------------------------------------------------------------------
# Item A / C: unique, other (a real but mismatched member), ambiguous and
# missing selection, across five owner spellings and six positions.
# ---------------------------------------------------------------------------

_RT = {
    "one/types": "enum Color\n  | Red\n  | Green\n",
    "two/types": "enum Color\n  | Red\n  | Blue\n",
}
_MN = {"m": "enum Color\n  | Red\n  | Green\n", "n": "enum Color\n  | Red\n  | Blue\n"}
_GB = {
    "gm": "enum Box[T]\n  | Full(value: T)\n  | Both\n  | Empty\n",
    "gn": "enum Box[T]\n  | Full(value: T)\n  | Both\n  | Other\n",
}
_LOC = (
    "scope a\n  enum E = X | Y\nend a",
    "scope b\n  enum E = X | Z\nend b",
    "use a::*",
    "use b::*",
    "let v: a::E = a::E::Y",
)

# name -> (extra modules, header entries, owner spelling, {outcome: member})
_FORMS: dict[str, tuple[dict[str, str], list[str], str, dict[str, str]]] = {
    "route": (
        _RT,
        [
            "import one/types",
            "import two/types",
            "let v: one/types::Color = one/types::Color::Green",
        ],
        "types::Color",
        {"unique": "Green", "other": "Blue", "amb": "Red", "missing": "NoSuch"},
    ),
    "bare": (
        _MN,
        ["import m", "import m::*", "import n::*", "let v: m::Color = m::Color::Green"],
        "Color",
        {"unique": "Green", "other": "Blue", "amb": "Red", "missing": "NoSuch"},
    ),
    "applied": (
        _GB,
        ["import gm", "import gm::*", "import gn::*", "let v: gm::Box[int] = gm::Box[int]::Empty"],
        "Box[int]",
        {"unique": "Empty", "other": "Other", "amb": "Both", "missing": "NoSuch"},
    ),
    "localuse": (
        {},
        ["use a::*", "use b::*", _LOC[0], _LOC[1], _LOC[4]],
        "E",
        {"unique": "Y", "other": "Z", "amb": "X", "missing": "NoSuch"},
    ),
    "moduse": (
        _MN,
        ["import m", "import n", "use m::*", "use n::*", "let v: m::Color = m::Color::Green"],
        "Color",
        {"unique": "Green", "other": "Blue", "amb": "Red", "missing": "NoSuch"},
    ),
}

_POS: dict[str, str] = {
    "value": "{q}",
    "pattern": "case v of\n  | {q} => 1\n  | _ => 2",
    "is": "v is {q}",
    "annot": "fn(x: {q}) => 1",
    "alias": "type CC = {q}\n1",
    "tyarg": "fn(x: array[{q}]) => 1",
}

# (phase, class) expected for each (form, outcome, position); phase
# "accepted" leaves the class column unused. Frozen from the fixed
# implementation's actual, verified behavior (tests/CLAUDE.md: assert
# behavior, not internals): "other" is a real member of a structurally
# distinct nominal type, so it is a genuine typecheck mismatch wherever the
# position actually checks the referenced value/pattern against ``v``'s type
# (pattern, ``is``) and otherwise just a legal type reference; "amb" is
# always scope's ambiguity verdict; "missing" is scope's selects-none
# verdict everywhere: the same-level ambiguity of an owner's own spelling --
# whether a bare unqualified name or a routed one (``route::Owner``) -- is
# decided full-path-first, so an owner ambiguous on its own spelling still
# selects "missing" once the requested member disambiguates it in no
# candidate.
_ACCEPTED = ("accepted", type(None))
_OTHER_MISMATCH = ("typecheck", AglTypeError)
_AMBIGUOUS = ("scope", AmbiguousQualificationError)
_MISSING = ("scope", UnknownMemberError)

_EXPECTED: dict[tuple[str, str, str], tuple[str, type[BaseException] | type[None]]] = {}
for _form in _FORMS:
    for _pos in _POS:
        _EXPECTED[(_form, "unique", _pos)] = _ACCEPTED
        _EXPECTED[(_form, "other", _pos)] = (
            _OTHER_MISMATCH if _pos in ("pattern", "is") else _ACCEPTED
        )
        _EXPECTED[(_form, "amb", _pos)] = _AMBIGUOUS
        _EXPECTED[(_form, "missing", _pos)] = _MISSING
del _form, _pos

# The span of a raised error covers exactly the qualifier chain's own text
# (``owner::member``), except an ``is``-test's mismatch error, whose span
# covers the whole ``subject is owner::member`` expression.
_IS_MISMATCH_SPAN_PREFIX = "v is "


def _matrix_case(
    form_name: str, outcome: str, pos_name: str
) -> tuple[dict[str, str], list[str], str, str]:
    modules, header, own, outcomes = _FORMS[form_name]
    q = f"{own}::{outcomes[outcome]}"
    entry = _POS[pos_name].format(q=q)
    return modules, header, entry, q


def _matrix_params() -> list[object]:
    return [
        pytest.param(form_name, outcome, pos_name, id=f"{form_name}-{outcome}-{pos_name}")
        for form_name, (_, _, _, outcomes) in _FORMS.items()
        for outcome in outcomes
        for pos_name in _POS
    ]


def _assert_matrix_verdict(
    source: str, verdict: Verdict, form_name: str, outcome: str, pos_name: str, q: str
) -> None:
    phase, cls, span = verdict
    expected_phase, expected_cls = _EXPECTED[(form_name, outcome, pos_name)]
    assert (phase, cls) == (expected_phase, expected_cls)
    if expected_phase == "accepted":
        return
    assert span is not None
    text = source[span.start_offset : span.end_offset]
    expected_text = _IS_MISMATCH_SPAN_PREFIX + q if pos_name == "is" and outcome == "other" else q
    assert text == expected_text


@pytest.mark.parametrize(("form_name", "outcome", "pos_name"), _matrix_params())
def test_qualifier_decision_matrix_file(
    tmp_path: Path, form_name: str, outcome: str, pos_name: str
) -> None:
    """Scope's unique/other/ambiguous/missing verdict is the same in every position."""
    modules, header, entry, q = _matrix_case(form_name, outcome, pos_name)
    src = "\n".join([*header, entry])
    verdict = _file_verdict(tmp_path, {"entry": src, **modules})
    _assert_matrix_verdict(src, verdict, form_name, outcome, pos_name, q)


@pytest.mark.parametrize(("form_name", "outcome", "pos_name"), _matrix_params())
def test_qualifier_decision_matrix_repl_grouped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, form_name: str, outcome: str, pos_name: str
) -> None:
    """The file-mode verdict above holds identically for one grouped REPL entry."""
    modules, header, entry, q = _matrix_case(form_name, outcome, pos_name)
    src = "\n".join([*header, entry])
    verdict = _repl_verdict(monkeypatch, tmp_path, modules, [src])
    _assert_matrix_verdict(src, verdict, form_name, outcome, pos_name, q)


# ---------------------------------------------------------------------------
# Item A: a local scope region beats a same-named imported nested region,
# identically whether the local declaration wins the position (rejecting a
# member it doesn't have) or nothing local conflicts at all (accepting the
# sole imported member) -- the exact inconsistency the reproducer named.
# ---------------------------------------------------------------------------

_GEO_LOCAL_ONLY = "scope Geo\n  enum Kind = Round | Flat\nend Geo"
_GEO_IMPORTED = "scope Geo\n  record Point\n    x: int\nend Geo\n"

# Positions meaningful for a plain, non-generic record owner. ``is`` (record
# types are not ``is``-testable) and applying type arguments (the record
# takes none) are exercised instead by the "applied" form above, whose
# owner is generic.
_RECORD_POS: dict[str, str] = {
    "value": "{q}(x = 1)",
    "pattern": "case 1 of\n  | {q}(x) => x\n  | _ => 2",
    "annot": "fn(p: {q}) => 1",
    "alias": "type A = {q}\n1",
    "tyarg": "fn(p: array[{q}]) => 1",
}

_ACCEPTED_RECORD_POS: dict[str, str] = {
    "value": "{q}(x = 1)",
    "pattern": "let v: {q} = {q}(x = 1)\ncase v of\n  | {q}(x) => x",
    "annot": "fn(p: {q}) => 1",
    "alias": "type A = {q}\n1",
    "tyarg": "fn(p: array[{q}]) => 1",
}


class TestLocalScopeShadowsSameNamedImport:
    """A local ``scope Geo`` lacking ``Point`` beats an imported ``Geo::Point``.

    Regression for mutation (a): a name lookup would find the imported
    ``Point`` and accept every position; the recorded decision (local scope
    wins) rejects all of them instead, matching the value/pattern positions
    that already rejected this before item A's fix reached the others.
    """

    @pytest.mark.parametrize("pos_name", sorted(_RECORD_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _RECORD_POS[pos_name].format(q="Geo::Point")
        src = "\n".join(["import shapes::*", _GEO_LOCAL_ONLY, entry])
        phase, cls, span = _file_verdict(tmp_path, {"entry": src, "shapes": _GEO_IMPORTED})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Point"

    @pytest.mark.parametrize("pos_name", sorted(_RECORD_POS))
    def test_repl_grouped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pos_name: str
    ) -> None:
        entry = _RECORD_POS[pos_name].format(q="Geo::Point")
        src = "\n".join(["import shapes::*", _GEO_LOCAL_ONLY, entry])
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {"shapes": _GEO_IMPORTED}, [src])
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Point"

    def test_repl_split_entries_agree_with_one_grouped_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The rejection does not depend on how the setup is grouped into entries."""
        entry = _RECORD_POS["annot"].format(q="Geo::Point")
        entries = ["import shapes::*", _GEO_LOCAL_ONLY, entry]
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {"shapes": _GEO_IMPORTED}, entries)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert entry[span.start_offset : span.end_offset] == "Geo::Point"


class TestImportedScopeRegionMemberIsAccepted:
    """A member uniquely selected through a scope-region prefix is accepted.

    Regression for mutation (b): scope's recording of a uniquely selected
    qualified type name when the prefix is a scope region (rather than a
    type owner), added by item A, is what lets the type positions (which
    have no fallback left to lean on) accept this at all.
    """

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_RECORD_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ACCEPTED_RECORD_POS[pos_name].format(q="Geo::Point")
        src = "\n".join(["import shapes::*", entry])
        phase, _cls, _span = _file_verdict(tmp_path, {"entry": src, "shapes": _GEO_IMPORTED})
        assert phase == "accepted"

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_RECORD_POS))
    def test_repl_grouped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pos_name: str
    ) -> None:
        entry = _ACCEPTED_RECORD_POS[pos_name].format(q="Geo::Point")
        src = "\n".join(["import shapes::*", entry])
        phase, _cls, _span = _repl_verdict(monkeypatch, tmp_path, {"shapes": _GEO_IMPORTED}, [src])
        assert phase == "accepted"

    def test_repl_split_entries_agree_with_one_grouped_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        entry = _ACCEPTED_RECORD_POS["annot"].format(q="Geo::Point")
        entries = ["import shapes::*", entry]
        phase, _cls, _span = _repl_verdict(
            monkeypatch, tmp_path, {"shapes": _GEO_IMPORTED}, entries
        )
        assert phase == "accepted"


# ---------------------------------------------------------------------------
# Item B: the owner is instantiated from scope's recorded key.
# ---------------------------------------------------------------------------


def test_nested_generic_alias_owner_is_instantiated_by_substitution(tmp_path: Path) -> None:
    """An alias target nested inside another generic type substitutes recursively.

    Regression for ``_instantiate_owner_template``: a naive type-argument
    swap (correct for a direct nominal declaration) produced
    ``Slot::Filled[int]`` for ``Rows[int]::Filled``, instead of the correct
    ``Slot::Filled[array[int]]``, because ``Rows[A]``'s own template is
    ``Slot[array[A]]`` -- an alias target that itself wraps another generic
    type, not a bare type parameter.
    """
    src = "\n".join(
        [
            "enum Slot[T]\n  | Filled(value: T)\n  | Empty",
            "type Rows[A] = Slot[array[A]]",
            "let row: Slot[array[int]] = Slot::Filled(value = [1, 2])",
            "case row of\n  | Rows[int]::Filled(value) => value.size() + 1\n  | Slot::Empty => 0",
        ]
    )
    graph = make_graph_from_files(tmp_path, {"entry": src})
    resolved = resolve_program(graph)
    tcp.check_program(resolved, base_caps())


# ---------------------------------------------------------------------------
# Item C: a qualifier whose route names nothing at all -- no module, no
# ``use`` alias, no local scope -- selects none, in every position.
# ---------------------------------------------------------------------------

_UNKNOWN_ROUTE_POS: dict[str, str] = {
    "value": "missing::Item(x = 1)",
    "annot": "fn(p: missing::Item) => 1",
    "alias": "type A = missing::Item\n1",
    "pattern": "let v = 1\ncase v of\n  | missing::Item => 1\n  | _ => 2",
    "is": "1 is missing::Item",
    "tyarg": "fn(p: array[missing::Item]) => 1",
    "applied": "fn(p: missing::Item[int]) => 1",
}


class TestUnknownQualifierRouteAcrossPositions:
    """``missing::Item`` selects none, decided by scope, in all seven positions."""

    @staticmethod
    def _expected_span_text(pos_name: str) -> str:
        # An ``is``-test's span covers the whole ``subject is owner::member``
        # expression, exactly like the "other" outcome's mismatch above.
        return "1 is missing::Item" if pos_name == "is" else "missing::Item"

    @pytest.mark.parametrize("pos_name", sorted(_UNKNOWN_ROUTE_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _UNKNOWN_ROUTE_POS[pos_name]
        phase, cls, span = _file_verdict(tmp_path, {"entry": entry})
        assert (phase, cls) == ("scope", UnknownQualifierError)
        assert span is not None
        assert entry[span.start_offset : span.end_offset] == self._expected_span_text(pos_name)

    @pytest.mark.parametrize("pos_name", sorted(_UNKNOWN_ROUTE_POS))
    def test_repl_grouped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pos_name: str
    ) -> None:
        entry = _UNKNOWN_ROUTE_POS[pos_name]
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {}, [entry])
        assert (phase, cls) == ("scope", UnknownQualifierError)
        assert span is not None
        assert entry[span.start_offset : span.end_offset] == self._expected_span_text(pos_name)


# ---------------------------------------------------------------------------
# A local, bare (non-scoped) nominal type beats a same-named wildcard-
# imported one, exactly as a local scope region does: the local owner's own
# member set is final, never merged with the import's.
# ---------------------------------------------------------------------------

_ENUM_LIB = "record Point\n  x: int\nenum Shape\n  | Circle\n"
_ENUM_LOCAL = "enum Shape\n  | Tri\n"

_ENUM_POS: dict[str, str] = {
    "value": "{q}",
    "pattern": "case 1 of\n  | {q} => 1\n  | _ => 2",
    "is": "let v = 1\nv is {q}",
    "annot": "fn(p: {q}) => 1",
}


class TestLocalEnumShadowsImportedEnum:
    """A local ``enum Shape`` lacking ``Circle`` beats an imported ``Shape::Circle``."""

    @pytest.mark.parametrize("pos_name", sorted(_ENUM_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ENUM_POS[pos_name].format(q="Shape::Circle")
        src = "\n".join(["import lib::*", _ENUM_LOCAL, entry])
        phase, cls, span = _file_verdict(tmp_path, {"entry": src, "lib": _ENUM_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Shape::Circle"

    @pytest.mark.parametrize("pos_name", sorted(_ENUM_POS))
    def test_repl_grouped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pos_name: str
    ) -> None:
        entry = _ENUM_POS[pos_name].format(q="Shape::Circle")
        src = "\n".join(["import lib::*", _ENUM_LOCAL, entry])
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {"lib": _ENUM_LIB}, [src])
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Shape::Circle"

    def test_repl_split_entries_agree_with_one_grouped_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        entry = _ENUM_POS["annot"].format(q="Shape::Circle")
        entries = ["import lib::*", _ENUM_LOCAL, entry]
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {"lib": _ENUM_LIB}, entries)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert entry[span.start_offset : span.end_offset] == "Shape::Circle"


# ---------------------------------------------------------------------------
# ``def Owner::method`` declares *Owner* as a scope path with no member set
# of its own (a method-owner namespace, unlike a scope region or a nominal
# type): it must never shadow an import route sharing its spelling, in the
# pattern and ``is`` positions just as in value and annotation positions --
# nor invent members when no import supplies any at all.
# ---------------------------------------------------------------------------

_DEFPATH_LIB = "scope Geo\n  record Point\n    x: int\nend Geo\n"
_DEFPATH_HEADER = ["import shapes::*", "def Geo::f() -> int = 1"]

_DEFPATH_ACCEPTED_POS: dict[str, str] = {
    "value": "Geo::Point(x = 1)",
    "annot": "fn(p: Geo::Point) => 1",
    "pattern": "let p = shapes::Geo::Point(x = 1)\ncase p of\n  | Geo::Point(x) => x",
}

# A record type is never ``is``-testable, so that position is exercised only
# by the missing-member reproducer below, whose scope-level rejection
# precedes any such typecheck concern.
_DEFPATH_MISSING_POS: dict[str, str] = {
    "value": "Geo::Point(x = 1)",
    "annot": "fn(p: Geo::Point) => 1",
    "pattern": "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2",
    "is": "1 is Geo::Point",
}

# The span a scope-level miss covers depends on its syntactic shape: a bare
# reference or annotation covers just the qualifier chain, a constructor
# pattern covers the whole pattern including its argument list, and an
# ``is``-test covers the whole ``subject is owner::member`` expression.
_DEFPATH_MISSING_SPAN_TEXT: dict[str, str] = {
    "value": "Geo::Point",
    "annot": "Geo::Point",
    "pattern": "Geo::Point(x)",
    "is": "1 is Geo::Point",
}


class TestDefCreatedScopePathDefersToImport:
    """A def-created path with an import in scope resolves through the import."""

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH_ACCEPTED_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH_ACCEPTED_POS[pos_name]
        src = "\n".join([*_DEFPATH_HEADER, entry])
        phase, _cls, _span = _file_verdict(tmp_path, {"entry": src, "shapes": _DEFPATH_LIB})
        assert phase == "accepted"

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH_ACCEPTED_POS))
    def test_repl_grouped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pos_name: str
    ) -> None:
        entry = _DEFPATH_ACCEPTED_POS[pos_name]
        src = "\n".join([*_DEFPATH_HEADER, entry])
        phase, _cls, _span = _repl_verdict(monkeypatch, tmp_path, {"shapes": _DEFPATH_LIB}, [src])
        assert phase == "accepted"

    def test_repl_split_entries_agree_with_one_grouped_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        entries = [*_DEFPATH_HEADER, _DEFPATH_ACCEPTED_POS["pattern"]]
        phase, _cls, _span = _repl_verdict(monkeypatch, tmp_path, {"shapes": _DEFPATH_LIB}, entries)
        assert phase == "accepted"


class TestDefCreatedScopePathWithoutImportIsUnknownMember:
    """A def-created path with no import at all rejects every position identically."""

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH_MISSING_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH_MISSING_POS[pos_name]
        src = "\n".join(["def Geo::f() -> int = 1", entry])
        phase, cls, span = _file_verdict(tmp_path, {"entry": src})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == _DEFPATH_MISSING_SPAN_TEXT[pos_name]

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH_MISSING_POS))
    def test_repl_grouped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pos_name: str
    ) -> None:
        entry = _DEFPATH_MISSING_POS[pos_name]
        src = "\n".join(["def Geo::f() -> int = 1", entry])
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {}, [src])
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == _DEFPATH_MISSING_SPAN_TEXT[pos_name]

    def test_repl_split_entries_agree_with_one_grouped_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        entries = ["def Geo::f() -> int = 1", self._entry()]
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {}, entries)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert self._entry()[span.start_offset : span.end_offset] == "Geo::Point"

    @staticmethod
    def _entry() -> str:
        return _DEFPATH_MISSING_POS["annot"]


# ---------------------------------------------------------------------------
# A local scope's own reading, once decisive, is walked exactly regardless of
# the qualifier chain's length: a 2-segment miss (the owner's own missing
# member) and a 3-segment miss (a member nested one level deeper, inside an
# owner nested in the same local scope) both reject with the identical
# verdict the local owner's member set alone decides.
# ---------------------------------------------------------------------------

_CHAIN_LIB = (
    "scope Geo\n  record Point\n    x: int\n  enum Shape\n    | Circle\n    | Square\nend Geo\n"
)
_CHAIN_LOCAL = "scope Geo\n  enum Kind = Round | Flat\nend Geo"

_CHAIN_LEN2_POS: dict[str, str] = {
    "value": "Geo::Point(x = 1)",
    "annot": "fn(p: Geo::Point) => 1",
    "pattern": "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2",
}
_CHAIN_LEN3_POS: dict[str, str] = {
    "value": "Geo::Shape::Circle",
    "annot": "fn(p: Geo::Shape::Circle) => 1",
    "pattern": "case 1 of\n  | Geo::Shape::Circle => 1\n  | _ => 2",
    "is": "1 is Geo::Shape::Circle",
}


class TestLocalScopeChainLengthsAgree:
    """A decisive local scope's missing member rejects at length 2 and length 3 alike."""

    @pytest.mark.parametrize("pos_name", sorted(_CHAIN_LEN2_POS))
    def test_length_two_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _CHAIN_LEN2_POS[pos_name]
        src = "\n".join(["import shapes::*", _CHAIN_LOCAL, entry])
        phase, cls, span = _file_verdict(tmp_path, {"entry": src, "shapes": _CHAIN_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Point"

    @pytest.mark.parametrize("pos_name", sorted(_CHAIN_LEN3_POS))
    def test_length_three_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _CHAIN_LEN3_POS[pos_name]
        src = "\n".join(["import shapes::*", _CHAIN_LOCAL, entry])
        phase, cls, span = _file_verdict(tmp_path, {"entry": src, "shapes": _CHAIN_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Shape"

    @pytest.mark.parametrize("pos_name", sorted(_CHAIN_LEN3_POS))
    def test_length_three_repl_grouped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pos_name: str
    ) -> None:
        entry = _CHAIN_LEN3_POS[pos_name]
        src = "\n".join(["import shapes::*", _CHAIN_LOCAL, entry])
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {"shapes": _CHAIN_LIB}, [src])
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Shape"

    def test_repl_split_entries_agree_with_one_grouped_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        entries = ["import shapes::*", _CHAIN_LOCAL, _CHAIN_LEN3_POS["annot"]]
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {"shapes": _CHAIN_LIB}, entries)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert _CHAIN_LEN3_POS["annot"][span.start_offset : span.end_offset] == "Shape"

    def test_length_four_misses_one_level_past_a_walked_hit_file(self, tmp_path: Path) -> None:
        """A 3-segment chain whose first two segments walk successfully rejects at the third.

        ``Shape`` nested inside the local ``Geo`` region exists (the walk's
        own first step hits), but nothing beneath it is named ``Missing``:
        the walk's own second step is what rejects, not its first.
        """
        entry = "Geo::Shape::Missing::X"
        src = "\n".join([_CHAIN_NESTED_LOCAL, entry])
        phase, cls, span = _file_verdict(tmp_path, {"entry": src})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Missing"

    def test_length_four_misses_one_level_past_a_walked_hit_repl(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        entry = "Geo::Shape::Missing::X"
        src = "\n".join([_CHAIN_NESTED_LOCAL, entry])
        phase, cls, span = _repl_verdict(monkeypatch, tmp_path, {}, [src])
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Missing"


_CHAIN_NESTED_LOCAL = (
    "scope Geo\n\n  scope Shape\n    enum Kind = Round | Flat\n  end Shape\nend Geo"
)

_ACCEPTED_CHAIN_LEN2_POS: dict[str, str] = {
    "value": "Geo::Point(x = 1)",
    "annot": "fn(p: Geo::Point) => 1",
    "pattern": "let v: Geo::Point = Geo::Point(x = 1)\ncase v of\n  | Geo::Point(x) => x",
}
_ACCEPTED_CHAIN_LEN3_POS: dict[str, str] = {
    "value": "Geo::Shape::Circle",
    "annot": "fn(p: Geo::Shape::Circle) => 1",
    "pattern": (
        "let v: Geo::Shape = Geo::Shape::Circle\ncase v of\n  | Geo::Shape::Circle => 1\n  | _ => 2"
    ),
    "is": "let v: Geo::Shape = Geo::Shape::Circle\nv is Geo::Shape::Circle",
}


class TestLocalScopeChainLengthsAreAcceptedWhenDeclaredLocally:
    """A decisive local scope's own member is accepted, walked exactly, at length 2 and 3 alike.

    ``_CHAIN_LIB``'s shape, declared locally instead of imported, exercises
    the same exact-path walk ``TestLocalScopeChainLengthsAgree`` exercises
    for a miss, this time to a hit at every step: the walk's own multi-
    segment success path, distinct from a same-spelled import ever being
    reachable at all.
    """

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_CHAIN_LEN2_POS))
    def test_length_two_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ACCEPTED_CHAIN_LEN2_POS[pos_name]
        src = "\n".join([_CHAIN_LIB, entry])
        phase, _cls, _span = _file_verdict(tmp_path, {"entry": src})
        assert phase == "accepted"

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_CHAIN_LEN3_POS))
    def test_length_three_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ACCEPTED_CHAIN_LEN3_POS[pos_name]
        src = "\n".join([_CHAIN_LIB, entry])
        phase, _cls, _span = _file_verdict(tmp_path, {"entry": src})
        assert phase == "accepted"

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_CHAIN_LEN3_POS))
    def test_length_three_repl_grouped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pos_name: str
    ) -> None:
        entry = _ACCEPTED_CHAIN_LEN3_POS[pos_name]
        src = "\n".join([_CHAIN_LIB, entry])
        phase, _cls, _span = _repl_verdict(monkeypatch, tmp_path, {}, [src])
        assert phase == "accepted"

    def test_repl_split_entries_agree_with_one_grouped_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        entries = [_CHAIN_LIB, _ACCEPTED_CHAIN_LEN3_POS["annot"]]
        phase, _cls, _span = _repl_verdict(monkeypatch, tmp_path, {}, entries)
        assert phase == "accepted"
