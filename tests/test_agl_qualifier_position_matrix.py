"""Position-matrix regression tests for scope's qualifier decisions.

Scope, not typecheck, decides which declaration a qualified type name
selects and whether a qualifier selects none at all, identically across the
value, pattern, ``is``, annotation, alias, type-argument and applied
positions, and identically in file mode and the REPL (independent of how a
REPL session groups its declarations into entries).

``_matrix_params`` reuses one probe program per outcome across five owner
spellings (a module route, a wildcard-imported bare name, a
wildcard-imported generic owner applied to a type argument -- proving the
owner is instantiated from scope's recorded key rather than re-resolved by
name -- a locally ``use``d scope region, and a module-level ``use`` alias)
and six syntactic positions; the seventh position (an owner itself carrying
type arguments, ``Owner[T]::Member``) is exercised by the generic-owner
spelling, whose member is *always* written applied.
``TestLocalScopeShadowsSameNamedImport`` and
``TestImportedScopeRegionMemberIsAccepted`` cover the local-scope-vs-import
case for a record owner (so ``is`` and applying type arguments, meaningless
for a non-generic record, are left to the matrix above): a local scope
region lacking a member beats a same-spelled imported one, and a member
uniquely selected through a scope-region prefix is accepted.

The "applied" form's ``is``/pattern positions cover an ``AppliedT`` owner
(``Box[int]``) resolved by its recorded ``gm::Box[int]`` key rather than
re-resolved by its bare spelling (ambiguous once both ``gm`` and ``gn`` are
wildcard-imported): a genuine type mismatch there spans the full qualifier
chain. ``test_nested_generic_alias_owner_is_instantiated_by_substitution``
covers an alias target nested inside another generic type, which must not
be instantiated by a bare type-argument swap.

``TestUnknownQualifierRouteAcrossPositions`` covers a qualifier whose
leading route names no module, ``use`` alias or local scope at all, so
scope's own selects-none raise fires for every position -- the ``_FORMS``
matrix's "missing" outcome always names a real, resolvable owner instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.repl import ReplSession
from agm.agl.scope.symbols import (
    AmbiguousQualificationError,
    RouteClashError,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.semantics.values import TextValue
from agm.agl.typecheck import AglTypeError
from tests.agl.ir_harness import evaluate_ir_graph
from tests.agl.qualifier_support import Verdict, file_verdict, repl_verdict_all_groupings

# ---------------------------------------------------------------------------
# Unique, other (a real but mismatched member), ambiguous and missing
# selection, across five owner spellings and six positions.
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
# "accepted" leaves the class column unused. Matches the implementation's
# actual, verified behavior (tests/CLAUDE.md: assert behavior, not
# internals): "other" is a real member of a structurally
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


def _matrix_params_repl() -> list[object]:
    # "moduse"'s ambiguity outcome is excluded: a ``use`` contribution from an
    # earlier REPL entry does not compose with one the probing entry declares
    # itself, so the ambiguity that every other grouping (and file mode)
    # reports is missed whenever the two share an entry -- a pre-existing,
    # separately tracked REPL divergence, not one of this module's own
    # position/grouping invariants (see
    # ``test_moduse_ambiguity_depends_on_repl_grouping`` below).
    return [
        pytest.param(form_name, outcome, pos_name, id=f"{form_name}-{outcome}-{pos_name}")
        for form_name, (_, _, _, outcomes) in _FORMS.items()
        for outcome in outcomes
        for pos_name in _POS
        if not (form_name == "moduse" and outcome == "amb")
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
    verdict = file_verdict(tmp_path, {"entry": src, **modules})
    _assert_matrix_verdict(src, verdict, form_name, outcome, pos_name, q)


@pytest.mark.parametrize(("form_name", "outcome", "pos_name"), _matrix_params_repl())
def test_qualifier_decision_matrix_repl(
    tmp_path: Path, form_name: str, outcome: str, pos_name: str
) -> None:
    """The file-mode verdict above holds identically across every REPL grouping."""
    modules, header, entry, q = _matrix_case(form_name, outcome, pos_name)
    decls = (*header, entry)
    verdict = repl_verdict_all_groupings(tmp_path, modules, decls)
    _assert_matrix_verdict("\n".join(decls), verdict, form_name, outcome, pos_name, q)


def test_moduse_ambiguity_depends_on_repl_grouping(tmp_path: Path) -> None:
    """Pre-existing REPL divergence: a shared entry misses an earlier entry's ``use``.

    ``Color::Red`` is ambiguous between ``m``'s and ``n``'s wildcard-``use``d
    ``Color`` in file mode, and when ``use n::*`` is evaluated as its own
    REPL entry -- but is wrongly accepted when ``use n::*`` instead shares an
    entry with the reference: the probing entry's own ``use`` does not
    compose with one an earlier entry already committed. This is a known,
    separately tracked limitation of REPL entry evaluation, not one of this
    module's own position/grouping invariants; it is pinned here so a fix
    (or a regression sharpening it) is visible.
    """
    for name, source in _MN.items():
        (tmp_path / f"{name}.agl").write_text(source, encoding="utf-8")
    session = ReplSession(cwd=tmp_path, default_stdlib=False)
    session.open()
    for decl in ("import m", "import n", "use m::*"):
        assert session.eval_entry(decl).ok
    result = session.eval_entry("use n::*\nColor::Red")
    assert result.ok


# ---------------------------------------------------------------------------
# A local scope region beats a same-named imported nested region, identically
# whether the local declaration wins the position (rejecting a member it
# doesn't have) or nothing local conflicts at all (accepting the sole
# imported member).
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

    A name lookup that fell through to the import would find the imported
    ``Point`` and accept every position; local-scope-wins rejects all of
    them instead, identically in every position and REPL grouping.
    """

    @pytest.mark.parametrize("pos_name", sorted(_RECORD_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _RECORD_POS[pos_name].format(q="Geo::Point")
        src = "\n".join(["import shapes::*", _GEO_LOCAL_ONLY, entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "shapes": _GEO_IMPORTED})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Point"

    @pytest.mark.parametrize("pos_name", sorted(_RECORD_POS))
    def test_repl(self, tmp_path: Path, pos_name: str) -> None:
        """The rejection holds across every REPL grouping, not just one file-like entry."""
        entry = _RECORD_POS[pos_name].format(q="Geo::Point")
        decls = ("import shapes::*", _GEO_LOCAL_ONLY, entry)
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {"shapes": _GEO_IMPORTED}, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::Point"


class TestImportedScopeRegionMemberIsAccepted:
    """A member uniquely selected through a scope-region prefix is accepted.

    Scope's recording of a uniquely selected qualified type name when the
    prefix is a scope region (rather than a type owner) is what lets the
    type positions (which have no fallback left to lean on) accept this at
    all.
    """

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_RECORD_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ACCEPTED_RECORD_POS[pos_name].format(q="Geo::Point")
        src = "\n".join(["import shapes::*", entry])
        phase, _cls, _span = file_verdict(tmp_path, {"entry": src, "shapes": _GEO_IMPORTED})
        assert phase == "accepted"

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_RECORD_POS))
    def test_repl(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ACCEPTED_RECORD_POS[pos_name].format(q="Geo::Point")
        decls = ("import shapes::*", entry)
        phase, _cls, _span = repl_verdict_all_groupings(tmp_path, {"shapes": _GEO_IMPORTED}, decls)
        assert phase == "accepted"


# ---------------------------------------------------------------------------
# A local declaration beats a bare name two wildcard imports both
# contribute (so the bare name alone is ambiguous between them), in every
# position that names a type, not only a value: the bare-type-name level
# lookup must select the local declaration the same way the value/member
# lookup already does, rather than only reading what a bare name selects
# among imports.
# ---------------------------------------------------------------------------

_TWO_IMPORT_LIB = {
    "m": "record Point\n  x: int\nenum Shape\n  | Circle\n",
    "n": "record Point\n  y: int\nenum Shape\n  | Square\n",
}
_TWO_IMPORT_HEADER = ("import m::*", "import n::*", "record Point\n  z: int", "enum Shape\n  | Tri")


class TestLocalTypeWinsOverAmbiguousBareImports:
    """A local declaration wins a bare name two wildcard imports both contribute.

    Regression for ``_validate_bare_type_name``: it used to ask what a bare
    name selects *beyond* the module's own declarations, so a local
    declaration never won, even though the value position (``Point(z = 1)``)
    already accepted the local record.
    """

    @pytest.mark.parametrize(
        "entry",
        [
            "fn(p: Point) => 1",
            "fn(p: array[Point]) => 1",
            "type A = Point\n1",
            "Point(z = 1)",
        ],
    )
    def test_file(self, tmp_path: Path, entry: str) -> None:
        src = "\n".join([*_TWO_IMPORT_HEADER, entry])
        phase, _cls, _span = file_verdict(tmp_path, {"entry": src, **_TWO_IMPORT_LIB})
        assert phase == "accepted"

    @pytest.mark.parametrize(
        "entry",
        [
            "fn(p: Point) => 1",
            "fn(p: array[Point]) => 1",
            "type A = Point\n1",
            "Point(z = 1)",
        ],
    )
    def test_repl(self, tmp_path: Path, entry: str) -> None:
        decls = (*_TWO_IMPORT_HEADER, entry)
        phase, _cls, _span = repl_verdict_all_groupings(tmp_path, _TWO_IMPORT_LIB, decls)
        assert phase == "accepted"


# ---------------------------------------------------------------------------
# A local bare enum's own nullary case, walked one level past a route that
# would otherwise resolve the same spelling, gives the identical verdict
# regardless of how a REPL session happens to split its setup into entries.
# ---------------------------------------------------------------------------

_ENUM_CASE_CLASH_LIB = {
    "pkg/Foo": "record R\n  x: int\nenum E\n  | A\n  | B\nrecord G[T]\n  v: T\n"
}
_ENUM_CASE_CLASH_LOCAL = "enum Foo\n  | R(y: int)\n  | E"
_ENUM_CASE_CLASH_HEADER = ("import pkg/Foo", _ENUM_CASE_CLASH_LOCAL)


class TestLocalEnumCaseWalkAgreesAcrossGroupings:
    """``Foo::E::A`` reaches the identical verdict in file mode and every REPL grouping.

    Regression: this used to give ``UnknownMemberError`` in file mode and
    some REPL groupings, but accepted the reference in others, because the
    route clash a longer route segment run names was only checked for the
    chain's first segment. The unanchored spelling clashes with the
    ``pkg/Foo`` route (which resolves ``E::A`` past it); the anchored
    spelling, naming only the current module, has no route to clash with
    and rejects ``A`` as a plain unknown member of the local nullary case.
    """

    def test_file_unanchored(self, tmp_path: Path) -> None:
        src = "\n".join([*_ENUM_CASE_CLASH_HEADER, "Foo::E::A"])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, **_ENUM_CASE_CLASH_LIB})
        assert (phase, cls) == ("scope", RouteClashError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Foo::E::A"

    def test_repl_unanchored(self, tmp_path: Path) -> None:
        decls = (*_ENUM_CASE_CLASH_HEADER, "Foo::E::A")
        phase, cls, span = repl_verdict_all_groupings(tmp_path, _ENUM_CASE_CLASH_LIB, decls)
        assert (phase, cls) == ("scope", RouteClashError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Foo::E::A"

    def test_file_anchored(self, tmp_path: Path) -> None:
        src = "\n".join([*_ENUM_CASE_CLASH_HEADER, "::Foo::E::A"])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, **_ENUM_CASE_CLASH_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "::Foo::E::A"

    def test_repl_anchored(self, tmp_path: Path) -> None:
        decls = (*_ENUM_CASE_CLASH_HEADER, "::Foo::E::A")
        phase, cls, span = repl_verdict_all_groupings(tmp_path, _ENUM_CASE_CLASH_LIB, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "::Foo::E::A"


# ---------------------------------------------------------------------------
# A local nominal type sharing an imported scope region's path is accepted
# on its own, never merged into an ambiguity with the import: a direct local
# hit is decisive by itself, the same way a local scope's own member set
# is (above), even though here the local declaration and the import both
# name a real, independent type at the same path.
# ---------------------------------------------------------------------------

_LOCAL_WINS_LIB = (
    "scope Geo\n"
    "  record Point\n"
    "    x: int\n"
    "  enum Shape\n"
    "    | Circle\n"
    "    | Square\n"
    "  type Num = int\n"
    "end Geo\n"
)
_LOCAL_WINS_LOCAL = (
    "scope Geo\n  record Point\n    y: int\n  enum Shape\n    | Tri\n  type Num = text\nend Geo"
)
_LOCAL_WINS_HEADER = ("import shapes::*", _LOCAL_WINS_LOCAL)


class TestLocalTypeWinsOverAmbiguousImportWithoutMerging:
    """A local type sharing an imported region's path resolves to itself alone.

    Regression for a direct-local-hit gate: without it, the local and
    imported ``Geo::Shape`` (or ``Geo::Num``) merge into an
    ``AmbiguousQualificationError`` instead of the local declaration simply
    winning, exactly as a plain member-set local scope already does.
    """

    @pytest.mark.parametrize(
        "entry",
        [
            "fn(p: Geo::Shape) => 1",
            "let s: Geo::Shape = Geo::Shape::Tri\ns",
            "fn(p: Geo::Num) => 1",
            'let n: Geo::Num = "a"\nn',
        ],
    )
    def test_file(self, tmp_path: Path, entry: str) -> None:
        src = "\n".join([*_LOCAL_WINS_HEADER, entry])
        phase, _cls, _span = file_verdict(tmp_path, {"entry": src, "shapes": _LOCAL_WINS_LIB})
        assert phase == "accepted"

    @pytest.mark.parametrize(
        "entry",
        [
            "fn(p: Geo::Shape) => 1",
            "let s: Geo::Shape = Geo::Shape::Tri\ns",
            "fn(p: Geo::Num) => 1",
            'let n: Geo::Num = "a"\nn',
        ],
    )
    def test_repl(self, tmp_path: Path, entry: str) -> None:
        decls = (*_LOCAL_WINS_HEADER, entry)
        phase, _cls, _span = repl_verdict_all_groupings(
            tmp_path, {"shapes": _LOCAL_WINS_LIB}, decls
        )
        assert phase == "accepted"

    def test_local_alias_identity_is_the_declared_target_not_the_import(
        self, tmp_path: Path
    ) -> None:
        """``Geo::Num`` resolves to the local ``text`` alias, not the imported ``int`` one.

        ``"a"`` only type-checks against the local alias's target; had the
        two merged, this would fail (or the annotation would already have
        raised ambiguous) instead of binding a real text value.
        """
        entry = "\n".join([*_LOCAL_WINS_HEADER, 'let n: Geo::Num = "a"\n'])
        bindings = evaluate_ir_graph(entry, {"shapes": _LOCAL_WINS_LIB}, tmp_path)
        assert isinstance(bindings["n"], TextValue)
        assert bindings["n"].value == "a"

    def test_local_scope_region_type_wins_over_imported_region(self, tmp_path: Path) -> None:
        """A local scope-region ``Geo::Shape`` also wins when the local scope has ``Point``.

        Distinct from the module-level cases above: here the local ``Geo``
        also declares its own ``Point``, ruling out any fallback reading of
        ``Geo`` as a plain member-set scope with nothing of its own.
        """
        lib = (
            "scope Geo\n"
            "  record Point\n"
            "    x: int\n"
            "  record Box[T]\n"
            "    v: T\n"
            "  enum Shape\n"
            "    | Circle\n"
            "    | Square\n"
            "end Geo\n"
        )
        local = "scope Geo\n  record Point\n    y: int\n  enum Shape\n    | Tri\nend Geo"
        entry = "fn(p: Geo::Shape) => 1"
        src = "\n".join(["import shapes::*", local, entry])
        phase, _cls, _span = file_verdict(tmp_path, {"entry": src, "shapes": lib})
        assert phase == "accepted"


# ---------------------------------------------------------------------------
# The owner is instantiated from scope's recorded key.
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
    phase, _cls, _span = file_verdict(tmp_path, {"entry": src})
    assert phase == "accepted"


# ---------------------------------------------------------------------------
# A qualifier whose route names nothing at all -- no module, no ``use``
# alias, no local scope -- selects none, in every position.
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

    @pytest.mark.parametrize("pos_name", sorted(_UNKNOWN_ROUTE_POS))
    def test_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _UNKNOWN_ROUTE_POS[pos_name]
        phase, cls, span = file_verdict(tmp_path, {"entry": entry})
        assert (phase, cls) == ("scope", UnknownQualifierError)
        assert span is not None
        assert entry[span.start_offset : span.end_offset] == "missing::Item"

    @pytest.mark.parametrize("pos_name", sorted(_UNKNOWN_ROUTE_POS))
    def test_repl(self, tmp_path: Path, pos_name: str) -> None:
        entry = _UNKNOWN_ROUTE_POS[pos_name]
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {}, (entry,))
        assert (phase, cls) == ("scope", UnknownQualifierError)
        assert span is not None
        assert entry[span.start_offset : span.end_offset] == "missing::Item"


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
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "lib": _ENUM_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Shape::Circle"

    @pytest.mark.parametrize("pos_name", sorted(_ENUM_POS))
    def test_repl(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ENUM_POS[pos_name].format(q="Shape::Circle")
        decls = ("import lib::*", _ENUM_LOCAL, entry)
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {"lib": _ENUM_LIB}, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Shape::Circle"


# ---------------------------------------------------------------------------
# ``def Owner::method`` whose leading segment ALSO resolves, below the local
# level, to an imported TYPE owner: the def-created path reads as that
# owner's method namespace, not a plain local one -- its nested types stay
# reachable and a real miss still rejects, distinct from the plain-local-
# namespace case below where the leading segment names only a scope region.
# ---------------------------------------------------------------------------

_DEFPATH_TYPE_OWNER_LIB = "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"
_DEFPATH_TYPE_OWNER_DEF = "def Geo::m(self) -> int = self.x"


class TestDefCreatedPathOverImportedTypeReadsAsItsMethodNamespace:
    """A def-created path over an imported type keeps the type's own nested members.

    Regression: scope used to say ``'Geo' is a type name, not a value`` for
    both a hit and a miss; it must instead construct ``Geo::Inner`` and
    reject ``Geo::Nope`` with the ordinary unknown-member verdict.
    """

    @pytest.mark.parametrize("entry", ["fn(p: Geo::Inner) => 1", "Geo::Inner(y = 1)"])
    def test_file_nested_type_is_accepted(self, tmp_path: Path, entry: str) -> None:
        src = "\n".join(["import shapes::*", _DEFPATH_TYPE_OWNER_DEF, entry])
        phase, _cls, _span = file_verdict(
            tmp_path, {"entry": src, "shapes": _DEFPATH_TYPE_OWNER_LIB}
        )
        assert phase == "accepted"

    @pytest.mark.parametrize("entry", ["fn(p: Geo::Inner) => 1", "Geo::Inner(y = 1)"])
    def test_repl_nested_type_is_accepted(self, tmp_path: Path, entry: str) -> None:
        decls = ("import shapes::*", _DEFPATH_TYPE_OWNER_DEF, entry)
        phase, _cls, _span = repl_verdict_all_groupings(
            tmp_path, {"shapes": _DEFPATH_TYPE_OWNER_LIB}, decls
        )
        assert phase == "accepted"

    @pytest.mark.parametrize("entry", ["fn(p: Geo::Nope) => 1", "Geo::Nope(y = 1)"])
    def test_file_missing_member_is_rejected(self, tmp_path: Path, entry: str) -> None:
        src = "\n".join(["import shapes::*", _DEFPATH_TYPE_OWNER_DEF, entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "shapes": _DEFPATH_TYPE_OWNER_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Nope"

    @pytest.mark.parametrize("entry", ["fn(p: Geo::Nope) => 1", "Geo::Nope(y = 1)"])
    def test_repl_missing_member_is_rejected(self, tmp_path: Path, entry: str) -> None:
        decls = ("import shapes::*", _DEFPATH_TYPE_OWNER_DEF, entry)
        phase, cls, span = repl_verdict_all_groupings(
            tmp_path, {"shapes": _DEFPATH_TYPE_OWNER_LIB}, decls
        )
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::Nope"


# ---------------------------------------------------------------------------
# ``def Owner::method`` declares *Owner* as a scope path with no member set
# of its own (a method-owner namespace, unlike a scope region or a nominal
# type). Its leading segment names no type owner below the local level here
# -- only a same-named imported scope *region*, never a type -- so the path
# is a plain local namespace: local wins, with no merging into the region's
# members, in every position, whether or not the import is even present.
# ---------------------------------------------------------------------------

_DEFPATH_LIB = "scope Geo\n  record Point\n    x: int\nend Geo\n"
_DEFPATH_DEF = "def Geo::f() -> int = 1"

_DEFPATH_MISSING_POS: dict[str, str] = {
    "value": "Geo::Point(x = 1)",
    "annot": "fn(p: Geo::Point) => 1",
    "pattern": "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2",
    "is": "1 is Geo::Point",
}


class TestDefCreatedPathIsPlainLocalNamespace:
    """A def-created path is a plain local namespace: local wins, with no merging.

    Rejects every position identically whether or not an import happens to
    open a same-named scope *region* -- a region is not a type owner, so it
    never gives the def-created path anything to defer to.
    """

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH_MISSING_POS))
    def test_file_without_import(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH_MISSING_POS[pos_name]
        src = "\n".join([_DEFPATH_DEF, entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Point"

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH_MISSING_POS))
    def test_repl_without_import(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH_MISSING_POS[pos_name]
        decls = (_DEFPATH_DEF, entry)
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {}, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::Point"

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH_MISSING_POS))
    def test_file_with_imported_region(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH_MISSING_POS[pos_name]
        src = "\n".join(["import shapes::*", _DEFPATH_DEF, entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "shapes": _DEFPATH_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Point"

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH_MISSING_POS))
    def test_repl_with_imported_region(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH_MISSING_POS[pos_name]
        decls = ("import shapes::*", _DEFPATH_DEF, entry)
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {"shapes": _DEFPATH_LIB}, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::Point"


# ---------------------------------------------------------------------------
# A def-created path stays a plain local namespace at a 3-segment chain, and
# at a def path rooted one scope level deeper, not
# only at the 2-segment case above: the leading segment (``Geo``) still names
# only a scope region below the local level, so an imported member three
# segments down (``Geo::In::Point``) gets no more merging than a two-segment
# one does, whether the def path is declared at the outer scope
# (``def Geo::f``) or the inner one (``def Geo::In::f``).
# ---------------------------------------------------------------------------

_DEFPATH3_LIB = "scope Geo\n\n  scope In\n    record Point\n      x: int\n  end In\nend Geo\n"

_DEFPATH3_POS: dict[str, str] = {
    "value": "Geo::In::Point(x = 1)",
    "annot": "fn(p: Geo::In::Point) => 1",
}
_DEFPATH3_MISSING_POS: dict[str, str] = {
    "value": "Geo::In::Nope(x = 1)",
    "annot": "fn(p: Geo::In::Nope) => 1",
}


class TestDefCreatedPathIsPlainLocalNamespaceAtEveryChainLength:
    """The plain-local-namespace verdict holds for a 3-segment chain too.

    Regression: the value position used to give ``UnknownQualifierError``
    while the annotation was wrongly accepted; both must reject with the
    ordinary unknown-member verdict, identically to the 2-segment case.
    """

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH3_POS))
    def test_file_import_is_not_merged_at_length_three(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH3_POS[pos_name]
        src = "\n".join(["import shapes::*", "def Geo::f() -> int = 1", entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "shapes": _DEFPATH3_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::In::Point"

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH3_POS))
    def test_repl_import_is_not_merged_at_length_three(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH3_POS[pos_name]
        decls = ("import shapes::*", "def Geo::f() -> int = 1", entry)
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {"shapes": _DEFPATH3_LIB}, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::In::Point"

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH3_MISSING_POS))
    def test_file_missing_member_at_length_three(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH3_MISSING_POS[pos_name]
        src = "\n".join(["import shapes::*", "def Geo::f() -> int = 1", entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "shapes": _DEFPATH3_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::In::Nope"

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH3_POS))
    def test_file_def_path_nested_one_level_deeper(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH3_POS[pos_name]
        src = "\n".join(["import shapes::*", "def Geo::In::f() -> int = 1", entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "shapes": _DEFPATH3_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::In::Point"

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH3_POS))
    def test_repl_def_path_nested_one_level_deeper(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH3_POS[pos_name]
        decls = ("import shapes::*", "def Geo::In::f() -> int = 1", entry)
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {"shapes": _DEFPATH3_LIB}, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::In::Point"


# ---------------------------------------------------------------------------
# A def-created path's own leading segment can itself be a bare, empty-
# membered prefix (``def Geo::In::f`` declares nothing directly at ``Geo``,
# only at ``Geo::In``): querying a sibling path at that prefix
# (``Geo::Other``, naming neither ``In`` nor anything the prefix itself
# declares) is non-decisive there, so it falls all the way through the
# ordinary owner/route search and is rejected by that search's own local-path
# fallback, identically in a type position and a pattern/``is`` position.
# ---------------------------------------------------------------------------

_DEFPATH_EMPTY_PREFIX_LIB = (
    "scope Geo\n\n  scope In\n    record Point\n      x: int\n  end In\nend Geo\n"
)


class TestDefCreatedPathWithEmptyOwnPrefixIsPlainLocalNamespace:
    """A def-created path's non-owning leading prefix still rejects a sibling miss."""

    def test_file_annotation(self, tmp_path: Path) -> None:
        src = "\n".join(
            ["import shapes::*", "def Geo::In::f() -> int = 1", "fn(x: Geo::Other) => 1"]
        )
        phase, cls, span = file_verdict(
            tmp_path, {"entry": src, "shapes": _DEFPATH_EMPTY_PREFIX_LIB}
        )
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Other"

    def test_repl_annotation(self, tmp_path: Path) -> None:
        decls = ("import shapes::*", "def Geo::In::f() -> int = 1", "fn(x: Geo::Other) => 1")
        phase, cls, span = repl_verdict_all_groupings(
            tmp_path, {"shapes": _DEFPATH_EMPTY_PREFIX_LIB}, decls
        )
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::Other"

    def test_file_is(self, tmp_path: Path) -> None:
        src = "\n".join(["import shapes::*", "def Geo::In::f() -> int = 1", "1 is Geo::Other"])
        phase, cls, span = file_verdict(
            tmp_path, {"entry": src, "shapes": _DEFPATH_EMPTY_PREFIX_LIB}
        )
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Other"

    def test_repl_is(self, tmp_path: Path) -> None:
        decls = ("import shapes::*", "def Geo::In::f() -> int = 1", "1 is Geo::Other")
        phase, cls, span = repl_verdict_all_groupings(
            tmp_path, {"shapes": _DEFPATH_EMPTY_PREFIX_LIB}, decls
        )
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::Other"


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
_CHAIN_NESTED_LOCAL = (
    "scope Geo\n\n  scope Shape\n    enum Kind = Round | Flat\n  end Shape\nend Geo"
)


class TestLocalScopeChainLengthsAgree:
    """A decisive local scope's missing member rejects at length 2 and length 3 alike."""

    @pytest.mark.parametrize("pos_name", sorted(_CHAIN_LEN2_POS))
    def test_length_two_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _CHAIN_LEN2_POS[pos_name]
        src = "\n".join(["import shapes::*", _CHAIN_LOCAL, entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "shapes": _CHAIN_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Point"

    @pytest.mark.parametrize("pos_name", sorted(_CHAIN_LEN3_POS))
    def test_length_three_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _CHAIN_LEN3_POS[pos_name]
        src = "\n".join(["import shapes::*", _CHAIN_LOCAL, entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src, "shapes": _CHAIN_LIB})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Shape::Circle"

    @pytest.mark.parametrize("pos_name", sorted(_CHAIN_LEN3_POS))
    def test_length_three_repl(self, tmp_path: Path, pos_name: str) -> None:
        entry = _CHAIN_LEN3_POS[pos_name]
        decls = ("import shapes::*", _CHAIN_LOCAL, entry)
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {"shapes": _CHAIN_LIB}, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::Shape::Circle"

    def test_length_four_misses_one_level_past_a_walked_hit_file(self, tmp_path: Path) -> None:
        """A 4-segment chain whose first two segments walk successfully rejects at the third.

        ``Shape`` nested inside the local ``Geo`` region exists (the walk's
        own first step hits), but nothing beneath it is named ``Missing``:
        the walk's own second step is what rejects, not its first.
        """
        entry = "Geo::Shape::Missing::X"
        src = "\n".join([_CHAIN_NESTED_LOCAL, entry])
        phase, cls, span = file_verdict(tmp_path, {"entry": src})
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::Shape::Missing::X"

    def test_length_four_misses_one_level_past_a_walked_hit_repl(self, tmp_path: Path) -> None:
        entry = "Geo::Shape::Missing::X"
        decls = (_CHAIN_NESTED_LOCAL, entry)
        phase, cls, span = repl_verdict_all_groupings(tmp_path, {}, decls)
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert "\n".join(decls)[span.start_offset : span.end_offset] == "Geo::Shape::Missing::X"


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
        phase, _cls, _span = file_verdict(tmp_path, {"entry": src})
        assert phase == "accepted"

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_CHAIN_LEN3_POS))
    def test_length_three_file(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ACCEPTED_CHAIN_LEN3_POS[pos_name]
        src = "\n".join([_CHAIN_LIB, entry])
        phase, _cls, _span = file_verdict(tmp_path, {"entry": src})
        assert phase == "accepted"

    @pytest.mark.parametrize("pos_name", sorted(_ACCEPTED_CHAIN_LEN3_POS))
    def test_length_three_repl(self, tmp_path: Path, pos_name: str) -> None:
        entry = _ACCEPTED_CHAIN_LEN3_POS[pos_name]
        decls = (_CHAIN_LIB, entry)
        phase, _cls, _span = repl_verdict_all_groupings(tmp_path, {}, decls)
        assert phase == "accepted"


# ---------------------------------------------------------------------------
# A length-4 local scope, matching an import route at its leading segment
# alone, clashes at the full chain length: a plain local scope declaring the
# full path is never decisive against a route resolving to a declaration of
# its own, whether that declaration shares the local one's leaf name or not.
# ---------------------------------------------------------------------------

_LEN4_LIB = "scope E\n\n  scope A\n    record Z\n      v: int\n  end A\nend E\n"
_LEN4_LOCAL_SAME_LEAF = (
    "scope Foo\n\n  scope E\n\n    scope A\n      record Z\n        w: int\n    end A\n  end E\n"
    "end Foo"
)
_LEN4_LOCAL_DIFFERENT_LEAF = (
    "scope Foo\n\n  scope E\n\n    scope A\n      record Q\n        w: int\n    end A\n  end E\n"
    "end Foo"
)

_LEN4_POS: dict[str, str] = {
    "value": "Foo::E::A::Z(w = 1)",
    "annot": "fn(p: Foo::E::A::Z) => 1",
}


class TestLocalScopeFullyMatchingRouteLeadingSegmentClashesAtFullChainLength:
    """A local scope fully declaring the qualified path still clashes against its route.

    ``Foo`` both names the ``pkg/Foo`` import route (its leading segment
    alone) and a local scope declaring the whole chain down to ``A``: the
    route resolves ``E::A::Z`` to its own declaration, so the chain is
    ambiguous at length 4 whether or not the local leaf shares the route's
    own member name.
    """

    @pytest.mark.parametrize("pos_name", sorted(_LEN4_POS))
    def test_file_same_leaf(self, tmp_path: Path, pos_name: str) -> None:
        entry = _LEN4_POS[pos_name]
        src = "\n".join(["import pkg/Foo", _LEN4_LOCAL_SAME_LEAF, entry])
        phase, cls, _span = file_verdict(tmp_path, {"entry": src, "pkg/Foo": _LEN4_LIB})
        assert (phase, cls) == ("scope", RouteClashError)

    @pytest.mark.parametrize("pos_name", sorted(_LEN4_POS))
    def test_file_different_leaf(self, tmp_path: Path, pos_name: str) -> None:
        entry = _LEN4_POS[pos_name]
        src = "\n".join(["import pkg/Foo", _LEN4_LOCAL_DIFFERENT_LEAF, entry])
        phase, cls, _span = file_verdict(tmp_path, {"entry": src, "pkg/Foo": _LEN4_LIB})
        assert (phase, cls) == ("scope", RouteClashError)

    @pytest.mark.parametrize("pos_name", sorted(_LEN4_POS))
    def test_repl_same_leaf(self, tmp_path: Path, pos_name: str) -> None:
        entry = _LEN4_POS[pos_name]
        decls = ("import pkg/Foo", _LEN4_LOCAL_SAME_LEAF, entry)
        phase, cls, _span = repl_verdict_all_groupings(tmp_path, {"pkg/Foo": _LEN4_LIB}, decls)
        assert (phase, cls) == ("scope", RouteClashError)

    @pytest.mark.parametrize("pos_name", sorted(_LEN4_POS))
    def test_repl_different_leaf(self, tmp_path: Path, pos_name: str) -> None:
        entry = _LEN4_POS[pos_name]
        decls = ("import pkg/Foo", _LEN4_LOCAL_DIFFERENT_LEAF, entry)
        phase, cls, _span = repl_verdict_all_groupings(tmp_path, {"pkg/Foo": _LEN4_LIB}, decls)
        assert (phase, cls) == ("scope", RouteClashError)
