"""Position-matrix regression tests for scope's qualifier decisions.

Scope, not typecheck, decides which declaration a qualified type name
selects and whether a qualifier selects none at all, identically across the
value, pattern, ``is``, cast, annotation, alias, type-argument and applied
positions, and identically in file mode and the REPL (independent of how a
REPL session groups its declarations into entries). ``assert_verdicts_for_grouping``/
``assert_verdict_for_grouping`` (see :mod:`tests.agl.qualifier_support`) are
the one assertion helpers behind both this file's hand-written cases and
(via ``_assert_matrix_verdict``) its own generated matrix, each checking one
REPL grouping per test instance -- every hand-written case parametrizes over
:func:`~tests.agl.qualifier_support.grouping_params` exactly as the matrix
already does, so a batch's full grouping coverage is spread across many
cheap tests rather than summed into one. Every accepted case's *identity* is
asserted against its exact expected rendering (``_accepted_identity``), not
merely non-``None`` -- proof scope's resolution reached real, checked node
types naming the declaration it actually picked, and (since these helpers
compare them) that file mode and this grouping agree on it exactly.

``_matrix_params`` (file mode) and ``_matrix_cases`` (REPL mode, feeding
:func:`~tests.agl.qualifier_support.repl_matrix_verdict_for_grouping`'s
session-sharing) reuse one probe program per outcome across five owner
spellings (a module
route, a wildcard-imported bare name, a wildcard-imported generic owner
applied to a type argument -- proving the owner is instantiated from
scope's recorded key rather than re-resolved by name, with a member that
mentions ``T`` so its accepted identity proves the substitution too -- a
locally ``use``d scope region, and a module-level ``use`` alias) and seven
syntactic positions; the eighth position (an owner itself carrying type
arguments, ``Owner[T]::Member``) is exercised by the generic-owner
spelling, whose member is *always* written applied. The ``pattern``
position uses an as-pattern so its own accepted identity is the bound
value's real type, not a literal's. The ``is`` position's own accepted
identity is always ``bool``, the only identity an ``is`` test itself can
bear; ``cast`` probes the definitionally equivalent ``v as? {q}`` (``x is
T`` holds exactly when ``x as? T`` is ``Some``) separately, since ``is``
goes through scope's own is-test constructor-candidate path while ``as?``
goes through the cast's type position -- different code that must still
agree, so both are checked -- and its own accepted identity is ``Option``
of the resolved declaration instead.

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

A qualifier chain typed alone at the REPL prompt reaches the type-entry
fallback (``_try_type_entry``) only when it names a type, never a value: a
member/constructor spelling is already a legal value expression on its own
and never reaches that fallback, so ``TestBareTypeEntryFallback`` below
probes it with type-only spellings (a module-qualified alias member, an
applied generic owner, a nested type-only path) instead of reusing this
file's own value-position matrix, and asserts ``kind == "type"`` directly
through :class:`~agm.agl.repl.session.ReplSession` rather than through the
``Verdict`` machinery, which never distinguishes a type entry from an
ordinary accepted value entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import ReferencedMemberError
from agm.agl.repl import ReplSession
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousQualificationError,
    RouteClashError,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.semantics.values import TextValue
from agm.agl.typecheck import AglTypeError
from tests.agl.ir_harness import evaluate_ir_graph
from tests.agl.qualifier_support import (
    FilePhase,
    LegalGroupings,
    Verdict,
    all_groupings,
    assert_verdict_for_grouping,
    assert_verdicts_for_grouping,
    file_verdict,
    grouping_cases,
    grouping_params,
    repl_matrix_verdict_for_grouping,
)

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
    "gn": "enum Box[T]\n  | Loaded(value: T)\n  | Both\n  | Other\n",
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
        {"unique": "Full", "other": "Other", "amb": "Both", "missing": "NoSuch"},
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
    "pattern": "case v of\n  | {q} as bound => bound\n  | _ => v",
    "is": "v is {q}",
    "cast": "v as? {q}",
    "annot": "fn(x: {q}) => 1",
    "alias": "type CC = {q}\nfn(x: CC) => 1",
    "tyarg": "fn(x: array[{q}]) => 1",
}

# (phase, class) expected for each (form, outcome, position); phase
# "accepted" leaves the class column unused. Matches the implementation's
# actual, verified behavior (tests/CLAUDE.md: assert behavior, not
# internals): "other" is a real member of a structurally
# distinct nominal type, so it is a genuine typecheck mismatch wherever the
# position actually checks the referenced value/pattern against ``v``'s type
# (pattern, ``is``, ``cast``) and otherwise just a legal type reference; "amb" is
# always scope's ambiguity verdict; "missing" is scope's selects-none
# verdict everywhere: the same-level ambiguity of an owner's own spelling --
# whether a bare unqualified name or a routed one (``route::Owner``) -- is
# decided full-path-first, so an owner ambiguous on its own spelling still
# selects "missing" once the requested member disambiguates it in no
# candidate.
_ACCEPTED: tuple[FilePhase, type[BaseException] | type[None]] = ("accepted", type(None))
_OTHER_MISMATCH: tuple[FilePhase, type[BaseException] | type[None]] = ("typecheck", AglTypeError)
_AMBIGUOUS: tuple[FilePhase, type[BaseException] | type[None]] = (
    "scope",
    AmbiguousQualificationError,
)
_MISSING: tuple[FilePhase, type[BaseException] | type[None]] = ("scope", UnknownMemberError)

# The canonical, fully qualified case type each (form, outcome) resolves to --
# the base every accepted position's own identity derives from (see
# ``_accepted_identity``). Spelled out directly (tests/CLAUDE.md: assert
# behavior, not internals) rather than derived from ``_FORMS``, since which
# real module or scope ends up owning each case is exactly what scope's own
# resolution decides.
_CASE_TYPE: dict[tuple[str, str], str] = {
    ("route", "unique"): "one/types::Color::Green",
    ("route", "other"): "two/types::Color::Blue",
    ("bare", "unique"): "m::Color::Green",
    ("bare", "other"): "n::Color::Blue",
    ("applied", "unique"): "gm::Box::Full[int]",
    ("applied", "other"): "gn::Box::Other",
    ("localuse", "unique"): "a::E::Y",
    ("localuse", "other"): "b::E::Z",
    ("moduse", "unique"): "m::Color::Green",
    ("moduse", "other"): "n::Color::Blue",
}

# ``value``'s own identity: a nullary case's bare reference is a value of the
# case's own record type, but "applied"'s "unique" case (``Full``) carries a
# payload, so its bare reference is the constructor function instead -- the
# one case the ``record {case_type}`` template cannot derive uniformly.
_VALUE_IDENTITY_OVERRIDE: dict[tuple[str, str], str] = {
    ("applied", "unique"): "int -> gm::Box::Full[int]",
}

# ``pattern``'s own identity: the as-pattern's bound value is always the whole
# matched *owner* enum (proof scope picked the right declaration among
# same-named candidates), never the specific case alone -- only "unique" is
# ever accepted at this position (see the ``_EXPECTED`` loop below).
_ENUM_IDENTITY: dict[str, str] = {
    "route": "enum one/types::Color\n  | Red\n  | Green",
    "bare": "enum m::Color\n  | Red\n  | Green",
    "applied": "enum gm::Box[int]\n  | Full(value: int)\n  | Both\n  | Empty",
    "localuse": "enum a::E\n  | X\n  | Y",
    "moduse": "enum m::Color\n  | Red\n  | Green",
}


def _accepted_identity(form_name: str, outcome: str, pos_name: str) -> str:
    """The expected identity for one accepted ``(form, outcome, pos)`` matrix case.

    ``annot``/``alias`` reference the case's own canonical type
    (``_CASE_TYPE``) as a function parameter, ``tyarg`` wraps it in
    ``array[...]``, and ``cast`` (``v as? {q}``) reports the ``Option`` it
    casts to -- all four derive uniformly from the template. ``is`` is
    always ``bool``, the only identity an ``is`` test itself can bear.
    ``value`` and ``pattern`` are spelled out (``_VALUE_IDENTITY_OVERRIDE``/
    ``_ENUM_IDENTITY``) since their own rendering depends on whether the case
    carries a payload, or shows the whole owner enum, not just the template.
    """
    case_type = _CASE_TYPE[(form_name, outcome)]
    if pos_name == "value":
        return _VALUE_IDENTITY_OVERRIDE.get((form_name, outcome), f"record {case_type}")
    if pos_name == "pattern":
        return _ENUM_IDENTITY[form_name]
    if pos_name == "is":
        return "bool"
    if pos_name == "cast":
        return f"enum std/option::Option[{case_type}]\n  | None\n  | Some(value: {case_type})"
    if pos_name == "tyarg":
        return f"array[{case_type}] -> int"
    return f"{case_type} -> int"  # annot, alias


_ExpectedCase = tuple[str, type[BaseException] | type[None], str | None]

_EXPECTED: dict[tuple[str, str, str], _ExpectedCase] = {}
for _form in _FORMS:
    for _pos in _POS:
        _EXPECTED[(_form, "unique", _pos)] = (*_ACCEPTED, _accepted_identity(_form, "unique", _pos))
        _other_accepted = _pos not in ("pattern", "is", "cast")
        _EXPECTED[(_form, "other", _pos)] = (
            (*_ACCEPTED, _accepted_identity(_form, "other", _pos))
            if _other_accepted
            else (*_OTHER_MISMATCH, None)
        )
        _EXPECTED[(_form, "amb", _pos)] = (*_AMBIGUOUS, None)
        _EXPECTED[(_form, "missing", _pos)] = (*_MISSING, None)
del _form, _pos, _other_accepted

# The span of a raised error covers exactly the qualifier chain's own text
# (``owner::member``), except an ``is``/``cast`` mismatch error, whose span
# covers the whole ``subject is/as? owner::member`` expression.
_MISMATCH_SPAN_PREFIX: dict[str, str] = {"is": "v is ", "cast": "v as? "}


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


def _prefix_required_legal_groupings(total: int, prefix: int) -> frozenset[tuple[int, ...]]:
    """Legal groupings of *total* items whose first entry must span at least *prefix* items.

    Matches a header whose leading ``use`` declarations forward-reference
    ``scope`` regions declared later in the same header: a REPL entry
    resolves its own items together, so every legal grouping's first entry
    must include that whole leading run; the remaining items impose no
    further constraint and may split any way.
    """
    return frozenset(
        (first, *rest)
        for first in range(prefix, total + 1)
        for rest in all_groupings(total - first)
    )


_LOCALUSE_HEADER_LEN = len(_FORMS["localuse"][1])
# The header's first four items (two ``use``s, then the ``scope``s they name)
# must share one entry; the fifth (the ``let``) is unconstrained. Legality is
# over the header plus one merged final (tail + probe) entry, so the total
# item count is the header's own length plus one.
_EXPECTED_LEGAL_GROUPINGS: dict[str, LegalGroupings] = {
    "localuse": _prefix_required_legal_groupings(_LOCALUSE_HEADER_LEN + 1, 4),
}


def _expected_legal_full_groupings(form_name: str) -> frozenset[tuple[int, ...]]:
    """*form_name*'s own legal-grouping set, over its header plus one merged final entry."""
    _, header, _, _ = _FORMS[form_name]
    expected = _EXPECTED_LEGAL_GROUPINGS.get(form_name, "ALL")
    return frozenset(all_groupings(len(header) + 1)) if expected == "ALL" else expected


def _matrix_cases(form_name: str) -> list[tuple[str, str]]:
    """Every ``(outcome, pos_name)`` pair this form's REPL matrix probes."""
    _, _, _, outcomes = _FORMS[form_name]
    return [(outcome, pos_name) for outcome in outcomes for pos_name in _POS]


def _assert_matrix_verdict(
    source: str,
    verdict: Verdict,
    form_name: str,
    outcome: str,
    pos_name: str,
    q: str,
    *,
    is_file: bool,
) -> None:
    """Assert *verdict* matches ``_EXPECTED``.

    FILE mode's ``phase`` distinguishes ``"scope"``/``"typecheck"``; a REPL
    entry's single ``check_only`` outcome cannot, so its own phase is only
    ``"accepted"``/``"rejected"`` -- asserted class and span still pin down
    which phase would have raised, since ``file_verdict`` computed for the
    identical case must have raised the same class (see :mod:`qualifier_support`).
    """
    phase, cls, span, identity = verdict
    expected_phase, expected_cls, expected_identity = _EXPECTED[(form_name, outcome, pos_name)]
    if is_file:
        assert (phase, cls) == (expected_phase, expected_cls)
    else:
        assert phase == ("accepted" if expected_phase == "accepted" else "rejected")
        assert cls == expected_cls
    if expected_phase == "accepted":
        # An accepted entry's echoed type is always the real, checked type of
        # its final item, matched exactly -- proof scope's resolution reached
        # real, checked node types naming the declaration it actually picked,
        # not just "did not raise".
        assert identity == expected_identity
        return
    assert span is not None
    text = source[span.start_offset : span.end_offset]
    prefix = _MISMATCH_SPAN_PREFIX.get(pos_name)
    expected_text = prefix + q if prefix is not None and outcome == "other" else q
    assert text == expected_text


@pytest.mark.parametrize(("form_name", "outcome", "pos_name"), _matrix_params())
def test_qualifier_decision_matrix_file(
    tmp_path: Path, form_name: str, outcome: str, pos_name: str
) -> None:
    """Scope's unique/other/ambiguous/missing verdict is the same in every position."""
    modules, header, entry, q = _matrix_case(form_name, outcome, pos_name)
    src = "\n".join([*header, entry])
    verdict = file_verdict(tmp_path, {"entry": src, **modules})
    _assert_matrix_verdict(src, verdict, form_name, outcome, pos_name, q, is_file=True)


def _matrix_repl_params() -> list[object]:
    return [
        pytest.param(form_name, sizes, id=f"{form_name}-{gid}")
        for form_name in sorted(_FORMS)
        for gid, sizes in grouping_cases(len(_FORMS[form_name][1]) + 1)
    ]


@pytest.mark.parametrize(("form_name", "sizes"), _matrix_repl_params())
def test_qualifier_decision_matrix_repl(
    tmp_path: Path, form_name: str, sizes: tuple[int, ...]
) -> None:
    """*sizes*'s own legality matches its expected set; a legal grouping also matches file mode.

    One REPL session, opened once against this test's own directory,
    replays *sizes*'s own setup prefix (``sizes[:-1]``, grouping this form's
    header) and asserts that succeeding is exactly equivalent to *sizes*
    being one of ``_expected_legal_full_groupings(form_name)``'s own
    members (see :func:`~tests.agl.qualifier_support.repl_matrix_verdict_for_grouping`)
    -- covering every candidate grouping, legal or not, with no separate
    legality-only pass. Only when legal does the same session go on to
    answer every outcome/position probe as its own final entry, checked
    against the file-mode verdict exactly as before; a ``check_only`` probe
    never promotes session state, so this is a pure reparametrization of
    trying every probe against this one grouping separately.
    """
    modules, header, own, outcomes = _FORMS[form_name]
    cases = _matrix_cases(form_name)
    probes = {
        (outcome, pos_name): _POS[pos_name].format(q=f"{own}::{outcomes[outcome]}")
        for outcome, pos_name in cases
    }
    is_legal, verdicts = repl_matrix_verdict_for_grouping(
        tmp_path, modules, tuple(header), sizes, probes
    )
    assert is_legal == (sizes in _expected_legal_full_groupings(form_name))
    if not is_legal:
        return
    tail = header[sum(sizes[:-1]) :]
    for outcome, pos_name in cases:
        q = f"{own}::{outcomes[outcome]}"
        # A grouping's own final entry is the header's own unconsumed tail
        # plus the probe, joined exactly as `repl_matrix_verdict_for_grouping`
        # joined it -- the returned span is relative to that whole text, not
        # the probe alone, whenever the tail is non-empty.
        final_text = "\n".join((*tail, probes[(outcome, pos_name)]))
        _assert_matrix_verdict(
            final_text,
            verdicts[(outcome, pos_name)],
            form_name,
            outcome,
            pos_name,
            q,
            is_file=False,
        )


class TestBareTypeEntryFallback:
    """A qualified type-only spelling, typed alone, reaches the REPL's type-entry fallback.

    Unlike a member/constructor spelling (already a legal value expression
    on its own, so it never reaches ``_try_type_entry``), each probe here
    names a real type and nothing else, so scope's own owner/member
    resolution is what the fallback's synthetic ``type <fresh> = text``
    alias exercises; rejection through the same fallback (a hidden or
    referenced member) is already covered in
    ``tests/test_agl_repl_session.py``.
    """

    def test_module_qualified_alias_member(self, tmp_path: Path) -> None:
        (tmp_path / "geo.agl").write_text("type Num = int\n", encoding="utf-8")
        session = ReplSession(cwd=tmp_path, default_stdlib=True)
        session.open()
        assert session.eval_entry("import geo").ok
        result = session.eval_entry("geo::Num", check_only=True)
        assert result.ok
        assert result.kind == "type"

    def test_applied_generic_owner(self, tmp_path: Path) -> None:
        (tmp_path / "gm.agl").write_text(_GB["gm"], encoding="utf-8")
        session = ReplSession(cwd=tmp_path, default_stdlib=True)
        session.open()
        assert session.eval_entry("import gm").ok
        result = session.eval_entry("gm::Box[int]", check_only=True)
        assert result.ok
        assert result.kind == "type"

    def test_nested_type_only_path(self, tmp_path: Path) -> None:
        session = ReplSession(cwd=tmp_path, default_stdlib=True)
        session.open()
        assert session.eval_entry(
            "scope Geo\n\n  scope In\n    type Num = int\n  end In\nend Geo"
        ).ok
        result = session.eval_entry("Geo::In::Num", check_only=True)
        assert result.ok
        assert result.kind == "type"


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
    "alias": "type A = {q}\nfn(x: A) => 1",
    "tyarg": "fn(p: array[{q}]) => 1",
}

_ACCEPTED_RECORD_POS: dict[str, str] = {
    "value": "{q}(x = 1)",
    "pattern": "let v: {q} = {q}(x = 1)\ncase v of\n  | {q}(x) => x",
    "annot": "fn(p: {q}) => 1",
    "alias": "type A = {q}\nfn(x: A) => 1",
    "tyarg": "fn(p: array[{q}]) => 1",
}


_SHADOW_HEADER = ("import shapes::*", _GEO_LOCAL_ONLY)
_IMPORTED_MEMBER_HEADER = ("import shapes::*",)


class TestLocalScopeShadowsSameNamedImport:
    """A local ``scope Geo`` lacking ``Point`` beats an imported ``Geo::Point``.

    A name lookup that fell through to the import would find the imported
    ``Point`` and accept every position; local-scope-wins rejects all of
    them instead, identically in every position, file mode and REPL
    grouping.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_SHADOW_HEADER) + 1))
    def test_rejected(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {pos: _RECORD_POS[pos].format(q="Geo::Point") for pos in _RECORD_POS}
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _GEO_IMPORTED},
            _SHADOW_HEADER,
            sizes,
            probes,
            {pos: ("scope", UnknownMemberError) for pos in probes},
            span_texts={pos: "Geo::Point" for pos in probes},
        )


class TestImportedScopeRegionMemberIsAccepted:
    """A member uniquely selected through a scope-region prefix is accepted.

    Scope's recording of a uniquely selected qualified type name when the
    prefix is a scope region (rather than a type owner) is what lets the
    type positions (which have no fallback left to lean on) accept this at
    all.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_IMPORTED_MEMBER_HEADER) + 1))
    def test_accepted(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {
            pos: _ACCEPTED_RECORD_POS[pos].format(q="Geo::Point") for pos in _ACCEPTED_RECORD_POS
        }
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _GEO_IMPORTED},
            _IMPORTED_MEMBER_HEADER,
            sizes,
            probes,
            {pos: _ACCEPTED for pos in probes},
        )


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

    A bare type-name lookup asks what the module's own declarations select
    first, exactly as the value position (``Point(z = 1)``) already does, so
    the local record wins the same way in every position.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_TWO_IMPORT_HEADER) + 1))
    def test_accepted(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {
            "annot": "fn(p: Point) => 1",
            "tyarg": "fn(p: array[Point]) => 1",
            "alias": "type A = Point\n1",
            "value": "Point(z = 1)",
        }
        assert_verdicts_for_grouping(
            tmp_path,
            _TWO_IMPORT_LIB,
            _TWO_IMPORT_HEADER,
            sizes,
            probes,
            {pos: _ACCEPTED for pos in probes},
        )


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

    A route clash is checked over the whole segment run a longer route
    names, not just the chain's first segment. The unanchored spelling
    clashes with the ``pkg/Foo`` route (which resolves ``E::A`` past it);
    the anchored spelling, naming only the current module, has no route to
    clash with
    and rejects ``A`` as a plain unknown member of the local nullary case.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_ENUM_CASE_CLASH_HEADER) + 1))
    def test_unanchored(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        decls = (*_ENUM_CASE_CLASH_HEADER, "Foo::E::A")
        assert_verdict_for_grouping(
            tmp_path,
            _ENUM_CASE_CLASH_LIB,
            decls,
            sizes,
            ("scope", RouteClashError),
            span_text="Foo::E::A",
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_ENUM_CASE_CLASH_HEADER) + 1))
    def test_anchored(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        decls = (*_ENUM_CASE_CLASH_HEADER, "::Foo::E::A")
        assert_verdict_for_grouping(
            tmp_path,
            _ENUM_CASE_CLASH_LIB,
            decls,
            sizes,
            ("scope", UnknownMemberError),
            span_text="::Foo::E::A",
        )


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
_LOCAL_WINS_NESTED_LIB = (
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
_LOCAL_WINS_NESTED_LOCAL = "scope Geo\n  record Point\n    y: int\n  enum Shape\n    | Tri\nend Geo"
_LOCAL_WINS_NESTED_HEADER = ("import shapes::*", _LOCAL_WINS_NESTED_LOCAL)


class TestLocalTypeWinsOverAmbiguousImportWithoutMerging:
    """A local type sharing an imported region's path resolves to itself alone.

    A direct local hit on ``Geo::Shape`` (or ``Geo::Num``) wins outright
    rather than merging with the imported region's own same-path
    declaration into an ambiguity, exactly as a plain member-set local
    scope already does.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_LOCAL_WINS_HEADER) + 1))
    def test_accepted(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {
            "annot-shape": "fn(p: Geo::Shape) => 1",
            "let-tri": "let s: Geo::Shape = Geo::Shape::Tri\ns",
            "annot-num": "fn(p: Geo::Num) => 1",
            "let-num": 'let n: Geo::Num = "a"\nn',
        }
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _LOCAL_WINS_LIB},
            _LOCAL_WINS_HEADER,
            sizes,
            probes,
            {pos: _ACCEPTED for pos in probes},
        )

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

    @pytest.mark.parametrize("sizes", grouping_params(len(_LOCAL_WINS_NESTED_HEADER) + 1))
    def test_local_scope_region_type_wins_over_imported_region(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        """A local scope-region ``Geo::Shape`` also wins when the local scope has ``Point``.

        Distinct from the module-level cases above: here the local ``Geo``
        also declares its own ``Point``, ruling out any fallback reading of
        ``Geo`` as a plain member-set scope with nothing of its own.
        """
        decls = (*_LOCAL_WINS_NESTED_HEADER, "fn(p: Geo::Shape) => 1")
        assert_verdict_for_grouping(
            tmp_path, {"shapes": _LOCAL_WINS_NESTED_LIB}, decls, sizes, _ACCEPTED
        )


# ---------------------------------------------------------------------------
# The owner is instantiated from scope's recorded key.
# ---------------------------------------------------------------------------

_NESTED_GENERIC_ALIAS_HEADER = (
    "enum Slot[T]\n  | Filled(value: T)\n  | Empty",
    "type Rows[A] = Slot[array[A]]",
    "let row: Slot[array[int]] = Slot::Filled(value = [1, 2])",
)


@pytest.mark.parametrize("sizes", grouping_params(len(_NESTED_GENERIC_ALIAS_HEADER) + 1))
def test_nested_generic_alias_owner_is_instantiated_by_substitution(
    tmp_path: Path, sizes: tuple[int, ...]
) -> None:
    """An alias target nested inside another generic type substitutes recursively.

    ``Rows[A]``'s own template is ``Slot[array[A]]`` -- an alias target
    that itself wraps another generic type, not a bare type parameter -- so
    ``Rows[int]::Filled`` instantiates to ``Slot::Filled[array[int]]``, not
    ``Slot::Filled[int]``.
    """
    decls = (
        *_NESTED_GENERIC_ALIAS_HEADER,
        "case row of\n  | Rows[int]::Filled(value) => value.size() + 1\n  | Slot::Empty => 0",
    )
    # The case expression's own arms both return ``int``, so the substitution
    # proof is behavioral, not visible in this identity directly: had
    # ``Rows[int]::Filled`` instantiated to ``Slot::Filled[int]`` instead, the
    # pattern's own bound ``value`` would be ``int``, and ``.size()`` (an
    # array-only method) would fail to typecheck rather than silently
    # accepting -- so acceptance itself is the proof; the identity is
    # asserted exactly anyway, per this module's own no-discard rule.
    assert_verdict_for_grouping(tmp_path, {}, decls, sizes, _ACCEPTED, expected_identity="int")


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

    def test_rejected(self, tmp_path: Path) -> None:
        # No header: the only grouping is one entry holding the probe alone.
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            (),
            (1,),
            _UNKNOWN_ROUTE_POS,
            {pos: ("scope", UnknownQualifierError) for pos in _UNKNOWN_ROUTE_POS},
            span_texts={pos: "missing::Item" for pos in _UNKNOWN_ROUTE_POS},
        )


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


_ENUM_SHADOW_HEADER = ("import lib::*", _ENUM_LOCAL)


class TestLocalEnumShadowsImportedEnum:
    """A local ``enum Shape`` lacking ``Circle`` beats an imported ``Shape::Circle``."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_ENUM_SHADOW_HEADER) + 1))
    def test_rejected(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {pos: _ENUM_POS[pos].format(q="Shape::Circle") for pos in _ENUM_POS}
        assert_verdicts_for_grouping(
            tmp_path,
            {"lib": _ENUM_LIB},
            _ENUM_SHADOW_HEADER,
            sizes,
            probes,
            {pos: ("scope", UnknownMemberError) for pos in probes},
            span_texts={pos: "Shape::Circle" for pos in probes},
        )


# ---------------------------------------------------------------------------
# ``def Owner::method`` whose leading segment ALSO resolves, below the local
# level, to an imported TYPE owner: the def-created path reads as that
# owner's method namespace, not a plain local one -- its nested types stay
# reachable and a real miss still rejects, distinct from the plain-local-
# namespace case below where the leading segment names only a scope region.
# ---------------------------------------------------------------------------

_DEFPATH_TYPE_OWNER_LIB = "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"
_DEFPATH_TYPE_OWNER_DEF = "def Geo::m(self) -> int = self.x"
_DEFPATH_TYPE_OWNER_HEADER = ("import shapes::*", _DEFPATH_TYPE_OWNER_DEF)


class TestDefCreatedPathOverImportedTypeReadsAsItsMethodNamespace:
    """A def-created path over an imported type keeps the type's own nested members.

    ``Geo::Inner`` constructs through the def-created path over the
    imported type ``Geo``, and ``Geo::Nope`` rejects with the ordinary
    unknown-member verdict, rather than either spelling treating ``Geo`` as
    only a method-owner namespace with no nested members of its own.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH_TYPE_OWNER_HEADER) + 1))
    def test_nested_type_is_accepted(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {"annot": "fn(p: Geo::Inner) => 1", "value": "Geo::Inner(y = 1)"}
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _DEFPATH_TYPE_OWNER_LIB},
            _DEFPATH_TYPE_OWNER_HEADER,
            sizes,
            probes,
            {pos: _ACCEPTED for pos in probes},
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH_TYPE_OWNER_HEADER) + 1))
    def test_missing_member_is_rejected(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {"annot": "fn(p: Geo::Nope) => 1", "value": "Geo::Nope(y = 1)"}
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _DEFPATH_TYPE_OWNER_LIB},
            _DEFPATH_TYPE_OWNER_HEADER,
            sizes,
            probes,
            {pos: ("scope", UnknownMemberError) for pos in probes},
            span_texts={pos: "Geo::Nope" for pos in probes},
        )


class TestCurrentModuleAnchoredDefCreatedPathIsAnOwnRootMiss:
    """A ``::``-anchored qualifier over a def-created path still walks it exactly.

    A ``::`` anchor still checks whether its leading segment resolves
    locally (as a def-created method-owner namespace) before reporting an
    unknown qualifier, so both ``::Geo::Nope`` (missing on the imported
    owner too) and ``::Geo::Inner`` (present on the imported owner, but not
    on this module's own root, which is all a ``::`` anchor ever names)
    report the local path's own missing member, spanned on the whole
    qualifier.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH_TYPE_OWNER_HEADER) + 1))
    def test_rejected(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        shapes = {"annot": "fn(p: ::Geo::{}) => 1", "value": "::Geo::{}(y = 1)"}
        members = ["Inner", "Nope"]
        probes = {
            (shape_name, member): shape.format(member)
            for shape_name, shape in shapes.items()
            for member in members
        }
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _DEFPATH_TYPE_OWNER_LIB},
            _DEFPATH_TYPE_OWNER_HEADER,
            sizes,
            probes,
            {key: ("scope", UnknownMemberError) for key in probes},
            span_texts={key: f"::Geo::{key[1]}" for key in probes},
        )


_REFERENCED_MEMBER_HEADER = ("record Saved\n  id: int", "enum Stored = ::Saved | Fresh")


class TestReferencedMemberWalkedOneSegmentPastNeverNamesAnUnknownQualifier:
    """A referenced enum member, walked one segment past, is a member miss.

    The leading lookup for the owner ``Stored::Saved`` recognizes it as a
    referenced enum member, so ``Stored::Saved::X`` reports the same
    referenced-member verdict a bare ``Stored::Saved`` already gets, rather
    than treating ``Stored`` as an unresolved module route.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_REFERENCED_MEMBER_HEADER) + 1))
    def test_rejected(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        probes = {"annot": "fn(p: Stored::Saved::X) => 1", "value": "Stored::Saved::X"}
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _REFERENCED_MEMBER_HEADER,
            sizes,
            probes,
            {pos: ("scope", ReferencedMemberError) for pos in probes},
            span_texts={pos: "Stored::Saved::X" for pos in probes},
        )


# ---------------------------------------------------------------------------
# A nearer ``use``-opened scope region blocks a farther, same-spelled
# imported type from ever merging into the leading lookup -- the one lookup
# shared by a bare type name, a qualifier's leading segment, and a
# ``def Owner::method`` receiver.
# ---------------------------------------------------------------------------

_NEAREST_REGION_LIB = "scope Geo\n  record Point\n    x: int\nend Geo\n"
_NEAREST_IMPORTED_TYPE_LIB = "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"
_NEAREST_MODULES = {"shapes": _NEAREST_REGION_LIB, "tl": _NEAREST_IMPORTED_TYPE_LIB}
_NEAREST_HEADER = ("import tl::*", "import shapes", "use shapes::*")


class TestNearestUseRegionBlocksFartherImportedTypeAsQualifierLeadingSegment:
    """A qualifier's own leading segment: the nearer ``use`` region always decides.

    ``tl::Geo`` (an imported type) and ``shapes::Geo`` (a locally ``use``d
    scope region) share a spelling; only the region -- the nearer
    contribution -- ever decides ``Geo``'s own leading reading, so a value
    reference to ``Geo::f()`` (naming no member of either) is a member the
    region lacks, exactly as for a region this module declares itself --
    never merged with the farther type (contrast
    ``test_control_without_the_nearer_region``, whose identical spelling,
    absent the ``use``, resolves to the farther type instead).
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_NEAREST_HEADER) + 1))
    def test_farther_type_member_is_not_reached(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        assert_verdict_for_grouping(
            tmp_path,
            _NEAREST_MODULES,
            (*_NEAREST_HEADER, "Geo::f()"),
            sizes,
            ("scope", UnknownMemberError),
            span_text="Geo::f",
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(("import tl::*",)) + 1))
    def test_control_without_the_nearer_region(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        assert_verdict_for_grouping(
            tmp_path,
            {"tl": _NEAREST_IMPORTED_TYPE_LIB},
            ("import tl::*", "Geo::f()"),
            sizes,
            ("scope", UnknownMemberError),
            span_text="Geo::f",
        )

    @pytest.mark.parametrize(
        "sizes", grouping_params(len((*_NEAREST_HEADER, "def Geo::f() -> int = 1")) + 1)
    )
    def test_a_competing_def_created_path_still_resolves_locally(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        assert_verdict_for_grouping(
            tmp_path,
            _NEAREST_MODULES,
            (*_NEAREST_HEADER, "def Geo::f() -> int = 1", "Geo::f()"),
            sizes,
            _ACCEPTED,
            expected_identity="int",
        )


_RECEIVER_NEAREST_MODULES = {"shapes": _NEAREST_REGION_LIB, "tl": "record Geo\n  x: int\n"}
_RECEIVER_NEAREST_HEADER = ("import tl::*", "import shapes", "use shapes::*")


class TestNearestUseRegionBlocksFartherImportedTypeAsReceiverOwner:
    """A ``def Owner::method`` receiver's own owner: the same lookup, the same answer.

    ``self``'s type is the receiver's resolved owner; once the nearer
    ``use``-opened region decides ``Geo`` (exactly as the qualifier leading
    segment above does), a region is never a type, so ``self`` itself
    cannot be typed and the receiver rejects -- the same scope-phase verdict
    the qualifier leading segment reaches for the identical shadowing.
    Absent the region (``test_control_without_the_nearer_region``), the
    farther type decides instead and the method type-checks normally.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_RECEIVER_NEAREST_HEADER) + 1))
    def test_receiver_owner_is_not_a_type(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdict_for_grouping(
            tmp_path,
            _RECEIVER_NEAREST_MODULES,
            (*_RECEIVER_NEAREST_HEADER, "def Geo::f(self) -> int = 1"),
            sizes,
            ("scope", AglScopeError),
            span_text="self",
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(("import tl::*", "import shapes")) + 1))
    def test_control_without_the_nearer_region(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        assert_verdict_for_grouping(
            tmp_path,
            _RECEIVER_NEAREST_MODULES,
            (
                "import tl::*",
                "import shapes",
                "def Geo::f(self) -> int = self.x\nlet g = Geo(x = 1)\ng.f()",
            ),
            sizes,
            _ACCEPTED,
            expected_identity="int",
        )


_BARE_NEAREST_MODULES = {
    "m": "record X\n  a: int\n",
    "n": "record X\n  b: int\n",
    "lib": "scope X\n  record Y\n    z: int\nend X\n",
}
_BARE_NEAREST_HEADER = ("import m::*", "import n::*", "import lib", "use lib::*")


class TestNearestUseRegionBlocksAmbiguousImportsAsBareTypeName:
    """A bare (unqualified) type name: the same lookup avoids the same ambiguity.

    ``X`` is ambiguous between two wildcard-imported records (``m::X``,
    ``n::X``) at the farther, IMPORTED layer; a nearer ``use``-opened scope
    region of the identical spelling (``lib::X``) decides it outright
    instead, so -- unlike the plain, genuinely ambiguous control below -- the
    verdict is never ambiguity. A region is never itself a type, so scope
    rejects the annotation as naming a region.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_BARE_NEAREST_HEADER) + 1))
    def test_nearer_region_names_no_type(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdict_for_grouping(
            tmp_path,
            _BARE_NEAREST_MODULES,
            (*_BARE_NEAREST_HEADER, "fn(p: X) => 1"),
            sizes,
            ("scope", AglScopeError),
            span_text="X",
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(("import m::*", "import n::*")) + 1))
    def test_control_without_the_nearer_region_is_ambiguous(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        assert_verdict_for_grouping(
            tmp_path,
            _BARE_NEAREST_MODULES,
            ("import m::*", "import n::*", "fn(p: X) => 1"),
            sizes,
            ("scope", AmbiguousQualificationError),
            span_text="X",
        )


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
_DEFPATH_NO_IMPORT_HEADER = (_DEFPATH_DEF,)
_DEFPATH_WITH_IMPORT_HEADER = ("import shapes::*", _DEFPATH_DEF)

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

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH_NO_IMPORT_HEADER) + 1))
    def test_without_import(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            _DEFPATH_NO_IMPORT_HEADER,
            sizes,
            _DEFPATH_MISSING_POS,
            {pos: ("scope", UnknownMemberError) for pos in _DEFPATH_MISSING_POS},
            span_texts={pos: "Geo::Point" for pos in _DEFPATH_MISSING_POS},
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH_WITH_IMPORT_HEADER) + 1))
    def test_with_imported_region(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _DEFPATH_LIB},
            _DEFPATH_WITH_IMPORT_HEADER,
            sizes,
            _DEFPATH_MISSING_POS,
            {pos: ("scope", UnknownMemberError) for pos in _DEFPATH_MISSING_POS},
            span_texts={pos: "Geo::Point" for pos in _DEFPATH_MISSING_POS},
        )


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
_DEFPATH3_HEADER_OUTER = ("import shapes::*", "def Geo::f() -> int = 1")
_DEFPATH3_HEADER_INNER = ("import shapes::*", "def Geo::In::f() -> int = 1")


class TestDefCreatedPathIsPlainLocalNamespaceAtLengthThree:
    """The plain-local-namespace verdict holds for a 3-segment chain too.

    The value position and the annotation position must reject with the
    same, ordinary unknown-member verdict, identically to the 2-segment case
    above, whether the def path is declared at the outer scope (``Geo``) or
    one level deeper (``Geo::In``).
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH3_HEADER_OUTER) + 1))
    def test_import_is_not_merged_at_length_three(
        self, tmp_path: Path, sizes: tuple[int, ...]
    ) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _DEFPATH3_LIB},
            _DEFPATH3_HEADER_OUTER,
            sizes,
            _DEFPATH3_POS,
            {pos: ("scope", UnknownMemberError) for pos in _DEFPATH3_POS},
            span_texts={pos: "Geo::In::Point" for pos in _DEFPATH3_POS},
        )

    @pytest.mark.parametrize("pos_name", sorted(_DEFPATH3_MISSING_POS))
    def test_file_missing_member_at_length_three(self, tmp_path: Path, pos_name: str) -> None:
        entry = _DEFPATH3_MISSING_POS[pos_name]
        src = "\n".join(["import shapes::*", "def Geo::f() -> int = 1", entry])
        phase, cls, span, _identity = file_verdict(
            tmp_path, {"entry": src, "shapes": _DEFPATH3_LIB}
        )
        assert (phase, cls) == ("scope", UnknownMemberError)
        assert span is not None
        assert src[span.start_offset : span.end_offset] == "Geo::In::Nope"

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH3_HEADER_INNER) + 1))
    def test_def_path_nested_one_level_deeper(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _DEFPATH3_LIB},
            _DEFPATH3_HEADER_INNER,
            sizes,
            _DEFPATH3_POS,
            {pos: ("scope", UnknownMemberError) for pos in _DEFPATH3_POS},
            span_texts={pos: "Geo::In::Point" for pos in _DEFPATH3_POS},
        )


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
_DEFPATH_EMPTY_PREFIX_HEADER = ("import shapes::*", "def Geo::In::f() -> int = 1")


class TestDefCreatedPathWithEmptyOwnPrefixIsPlainLocalNamespace:
    """A def-created path's non-owning leading prefix still rejects a sibling miss."""

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH_EMPTY_PREFIX_HEADER) + 1))
    def test_annotation(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        decls = (*_DEFPATH_EMPTY_PREFIX_HEADER, "fn(x: Geo::Other) => 1")
        assert_verdict_for_grouping(
            tmp_path,
            {"shapes": _DEFPATH_EMPTY_PREFIX_LIB},
            decls,
            sizes,
            ("scope", UnknownMemberError),
            span_text="Geo::Other",
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_DEFPATH_EMPTY_PREFIX_HEADER) + 1))
    def test_is(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        decls = (*_DEFPATH_EMPTY_PREFIX_HEADER, "1 is Geo::Other")
        assert_verdict_for_grouping(
            tmp_path,
            {"shapes": _DEFPATH_EMPTY_PREFIX_LIB},
            decls,
            sizes,
            ("scope", UnknownMemberError),
            span_text="Geo::Other",
        )


# ---------------------------------------------------------------------------
# A local scope's own reading, once decisive, is walked exactly regardless of
# the qualifier chain's length: a 2-segment miss (the owner's own missing
# member) and a 3-segment miss (a member nested one level deeper, inside an
# owner nested in the same local scope) both reject with the identical
# verdict the local owner's member set alone decides.
# ---------------------------------------------------------------------------

# ``Geo`` declares a member at every chain length: ``Geo::Point`` (2),
# ``Geo::Shape::Circle`` (3) and ``Geo::Deep::Kind::Round`` (4). Declared
# locally, every one is accepted; imported beside a local ``Geo`` region
# declaring only ``Deep`` (empty of ``Kind``), every one is an unknown member
# of that region -- at length 4 one level past a walked hit.
_CHAIN_GEO = (
    "scope Geo\n  record Point\n    x: int\n  enum Shape = Circle | Square\n"
    "\n  scope Deep\n    enum Kind = Round | Flat\n  end Deep\nend Geo"
)
_CHAIN_LOCAL_DEEP = "scope Geo\n\n  scope Deep\n    enum Other = O\n  end Deep\nend Geo"
_CHAIN_NESTED_LOCAL = (
    "scope Geo\n\n  scope Shape\n    enum Kind = Round | Flat\n  end Shape\nend Geo"
)
# Each length's spelling and, for a member, its enum.
_CHAIN_SPELLINGS: dict[int, tuple[str, str | None]] = {
    2: ("Geo::Point", None),
    3: ("Geo::Shape::Circle", "Geo::Shape"),
    4: ("Geo::Deep::Kind::Round", "Geo::Deep::Kind"),
}


def _chain_probes(length: int, *, declared: bool) -> dict[str, str]:
    """Every position's probe of the length-*length* spelling, well typed when *declared*."""
    q, owner = _CHAIN_SPELLINGS[length]
    value = f"{q}(x = 1)" if owner is None else q
    probes = {
        "value": value,
        "annot": f"fn(p: {q}) => 1",
        "alias": f"type AA = {q}\nfn(p: AA) => 1",
        "tyarg": f"fn(p: array[{q}]) => 1",
    }
    if owner is None:
        probes["pattern"] = (
            f"case {value} of\n  | {q}(x) => x"
            if declared
            else f"case 1 of\n  | {q}(x) => x\n  | _ => 2"
        )
    elif declared:
        probes["pattern"] = f"let v: {owner} = {q}\ncase v of\n  | {q} => 1\n  | _ => 2"
        probes["is"] = f"let v: {owner} = {q}\nv is {q}"
    else:
        probes["pattern"] = f"case 1 of\n  | {q} => 1\n  | _ => 2"
        probes["is"] = f"1 is {q}"
    return probes


def _chain_identities(length: int) -> dict[str, str]:
    """Each accepted length-*length* probe's identity."""
    q, owner = _CHAIN_SPELLINGS[length]
    return {
        "value": f"record {q}\n  x: int" if owner is None else f"record {q}",
        "annot": f"{q} -> int",
        "alias": f"{q} -> int",
        "tyarg": f"array[{q}] -> int",
        "pattern": "int",
        "is": "bool",
    }


class TestLocalScopeChainAtEveryLength:
    """A local scope's chain is walked exactly at lengths 2 to 4, in every position."""

    @pytest.mark.parametrize("length", sorted(_CHAIN_SPELLINGS))
    @pytest.mark.parametrize("sizes", grouping_params(3))
    def test_missing_member(self, tmp_path: Path, sizes: tuple[int, ...], length: int) -> None:
        probes = _chain_probes(length, declared=False)
        assert_verdicts_for_grouping(
            tmp_path,
            {"shapes": _CHAIN_GEO},
            ("import shapes::*", _CHAIN_LOCAL_DEEP),
            sizes,
            probes,
            {pos: ("scope", UnknownMemberError) for pos in probes},
            span_texts=dict.fromkeys(probes, _CHAIN_SPELLINGS[length][0]),
        )

    @pytest.mark.parametrize("length", sorted(_CHAIN_SPELLINGS))
    @pytest.mark.parametrize("sizes", grouping_params(2))
    def test_declared_member(self, tmp_path: Path, sizes: tuple[int, ...], length: int) -> None:
        probes = _chain_probes(length, declared=True)
        identities = _chain_identities(length)
        assert_verdicts_for_grouping(
            tmp_path,
            {},
            (_CHAIN_GEO,),
            sizes,
            probes,
            {pos: _ACCEPTED for pos in probes},
            expected_identities={pos: identities[pos] for pos in probes},
        )


_ROUTE_AND_PARTIAL_LOCAL_MISS_LIB = {"Geo": "let x = 1\n"}
_ROUTE_AND_PARTIAL_LOCAL_MISS_HEADER = ("import Geo", _CHAIN_NESTED_LOCAL)


@pytest.mark.parametrize("sizes", grouping_params(len(_ROUTE_AND_PARTIAL_LOCAL_MISS_HEADER) + 1))
def test_local_scope_partial_miss_clashes_with_a_same_named_route(
    tmp_path: Path, sizes: tuple[int, ...]
) -> None:
    """A partial local-scope miss still clashes with a same-named import route.

    ``Geo::Shape::Missing::X``'s walk matches ``Geo::Shape`` (both nested
    scope regions) but misses at ``Missing``; ``Geo`` also names a genuine
    import route (not merely a same-spelled scope), so the missing step
    defers to the route-clash check instead of rejecting directly -- and a
    plain scope region never declares ``Missing`` itself, so any such route
    at all is a clash.
    """
    decls = (*_ROUTE_AND_PARTIAL_LOCAL_MISS_HEADER, "Geo::Shape::Missing::X")
    assert_verdict_for_grouping(
        tmp_path,
        _ROUTE_AND_PARTIAL_LOCAL_MISS_LIB,
        decls,
        sizes,
        ("scope", RouteClashError),
        span_text="Geo::Shape::Missing::X",
    )


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


_LEN4_HEADER_SAME_LEAF = ("import pkg/Foo", _LEN4_LOCAL_SAME_LEAF)
_LEN4_HEADER_DIFFERENT_LEAF = ("import pkg/Foo", _LEN4_LOCAL_DIFFERENT_LEAF)


class TestLocalScopeFullyMatchingRouteLeadingSegmentClashesAtFullChainLength:
    """A local scope fully declaring the qualified path still clashes against its route.

    ``Foo`` both names the ``pkg/Foo`` import route (its leading segment
    alone) and a local scope declaring the whole chain down to ``A``: the
    route resolves ``E::A::Z`` to its own declaration, so the chain is
    ambiguous at length 4 whether or not the local leaf shares the route's
    own member name.
    """

    @pytest.mark.parametrize("sizes", grouping_params(len(_LEN4_HEADER_SAME_LEAF) + 1))
    def test_same_leaf(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {"pkg/Foo": _LEN4_LIB},
            _LEN4_HEADER_SAME_LEAF,
            sizes,
            _LEN4_POS,
            {pos: ("scope", RouteClashError) for pos in _LEN4_POS},
            span_texts={pos: "Foo::E::A::Z" for pos in _LEN4_POS},
        )

    @pytest.mark.parametrize("sizes", grouping_params(len(_LEN4_HEADER_DIFFERENT_LEAF) + 1))
    def test_different_leaf(self, tmp_path: Path, sizes: tuple[int, ...]) -> None:
        assert_verdicts_for_grouping(
            tmp_path,
            {"pkg/Foo": _LEN4_LIB},
            _LEN4_HEADER_DIFFERENT_LEAF,
            sizes,
            _LEN4_POS,
            {pos: ("scope", RouteClashError) for pos in _LEN4_POS},
            span_texts={pos: "Foo::E::A::Z" for pos in _LEN4_POS},
        )
