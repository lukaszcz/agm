"""Shared REPL-grouping and qualifier-verdict helpers.

A qualifier verdict (:data:`Verdict`) is a 4-tuple ``(phase, cls, span,
identity)``:

``phase``
    FILE mode (:func:`graph_verdict`, :func:`file_verdict`) is
    ``"scope"``/``"typecheck"``/``"matchcompile"``/``"accepted"``, classified
    by which real pipeline call raised -- never by the raised exception's
    class alone, since more than one phase can raise the same class. REPL
    mode (:func:`assert_verdicts_for_grouping`,
    :func:`repl_matrix_verdict_for_grouping`) has no phase-by-phase call of
    its own -- :class:`~agm.agl.repl.entry.EntryResult`
    carries one ``check_only`` outcome -- so its own phase is only
    ``"accepted"``/``"rejected"``, or ``"type-entry"`` when an entry that is
    no value is accepted as a type spelling alone; a REPL verdict's *class*
    and *span* are asserted to match a fresh file-mode verdict for the
    identical case instead, which implies the same phase.
``cls``/``span``
    The raised exception's class and span, or ``(NoneType, None)`` when
    accepted.
``identity``
    Set only when accepted: the rendered static type of the entry's final
    expression (FILE mode: the checked entry module's own node types; REPL
    mode: ``EntryResult.value_type``, rendered the same way) -- proof that an
    owner is instantiated from scope's recorded key, not re-resolved by name.

The REPL helpers run one caller-chosen way to group a declaration sequence
into REPL entries (:func:`all_groupings`, :func:`legal_groupings`): every
entry but a grouping's last runs through
:meth:`~agm.agl.repl.session.ReplSession.eval_entry` (accumulating session
state exactly as production does); the last runs through
``eval_entry(check_only=True)`` -- the same real entry path a production REPL
uses to classify one entry without lowering, evaluating, or promoting it, so
a rejection's structured cause is ``EntryResult.failure``. A caller covers
every candidate grouping by parametrizing its own test over
:func:`grouping_params` -- one pytest instance per grouping, so a batch's
full grouping coverage is spread across many cheap tests rather than summed
into one -- and each instance asserts its own legality against the expected
legal set: ``"ALL"`` when every grouping tried is legal, otherwise the
literal set.

:func:`_grouping_probe_outcomes` is the one per-grouping engine underlying
:func:`repl_matrix_verdict_for_grouping`: a *header* declaration sequence
plus a batch of *probes* (a probe is never built from *header* alone) all
sharing that header's own setup. ``sizes[:-1]`` groups *header*'s own
leading items into that many setup entries, decided in one session; whether
they all succeed *is* that grouping's own legality, and only when legal does
the same session go on to answer every *probes* value as its own
``check_only`` final entry against *header*'s own unconsumed tail (never
promoting state). Both the matrix test and :func:`assert_verdicts_for_grouping`
call :func:`repl_matrix_verdict_for_grouping` directly for one caller-chosen
full grouping over ``len(header) + 1`` items (the ``+ 1`` standing in for
whichever probe follows) -- the one shared engine, so a batch of probes
sharing one header is checked in one session per grouping, never one
session per probe per grouping.

:func:`assert_verdicts_for_grouping` is the one assertion helper, for one
grouping at a time: for every probe in a batch sharing one *header*, it
asserts that *sizes*'s own setup is legal exactly when *sizes* is in the
expected legal set, and -- only when legal -- computes each probe's own file
verdict (may be recomputed per call; it is cheap) and asserts it, and this
grouping's own REPL verdict, against the caller's expected ``(phase, cls)``
-- including, when accepted, that file mode's and this grouping's own
*identity* are equal (and, when the caller supplies one, equal to an
expected literal too) and, when the caller supplies expected origins, that
file mode's and this grouping's own ambiguity origins (:func:`origin_kinds`)
equal them, each listed once. :func:`assert_verdict_for_grouping` is its
single-probe special case, kept for callers with only one probe to check
against a header. :func:`grouping_params` builds one ``pytest.param(sizes)``
per candidate grouping over a header, so a caller parametrizes rather than
loops.

:func:`all_groupings`, :func:`eval_setup_entries`, :func:`eval_grouped_final`
and :func:`legal_groupings` are the general-purpose entry-grouping helpers
shared with ``tests/test_agl_repl_session.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import pytest

from agm.agl.diagnostics import AglError
from agm.agl.matchcompile import compile_program_matches, match_issue_error
from agm.agl.modules.ids import ENTRY_ID, spell_declaration
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.repl.type_display import format_type_for_repl
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AmbiguousQualificationError, to_bare_path
from agm.agl.syntax.nodes import Block, FuncDef
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files

if TYPE_CHECKING:
    from agm.agl.modules.loader import ModuleGraph
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.types import Type
    from agm.agl.typecheck.env import CheckedModule

FilePhase = Literal["scope", "typecheck", "matchcompile", "accepted"]
ReplPhase = Literal["accepted", "type-entry", "rejected"]
FileVerdict = tuple[FilePhase, type[BaseException] | type[None], SourceSpan | None, str | None]
ReplVerdict = tuple[ReplPhase, type[BaseException] | type[None], SourceSpan | None, str | None]
Verdict = FileVerdict | ReplVerdict
LegalGroupings = frozenset[tuple[int, ...]] | Literal["ALL"]

K = TypeVar("K")


def all_groupings(n: int) -> tuple[tuple[int, ...], ...]:
    """Every way to split *n* declarations, in order, into one or more entries."""
    if n == 0:
        return ((),)
    return tuple((first, *rest) for first in range(1, n + 1) for rest in all_groupings(n - first))


def eval_setup_entries(
    session: ReplSession, decls: tuple[str, ...], sizes: tuple[int, ...]
) -> bool:
    """Evaluate one entry per *sizes* against *session*, in order; ``True`` iff every one succeeds.

    Stops at the first failing entry. The one entry-grouping walk shared by
    every caller that only needs a legal/illegal verdict for a grouping's
    setup entries -- :func:`legal_groupings` and :func:`eval_grouped_final`
    build on it instead of repeating the walk.
    """
    start = 0
    for size in sizes:
        if not session.eval_entry("\n".join(decls[start : start + size])).ok:
            return False
        start += size
    return True


def eval_grouped_final(
    session: ReplSession, decls: tuple[str, ...], sizes: tuple[int, ...]
) -> EntryResult:
    """Evaluate *decls* as one entry per *sizes*; every entry but the last must succeed.

    Returns the last entry's result unchecked, for a caller that expects it
    to fail depending on how the grouping combines declarations into
    entries. *sizes* is presumed already known legal (see
    :func:`legal_groupings`); a non-final entry that fails to set up anyway
    is a caller bug, not a case this helper classifies.
    """
    start = sum(sizes[:-1])
    if not eval_setup_entries(session, decls, sizes[:-1]):
        raise AssertionError(f"a non-final entry failed to set up: sizes={sizes}")
    return session.eval_entry("\n".join(decls[start:]))


def legal_groupings(
    groupings: tuple[tuple[int, ...], ...],
    make_session: Callable[[], ReplSession],
    is_legal: Callable[[ReplSession, tuple[int, ...]], bool],
) -> frozenset[tuple[int, ...]]:
    """Every grouping in *groupings* whose fresh session *is_legal* accepts.

    *make_session* returns one fresh, already-open session per grouping (a
    factory that reuses one fixed, already-written directory across every
    call is fine, since each call still returns its own new session).
    *is_legal* runs the grouping's own entries against that session --
    typically via :func:`eval_setup_entries` -- and decides whether the
    grouping counts as legal; it may also record anything else it needs from
    that session as a side effect before returning. Used by
    ``tests/test_agl_repl_session.py``'s single legal-grouping check (whose
    own *is_legal* also gates on a probe entry's own success).
    """
    return frozenset(sizes for sizes in groupings if is_legal(make_session(), sizes))


def graph_verdict(graph: "ModuleGraph") -> FileVerdict:
    """Resolve, type-check, and match-compile *graph*; report which phase, if any, raised.

    Classifies by which call raised, not by the exception's class: scope
    resolution can itself raise ``AglTypeError`` for a type-name-as-value
    mistake caught while resolving, so only a genuine typecheck-phase
    rejection -- one the (successfully resolved) second call raises -- is
    reported as ``"typecheck"``. On acceptance, *identity* is the rendered
    static type of the entry module's own final item.
    """
    return _graph_outcome(graph)[0]


def _graph_outcome(graph: "ModuleGraph") -> tuple[FileVerdict, AglError | None]:
    """:func:`graph_verdict`'s verdict paired with the raised error itself, if any."""
    try:
        resolved = resolve_program(graph)
    except AglError as exc:
        return ("scope", type(exc), exc.span, None), exc
    try:
        checked_program = check_program(resolved, base_caps())
    except AglError as exc:
        return ("typecheck", type(exc), exc.span, None), exc
    match_result = compile_program_matches(checked_program)
    if match_result.compiled is None:
        match_error = match_issue_error(match_result.issues[0])
        return ("matchcompile", type(match_error), match_error.span, None), match_error
    entry = checked_program.modules[checked_program.entry_id]
    identity = _rendered_identity(_entry_final_type(entry), entry.type_env.type_table)
    return ("accepted", type(None), None, identity), None


def _rendered_identity(value_type: "Type | None", type_table: "TypeTable | None") -> str | None:
    """Render *value_type* the way the REPL echoes it, or ``None`` when there is none."""
    if value_type is None:
        return None
    return format_type_for_repl(value_type, type_table)


def _entry_final_type(entry: "CheckedModule") -> "Type | None":
    """Static type of *entry*'s final source item.

    An inline command's items sit inside the host's synthetic entry
    function, whose body is always a ``Block``, not directly at module top
    level; a real, source-declared ``program def`` needs no such unwrapping.
    """
    items = entry.resolved.program.body.items
    if items and isinstance(items[-1], FuncDef) and items[-1].is_synthetic:
        body = items[-1].body
        items = body.items if isinstance(body, Block) else ()
    if not items:
        return None
    return entry.node_types.get(items[-1].node_id)


def file_verdict(tmp_path: Path, modules: dict[str, str], *, stdlib: bool = True) -> FileVerdict:
    """Build *modules* as files into one real graph and classify it via :func:`graph_verdict`."""
    graph = make_graph_from_files(tmp_path, modules, default_stdlib=stdlib)
    return graph_verdict(graph)


def _check_only_outcome(session: ReplSession, text: str) -> tuple[ReplVerdict, AglError | None]:
    """Evaluate *text* as a ``check_only`` entry against *session*; report its verdict and error.

    The real entry path (see module docstring): every ``check_only``
    rejection is a static one (parse/scope/typecheck/match-compile), so
    ``EntryResult.failure`` is always set on rejection.
    """
    result = session.eval_entry(text, check_only=True)
    if result.ok:
        identity = _rendered_identity(result.value_type, result.type_table)
        phase: ReplPhase = "type-entry" if result.kind == "type" else "accepted"
        return (phase, type(None), None, identity), None
    failure = result.failure
    if failure is None:
        raise AssertionError("a check_only rejection always carries its static failure")
    return ("rejected", type(failure), failure.span, None), failure


def _write_modules(session_dir: Path, modules: dict[str, str]) -> None:
    """Write every module except ``entry`` under *session_dir*, one file per module."""
    for name, source in modules.items():
        if name == "entry":
            continue
        path = session_dir / f"{name}.agl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _grouping_probe_outcomes(
    session: ReplSession, header: tuple[str, ...], sizes: tuple[int, ...], probes: Mapping[K, str]
) -> dict[K, tuple[ReplVerdict, AglError | None]] | None:
    """Every *probes* value's own outcome for *sizes* against *session*, or ``None`` if illegal.

    The one per-grouping engine (see module docstring): ``sizes[:-1]``
    groups *header*'s own leading items into that many setup entries via
    :func:`eval_setup_entries`; only when they all succeed does the same
    session go on to answer every *probes* value as its own ``check_only``
    final entry against *header*'s own unconsumed tail.
    """
    if not eval_setup_entries(session, header, sizes[:-1]):
        return None
    tail = header[sum(sizes[:-1]) :]
    return {
        key: _check_only_outcome(session, "\n".join((*tail, probe)))
        for key, probe in probes.items()
    }


def repl_matrix_verdict_for_grouping(
    tmp_path: Path,
    modules: dict[str, str],
    header: tuple[str, ...],
    sizes: tuple[int, ...],
    probes: Mapping[K, str],
    *,
    stdlib: bool = True,
) -> tuple[bool, dict[K, ReplVerdict]]:
    """Decide *sizes*'s own legality and, when legal, every *probes* verdict, in one session.

    *sizes* is a full grouping over ``len(header) + 1`` items (the ``+ 1``
    standing in for whichever probe follows). Returns ``(False, {})`` when
    illegal.
    """
    is_legal, outcomes = _repl_outcomes_for_grouping(
        tmp_path, modules, header, sizes, probes, stdlib=stdlib
    )
    return is_legal, {key: verdict for key, (verdict, _failure) in outcomes.items()}


def _repl_outcomes_for_grouping(
    tmp_path: Path,
    modules: dict[str, str],
    header: tuple[str, ...],
    sizes: tuple[int, ...],
    probes: Mapping[K, str],
    *,
    stdlib: bool,
) -> tuple[bool, dict[K, tuple[ReplVerdict, AglError | None]]]:
    """:func:`repl_matrix_verdict_for_grouping` with each probe's raised error kept too."""
    _write_modules(tmp_path, modules)
    session = ReplSession(cwd=tmp_path, default_stdlib=stdlib)
    session.open()
    outcomes = _grouping_probe_outcomes(session, header, sizes, probes)
    return outcomes is not None, outcomes or {}


def origin_kinds(error: AglError | None) -> frozenset[tuple[type, str]]:
    """*error*'s ambiguity origins as ``(origin class, declaration spelling)`` pairs.

    A declaration is spelled with its module's label unless it lives in the
    entry module (see :func:`~agm.agl.modules.ids.spell_declaration`), so a
    file-mode and a REPL origin of the same candidate compare equal.
    """
    if not isinstance(error, AmbiguousQualificationError):
        return frozenset()
    return frozenset(
        (
            type(origin),
            spell_declaration(
                origin.declaration[0], to_bare_path(origin.declaration[1]), local_to=ENTRY_ID
            ),
        )
        for origin in error.origins
    )


def origins_listed_once(error: AglError | None) -> bool:
    """Whether *error* lists each of its ambiguity origins once."""
    return not isinstance(error, AmbiguousQualificationError) or len(set(error.origins)) == len(
        error.origins
    )


def ambiguity_spelling(error: AglError | None) -> str | None:
    """*error*'s ambiguous spelling as written, or ``None`` for any other error."""
    return error.spelling if isinstance(error, AmbiguousQualificationError) else None


def grouping_cases(n: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Every grouping over *n* items, paired with a dotted id, for pytest parametrization."""
    return tuple((".".join(map(str, sizes)), sizes) for sizes in all_groupings(n))


def grouping_params(n: int) -> list[object]:
    """``pytest.param(sizes)`` per grouping over *n* items, id'd by its own dotted sizes."""
    return [pytest.param(sizes, id=gid) for gid, sizes in grouping_cases(n)]


def assert_verdicts_for_grouping(
    tmp_path: Path,
    modules: dict[str, str],
    header: tuple[str, ...],
    sizes: tuple[int, ...],
    probes: Mapping[K, str],
    expected: Mapping[K, tuple[FilePhase, type[BaseException] | type[None]]],
    *,
    span_texts: Mapping[K, str] | None = None,
    expected_identities: Mapping[K, str] | None = None,
    expected_origins: Mapping[K, frozenset[tuple[type, str]]] | None = None,
    expected_spellings: Mapping[K, str] | None = None,
    type_entries: Mapping[K, str] | None = None,
    stdlib: bool = True,
    expected_legal_groupings: LegalGroupings = "ALL",
) -> None:
    """Assert *sizes*'s own legality, and (when legal) every *probes* verdict, against file mode.

    One grouping at a time (see module docstring); a caller covers every
    candidate by parametrizing its own test over :func:`grouping_params`.
    *expected* is keyed like *probes*, a file-mode ``(phase, cls)`` pair;
    REPL's own phase is normalized to ``"accepted"``/``"rejected"`` before
    comparison. For a probe whose ``expected[key][0] == "accepted"``,
    *span_texts* must have no entry for *key* and both modes' own *identity*
    must be set and equal to each other -- proof file mode and this grouping
    resolve to the identical declaration, not merely that neither raised --
    and, when *expected_identities* has an entry for *key*, equal to it too;
    otherwise *span_texts[key]* is the exact text the raised error's span
    must slice out, in both modes, both modes raise the same message, and,
    when *expected_origins* has an entry for *key*, both modes' ambiguity
    origins (:func:`origin_kinds`) equal it, each listed once, and when *expected_spellings*
    has one, both modes' ambiguity ``spelling`` equals it. *type_entries* maps a probe file
    mode rejects as a value to the type its spelling names: alone in this
    grouping's final entry, it is instead a REPL ``"type-entry"`` of that
    rendered *identity*.
    """
    span_texts = span_texts or {}
    expected_identities = expected_identities or {}
    expected_origins = expected_origins or {}
    expected_spellings = expected_spellings or {}
    type_entries = type_entries or {}
    expected_legal = (
        frozenset(all_groupings(len(header) + 1))
        if expected_legal_groupings == "ALL"
        else expected_legal_groupings
    )
    session_dir = tmp_path / "repl"
    session_dir.mkdir()
    is_legal, outcomes = _repl_outcomes_for_grouping(
        session_dir, modules, header, sizes, probes, stdlib=stdlib
    )
    assert is_legal == (sizes in expected_legal), sizes
    if not is_legal:
        return
    tail = header[sum(sizes[:-1]) :]
    for key, probe in probes.items():
        expected_phase, expected_cls = expected[key]
        src = "\n".join((*header, probe))
        file_outcome, file_failure = _graph_outcome(
            make_graph_from_files(tmp_path, {"entry": src, **modules}, default_stdlib=stdlib)
        )
        file_phase, file_cls, file_span, file_identity = file_outcome
        assert (file_phase, file_cls) == (expected_phase, expected_cls), key
        if expected_phase == "accepted":
            assert key not in span_texts, key
            assert file_identity is not None, key
            if key in expected_identities:
                assert file_identity == expected_identities[key], key
        else:
            assert file_span is not None, key
            assert src[file_span.start_offset : file_span.end_offset] == span_texts[key], key
        (repl_phase, repl_cls, repl_span, repl_identity), repl_failure = outcomes[key]
        if key in type_entries and not tail:
            assert (repl_phase, repl_identity) == ("type-entry", type_entries[key]), key
            continue
        assert repl_phase == ("accepted" if expected_phase == "accepted" else "rejected"), key
        assert repl_cls == expected_cls, key
        if expected_phase == "accepted":
            assert repl_identity == file_identity, key
        else:
            text = "\n".join((*tail, probe))
            sliced = (
                text[repl_span.start_offset : repl_span.end_offset]
                if repl_span is not None
                else None
            )
            assert sliced == span_texts[key], key
            assert str(repl_failure) == str(file_failure), key
            if key in expected_origins:
                for failure in (file_failure, repl_failure):
                    assert origin_kinds(failure) == expected_origins[key], key
                    assert origins_listed_once(failure), key
            if key in expected_spellings:
                assert ambiguity_spelling(file_failure) == expected_spellings[key], key
                assert ambiguity_spelling(repl_failure) == expected_spellings[key], key


def assert_verdict_for_grouping(
    tmp_path: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    sizes: tuple[int, ...],
    expected: tuple[FilePhase, type[BaseException] | type[None]],
    *,
    span_text: str | None = None,
    expected_identity: str | None = None,
    expected_origins: frozenset[tuple[type, str]] | None = None,
    stdlib: bool = True,
    expected_legal_groupings: LegalGroupings = "ALL",
) -> None:
    """Single-probe special case of :func:`assert_verdicts_for_grouping`.

    *decls*'s last item is the probe, and every earlier item is the shared
    header.
    """
    assert_verdicts_for_grouping(
        tmp_path,
        modules,
        decls[:-1],
        sizes,
        {"_": decls[-1]},
        {"_": expected},
        span_texts=None if span_text is None else {"_": span_text},
        expected_identities=None if expected_identity is None else {"_": expected_identity},
        expected_origins=None if expected_origins is None else {"_": expected_origins},
        stdlib=stdlib,
        expected_legal_groupings=expected_legal_groupings,
    )


@dataclass(frozen=True)
class Probe:
    """One probe's source and expected file-mode verdict, for a :class:`Scenario` table.

    *detail* is the rendered identity when accepted, else the exact text the
    error's span slices out; *type_entry* is the identity the REPL renders
    when the probe alone is read as a type spelling (see
    :func:`assert_verdicts_for_grouping`).
    """

    text: str
    phase: FilePhase
    error: type[AglError] | None
    detail: str
    type_entry: str | None = None


def accepted(text: str, identity: str) -> Probe:
    """A probe file mode accepts with rendered *identity*."""
    return Probe(text, "accepted", None, identity)


def rejected(
    text: str,
    error: type[AglError],
    span: str,
    *,
    phase: FilePhase = "scope",
    type_entry: str | None = None,
) -> Probe:
    """A probe file mode rejects in *phase* with *error* spanning *span*."""
    return Probe(text, phase, error, span, type_entry)


@dataclass(frozen=True)
class Scenario:
    """Modules, a shared header and its probes, with the header's legal REPL groupings."""

    header: tuple[str, ...]
    probes: Mapping[str, Probe]
    modules: Mapping[str, str] = field(default_factory=dict)
    legal: LegalGroupings = "ALL"


_PROBES_PER_CASE = 12
"""Probes one scenario test case checks: each probe costs a file build and a REPL entry."""


def scenario_params(scenarios: Mapping[str, Scenario]) -> list[object]:
    """``pytest.param(scenario, sizes)`` per scenario, grouping and chunk of its probes.

    Every chunk replays the scenario's header in its own session, so a large
    probe table spreads over several cheap test cases instead of one costly one.
    """
    params: list[object] = []
    for name, scenario in scenarios.items():
        keys = list(scenario.probes)
        chunks = [keys[i : i + _PROBES_PER_CASE] for i in range(0, len(keys), _PROBES_PER_CASE)]
        for index, chunk in enumerate(chunks):
            part = replace(scenario, probes={key: scenario.probes[key] for key in chunk})
            suffix = "" if len(chunks) == 1 else f"-part{index + 1}"
            params.extend(
                pytest.param(part, sizes, id=f"{name}{suffix}-{gid}")
                for gid, sizes in grouping_cases(len(scenario.header) + 1)
            )
    return params


def assert_scenario_for_grouping(
    tmp_path: Path, scenario: Scenario, sizes: tuple[int, ...]
) -> None:
    """:func:`assert_verdicts_for_grouping` for every probe of *scenario*."""
    probes = scenario.probes
    assert_verdicts_for_grouping(
        tmp_path,
        dict(scenario.modules),
        scenario.header,
        sizes,
        {key: probe.text for key, probe in probes.items()},
        {
            key: (probe.phase, type(None) if probe.error is None else probe.error)
            for key, probe in probes.items()
        },
        span_texts={key: p.detail for key, p in probes.items() if p.error is not None},
        expected_identities={key: p.detail for key, p in probes.items() if p.error is None},
        type_entries={key: p.type_entry for key, p in probes.items() if p.type_entry is not None},
        expected_legal_groupings=scenario.legal,
    )
