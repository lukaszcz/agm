"""Shared REPL-grouping and qualifier-verdict helpers.

A qualifier verdict (:data:`Verdict`) is a 4-tuple ``(phase, cls, span,
identity)``:

``phase``
    FILE mode (:func:`graph_verdict`, :func:`file_verdict`) is
    ``"scope"``/``"typecheck"``/``"matchcompile"``/``"accepted"``, classified
    by which real pipeline call raised -- never by the raised exception's
    class alone, since more than one phase can raise the same class. A REPL
    entry has no phase-by-phase call of its own --
    :class:`~agm.agl.repl.entry.EntryResult` carries one ``check_only``
    outcome -- so its own phase is only ``"accepted"``/``"rejected"``, or
    ``"type-entry"`` when an entry that is no value is accepted as a type
    spelling alone; a REPL verdict's *class* and *span* are asserted to match
    the file-mode verdict for the identical text instead, which implies the
    same phase.
``cls``/``span``
    The raised exception's class and span, or ``(NoneType, None)`` when
    accepted.
``identity``
    Set only when accepted: the rendered static type of the entry's final
    expression (FILE mode: the checked entry module's own node types; REPL
    mode: ``EntryResult.value_type``, rendered the same way) -- proof of
    which declaration was selected, not merely that nothing raised.

A case is a *header* declaration sequence plus a batch of :class:`Probe`\\ s
sharing it. :func:`assert_verdicts` checks every probe once in file mode
(the header and the probe as one inline entry) and then in every way to
group the header into REPL entries (:func:`all_groupings` over
``len(header) + 1`` items, the ``+ 1`` standing for whichever probe
follows): per grouping, one session evaluates ``sizes[:-1]`` setup entries
through :meth:`~agm.agl.repl.session.ReplSession.eval_entry`; whether they
all succeed *is* the grouping's legality, asserted against the expected
legal set. Only a legal grouping goes on to answer every probe as its own
``eval_entry(check_only=True)`` final entry against the header's unconsumed
tail -- the real entry path, never promoting state, so a rejection's
structured cause is ``EntryResult.failure``. Both modes must reach the
probe's expected phase-and-class, span text or identity, and a rejected
probe the same message in both. An ``info`` probe has no file mode: once
the tail is promoted, ``:info`` of its name must report the expected
description, or raise the expected class spanning the expected text.

Probes come from :func:`accepted`, :func:`rejected`, :func:`info` and
:func:`info_rejected`; :func:`type_positions` and
:func:`type_positions_rejected` spell one qualifier in every type position,
and :func:`option_identity` is the identity an ``as?`` cast pins.
:func:`span_text` slices an error's span out of its source for tests that
check a rejection outside the harness.

:class:`Scenario` tables feed :func:`assert_scenario` through
:func:`scenario_params`, one test per chunk of a scenario's probes and batch
of its groupings, so each test stays cheap; a test computes file mode once
per probe, not once per grouping. A test that calls :func:`assert_verdicts`
itself batches its groupings through :func:`grouping_batches`.
:func:`assert_repl_verdicts` checks one REPL history of separate entries,
for histories that no file shares. :func:`assert_file_resolves_like_inline_entry`
compares the inline ``agm exec -c`` graph with the same text as a real
``agm exec <file>`` entry (see :func:`file_source`).

:func:`all_groupings`, :func:`eval_setup_entries`, :func:`eval_grouped_final`
and :func:`legal_groupings` are the general-purpose entry-grouping helpers
shared with ``tests/test_agl_repl_session.py``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import pytest

from agm.agl.diagnostics import AglError
from agm.agl.matchcompile import compile_program_matches, match_issue_error
from agm.agl.modules.ids import ENTRY_ID, Reader, spell_declaration
from agm.agl.modules.loader import parse_entry_module
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.repl.type_display import format_type_for_repl
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AmbiguousQualificationError, to_bare_path
from agm.agl.syntax.nodes import Block, FuncDef, Item, LetDecl, VarDecl
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_file_graph_from_files, make_graph_from_files

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
Origins = frozenset[tuple[type, str]]
Groupings = tuple[tuple[int, ...], ...]

K = TypeVar("K")
T = TypeVar("T")


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


def origin_kinds(error: AglError | None) -> Origins:
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
                origin.declaration[0], to_bare_path(origin.declaration[1]), reader=Reader(ENTRY_ID)
            ),
        )
        for origin in error.origins
    )


def _origins_listed_once(error: AglError | None) -> bool:
    """Whether *error* lists each of its ambiguity origins once."""
    return not isinstance(error, AmbiguousQualificationError) or len(set(error.origins)) == len(
        error.origins
    )


def _ambiguity_spelling(error: AglError | None) -> str | None:
    """*error*'s ambiguous spelling as written, or ``None`` for any other error."""
    return error.spelling if isinstance(error, AmbiguousQualificationError) else None


def grouping_cases(n: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Every grouping over *n* items, paired with a dotted id, for pytest parametrization."""
    return tuple((".".join(map(str, sizes)), sizes) for sizes in all_groupings(n))


def grouping_params(n: int) -> list[object]:
    """``pytest.param(sizes)`` per grouping over *n* items, id'd by its own dotted sizes."""
    return [pytest.param(sizes, id=gid) for gid, sizes in grouping_cases(n)]


@dataclass(frozen=True)
class Probe:
    """One probe's source and expected verdict.

    *detail* is the rendered identity when accepted (``None`` only for a
    caller that pins no identity, whose two modes must still agree), else
    the exact text the error's span slices out. *type_entry* is the identity
    the REPL renders when the probe alone is read as a type spelling;
    *origins* and *spelling* pin an ambiguity's structured fields. An *info*
    probe's text is a name ``:info`` describes: *detail* is the description
    when accepted.
    """

    text: str
    phase: FilePhase
    error: type[AglError] | None
    detail: str | None
    type_entry: str | None = None
    origins: Origins | None = None
    spelling: str | None = None
    info: bool = False


def accepted(text: str, identity: str) -> Probe:
    """A probe file mode accepts with rendered *identity*."""
    return Probe(text, "accepted", None, identity)


def option_identity(selected: str) -> str:
    """The rendered identity of ``v as? q`` when ``q`` selects *selected*."""
    return f"enum std/option::Option[{selected}]\n  | None\n  | Some(value: {selected})"


def rejected(
    text: str,
    error: type[AglError],
    span: str,
    *,
    phase: FilePhase = "scope",
    type_entry: str | None = None,
    origins: Origins | None = None,
    spelling: str | None = None,
) -> Probe:
    """A probe file mode rejects in *phase* with *error* spanning *span*."""
    return Probe(text, phase, error, span, type_entry, origins, spelling)


def info(name: str, description: str) -> Probe:
    """``:info name`` describing the selected declaration as *description*."""
    return Probe(name, "accepted", None, description, info=True)


def info_rejected(name: str, error: type[AglError], span: str) -> Probe:
    """``:info name`` raising scope's *error* spanning *span*."""
    return Probe(name, "scope", error, span, info=True)


_TYPE_POSITIONS: dict[str, tuple[str, str]] = {
    "annot": ("fn(p: {q}) => p", "{s} -> {s}"),
    "alias": ("type AA = {q}\nfn(p: AA) => p", "{s} -> {s}"),
    "tyarg": ("fn(p: array[{q}]) => p", "array[{s}] -> array[{s}]"),
    "cast": ("fn(p: text) => p as? {q}", "text -> std/option::Option[{s}]"),
}
"""Each type position's probe of a spelling ``{q}`` and its identity when it selects ``{s}``.

An applied spelling (``Box[int]``) is the applied-type position of the same
templates; ``is``, patterns and values take constructors, so each table
writes those for its own declaration kind.
"""


def type_positions(prefix: str, q: str, selected: str) -> dict[str, Probe]:
    """Probes of *q* in every type position, each selecting the type rendered *selected*.

    Keyed ``{prefix}-{position}``.
    """
    return {
        f"{prefix}-{position}": accepted(text.format(q=q), identity.format(s=selected))
        for position, (text, identity) in _TYPE_POSITIONS.items()
    }


def type_positions_rejected(
    prefix: str,
    q: str,
    error: type[AglError],
    span: str | None = None,
    *,
    phase: FilePhase = "scope",
) -> dict[str, Probe]:
    """Probes of *q* in every type position, each rejected with *error* spanning *span* (*q*)."""
    return {
        f"{prefix}-{position}": rejected(
            text.format(q=q), error, q if span is None else span, phase=phase
        )
        for position, (text, _) in _TYPE_POSITIONS.items()
    }


def probe_table(
    texts: Mapping[K, str],
    expected: Mapping[K, tuple[FilePhase, type[BaseException] | type[None]]],
    *,
    span_texts: Mapping[K, str] | None = None,
    identities: Mapping[K, str] | None = None,
    origins: Mapping[K, Origins] | None = None,
    spellings: Mapping[K, str] | None = None,
    type_entries: Mapping[K, str] | None = None,
) -> dict[K, Probe]:
    """One :class:`Probe` per *texts* entry, from its ``(phase, class)`` in *expected*.

    A rejected probe's *span_texts* entry is required; every other mapping
    is optional per key.
    """
    span_texts = span_texts or {}
    identities = identities or {}
    origins = origins or {}
    spellings = spellings or {}
    type_entries = type_entries or {}
    table: dict[K, Probe] = {}
    for key, text in texts.items():
        phase, cls = expected[key]
        if phase == "accepted":
            table[key] = Probe(text, phase, None, identities.get(key))
            continue
        assert issubclass(cls, AglError), key
        table[key] = rejected(
            text,
            cls,
            span_texts[key],
            phase=phase,
            type_entry=type_entries.get(key),
            origins=origins.get(key),
            spelling=spellings.get(key),
        )
    return table


def _probe_class(probe: Probe) -> type[BaseException] | type[None]:
    return type(None) if probe.error is None else probe.error


def span_text(text: str, span: SourceSpan | None) -> str | None:
    """The slice of *text* that *span* covers."""
    return None if span is None else text[span.start_offset : span.end_offset]


def _check_file_probe(
    tmp_path: Path,
    modules: Mapping[str, str],
    header: tuple[str, ...],
    key: object,
    probe: Probe,
    stdlib: bool,
) -> tuple[str | None, AglError | None]:
    """Assert *probe*'s file-mode verdict; return its identity and raised error."""
    src = "\n".join((*header, probe.text))
    (phase, cls, span, identity), failure = _graph_outcome(
        make_graph_from_files(tmp_path, {"entry": src, **modules}, default_stdlib=stdlib)
    )
    assert (phase, cls) == (probe.phase, _probe_class(probe)), key
    if probe.error is None:
        assert identity is not None, key
        assert probe.detail is None or identity == probe.detail, key
    else:
        assert span_text(src, span) == probe.detail, key
        _assert_ambiguity_fields(key, failure, probe)
    return identity, failure


def _assert_ambiguity_fields(key: object, failure: AglError | None, probe: Probe) -> None:
    if probe.origins is not None:
        assert origin_kinds(failure) == probe.origins, key
        assert _origins_listed_once(failure), key
    if probe.spelling is not None:
        assert _ambiguity_spelling(failure) == probe.spelling, key


def _check_only_outcome(session: ReplSession, text: str) -> tuple[ReplVerdict, AglError | None]:
    """Evaluate *text* as a ``check_only`` entry against *session*; report its verdict and error.

    Every ``check_only`` rejection is a static one (parse/scope/typecheck/
    match-compile), so ``EntryResult.failure`` is always set on rejection.
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


def _info_outcome(session: ReplSession, name: str) -> tuple[ReplVerdict, AglError | None]:
    """``:info`` of *name* against *session*'s promoted state, as a verdict."""
    try:
        description = session.info_of(name)
    except AglError as exc:
        return ("rejected", type(exc), exc.span, None), exc
    return ("accepted", type(None), None, description), None


def _write_modules(session_dir: Path, modules: Mapping[str, str]) -> None:
    """Write every module except ``entry`` under *session_dir*, one file per module."""
    for name, source in modules.items():
        if name == "entry":
            continue
        path = session_dir / f"{name}.agl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _assert_grouping(
    tmp_path: Path,
    modules: Mapping[str, str],
    header: tuple[str, ...],
    sizes: tuple[int, ...],
    probes: Mapping[K, Probe],
    file_outcomes: Mapping[K, tuple[str | None, AglError | None]] | None,
    legal: frozenset[tuple[int, ...]],
    stdlib: bool,
) -> None:
    """Assert *sizes*'s legality and, when legal, every probe's REPL verdict (module docstring).

    Without *file_outcomes*, an accepted probe's identity is its own *detail*
    and a rejection's message is compared with nothing.
    """
    session_dir = tmp_path / ("repl-" + ".".join(map(str, sizes)))
    session_dir.mkdir()
    _write_modules(session_dir, modules)
    session = ReplSession(cwd=session_dir, default_stdlib=stdlib)
    session.open()
    is_legal = eval_setup_entries(session, header, sizes[:-1])
    assert is_legal == (sizes in legal), sizes
    if not is_legal:
        return
    tail = header[sum(sizes[:-1]) :]
    outcomes = {
        key: _check_only_outcome(session, "\n".join((*tail, probe.text)))
        for key, probe in probes.items()
        if not probe.info
    }
    if any(probe.info for probe in probes.values()):
        assert not tail or session.eval_entry("\n".join(tail)).ok, sizes
        outcomes.update(
            {key: _info_outcome(session, probe.text) for key, probe in probes.items() if probe.info}
        )
    for key, probe in probes.items():
        (phase, cls, span, identity), failure = outcomes[key]
        if probe.info:
            assert cls == _probe_class(probe), (key, sizes)
            detail = identity if probe.error is None else span_text(probe.text, span)
            assert detail == probe.detail, (key, sizes)
            continue
        if probe.type_entry is not None and not tail:
            assert (phase, identity) == ("type-entry", probe.type_entry), (key, sizes)
            continue
        assert phase == ("accepted" if probe.error is None else "rejected"), (key, sizes)
        assert cls == _probe_class(probe), (key, sizes)
        file_identity, file_failure = (
            (probe.detail, None) if file_outcomes is None else file_outcomes[key]
        )
        if probe.error is None:
            assert identity == file_identity, (key, sizes)
            continue
        assert span_text("\n".join((*tail, probe.text)), span) == probe.detail, (key, sizes)
        assert file_outcomes is None or str(failure) == str(file_failure), (key, sizes)
        _assert_ambiguity_fields((key, sizes), failure, probe)


def _legal_set(header: tuple[str, ...], legal: LegalGroupings) -> frozenset[tuple[int, ...]]:
    return frozenset(all_groupings(len(header) + 1)) if legal == "ALL" else legal


def _file_outcomes(
    tmp_path: Path,
    modules: Mapping[str, str],
    header: tuple[str, ...],
    probes: Mapping[K, Probe],
    stdlib: bool,
) -> dict[K, tuple[str | None, AglError | None]]:
    file_dir = tmp_path / "file"
    file_dir.mkdir()
    return {
        key: _check_file_probe(file_dir, modules, header, key, probe, stdlib)
        for key, probe in probes.items()
        if not probe.info
    }


def assert_verdicts(
    tmp_path: Path,
    modules: Mapping[str, str],
    header: tuple[str, ...],
    probes: Mapping[K, Probe],
    *,
    legal: LegalGroupings = "ALL",
    stdlib: bool = True,
    groupings: Groupings | None = None,
) -> None:
    """Assert every probe in file mode once, then in each REPL grouping of *header*.

    *legal* is the set of full groupings (over ``len(header) + 1`` items)
    whose setup entries succeed, ``"ALL"`` when every one does. *groupings*
    narrows the groupings replayed to one batch of :func:`grouping_batches`;
    ``None`` is every grouping.
    """
    groupings = groupings or all_groupings(len(header) + 1)
    assert all(sum(sizes) == len(header) + 1 for sizes in groupings), groupings
    file_outcomes = _file_outcomes(tmp_path, modules, header, probes, stdlib)
    expected_legal = _legal_set(header, legal)
    for sizes in groupings:
        _assert_grouping(
            tmp_path, modules, header, sizes, probes, file_outcomes, expected_legal, stdlib
        )


def assert_verdict(
    tmp_path: Path,
    modules: Mapping[str, str],
    decls: tuple[str, ...],
    expected: tuple[FilePhase, type[BaseException] | type[None]],
    *,
    span_text: str | None = None,
    identity: str | None = None,
    origins: Origins | None = None,
    legal: LegalGroupings = "ALL",
    stdlib: bool = True,
    groupings: Groupings | None = None,
) -> None:
    """:func:`assert_verdicts` for one probe: *decls*'s last item after the header before it."""
    assert_verdicts(
        tmp_path,
        modules,
        decls[:-1],
        probe_table(
            {"probe": decls[-1]},
            {"probe": expected},
            span_texts=None if span_text is None else {"probe": span_text},
            identities=None if identity is None else {"probe": identity},
            origins=None if origins is None else {"probe": origins},
        ),
        legal=legal,
        stdlib=stdlib,
        groupings=groupings,
    )


def assert_repl_verdicts(
    tmp_path: Path,
    modules: Mapping[str, str],
    entries: tuple[str, ...],
    probes: Mapping[K, Probe],
    *,
    stdlib: bool = True,
) -> None:
    """Assert every probe's REPL verdict after *entries*, each its own successful entry.

    For a REPL-only history that no file shares -- a later entry importing or
    redeclaring what an earlier one already decided -- so no file mode is
    compared: an accepted probe's identity is its expected *detail*.
    """
    sizes = (*(1 for _ in entries), 1)
    _assert_grouping(tmp_path, modules, entries, sizes, probes, None, frozenset({sizes}), stdlib)


@dataclass(frozen=True)
class Scenario:
    """Modules, a shared header and its probes, with the header's legal REPL groupings.

    *groupings* narrows the REPL groupings a check replays to one batch of
    them (see :func:`scenario_params`); ``None`` is every grouping.
    """

    header: tuple[str, ...]
    probes: Mapping[str, Probe]
    modules: Mapping[str, str] = field(default_factory=dict)
    legal: LegalGroupings = "ALL"
    groupings: Groupings | None = None


def assert_scenario(tmp_path: Path, scenario: Scenario) -> None:
    """:func:`assert_verdicts` for every probe of *scenario*, over its groupings."""
    assert_verdicts(
        tmp_path,
        scenario.modules,
        scenario.header,
        scenario.probes,
        legal=scenario.legal,
        groupings=scenario.groupings,
    )


_GROUPINGS_PER_CASE = 2
"""REPL groupings one test case replays at most: each opens a session and sets its header up."""

_PROBES_PER_CASE = 8
"""Probes one scenario test case checks: each in file mode, then once per grouping."""


def _batches(items: Sequence[T], size: int) -> list[tuple[T, ...]]:
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]


def _grouping_batches(n: int) -> list[Groupings]:
    return _batches(all_groupings(n), _GROUPINGS_PER_CASE)


def grouping_batches(n: int) -> list[object]:
    """``pytest.param(groupings)`` per batch of the groupings over *n* items.

    For a test that passes it as :func:`assert_verdicts`' *groupings*, so each
    of its cases replays few enough groupings to stay cheap.
    """
    return [
        pytest.param(batch, id=f"groupings{index + 1}")
        for index, batch in enumerate(_grouping_batches(n))
    ]


def _scenario_parts(
    name: str,
    scenario: Scenario,
    probes_per_part: int,
    batches: Sequence[Groupings | None],
) -> list[object]:
    """``pytest.param`` per chunk of *scenario*'s probes by each of its grouping *batches*."""
    chunks = _batches(list(scenario.probes), probes_per_part)
    params: list[object] = []
    for index, chunk in enumerate(chunks):
        probes = {key: scenario.probes[key] for key in chunk}
        for batch_index, batch in enumerate(batches):
            part_id = name if len(chunks) == 1 else f"{name}-part{index + 1}"
            if len(batches) > 1:
                part_id += f"-groupings{batch_index + 1}"
            part = replace(scenario, probes=probes, groupings=batch)
            params.append(pytest.param(part, id=part_id))
    return params


def scenario_params(scenarios: Mapping[str, Scenario]) -> list[object]:
    """``pytest.param(scenario)`` per chunk of every scenario's probes and batch of its groupings.

    For :func:`assert_scenario`: every part replays the scenario's header in
    its own sessions, few enough times to stay cheap.
    """
    params: list[object] = []
    for name, scenario in scenarios.items():
        batches = _grouping_batches(len(scenario.header) + 1)
        params += _scenario_parts(name, scenario, _PROBES_PER_CASE, batches)
    return params


_FILE_PROBES_PER_CASE = 16
"""Probes one :func:`assert_file_resolves_like_inline_entry` case resolves, twice each."""


def file_params(scenarios: Mapping[str, Scenario]) -> list[object]:
    """``pytest.param(scenario)`` per chunk of every scenario's probes.

    For :func:`assert_file_resolves_like_inline_entry`, which replays no REPL
    grouping.
    """
    params: list[object] = []
    for name, scenario in scenarios.items():
        params += _scenario_parts(name, scenario, _FILE_PROBES_PER_CASE, (None,))
    return params


_FILE_ENTRY_HEAD = "def probe() =\n"
"""The function a file holds an inline entry's statements in."""


def file_source(source: str) -> tuple[str, int]:
    """*source* as a file declaring what its inline entry declares, and where its statements start.

    A file's root holds bindings but no expressions or assignments, so the
    inline entry's statements from its first such one on -- which must follow
    every declaration -- become the body of a trailing ``def``, each line
    indented. Returns the file text and the offset its body starts at: the
    text's length when there is none.
    """
    items = parse_entry_module(source, entry_path=None, inline_command=True).program.body.items
    statements: tuple[Item, ...] = ()
    for item in items:
        if isinstance(item, FuncDef) and item.is_synthetic and isinstance(item.body, Block):
            statements = item.body.items
    executable = [item for item in statements if not isinstance(item, (LetDecl, VarDecl))]
    if not executable:
        return source, len(source)
    start = executable[0].span.start_offset
    assert all(
        item.span.end_offset <= start
        for item in items
        if not (isinstance(item, FuncDef) and item.is_synthetic)
    ), source
    head = source[:start] + _FILE_ENTRY_HEAD
    return head + textwrap.indent(source[start:], "  "), len(head)


def _scope_failure(graph: "ModuleGraph") -> AglError | None:
    """The error resolving *graph* raises, or ``None``."""
    try:
        resolve_program(graph)
    except AglError as exc:
        return exc
    return None


def assert_file_resolves_like_inline_entry(tmp_path: Path, scenario: Scenario) -> None:
    """Assert every probe of *scenario* resolves as a file exactly as as an inline entry.

    ``agm exec <file>`` and ``agm exec -c`` read the same text in source
    order: scope accepts both, or rejects both with the same class, span and
    message. The file holds the entry's statements in a function
    (:func:`file_source`); only resolution is compared, since a file's root
    initializers must also be constant. ``info`` probes have no file.
    """
    modules = dict(scenario.modules)
    for key, probe in scenario.probes.items():
        if probe.info:
            continue
        source = "\n".join((*scenario.header, probe.text))
        inline = _scope_failure(
            make_graph_from_files(tmp_path / key / "inline", {"entry": source, **modules})
        )
        text, body_start = file_source(source)
        file_dir = tmp_path / key / "file"
        file_dir.mkdir(parents=True)
        file = _scope_failure(
            make_file_graph_from_files(
                file_dir, {"entry": text, **modules}, entry_path=file_dir / "entry.agl"
            )
        )
        assert type(file) is type(inline), key
        if file is None or inline is None:
            continue
        assert file.span is not None and inline.span is not None, key
        sliced = text[file.span.start_offset : file.span.end_offset]
        if file.span.start_offset >= body_start:
            sliced = sliced.replace("\n  ", "\n")
        assert sliced == source[inline.span.start_offset : inline.span.end_offset], key
        assert str(file) == str(inline), key
