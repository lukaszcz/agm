"""Shared REPL-grouping and qualifier-verdict helpers.

A qualifier verdict (:data:`Verdict`) is a 4-tuple ``(phase, cls, span,
identity)``:

``phase``
    A whole program -- the inline ``agm exec -c`` graph (:func:`graph_verdict`,
    :func:`inline_verdict`) or a real ``agm exec <file>`` entry -- has a
    :data:`Phase`: ``"scope"``/``"typecheck"``/``"matchcompile"``/``"accepted"``,
    classified by which real pipeline call raised -- never by the raised
    exception's class alone, since more than one phase can raise the same
    class. A REPL entry has no phase-by-phase call of its own --
    :class:`~agm.agl.repl.entry.EntryResult` carries one ``check_only``
    outcome -- so its own phase is only ``"accepted"``/``"rejected"``, or
    ``"type-entry"`` when an entry that is no value is accepted as a type
    spelling alone; a REPL verdict's *class* and *span* are asserted to match
    the expected ones instead, which implies the same phase.
``cls``/``span``
    The raised exception's class and span, or ``(NoneType, None)`` when
    accepted.
``identity``
    Set only when accepted: the rendered static type of the entry's final
    expression (a program: the checked entry module's own node types; REPL
    mode: ``EntryResult.value_type``, rendered the same way) -- proof of
    which declaration was selected, not merely that nothing raised.

A case is a *header* declaration sequence plus a batch of :class:`Probe`\\ s
sharing it. :func:`assert_verdicts` checks every probe against its expected
verdict in its *file part* -- the header and the probe as one inline entry,
the same text as a real file (:func:`file_source`), and the same text as one
REPL entry, all three rejecting with one message -- and in every other way to
group the header into REPL entries (:func:`all_groupings` over
``len(header) + 1`` items, the ``+ 1`` standing for whichever probe follows),
each rejecting with the inline entry's message too.
Per grouping, one session evaluates ``sizes[:-1]`` setup entries through
:meth:`~agm.agl.repl.session.ReplSession.eval_entry`; whether they all succeed
*is* the grouping's legality, asserted against the expected legal set. Only a
legal grouping goes on to answer every probe as its own
``eval_entry(check_only=True)`` final entry against the header's unconsumed
tail -- the real entry path, never promoting state, so a rejection's
structured cause is ``EntryResult.failure``. An ``info`` probe has neither an
inline entry nor a file: once the tail is promoted, ``:info`` of its name must
report the expected description, or raise the expected class spanning the
expected text.

Probes come from :func:`accepted`, :func:`rejected`, :func:`info` and
:func:`info_rejected`; :func:`type_positions` and
:func:`type_positions_rejected` spell one qualifier in every type position,
and :func:`option_identity` is the identity an ``as?`` cast pins.
:func:`span_text` slices an error's span out of its source for tests that
check a rejection outside the harness.

A check splits into :data:`Part`\\ s, one test case each, so each case stays
cheap: :class:`Scenario` tables feed :func:`assert_scenario` through
:func:`scenario_params`, one part per chunk of a scenario's probes and either
its file part or a batch of its groupings; a test that calls
:func:`assert_verdicts` itself takes its parts from :func:`verdict_parts`.
:func:`assert_repl_verdicts` checks one REPL history of separate entries, for
histories that no file shares.

:func:`all_groupings`, :func:`eval_setup_entries`, :func:`eval_grouped_final`
and :func:`legal_groupings` are the general-purpose entry-grouping helpers
shared with ``tests/test_agl_repl_session.py``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import pytest

from agm.agl.diagnostics import AglError, AglTypeError
from agm.agl.matchcompile import compile_program_matches, match_issue_error
from agm.agl.modules.ids import ENTRY_ID, Reader, spell_declaration
from agm.agl.modules.loader import parse_entry_module
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.repl.type_display import format_type_for_repl
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AmbiguousQualificationError, to_bare_path
from agm.agl.syntax.nodes import Block, FuncDef, Item, LetDecl, VarDecl
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_file_graph_from_files, make_inline_graph_from_files

if TYPE_CHECKING:
    from agm.agl.modules.loader import ModuleGraph
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.types import Type
    from agm.agl.typecheck.env import CheckedModule

Phase = Literal["scope", "typecheck", "matchcompile", "accepted"]
ReplPhase = Literal["accepted", "type-entry", "rejected"]
ProgramVerdict = tuple[Phase, type[BaseException] | type[None], SourceSpan | None, str | None]
ReplVerdict = tuple[ReplPhase, type[BaseException] | type[None], SourceSpan | None, str | None]
Verdict = ProgramVerdict | ReplVerdict
LegalGroupings = frozenset[tuple[int, ...]] | Literal["ALL"]
Origins = frozenset[tuple[type, str]]
Groupings = tuple[tuple[int, ...], ...]
Part = Literal["file"] | Groupings
"""One test case's share of a verdict check: the file part, or a batch of other groupings."""

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


def graph_verdict(graph: "ModuleGraph") -> ProgramVerdict:
    """Resolve, type-check, and match-compile *graph*; report which phase, if any, raised.

    Classifies by which call raised, not by the exception's class: scope
    resolution can itself raise ``AglTypeError`` for a type-name-as-value
    mistake caught while resolving, so only a genuine typecheck-phase
    rejection -- one the (successfully resolved) second call raises -- is
    reported as ``"typecheck"``. On acceptance, *identity* is the rendered
    static type of the entry module's own final item.
    """
    return _graph_outcome(graph)[0]


def _graph_outcome(
    graph: "ModuleGraph", *, wrapped: bool = False
) -> tuple[ProgramVerdict, AglError | None]:
    """:func:`graph_verdict`'s verdict paired with the raised error itself, if any.

    *wrapped* says a real file holds the entry's statements in its final
    ``def`` (:func:`file_source`), whose body's final item the identity reads.
    """
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
        match_error = match_issue_error(match_result.issues[0], resolved.speller)
        return ("matchcompile", type(match_error), match_error.span, None), match_error
    entry = checked_program.modules[checked_program.entry_id]
    identity = _rendered_identity(_entry_final_type(entry, wrapped), entry.type_env.type_table)
    return ("accepted", type(None), None, identity), None


def _rendered_identity(value_type: "Type | None", type_table: "TypeTable | None") -> str | None:
    """Render *value_type* the way the REPL echoes it, or ``None`` when there is none."""
    if value_type is None:
        return None
    return format_type_for_repl(value_type, type_table)


def _entry_final_type(entry: "CheckedModule", wrapped: bool) -> "Type | None":
    """Static type of *entry*'s final source item.

    An inline command's items sit inside the host's synthetic entry
    function, and a *wrapped* file's inside its final ``def``, not directly
    at module top level.
    """
    items = entry.resolved.program.body.items
    if items and isinstance(items[-1], FuncDef) and (items[-1].is_synthetic or wrapped):
        body = items[-1].body
        items = body.items if isinstance(body, Block) else () if body is None else (body,)
    if not items:
        return None
    return entry.node_types.get(items[-1].node_id)


def inline_verdict(
    tmp_path: Path, modules: dict[str, str], *, stdlib: bool = True
) -> ProgramVerdict:
    """Build *modules* into one inline graph and classify it via :func:`graph_verdict`."""
    graph = make_inline_graph_from_files(tmp_path, modules, default_stdlib=stdlib)
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

    *detail* is the rendered identity when accepted, else the exact text
    the error's span slices out. *type_entry* is the identity the REPL
    renders when the probe alone is read as a type spelling; *origins* and
    *spelling* pin an ambiguity's structured fields. An *info* probe's text
    is a name ``:info`` describes: *detail* is the description when
    accepted. *in_file* is the verdict a real file holding the text reaches
    instead, where a file reads it differently (an ``extern def``, a
    non-constant binding; see :func:`nonconstant_in_file`).
    """

    text: str
    phase: Phase
    error: type[AglError] | None
    detail: str
    type_entry: str | None = None
    origins: Origins | None = None
    spelling: str | None = None
    info: bool = False
    in_file: Probe | None = None


def accepted(text: str, identity: str) -> Probe:
    """A probe every mode accepts with rendered *identity*."""
    return Probe(text, "accepted", None, identity)


def option_identity(selected: str) -> str:
    """The rendered identity of ``v as? q`` when ``q`` selects *selected*."""
    return f"enum std/option::Option[{selected}]\n  | None\n  | Some(value: {selected})"


def rejected(
    text: str,
    error: type[AglError],
    span: str,
    *,
    phase: Phase = "scope",
    type_entry: str | None = None,
    origins: Origins | None = None,
    spelling: str | None = None,
) -> Probe:
    """A probe every mode rejects, a program in *phase*, with *error* spanning *span*."""
    return Probe(text, phase, error, span, type_entry, origins, spelling)


def info(name: str, description: str) -> Probe:
    """``:info name`` describing the selected declaration as *description*."""
    return Probe(name, "accepted", None, description, info=True)


def info_rejected(name: str, error: type[AglError], span: str) -> Probe:
    """``:info name`` raising scope's *error* spanning *span*."""
    return Probe(name, "scope", error, span, info=True)


def nonconstant_in_file(probes: Mapping[K, Probe], keys: Collection[K]) -> dict[K, Probe]:
    """*probes*, where a real file rejects each of *keys*' last binding as not constant.

    An inline entry's bindings, in its scope regions too, need no constant
    initializer; a file's root and scoped ones do. The rejection spans the
    initializer of the last ``let`` or ``var`` the probe's text writes.
    """
    return {
        key: probe
        if key not in keys
        else replace(
            probe,
            in_file=rejected(
                probe.text, AglTypeError, _last_initializer(probe.text), phase="typecheck"
            ),
        )
        for key, probe in probes.items()
    }


def _last_initializer(text: str) -> str:
    """The initializer text of the last ``let`` or ``var`` binding *text* writes."""
    bindings: list[LetDecl | VarDecl] = []
    walk(
        parse_entry_module(text, entry_path=None, inline_code=True).program,
        lambda node: bindings.append(node) if isinstance(node, (LetDecl, VarDecl)) else None,
    )
    span = max(bindings, key=lambda binding: binding.span.start_offset).value.span
    return text[span.start_offset : span.end_offset]


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
    phase: Phase = "scope",
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
    expected: Mapping[K, tuple[Phase, type[BaseException] | type[None]]],
    *,
    span_texts: Mapping[K, str] | None = None,
    identities: Mapping[K, str] | None = None,
    origins: Mapping[K, Origins] | None = None,
    spellings: Mapping[K, str] | None = None,
    type_entries: Mapping[K, str] | None = None,
) -> dict[K, Probe]:
    """One :class:`Probe` per *texts* entry, from its ``(phase, class)`` in *expected*.

    An accepted probe's *identities* entry and a rejected probe's
    *span_texts* entry are required; every other mapping is optional per key.
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
            table[key] = accepted(text, identities[key])
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


def _assert_program_outcome(
    key: object,
    outcome: tuple[ProgramVerdict, AglError | None],
    probe: Probe,
    sliced: Callable[[SourceSpan | None], str | None],
) -> None:
    """Assert a whole program's *outcome* is *probe*'s; *sliced* reads a span's source text."""
    (phase, cls, span, identity), failure = outcome
    assert (phase, cls) == (probe.phase, _probe_class(probe)), key
    if probe.error is None:
        assert identity == probe.detail, key
        return
    assert sliced(span) == probe.detail, key
    _assert_ambiguity_fields(key, failure, probe)


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
    inline_failures: Mapping[K, AglError | None] | None,
    legal: frozenset[tuple[int, ...]],
    stdlib: bool,
) -> None:
    """Assert *sizes*'s legality and, when legal, every probe's REPL verdict (module docstring).

    With *inline_failures*, a rejection's message must also be the inline
    entry's (``None`` only for a REPL history no inline entry shares).
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
        if probe.error is None:
            assert identity == probe.detail, (key, sizes)
            continue
        assert span_text("\n".join((*tail, probe.text)), span) == probe.detail, (key, sizes)
        assert inline_failures is None or str(failure) == str(inline_failures[key]), (key, sizes)
        _assert_ambiguity_fields((key, sizes), failure, probe)


def _legal_set(header: tuple[str, ...], legal: LegalGroupings) -> frozenset[tuple[int, ...]]:
    return frozenset(all_groupings(len(header) + 1)) if legal == "ALL" else legal


def _inline_outcome(
    tmp_path: Path,
    modules: Mapping[str, str],
    header: tuple[str, ...],
    probe: Probe,
    stdlib: bool,
) -> tuple[ProgramVerdict, AglError | None]:
    """*probe*'s outcome as one inline entry after *header*."""
    source = "\n".join((*header, probe.text))
    return _graph_outcome(
        make_inline_graph_from_files(
            tmp_path / "inline", {"entry": source, **modules}, default_stdlib=stdlib
        )
    )


def _inline_failures(
    tmp_path: Path,
    modules: Mapping[str, str],
    header: tuple[str, ...],
    probes: Mapping[K, Probe],
    stdlib: bool,
) -> dict[K, AglError | None]:
    """Each rejected probe's inline-entry failure, which its REPL entries must repeat."""
    return {
        key: _inline_outcome(tmp_path, modules, header, probe, stdlib)[1]
        for key, probe in probes.items()
        if probe.error is not None and not probe.info
    }


def _assert_file_part(
    tmp_path: Path,
    modules: Mapping[str, str],
    header: tuple[str, ...],
    probes: Mapping[K, Probe],
    legal: frozenset[tuple[int, ...]],
    stdlib: bool,
) -> dict[K, AglError | None]:
    """Assert every probe as an inline entry, a real file and one REPL entry (module docstring).

    The real file rejects with the inline entry's message; the REPL entry, a
    grouping every header has, with the inline entry's too. Returns each
    probe's inline-entry failure.
    """
    file_dir = tmp_path / "file"
    file_dir.mkdir()
    # The real file's Python companion, which a file's ``extern def`` needs.
    (file_dir / "entry.py").touch()
    inline_failures: dict[K, AglError | None] = {}
    for key, probe in probes.items():
        if probe.info:
            continue
        source = "\n".join((*header, probe.text))
        inline = _inline_outcome(tmp_path, modules, header, probe, stdlib)
        _assert_program_outcome(key, inline, probe, partial(span_text, source))
        inline_failures[key] = inline[1]
        text, body_start = file_source(source)
        file = _graph_outcome(
            make_file_graph_from_files(
                file_dir,
                {"entry": text, **modules},
                default_stdlib=stdlib,
                entry_path=file_dir / "entry.agl",
            ),
            wrapped=body_start < len(text),
        )
        file_probe = probe.in_file or probe
        _assert_program_outcome(key, file, file_probe, partial(_file_span_text, text, body_start))
        assert probe.in_file is not None or str(file[1]) == str(inline[1]), key
    _assert_grouping(
        tmp_path, modules, header, (len(header) + 1,), probes, inline_failures, legal, stdlib
    )
    return inline_failures


def assert_verdicts(
    tmp_path: Path,
    modules: Mapping[str, str],
    header: tuple[str, ...],
    probes: Mapping[K, Probe],
    *,
    legal: LegalGroupings = "ALL",
    stdlib: bool = True,
    part: Part | None = None,
) -> None:
    """Assert every probe in its file part, then in each other REPL grouping of *header*.

    *legal* is the set of full groupings (over ``len(header) + 1`` items)
    whose setup entries succeed, ``"ALL"`` when every one does. *part*
    narrows the check to one :func:`verdict_parts` part; ``None`` is all of
    it.
    """
    legal_set = _legal_set(header, legal)
    inline_failures = (
        _assert_file_part(tmp_path, modules, header, probes, legal_set, stdlib)
        if part is None or part == "file"
        else None
    )
    groupings = _other_groupings(len(header) + 1) if part is None else part
    if groupings == "file":
        return
    assert set(groupings) <= set(_other_groupings(len(header) + 1)), groupings
    if inline_failures is None:
        inline_failures = _inline_failures(tmp_path, modules, header, probes, stdlib)
    for sizes in groupings:
        _assert_grouping(
            tmp_path, modules, header, sizes, probes, inline_failures, legal_set, stdlib
        )


def assert_verdict(
    tmp_path: Path,
    modules: Mapping[str, str],
    decls: tuple[str, ...],
    expected: tuple[Phase, type[BaseException] | type[None]],
    *,
    span_text: str | None = None,
    identity: str | None = None,
    origins: Origins | None = None,
    legal: LegalGroupings = "ALL",
    stdlib: bool = True,
    part: Part | None = None,
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
        part=part,
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
    redeclaring what an earlier one already decided -- so no inline entry
    or file is compared.
    """
    sizes = (*(1 for _ in entries), 1)
    _assert_grouping(tmp_path, modules, entries, sizes, probes, None, frozenset({sizes}), stdlib)


@dataclass(frozen=True)
class Scenario:
    """Modules, a shared header and its probes, with the header's legal REPL groupings.

    *part* narrows a check to one part of it (see :func:`scenario_params`);
    ``None`` is all of it.
    """

    header: tuple[str, ...]
    probes: Mapping[str, Probe]
    modules: Mapping[str, str] = field(default_factory=dict)
    legal: LegalGroupings = "ALL"
    part: Part | None = None


def assert_scenario(tmp_path: Path, scenario: Scenario) -> None:
    """:func:`assert_verdicts` for every probe of *scenario*, over its part."""
    assert_verdicts(
        tmp_path,
        scenario.modules,
        scenario.header,
        scenario.probes,
        legal=scenario.legal,
        part=scenario.part,
    )


_CHECKS_PER_CASE = 24
"""Cost one test case spends at most, in ``check_only`` entries against a set-up header."""

_SESSION_CHECKS = 4
"""What opening a session and setting its header up costs, in ``check_only`` entries."""

_PROBES_PER_FILE_CASE = 16
"""Probes one file-part case checks, each as an inline entry, a real file and a REPL entry."""


def _batches(items: Sequence[T], size: int) -> list[tuple[T, ...]]:
    return [tuple(items[i : i + size]) for i in range(0, len(items), size)]


def _other_groupings(n: int) -> Groupings:
    """Every grouping over *n* items but the single entry, which the file part replays."""
    return all_groupings(n)[:-1]


def _grouping_parts(n: int, probes: int) -> list[tuple[Part, str]]:
    """Batches of the other groupings over *n* items, each with its id, for *probes* probes.

    Each grouping opens one session and answers every probe in it, so fewer
    groupings share a case the more probes each answers.
    """
    per_case = max(1, _CHECKS_PER_CASE // (_SESSION_CHECKS + probes))
    return [
        (batch, "groupings-" + ".".join(map(str, batch[0])))
        for batch in _batches(_other_groupings(n), per_case)
    ]


def verdict_parts(n: int, *, probes: int = 8) -> list[object]:
    """``pytest.param(part)`` per part of an :func:`assert_verdicts` check over *n* items.

    *probes* is how many probes the check asserts, which sizes its batches.
    """
    return [
        pytest.param("file", id="file"),
        *(pytest.param(batch, id=part_id) for batch, part_id in _grouping_parts(n, probes)),
    ]


def scenario_params(scenarios: Mapping[str, Scenario]) -> list[object]:
    """``pytest.param(scenario)`` per part of every chunk of every scenario's probes.

    For :func:`assert_scenario`. The file part checks chunks of
    :data:`_PROBES_PER_FILE_CASE` probes; the groupings, chunks of as many
    as one case answers in one session. A chunk is named by its first probe
    key when its scenario has more than one.
    """
    params: list[object] = []
    for name, scenario in scenarios.items():
        n = len(scenario.header) + 1
        keys = list(scenario.probes)
        for size, parts in (
            (_PROBES_PER_FILE_CASE, lambda _probes: [("file", "file")]),
            (_CHECKS_PER_CASE - _SESSION_CHECKS, lambda probes: _grouping_parts(n, probes)),
        ):
            chunks = _batches(keys, size)
            for chunk in chunks:
                prefix = name if len(chunks) == 1 else f"{name}-{chunk[0]}"
                probes = {key: scenario.probes[key] for key in chunk}
                params += [
                    pytest.param(replace(scenario, probes=probes, part=part), id=f"{prefix}-{id_}")
                    for part, id_ in parts(len(chunk))
                ]
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
    items = parse_entry_module(source, entry_path=None, inline_code=True).program.body.items
    statements: tuple[Item, ...] = ()
    for item in items:
        if isinstance(item, FuncDef) and item.is_synthetic and isinstance(item.body, Block):
            statements = item.body.items
    executable = [item for item in statements if not isinstance(item, (LetDecl, VarDecl))]
    if not executable:
        return source, len(source)
    # A statement's own span may begin inside it, after an opening parenthesis.
    start = source.rfind("\n", 0, executable[0].span.start_offset) + 1
    assert all(
        item.span.end_offset <= start
        for item in items
        if not (isinstance(item, FuncDef) and item.is_synthetic)
    ), source
    return source[:start] + file_statements(source[start:]), start + len(_FILE_ENTRY_HEAD)


def file_statements(statements: str) -> str:
    """*statements* in the function a file holds its inline entry's statements in."""
    return _FILE_ENTRY_HEAD + textwrap.indent(statements, "  ")


def _file_span_text(text: str, body_start: int, span: SourceSpan | None) -> str | None:
    """The slice of file *text* that *span* covers, with its body lines' indent removed."""
    sliced = span_text(text, span)
    if sliced is None or span is None or span.start_offset < body_start:
        return sliced
    return sliced.replace("\n  ", "\n")
