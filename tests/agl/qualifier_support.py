"""Shared REPL-grouping and qualifier-verdict helpers.

A qualifier verdict is which phase, if any, rejected a program -- ``"scope"``
(the resolve phase, which can raise either
:class:`~agm.agl.scope.AglScopeError` or, for a type-name-as-value mistake
caught while resolving, :class:`~agm.agl.typecheck.AglTypeError`),
``"typecheck"`` (the check phase, :class:`~agm.agl.typecheck.AglTypeError`),
or ``"accepted"`` -- plus the raised exception's class and span, obtained by
running the real ``resolve_program``/``check_program`` phases directly and
catching what they raise, never by monkeypatching an internal to observe it.
Classification is by which phase raised, never by the exception's class
alone, since either phase can raise either error class.

:func:`file_verdict` does this for a file-mode module graph.
:func:`repl_verdict_all_groupings` does the REPL equivalent, but a REPL
session accumulates state entry by entry, so *how* a declaration sequence is
split into entries can matter: it evaluates every legal grouping (via
:func:`all_groupings`, skipping a split whose setup does not itself form a
legal session) in its own fresh session, asserts they all reach the same
verdict, and returns it. Every entry but a grouping's last runs through
:meth:`~agm.agl.repl.session.ReplSession.eval_entry` (the real entry
pipeline, accumulating session state exactly as production does); the last is
resolved and checked directly through
:meth:`~agm.agl.repl.entry_pipeline.EntryPipeline.resolve_and_check_program`
-- the same production seam ``ReplSession.type_of`` and its bare-type-entry
fallback use to classify one entry without lowering, evaluating or promoting
it -- so its verdict is obtained the same way :func:`file_verdict` obtains a
file-mode one.

Both helpers take one ``stdlib`` flag: whether the module graph or session
loads the standard library (``std/prelude``), so a caller need not track two
separate defaults for file and REPL mode.

:func:`reptype_verdict_all_groupings` is the REPL-only bare-type-entry
counterpart: it probes a qualifier chain typed alone at the prompt (the
fallback :meth:`~agm.agl.repl.session.ReplSession.eval_entry` reaches for
when an entry does not resolve as a value expression), rather than as an
expression, so it takes its probe separately from its setup declarations.

:func:`all_groupings` and :func:`eval_grouped_final` are the general-purpose
entry-grouping helpers shared with ``tests/test_agl_repl_session.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agm.agl.diagnostics import AglError
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.scope.program import resolve_program
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files

Verdict = tuple[str, type[BaseException] | type[None], SourceSpan | None]


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
    setup entries -- :func:`eval_grouped_final`'s caller-visible diagnostics
    on a bad setup entry, and this module's own :func:`_repl_grouping_verdict`,
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

    Returns the last entry's result unchecked, for a caller that expects it to
    fail depending on how the grouping combines declarations into entries.
    """
    start = sum(sizes[:-1])
    assert eval_setup_entries(session, decls, sizes[:-1]), "a non-final entry failed to set up"
    return session.eval_entry("\n".join(decls[start:]))


def file_verdict(tmp_path: Path, modules: dict[str, str], *, stdlib: bool = True) -> Verdict:
    """Resolve and type-check *modules* as files; report which phase, if any, raised.

    Classifies by which call raised, not by the exception's class: scope
    resolution can itself raise ``AglTypeError`` for a type-name-as-value
    mistake caught while resolving, so only a genuine typecheck-phase
    rejection -- one the (successfully resolved) second call raises --
    is reported as ``"typecheck"``.
    """
    graph = make_graph_from_files(tmp_path, modules, default_stdlib=stdlib)
    try:
        resolved = resolve_program(graph)
    except AglError as exc:
        return "scope", type(exc), exc.span
    try:
        check_program(resolved, base_caps())
    except AglError as exc:
        return "typecheck", type(exc), exc.span
    return "accepted", type(None), None


def _phase_verdict(
    resolve_only: Callable[[], None], resolve_and_check: Callable[[], None]
) -> Verdict:
    """Classify by which of two phase-ordered calls raises, not by exception class.

    Shared by every entry-based verdict helper below: *resolve_only* re-runs
    just the scope phase; *resolve_and_check* re-runs it together with
    typecheck. Both re-resolve the identical input deterministically, so
    *resolve_and_check* raises ``AglScopeError`` only when *resolve_only*
    already did -- a genuine typecheck-phase rejection is one it raises
    after *resolve_only* already succeeded.
    """
    try:
        resolve_only()
    except AglError as exc:
        return "scope", type(exc), exc.span
    try:
        resolve_and_check()
    except AglError as exc:
        return "typecheck", type(exc), exc.span
    return "accepted", type(None), None


def _final_entry_verdict(session: ReplSession, text: str) -> Verdict:
    """Resolve and type-check *text* against *session*'s accumulated state; report the verdict.

    As :func:`file_verdict`, classifies by which call raised: a lone
    :meth:`~agm.agl.repl.session.ReplSession.resolve_entry` re-resolves the
    same text deterministically against the same, already-committed session
    state, so it raises only when the full
    :meth:`~agm.agl.repl.session.ReplSession.resolve_and_check_entry` call
    would raise at the identical, scope-phase point.
    """
    return _phase_verdict(
        lambda: session.resolve_entry(text), lambda: session.resolve_and_check_entry(text)
    )


def _reptype_entry_verdict(session: ReplSession, text: str) -> Verdict:
    """As :func:`_final_entry_verdict`, but for the REPL-only bare-type-entry position.

    Probes *text* the way :meth:`~agm.agl.repl.session.ReplSession.eval_entry`'s
    bare-type fallback does (:meth:`~agm.agl.repl.session.ReplSession.resolve_type_entry`/
    :meth:`~agm.agl.repl.session.ReplSession.resolve_and_check_type_entry`),
    rather than as a value expression: a qualifier chain typed alone at the
    REPL prompt (``Owner::Member``, no ``type X = ...`` wrapper) resolves
    through this position, not the ordinary entry pipeline.
    """
    return _phase_verdict(
        lambda: session.resolve_type_entry(text),
        lambda: session.resolve_and_check_type_entry(text),
    )


def _repl_grouping_verdict(
    session_dir: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    sizes: tuple[int, ...],
    *,
    stdlib: bool,
    final_verdict: Callable[[ReplSession, str], Verdict],
) -> tuple[Verdict, str] | None:
    """Evaluate *decls* as one entry per *sizes* on a fresh session.

    Returns the last entry's verdict (via *final_verdict* -- ordinarily
    :func:`_final_entry_verdict`, or :func:`_reptype_entry_verdict` for a
    bare-type-entry probe) and its own text, or ``None`` when an earlier
    entry fails to set up: not every grouping of a declaration sequence into
    REPL entries is a legal session (an entry's header items must precede
    its non-header ones, and some setup -- a ``use`` naming a scope region --
    needs that region already declared in an earlier entry), so an illegal
    grouping is skipped rather than asserted to succeed.
    """
    for name, source in modules.items():
        if name == "entry":
            continue
        path = session_dir / f"{name}.agl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    session = ReplSession(cwd=session_dir, default_stdlib=stdlib)
    session.open()
    if not eval_setup_entries(session, decls, sizes[:-1]):
        return None
    final_text = "\n".join(decls[sum(sizes[:-1]) :])
    return final_verdict(session, final_text), final_text


def _verdict_over_groupings(
    tmp_path: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    groupings: tuple[tuple[int, ...], ...],
    *,
    stdlib: bool,
    final_verdict: Callable[[ReplSession, str], Verdict],
    pick: Callable[[tuple[int, ...]], bool],
) -> Verdict:
    """Evaluate *decls* over *groupings*; every legal one must reach the identical verdict.

    Shared core for :func:`repl_verdict_all_groupings` and
    :func:`reptype_verdict_all_groupings`, which differ only in which
    groupings are legal to try at all (see :func:`_repl_grouping_verdict`)
    and which one's own verdict *pick* selects to return -- every legal
    grouping's own last-entry text differs, so the returned span is only
    meaningful relative to the one grouping *pick* names; the assertion below
    is what guarantees every other legal grouping agrees with it anyway.
    """
    canonical: tuple[str, type[BaseException] | type[None], str | None] | None = None
    combined: Verdict | None = None
    legal_groupings = 0
    for index, sizes in enumerate(groupings):
        session_dir = tmp_path / str(index)
        session_dir.mkdir()
        outcome = _repl_grouping_verdict(
            session_dir, modules, decls, sizes, stdlib=stdlib, final_verdict=final_verdict
        )
        if outcome is None:
            continue
        legal_groupings += 1
        verdict, text = outcome
        phase, cls, span = verdict
        sliced = text[span.start_offset : span.end_offset] if span is not None else None
        current = (phase, cls, sliced)
        if canonical is None:
            canonical = current
        else:
            assert current == canonical, (sizes, current, canonical)
        if pick(sizes):
            combined = verdict
    assert legal_groupings > 0
    assert combined is not None
    return combined


def repl_verdict_all_groupings(
    tmp_path: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    *,
    stdlib: bool = False,
    final_verdict: Callable[[ReplSession, str], Verdict] = _final_entry_verdict,
) -> Verdict:
    """Evaluate *decls* over every legal way to group them into REPL entries.

    Returns the verdict from the one grouping that puts every declaration in
    a single entry -- its span is relative to ``"\\n".join(decls)``, exactly
    like :func:`file_verdict`'s own, so a caller slices both the same way.
    Every other, legal grouping must reach the identical verdict: a REPL
    session's verdict for a declaration sequence never depends on how it
    happened to be split into entries, only on whether the split is a legal
    session at all. *final_verdict* classifies the grouping's last entry (see
    :func:`_repl_grouping_verdict`).
    """
    return _verdict_over_groupings(
        tmp_path,
        modules,
        decls,
        all_groupings(len(decls)),
        stdlib=stdlib,
        final_verdict=final_verdict,
        pick=lambda sizes: sizes == (len(decls),),
    )


def reptype_verdict_all_groupings(
    tmp_path: Path,
    modules: dict[str, str],
    header: tuple[str, ...],
    probe: str,
    *,
    stdlib: bool = False,
) -> Verdict:
    """As :func:`repl_verdict_all_groupings`, probing the bare-type-entry position.

    *probe* is typed alone at the REPL prompt as a bare type reference
    (``Owner::Member``, never a ``type X = ...`` wrapper) -- the REPL-only
    fallback position :meth:`~agm.agl.repl.session.ReplSession.eval_entry`
    falls back to when an entry does not resolve as a value expression, and
    itself only a single type expression, never combined with *header*'s
    setup declarations in one entry. Every legal way to group *header* into
    entries ahead of *probe*'s own is tried; the returned verdict's span is
    relative to *probe* alone, which every grouping's final entry equals.
    """
    decls = (*header, probe)
    header_groupings = all_groupings(len(header))
    combined_header = (len(header),) if header else ()
    return _verdict_over_groupings(
        tmp_path,
        modules,
        decls,
        tuple((*header_sizes, 1) for header_sizes in header_groupings),
        stdlib=stdlib,
        final_verdict=_reptype_entry_verdict,
        pick=lambda sizes: sizes == (*combined_header, 1),
    )
