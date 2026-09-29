"""Shared REPL-grouping and qualifier-verdict helpers.

A qualifier verdict (:data:`Verdict`) is a 4-tuple ``(phase, cls, span,
identity)``:

``phase``
    ``"scope"``/``"typecheck"``/``"accepted"`` in FILE mode (see
    :func:`file_verdict`, which obtains it by running the real
    ``resolve_program``/``check_program`` phases directly and catching what
    each raises -- classification is by which call raised, never by the
    raised exception's class alone, since either phase can raise either
    error class (a type-name-as-value mistake is ``AglTypeError`` even when
    caught while resolving). REPL mode (:func:`repl_verdict_all_groupings`,
    :func:`reptype_verdict_all_groupings`) has no equivalent two-step
    call -- :class:`~agm.agl.repl.entry.EntryResult` carries one
    ``check_only`` outcome, not a phase-by-phase one -- so its own ``phase``
    is only ``"accepted"``/``"rejected"``; a REPL verdict's *class* and
    *span* are asserted to match a freshly computed file-mode verdict for
    the identical case instead, which implies the same phase.
``cls``/``span``
    The raised exception's class and span, or ``(NoneType, None)`` when
    accepted.
``identity``
    Set only when accepted: the rendered static type of the entry's final
    expression (FILE mode: from the checked entry module's own node types;
    REPL mode: ``EntryResult.value_type``, rendered the same way) -- proof
    that an owner is instantiated from scope's recorded key, not re-resolved
    by name.

Both REPL helpers run every legal way to group *decls* into REPL entries
(:func:`all_groupings`, one fresh :class:`~agm.agl.repl.session.ReplSession`
per grouping): every entry but a grouping's last runs through
:meth:`~agm.agl.repl.session.ReplSession.eval_entry` (the real entry
pipeline, accumulating session state exactly as production does); the last
runs through ``eval_entry(check_only=True)`` -- the same real entry path a
production REPL uses to classify one entry without lowering, evaluating, or
promoting it, so a rejection's structured cause is
``EntryResult.failure`` (never a second, resolve-only call). Every legal
grouping must reach the identical verdict, and the exact set of legal
groupings (never silently narrowed by skipping an illegal one) must equal
*expected_legal_groupings* -- ``"ALL"`` when every grouping tried is legal,
otherwise the literal set.

:func:`reptype_verdict_all_groupings` probes a qualifier chain typed alone
at the REPL prompt (the same real ``eval_entry`` fallback a production REPL
falls back to when an entry does not resolve as a value expression, not a
dedicated test hook), so it takes its probe separately from its setup
declarations and keeps the probe as the grouping's own final, single-entry
tail.

Both helpers take one ``stdlib`` flag: whether the module graph or session
loads the standard library (``std/prelude``), so a caller need not track two
separate defaults for file and REPL mode.

:func:`all_groupings` and :func:`eval_grouped_final` are the general-purpose
entry-grouping helpers shared with ``tests/test_agl_repl_session.py``;
:func:`phase_verdict` is the "which of two phase-ordered calls raised"
classifier shared with ``tests/test_agl_qualifier_resolution.py``'s own
non-``ModuleGraph`` inline-entry resolver.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from agm.agl.diagnostics import AglError
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.repl.type_display import format_type_for_repl
from agm.agl.scope.program import resolve_program
from agm.agl.syntax.nodes import FuncDef
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files

if TYPE_CHECKING:
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.types import Type
    from agm.agl.typecheck.env import CheckedModule

Verdict = tuple[str, type[BaseException] | type[None], SourceSpan | None, str | None]
LegalGroupings = frozenset[tuple[int, ...]] | Literal["ALL"]


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
    on a bad setup entry, and this module's own grouping walk below -- build
    on it instead of repeating the walk.
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


def phase_verdict(
    resolve_only: Callable[[], object], resolve_and_check: Callable[[], object]
) -> tuple[str, type[BaseException] | type[None], SourceSpan | None]:
    """Classify by which of two phase-ordered calls raises, not by exception class.

    *resolve_only* re-runs just the scope phase; *resolve_and_check* re-runs
    it together with typecheck. Both re-resolve the identical input
    deterministically, so *resolve_and_check* raises ``AglScopeError`` only
    when *resolve_only* already did -- a genuine typecheck-phase rejection is
    one it raises after *resolve_only* already succeeded.
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


def _rendered_identity(value_type: "Type | None", type_table: "TypeTable | None") -> str | None:
    """Render *value_type* the way the REPL echoes it, or ``None`` when there is none."""
    if value_type is None:
        return None
    return format_type_for_repl(value_type, type_table)


def _entry_final_type(entry: "CheckedModule") -> "Type | None":
    """Static type of *entry*'s final source item.

    An inline command's items sit inside the host's synthetic entry
    function, not directly at module top level; a real, source-declared
    ``program def`` needs no such unwrapping.
    """
    items = entry.resolved.program.body.items
    if items and isinstance(items[-1], FuncDef) and items[-1].is_synthetic:
        items = items[-1].body.items
    if not items:
        return None
    return entry.node_types.get(items[-1].node_id)


def file_verdict(tmp_path: Path, modules: dict[str, str], *, stdlib: bool = True) -> Verdict:
    """Resolve and type-check *modules* as files; report which phase, if any, raised.

    Classifies by which call raised, not by the exception's class: scope
    resolution can itself raise ``AglTypeError`` for a type-name-as-value
    mistake caught while resolving, so only a genuine typecheck-phase
    rejection -- one the (successfully resolved) second call raises --
    is reported as ``"typecheck"``. On acceptance, *identity* is the
    rendered static type of the entry module's own final item.
    """
    graph = make_graph_from_files(tmp_path, modules, default_stdlib=stdlib)
    try:
        resolved = resolve_program(graph)
    except AglError as exc:
        return "scope", type(exc), exc.span, None
    try:
        checked_program = check_program(resolved, base_caps())
    except AglError as exc:
        return "typecheck", type(exc), exc.span, None
    entry = checked_program.modules[checked_program.entry_id]
    identity = _rendered_identity(_entry_final_type(entry), entry.type_env.type_table)
    return "accepted", type(None), None, identity


def _check_only_verdict(session: ReplSession, text: str) -> Verdict:
    """Evaluate *text* as a ``check_only`` entry against *session*; report its verdict.

    The real entry path (see module docstring): every ``check_only``
    rejection is a static one (parse/scope/typecheck/match-compile), so
    ``EntryResult.failure`` is always set on rejection.
    """
    result = session.eval_entry(text, check_only=True)
    if result.ok:
        identity = _rendered_identity(result.value_type, result.type_table)
        return "accepted", type(None), None, identity
    failure = result.failure
    assert failure is not None, "a check_only rejection always carries its static failure"
    return "rejected", type(failure), failure.span, None


def _repl_grouping_verdict(
    session_dir: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    sizes: tuple[int, ...],
    *,
    stdlib: bool,
) -> tuple[Verdict, str] | None:
    """Evaluate *decls* as one entry per *sizes* on a fresh session.

    Returns the last entry's verdict (via :func:`_check_only_verdict`) and
    its own text, or ``None`` when an earlier entry fails to set up: not
    every grouping of a declaration sequence into REPL entries is a legal
    session (an entry's header items must precede its non-header ones, and
    some setup -- a ``use`` naming a scope region -- needs that region
    already declared in an earlier entry), so an illegal grouping is skipped
    rather than asserted to succeed.
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
    return _check_only_verdict(session, final_text), final_text


def _verdict_over_groupings(
    tmp_path: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    groupings: tuple[tuple[int, ...], ...],
    *,
    stdlib: bool,
    pick: Callable[[tuple[int, ...]], bool],
    expected_legal_groupings: LegalGroupings,
) -> Verdict:
    """Evaluate *decls* over *groupings*; every legal one must reach the identical verdict.

    Shared core for :func:`repl_verdict_all_groupings` and
    :func:`reptype_verdict_all_groupings`, which differ only in which
    groupings are tried at all (see :func:`_repl_grouping_verdict`) and
    which one's own verdict *pick* selects to return -- every legal
    grouping's own last-entry text differs, so the returned span is only
    meaningful relative to the one grouping *pick* names; the assertion below
    is what guarantees every other legal grouping agrees with it anyway. The
    exact set of groupings that turn out legal must equal
    *expected_legal_groupings* -- an illegal grouping is never silently
    skipped without that set being asserted.
    """
    canonical: tuple[type[BaseException] | type[None], str | None, str | None] | None = None
    combined: Verdict | None = None
    legal: set[tuple[int, ...]] = set()
    for index, sizes in enumerate(groupings):
        session_dir = tmp_path / str(index)
        session_dir.mkdir()
        outcome = _repl_grouping_verdict(session_dir, modules, decls, sizes, stdlib=stdlib)
        if outcome is None:
            continue
        legal.add(sizes)
        verdict, text = outcome
        _phase, cls, span, identity = verdict
        sliced = text[span.start_offset : span.end_offset] if span is not None else None
        current = (cls, sliced, identity)
        if canonical is None:
            canonical = current
        else:
            assert current == canonical, (sizes, current, canonical)
        if pick(sizes):
            combined = verdict
    expected = (
        frozenset(groupings) if expected_legal_groupings == "ALL" else expected_legal_groupings
    )
    assert legal == expected, (legal, expected)
    assert combined is not None
    return combined


def repl_verdict_all_groupings(
    tmp_path: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    *,
    stdlib: bool = True,
    expected_legal_groupings: LegalGroupings = "ALL",
) -> Verdict:
    """Evaluate *decls* over every legal way to group them into REPL entries.

    Returns the verdict from the one grouping that puts every declaration in
    a single entry -- its span is relative to ``"\\n".join(decls)``, exactly
    like :func:`file_verdict`'s own, so a caller slices both the same way.
    Every other, legal grouping must reach the identical verdict: a REPL
    session's verdict for a declaration sequence never depends on how it
    happened to be split into entries, only on whether the split is a legal
    session at all.
    """
    return _verdict_over_groupings(
        tmp_path,
        modules,
        decls,
        all_groupings(len(decls)),
        stdlib=stdlib,
        pick=lambda sizes: sizes == (len(decls),),
        expected_legal_groupings=expected_legal_groupings,
    )


def reptype_verdict_all_groupings(
    tmp_path: Path,
    modules: dict[str, str],
    header: tuple[str, ...],
    probe: str,
    *,
    stdlib: bool = True,
    expected_legal_groupings: LegalGroupings = "ALL",
) -> Verdict:
    """As :func:`repl_verdict_all_groupings`, keeping *probe* as its own final entry.

    *probe* is typed alone at the REPL prompt -- never combined with
    *header*'s setup declarations in one entry -- and evaluated through the
    same real ``eval_entry(check_only=True)`` path; when *probe* does not
    resolve as an ordinary value expression, that path is production's own
    REPL-only bare-type-entry fallback. Every legal way to group *header*
    into entries ahead of *probe*'s own is tried; the returned verdict's span
    is relative to *probe* alone, which every grouping's final entry equals.
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
        pick=lambda sizes: sizes == (*combined_header, 1),
        expected_legal_groupings=expected_legal_groupings,
    )
