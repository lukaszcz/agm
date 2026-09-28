"""Shared REPL-grouping and qualifier-verdict helpers.

A qualifier verdict is which phase, if any, rejected a program -- ``"scope"``
(:class:`~agm.agl.scope.AglScopeError`), ``"typecheck"``
(:class:`~agm.agl.typecheck.AglTypeError`), or ``"accepted"`` -- plus the
raised exception's class and span, obtained by running the real
``resolve_program``/``check_program`` phases directly and catching what they
raise, never by monkeypatching an internal to observe it.

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

:func:`all_groupings` and :func:`eval_grouped_final` are the general-purpose
entry-grouping helpers shared with ``tests/test_agl_repl_session.py``.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl.repl import EntryResult, ReplSession
from agm.agl.scope import AglScopeError
from agm.agl.scope.program import resolve_program
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files

Verdict = tuple[str, type[BaseException] | type[None], SourceSpan | None]


def all_groupings(n: int) -> tuple[tuple[int, ...], ...]:
    """Every way to split *n* declarations, in order, into one or more entries."""
    if n == 0:
        return ((),)
    return tuple((first, *rest) for first in range(1, n + 1) for rest in all_groupings(n - first))


def eval_grouped_final(
    session: ReplSession, decls: tuple[str, ...], sizes: tuple[int, ...]
) -> EntryResult:
    """Evaluate *decls* as one entry per *sizes*; every entry but the last must succeed.

    Returns the last entry's result unchecked, for a caller that expects it to
    fail depending on how the grouping combines declarations into entries.
    """
    start = 0
    result: EntryResult | None = None
    for size in sizes:
        result = session.eval_entry("\n".join(decls[start : start + size]))
        if start + size < len(decls):
            assert result.ok, result.diagnostics
        start += size
    assert result is not None
    return result


def file_verdict(tmp_path: Path, modules: dict[str, str]) -> Verdict:
    """Resolve and type-check *modules* as files; report which phase, if any, raised."""
    graph = make_graph_from_files(tmp_path, modules)
    try:
        resolved = resolve_program(graph)
    except AglScopeError as exc:
        return "scope", type(exc), exc.span
    try:
        check_program(resolved, base_caps())
    except AglTypeError as exc:
        return "typecheck", type(exc), exc.span
    return "accepted", type(None), None


def _final_entry_verdict(session: ReplSession, text: str) -> Verdict:
    """Resolve and type-check *text* against *session*'s accumulated state; report the verdict."""
    from agm.agl.lexer import spaced_qualifier_collector
    from agm.agl.parser import parse_program_seeded

    host_env = session._runtime.host_environment()
    with spaced_qualifier_collector() as spaced_sink:
        program, next_node_id = parse_program_seeded(
            text, start_id=session._next_node_id, resolve_infix=False
        )
    try:
        session._entry_pipeline.resolve_and_check_program(
            program, next_node_id, host_env, spaced_qualifiers=tuple(spaced_sink)
        )
    except AglScopeError as exc:
        return "scope", type(exc), exc.span
    except AglTypeError as exc:
        return "typecheck", type(exc), exc.span
    return "accepted", type(None), None


def _repl_grouping_verdict(
    session_dir: Path, modules: dict[str, str], decls: tuple[str, ...], sizes: tuple[int, ...]
) -> tuple[Verdict, str] | None:
    """Evaluate *decls* as one entry per *sizes* on a fresh session.

    Returns the last entry's verdict and its own text, or ``None`` when an
    earlier entry fails to set up: not every grouping of a declaration
    sequence into REPL entries is a legal session (an entry's header items
    must precede its non-header ones, and some setup -- a ``use`` naming a
    scope region -- needs that region already declared in an earlier entry),
    so an illegal grouping is skipped rather than asserted to succeed.
    """
    for name, source in modules.items():
        if name == "entry":
            continue
        path = session_dir / f"{name}.agl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    session = ReplSession(cwd=session_dir, default_stdlib=False)
    session.open()
    start = 0
    for size in sizes[:-1]:
        result = session.eval_entry("\n".join(decls[start : start + size]))
        if not result.ok:
            return None
        start += size
    final_text = "\n".join(decls[start:])
    return _final_entry_verdict(session, final_text), final_text


def repl_verdict_all_groupings(
    tmp_path: Path, modules: dict[str, str], decls: tuple[str, ...]
) -> Verdict:
    """Evaluate *decls* over every legal way to group them into REPL entries.

    Returns the verdict from the one grouping that puts every declaration in
    a single entry -- its span is relative to ``"\\n".join(decls)``, exactly
    like :func:`file_verdict`'s own, so a caller slices both the same way.
    Every other, legal grouping (see :func:`_repl_grouping_verdict`) must
    reach the identical verdict, its span text sliced from its own entry's
    text: a REPL session's verdict for a declaration sequence never depends
    on how it happened to be split into entries, only on whether the split
    is a legal session at all.
    """
    canonical: tuple[str, type[BaseException] | type[None], str | None] | None = None
    combined: Verdict | None = None
    legal_groupings = 0
    for index, sizes in enumerate(all_groupings(len(decls))):
        session_dir = tmp_path / str(index)
        session_dir.mkdir()
        outcome = _repl_grouping_verdict(session_dir, modules, decls, sizes)
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
        if sizes == (len(decls),):
            combined = verdict
    assert legal_groupings > 0
    assert combined is not None
    return combined
