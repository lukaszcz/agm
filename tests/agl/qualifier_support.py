"""Shared REPL-grouping and qualifier-verdict helpers.

A qualifier verdict (:data:`Verdict`) is a 4-tuple ``(phase, cls, span,
identity)``:

``phase``
    FILE mode (:func:`graph_verdict`, :func:`file_verdict`) is
    ``"scope"``/``"typecheck"``/``"matchcompile"``/``"accepted"``, classified
    by which real pipeline call raised -- never by the raised exception's
    class alone, since more than one phase can raise the same class. REPL
    mode (:func:`repl_verdict_all_groupings`, :func:`repl_matrix_verdicts`)
    has no phase-by-phase call of its own -- :class:`~agm.agl.repl.entry.EntryResult`
    carries one ``check_only`` outcome -- so its own phase is only
    ``"accepted"``/``"rejected"``; a REPL verdict's *class* and *span* are
    asserted to match a fresh file-mode verdict for the identical case
    instead, which implies the same phase.
``cls``/``span``
    The raised exception's class and span, or ``(NoneType, None)`` when
    accepted.
``identity``
    Set only when accepted: the rendered static type of the entry's final
    expression (FILE mode: the checked entry module's own node types; REPL
    mode: ``EntryResult.value_type``, rendered the same way) -- proof that an
    owner is instantiated from scope's recorded key, not re-resolved by name.

The REPL helpers run every legal way to group a declaration sequence into
REPL entries (:func:`all_groupings`, :func:`legal_groupings`): every entry
but a grouping's last runs through
:meth:`~agm.agl.repl.session.ReplSession.eval_entry` (accumulating session
state exactly as production does); the last runs through
``eval_entry(check_only=True)`` -- the same real entry path a production REPL
uses to classify one entry without lowering, evaluating, or promoting it, so
a rejection's structured cause is ``EntryResult.failure``. Every legal
grouping must reach the identical verdict, and the exact set of legal
groupings must equal *expected_legal_groupings* -- ``"ALL"`` when every
grouping tried is legal, otherwise the literal set.

:func:`repl_matrix_verdicts` answers many probes (for example every
outcome/position combination for one qualifier form) over the same *header*
declarations at once: since a ``check_only`` probe never promotes state, one
session per way of grouping *header*'s own leading items checks every probe
as its own final entry, instead of rebuilding a session per probe -- a pure
reparametrization of calling :func:`repl_verdict_all_groupings` once per
probe, trying the identical set of (probe, grouping) combinations.

:func:`assert_verdict_everywhere` is the one assertion helper: it computes
the file verdict, asserts it against an expected ``(phase, cls)``, then
asserts every REPL grouping's own verdict agrees.

:func:`all_groupings`, :func:`eval_setup_entries`, :func:`eval_grouped_final`
and :func:`legal_groupings` are the general-purpose entry-grouping helpers
shared with ``tests/test_agl_repl_session.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from agm.agl.diagnostics import AglError
from agm.agl.matchcompile import compile_program_matches, match_issue_error
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.repl.type_display import format_type_for_repl
from agm.agl.scope.program import resolve_program
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
ReplPhase = Literal["accepted", "rejected"]
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
    that session as a side effect before returning. Shared by
    :func:`_verdict_over_groupings` (whose own *is_legal* also records the
    final entry's own check-only verdict) and
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
    try:
        resolved = resolve_program(graph)
    except AglError as exc:
        return "scope", type(exc), exc.span, None
    try:
        checked_program = check_program(resolved, base_caps())
    except AglError as exc:
        return "typecheck", type(exc), exc.span, None
    match_result = compile_program_matches(checked_program)
    if match_result.compiled is None:
        match_error = match_issue_error(match_result.issues[0])
        return "matchcompile", type(match_error), match_error.span, None
    entry = checked_program.modules[checked_program.entry_id]
    identity = _rendered_identity(_entry_final_type(entry), entry.type_env.type_table)
    return "accepted", type(None), None, identity


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


def _check_only_verdict(session: ReplSession, text: str) -> ReplVerdict:
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
    if failure is None:
        raise AssertionError("a check_only rejection always carries its static failure")
    return "rejected", type(failure), failure.span, None


def _write_modules(session_dir: Path, modules: dict[str, str]) -> None:
    """Write every module except ``entry`` under *session_dir*, one file per module."""
    for name, source in modules.items():
        if name == "entry":
            continue
        path = session_dir / f"{name}.agl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _verdict_over_groupings(
    tmp_path: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    groupings: tuple[tuple[int, ...], ...],
    *,
    stdlib: bool,
    pick: Callable[[tuple[int, ...]], bool],
    expected_legal_groupings: LegalGroupings,
) -> ReplVerdict:
    """Evaluate *decls* over *groupings*; every legal one must reach the identical verdict.

    Shared core for :func:`repl_verdict_all_groupings`, whose own *pick*
    selects the one grouping whose verdict is returned -- every legal
    grouping's own last-entry text differs, so the returned span is only
    meaningful relative to the one grouping *pick* names; the assertion
    below is what guarantees every other legal grouping agrees with it
    anyway. The exact set of groupings that turn out legal must equal
    *expected_legal_groupings*.
    """
    canonical: tuple[type[BaseException] | type[None], str | None, str | None] | None = None
    combined: ReplVerdict | None = None
    verdicts: dict[tuple[int, ...], tuple[ReplVerdict, str]] = {}
    counter = 0

    def make_session() -> ReplSession:
        nonlocal counter
        session_dir = tmp_path / str(counter)
        counter += 1
        session_dir.mkdir()
        _write_modules(session_dir, modules)
        session = ReplSession(cwd=session_dir, default_stdlib=stdlib)
        session.open()
        return session

    def is_legal(session: ReplSession, sizes: tuple[int, ...]) -> bool:
        if not eval_setup_entries(session, decls, sizes[:-1]):
            return False
        final_text = "\n".join(decls[sum(sizes[:-1]) :])
        verdicts[sizes] = (_check_only_verdict(session, final_text), final_text)
        return True

    legal = legal_groupings(groupings, make_session, is_legal)
    for sizes in legal:
        verdict, text = verdicts[sizes]
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
    if combined is None:
        raise AssertionError("the picked grouping was not among the legal ones")
    return combined


def repl_verdict_all_groupings(
    tmp_path: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    *,
    stdlib: bool = True,
    expected_legal_groupings: LegalGroupings = "ALL",
) -> ReplVerdict:
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


def repl_matrix_verdicts(
    tmp_path: Path,
    modules: dict[str, str],
    header: tuple[str, ...],
    probes: Mapping[K, str],
    *,
    stdlib: bool = True,
    expected_legal_groupings: LegalGroupings = "ALL",
) -> dict[K, ReplVerdict]:
    """Evaluate every *probes* value as its own entry after *header*, over every REPL grouping.

    A pure reparametrization of calling :func:`repl_verdict_all_groupings`
    once per probe on ``(*header, probe)``: enumerating every grouping of
    ``len(header) + 1`` items bijects with choosing how many of *header*'s
    own leading items (``m``) are grouped ahead of a merged final entry
    (*header*'s own remaining tail plus one probe), then a grouping of just
    those ``m`` items. Since a ``check_only`` entry never promotes state,
    the session built for one such ``(m, prefix grouping)`` pair answers
    every probe's own final entry without being rebuilt -- every (probe,
    grouping) combination is still tried, at the identical verdict, only
    the session that answers it is shared across probes.

    Returns, for each key in *probes*, the verdict from whichever grouping
    fully consumes *header* ahead of its own probe (so the probe is always
    that grouping's own final, standalone entry -- the returned span, when
    rejected, is always relative to the probe's own text alone), after
    asserting every other legal grouping reaches the identical verdict for
    that probe, and that the exact set of legal groupings (over
    ``len(header) + 1`` items, as for :func:`repl_verdict_all_groupings`)
    equals *expected_legal_groupings*.
    """
    canonical: dict[K, tuple[type[BaseException] | type[None], str | None, str | None]] = {}
    combined: dict[K, ReplVerdict] = {}
    legal: set[tuple[int, ...]] = set()
    counter = 0

    for m in range(len(header) + 1):
        tail = header[m:]
        for prefix_sizes in all_groupings(m):
            session_dir = tmp_path / str(counter)
            counter += 1
            session_dir.mkdir()
            _write_modules(session_dir, modules)
            session = ReplSession(cwd=session_dir, default_stdlib=stdlib)
            session.open()
            if not eval_setup_entries(session, header[:m], prefix_sizes):
                continue
            sizes = (*prefix_sizes, len(tail) + 1)
            legal.add(sizes)
            for key, probe in probes.items():
                final_text = "\n".join((*tail, probe))
                verdict = _check_only_verdict(session, final_text)
                _phase, cls, span, identity = verdict
                sliced = (
                    final_text[span.start_offset : span.end_offset] if span is not None else None
                )
                current = (cls, sliced, identity)
                if key not in canonical:
                    canonical[key] = current
                else:
                    assert current == canonical[key], (key, sizes, current, canonical[key])
                if m == len(header):
                    combined[key] = verdict

    expected = (
        frozenset(all_groupings(len(header) + 1))
        if expected_legal_groupings == "ALL"
        else expected_legal_groupings
    )
    assert legal == expected, (legal, expected)
    assert set(combined) == set(probes), (set(combined), set(probes))
    return combined


def assert_verdict_everywhere(
    tmp_path: Path,
    modules: dict[str, str],
    decls: tuple[str, ...],
    expected: tuple[FilePhase, type[BaseException] | type[None]],
    *,
    span_text: str | None = None,
    stdlib: bool = True,
    expected_legal_groupings: LegalGroupings = "ALL",
) -> None:
    """Assert *decls* reaches *expected* identically in file mode and every REPL grouping.

    *expected* is a file-mode ``(phase, cls)`` pair; REPL's own phase is
    normalized to ``"accepted"``/``"rejected"`` before comparison (see
    module docstring). When ``expected[0] == "accepted"``, *span_text* must
    be ``None`` and both modes' own *identity* must be set instead;
    otherwise *span_text* is the exact text the raised error's span must
    slice out of *decls*, joined by newline, in both modes.
    """
    src = "\n".join(decls)
    expected_phase, expected_cls = expected
    file_phase, file_cls, file_span, file_identity = file_verdict(
        tmp_path, {"entry": src, **modules}, stdlib=stdlib
    )
    assert (file_phase, file_cls) == (expected_phase, expected_cls)
    repl_phase, repl_cls, repl_span, repl_identity = repl_verdict_all_groupings(
        tmp_path,
        modules,
        decls,
        stdlib=stdlib,
        expected_legal_groupings=expected_legal_groupings,
    )
    assert repl_phase == ("accepted" if expected_phase == "accepted" else "rejected")
    assert repl_cls == expected_cls
    if expected_phase == "accepted":
        assert span_text is None
        assert file_identity is not None
        assert repl_identity is not None
        return
    assert span_text is not None
    assert file_span is not None
    assert src[file_span.start_offset : file_span.end_offset] == span_text
    assert repl_span is not None
    assert src[repl_span.start_offset : repl_span.end_offset] == span_text
