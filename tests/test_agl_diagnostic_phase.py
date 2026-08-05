"""Diagnostics are tagged with the frontend pass that produced them.

``Diagnostic.phase`` lets a caller classify a failure programmatically (e.g.
distinguish a name-resolution failure from a type error) instead of matching
on ``message`` text. This suite drives the real pipeline and REPL entry path
rather than constructing pipeline exceptions by hand, and confirms the tag is
purely a classification aid: it never changes rendered diagnostic output.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

from agm.agl import PipelineDriver
from agm.agl.diagnostics import (
    Diagnostic,
    DiagnosticPhase,
    RelatedDiagnostic,
    SourceSpan,
    diagnostic_from_span,
    format_diagnostic,
)
from agm.agl.repl.session import ReplSession
from agm.agl.scope import AglScopeError
from agm.agl.syntax.spans import SourceId
from agm.agl.typecheck import AglTypeError

_UNDEFINED_NAME_SOURCE = "some_totally_undefined_name\n"
_TYPE_MISMATCH_SOURCE = 'let x: int = "not an int"\nx\n'
_SYNTAX_ERROR_SOURCE = "let x = (\n"
_TEXT_PARSE_POLICY_SOURCE = 'let response: text = ask("Q", on_parse_error = Abort())\n'
_LEADING_TAB_SOURCE = "\tprint\n"


def _no_agents(_request: object) -> str:
    return ""


def test_undefined_name_diagnostic_is_phased_scope() -> None:
    result = PipelineDriver(agent_dispatcher=_no_agents).run(_UNDEFINED_NAME_SOURCE)

    assert not result.ok
    assert [d.phase for d in result.diagnostics] == [DiagnosticPhase.SCOPE]


def test_type_mismatch_diagnostic_is_phased_typecheck() -> None:
    result = PipelineDriver(agent_dispatcher=_no_agents).run(_TYPE_MISMATCH_SOURCE)

    assert not result.ok
    assert [d.phase for d in result.diagnostics] == [DiagnosticPhase.TYPECHECK]


def test_syntax_error_diagnostic_has_no_phase() -> None:
    result = PipelineDriver(agent_dispatcher=_no_agents).run(_SYNTAX_ERROR_SOURCE)

    assert not result.ok
    assert [d.phase for d in result.diagnostics] == [None]


def test_prepare_and_discover_params_surface_the_same_scope_phase() -> None:
    runtime = PipelineDriver(agent_dispatcher=_no_agents)
    prepared = runtime.prepare_program(_UNDEFINED_NAME_SOURCE)
    discovery = runtime.discover_params(prepared)

    assert [d.phase for d in prepared.diagnostics] == [DiagnosticPhase.SCOPE]
    assert [d.phase for d in discovery.diagnostics] == [DiagnosticPhase.SCOPE]


def test_repl_undefined_name_entry_is_phased_scope() -> None:
    result = ReplSession(agent_dispatcher=_no_agents).eval_entry(_UNDEFINED_NAME_SOURCE)

    assert not result.ok
    assert [d.phase for d in result.diagnostics] == [DiagnosticPhase.SCOPE]


def test_repl_type_mismatch_entry_is_phased_typecheck() -> None:
    result = ReplSession(agent_dispatcher=_no_agents).eval_entry(_TYPE_MISMATCH_SOURCE)

    assert not result.ok
    assert [d.phase for d in result.diagnostics] == [DiagnosticPhase.TYPECHECK]


def test_repl_syntax_error_entry_has_no_phase() -> None:
    result = ReplSession(agent_dispatcher=_no_agents).eval_entry(_SYNTAX_ERROR_SOURCE)

    assert not result.ok
    assert [d.phase for d in result.diagnostics] == [None]


# ---------------------------------------------------------------------------
# Warnings are classified by producing pass too
# ---------------------------------------------------------------------------


def test_checker_warning_is_phased_typecheck() -> None:
    """A warning raised by the typecheck pass carries that pass's tag."""
    result = PipelineDriver(agent_dispatcher=_no_agents).run(
        _TEXT_PARSE_POLICY_SOURCE, check_only=True
    )

    assert [w.phase for w in result.warnings] == [DiagnosticPhase.TYPECHECK]


def test_lexer_warning_has_no_phase() -> None:
    """A pass without a dedicated tag leaves the classification unset."""
    result = PipelineDriver(agent_dispatcher=_no_agents).run(_LEADING_TAB_SOURCE)

    assert [w.phase for w in result.warnings] == [None]


# ---------------------------------------------------------------------------
# Non-``AglError`` escapes from a pass are classified by the catching boundary
# ---------------------------------------------------------------------------


def test_unexpected_scope_pass_failure_is_phased_scope() -> None:
    with patch("agm.agl.scope.program.resolve_program", side_effect=RuntimeError("boom")):
        prepared = PipelineDriver.prepare_program("let x = 1\nx\n")

    assert [d.phase for d in prepared.diagnostics] == [DiagnosticPhase.SCOPE]


def test_unexpected_typecheck_pass_failure_is_phased_typecheck() -> None:
    runtime = PipelineDriver(agent_dispatcher=_no_agents)
    prepared = runtime.prepare_program("let x = 1\nx\n")
    with patch("agm.agl.typecheck.program.check_program", side_effect=RuntimeError("boom")):
        discovery = runtime.discover_params(prepared)

    assert [d.phase for d in discovery.diagnostics] == [DiagnosticPhase.TYPECHECK]


# ---------------------------------------------------------------------------
# ``AglError.to_diagnostic`` stamps on both of its construction paths
# ---------------------------------------------------------------------------


def test_span_less_error_still_carries_its_phase() -> None:
    diagnostic = AglScopeError("boom").to_diagnostic()

    assert diagnostic.phase is DiagnosticPhase.SCOPE
    assert (diagnostic.message, diagnostic.line) == ("boom", 1)


def test_spanned_error_carries_its_phase_and_keeps_every_span_derived_field() -> None:
    span = SourceSpan(2, 4, 2, 9, 10, 15, source=SourceId("/tmp/example.agl"))
    note_span = SourceSpan(7, 1, 7, 3, 40, 42, source=SourceId("/tmp/other.agl"))

    diagnostic = AglTypeError("boom", span=span, related=(("because", note_span),)).to_diagnostic()

    assert diagnostic.phase is DiagnosticPhase.TYPECHECK
    assert diagnostic.related == (RelatedDiagnostic("because", 7, 1, 7, 3, "/tmp/other.agl"),)
    # Stamping the phase and the related notes must leave every other field
    # exactly as the span-derived diagnostic produced it.
    stripped = dataclasses.replace(diagnostic, related=(), phase=None)
    assert stripped == diagnostic_from_span("boom", span)


# ---------------------------------------------------------------------------
# The tag is never rendered
# ---------------------------------------------------------------------------

_RENDERABLE_SHAPES: tuple[Diagnostic, ...] = (
    Diagnostic(message="boom", line=3),
    Diagnostic(message="boom", line=3, column=5),
    Diagnostic(message="boom", line=3, column=5, end_line=3, end_column=9),
    Diagnostic(message="boom", line=3, column=5, end_line=6, end_column=2),
    Diagnostic(message="boom", line=3, column=5, severity="warning"),
    Diagnostic(message="boom", line=3, column=5, source_label="/tmp/example.agl"),
    Diagnostic(
        message="boom",
        line=3,
        column=5,
        end_line=3,
        end_column=9,
        source_label="/tmp/example.agl",
        related=(
            RelatedDiagnostic("because", 7, 1, 7, 3, "/tmp/other.agl"),
            RelatedDiagnostic("and also", 9, 2),
        ),
    ),
)


@pytest.mark.parametrize("unphased", _RENDERABLE_SHAPES, ids=range(len(_RENDERABLE_SHAPES)))
@pytest.mark.parametrize("phase", list(DiagnosticPhase))
def test_format_diagnostic_output_is_unaffected_by_phase(
    unphased: Diagnostic, phase: DiagnosticPhase
) -> None:
    """No rendering branch — location, severity, label, notes — reveals the tag."""
    phased = dataclasses.replace(unphased, phase=phase)

    for source_name in ("<agl>", None):
        rendered = format_diagnostic(phased, source_name=source_name)
        assert rendered == format_diagnostic(unphased, source_name=source_name)
        assert phase.value not in rendered
