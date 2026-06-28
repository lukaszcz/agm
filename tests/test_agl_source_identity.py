"""Tests for SourceId / source-aware SourceSpan / Diagnostic source_label (Task A).

Covers:
- SourceId dataclass: creation, equality, frozen, label.
- UNKNOWN_SOURCE sentinel: label is "<agl>".
- SourceSpan.source field: defaults to UNKNOWN_SOURCE, compare=False.
- parse_program with source= parameter: AST spans carry the supplied SourceId.
- parse_program_seeded with source= parameter: same behaviour.
- Default (no source=): spans carry UNKNOWN_SOURCE.
- AglSyntaxError stamped with source when parse_program is called with source=.
- Diagnostic.source_label: added field; None by default.
- diagnostic_from_span populates source_label from span.source.label.
- format_diagnostic_location: span-sourced label takes precedence over source_name.
- format_diagnostic: same precedence behaviour.
- Full backward compatibility: existing callers that do not pass source= still work.
"""

from __future__ import annotations

import pytest

from agm.agl.diagnostics import (
    AglError,
    Diagnostic,
    diagnostic_from_span,
    format_diagnostic,
    format_diagnostic_location,
)
from agm.agl.parser import AglSyntaxError, parse_program, parse_program_seeded
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceId, SourceSpan

# ---------------------------------------------------------------------------
# SourceId
# ---------------------------------------------------------------------------


class TestSourceId:
    def test_construction_and_label(self) -> None:
        sid = SourceId(label="/path/to/foo.agl")
        assert sid.label == "/path/to/foo.agl"

    def test_frozen(self) -> None:
        sid = SourceId(label="x")
        with pytest.raises((AttributeError, TypeError)):
            setattr(sid, "label", "y")

    def test_equality(self) -> None:
        assert SourceId(label="a") == SourceId(label="a")
        assert SourceId(label="a") != SourceId(label="b")

    def test_hashable(self) -> None:
        s = {SourceId(label="a"), SourceId(label="b"), SourceId(label="a")}
        assert len(s) == 2

    def test_repr_contains_label(self) -> None:
        sid = SourceId(label="myfile.agl")
        assert "myfile.agl" in repr(sid)


class TestUnknownSource:
    def test_sentinel_label(self) -> None:
        assert UNKNOWN_SOURCE.label == "<agl>"

    def test_is_source_id(self) -> None:
        assert isinstance(UNKNOWN_SOURCE, SourceId)

    def test_sentinel_equality(self) -> None:
        assert UNKNOWN_SOURCE == SourceId(label="<agl>")


# ---------------------------------------------------------------------------
# SourceSpan.source field
# ---------------------------------------------------------------------------


class TestSourceSpanSourceField:
    def _make_span(self, **kwargs: object) -> SourceSpan:
        defaults: dict[str, int] = {
            "start_line": 1, "start_col": 1, "end_line": 1, "end_col": 2,
            "start_offset": 0, "end_offset": 1,
        }
        defaults.update(kwargs)  # type: ignore[arg-type]
        return SourceSpan(**defaults)  # type: ignore[arg-type]

    def test_default_source_is_unknown(self) -> None:
        span = self._make_span()
        assert span.source is UNKNOWN_SOURCE

    def test_custom_source(self) -> None:
        sid = SourceId(label="/foo/bar.agl")
        span = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2,
            start_offset=0, end_offset=1,
            source=sid,
        )
        assert span.source == sid

    def test_source_compare_false_same_positions(self) -> None:
        """SourceSpans with different sources but same positions compare equal."""
        sid_a = SourceId(label="a.agl")
        sid_b = SourceId(label="b.agl")
        span_a = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2,
            start_offset=0, end_offset=1,
            source=sid_a,
        )
        span_b = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2,
            start_offset=0, end_offset=1,
            source=sid_b,
        )
        assert span_a == span_b

    def test_source_excluded_from_hash(self) -> None:
        """Spans with different sources but same positions have the same hash."""
        sid_a = SourceId(label="a.agl")
        sid_b = SourceId(label="b.agl")
        span_a = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2,
            start_offset=0, end_offset=1,
            source=sid_a,
        )
        span_b = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2,
            start_offset=0, end_offset=1,
            source=sid_b,
        )
        assert hash(span_a) == hash(span_b)


# ---------------------------------------------------------------------------
# parse_program / parse_program_seeded with source= parameter
# ---------------------------------------------------------------------------


class TestParseWithSource:
    def test_parse_program_no_source_gives_unknown(self) -> None:
        """Spans default to UNKNOWN_SOURCE when source= is omitted."""
        prog = parse_program("let x = 1")
        assert prog.span.source is UNKNOWN_SOURCE

    def test_parse_program_with_source_stamps_span(self) -> None:
        """parse_program(source=sid) stamps spans with sid."""
        sid = SourceId(label="/path/to/myfile.agl")
        prog = parse_program("let x = 1", source=sid)
        assert prog.span.source == sid

    def test_parse_program_with_source_stamps_child_spans(self) -> None:
        """Child spans also carry the supplied SourceId."""
        sid = SourceId(label="child_test.agl")
        prog = parse_program("let x = 1\nlet y = 2", source=sid)
        # Program body is a Block; its items are LetDecls.
        for item in prog.body.items:
            assert item.span.source == sid

    def test_parse_program_seeded_with_source_stamps_span(self) -> None:
        """parse_program_seeded(source=sid) stamps spans too."""
        sid = SourceId(label="seeded.agl")
        prog, _next = parse_program_seeded("let a = 42", start_id=0, source=sid)
        assert prog.span.source == sid

    def test_parse_program_seeded_no_source_gives_unknown(self) -> None:
        """parse_program_seeded without source= defaults to UNKNOWN_SOURCE."""
        prog, _next = parse_program_seeded("let a = 42", start_id=0)
        assert prog.span.source is UNKNOWN_SOURCE

    def test_source_none_gives_unknown(self) -> None:
        """Explicitly passing source=None uses UNKNOWN_SOURCE."""
        prog = parse_program("let x = 1", source=None)
        assert prog.span.source is UNKNOWN_SOURCE

    def test_source_does_not_affect_node_ids(self) -> None:
        """The source parameter must not change node id assignment."""
        sid = SourceId(label="test.agl")
        prog_with = parse_program("let x = 1", source=sid)
        prog_without = parse_program("let x = 1")
        assert prog_with.node_id == prog_without.node_id

    def test_source_repl_label(self) -> None:
        """REPL-style <repl> label is propagated correctly."""
        sid = SourceId(label="<repl>")
        prog = parse_program("1 + 2", source=sid)
        assert prog.span.source.label == "<repl>"

    def test_source_command_label(self) -> None:
        """<command> label used for exec -c is propagated correctly."""
        sid = SourceId(label="<command>")
        prog = parse_program("1 + 2", source=sid)
        assert prog.span.source.label == "<command>"


# ---------------------------------------------------------------------------
# AglSyntaxError stamped with source
# ---------------------------------------------------------------------------


class TestAglSyntaxErrorWithSource:
    def test_syntax_error_without_source_has_unknown(self) -> None:
        """Without source=, a parse error's span has UNKNOWN_SOURCE."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("x = y")
        err = exc_info.value
        assert err.span is not None
        assert err.span.source is UNKNOWN_SOURCE

    def test_syntax_error_with_source_has_label(self) -> None:
        """With source=sid, a parse error's span is stamped with sid."""
        sid = SourceId(label="/modules/broken.agl")
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("x = y", source=sid)
        err = exc_info.value
        assert err.span is not None
        assert err.span.source == sid

    def test_lex_error_with_source_has_label(self) -> None:
        """Lex-level errors (bad characters) also carry the source id."""
        sid = SourceId(label="/modules/broken.agl")
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let x = \x00oops", source=sid)
        err = exc_info.value
        assert err.span is not None
        assert err.span.source == sid

    def test_transformer_error_with_source_has_label(self) -> None:
        """AglSyntaxErrors raised inside the transformer carry the source id.

        Regression: errors raised in the transformer (e.g. "list[] arg count")
        were unwrapped from VisitError and re-raised WITHOUT the source stamp,
        so they reported ``<agl>`` instead of the module path.
        """
        sid = SourceId(label="/mods/broken.agl")
        with pytest.raises(AglSyntaxError) as exc_info:
            # "list[int, text]" triggers an arg-count error inside the
            # AstBuilder transformer, wrapped in a VisitError.
            parse_program("def f() -> list[int, text] = 1", source=sid)
        err = exc_info.value
        assert err.span is not None
        assert err.span.source.label == "/mods/broken.agl"


# ---------------------------------------------------------------------------
# Diagnostic.source_label and format_diagnostic_location / format_diagnostic
# ---------------------------------------------------------------------------


class TestDiagnosticSourceLabel:
    def test_diagnostic_default_source_label_is_none(self) -> None:
        d = Diagnostic(message="oops", line=1)
        assert d.source_label is None

    def test_diagnostic_explicit_source_label(self) -> None:
        d = Diagnostic(message="oops", line=1, source_label="/foo/bar.agl")
        assert d.source_label == "/foo/bar.agl"

    def test_diagnostic_from_span_populates_source_label(self) -> None:
        sid = SourceId(label="/src/mymodule.agl")
        span = SourceSpan(
            start_line=3, start_col=5, end_line=3, end_col=10,
            start_offset=20, end_offset=25,
            source=sid,
        )
        d = diagnostic_from_span("something broke", span)
        assert d.source_label == "/src/mymodule.agl"

    def test_diagnostic_from_span_unknown_source_gives_none_label(self) -> None:
        """UNKNOWN_SOURCE yields None source_label (falls back to caller's source_name)."""
        span = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2,
            start_offset=0, end_offset=1,
        )
        d = diagnostic_from_span("err", span)
        assert d.source_label is None

    def test_format_diagnostic_location_uses_span_label(self) -> None:
        """When source_label is set on the diagnostic, it overrides source_name."""
        d = Diagnostic(message="oops", line=5, column=3, source_label="/foo/bar.agl")
        loc = format_diagnostic_location(d, source_name="<agl>")
        assert loc == "/foo/bar.agl:5:3"

    def test_format_diagnostic_location_no_source_label_uses_source_name(self) -> None:
        """Without source_label, the source_name argument is used (backward compat)."""
        d = Diagnostic(message="oops", line=5, column=3)
        loc = format_diagnostic_location(d, source_name="<agl>")
        assert loc == "<agl>:5:3"

    def test_format_diagnostic_location_source_label_overrides_none_source_name(self) -> None:
        """Even if source_name=None, source_label still wins."""
        d = Diagnostic(message="oops", line=5, column=3, source_label="/foo/bar.agl")
        loc = format_diagnostic_location(d, source_name=None)
        assert loc == "/foo/bar.agl:5:3"

    def test_format_diagnostic_uses_span_label(self) -> None:
        """format_diagnostic uses span-sourced label."""
        d = Diagnostic(
            message="some error", line=3, column=5, end_line=3, end_column=9,
            source_label="/path/to/file.agl",
        )
        result = format_diagnostic(d)
        assert result == "/path/to/file.agl:3:5-8: error: some error"

    def test_format_diagnostic_backward_compat_no_source_label(self) -> None:
        """Without source_label, format_diagnostic keeps existing <agl> behavior."""
        d = Diagnostic(message="some error", line=3, column=5, end_line=3, end_column=9)
        result = format_diagnostic(d)
        assert result == "<agl>:3:5-8: error: some error"

    def test_format_diagnostic_location_source_name_none_no_label(self) -> None:
        """source_name=None with no source_label still omits the prefix."""
        d = Diagnostic(message="err", line=2, column=1)
        loc = format_diagnostic_location(d, source_name=None)
        assert loc == "2:1"

    def test_format_diagnostic_location_range_with_file_label(self) -> None:
        """Range formatting works correctly with a file-sourced label."""
        d = Diagnostic(
            message="err", line=3, column=5, end_line=3, end_column=9,
            source_label="/a/b.agl",
        )
        assert format_diagnostic_location(d) == "/a/b.agl:3:5-8"

    def test_diagnostic_from_span_then_format(self) -> None:
        """Round-trip: parse → span → diagnostic_from_span → format."""
        sid = SourceId(label="/project/src/utils.agl")
        prog = parse_program("let x = 1", source=sid)
        diag = diagnostic_from_span("test message", prog.span)
        result = format_diagnostic(diag)
        assert "/project/src/utils.agl:1" in result
        assert "test message" in result


# ---------------------------------------------------------------------------
# Backward compatibility: callers that never pass source= still work
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_span_without_source_arg_still_works(self) -> None:
        """Constructing SourceSpan without source= keeps working."""
        span = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2,
            start_offset=0, end_offset=1,
        )
        assert span.start_line == 1

    def test_diagnostic_without_source_label_still_works(self) -> None:
        """Constructing Diagnostic without source_label keeps working."""
        d = Diagnostic(message="err", line=3)
        assert d.message == "err"
        assert d.line == 3

    def test_format_diagnostic_default_source_name_unchanged(self) -> None:
        """format_diagnostic with default source_name='<agl>' unchanged."""
        d = Diagnostic(message="some error", line=3, column=5, end_line=3, end_column=9)
        assert format_diagnostic(d) == "<agl>:3:5-8: error: some error"

    def test_agl_error_to_diagnostic_backward_compat(self) -> None:
        """AglError.to_diagnostic() still produces a valid diagnostic."""
        span = SourceSpan(
            start_line=2, start_col=1, end_line=2, end_col=5,
            start_offset=10, end_offset=14,
        )
        err = AglError("test error", span=span)
        d = err.to_diagnostic()
        assert d.line == 2
        assert d.column == 1

    def test_parse_program_no_source_backward_compat(self) -> None:
        """parse_program without source= still returns a valid Program."""
        prog = parse_program("let x = 1")
        assert prog.span.start_line == 1
