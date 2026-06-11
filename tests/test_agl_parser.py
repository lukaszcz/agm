"""Tests for the AgL parser (agm.agl.parser) — Component 2.

Covers:
- LALR(1) conflict-guard: zero shift/reduce and reduce/reduce conflicts.
- Parsing every M1 construct to the expected AST shape.
- Inline semicolons vs newlines produce identical ASTs.
- Template strings: plain text, single/multiple interpolations, renderer.
- Agent calls: with/without call_options, every option type.
- Type annotations: all primitives, list[T], dict[text, T], named types.
- Input declarations with and without annotation.
- Error cases: == produces friendly AglSyntaxError; malformed inputs raise
  AglSyntaxError (not bare lark exceptions).

NOTE: This file must NOT modify tests/test_agl_e2e.py or tests/agl/.
      No static-analysis suppression comments in this file.
"""

from __future__ import annotations

import decimal
import importlib.resources
import logging
import logging.handlers

import pytest

from agm.agl.lexer.lexer import AglLexer
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.syntax import (
    AbortPolicy,
    AgentCall,
    BoolLit,
    CallOptions,
    DecimalLit,
    ExprStmt,
    InputDecl,
    InterpSegment,
    IntLit,
    LetDecl,
    NullLit,
    PassStmt,
    PrintStmt,
    Program,
    RetryPolicy,
    SetStmt,
    Template,
    TextSegment,
    VarDecl,
    VarRef,
)
from agm.agl.syntax.types import (
    BoolT,
    DecimalT,
    DictT,
    IntT,
    JsonT,
    ListT,
    NameT,
    TextT,
)

# ---------------------------------------------------------------------------
# LALR(1) conflict guard
#
# Build the Lark parser with DEBUG logging enabled and assert that no
# "Shift/Reduce" conflicts appear.  This is a mandatory regression test:
# any accidental grammar ambiguity will cause this test to fail before the
# first developer notices a parse anomaly.
# ---------------------------------------------------------------------------


class TestConflictGuard:
    """Verify that the M1 grammar has zero LALR conflicts at construction time."""

    def test_no_shift_reduce_conflicts(self) -> None:
        """Grammar must have zero shift/reduce and reduce/reduce conflicts."""
        import lark

        handler = logging.handlers.MemoryHandler(capacity=10000)
        lark.logger.addHandler(handler)
        old_level = lark.logger.level
        lark.logger.setLevel(logging.DEBUG)
        try:
            grammar_text = (
                importlib.resources.files("agm.agl")
                .joinpath("grammar/agl.lark")
                .read_text(encoding="utf-8")
            )
            lark.Lark(
                grammar_text,
                parser="lalr",
                lexer=AglLexer,
                propagate_positions=True,
                maybe_placeholders=True,
            )
        finally:
            lark.logger.setLevel(old_level)
            lark.logger.removeHandler(handler)

        records = handler.buffer
        conflicts = [
            r
            for r in records
            if "Shift/Reduce" in r.getMessage() or "Reduce/Reduce" in r.getMessage()
        ]
        conflict_msgs = "\n".join(r.getMessage() for r in conflicts)
        assert conflicts == [], f"LALR conflicts detected:\n{conflict_msgs}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_one(src: str) -> object:
    """Parse a single-statement program and return that statement."""
    prog = parse_program(src)
    assert isinstance(prog, Program)
    assert len(prog.body) == 1
    return prog.body[0]


# ---------------------------------------------------------------------------
# PassStmt
# ---------------------------------------------------------------------------


class TestPassStmt:
    def test_parse_pass(self) -> None:
        stmt = _parse_one("pass")
        assert isinstance(stmt, PassStmt)

    def test_pass_span_starts_at_1_1(self) -> None:
        stmt = _parse_one("pass")
        assert isinstance(stmt, PassStmt)
        assert stmt.span.start_line == 1
        assert stmt.span.start_col == 1


# ---------------------------------------------------------------------------
# LetDecl
# ---------------------------------------------------------------------------


class TestLetDecl:
    def test_let_int(self) -> None:
        stmt = _parse_one("let x = 42")
        assert isinstance(stmt, LetDecl)
        assert stmt.name == "x"
        assert stmt.type_ann is None
        assert isinstance(stmt.value, IntLit)
        assert stmt.value.value == 42

    def test_let_decimal(self) -> None:
        stmt = _parse_one("let x = 3.14")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, DecimalLit)
        assert stmt.value.value == decimal.Decimal("3.14")

    def test_let_true(self) -> None:
        stmt = _parse_one("let x = true")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BoolLit)
        assert stmt.value.value is True

    def test_let_false(self) -> None:
        stmt = _parse_one("let x = false")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BoolLit)
        assert stmt.value.value is False

    def test_let_null(self) -> None:
        stmt = _parse_one("let x = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, NullLit)

    def test_let_var_ref(self) -> None:
        stmt = _parse_one("let y = x")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, VarRef)
        assert stmt.value.name == "x"

    def test_let_paren_expr(self) -> None:
        stmt = _parse_one("let x = (42)")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, IntLit)
        assert stmt.value.value == 42

    def test_let_with_type_ann_text(self) -> None:
        stmt = _parse_one("let x: text = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, TextT)

    def test_let_with_type_ann_json(self) -> None:
        stmt = _parse_one("let x: json = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, JsonT)

    def test_let_with_type_ann_bool(self) -> None:
        stmt = _parse_one("let x: bool = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, BoolT)

    def test_let_with_type_ann_int(self) -> None:
        stmt = _parse_one("let x: int = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, IntT)

    def test_let_with_type_ann_decimal(self) -> None:
        stmt = _parse_one("let x: decimal = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, DecimalT)

    def test_let_with_list_type(self) -> None:
        stmt = _parse_one("let x: list[text] = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, ListT)
        assert isinstance(stmt.type_ann.elem, TextT)

    def test_let_with_dict_type(self) -> None:
        stmt = _parse_one("let x: dict[text, json] = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, DictT)
        assert isinstance(stmt.type_ann.value, JsonT)

    def test_let_with_named_type(self) -> None:
        stmt = _parse_one("let x: MyRecord = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, NameT)
        assert stmt.type_ann.name == "MyRecord"


# ---------------------------------------------------------------------------
# VarDecl
# ---------------------------------------------------------------------------


class TestVarDecl:
    def test_var_int(self) -> None:
        stmt = _parse_one("var x = 1")
        assert isinstance(stmt, VarDecl)
        assert stmt.name == "x"
        assert isinstance(stmt.value, IntLit)
        assert stmt.value.value == 1

    def test_var_with_type_ann(self) -> None:
        stmt = _parse_one("var x: text = null")
        assert isinstance(stmt, VarDecl)
        assert isinstance(stmt.type_ann, TextT)


# ---------------------------------------------------------------------------
# SetStmt
# ---------------------------------------------------------------------------


class TestSetStmt:
    def test_set_stmt(self) -> None:
        stmt = _parse_one("set x = 2")
        assert isinstance(stmt, SetStmt)
        assert stmt.target == "x"
        assert isinstance(stmt.value, IntLit)
        assert stmt.value.value == 2


# ---------------------------------------------------------------------------
# InputDecl
# ---------------------------------------------------------------------------


class TestInputDecl:
    def test_input_no_annotation(self) -> None:
        stmt = _parse_one("input name")
        assert isinstance(stmt, InputDecl)
        assert stmt.name == "name"
        assert stmt.annotation is None

    def test_input_with_text_ann(self) -> None:
        stmt = _parse_one("input spec: text")
        assert isinstance(stmt, InputDecl)
        assert stmt.name == "spec"
        assert isinstance(stmt.annotation, TextT)

    def test_input_with_json_ann(self) -> None:
        stmt = _parse_one("input data: json")
        assert isinstance(stmt, InputDecl)
        assert isinstance(stmt.annotation, JsonT)

    def test_input_with_named_type(self) -> None:
        stmt = _parse_one("input spec: MyType")
        assert isinstance(stmt, InputDecl)
        assert isinstance(stmt.annotation, NameT)
        assert stmt.annotation.name == "MyType"

    def test_input_with_list_type(self) -> None:
        stmt = _parse_one("input items: list[text]")
        assert isinstance(stmt, InputDecl)
        assert isinstance(stmt.annotation, ListT)
        assert isinstance(stmt.annotation.elem, TextT)


# ---------------------------------------------------------------------------
# PrintStmt
# ---------------------------------------------------------------------------


class TestPrintStmt:
    def test_print_var_ref(self) -> None:
        stmt = _parse_one("print x")
        assert isinstance(stmt, PrintStmt)
        assert isinstance(stmt.value, VarRef)
        assert stmt.value.name == "x"

    def test_print_int(self) -> None:
        stmt = _parse_one("print 42")
        assert isinstance(stmt, PrintStmt)
        assert isinstance(stmt.value, IntLit)


# ---------------------------------------------------------------------------
# ExprStmt / AgentCall
# ---------------------------------------------------------------------------


class TestAgentCall:
    def test_agent_call_no_options(self) -> None:
        stmt = _parse_one('prompt "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.agent == "prompt"
        assert isinstance(call.options, CallOptions)
        assert call.options.format is None
        assert call.options.strict_json is None
        assert call.options.parse_policy is None

    def test_agent_call_format_json(self) -> None:
        stmt = _parse_one('prompt[format: json] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "json"

    def test_agent_call_format_text(self) -> None:
        stmt = _parse_one('prompt[format: text] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "text"

    def test_agent_call_strict_json_true(self) -> None:
        stmt = _parse_one('prompt[strict_json: true] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.strict_json is True

    def test_agent_call_strict_json_false(self) -> None:
        stmt = _parse_one('prompt[strict_json: false] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.strict_json is False

    def test_agent_call_on_parse_error_abort(self) -> None:
        stmt = _parse_one('prompt[on_parse_error: abort] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert isinstance(call.options.parse_policy, AbortPolicy)

    def test_agent_call_on_parse_error_retry(self) -> None:
        stmt = _parse_one('prompt[on_parse_error: retry[3]] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert isinstance(call.options.parse_policy, RetryPolicy)
        assert call.options.parse_policy.extra == 3

    def test_agent_call_multiple_options(self) -> None:
        stmt = _parse_one('prompt[format: json, strict_json: true] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "json"
        assert call.options.strict_json is True

    def test_agent_call_trailing_comma_in_options(self) -> None:
        stmt = _parse_one('prompt[format: json,] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "json"

    def test_duplicate_format_option_rejected(self) -> None:
        src = 'prompt[format: json, format: text] "hello"'
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "duplicate option 'format'" in str(err)
        # Span must point at the SECOND (duplicate) occurrence.
        span = err.source_span
        dup_col = src.index("format", src.index("format") + 1) + 1
        assert span.start_line == 1
        assert span.start_col == dup_col

    def test_duplicate_strict_json_option_rejected(self) -> None:
        src = 'prompt[strict_json: true, strict_json: false] "hello"'
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "duplicate option 'strict_json'" in str(err)
        dup_col = src.index("strict_json", src.index("strict_json") + 1) + 1
        assert err.source_span.start_col == dup_col

    def test_duplicate_on_parse_error_option_rejected(self) -> None:
        src = 'prompt[on_parse_error: abort, on_parse_error: retry[2]] "hello"'
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "duplicate option 'on_parse_error'" in str(err)
        dup_col = src.index("on_parse_error", src.index("on_parse_error") + 1) + 1
        assert err.source_span.start_col == dup_col

    def test_distinct_options_twin_parses(self) -> None:
        # Accept-twin: distinct option keys parse fine even with three options.
        stmt = _parse_one(
            'prompt[format: json, strict_json: true, on_parse_error: abort] "hi"'
        )
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "json"
        assert call.options.strict_json is True
        assert isinstance(call.options.parse_policy, AbortPolicy)


# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------


class TestTemplate:
    def test_plain_text(self) -> None:
        stmt = _parse_one('prompt "hello world"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        tmpl = call.template
        assert isinstance(tmpl, Template)
        # One TextSegment for the content
        text_segs = [s for s in tmpl.segments if isinstance(s, TextSegment)]
        assert any(s.text == "hello world" for s in text_segs)

    def test_single_interpolation(self) -> None:
        stmt = _parse_one('prompt "hello ${x}"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        tmpl = call.template
        interps = [s for s in tmpl.segments if isinstance(s, InterpSegment)]
        assert len(interps) == 1
        assert isinstance(interps[0].expr, VarRef)
        assert interps[0].expr.name == "x"
        assert interps[0].render is None

    def test_interp_with_renderer(self) -> None:
        stmt = _parse_one('prompt "hello ${x as raw}"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        tmpl = call.template
        interps = [s for s in tmpl.segments if isinstance(s, InterpSegment)]
        assert len(interps) == 1
        assert interps[0].render == "raw"

    def test_multiple_interpolations(self) -> None:
        stmt = _parse_one('prompt "hello ${x} and ${y as raw}"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        tmpl = call.template
        interps = [s for s in tmpl.segments if isinstance(s, InterpSegment)]
        assert len(interps) == 2
        assert interps[0].render is None
        assert interps[1].render == "raw"

    def test_template_text_segment_content(self) -> None:
        stmt = _parse_one('prompt "hello ${x} world"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        tmpl = call.template
        text_segs = [s for s in tmpl.segments if isinstance(s, TextSegment)]
        assert any("hello" in s.text for s in text_segs)
        assert any("world" in s.text for s in text_segs)

    def test_adjacent_interps_have_no_text_segments(self) -> None:
        # "${a}${b}" yields exactly two InterpSegments and no TextSegments:
        # empty fragments between/around interps carry no information and are
        # normalized away.
        stmt = _parse_one('prompt "${a}${b}"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        segs = call.template.segments
        assert all(isinstance(s, InterpSegment) for s in segs)
        assert len(segs) == 2
        assert isinstance(segs[0], InterpSegment)
        assert isinstance(segs[1], InterpSegment)
        assert isinstance(segs[0].expr, VarRef)
        assert segs[0].expr.name == "a"
        assert isinstance(segs[1].expr, VarRef)
        assert segs[1].expr.name == "b"

    def test_leading_text_then_interp(self) -> None:
        # "x${a}" yields Text("x") followed by an InterpSegment; no empty
        # trailing TextSegment.
        stmt = _parse_one('prompt "x${a}"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        segs = call.template.segments
        assert len(segs) == 2
        assert isinstance(segs[0], TextSegment)
        assert segs[0].text == "x"
        assert isinstance(segs[1], InterpSegment)

    def test_no_empty_text_segments_in_any_template(self) -> None:
        # Invariant: a template never contains an empty TextSegment.
        stmt = _parse_one('prompt "${a} mid ${b}"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        for s in call.template.segments:
            if isinstance(s, TextSegment):
                assert s.text != ""

    def test_interp_expr_can_be_let_value(self) -> None:
        # Interpolation can reference any expression
        prog = parse_program("let x = 42")
        stmt = prog.body[0]
        assert isinstance(stmt, LetDecl)
        # And a template using that var
        prog2 = parse_program('prompt "${x}"')
        stmt2 = prog2.body[0]
        assert isinstance(stmt2, ExprStmt)


# ---------------------------------------------------------------------------
# Multi-statement programs
# ---------------------------------------------------------------------------


class TestMultiStatement:
    def test_semicolon_separator(self) -> None:
        prog = parse_program("let x = 1; let y = 2")
        assert isinstance(prog, Program)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], LetDecl)
        assert isinstance(prog.body[1], LetDecl)

    def test_newline_separator(self) -> None:
        prog = parse_program("let x = 1\nlet y = 2")
        assert isinstance(prog, Program)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], LetDecl)
        assert isinstance(prog.body[1], LetDecl)

    def test_semicolon_and_newline_same_ast(self) -> None:
        prog_semi = parse_program("let x = 1; let y = 2")
        prog_newline = parse_program("let x = 1\nlet y = 2")
        # Compare by structure, not span/node_id (both use compare=False)
        assert prog_semi.body[0] == prog_newline.body[0]
        assert prog_semi.body[1] == prog_newline.body[1]

    def test_trailing_semicolon_ok(self) -> None:
        prog = parse_program("let x = 1;")
        assert isinstance(prog, Program)
        assert len(prog.body) == 1

    def test_trailing_newline_ok(self) -> None:
        prog = parse_program("let x = 1\n")
        assert isinstance(prog, Program)
        assert len(prog.body) == 1

    def test_three_statements(self) -> None:
        prog = parse_program("input x\nlet y = x\nprint y")
        assert len(prog.body) == 3
        assert isinstance(prog.body[0], InputDecl)
        assert isinstance(prog.body[1], LetDecl)
        assert isinstance(prog.body[2], PrintStmt)


# ---------------------------------------------------------------------------
# Source spans
# ---------------------------------------------------------------------------


class TestSourceSpans:
    def test_span_is_1_based(self) -> None:
        stmt = _parse_one("pass")
        assert isinstance(stmt, PassStmt)
        assert stmt.span.start_line == 1
        assert stmt.span.start_col == 1

    def test_offset_is_0_based(self) -> None:
        stmt = _parse_one("pass")
        assert isinstance(stmt, PassStmt)
        assert stmt.span.start_offset == 0

    def test_let_span_covers_full_statement(self) -> None:
        stmt = _parse_one("let x = 42")
        assert isinstance(stmt, LetDecl)
        # Statement starts at col 1 and ends after '42'
        assert stmt.span.start_col == 1
        assert stmt.span.end_col > stmt.span.start_col


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_eq_eq_produces_friendly_error(self) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let x == 1")
        assert "Use `=` for equality." in str(exc_info.value)

    def test_syntax_error_message_for_eq_eq(self) -> None:
        """The == error message must contain the phrase 'Use `=` for equality.'"""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let x == 1")
        assert "Use `=` for equality." in str(exc_info.value)

    def test_syntax_error_has_span(self) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let x == 1")
        err = exc_info.value
        assert hasattr(err, "source_span")
        span = err.source_span
        assert span.start_line >= 1

    def test_bare_lark_error_not_raised(self) -> None:
        """Parser must wrap all Lark errors into AglSyntaxError."""
        with pytest.raises(AglSyntaxError):
            parse_program("let = 1")  # missing name

    def test_unknown_token_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("@@@")

    def test_unterminated_template_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('prompt "unterminated')

    @pytest.mark.parametrize(
        "garbage",
        [")", "]", "= = =", "let let let", "[[[", "print print"],
    )
    def test_garbage_inputs_wrap_into_agl_syntax_error(self, garbage: str) -> None:
        # No raw lark exception may escape parse_program for malformed input;
        # every lark error is wrapped into AglSyntaxError.
        with pytest.raises(AglSyntaxError):
            parse_program(garbage)

    def test_wrong_format_option_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('prompt[format: true] "hello"')

    def test_wrong_strict_json_option_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('prompt[strict_json: json] "hello"')

    def test_wrong_retry_name_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('prompt[on_parse_error: loop[3]] "hello"')

    def test_unknown_option_key_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('prompt[unknown_key: true] "hello"')


# ---------------------------------------------------------------------------
# Node ID uniqueness
# ---------------------------------------------------------------------------


class TestNodeIds:
    def _collect_ids(self, prog: Program) -> list[int]:
        ids: list[int] = []

        def visit(node: object) -> None:
            nid = getattr(node, "node_id", None)
            if isinstance(nid, int):
                ids.append(nid)
            for field_name in getattr(node, "__dataclass_fields__", {}):
                child = getattr(node, field_name)
                if isinstance(child, tuple):
                    for c in child:
                        visit(c)
                elif hasattr(child, "__dataclass_fields__"):
                    visit(child)

        visit(prog)
        return ids

    def test_node_ids_are_unique(self) -> None:
        prog = parse_program('let x = 42; prompt "hello ${x as raw}"')
        ids = self._collect_ids(prog)
        assert len(ids) == len(set(ids)), f"Duplicate node IDs found: {ids}"

    def test_node_ids_reset_per_parse(self) -> None:
        prog1 = parse_program("pass")
        prog2 = parse_program("pass")
        # Both programs' PassStmt should have node_id=0
        stmt1 = prog1.body[0]
        stmt2 = prog2.body[0]
        assert isinstance(stmt1, PassStmt)
        assert isinstance(stmt2, PassStmt)
        assert stmt1.node_id == stmt2.node_id


# ---------------------------------------------------------------------------
# Type annotation edge cases
# ---------------------------------------------------------------------------


class TestTypeAnnotationEdgeCases:
    def test_var_name_in_type_position_is_name_t(self) -> None:
        """A VAR_NAME in type position that isn't a primitive maps to NameT."""
        stmt = _parse_one("let x: mytype = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, NameT)
        assert stmt.type_ann.name == "mytype"

    def test_unknown_generic_type_raises_agl_syntax_error(self) -> None:
        """An unknown generic type constructor raises AglSyntaxError."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let x: foo[text] = null")
        assert "Unknown generic type" in str(exc_info.value)

    def test_on_parse_error_bad_policy_raises_agl_syntax_error(self) -> None:
        """on_parse_error with non-abort/retry name raises AglSyntaxError."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program('prompt[on_parse_error: foo] "hello"')
        assert "on_parse_error expects" in str(exc_info.value)

    def test_dict_with_text_key_accepted(self) -> None:
        """dict[text, V] is accepted — text is the only legal key type in v1."""
        stmt = _parse_one("let x: dict[text, int] = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, DictT)
        assert isinstance(stmt.type_ann.value, IntT)

    def test_dict_with_int_key_rejected(self) -> None:
        """dict[int, V] is rejected; span points at the key token."""
        src = "let x: dict[int, int] = null"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "dict keys are always text" in str(err)
        key_col = src.index("int,") + 1
        assert err.source_span.start_col == key_col

    def test_dict_with_bogus_key_rejected(self) -> None:
        """dict[bogus, V] is rejected; span points at the key token."""
        src = "let x: dict[bogus, int] = null"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "dict keys are always text" in str(err)
        key_col = src.index("bogus") + 1
        assert err.source_span.start_col == key_col


# ---------------------------------------------------------------------------
# errors.py: direct coverage of all exception conversion paths
# ---------------------------------------------------------------------------


class TestSyntaxErrorFromLark:
    """Cover the individual branches of syntax_error_from_lark directly."""

    def test_unexpected_characters(self) -> None:
        """UnexpectedCharacters maps to AglSyntaxError with character info."""
        from lark.exceptions import UnexpectedCharacters

        from agm.agl.parser.errors import syntax_error_from_lark

        exc = UnexpectedCharacters.__new__(UnexpectedCharacters)
        exc.line = 3
        exc.column = 7
        exc.pos_in_stream = 20
        err = syntax_error_from_lark(exc)
        assert isinstance(err, AglSyntaxError)
        assert err.source_span.start_line == 3
        assert err.source_span.start_col == 7

    def test_unexpected_eof(self) -> None:
        """UnexpectedEOF maps to AglSyntaxError with generic message."""
        from lark.exceptions import UnexpectedEOF

        from agm.agl.parser.errors import syntax_error_from_lark

        exc = UnexpectedEOF.__new__(UnexpectedEOF)
        exc.expected = {"ID"}
        err = syntax_error_from_lark(exc)
        assert isinstance(err, AglSyntaxError)
        assert "end of input" in str(err).lower() or "unexpected" in str(err).lower()

    def test_generic_exception_fallback(self) -> None:
        """An unrecognized exception maps to a generic AglSyntaxError."""
        from agm.agl.parser.errors import syntax_error_from_lark

        err = syntax_error_from_lark(ValueError("something went wrong"))
        assert isinstance(err, AglSyntaxError)
        assert "something went wrong" in str(err)


# ---------------------------------------------------------------------------
# parser.py: exception handling paths
# ---------------------------------------------------------------------------


class TestParserExceptionPaths:
    def test_lark_error_from_parse_is_wrapped(self) -> None:
        """A generic LarkError from _PARSER.parse() is wrapped in AglSyntaxError."""
        from unittest.mock import patch

        from lark.exceptions import LarkError

        import agm.agl.parser.parser as parser_mod

        with patch.object(
            parser_mod._PARSER, "parse", side_effect=LarkError("boom")
        ):
            with pytest.raises(AglSyntaxError) as exc_info:
                parse_program("let x = 1")
            assert "boom" in str(exc_info.value)

    def test_non_lark_parse_exception_surfaces(self) -> None:
        """A non-LarkError from _PARSER.parse() (an internal bug) is NOT masked.

        F6 contract: only lark errors are wrapped into AglSyntaxError; genuine
        internal bugs such as RuntimeError/AssertionError propagate unchanged so
        they are not misreported as syntax errors.
        """
        from unittest.mock import patch

        import agm.agl.parser.parser as parser_mod

        with patch.object(
            parser_mod._PARSER, "parse", side_effect=RuntimeError("internal bug")
        ):
            with pytest.raises(RuntimeError, match="internal bug"):
                parse_program("let x = 1")

    def test_visit_error_with_non_agl_exc_is_wrapped(self) -> None:
        """VisitError wrapping a non-AglSyntaxError is converted."""
        from typing import cast
        from unittest.mock import patch

        import lark
        from lark.exceptions import VisitError

        from agm.agl.parser.transform import AstBuilder

        # Wrap a RuntimeError in VisitError, simulating a bad transformer
        rt_err = RuntimeError("transformer exploded")
        sentinel_tree = cast(lark.Tree, object())

        def bad_transform(self_: object, tree: lark.Tree) -> object:
            raise VisitError("some_rule", sentinel_tree, rt_err)

        with patch.object(AstBuilder, "transform", new=bad_transform):
            with pytest.raises(AglSyntaxError) as exc_info:
                parse_program("let x = 1")
            assert "transformer exploded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Internal helper coverage
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Test internal helper functions directly to ensure full coverage."""

    def test_find_type_expr_raises_on_empty(self) -> None:
        """_find_type_expr raises AssertionError if no TypeExpr is found."""
        from agm.agl.parser.transform import _find_type_expr

        with pytest.raises(AssertionError, match="_find_type_expr"):
            _find_type_expr([])

    def test_find_expr_raises_on_empty(self) -> None:
        """_find_expr raises AssertionError if no Expr is found."""
        from agm.agl.parser.transform import _find_expr

        with pytest.raises(AssertionError, match="_find_expr"):
            _find_expr([])

    def test_syntax_error_from_meta_creates_agl_syntax_error(self) -> None:
        """syntax_error_from_meta builds an AglSyntaxError from a Meta object."""
        from lark.tree import Meta

        from agm.agl.parser.transform import syntax_error_from_meta

        meta = Meta.__new__(Meta)
        meta.line = 2
        meta.column = 5
        meta.end_line = 2
        meta.end_column = 10
        meta.start_pos = 10
        meta.end_pos = 15
        meta.empty = False
        err = syntax_error_from_meta(meta, "test error")
        assert isinstance(err, AglSyntaxError)
        assert "test error" in str(err)
        assert err.source_span.start_line == 2

    def test_paren_expr_with_no_expr_raises(self) -> None:
        """paren_expr raises AssertionError when no Expr child is present."""
        from lark.lexer import Token as LarkToken
        from lark.tree import Meta

        from agm.agl.parser.transform import AstBuilder

        meta = Meta.__new__(Meta)
        meta.line = 1
        meta.column = 1
        meta.end_line = 1
        meta.end_column = 5
        meta.start_pos = 0
        meta.end_pos = 4
        meta.empty = False
        builder = AstBuilder()
        # Passing all-Token args triggers the assertion
        args = [LarkToken("LPAR", "("), LarkToken("RPAR", ")")]
        with pytest.raises(AssertionError, match="paren_expr"):
            builder.paren_expr(meta, args)
