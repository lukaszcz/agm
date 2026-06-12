"""Tests for the AgL parser (agm.agl.parser) — Component 2.

Covers:
- LALR(1) conflict-guard: zero shift/reduce and reduce/reduce conflicts.
- Parsing every M1 construct to the expected AST shape.
- M2: RecordDef, EnumDef, TypeAlias, Constructor, FieldAccess, ListLit, DictLit.
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
    Constructor,
    DecimalLit,
    DictEntry,
    DictLit,
    EnumDef,
    ExprStmt,
    FieldAccess,
    FieldDef,
    InputDecl,
    InterpSegment,
    IntLit,
    LetDecl,
    ListLit,
    NamedArg,
    NullLit,
    PassStmt,
    PrintStmt,
    Program,
    RecordDef,
    RetryPolicy,
    SetStmt,
    StringLit,
    Template,
    TextSegment,
    TypeAlias,
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
            nid: object = getattr(node, "node_id", None)
            if isinstance(nid, int):
                ids.append(nid)
            fields: object = getattr(node, "__dataclass_fields__", {})
            assert isinstance(fields, dict)
            for field_name in fields:
                child: object = getattr(node, field_name)
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
        args: list[object] = [LarkToken("LPAR", "("), LarkToken("RPAR", ")")]
        with pytest.raises(AssertionError, match="paren_expr"):
            builder.paren_expr(meta, args)


# ---------------------------------------------------------------------------
# M2: RecordDef
# ---------------------------------------------------------------------------


class TestRecordDef:
    def test_simple_record(self) -> None:
        prog = parse_program("record Point\n  x: int\n  y: int")
        assert isinstance(prog, Program)
        assert len(prog.body) == 1
        stmt = prog.body[0]
        assert isinstance(stmt, RecordDef)
        assert stmt.name == "Point"
        assert len(stmt.fields) == 2
        f0, f1 = stmt.fields
        assert isinstance(f0, FieldDef)
        assert f0.name == "x"
        assert isinstance(f1, FieldDef)
        assert f1.name == "y"

    def test_record_field_types(self) -> None:
        src = "record Issue\n  title: text\n  severity: int\n  weight: decimal\n  tags: list[text]"
        stmt = _parse_one(src)
        assert isinstance(stmt, RecordDef)
        assert stmt.name == "Issue"
        assert len(stmt.fields) == 4
        from agm.agl.syntax.types import DecimalT, IntT, ListT, TextT
        assert isinstance(stmt.fields[0].type_expr, TextT)
        assert isinstance(stmt.fields[1].type_expr, IntT)
        assert isinstance(stmt.fields[2].type_expr, DecimalT)
        assert isinstance(stmt.fields[3].type_expr, ListT)
        assert isinstance(stmt.fields[3].type_expr.elem, TextT)

    def test_record_field_named_type(self) -> None:
        src = "record Wrapper\n  inner: MyType"
        stmt = _parse_one(src)
        assert isinstance(stmt, RecordDef)
        from agm.agl.syntax.types import NameT
        assert isinstance(stmt.fields[0].type_expr, NameT)
        assert stmt.fields[0].type_expr.name == "MyType"

    def test_record_span_covers_full_declaration(self) -> None:
        src = "record Point\n  x: int\n  y: int"
        stmt = _parse_one(src)
        assert isinstance(stmt, RecordDef)
        assert stmt.span.start_line == 1
        assert stmt.span.start_col == 1
        assert stmt.span.end_line >= 3

    def test_record_in_program_with_other_stmts(self) -> None:
        src = "record Point\n  x: int\nlet p = null"
        prog = parse_program(src)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], RecordDef)
        assert isinstance(prog.body[1], LetDecl)

    def test_two_records(self) -> None:
        src = "record A\n  x: int\nrecord B\n  y: text"
        prog = parse_program(src)
        assert len(prog.body) == 2
        rec_a = prog.body[0]
        rec_b = prog.body[1]
        assert isinstance(rec_a, RecordDef)
        assert isinstance(rec_b, RecordDef)
        assert rec_a.name == "A"
        assert rec_b.name == "B"

    def test_record_requires_indented_block(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("record Point")  # missing field block


# ---------------------------------------------------------------------------
# M2: EnumDef
# ---------------------------------------------------------------------------


class TestEnumDef:
    def test_simple_nullary_variants(self) -> None:
        src = "enum Color\n  | Red\n  | Green\n  | Blue"
        stmt = _parse_one(src)
        assert isinstance(stmt, EnumDef)
        assert stmt.name == "Color"
        assert len(stmt.variants) == 3
        v0, v1, v2 = stmt.variants
        assert v0.name == "Red"
        assert v1.name == "Green"
        assert v2.name == "Blue"
        assert len(v0.fields) == 0
        assert len(v1.fields) == 0
        assert len(v2.fields) == 0

    def test_variant_with_single_field(self) -> None:
        src = "enum Status\n  | Done\n  | Failed(reason: text)"
        stmt = _parse_one(src)
        assert isinstance(stmt, EnumDef)
        assert stmt.name == "Status"
        assert len(stmt.variants) == 2
        v0, v1 = stmt.variants
        assert v0.name == "Done"
        assert len(v0.fields) == 0
        assert v1.name == "Failed"
        assert len(v1.fields) == 1
        assert v1.fields[0].name == "reason"
        from agm.agl.syntax.types import TextT
        assert isinstance(v1.fields[0].type_expr, TextT)

    def test_variant_with_multiple_fields(self) -> None:
        src = "enum FixResult\n  | Blocked(reason: text, recoverable: bool)"
        stmt = _parse_one(src)
        assert isinstance(stmt, EnumDef)
        v = stmt.variants[0]
        assert v.name == "Blocked"
        assert len(v.fields) == 2
        assert v.fields[0].name == "reason"
        assert v.fields[1].name == "recoverable"

    def test_same_column_pipe_continuation(self) -> None:
        """| variants at same indentation level as the enum keyword."""
        src = "enum Status\n  | Done\n  | Failed(reason: text, fatal: bool)"
        stmt = _parse_one(src)
        assert isinstance(stmt, EnumDef)
        assert len(stmt.variants) == 2

    def test_indented_pipe_variants(self) -> None:
        """Variants at deeper indentation (deeper indented block)."""
        src = "enum Status\n  | Done\n  | Partial(left: int)\n  | Failed(reason: text)"
        stmt = _parse_one(src)
        assert isinstance(stmt, EnumDef)
        assert len(stmt.variants) == 3

    def test_enum_span_covers_full_declaration(self) -> None:
        src = "enum Status\n  | Done\n  | Failed(reason: text)"
        stmt = _parse_one(src)
        assert isinstance(stmt, EnumDef)
        assert stmt.span.start_line == 1
        assert stmt.span.end_line >= 3

    def test_enum_then_let(self) -> None:
        src = "enum Status\n  | Done\nlet x = null"
        prog = parse_program(src)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], EnumDef)
        assert isinstance(prog.body[1], LetDecl)

    def test_variant_trailing_comma_in_field_list(self) -> None:
        src = "enum Foo\n  | Bar(x: int, y: text,)"
        stmt = _parse_one(src)
        assert isinstance(stmt, EnumDef)
        v = stmt.variants[0]
        assert len(v.fields) == 2

    def test_variant_empty_payload(self) -> None:
        """Variant with empty parens is accepted and has no fields."""
        src = "enum Foo\n  | Empty()"
        stmt = _parse_one(src)
        assert isinstance(stmt, EnumDef)
        v = stmt.variants[0]
        assert v.name == "Empty"
        assert len(v.fields) == 0

    def test_enum_requires_indented_block(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("enum Status")  # missing variant block


# ---------------------------------------------------------------------------
# M2: TypeAlias
# ---------------------------------------------------------------------------


class TestTypeAlias:
    def test_simple_alias_to_named_type(self) -> None:
        stmt = _parse_one("type Status = Review")
        assert isinstance(stmt, TypeAlias)
        assert stmt.name == "Status"
        from agm.agl.syntax.types import NameT
        assert isinstance(stmt.type_expr, NameT)
        assert stmt.type_expr.name == "Review"

    def test_alias_to_list_type(self) -> None:
        stmt = _parse_one("type IssueList = list[Issue]")
        assert isinstance(stmt, TypeAlias)
        from agm.agl.syntax.types import ListT, NameT
        assert isinstance(stmt.type_expr, ListT)
        assert isinstance(stmt.type_expr.elem, NameT)
        assert stmt.type_expr.elem.name == "Issue"

    def test_alias_to_dict_type(self) -> None:
        stmt = _parse_one("type IssueMap = dict[text, Issue]")
        assert isinstance(stmt, TypeAlias)
        from agm.agl.syntax.types import DictT, NameT
        assert isinstance(stmt.type_expr, DictT)
        assert isinstance(stmt.type_expr.value, NameT)

    def test_alias_to_primitive_type(self) -> None:
        stmt = _parse_one("type Msg = text")
        assert isinstance(stmt, TypeAlias)
        from agm.agl.syntax.types import TextT
        assert isinstance(stmt.type_expr, TextT)

    def test_alias_span(self) -> None:
        stmt = _parse_one("type Status = Review")
        assert isinstance(stmt, TypeAlias)
        assert stmt.span.start_line == 1
        assert stmt.span.start_col == 1

    def test_alias_in_program(self) -> None:
        src = "type Status = Review\nlet x = null"
        prog = parse_program(src)
        assert len(prog.body) == 2
        assert isinstance(prog.body[0], TypeAlias)
        assert isinstance(prog.body[1], LetDecl)


# ---------------------------------------------------------------------------
# M2: List literals
# ---------------------------------------------------------------------------


class TestListLit:
    def test_empty_list(self) -> None:
        stmt = _parse_one("let x = []")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, ListLit)
        assert len(stmt.value.elements) == 0

    def test_single_element(self) -> None:
        stmt = _parse_one("let x = [1]")
        assert isinstance(stmt, LetDecl)
        lst = stmt.value
        assert isinstance(lst, ListLit)
        assert len(lst.elements) == 1
        elem = lst.elements[0]
        assert isinstance(elem, IntLit)
        assert elem.value == 1

    def test_multiple_elements(self) -> None:
        stmt = _parse_one('let x = [1, 2, 3]')
        assert isinstance(stmt, LetDecl)
        lst = stmt.value
        assert isinstance(lst, ListLit)
        assert len(lst.elements) == 3

    def test_trailing_comma(self) -> None:
        stmt = _parse_one('let x = [1, 2,]')
        assert isinstance(stmt, LetDecl)
        lst = stmt.value
        assert isinstance(lst, ListLit)
        assert len(lst.elements) == 2

    def test_nested_list(self) -> None:
        stmt = _parse_one("let x = [[1, 2], [3]]")
        assert isinstance(stmt, LetDecl)
        lst = stmt.value
        assert isinstance(lst, ListLit)
        assert len(lst.elements) == 2
        assert isinstance(lst.elements[0], ListLit)
        assert isinstance(lst.elements[1], ListLit)

    def test_list_of_strings(self) -> None:
        stmt = _parse_one('let x = ["a", "b"]')
        assert isinstance(stmt, LetDecl)
        lst = stmt.value
        assert isinstance(lst, ListLit)
        assert len(lst.elements) == 2
        first = lst.elements[0]
        assert isinstance(first, StringLit)
        assert first.value == "a"

    def test_list_expr_stmt(self) -> None:
        stmt = _parse_one("[1, 2]")
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, ListLit)

    def test_list_multiline(self) -> None:
        src = 'let x = [\n  "a",\n  "b"\n]'
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, ListLit)
        assert len(stmt.value.elements) == 2


# ---------------------------------------------------------------------------
# M2: Dict literals
# ---------------------------------------------------------------------------


class TestDictLit:
    def test_empty_dict(self) -> None:
        stmt = _parse_one("let x = {}")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, DictLit)
        assert len(stmt.value.entries) == 0

    def test_identifier_key(self) -> None:
        stmt = _parse_one('let x = {source: "reviewer"}')
        assert isinstance(stmt, LetDecl)
        d = stmt.value
        assert isinstance(d, DictLit)
        assert len(d.entries) == 1
        e = d.entries[0]
        assert isinstance(e, DictEntry)
        assert isinstance(e.key, StringLit)
        assert e.key.value == "source"
        assert isinstance(e.value, StringLit)

    def test_string_key(self) -> None:
        stmt = _parse_one('let x = {"source": "reviewer"}')
        assert isinstance(stmt, LetDecl)
        d = stmt.value
        assert isinstance(d, DictLit)
        assert len(d.entries) == 1
        e = d.entries[0]
        assert isinstance(e.key, StringLit)
        assert e.key.value == "source"

    def test_multiple_entries(self) -> None:
        stmt = _parse_one('let x = {source: "reviewer", attempt: 2}')
        assert isinstance(stmt, LetDecl)
        d = stmt.value
        assert isinstance(d, DictLit)
        assert len(d.entries) == 2
        assert d.entries[0].key.value == "source"
        assert d.entries[1].key.value == "attempt"

    def test_trailing_comma(self) -> None:
        stmt = _parse_one('let x = {a: 1, b: 2,}')
        assert isinstance(stmt, LetDecl)
        d = stmt.value
        assert isinstance(d, DictLit)
        assert len(d.entries) == 2

    def test_null_value(self) -> None:
        stmt = _parse_one("{retries: null}")
        assert isinstance(stmt, ExprStmt)
        d = stmt.expr
        assert isinstance(d, DictLit)
        assert isinstance(d.entries[0].value, NullLit)

    def test_nested_list_value(self) -> None:
        stmt = _parse_one('{tags: ["a", "b"]}')
        assert isinstance(stmt, ExprStmt)
        d = stmt.expr
        assert isinstance(d, DictLit)
        assert isinstance(d.entries[0].value, ListLit)

    def test_dict_multiline(self) -> None:
        src = 'let x = {\n  source: "reviewer",\n  attempt: 2\n}'
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, DictLit)
        assert len(stmt.value.entries) == 2

    def test_identifier_key_is_string_lit(self) -> None:
        """Identifier shorthand keys are converted to StringLit by the builder."""
        stmt = _parse_one("{foo: 1}")
        assert isinstance(stmt, ExprStmt)
        d = stmt.expr
        assert isinstance(d, DictLit)
        e = d.entries[0]
        assert isinstance(e.key, StringLit)
        assert e.key.value == "foo"

    def test_interpolated_string_key(self) -> None:
        """A dict key that is a template with interpolation is handled.

        An interpolated key (e.g. ``"${x}"``) only captures plain-text segments;
        the interp segments are dropped, producing an empty-string StringLit key.
        """
        src = 'let d = {"${x}": 1}'
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        d = stmt.value
        assert isinstance(d, DictLit)
        assert len(d.entries) == 1
        e = d.entries[0]
        # Interpolated keys are normalised to StringLit (interp segments stripped).
        assert isinstance(e.key, StringLit)
        assert isinstance(e.value, IntLit)


# ---------------------------------------------------------------------------
# M2: Constructor expressions
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_nullary_unqualified(self) -> None:
        """Unqualified nullary constructor: just TYPE_NAME."""
        stmt = _parse_one("let x = Done")
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.qualifier is None
        assert c.name == "Done"
        assert len(c.args) == 0

    def test_qualified_nullary(self) -> None:
        """Qualified nullary: TYPE_NAME.TYPE_NAME."""
        stmt = _parse_one("let x = Status.Done")
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.qualifier == "Status"
        assert c.name == "Done"
        assert len(c.args) == 0

    def test_unqualified_with_args(self) -> None:
        """Unqualified constructor with named args: Type(x: 1, y: 2)."""
        stmt = _parse_one("let p = Point(x: 1, y: 2)")
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.qualifier is None
        assert c.name == "Point"
        assert len(c.args) == 2
        a0, a1 = c.args
        assert isinstance(a0, NamedArg)
        assert a0.name == "x"
        assert isinstance(a0.value, IntLit)
        assert a0.value.value == 1
        assert a1.name == "y"
        assert isinstance(a1.value, IntLit)
        assert a1.value.value == 2

    def test_qualified_with_args(self) -> None:
        """Qualified constructor with named args."""
        stmt = _parse_one('let r = Review.Fail(issues: ["missing tests"])')
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.qualifier == "Review"
        assert c.name == "Fail"
        assert len(c.args) == 1
        assert c.args[0].name == "issues"
        assert isinstance(c.args[0].value, ListLit)

    def test_constructor_trailing_comma(self) -> None:
        """Trailing comma in constructor args is accepted."""
        stmt = _parse_one("let p = Point(x: 1, y: 2,)")
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert len(c.args) == 2

    def test_constructor_empty_parens(self) -> None:
        """TYPE_NAME() — constructor with empty argument list."""
        stmt = _parse_one("let x = Done()")
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.name == "Done"
        assert len(c.args) == 0

    def test_constructor_expr_stmt(self) -> None:
        """Constructor as expression statement."""
        stmt = _parse_one("Done")
        assert isinstance(stmt, ExprStmt)
        ctor = stmt.expr
        assert isinstance(ctor, Constructor)
        assert ctor.name == "Done"

    def test_duplicate_named_arg_rejected(self) -> None:
        """Duplicate named arg is rejected; error points at the duplicate."""
        src = "let p = Point(x: 1, x: 2)"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "duplicate" in str(err).lower()
        # Error span must point at the second (duplicate) occurrence of 'x'
        dup_col = src.index("x: 2") + 1
        assert err.source_span.start_col == dup_col

    def test_positional_arg_rejected(self) -> None:
        """Positional (non-named) constructor args are not in v1."""
        with pytest.raises(AglSyntaxError):
            parse_program("let x = Point(1, 2)")

    def test_constructor_multiline(self) -> None:
        """Multi-line constructor with trailing comma."""
        src = "let issue = Issue(\n  title: null,\n  severity: 3,\n)"
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.name == "Issue"
        assert len(c.args) == 2

    def test_constructor_nested(self) -> None:
        """Nested constructor: Author(name: ...) inside Issue(author: ...)."""
        src = 'let x = Issue(author: Author(name: "Ada", active: true))'
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.name == "Issue"
        inner = c.args[0].value
        assert isinstance(inner, Constructor)
        assert inner.name == "Author"


# ---------------------------------------------------------------------------
# M2: Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    def test_simple_field_access(self) -> None:
        stmt = _parse_one("print issue.title")
        assert isinstance(stmt, PrintStmt)
        fa = stmt.value
        assert isinstance(fa, FieldAccess)
        assert isinstance(fa.obj, VarRef)
        assert fa.obj.name == "issue"
        assert fa.field == "title"

    def test_chained_field_access(self) -> None:
        stmt = _parse_one("print issue.author.name")
        assert isinstance(stmt, PrintStmt)
        fa = stmt.value
        assert isinstance(fa, FieldAccess)
        assert fa.field == "name"
        inner = fa.obj
        assert isinstance(inner, FieldAccess)
        assert inner.field == "author"
        assert isinstance(inner.obj, VarRef)
        assert inner.obj.name == "issue"

    def test_field_access_in_let(self) -> None:
        stmt = _parse_one("let t = issue.title")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, FieldAccess)

    def test_field_access_span(self) -> None:
        stmt = _parse_one("let t = issue.title")
        assert isinstance(stmt, LetDecl)
        fa = stmt.value
        assert isinstance(fa, FieldAccess)
        assert fa.span.start_line == 1

    def test_var_ref_not_wrapped_in_field_access(self) -> None:
        """A plain VAR_NAME reference without '.' must not become FieldAccess."""
        stmt = _parse_one("let x = y")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, VarRef)

    def test_field_access_on_paren_expr(self) -> None:
        """Field access can be applied to parenthesized expressions."""
        stmt = _parse_one("let t = (issue).title")
        assert isinstance(stmt, LetDecl)
        fa = stmt.value
        assert isinstance(fa, FieldAccess)
        assert fa.field == "title"

    def test_type_name_access_on_non_constructor(self) -> None:
        """VAR_NAME.TypeName — type_access where LHS is not a Constructor.

        This covers the else-branch in type_access: when the object expression
        is not a bare unqualified Constructor (e.g. a VarRef), the result is
        a Constructor with no qualifier so the scope checker can report the error.
        """
        stmt = _parse_one("let t = x.Done")
        assert isinstance(stmt, LetDecl)
        # The result is a Constructor built from the TYPE_NAME part.
        # Scope checking (M3) will validate whether this is legal.
        ctor = stmt.value
        assert isinstance(ctor, Constructor)
        assert ctor.qualifier is None
        assert ctor.name == "Done"


# ---------------------------------------------------------------------------
# M2: Sanity — parse types/*.agl programs to a Program (parse-only)
# ---------------------------------------------------------------------------


class TestTypeProgramFiles:
    @pytest.mark.xfail(
        reason="records.agl uses M3 constructs (if_stmt, comparison); passes after M3.",
        strict=False,
    )
    def test_parse_records_agl(self) -> None:
        """tests/agl/programs/types/records.agl must parse to a Program."""
        import pathlib

        src_path = pathlib.Path("tests/agl/programs/types/records.agl")
        src = src_path.read_text(encoding="utf-8")
        prog = parse_program(src)
        assert isinstance(prog, Program)
        assert len(prog.body) > 0
        # Must contain at least one RecordDef
        assert any(isinstance(s, RecordDef) for s in prog.body)

    @pytest.mark.xfail(
        reason="enums.agl uses M3 constructs (if_stmt, case_stmt, is/is not); passes after M3.",
        strict=False,
    )
    def test_parse_enums_agl(self) -> None:
        """tests/agl/programs/types/enums.agl must parse to a Program."""
        import pathlib

        src_path = pathlib.Path("tests/agl/programs/types/enums.agl")
        src = src_path.read_text(encoding="utf-8")
        prog = parse_program(src)
        assert isinstance(prog, Program)
        assert len(prog.body) > 0
        # Must contain at least one EnumDef
        assert any(isinstance(s, EnumDef) for s in prog.body)
