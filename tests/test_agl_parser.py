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
from agm.agl.parser import (
    AglSyntaxError,
    is_incomplete_source,
    parse_program,
    parse_program_seeded,
)
from agm.agl.syntax import (
    AbortPolicy,
    AgentCall,
    AgentDecl,
    BinaryOp,
    BinOp,
    BoolLit,
    CallOptions,
    CaseExpr,
    CaseExprBranch,
    CaseStmt,
    CatchClause,
    Constructor,
    ConstructorPattern,
    DecimalLit,
    DictEntry,
    DictLit,
    DoUntil,
    EnumDef,
    ExprStmt,
    FieldAccess,
    FieldDef,
    IfBranch,
    IfExpr,
    IfExprBranch,
    IfStmt,
    InputDecl,
    InterpSegment,
    IntLit,
    IsTest,
    LetDecl,
    ListLit,
    LiteralPattern,
    NamedArg,
    NullLit,
    PassStmt,
    PatternField,
    PrintStmt,
    Program,
    Raise,
    RecordDef,
    RetryPolicy,
    SetStmt,
    StringLit,
    Template,
    TextSegment,
    TryCatch,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.nodes import ELSE
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
from tests._agl_helpers import all_node_ids

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
# AgentDecl
# ---------------------------------------------------------------------------


class TestAgentDecl:
    def test_bare_declaration(self) -> None:
        stmt = _parse_one("agent reviewer")
        assert isinstance(stmt, AgentDecl)
        assert stmt.name == "reviewer"
        assert stmt.runner is None

    def test_declaration_with_runner(self) -> None:
        stmt = _parse_one('agent impl = "claude -p %{PROMPT_FILE}"')
        assert isinstance(stmt, AgentDecl)
        assert stmt.name == "impl"
        assert stmt.runner == "claude -p %{PROMPT_FILE}"

    def test_declaration_with_empty_runner(self) -> None:
        stmt = _parse_one('agent impl = ""')
        assert isinstance(stmt, AgentDecl)
        assert stmt.name == "impl"
        assert stmt.runner == ""

    def test_runner_with_interpolation_rejected(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('agent impl = "run ${x}"')

    def test_agent_usable_as_field_name(self) -> None:
        # `agent` is reserved, but stays valid as a record-field name so that
        # built-in exception fields like `AgentCallError.agent` keep working.
        stmt = _parse_one("print e.agent")
        assert isinstance(stmt, PrintStmt)
        assert isinstance(stmt.value, FieldAccess)
        assert stmt.value.field == "agent"


# ---------------------------------------------------------------------------
# `agent` reserved word used as a plain field/key name (field_name nonterminal)
#
# `agent` is reserved (it leads `agent_decl`), but it is also the field name on
# built-in exception records (AgentCallError.agent / AgentParseError.agent).
# It must therefore remain usable wherever a record/struct field or named key
# is referenced — field definitions, named constructor args, dict shorthand
# keys, and postfix field access.  It must NOT become bindable as a variable
# name (let/var/set/input/...).
# ---------------------------------------------------------------------------


class TestAgentAsFieldName:
    def test_record_field_named_agent(self) -> None:
        stmt = _parse_one("record R\n  agent: text")
        assert isinstance(stmt, RecordDef)
        assert stmt.name == "R"
        assert len(stmt.fields) == 1
        assert isinstance(stmt.fields[0], FieldDef)
        assert stmt.fields[0].name == "agent"
        assert isinstance(stmt.fields[0].type_expr, TextT)

    def test_record_field_named_agent_among_others(self) -> None:
        stmt = _parse_one("record R\n  message: text\n  agent: text\n  attempt: int")
        assert isinstance(stmt, RecordDef)
        assert [f.name for f in stmt.fields] == ["message", "agent", "attempt"]

    def test_enum_variant_field_named_agent(self) -> None:
        stmt = _parse_one("enum E\n  | Failure(agent: text)")
        assert isinstance(stmt, EnumDef)
        variant = stmt.variants[0]
        assert variant.name == "Failure"
        assert variant.fields[0].name == "agent"

    def test_dict_shorthand_key_agent(self) -> None:
        stmt = _parse_one("{agent: 1}")
        assert isinstance(stmt, ExprStmt)
        assert isinstance(stmt.expr, DictLit)
        entry = stmt.expr.entries[0]
        assert isinstance(entry, DictEntry)
        assert isinstance(entry.key, StringLit)
        assert entry.key.value == "agent"

    def test_named_constructor_arg_agent(self) -> None:
        stmt = _parse_one('Foo(agent: "n")')
        assert isinstance(stmt, ExprStmt)
        ctor = stmt.expr
        assert isinstance(ctor, Constructor)
        assert ctor.name == "Foo"
        assert ctor.args[0].name == "agent"

    def test_raise_builtin_exception_with_agent_field(self) -> None:
        src = (
            'raise AgentCallError(agent: "x", message: "m", '
            'trace_id: "t", cause: "c", metadata: null)'
        )
        stmt = _parse_one(src)
        assert isinstance(stmt, Raise)
        ctor = stmt.exc
        assert isinstance(ctor, Constructor)
        assert ctor.name == "AgentCallError"
        assert [a.name for a in ctor.args] == [
            "agent",
            "message",
            "trace_id",
            "cause",
            "metadata",
        ]

    def test_field_access_agent_still_parses(self) -> None:
        stmt = _parse_one("print e.agent")
        assert isinstance(stmt, PrintStmt)
        assert isinstance(stmt.value, FieldAccess)
        assert stmt.value.field == "agent"

    def test_agent_not_bindable_as_variable(self) -> None:
        # `agent` must stay reserved as a binder: `let agent = …` is the start
        # of neither a valid let nor an agent decl (no name after the keyword).
        with pytest.raises(AglSyntaxError):
            parse_program("let agent = 1")


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
        stmt = _parse_one('ask "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.agent == "ask"
        assert isinstance(call.options, CallOptions)
        assert call.options.format is None
        assert call.options.strict_json is None
        assert call.options.parse_policy is None

    def test_agent_call_format_json(self) -> None:
        stmt = _parse_one('ask[format: json] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "json"

    def test_agent_call_format_text(self) -> None:
        stmt = _parse_one('ask[format: text] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "text"

    def test_agent_call_strict_json_true(self) -> None:
        stmt = _parse_one('ask[strict_json: true] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.strict_json is True

    def test_agent_call_strict_json_false(self) -> None:
        stmt = _parse_one('ask[strict_json: false] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.strict_json is False

    def test_agent_call_on_parse_error_abort(self) -> None:
        stmt = _parse_one('ask[on_parse_error: abort] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert isinstance(call.options.parse_policy, AbortPolicy)

    def test_agent_call_on_parse_error_retry(self) -> None:
        stmt = _parse_one('ask[on_parse_error: retry[3]] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert isinstance(call.options.parse_policy, RetryPolicy)
        assert call.options.parse_policy.extra == 3

    def test_agent_call_multiple_options(self) -> None:
        stmt = _parse_one('ask[format: json, strict_json: true] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "json"
        assert call.options.strict_json is True

    def test_agent_call_trailing_comma_in_options(self) -> None:
        stmt = _parse_one('ask[format: json,] "hello"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        assert call.options.format == "json"

    def test_duplicate_format_option_rejected(self) -> None:
        src = 'ask[format: json, format: text] "hello"'
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
        src = 'ask[strict_json: true, strict_json: false] "hello"'
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "duplicate option 'strict_json'" in str(err)
        dup_col = src.index("strict_json", src.index("strict_json") + 1) + 1
        assert err.source_span.start_col == dup_col

    def test_duplicate_on_parse_error_option_rejected(self) -> None:
        src = 'ask[on_parse_error: abort, on_parse_error: retry[2]] "hello"'
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "duplicate option 'on_parse_error'" in str(err)
        dup_col = src.index("on_parse_error", src.index("on_parse_error") + 1) + 1
        assert err.source_span.start_col == dup_col

    def test_distinct_options_twin_parses(self) -> None:
        # Accept-twin: distinct option keys parse fine even with three options.
        stmt = _parse_one(
            'ask[format: json, strict_json: true, on_parse_error: abort] "hi"'
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
        stmt = _parse_one('ask "hello world"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        tmpl = call.template
        assert isinstance(tmpl, Template)
        # One TextSegment for the content
        text_segs = [s for s in tmpl.segments if isinstance(s, TextSegment)]
        assert any(s.text == "hello world" for s in text_segs)

    def test_single_interpolation(self) -> None:
        stmt = _parse_one('ask "hello ${x}"')
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
        stmt = _parse_one('ask "hello ${x as raw}"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        tmpl = call.template
        interps = [s for s in tmpl.segments if isinstance(s, InterpSegment)]
        assert len(interps) == 1
        assert interps[0].render == "raw"

    def test_multiple_interpolations(self) -> None:
        stmt = _parse_one('ask "hello ${x} and ${y as raw}"')
        assert isinstance(stmt, ExprStmt)
        call = stmt.expr
        assert isinstance(call, AgentCall)
        tmpl = call.template
        interps = [s for s in tmpl.segments if isinstance(s, InterpSegment)]
        assert len(interps) == 2
        assert interps[0].render is None
        assert interps[1].render == "raw"

    def test_template_text_segment_content(self) -> None:
        stmt = _parse_one('ask "hello ${x} world"')
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
        stmt = _parse_one('ask "${a}${b}"')
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
        stmt = _parse_one('ask "x${a}"')
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
        stmt = _parse_one('ask "${a} mid ${b}"')
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
        prog2 = parse_program('ask "${x}"')
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
            parse_program('ask "unterminated')

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
            parse_program('ask[format: true] "hello"')

    def test_wrong_strict_json_option_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('ask[strict_json: json] "hello"')

    def test_wrong_retry_name_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('ask[on_parse_error: loop[3]] "hello"')

    def test_unknown_option_key_raises_agl_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program('ask[unknown_key: true] "hello"')


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
        prog = parse_program('let x = 42; ask "hello ${x as raw}"')
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
            parse_program('ask[on_parse_error: foo] "hello"')
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

    # --- Task 2: dict_type head validation ---

    def test_dict_head_foo_rejected(self) -> None:
        """foo[text, int] must be rejected with a quoted type name in the message."""
        src = "let x: foo[text, int] = null"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "'foo'" in str(err)
        head_col = src.index("foo[") + 1
        assert err.source_span.start_col == head_col

    def test_dict_head_list_rejected(self) -> None:
        """list[text, int] must be rejected — Unknown generic type: 'list'."""
        src = "let x: list[text, int] = null"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "'list'" in str(err)
        head_col = src.index("list[") + 1
        assert err.source_span.start_col == head_col

    def test_dict_head_dict_still_accepted(self) -> None:
        """dict[text, int] must still parse to DictT(IntT)."""
        stmt = _parse_one("let x: dict[text, int] = null")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.type_ann, DictT)
        assert isinstance(stmt.type_ann.value, IntT)


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

    def test_find_name_token_raises_on_empty(self) -> None:
        """_find_name_token raises AssertionError if no name token is found."""
        from agm.agl.parser.transform import _find_name_token

        with pytest.raises(AssertionError, match="_find_name_token"):
            _find_name_token([])

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

    def test_interpolated_string_key_rejected(self) -> None:
        """F5: a dict key carrying interpolation is rejected, not coerced.

        ``{"${a}": 1}`` previously silently produced an empty-string StringLit
        key.  Per design §10.14 dict keys must be literal strings; the error
        pins to the key's span.
        """
        src = 'let d = {"${a}": 1}'
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "interpolation" in str(err).lower()
        span = err.source_span
        assert span.start_line == 1
        # The error points at the offending key (the opening quote).
        key_col = src.index('"${a}"') + 1
        assert span.start_col == key_col

    def test_plain_string_key_still_parses(self) -> None:
        """Accept-twin: a non-interpolated quoted key parses to StringLit."""
        stmt = _parse_one('let d = {"k": 1}')
        assert isinstance(stmt, LetDecl)
        d = stmt.value
        assert isinstance(d, DictLit)
        e = d.entries[0]
        assert isinstance(e.key, StringLit)
        assert e.key.value == "k"
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

    def test_double_payload_rejected(self) -> None:
        """F3: ``Issue(a: 1)(b: 2)`` — a second payload is rejected.

        Previously the first payload was silently dropped.  The error pins to
        the second payload's span.
        """
        src = "let t = Issue(a: 1)(b: 2)"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "single argument list" in str(err).lower()
        span = err.source_span
        assert span.start_line == 1
        # Second payload begins at the '(' right after the first ')'.
        assert span.start_col == src.index("(b: 2)") + 1

    def test_double_payload_after_empty_first_rejected(self) -> None:
        """F3 edge: an empty first payload still counts as applied.

        ``Empty()(b: 2)`` — the first ``()`` produces empty args; a second
        payload is still rejected (state is tracked, not inferred from len).
        """
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let t = Empty()(b: 2)")
        assert "single argument list" in str(exc_info.value).lower()

    def test_qualified_first_payload_still_parses(self) -> None:
        """Accept-twin for F3: ``Status.Done(x: 1)`` — first payload is legal.

        The qualified no-arg constructor produced by type_access has empty
        args; attaching its first payload must still work.
        """
        stmt = _parse_one("let r = Status.Done(x: 1)")
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.qualifier == "Status"
        assert c.name == "Done"
        assert len(c.args) == 1
        assert c.args[0].name == "x"

    def test_empty_payload_still_parses(self) -> None:
        """Accept-twin for F3: ``Empty()`` — a single empty payload is legal."""
        stmt = _parse_one("let t = Empty()")
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.name == "Empty"
        assert len(c.args) == 0

    def test_method_call_syntax_rejected(self) -> None:
        """F4: ``x.f(a: 1)`` — payload on a non-constructor base is rejected.

        Previously this leaked an AssertionError wrapped as VisitError noise
        with a degenerate span.  Now it is a clean AglSyntaxError carrying the
        rule's meta span.
        """
        src = "let t = x.f(a: 1)"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        msg = str(err).lower()
        assert "method-call syntax" in msg
        assert "type name" in msg
        span = err.source_span
        assert span.start_line == 1
        # Span covers the whole offending expression, not a degenerate 1:1 span.
        assert span.end_offset > span.start_offset

    def test_payload_on_field_access_rejected(self) -> None:
        """F4 variant: ``issue.title(a: 1)`` — payload on a FieldAccess base."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let t = issue.title(a: 1)")
        assert "method-call syntax" in str(exc_info.value).lower()

    def test_payload_on_paren_expr_rejected(self) -> None:
        """F4 variant: ``(x)(a: 1)`` — payload on a parenthesized expression."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let t = (x)(a: 1)")
        assert "method-call syntax" in str(exc_info.value).lower()


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

    def test_type_name_access_on_non_constructor_rejected(self) -> None:
        """F1: ``x.Done`` (VAR_NAME LHS) is rejected, not silently dropped.

        Only ``TYPE_NAME . TYPE_NAME`` qualification is legal (design §10.13).
        Previously ``x`` was dropped and a bare ``Constructor('Done')`` was
        fabricated; now it is a clean AglSyntaxError pinned at ``.Done``.
        """
        src = "let t = x.Done"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "type name" in str(err).lower()
        span = err.source_span
        assert span.start_line == 1
        assert span.start_col == src.index("Done") + 1

    def test_type_name_access_on_field_access_rejected(self) -> None:
        """F1 variant: ``issue.title.Done`` — LHS is a record FieldAccess."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let t = issue.title.Done")
        assert "type name" in str(exc_info.value).lower()

    def test_double_qualified_type_access_rejected(self) -> None:
        """F2: ``A.B.C`` — the LHS is already a qualified Constructor.

        Previously ``A.B`` was dropped, yielding ``Constructor('C')``; now the
        second ``.TypeName`` is rejected with the span pinned at ``.C``.
        """
        src = "let t = A.B.C"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "type name" in str(err).lower()
        assert err.source_span.start_col == src.rindex("C") + 1

    def test_type_access_on_constructor_with_args_rejected(self) -> None:
        """F2 variant: ``Empty(a: 1).Done`` — LHS already carries args."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let t = Empty(a: 1).Done")
        assert "type name" in str(exc_info.value).lower()

    def test_qualified_nullary_still_parses(self) -> None:
        """Accept-twin for F1/F2: ``Status.Done`` is a valid qualified ctor."""
        stmt = _parse_one("let x = Status.Done")
        assert isinstance(stmt, LetDecl)
        c = stmt.value
        assert isinstance(c, Constructor)
        assert c.qualifier == "Status"
        assert c.name == "Done"
        assert len(c.args) == 0

    def test_field_access_on_constructor_result_still_parses(self) -> None:
        """Accept-twin: ``Issue(a: 1).title`` — field access on a ctor result.

        Per design §10.13 ``access ::= atom ("." VAR_NAME)*`` allows a
        lowercase ``.field`` suffix over any atom, including a constructor
        application.  This must keep parsing as a FieldAccess.
        """
        stmt = _parse_one("let t = Issue(a: 1).title")
        assert isinstance(stmt, LetDecl)
        fa = stmt.value
        assert isinstance(fa, FieldAccess)
        assert fa.field == "title"
        assert isinstance(fa.obj, Constructor)
        assert fa.obj.name == "Issue"
        assert len(fa.obj.args) == 1


# ---------------------------------------------------------------------------
# M2: Sanity — parse types/*.agl programs to a Program (parse-only)
# ---------------------------------------------------------------------------


class TestTypeProgramFiles:
    def test_parse_records_agl(self) -> None:
        """tests/agl/programs/types/records.agl must parse to a Program (M3)."""
        import pathlib

        src_path = pathlib.Path("tests/agl/programs/types/records.agl")
        src = src_path.read_text(encoding="utf-8")
        prog = parse_program(src)
        assert isinstance(prog, Program)
        assert len(prog.body) > 0
        # Must contain at least one RecordDef
        assert any(isinstance(s, RecordDef) for s in prog.body)

    def test_parse_enums_agl(self) -> None:
        """tests/agl/programs/types/enums.agl must parse to a Program (M3)."""
        import pathlib

        src_path = pathlib.Path("tests/agl/programs/types/enums.agl")
        src = src_path.read_text(encoding="utf-8")
        prog = parse_program(src)
        assert isinstance(prog, Program)
        assert len(prog.body) > 0
        # Must contain at least one EnumDef
        assert any(isinstance(s, EnumDef) for s in prog.body)


# ---------------------------------------------------------------------------
# M3: Binary operators (precedence chain)
# ---------------------------------------------------------------------------


class TestBinaryOperators:
    def test_equality(self) -> None:
        stmt = _parse_one("let x = a = b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.EQ

    def test_inequality(self) -> None:
        stmt = _parse_one("let x = a != b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.NEQ

    def test_less_than(self) -> None:
        stmt = _parse_one("let x = a < b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.LT

    def test_less_equal(self) -> None:
        stmt = _parse_one("let x = a <= b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.LE

    def test_greater_than(self) -> None:
        stmt = _parse_one("let x = a > b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.GT

    def test_greater_equal(self) -> None:
        stmt = _parse_one("let x = a >= b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.GE

    def test_in_operator(self) -> None:
        stmt = _parse_one("let x = a in b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.IN

    def test_addition(self) -> None:
        stmt = _parse_one("let x = a + b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.ADD

    def test_subtraction(self) -> None:
        stmt = _parse_one("let x = a - b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.SUB

    def test_multiplication(self) -> None:
        stmt = _parse_one("let x = a * b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.MUL

    def test_division(self) -> None:
        stmt = _parse_one("let x = a / b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.DIV

    def test_or_operator(self) -> None:
        stmt = _parse_one("let x = a or b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.OR

    def test_and_operator(self) -> None:
        stmt = _parse_one("let x = a and b")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == BinOp.AND

    def test_precedence_add_mul(self) -> None:
        """a + b * c parses as a + (b * c) — MUL binds tighter than ADD."""
        stmt = _parse_one("let x = a + b * c")
        assert isinstance(stmt, LetDecl)
        top = stmt.value
        assert isinstance(top, BinaryOp)
        assert top.op == BinOp.ADD
        assert isinstance(top.right, BinaryOp)
        assert top.right.op == BinOp.MUL

    def test_precedence_neg_add(self) -> None:
        """- a + b parses as (-a) + b — unary minus binds tighter than ADD."""
        stmt = _parse_one("let x = -a + b")
        assert isinstance(stmt, LetDecl)
        top = stmt.value
        assert isinstance(top, BinaryOp)
        assert top.op == BinOp.ADD
        assert isinstance(top.left, UnaryNeg)

    def test_precedence_not_and(self) -> None:
        """not a and b parses as (not a) and b — NOT binds tighter than AND."""
        stmt = _parse_one("let x = not a and b")
        assert isinstance(stmt, LetDecl)
        top = stmt.value
        assert isinstance(top, BinaryOp)
        assert top.op == BinOp.AND
        assert isinstance(top.left, UnaryNot)

    def test_precedence_and_or(self) -> None:
        """a and b or c parses as (a and b) or c — AND binds tighter than OR."""
        stmt = _parse_one("let x = a and b or c")
        assert isinstance(stmt, LetDecl)
        top = stmt.value
        assert isinstance(top, BinaryOp)
        assert top.op == BinOp.OR
        assert isinstance(top.left, BinaryOp)
        assert top.left.op == BinOp.AND

    def test_associativity_add(self) -> None:
        """a + b + c parses left-associatively as (a + b) + c."""
        stmt = _parse_one("let x = a + b + c")
        assert isinstance(stmt, LetDecl)
        top = stmt.value
        assert isinstance(top, BinaryOp)
        assert top.op == BinOp.ADD
        assert isinstance(top.left, BinaryOp)
        assert top.left.op == BinOp.ADD

    def test_chained_comparison_rejected(self) -> None:
        """a = b = c is rejected (comparison is non-associative)."""
        with pytest.raises(AglSyntaxError):
            parse_program("let ok = (x = 1 = 2)")

    def test_operands_are_correct(self) -> None:
        """BinaryOp.left and .right are set correctly."""
        stmt = _parse_one("let x = a + b")
        assert isinstance(stmt, LetDecl)
        op = stmt.value
        assert isinstance(op, BinaryOp)
        assert isinstance(op.left, VarRef)
        assert op.left.name == "a"
        assert isinstance(op.right, VarRef)
        assert op.right.name == "b"


# ---------------------------------------------------------------------------
# M3: Unary operators
# ---------------------------------------------------------------------------


class TestUnaryOperators:
    def test_unary_not(self) -> None:
        stmt = _parse_one("let x = not a")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, UnaryNot)
        assert isinstance(stmt.value.operand, VarRef)

    def test_unary_neg(self) -> None:
        stmt = _parse_one("let x = -a")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, UnaryNeg)
        assert isinstance(stmt.value.operand, VarRef)

    def test_double_not(self) -> None:
        stmt = _parse_one("let x = not not a")
        assert isinstance(stmt, LetDecl)
        outer = stmt.value
        assert isinstance(outer, UnaryNot)
        assert isinstance(outer.operand, UnaryNot)

    def test_double_neg(self) -> None:
        stmt = _parse_one("let x = - -a")
        assert isinstance(stmt, LetDecl)
        outer = stmt.value
        assert isinstance(outer, UnaryNeg)
        assert isinstance(outer.operand, UnaryNeg)

    def test_neg_literal(self) -> None:
        stmt = _parse_one("let x = -42")
        assert isinstance(stmt, LetDecl)
        outer = stmt.value
        assert isinstance(outer, UnaryNeg)
        assert isinstance(outer.operand, IntLit)
        assert outer.operand.value == 42


# ---------------------------------------------------------------------------
# M3: is / is not tests
# ---------------------------------------------------------------------------


class TestIsTest:
    def test_is_simple(self) -> None:
        stmt = _parse_one("let x = v is Pass")
        assert isinstance(stmt, LetDecl)
        t = stmt.value
        assert isinstance(t, IsTest)
        assert isinstance(t.expr, VarRef)
        assert t.expr.name == "v"
        assert t.qualifier is None
        assert t.variant == "Pass"
        assert t.negated is False

    def test_is_qualified(self) -> None:
        stmt = _parse_one("let x = v is Review.Pass")
        assert isinstance(stmt, LetDecl)
        t = stmt.value
        assert isinstance(t, IsTest)
        assert t.qualifier == "Review"
        assert t.variant == "Pass"
        assert t.negated is False

    def test_is_not_simple(self) -> None:
        stmt = _parse_one("let x = v is not Fail")
        assert isinstance(stmt, LetDecl)
        t = stmt.value
        assert isinstance(t, IsTest)
        assert t.variant == "Fail"
        assert t.negated is True

    def test_is_not_qualified(self) -> None:
        stmt = _parse_one("let x = v is not Review.Fail")
        assert isinstance(stmt, LetDecl)
        t = stmt.value
        assert isinstance(t, IsTest)
        assert t.qualifier == "Review"
        assert t.variant == "Fail"
        assert t.negated is True

    def test_is_in_condition(self) -> None:
        src = 'do pass until status is Complete'
        prog = parse_program(src)
        assert len(prog.body) == 1
        du = prog.body[0]
        assert isinstance(du, DoUntil)
        assert isinstance(du.condition, IsTest)
        assert du.condition.variant == "Complete"

    def test_is_not_in_condition(self) -> None:
        src = 'do pass until review is not Pass'
        prog = parse_program(src)
        du = prog.body[0]
        assert isinstance(du, DoUntil)
        assert isinstance(du.condition, IsTest)
        assert du.condition.negated is True


# ---------------------------------------------------------------------------
# M3: raise_stmt
# ---------------------------------------------------------------------------


class TestRaiseStmt:
    def test_raise_var_ref(self) -> None:
        stmt = _parse_one("raise e")
        assert isinstance(stmt, Raise)
        assert isinstance(stmt.exc, VarRef)
        assert stmt.exc.name == "e"

    def test_raise_constructor(self) -> None:
        stmt = _parse_one('raise Abort(message: "oops")')
        assert isinstance(stmt, Raise)
        assert isinstance(stmt.exc, Constructor)
        assert stmt.exc.name == "Abort"

    def test_raise_in_if_branch(self) -> None:
        src = 'if true => raise Abort(message: "bad")'
        prog = parse_program(src)
        assert len(prog.body) == 1
        stmt = prog.body[0]
        assert isinstance(stmt, IfStmt)
        branch = stmt.branches[0]
        assert len(branch.body) == 1
        assert isinstance(branch.body[0], Raise)


# ---------------------------------------------------------------------------
# M3: do_until
# ---------------------------------------------------------------------------


class TestDoUntil:
    def test_simple_bounded(self) -> None:
        src = "do[5] pass until true"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert stmt.limit == 5
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], PassStmt)
        assert isinstance(stmt.condition, BoolLit)
        assert stmt.condition.value is True

    def test_unbounded(self) -> None:
        src = "do pass until true"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert stmt.limit is None

    def test_multiline(self) -> None:
        src = "do[3]\n  pass\nuntil true"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert stmt.limit == 3
        assert len(stmt.body) == 1

    def test_multiline_multi_stmts(self) -> None:
        src = "do[3]\n  let x = 1\n  pass\nuntil true"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert len(stmt.body) == 2
        assert isinstance(stmt.body[0], LetDecl)
        assert isinstance(stmt.body[1], PassStmt)

    def test_inline_multi_stmts(self) -> None:
        src = "do[2] let x = 1; pass until true"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert stmt.limit == 2
        assert len(stmt.body) == 2

    def test_until_condition_is_expr(self) -> None:
        src = "do pass until status is Complete"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert isinstance(stmt.condition, IsTest)

    def test_zero_bound_rejected(self) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("do[0] pass until true")
        assert "positive" in str(exc_info.value).lower()

    def test_inline_do_with_trailing_if(self) -> None:
        """Prototype case: inline do body may end in if_stmt (sealed by until)."""
        src = "do[5] let r = a; if r = 1 => pass | else => pass until r = 1"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert stmt.limit == 5
        # Last stmt in body is IfStmt
        assert isinstance(stmt.body[-1], IfStmt)

    def test_inline_type_alias_survives_to_ast(self) -> None:
        """do[2] type T = int until true — TypeAlias must appear in body, not be dropped."""
        stmt = _parse_one("do[2] type T = int until true")
        assert isinstance(stmt, DoUntil)
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], TypeAlias)

    def test_inline_enum_survives_to_ast(self) -> None:
        """do[2] enum E | A until true — EnumDef must appear in body, not be dropped."""
        stmt = _parse_one("do[2] enum E | A until true")
        assert isinstance(stmt, DoUntil)
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], EnumDef)

    def test_inline_input_decl_survives_to_ast(self) -> None:
        """do[2] input x until true — InputDecl must appear in body, not be dropped."""
        stmt = _parse_one("do[2] input x until true")
        assert isinstance(stmt, DoUntil)
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], InputDecl)

    def test_bare_case_expr_after_until_rejected(self) -> None:
        """bare case_expr after until must be rejected (bar-safe violation)."""
        src = "do[2] pass until case s of\n  | A => true\n  | B => false"
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_parenthesized_case_expr_after_until_ok(self) -> None:
        """Parenthesized case_expr after until is accepted."""
        src = (
            "enum S\n  | A\n  | B\nlet s: S = A\n"
            "do[2] pass until (case s of | A => true | B => false)"
        )
        prog = parse_program(src)
        du = prog.body[-1]
        assert isinstance(du, DoUntil)
        assert isinstance(du.condition, CaseExpr)


# ---------------------------------------------------------------------------
# M3: if_stmt
# ---------------------------------------------------------------------------


class TestIfStmt:
    def test_simple_if_else(self) -> None:
        src = "if true => pass | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert len(stmt.branches) == 2
        b0, b1 = stmt.branches
        assert isinstance(b0, IfBranch)
        assert isinstance(b0.cond, BoolLit)
        assert b0.cond.value is True
        assert isinstance(b1, IfBranch)
        assert b1.cond is ELSE

    def test_single_condition_no_else(self) -> None:
        src = "if x => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert len(stmt.branches) == 1
        assert isinstance(stmt.branches[0].cond, VarRef)

    def test_multiple_conditions(self) -> None:
        src = "if a => pass | b => pass | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert len(stmt.branches) == 3
        assert stmt.branches[2].cond is ELSE

    def test_multiline_branches(self) -> None:
        src = "if true =>\n  pass\n| else =>\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert len(stmt.branches) == 2

    def test_branch_body_has_stmts(self) -> None:
        src = "if cond =>\n  let x = 1\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert len(stmt.branches) == 1
        assert len(stmt.branches[0].body) == 2

    def test_else_not_last_rejected(self) -> None:
        """else branch must be last — rejected with diagnostic about 'else'."""
        src = "let k = 1\nif k = 1 => pass\n| else => pass\n| k = 2 => pass"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        assert "else" in str(exc_info.value).lower()

    def test_nested_if_inline_branch_rejected(self) -> None:
        """Nested if in inline branch body is rejected (bar-safe violation)."""
        src = "if true => if false => pass | else => pass"
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_missing_arrow_rejected(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("if true pass")

    def test_inline_if(self) -> None:
        """if condition => stmt | else => stmt on one line."""
        src = "if x => pass | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)

    def test_design_3_3_if_inline(self) -> None:
        """Design §3.3: if code is Fail or design is Fail => ... | else => pass."""
        src = ("if code is Fail or design is Fail => pass | else => pass")
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        cond = stmt.branches[0].cond
        assert isinstance(cond, BinaryOp)
        assert cond.op == BinOp.OR


# ---------------------------------------------------------------------------
# M3: case_stmt
# ---------------------------------------------------------------------------


class TestCaseStmt:
    def test_simple_case(self) -> None:
        src = "case result of\n  | Pass => pass\n  | Fail => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, CaseStmt)
        assert isinstance(stmt.subject, VarRef)
        assert stmt.subject.name == "result"
        assert len(stmt.branches) == 2

    def test_wildcard_branch(self) -> None:
        src = "case x of\n  | _ => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, CaseStmt)
        b = stmt.branches[0]
        assert isinstance(b.pattern, WildcardPattern)

    def test_constructor_pattern_no_fields(self) -> None:
        src = "case result of\n  | Pass => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, ConstructorPattern)
        assert b.pattern.name == "Pass"
        assert b.pattern.qualifier is None

    def test_constructor_pattern_with_fields(self) -> None:
        src = "case result of\n  | Fail(issues) => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, ConstructorPattern)
        assert b.pattern.name == "Fail"
        assert len(b.pattern.fields) == 1
        f = b.pattern.fields[0]
        assert isinstance(f, PatternField)
        assert f.name == "issues"
        # Shorthand: Fail(issues) means issues: issues
        assert isinstance(f.pattern, VarPattern)
        assert f.pattern.name == "issues"

    def test_constructor_pattern_field_bind(self) -> None:
        src = "case result of\n  | Blocked(reason: why) => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, ConstructorPattern)
        f = b.pattern.fields[0]
        assert f.name == "reason"
        assert isinstance(f.pattern, VarPattern)
        assert f.pattern.name == "why"

    def test_qualified_constructor_pattern(self) -> None:
        src = "case result of\n  | Review.Pass => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, ConstructorPattern)
        assert b.pattern.qualifier == "Review"
        assert b.pattern.name == "Pass"

    def test_var_pattern(self) -> None:
        src = "case code of\n  | other => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, VarPattern)
        assert b.pattern.name == "other"

    def test_literal_pattern_int(self) -> None:
        src = "case code of\n  | 0 => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, LiteralPattern)
        assert isinstance(b.pattern.literal, IntLit)
        assert b.pattern.literal.value == 0

    def test_literal_pattern_bool(self) -> None:
        src = "case flag of\n  | true => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, LiteralPattern)
        assert isinstance(b.pattern.literal, BoolLit)
        assert b.pattern.literal.value is True

    def test_literal_pattern_str(self) -> None:
        src = 'case who of\n  | "alice" => pass'
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, LiteralPattern)
        assert isinstance(b.pattern.literal, StringLit)
        assert b.pattern.literal.value == "alice"

    def test_case_stmt_inline(self) -> None:
        src = "case result of | Pass => pass | Fail => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, CaseStmt)
        assert len(stmt.branches) == 2

    def test_stmt_in_case_expr_branch_rejected(self) -> None:
        """case expression branch must be an expression, not a statement."""
        src = (
            "enum S\n  | A\n  | B\nlet s: S = A\n"
            "let x = case s of\n  | A => let y = 1\n  | B => 2"
        )
        with pytest.raises(AglSyntaxError):
            parse_program(src)


# ---------------------------------------------------------------------------
# M3: case_expr
# ---------------------------------------------------------------------------


class TestCaseExpr:
    def test_simple_case_expr(self) -> None:
        src = "let x = case v of\n  | Pass => 1\n  | Fail => 0"
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        ce = stmt.value
        assert isinstance(ce, CaseExpr)
        assert isinstance(ce.subject, VarRef)
        assert len(ce.branches) == 2

    def test_case_expr_branch_body_is_expr(self) -> None:
        src = 'let x = case v of\n  | Pass => "ok"\n  | Fail => "no"'
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        ce = stmt.value
        assert isinstance(ce, CaseExpr)
        b0 = ce.branches[0]
        assert isinstance(b0, CaseExprBranch)
        assert isinstance(b0.body, StringLit)

    def test_case_expr_inline(self) -> None:
        src = "let x = case v of | Pass => 1 | Fail => 0"
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, CaseExpr)

    def test_bare_case_expr_at_block_level(self) -> None:
        """A bare case_expr at block level is valid (binding RHS is not bar-safe)."""
        src = "let x = case s of\n  | A => true\n  | B => false"
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, CaseExpr)

    def test_bare_case_expr_in_bar_position_rejected(self) -> None:
        """Bare case_expr in an if-branch body is rejected (bar-safe)."""
        src = "if ok => let x = case v of | a => 1 | b => 2"
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_parenthesized_case_expr_in_bar_position_ok(self) -> None:
        """Parenthesized case_expr in an if-branch body is accepted."""
        src = "if ok => let x = (case v of | a => 1 | b => 2) | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        b0 = stmt.branches[0]
        assert len(b0.body) == 1
        inner = b0.body[0]
        assert isinstance(inner, LetDecl)
        assert isinstance(inner.value, CaseExpr)


# ---------------------------------------------------------------------------
# M3: try_stmt / catch / raise
# ---------------------------------------------------------------------------


class TestTryCatch:
    def test_simple_try_catch(self) -> None:
        src = "try\n  pass\ncatch _ =>\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], PassStmt)
        assert len(stmt.handlers) == 1
        h = stmt.handlers[0]
        assert isinstance(h, CatchClause)
        assert h.exc_type is None  # wildcard
        assert h.binding is None

    def test_catch_with_binding(self) -> None:
        src = "try\n  pass\ncatch _ as e =>\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        h = stmt.handlers[0]
        assert h.exc_type is None
        assert h.binding == "e"

    def test_catch_type(self) -> None:
        src = "try\n  pass\ncatch AgentParseError =>\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        h = stmt.handlers[0]
        assert h.exc_type == "AgentParseError"
        assert h.binding is None

    def test_catch_type_with_binding(self) -> None:
        src = "try\n  pass\ncatch AgentParseError as e =>\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        h = stmt.handlers[0]
        assert h.exc_type == "AgentParseError"
        assert h.binding == "e"

    def test_multiple_catch_clauses(self) -> None:
        src = "try\n  pass\ncatch AgentParseError as e =>\n  pass\ncatch _ as e =>\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        assert len(stmt.handlers) == 2

    def test_inline_try(self) -> None:
        """Inline try body is closed statements only."""
        src = "try pass catch _ => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)

    def test_inline_try_multi_closed(self) -> None:
        src = "try let x = 1; pass catch _ => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        assert len(stmt.body) == 2

    def test_if_in_inline_catch_rejected(self) -> None:
        """An if_stmt in an inline catch body is rejected."""
        src = "try pass catch _ => if true => pass | else => pass"
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_open_stmt_in_inline_try_rejected(self) -> None:
        """An open stmt (if) in inline try body is rejected."""
        src = "try let x = 1; if true => pass | else => pass catch _ => pass"
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_try_body_multi_stmts(self) -> None:
        src = "try\n  let x = 1\n  let y = 2\n  pass\ncatch _ =>\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        assert len(stmt.body) == 3

    def test_try_with_bar_safe_catch_body(self) -> None:
        """Try-in-branch + bar-safe catch: prototype case §4.1."""
        src = "if ok => try pass catch _ => pass | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        b0 = stmt.branches[0]
        assert len(b0.body) == 1
        assert isinstance(b0.body[0], TryCatch)

    # --- Task 1: catch wildcard validation ---

    def test_catch_lowercase_name_rejected(self) -> None:
        """catch <lowercase-name> (not '_') must be rejected with a targeted error."""
        src = "try pass catch foo => pass"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        assert "'foo' is not an exception type name" in str(err)
        # Span must point at the offending name token.
        assert err.source_span.start_col == src.index("foo") + 1

    def test_catch_lowercase_name_message_includes_guidance(self) -> None:
        """Error message for a bad catch name must include guidance."""
        src = "try pass catch myerr => pass"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        msg = str(exc_info.value)
        assert "exception type" in msg

    def test_catch_wildcard_still_accepted(self) -> None:
        """catch _ => ... must still parse as a wildcard (exc_type=None)."""
        stmt = _parse_one("try pass catch _ => pass")
        assert isinstance(stmt, TryCatch)
        assert stmt.handlers[0].exc_type is None
        assert stmt.handlers[0].binding is None

    def test_catch_wildcard_with_binding_still_accepted(self) -> None:
        """catch _ as e => ... must still parse correctly."""
        stmt = _parse_one("try pass catch _ as e => pass")
        assert isinstance(stmt, TryCatch)
        assert stmt.handlers[0].exc_type is None
        assert stmt.handlers[0].binding == "e"

    def test_catch_type_name_still_accepted(self) -> None:
        """catch AgentParseError as e => ... must still parse correctly."""
        stmt = _parse_one("try\n  pass\ncatch AgentParseError as e =>\n  pass")
        assert isinstance(stmt, TryCatch)
        assert stmt.handlers[0].exc_type == "AgentParseError"
        assert stmt.handlers[0].binding == "e"


# ---------------------------------------------------------------------------
# M3: Rejection fixtures (acceptance suite integration)
# ---------------------------------------------------------------------------


class TestRejectionFixtures:
    """Verify that the acceptance rejection fixtures fail as expected.

    Each fixture has a .agl file and a .expect.json describing the expected
    diagnostic (line number and/or message_contains).
    """

    def _parse_and_check(
        self,
        src: str,
        expected_line: int | None,
        message_contains: list[str] | None,
    ) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        err = exc_info.value
        if expected_line is not None:
            assert err.source_span.start_line == expected_line, (
                f"Expected line {expected_line}, got {err.source_span.start_line}; "
                f"message: {err}"
            )
        if message_contains:
            msg = str(err).lower()
            for fragment in message_contains:
                assert fragment.lower() in msg, (
                    f"Expected {fragment!r} in error message: {err}"
                )

    def test_bare_case_after_until(self) -> None:
        src = (
            "enum S\n  | A\n  | B\n"
            "let s: S = A\n"
            "var n: int = 0\n"
            "do[2] set n = n + 1 until case s of | A => true | B => false"
        )
        self._parse_and_check(src, expected_line=6, message_contains=None)

    def test_chained_comparison(self) -> None:
        src = "let x = 1\nlet ok = (x = 1 = 2)"
        self._parse_and_check(src, expected_line=2, message_contains=None)

    def test_else_not_last(self) -> None:
        src = "let k = 1\nif k = 1 => pass\n| else => pass\n| k = 2 => pass"
        self._parse_and_check(src, expected_line=None, message_contains=["else"])

    def test_if_in_inline_catch(self) -> None:
        src = "try pass catch _ => if true => pass | else => pass"
        self._parse_and_check(src, expected_line=1, message_contains=None)

    def test_missing_arrow(self) -> None:
        src = "if true pass"
        self._parse_and_check(src, expected_line=1, message_contains=None)

    def test_nested_if_inline_branch(self) -> None:
        src = "if true => if false => pass | else => pass"
        self._parse_and_check(src, expected_line=1, message_contains=None)

    def test_open_stmt_in_inline_try(self) -> None:
        src = "try let x = 1; if true => pass | else => pass catch _ => pass"
        self._parse_and_check(src, expected_line=1, message_contains=None)

    def test_stmt_in_case_expr_branch(self) -> None:
        src = (
            "enum S\n  | A\n  | B\n"
            "let s: S = A\n"
            "let x = case s of\n  | A => let y = 1\n  | B => 2"
        )
        self._parse_and_check(src, expected_line=6, message_contains=None)

    def test_zero_loop_bound(self) -> None:
        src = "do[0] pass until true"
        self._parse_and_check(src, expected_line=1, message_contains=["positive"])

    def test_bare_assignment_rejected(self) -> None:
        """bare `n = 2` is rejected at parse time as a bare assignment attempt."""
        src = "var n: int = 1\nn = 2"
        self._parse_and_check(src, expected_line=2, message_contains=None)

    def test_annotated_assignment_rejected(self) -> None:
        """annotated assignment `count: int = 5` is rejected at parse time."""
        src = "count: int = 5"
        with pytest.raises(AglSyntaxError):
            parse_program(src)


# ---------------------------------------------------------------------------
# M3: Design §3 canonical examples (parse-level check)
# ---------------------------------------------------------------------------


class TestDesignExamples:
    def test_design_3_1_one_liner(self) -> None:
        """Design §3.1 inline form."""
        src = (
            'do[5] let status: Status = ask[on_parse_error: retry[2]] "Do X."'
            " until status is Complete"
        )
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert stmt.limit == 5
        assert isinstance(stmt.condition, IsTest)
        assert stmt.condition.variant == "Complete"

    def test_design_3_2_do_loop_case(self) -> None:
        """Design §3.2 one-liner prototype case (prototype test_conflicts.py §14)."""
        src = (
            "var artifact: text = impl \"Implement ${spec}\"; "
            "do[5] let review: Review = reviewer[on_parse_error: retry[2]] \"Review ${artifact}\"; "
            "case review of "
            "| Fail(issues) => set artifact = impl \"Fix ${issues} in ${artifact}\" "
            "| Pass => pass "
            "until review is Pass"
        )
        prog = parse_program(src)
        assert len(prog.body) == 2
        du = prog.body[1]
        assert isinstance(du, DoUntil)
        # body has: let review + case review
        assert len(du.body) == 2
        assert isinstance(du.body[0], LetDecl)
        assert isinstance(du.body[1], CaseStmt)

    def test_design_3_4_case_with_fields(self) -> None:
        src = (
            "case review of\n"
            "  | Pass => pass\n"
            "  | Fail(issues) => set artifact = impl \"Fix:\\n${issues}\""
        )
        stmt = _parse_one(src)
        assert isinstance(stmt, CaseStmt)
        assert len(stmt.branches) == 2
        b1 = stmt.branches[1]
        assert isinstance(b1.pattern, ConstructorPattern)
        assert b1.pattern.name == "Fail"
        assert len(b1.pattern.fields) == 1

    def test_design_3_5_case_expr(self) -> None:
        src = (
            "let next_prompt: text = case action of\n"
            "  | Stop => \"Stop.\"\n"
            "  | Continue(prompt) => prompt\n"
            '  | Escalate(reason) => "Investigate blocker:\\n${reason}"'
        )
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, CaseExpr)
        assert len(stmt.value.branches) == 3

    def test_design_3_7_equality(self) -> None:
        src = (
            "let expected_count = 3\n"
            "let actual_count: int = 5\n"
            "if actual_count = expected_count => pass\n"
            "| else =>\n"
            '  raise Abort(message: "unexpected count")'
        )
        prog = parse_program(src)
        assert len(prog.body) == 3
        if_stmt = prog.body[2]
        assert isinstance(if_stmt, IfStmt)
        cond = if_stmt.branches[0].cond
        assert isinstance(cond, BinaryOp)
        assert cond.op == BinOp.EQ

    def test_design_3_8_try_catch(self) -> None:
        src = (
            "try\n"
            "  let review: text = reviewer \"Review ${artifact}\"\n"
            "catch AgentParseError as e =>\n"
            "  raise e"
        )
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        assert len(stmt.handlers) == 1
        h = stmt.handlers[0]
        assert h.exc_type == "AgentParseError"
        assert h.binding == "e"

    def test_design_section_14_canonical(self) -> None:
        """Parse the design §14 canonical review_fix.agl program."""
        import pathlib

        src_path = pathlib.Path("tests/agl/programs/canonical/review_fix.agl")
        src = src_path.read_text(encoding="utf-8")
        prog = parse_program(src)
        assert isinstance(prog, Program)
        assert len(prog.body) > 0

    def test_inline_multiline_equivalence(self) -> None:
        """Inline semicolons produce the same AST as newlines for do/if."""
        inline = "if a => pass | b => pass"
        multiline = "if a => pass\n| b => pass"
        p1 = parse_program(inline)
        p2 = parse_program(multiline)
        assert p1.body[0] == p2.body[0]


# ---------------------------------------------------------------------------
# M3: Pattern matching edge cases
# ---------------------------------------------------------------------------


class TestPatterns:
    def test_nested_constructor_pattern(self) -> None:
        """Nested constructor pattern: Inner(shape: Line(len: n))."""
        src = "case wrapped of\n  | Inner(shape: Line(len: n)) => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, CaseStmt)
        b = stmt.branches[0]
        outer = b.pattern
        assert isinstance(outer, ConstructorPattern)
        assert outer.name == "Inner"
        assert len(outer.fields) == 1
        inner_field = outer.fields[0]
        assert inner_field.name == "shape"
        inner_pat = inner_field.pattern
        assert isinstance(inner_pat, ConstructorPattern)
        assert inner_pat.name == "Line"
        assert inner_pat.fields[0].name == "len"
        bound_var = inner_pat.fields[0].pattern
        assert isinstance(bound_var, VarPattern)
        assert bound_var.name == "n"

    def test_multiple_field_pattern(self) -> None:
        src = "case result of\n  | Box(w, h, label) => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, ConstructorPattern)
        assert len(b.pattern.fields) == 3

    def test_pattern_field_rename(self) -> None:
        src = "case result of\n  | Box(label: tag) => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        f = b.pattern.fields[0]
        assert f.name == "label"
        assert isinstance(f.pattern, VarPattern)
        assert f.pattern.name == "tag"

    def test_wildcard_in_case_stmt(self) -> None:
        src = "case x of\n  | _ => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, WildcardPattern)

    def test_var_pattern_in_case(self) -> None:
        src = "case code of\n  | 0 => pass\n  | other => pass"
        stmt = _parse_one(src)
        b = stmt.branches[1]
        assert isinstance(b.pattern, VarPattern)
        assert b.pattern.name == "other"


# ---------------------------------------------------------------------------
# M3: Program files (control and canonical)
# ---------------------------------------------------------------------------


class TestM3ProgramFiles:
    @staticmethod
    def _read(rel: str) -> str:
        import pathlib
        return pathlib.Path(rel).read_text(encoding="utf-8")

    def test_parse_do_until_agl(self) -> None:
        prog = parse_program(self._read("tests/agl/programs/control/do_until.agl"))
        assert isinstance(prog, Program)
        assert any(isinstance(s, DoUntil) for s in prog.body)

    def test_parse_case_patterns_agl(self) -> None:
        prog = parse_program(self._read("tests/agl/programs/control/case_patterns.agl"))
        assert isinstance(prog, Program)
        assert any(isinstance(s, CaseStmt) for s in prog.body)

    def test_parse_case_expr_agl(self) -> None:
        prog = parse_program(self._read("tests/agl/programs/control/case_expr.agl"))
        assert isinstance(prog, Program)
        assert any(
            isinstance(s, LetDecl) and isinstance(s.value, CaseExpr) for s in prog.body
        )

    def test_parse_nested_control_agl(self) -> None:
        prog = parse_program(self._read("tests/agl/programs/control/nested_control.agl"))
        assert isinstance(prog, Program)

    def test_parse_review_fix_agl(self) -> None:
        prog = parse_program(self._read("tests/agl/programs/canonical/review_fix.agl"))
        assert isinstance(prog, Program)

    def test_parse_one_liner_agl(self) -> None:
        prog = parse_program(self._read("tests/agl/programs/canonical/one_liner.agl"))
        assert isinstance(prog, Program)

    def test_parse_multi_agent_agl(self) -> None:
        prog = parse_program(self._read("tests/agl/programs/canonical/multi_agent.agl"))
        assert isinstance(prog, Program)

    def test_parse_dialogue_agl(self) -> None:
        src = self._read("tests/agl/programs/canonical/dialogue.agl")
        prog = parse_program(src)
        assert isinstance(prog, Program)


# ---------------------------------------------------------------------------
# M3: Coverage completion — edge cases for full 100% branch coverage
# ---------------------------------------------------------------------------


class TestM3CoverageEdgeCases:
    """Tests that exercise specific uncovered branches in transform.py."""

    def test_bar_var_decl_in_if_branch(self) -> None:
        """bar_var_decl: var declaration in a bar-safe inline branch body."""
        src = "if cond => var x: int = 1 | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        b0 = stmt.branches[0]
        assert len(b0.body) == 1
        assert isinstance(b0.body[0], VarDecl)

    def test_bar_set_stmt_in_if_branch(self) -> None:
        """bar_set_stmt: set in a bar-safe inline branch body."""
        src = "var x: int = 0\nif cond => set x = 1 | else => pass"
        prog = parse_program(src)
        if_stmt = prog.body[1]
        assert isinstance(if_stmt, IfStmt)
        b0 = if_stmt.branches[0]
        assert isinstance(b0.body[0], SetStmt)

    def test_bar_print_stmt_in_if_branch(self) -> None:
        """bar_print_stmt: print in a bar-safe inline branch body."""
        src = 'if cond => print "hello" | else => pass'
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert isinstance(stmt.branches[0].body[0], PrintStmt)

    def test_bar_raise_stmt_in_if_branch(self) -> None:
        """bar_raise_stmt: raise in a bar-safe inline branch body."""
        src = "if cond => raise e | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert isinstance(stmt.branches[0].body[0], Raise)

    def test_pat_lit_decimal(self) -> None:
        """LiteralPattern with a decimal literal."""
        src = "case x of\n  | 3.14 => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, CaseStmt)
        b = stmt.branches[0]
        assert isinstance(b.pattern, LiteralPattern)
        assert isinstance(b.pattern.literal, DecimalLit)

    def test_pat_lit_false(self) -> None:
        """LiteralPattern with false literal."""
        src = "case x of\n  | false => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, LiteralPattern)
        assert isinstance(b.pattern.literal, BoolLit)
        assert b.pattern.literal.value is False

    def test_pat_lit_null(self) -> None:
        """LiteralPattern with null literal."""
        src = "case x of\n  | null => pass"
        stmt = _parse_one(src)
        b = stmt.branches[0]
        assert isinstance(b.pattern, LiteralPattern)
        assert isinstance(b.pattern.literal, NullLit)

    def test_pat_lit_str_with_interp_rejected(self) -> None:
        """Pattern string literal with interpolation is rejected."""
        src = 'case x of\n  | "${y}" => pass'
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_branch_body_returns_try_stmt(self) -> None:
        """branch_body containing a try_stmt is accepted."""
        src = "if ok => try pass catch _ => pass | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        b0 = stmt.branches[0]
        assert len(b0.body) == 1
        assert isinstance(b0.body[0], TryCatch)

    def test_catch_body_with_suite(self) -> None:
        """catch_body with a suite (indented block) is handled."""
        src = "try\n  pass\ncatch _ =>\n  let x = 1\n  pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        h = stmt.handlers[0]
        assert len(h.body) == 2

    def test_catch_body_inline_bar_closed(self) -> None:
        """catch_body with inline bar_closed_stmt sets body correctly."""
        src = "try pass catch _ => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        assert len(stmt.handlers[0].body) == 1
        assert isinstance(stmt.handlers[0].body[0], PassStmt)

    def test_try_stmt_with_two_handlers(self) -> None:
        """try_stmt with two catch_clauses covers the CatchClause append path."""
        src = (
            "try\n  pass\n"
            "catch AgentParseError as e =>\n  pass\n"
            "catch _ =>\n  pass"
        )
        stmt = _parse_one(src)
        assert isinstance(stmt, TryCatch)
        assert len(stmt.handlers) == 2

    def test_do_until_multiline_suite(self) -> None:
        """do_until with suite body: covers the suite non-None path."""
        src = "do[2]\n  let x = 1\n  pass\nuntil true"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert len(stmt.body) == 2

    def test_if_branch_with_suite(self) -> None:
        """branch_body receiving a suite returns tuple from it."""
        src = "if cond =>\n  let x = 1\n  pass\n| else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        b0 = stmt.branches[0]
        assert len(b0.body) == 2

    def test_case_stmt_branch_with_suite(self) -> None:
        """case_stmt_branch with suite body."""
        src = "case result of\n  | Pass =>\n    let x = 1\n    pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, CaseStmt)
        b = stmt.branches[0]
        assert len(b.body) == 2


# ---------------------------------------------------------------------------
# F8: Targeted diagnostics for chained comparison and bar-safe violations
# ---------------------------------------------------------------------------


class TestTargetedDiagnostics:
    """F8/§4.4: the design promises targeted diagnostic messages.

    Chained comparison (x = y = z) is non-associative per §4.3/§12.5 and
    must produce a targeted error about non-associativity, not a generic
    "Unexpected token" message.
    """

    def _assert_chained_comparison_message(self, source: str) -> None:
        """Parse *source*, assert it raises AglSyntaxError with a targeted message."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(source)
        err = exc_info.value
        msg = str(err).lower()
        # The message must reference associativity, parenthesization, or comparison.
        assert (
            "non-associative" in msg
            or "associative" in msg
            or "parenthes" in msg
            or "comparison" in msg
            or "chain" in msg
        ), f"Expected targeted chained-comparison message, got: {err}"
        # Must not be a plain "unexpected token" message.
        assert "unexpected token" not in msg or any(
            kw in msg for kw in ("non-associative", "parenthes", "comparison", "chain")
        ), f"Message is too generic: {err}"

    def test_chained_comparison_targeted_message(self) -> None:
        """Chained EQ rejects with a targeted non-associative message (x = y = z)."""
        self._assert_chained_comparison_message("let x = 1\nlet ok = (x = 1 = 2)")

    def test_chained_lt_targeted_message(self) -> None:
        """Chained LT rejects with a targeted non-associative message (1 < 2 < 3)."""
        self._assert_chained_comparison_message("let ok = (1 < 2 < 3)")

    def test_chained_le_neq_targeted_message(self) -> None:
        """Mixed chained comparison rejects with a targeted message (a <= b != c)."""
        source = "let a = 1\nlet b = 2\nlet c = 3\nlet ok = (a <= b != c)"
        self._assert_chained_comparison_message(source)


class TestInlineCompoundDiagnostics:
    """§12.5 item 9 / §4.4: bar-safe inline-form rejections get targeted text.

    A nested ``if`` / ``case`` / ``try`` in an inline (bar-safe) position must
    not surface the generic "Unexpected token" fallback with a raw Lark
    ``Token`` repr.  It must produce a targeted, honest message that
    distinguishes a statement position (write an indented block) from an
    expression position (parenthesize a ``case`` expression).
    """

    def _error(self, source: str) -> AglSyntaxError:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(source)
        return exc_info.value

    def test_nested_if_in_inline_branch(self) -> None:
        """Nested ``if`` in an inline ``=>`` branch body → suite guidance."""
        err = self._error("if true => if false => pass | else => pass")
        msg = str(err)
        assert msg == (
            "`if` is not allowed inline here; "
            "write it as an indented block instead."
        )
        assert "Token(" not in msg

    def test_if_as_inline_catch_body(self) -> None:
        """``if`` as an inline ``catch`` body → suite guidance."""
        err = self._error("try pass catch _ => if true => pass | else => pass")
        assert str(err) == (
            "`if` is not allowed inline here; "
            "write it as an indented block instead."
        )

    def test_nested_inline_try(self) -> None:
        """A nested inline ``try`` → suite guidance."""
        err = self._error("try try pass catch _ => pass catch _ => pass")
        assert str(err) == (
            "`try` is not allowed inline here; "
            "write it as an indented block instead."
        )

    def test_open_statement_with_if_in_inline_try(self) -> None:
        """An open statement followed by ``if`` in an inline ``try`` → suite guidance."""
        src = "try let x = 1; if true => pass | else => pass catch _ => pass"
        assert str(self._error(src)) == (
            "`if` is not allowed inline here; "
            "write it as an indented block instead."
        )

    def test_bare_case_after_until(self) -> None:
        """A bare ``case`` expression after ``until`` → parenthesize guidance."""
        src = (
            "enum S\n  | A\n  | B\nlet s: S = A\nvar n: int = 0\n"
            "do[2] set n = n + 1 until case s of | A => true | B => false"
        )
        err = self._error(src)
        assert str(err) == (
            "`case` is not allowed inline here; "
            "parenthesize the case expression, e.g. `(case x of ...)`."
        )
        assert err.source_span.start_line == 6

    def test_if_in_expression_position_now_valid(self) -> None:
        """``if`` in an expression position is now valid (if_expr form)."""
        # Since if_expr was added, `let x = if cond => 1 | else => 2` parses
        # as a LetDecl whose value is an IfExpr — not a syntax error.
        stmt = _parse_one("let x = if true => 1 | else => 2")
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, IfExpr)

    def test_try_in_expression_position_suite_guidance(self) -> None:
        """``try`` where an expression is expected → suite guidance (no expr form)."""
        err = self._error("let x = try 1 catch _ => 2")
        assert str(err) == (
            "`try` is not allowed inline here; "
            "write it as an indented block instead."
        )

    def test_friendly_fallback_renders_token_value(self) -> None:
        """The generic fallback renders the friendly token value, not a Token repr."""
        src = (
            "enum S\n  | A\n  | B\nlet s: S = A\n"
            "let x = case s of\n  | A => let y = 1\n  | B => 2"
        )
        err = self._error(src)
        msg = str(err)
        assert msg == "Unexpected 'let'."
        assert "Token(" not in msg


# ---------------------------------------------------------------------------
# Seeded / incremental parsing (REPL pass seam)
# ---------------------------------------------------------------------------


class TestParseProgramSeeded:
    """``parse_program_seeded`` keeps node ids unique across incremental entries."""

    def test_default_start_id_preserves_existing_ids(self) -> None:
        """``start_id=0`` (the default) yields the same ids as ``parse_program``."""
        src = "let x = 1\nlet y = 2\nprint x"
        baseline = parse_program(src)
        prog, _next_id = parse_program_seeded(src, start_id=0)
        assert all_node_ids(prog) == all_node_ids(baseline)

    def test_consecutive_calls_produce_disjoint_ranges(self) -> None:
        """Two consecutive seeded parses produce disjoint node-id ranges."""
        prog1, next1 = parse_program_seeded("let x = 1", start_id=0)
        prog2, next2 = parse_program_seeded("let y = 2", start_id=next1)
        ids1 = all_node_ids(prog1)
        ids2 = all_node_ids(prog2)
        assert ids1.isdisjoint(ids2)
        # The second range begins at the first range's reported next seed.
        assert min(ids2) >= next1
        assert next2 > next1

    def test_next_start_id_is_first_unconsumed(self) -> None:
        """``next_start_id`` is the first id NOT consumed, derived from the counter.

        For a simple program the root holds the maximum id, so the next seed is
        ``program.node_id + 1``; the value is read from the builder counter, not
        hardcoded to that assumption.
        """
        prog, next_id = parse_program_seeded("let x = 1", start_id=0)
        assert next_id == max(all_node_ids(prog)) + 1
        assert next_id == prog.node_id + 1

    def test_start_id_offsets_all_ids(self) -> None:
        """A non-zero ``start_id`` shifts every node id by that offset."""
        base, _ = parse_program_seeded("let x = 1", start_id=0)
        shifted, _ = parse_program_seeded("let x = 1", start_id=100)
        base_ids = sorted(all_node_ids(base))
        shifted_ids = sorted(all_node_ids(shifted))
        assert shifted_ids == [i + 100 for i in base_ids]

    def test_parse_program_accepts_start_id(self) -> None:
        """``parse_program`` honours ``start_id`` while returning only the program."""
        prog = parse_program("let x = 1", start_id=50)
        assert min(all_node_ids(prog)) >= 50

    def test_parse_error_lets_caller_reuse_start_id(self) -> None:
        """A failed seeded parse raises and never advances the caller's seed.

        Contract: ``parse_program_seeded`` returns ``next_start_id`` only on
        success, so a syntax error leaves the caller free to reuse the SAME
        ``start_id`` for the corrected entry — which then yields the expected
        disjoint range as if the bad call never happened.
        """
        # A bad entry at start_id=10 raises (no next_start_id is produced).
        with pytest.raises(AglSyntaxError):
            parse_program_seeded("let x ==", start_id=10)
        # Reusing the same start_id for a valid entry yields ids at/above it,
        # disjoint from a subsequent entry seeded with its reported next id.
        prog, next_id = parse_program_seeded("let x = 1", start_id=10)
        assert min(all_node_ids(prog)) >= 10
        prog2, _ = parse_program_seeded("let y = 2", start_id=next_id)
        assert all_node_ids(prog).isdisjoint(all_node_ids(prog2))


class TestIsIncompleteSource:
    """The structured incompleteness signal used by the REPL multiline predicate."""

    @pytest.mark.parametrize(
        "source",
        [
            "record R",  # unterminated block header ($END)
            "enum E",
            "case x of",
            "try",
            "do agent",
            "if x = 1 =>",  # block-opening arrow, no body yet
            "if x = 1 =>\n",  # newline after =>, awaiting an INDENT
            "1 +",  # dangling binary operator
            "let x =",  # open initializer
        ],
    )
    def test_incomplete_sources(self, source: str) -> None:
        assert is_incomplete_source(source) is True

    @pytest.mark.parametrize(
        "source",
        [
            "1 + 2",  # a clean parse
            "let x = 1",
            "record R\n  x: int",  # a full block
            "let = 5",  # a real error mid-line (not $END / INDENT)
            "x == y",  # the == friendly-error case
            '"unterminated',  # a lexical error → submit so the user sees it
        ],
    )
    def test_complete_or_erroring_sources(self, source: str) -> None:
        assert is_incomplete_source(source) is False


# ---------------------------------------------------------------------------
# if_stmt optional leading pipe
# ---------------------------------------------------------------------------


class TestIfStmtLeadingPipe:
    def test_leading_pipe_two_branches_same_as_no_pipe(self) -> None:
        """if | A => B | else => C parses to same IfStmt as if A => B | else => C."""
        src_pipe = "if | true => pass | else => pass"
        src_no_pipe = "if true => pass | else => pass"
        stmt_pipe = _parse_one(src_pipe)
        stmt_no_pipe = _parse_one(src_no_pipe)
        assert isinstance(stmt_pipe, IfStmt)
        assert isinstance(stmt_no_pipe, IfStmt)
        # Structural equality ignores span/node_id (compare=False)
        assert stmt_pipe == stmt_no_pipe

    def test_leading_pipe_single_branch(self) -> None:
        """Single-branch if with leading pipe: if | A => B."""
        src = "if | x => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert len(stmt.branches) == 1
        assert isinstance(stmt.branches[0].cond, VarRef)

    def test_leading_pipe_three_branches(self) -> None:
        """Three-branch if with leading pipe."""
        src = "if | a => pass | b => pass | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert len(stmt.branches) == 3
        assert stmt.branches[2].cond is ELSE

    def test_leading_pipe_multiline(self) -> None:
        """Leading pipe on its own line (layout continuation)."""
        src = "if\n| cond => pass\n| else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        assert len(stmt.branches) == 2
        assert stmt.branches[1].cond is ELSE

    def test_else_not_last_with_leading_pipe_rejected(self) -> None:
        """else-not-last is a syntax error even with leading pipe."""
        src = "let k = 1\nif | else => pass\n| k = 1 => pass"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        assert "else" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# if_expr
# ---------------------------------------------------------------------------


class TestIfExpr:
    def test_if_expr_in_let_binding(self) -> None:
        """let x = if A => 1 | else => 2 produces LetDecl with IfExpr."""
        src = "let x = if a => 1 | else => 2"
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        ie = stmt.value
        assert isinstance(ie, IfExpr)
        assert len(ie.branches) == 2
        b0, b1 = ie.branches
        assert isinstance(b0, IfExprBranch)
        assert isinstance(b0.cond, VarRef)
        assert b0.cond.name == "a"
        assert isinstance(b0.body, IntLit)
        assert b0.body.value == 1
        assert isinstance(b1, IfExprBranch)
        assert b1.cond is ELSE
        assert isinstance(b1.body, IntLit)
        assert b1.body.value == 2

    def test_if_expr_with_leading_pipe(self) -> None:
        """print (if | A => 1 | else => 2) uses parenthesized if_expr."""
        src = "print (if | a => 1 | else => 2)"
        stmt = _parse_one(src)
        assert isinstance(stmt, PrintStmt)
        ie = stmt.value
        assert isinstance(ie, IfExpr)
        assert len(ie.branches) == 2

    def test_bare_if_expr_in_print(self) -> None:
        """bare print if A => 1 | else => 2 is legal (print takes general expr)."""
        src = "print if a => 1 | else => 2"
        stmt = _parse_one(src)
        assert isinstance(stmt, PrintStmt)
        assert isinstance(stmt.value, IfExpr)

    def test_if_expr_nested_in_list(self) -> None:
        """if_expr nested inside a list literal."""
        src = "let xs = [if a => 1 | else => 2, 3]"
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        lst = stmt.value
        assert isinstance(lst, ListLit)
        assert isinstance(lst.elements[0], IfExpr)

    def test_if_expr_nested_in_interpolation(self) -> None:
        """if_expr nested inside a template interpolation."""
        src = 'let s = "${if a => 1 | else => 2}"'
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        tmpl = stmt.value
        assert isinstance(tmpl, Template)
        seg = tmpl.segments[0]
        assert isinstance(seg, InterpSegment)
        assert isinstance(seg.expr, IfExpr)

    def test_if_expr_nested_in_dict(self) -> None:
        """if_expr nested as a dict value."""
        src = 'let d = {"k": if a => 1 | else => 2}'
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        dct = stmt.value
        assert isinstance(dct, DictLit)
        assert len(dct.entries) == 1
        entry = dct.entries[0]
        ie = entry.value
        assert isinstance(ie, IfExpr)
        assert len(ie.branches) == 2
        b0, b1 = ie.branches
        assert isinstance(b0, IfExprBranch)
        assert isinstance(b0.cond, VarRef)
        assert b0.cond.name == "a"
        assert isinstance(b0.body, IntLit)
        assert b0.body.value == 1
        assert isinstance(b1, IfExprBranch)
        assert b1.cond is ELSE
        assert isinstance(b1.body, IntLit)
        assert b1.body.value == 2

    def test_if_expr_in_bar_safe_condition_requires_parens(self) -> None:
        """Unparenthesized if_expr as an if condition (bar-safe) is a syntax error."""
        # An if_expr used as the condition of an if_stmt branch body is bar-safe.
        # The condition itself is bar_expr which does NOT admit if_expr.
        src = "if if a => true | else => false => pass"
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_if_expr_parenthesized_in_bar_safe_position_ok(self) -> None:
        """Parenthesized if_expr as condition of if_stmt is accepted."""
        src = "if (if a => true | else => false) => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        cond = stmt.branches[0].cond
        assert isinstance(cond, IfExpr)

    def test_if_expr_in_branch_body_requires_parens(self) -> None:
        """Unparenthesized if_expr as a branch body (bar-safe) is a syntax error."""
        src = "if ok => if a => 1 | else => 2"
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_if_expr_parenthesized_in_branch_body_ok(self) -> None:
        """Parenthesized if_expr in a branch body is accepted."""
        src = "if ok => let x = (if a => 1 | else => 2) | else => pass"
        stmt = _parse_one(src)
        assert isinstance(stmt, IfStmt)
        b0 = stmt.branches[0]
        inner = b0.body[0]
        assert isinstance(inner, LetDecl)
        assert isinstance(inner.value, IfExpr)

    def test_if_expr_else_not_last_rejected(self) -> None:
        """else-not-last in if_expr is a syntax error."""
        src = "let x = if else => 1 | a => 2"
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(src)
        assert "else" in str(exc_info.value).lower()

    def test_if_expr_with_leading_pipe_branches(self) -> None:
        """if_expr with leading pipe: let x = if | a => 1 | else => 2."""
        src = "let x = if | a => 1 | else => 2"
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        ie = stmt.value
        assert isinstance(ie, IfExpr)
        assert len(ie.branches) == 2

    def test_if_expr_ast_structure(self) -> None:
        """Transformer produces correct IfExpr/IfExprBranch AST shape."""
        src = "let x = if cond => 10 | else => 20"
        stmt = _parse_one(src)
        assert isinstance(stmt, LetDecl)
        ie = stmt.value
        assert isinstance(ie, IfExpr)
        assert len(ie.branches) == 2
        cond_branch, else_branch = ie.branches
        assert isinstance(cond_branch, IfExprBranch)
        assert isinstance(cond_branch.cond, VarRef)
        assert cond_branch.cond.name == "cond"
        assert isinstance(cond_branch.body, IntLit)
        assert cond_branch.body.value == 10
        assert isinstance(else_branch, IfExprBranch)
        assert else_branch.cond is ELSE
        assert isinstance(else_branch.body, IntLit)
        assert else_branch.body.value == 20

    def test_if_expr_in_until_condition_requires_parens(self) -> None:
        """Unparenthesized if_expr after 'until' is a syntax error (bar-safe)."""
        src = "do pass until if a => true | else => false"
        with pytest.raises(AglSyntaxError):
            parse_program(src)

    def test_if_expr_parenthesized_after_until_ok(self) -> None:
        """Parenthesized if_expr after 'until' is accepted."""
        src = "do pass until (if a => true | else => false)"
        stmt = _parse_one(src)
        assert isinstance(stmt, DoUntil)
        assert isinstance(stmt.condition, IfExpr)
