"""Tests for the AgL v2 parser (agm.agl.parser) — Component 2.

Covers:
- LALR(1) conflict-guard: zero shift/reduce and reduce/reduce conflicts.
- Parsing v2 constructs to the expected AST shape.
- Records, enums, type aliases, constructors, field access, lists, dicts.
- Templates: plain text, interpolations.
- Uniform calls: paren-call with positional+named args; single-arg sugar.
- Function declarations (def) and lambda expressions.
- Type expressions: primitives, list[T], dict[text, T], func_type, unit, agent.
- Control flow: if/case/do/try expressions; suite bodies; multi-line branches.
- Binders: let/var/set.
- Input/agent/config declarations.
- Error cases: == produces friendly AglSyntaxError; bad syntax raises AglSyntaxError.
- REPL seam: parse_program_seeded and is_incomplete_source.
- Negative cases: f a b (juxt does not chain); print classify(x) (call needs parens).

NOTE: This file must NOT modify tests/test_agl_e2e.py or tests/agl/.
      No static-analysis suppression comments in this file.
"""

from __future__ import annotations

import decimal
import importlib.resources
import logging
import logging.handlers

import pytest

from agm.agl.parser import (
    AglSyntaxError,
    is_incomplete_source,
    parse_program,
    parse_program_seeded,
)
from agm.agl.syntax import (
    AgentDecl,
    BinaryOp,
    BinOp,
    Block,
    BoolLit,
    Call,
    Case,
    CaseBranch,
    CatchClause,
    ConfigPragma,
    Constructor,
    ConstructorPattern,
    DecimalLit,
    DictEntry,
    DictLit,
    Do,
    EnumDef,
    FieldAccess,
    FieldDef,
    FuncDef,
    If,
    IfBranch,
    InputDecl,
    InterpSegment,
    IntLit,
    IsTest,
    Lambda,
    LetDecl,
    ListLit,
    LiteralPattern,
    NamedArg,
    NullLit,
    Param,
    PatternField,
    Program,
    Raise,
    RecordDef,
    SetStmt,
    StringLit,
    Template,
    TextSegment,
    Try,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.nodes import ELSE
from agm.agl.syntax.types import (
    AgentT,
    BoolT,
    DecimalT,
    DictT,
    FuncT,
    IntT,
    JsonT,
    ListT,
    NameT,
    TextT,
    UnitT,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def parse(src: str) -> Program:
    """Parse *src* and return the Program root."""
    return parse_program(src.strip())


def items(prog: Program) -> tuple[object, ...]:
    """Return the top-level block items."""
    return prog.body.items


def first(prog: Program) -> object:
    """Return the first top-level item."""
    return prog.body.items[0]


# ---------------------------------------------------------------------------
# Conflict guard — MANDATORY regression
# ---------------------------------------------------------------------------


class TestConflictGuard:
    """Asserts the grammar has 0 shift/reduce and 0 reduce/reduce conflicts.

    The Lark LALR parser emits conflict warnings at DEBUG level; this test
    captures that log stream and verifies it is clean.  Any conflict message
    causes an immediate failure so a regression is caught before it ships.
    """

    def test_zero_conflicts(self) -> None:
        """Build the Lark parser and assert the debug log contains no conflicts.

        Two kinds of LALR(1) conflict manifest differently:
        - Shift/Reduce: Lark logs a DEBUG message containing "Shift/Reduce".
        - Reduce/Reduce: Lark raises ``lark.exceptions.GrammarError`` at
          parser-construction time rather than logging.

        Both are caught and surfaced as explicit pytest failures so that any
        regression introduced by a grammar change is immediately visible.
        """
        import io

        grammar_text = (
            importlib.resources.files("agm.agl")
            .joinpath("grammar/agl.lark")
            .read_text(encoding="utf-8")
        )

        from lark import Lark
        from lark.exceptions import GrammarError

        from agm.agl.lexer.lexer import AglLexer

        # Capture the lark.grammar / lark.parsers.lalr_analysis DEBUG stream
        # so that Shift/Reduce conflicts (which are logged, not raised) are caught.
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        root_logger = logging.getLogger("lark")
        old_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(handler)
        try:
            Lark(
                grammar_text,
                parser="lalr",
                lexer=AglLexer,
                propagate_positions=True,
                maybe_placeholders=True,
            )
        except GrammarError as exc:
            # Reduce/Reduce conflicts raise GrammarError at construction time.
            pytest.fail(f"LALR(1) conflict (GrammarError) detected: {exc}")
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)

        log_output = stream.getvalue()
        # Any line containing "Shift/Reduce" or "Reduce/Reduce" is a conflict.
        conflict_lines = [
            line for line in log_output.splitlines()
            if "Shift/Reduce" in line or "Reduce/Reduce" in line
        ]
        assert conflict_lines == [], (
            "LALR(1) conflicts detected:\n" + "\n".join(conflict_lines)
        )


# ---------------------------------------------------------------------------
# Program structure
# ---------------------------------------------------------------------------


class TestProgramRoot:
    def test_single_expr(self) -> None:
        prog = parse("42")
        assert isinstance(prog, Program)
        assert isinstance(prog.body, Block)
        assert isinstance(first(prog), IntLit)

    def test_empty_block_raises(self) -> None:
        # An empty source is not valid (block needs at least one item).
        with pytest.raises(AglSyntaxError):
            parse("")

    def test_block_multiple_items(self) -> None:
        prog = parse("1\n2\n3")
        assert len(items(prog)) == 3

    def test_semicolon_separator(self) -> None:
        prog = parse("1; 2; 3")
        assert len(items(prog)) == 3

    def test_node_id_uniqueness(self) -> None:
        prog = parse("let x = 1\nlet y = 2\nx")
        ids = [node.node_id for node in [prog, prog.body]]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_int(self) -> None:
        n = first(parse("42"))
        assert isinstance(n, IntLit) and n.value == 42

    def test_decimal(self) -> None:
        d = first(parse("3.14"))
        assert isinstance(d, DecimalLit) and d.value == decimal.Decimal("3.14")

    def test_bool_true(self) -> None:
        b = first(parse("true"))
        assert isinstance(b, BoolLit) and b.value is True

    def test_bool_false(self) -> None:
        b = first(parse("false"))
        assert isinstance(b, BoolLit) and b.value is False

    def test_null(self) -> None:
        assert isinstance(first(parse("null")), NullLit)

    def test_unit_lit(self) -> None:
        u = first(parse("()"))
        assert isinstance(u, UnitLit)

    def test_string_plain(self) -> None:
        s = first(parse('"hello"'))
        assert isinstance(s, StringLit) and s.value == "hello"

    def test_string_interpolated(self) -> None:
        tmpl = first(parse('"hello ${name}"'))
        assert isinstance(tmpl, Template)
        assert len(tmpl.segments) == 2
        assert isinstance(tmpl.segments[0], TextSegment)
        assert isinstance(tmpl.segments[1], InterpSegment)

    def test_list_empty(self) -> None:
        lst = first(parse("[]"))
        assert isinstance(lst, ListLit) and lst.elements == ()

    def test_list_elements(self) -> None:
        lst = first(parse("[1, 2, 3]"))
        assert isinstance(lst, ListLit) and len(lst.elements) == 3

    def test_dict_empty(self) -> None:
        d = first(parse("{}"))
        assert isinstance(d, DictLit) and d.entries == ()

    def test_dict_entries(self) -> None:
        d = first(parse('{"a": 1, "b": 2}'))
        assert isinstance(d, DictLit) and len(d.entries) == 2
        assert all(isinstance(e, DictEntry) for e in d.entries)

    def test_dict_shorthand_key(self) -> None:
        d = first(parse("{name: 1}"))
        assert isinstance(d, DictLit)
        assert d.entries[0].key.value == "name"


# ---------------------------------------------------------------------------
# Binders
# ---------------------------------------------------------------------------


class TestBinders:
    def test_let_decl_simple(self) -> None:
        let = first(parse("let x = 5"))
        assert isinstance(let, LetDecl)
        assert let.name == "x"
        assert let.type_ann is None
        assert isinstance(let.value, IntLit)

    def test_let_decl_annotated(self) -> None:
        let = first(parse("let x: int = 5"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, IntT)

    def test_var_decl(self) -> None:
        v = first(parse("var count: int = 0"))
        assert isinstance(v, VarDecl)
        assert v.name == "count"
        assert isinstance(v.type_ann, IntT)

    def test_set_stmt(self) -> None:
        s = first(parse("set x = 10"))
        assert isinstance(s, SetStmt)
        assert s.target == "x"
        assert isinstance(s.value, IntLit)

    def test_let_continuation(self) -> None:
        """let-continuation: let x = 1; x parses as two block items."""
        prog = parse("let x = 1\nx")
        assert len(items(prog)) == 2
        assert isinstance(items(prog)[0], LetDecl)
        assert isinstance(items(prog)[1], VarRef)


# ---------------------------------------------------------------------------
# Type expressions
# ---------------------------------------------------------------------------


class TestTypeExpressions:
    def test_text(self) -> None:
        let = first(parse("let x: text = x"))
        assert isinstance(let, LetDecl) and isinstance(let.type_ann, TextT)

    def test_int(self) -> None:
        let = first(parse("let x: int = 1"))
        assert isinstance(let, LetDecl) and isinstance(let.type_ann, IntT)

    def test_decimal(self) -> None:
        let = first(parse("let x: decimal = 1.0"))
        assert isinstance(let, LetDecl) and isinstance(let.type_ann, DecimalT)

    def test_bool(self) -> None:
        let = first(parse("let x: bool = true"))
        assert isinstance(let, LetDecl) and isinstance(let.type_ann, BoolT)

    def test_json(self) -> None:
        let = first(parse("let x: json = null"))
        assert isinstance(let, LetDecl) and isinstance(let.type_ann, JsonT)

    def test_list_of_int(self) -> None:
        let = first(parse("let xs: list[int] = []"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, ListT)
        assert isinstance(let.type_ann.elem, IntT)

    def test_dict_type(self) -> None:
        let = first(parse("let d: dict[text, int] = {}"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, DictT)
        assert isinstance(let.type_ann.value, IntT)

    def test_named_type(self) -> None:
        let = first(parse("let r: Review = x"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, NameT)
        assert let.type_ann.name == "Review"

    def test_unit_type(self) -> None:
        let = first(parse("let u: unit = ()"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, UnitT)

    def test_agent_type(self) -> None:
        let = first(parse("let a: agent = rev"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, AgentT)

    def test_func_type_one_param(self) -> None:
        let = first(parse("let f: (int) -> text = classify"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, FuncT)
        assert len(let.type_ann.params) == 1
        assert isinstance(let.type_ann.params[0], IntT)
        assert isinstance(let.type_ann.result, TextT)

    def test_func_type_two_params(self) -> None:
        let = first(parse("let f: (int, text) -> bool = g"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, FuncT)
        assert len(let.type_ann.params) == 2

    def test_func_type_unit_domain(self) -> None:
        """() -> text is a zero-param function type."""
        let = first(parse("let f: () -> text = g"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, FuncT)
        assert let.type_ann.params == ()
        assert isinstance(let.type_ann.result, TextT)

    def test_func_type_returns_func(self) -> None:
        """Higher-order: (int) -> (int) -> int."""
        let = first(parse("let f: (int) -> (int) -> int = g"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, FuncT)
        assert isinstance(let.type_ann.result, FuncT)


# ---------------------------------------------------------------------------
# Declarations: record / enum / type alias / input / agent / config
# ---------------------------------------------------------------------------


class TestDeclarations:
    def test_record_def(self) -> None:
        prog = parse("record Issue\n  title: text\n  severity: int")
        rec = first(prog)
        assert isinstance(rec, RecordDef)
        assert rec.name == "Issue"
        assert len(rec.fields) == 2
        assert isinstance(rec.fields[0], FieldDef)
        assert rec.fields[0].name == "title"
        assert isinstance(rec.fields[0].type_expr, TextT)

    def test_enum_def(self) -> None:
        prog = parse("enum Status\n  | Pass\n  | Fail")
        en = first(prog)
        assert isinstance(en, EnumDef)
        assert en.name == "Status"
        assert len(en.variants) == 2
        assert en.variants[0].name == "Pass"
        assert en.variants[1].name == "Fail"

    def test_enum_with_payload(self) -> None:
        prog = parse("enum Result\n  | Ok(value: int)\n  | Err(msg: text)")
        en = first(prog)
        assert isinstance(en, EnumDef)
        ok = en.variants[0]
        assert ok.name == "Ok"
        assert len(ok.fields) == 1
        assert isinstance(ok.fields[0], FieldDef)

    def test_type_alias(self) -> None:
        ta = first(parse("type Name = text"))
        assert isinstance(ta, TypeAlias)
        assert ta.name == "Name"
        assert isinstance(ta.type_expr, TextT)

    def test_input_decl_no_annotation(self) -> None:
        inp = first(parse("input spec"))
        assert isinstance(inp, InputDecl)
        assert inp.name == "spec"
        assert inp.annotation is None

    def test_input_decl_annotated(self) -> None:
        inp = first(parse("input count: int"))
        assert isinstance(inp, InputDecl)
        assert isinstance(inp.annotation, IntT)

    def test_agent_decl_bare(self) -> None:
        ag = first(parse("agent reviewer"))
        assert isinstance(ag, AgentDecl)
        assert ag.name == "reviewer"
        assert ag.runner is None

    def test_agent_decl_with_runner(self) -> None:
        ag = first(parse('agent planner = "claude -p %{PROMPT_FILE}"'))
        assert isinstance(ag, AgentDecl)
        assert ag.runner == "claude -p %{PROMPT_FILE}"

    def test_config_pragma_bool(self) -> None:
        cfg = first(parse("config log = true"))
        assert isinstance(cfg, ConfigPragma)
        assert cfg.key == "log"
        assert cfg.value is True

    def test_config_pragma_int(self) -> None:
        cfg = first(parse("config max_iters = 10"))
        assert isinstance(cfg, ConfigPragma)
        assert cfg.value == 10

    def test_config_pragma_string(self) -> None:
        cfg = first(parse('config runner = "claude"'))
        assert isinstance(cfg, ConfigPragma)
        assert cfg.value == "claude"


# ---------------------------------------------------------------------------
# Function declarations (def)
# ---------------------------------------------------------------------------


class TestFuncDef:
    def test_def_no_params(self) -> None:
        fd = first(parse("def greet() -> text = x"))
        assert isinstance(fd, FuncDef)
        assert fd.name == "greet"
        assert fd.params == ()
        assert isinstance(fd.return_type, TextT)

    def test_def_required_params(self) -> None:
        fd = first(parse("def add(x: int, y: int) -> int = z"))
        assert isinstance(fd, FuncDef)
        assert len(fd.params) == 2
        p0 = fd.params[0]
        assert isinstance(p0, Param)
        assert p0.name == "x"
        assert isinstance(p0.type_expr, IntT)
        assert p0.default is None

    def test_def_with_default(self) -> None:
        fd = first(parse("def summarize(doc: text, limit: int = 3) -> text = x"))
        assert isinstance(fd, FuncDef)
        assert len(fd.params) == 2
        p1 = fd.params[1]
        assert isinstance(p1, Param)
        assert p1.name == "limit"
        assert isinstance(p1.default, IntLit)
        assert p1.default.value == 3

    def test_def_expression_body(self) -> None:
        fd = first(parse("def fact(n: int) -> int = n"))
        assert isinstance(fd, FuncDef)
        assert isinstance(fd.body, VarRef)

    def test_def_suite_body(self) -> None:
        src = "def summarize(doc: text) -> text =\n  let head = ask\n  head"
        fd = first(parse(src))
        assert isinstance(fd, FuncDef)
        assert isinstance(fd.body, Block)
        assert len(fd.body.items) == 2

    def test_def_if_body(self) -> None:
        src = "def classify(n: int) -> text = if n > 0 => pos | else => neg"
        fd = first(parse(src))
        assert isinstance(fd, FuncDef)
        assert isinstance(fd.body, If)

    def test_def_trailing_comma_params(self) -> None:
        fd = first(parse("def f(x: int,) -> int = x"))
        assert isinstance(fd, FuncDef)
        assert len(fd.params) == 1


# ---------------------------------------------------------------------------
# Lambda expressions
# ---------------------------------------------------------------------------


class TestLambda:
    def test_lambda_with_return_type(self) -> None:
        src = "let dbl = fn(x: int) -> int => x"
        let = first(parse(src))
        assert isinstance(let, LetDecl)
        lam = let.value
        assert isinstance(lam, Lambda)
        assert len(lam.params) == 1
        assert isinstance(lam.return_type, IntT)
        assert isinstance(lam.body, VarRef)

    def test_lambda_without_return_type(self) -> None:
        src = "let dbl = fn(x: int) => x"
        let = first(parse(src))
        assert isinstance(let, LetDecl)
        lam = let.value
        assert isinstance(lam, Lambda)
        assert lam.return_type is None

    def test_lambda_no_params(self) -> None:
        src = "let f = fn() => 1"
        let = first(parse(src))
        assert isinstance(let, LetDecl)
        lam = let.value
        assert isinstance(lam, Lambda)
        assert lam.params == ()
        assert isinstance(lam.body, IntLit)

    def test_lambda_multi_params(self) -> None:
        src = "let add = fn(x: int, y: int) => x"
        let = first(parse(src))
        assert isinstance(let, LetDecl)
        lam = let.value
        assert isinstance(lam, Lambda)
        assert len(lam.params) == 2

    def test_lambda_as_call_arg(self) -> None:
        """Lambda as argument must be parenthesized."""
        src = "let r = map(fn(x: int) -> int => x, xs)"
        let = first(parse(src))
        assert isinstance(let, LetDecl)
        call = let.value
        assert isinstance(call, Call)
        # First arg is a Lambda (parenthesized by the outer parens of the call)
        assert isinstance(call.args[0], Lambda)


# ---------------------------------------------------------------------------
# Uniform calls
# ---------------------------------------------------------------------------


class TestCalls:
    def test_paren_call_no_args(self) -> None:
        """f() produces a Call with empty args."""
        call = first(parse("f()"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "f"
        assert call.args == ()
        assert call.named_args == ()

    def test_paren_call_one_positional(self) -> None:
        call = first(parse("f(1)"))
        assert isinstance(call, Call)
        assert len(call.args) == 1
        assert isinstance(call.args[0], IntLit)

    def test_paren_call_multiple_positional(self) -> None:
        call = first(parse("f(1, 2, 3)"))
        assert isinstance(call, Call)
        assert len(call.args) == 3

    def test_paren_call_named_arg(self) -> None:
        call = first(parse("ask(x, agent: reviewer)"))
        assert isinstance(call, Call)
        assert len(call.args) == 1
        assert len(call.named_args) == 1
        na = call.named_args[0]
        assert isinstance(na, NamedArg)
        assert na.name == "agent"

    def test_paren_call_multiple_named(self) -> None:
        call = first(parse("ask(x, agent: rev, format: json)"))
        assert isinstance(call, Call)
        assert len(call.named_args) == 2

    def test_paren_call_trailing_comma(self) -> None:
        call = first(parse("f(1, 2,)"))
        assert isinstance(call, Call)
        assert len(call.args) == 2

    def test_unit_call(self) -> None:
        """f() with no args produces a Call with empty args (not a UnitLit)."""
        call = first(parse("f()"))
        assert isinstance(call, Call)
        assert call.args == ()

    def test_juxt_call_varref(self) -> None:
        """print x desugars to Call(print, (x,), ())."""
        call = first(parse("print x"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "print"
        assert len(call.args) == 1
        assert isinstance(call.args[0], VarRef)
        assert call.args[0].name == "x"
        assert call.named_args == ()

    def test_juxt_call_string(self) -> None:
        """ask "hi" desugars to Call(ask, ("hi",), ())."""
        call = first(parse('ask "hi"'))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "ask"
        assert isinstance(call.args[0], StringLit)
        assert call.args[0].value == "hi"

    def test_juxt_call_int(self) -> None:
        """f 5 desugars to Call(f, (5,), ())."""
        call = first(parse("f 5"))
        assert isinstance(call, Call)
        assert len(call.args) == 1
        assert isinstance(call.args[0], IntLit)

    def test_juxt_call_field_access(self) -> None:
        """print res.stdout desugars to Call(print, (FieldAccess(res, stdout),), ())."""
        call = first(parse("print res.stdout"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "print"
        assert len(call.args) == 1
        fa = call.args[0]
        assert isinstance(fa, FieldAccess)
        assert isinstance(fa.obj, VarRef)
        assert fa.obj.name == "res"
        assert fa.field == "stdout"

    def test_juxt_call_deep_field_access(self) -> None:
        """print a.b.c desugars to Call(print, (FieldAccess(FieldAccess(a, b), c),), ())."""
        call = first(parse("print a.b.c"))
        assert isinstance(call, Call)
        arg = call.args[0]
        assert isinstance(arg, FieldAccess)
        assert arg.field == "c"
        assert isinstance(arg.obj, FieldAccess)
        assert arg.obj.field == "b"

    def test_paren_call_with_call_result(self) -> None:
        """print(classify(x)) — nested call in paren form."""
        call = first(parse("print(classify(x))"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "print"
        inner = call.args[0]
        assert isinstance(inner, Call)
        assert isinstance(inner.callee, VarRef)
        assert inner.callee.name == "classify"

    def test_chained_paren_calls(self) -> None:
        """f(x).g(y) chains fine via postfix."""
        call = first(parse("f(x).g(y)"))
        assert isinstance(call, Call)
        # callee is a FieldAccess on f(x)
        callee = call.callee
        assert isinstance(callee, FieldAccess)

    def test_duplicate_named_arg_raises(self) -> None:
        with pytest.raises(AglSyntaxError, match="duplicate"):
            parse("f(a: 1, a: 2)")


# ---------------------------------------------------------------------------
# Field access and qualified constructors
# ---------------------------------------------------------------------------


class TestFieldAccessAndConstructors:
    def test_field_access(self) -> None:
        fa = first(parse("r.field"))
        assert isinstance(fa, FieldAccess)
        assert fa.field == "field"

    def test_chained_field_access(self) -> None:
        fa = first(parse("a.b.c"))
        assert isinstance(fa, FieldAccess)
        assert fa.field == "c"
        assert isinstance(fa.obj, FieldAccess)
        assert fa.obj.field == "b"

    def test_constructor_bare(self) -> None:
        c = first(parse("Pass"))
        assert isinstance(c, Constructor)
        assert c.qualifier is None
        assert c.name == "Pass"
        assert c.args == ()

    def test_constructor_with_args(self) -> None:
        c = first(parse("Issue(title: x, severity: 1)"))
        assert isinstance(c, Constructor)
        assert c.name == "Issue"
        assert len(c.args) == 2

    def test_qualified_constructor(self) -> None:
        c = first(parse("Review.Pass"))
        assert isinstance(c, Constructor)
        assert c.qualifier == "Review"
        assert c.name == "Pass"

    def test_double_payload_raises(self) -> None:
        with pytest.raises(AglSyntaxError, match="single argument list"):
            parse("Issue(a: 1)(b: 2)")

    def test_bad_qualification_raises(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse("x.Done")


# ---------------------------------------------------------------------------
# Binary operators
# ---------------------------------------------------------------------------


class TestBinaryOperators:
    def test_arithmetic(self) -> None:
        e = first(parse("1 + 2 * 3"))
        # Should parse as 1 + (2 * 3) due to precedence.
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.ADD

    def test_comparison(self) -> None:
        e = first(parse("x > 0"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.GT

    def test_equality(self) -> None:
        e = first(parse("x = y"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.EQ

    def test_logical_and_or(self) -> None:
        e = first(parse("a or b and c"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.OR

    def test_not(self) -> None:
        e = first(parse("not x"))
        assert isinstance(e, UnaryNot)

    def test_unary_neg(self) -> None:
        e = first(parse("-1"))
        assert isinstance(e, UnaryNeg)

    def test_is_test(self) -> None:
        e = first(parse("x is Pass"))
        assert isinstance(e, IsTest)
        assert e.variant == "Pass"
        assert not e.negated

    def test_is_not_test(self) -> None:
        e = first(parse("x is not Pass"))
        assert isinstance(e, IsTest)
        assert e.negated

    def test_is_qualified(self) -> None:
        e = first(parse("x is Review.Pass"))
        assert isinstance(e, IsTest)
        assert e.qualifier == "Review"
        assert e.variant == "Pass"

    def test_in(self) -> None:
        e = first(parse("x in [1, 2, 3]"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.IN

    def test_eq_eq_raises(self) -> None:
        """== is not valid AgL; lexer emits EQ_EQ which triggers a parse error."""
        with pytest.raises(AglSyntaxError):
            parse("x == y")


# ---------------------------------------------------------------------------
# Control flow: if_expr
# ---------------------------------------------------------------------------


class TestIfExpr:
    def test_if_single_branch(self) -> None:
        e = first(parse("if x => 1"))
        assert isinstance(e, If)
        assert len(e.branches) == 1
        b = e.branches[0]
        assert isinstance(b, IfBranch)
        assert isinstance(b.cond, VarRef)

    def test_if_with_else(self) -> None:
        e = first(parse("if x > 0 => pos | else => neg"))
        assert isinstance(e, If)
        assert len(e.branches) == 2
        assert e.branches[-1].cond is ELSE

    def test_if_with_else_without_pipe(self) -> None:
        e = first(parse("if x > 0 => pos else => neg"))
        assert isinstance(e, If)
        assert len(e.branches) == 2
        assert e.branches[-1].cond is ELSE

    def test_if_multiple_branches(self) -> None:
        e = first(parse("if n > 0 => pos | n < 0 => neg | else => zero"))
        assert isinstance(e, If)
        assert len(e.branches) == 3

    def test_if_multiple_branches_with_else_without_pipe(self) -> None:
        e = first(parse("if n > 0 => pos | n < 0 => neg else => zero"))
        assert isinstance(e, If)
        assert len(e.branches) == 3
        assert e.branches[-1].cond is ELSE

    def test_if_with_leading_pipe(self) -> None:
        e = first(parse("if | n > 0 => pos | else => neg"))
        assert isinstance(e, If)
        assert len(e.branches) == 2

    def test_if_with_leading_pipe_and_else_without_pipe(self) -> None:
        e = first(parse("if | n > 0 => pos else => neg"))
        assert isinstance(e, If)
        assert len(e.branches) == 2
        assert e.branches[-1].cond is ELSE

    def test_if_cannot_have_branch_after_else(self) -> None:
        with pytest.raises(AglSyntaxError, match=r"\|"):
            parse("if n > 0 => y else => x | n < 0 => z")

    def test_if_else_missing_arrow_message(self) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse("if true => false else true")

        assert str(exc_info.value) == "Missing `=>` after `else`."

    def test_if_suite_branch_body(self) -> None:
        src = "if x =>\n  let y = 1\n  y\n| else => z"
        e = first(parse(src))
        assert isinstance(e, If)
        body = e.branches[0].body
        assert isinstance(body, Block)
        assert len(body.items) == 2

    def test_if_suite_else_without_pipe(self) -> None:
        src = "if x =>\n  let y = 1\n  y\nelse => z"
        e = first(parse(src))
        assert isinstance(e, If)
        assert len(e.branches) == 2
        assert e.branches[-1].cond is ELSE

    def test_if_multiline_leading_pipe_else_without_pipe(self) -> None:
        src = "if\n  | x => y\n  else => z"
        e = first(parse(src))
        assert isinstance(e, If)
        assert len(e.branches) == 2
        assert e.branches[-1].cond is ELSE


# ---------------------------------------------------------------------------
# Control flow: case_expr
# ---------------------------------------------------------------------------


class TestCaseExpr:
    def test_case_simple(self) -> None:
        src = "case x of | Pass => ok | Fail => err"
        e = first(parse(src))
        assert isinstance(e, Case)
        assert isinstance(e.subject, VarRef)
        assert len(e.branches) == 2

    def test_case_with_var_pattern(self) -> None:
        src = "case x of | n => n"
        e = first(parse(src))
        assert isinstance(e, Case)
        pat = e.branches[0].pattern
        assert isinstance(pat, VarPattern)

    def test_case_with_wildcard(self) -> None:
        src = "case x of | _ => default"
        e = first(parse(src))
        assert isinstance(e, Case)
        assert isinstance(e.branches[0].pattern, WildcardPattern)

    def test_case_branch_body_is_expr(self) -> None:
        src = "case x of | Pass => ok | Fail => err"
        e = first(parse(src))
        assert isinstance(e, Case)
        branch = e.branches[0]
        assert isinstance(branch, CaseBranch)
        assert isinstance(branch.body, VarRef)

    def test_case_suite_branch_body(self) -> None:
        src = "case x of | Pass =>\n  let r = 1\n  r\n| Fail => err"
        e = first(parse(src))
        assert isinstance(e, Case)
        body = e.branches[0].body
        assert isinstance(body, Block)


# ---------------------------------------------------------------------------
# Control flow: do_expr
# ---------------------------------------------------------------------------


class TestDoExpr:
    def test_do_simple(self) -> None:
        e = first(parse("do set x = 1 until x > 5"))
        assert isinstance(e, Do)
        assert e.limit is None
        assert isinstance(e.condition, BinaryOp)

    def test_do_with_bound(self) -> None:
        e = first(parse("do[10] set x = 1 until x > 5"))
        assert isinstance(e, Do)
        assert e.limit == 10

    def test_do_zero_bound_raises(self) -> None:
        with pytest.raises(AglSyntaxError, match="positive"):
            parse("do[0] x until true")

    def test_do_suite_body(self) -> None:
        src = "do\n  set x = 1\n  set y = 2\nuntil x > 5"
        e = first(parse(src))
        assert isinstance(e, Do)
        assert isinstance(e.body, Block)


# ---------------------------------------------------------------------------
# Control flow: try_expr
# ---------------------------------------------------------------------------


class TestTryExpr:
    def test_try_catch_type(self) -> None:
        src = "try x catch AgentCallError => err"
        e = first(parse(src))
        assert isinstance(e, Try)
        assert len(e.handlers) == 1
        h = e.handlers[0]
        assert isinstance(h, CatchClause)
        assert h.exc_type == "AgentCallError"
        assert h.binding is None

    def test_try_catch_with_binding(self) -> None:
        src = "try x catch AgentCallError as err => err"
        e = first(parse(src))
        assert isinstance(e, Try)
        h = e.handlers[0]
        assert h.binding == "err"

    def test_try_catch_wildcard(self) -> None:
        src = "try x catch _ => default"
        e = first(parse(src))
        assert isinstance(e, Try)
        h = e.handlers[0]
        assert h.exc_type is None

    def test_try_multiple_handlers(self) -> None:
        src = "try x catch AgentCallError => e1 catch _ => e2"
        e = first(parse(src))
        assert isinstance(e, Try)
        assert len(e.handlers) == 2

    def test_try_suite_body(self) -> None:
        src = "try\n  let r = x\n  r\ncatch _ => err"
        e = first(parse(src))
        assert isinstance(e, Try)
        assert isinstance(e.body, Block)

    def test_try_suite_catch_body(self) -> None:
        src = "try x catch _ =>\n  let e = err\n  e"
        e = first(parse(src))
        assert isinstance(e, Try)
        h = e.handlers[0]
        assert isinstance(h.body, Block)


# ---------------------------------------------------------------------------
# raise_expr
# ---------------------------------------------------------------------------


class TestRaiseExpr:
    def test_raise_simple(self) -> None:
        e = first(parse("raise x"))
        assert isinstance(e, Raise)
        assert isinstance(e.exc, VarRef)

    def test_raise_constructor(self) -> None:
        e = first(parse("raise Error(msg: m)"))
        assert isinstance(e, Raise)
        assert isinstance(e.exc, Constructor)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


class TestPatterns:
    def test_literal_int_pattern(self) -> None:
        e = first(parse("case x of | 1 => a | 2 => b"))
        assert isinstance(e, Case)
        assert isinstance(e.branches[0].pattern, LiteralPattern)

    def test_constructor_pattern_with_fields(self) -> None:
        src = "case r of | Issue(title: t, severity: s) => ok"
        e = first(parse(src))
        assert isinstance(e, Case)
        pat = e.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.name == "Issue"
        assert len(pat.fields) == 2

    def test_constructor_pattern_shorthand(self) -> None:
        src = "case r of | Issue(title) => ok"
        e = first(parse(src))
        assert isinstance(e, Case)
        pat = e.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert isinstance(pat.fields[0], PatternField)
        assert pat.fields[0].name == "title"

    def test_qualified_constructor_pattern(self) -> None:
        src = "case r of | Review.Pass => ok | Review.Fail => err"
        e = first(parse(src))
        assert isinstance(e, Case)
        assert e.branches[0].pattern.qualifier == "Review"


# ---------------------------------------------------------------------------
# Template / string interpolation
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_plain_template_collapses_to_string_lit(self) -> None:
        """A template with no interpolations becomes StringLit."""
        s = first(parse('"hello world"'))
        assert isinstance(s, StringLit)
        assert s.value == "hello world"

    def test_interpolated_template(self) -> None:
        t = first(parse('"Hello ${name}"'))
        assert isinstance(t, Template)
        assert any(isinstance(seg, InterpSegment) for seg in t.segments)

    def test_multi_interp_template(self) -> None:
        t = first(parse('"${a} and ${b}"'))
        assert isinstance(t, Template)
        interps = [s for s in t.segments if isinstance(s, InterpSegment)]
        assert len(interps) == 2

    def test_interpolated_agent_runner_raises(self) -> None:
        with pytest.raises(AglSyntaxError, match="literal string"):
            parse('agent reviewer = "runner ${x}"')

    def test_pattern_interpolated_string_raises(self) -> None:
        with pytest.raises(AglSyntaxError, match="interpolation"):
            parse('case x of | "${y}" => ok')


# ---------------------------------------------------------------------------
# Multi-line if/case/try (|-continuation)
# ---------------------------------------------------------------------------


class TestMultiLineBranches:
    def test_multiline_if_suite_bodies(self) -> None:
        """if with suite bodies (indented blocks after =>)."""
        src = (
            "if n > 0 =>\n"
            "  pos\n"
            "| n < 0 =>\n"
            "  neg\n"
            "| else =>\n"
            "  zero"
        )
        e = first(parse(src))
        assert isinstance(e, If)
        assert len(e.branches) == 3

    def test_multiline_if_pipe_continuation(self) -> None:
        """if with | on new lines — the |-continuation layout rule suppresses _NEWLINE."""
        src = "if n > 0 => pos\n| n < 0 => neg\n| else => zero"
        e = first(parse(src))
        assert isinstance(e, If)
        assert len(e.branches) == 3

    def test_multiline_case(self) -> None:
        """case with | on new lines via |-continuation."""
        src = "case x of\n| Pass => ok\n| Fail => err"
        e = first(parse(src))
        assert isinstance(e, Case)
        assert len(e.branches) == 2

    def test_multiline_case_suite_bodies(self) -> None:
        """case with suite bodies."""
        src = (
            "case x of\n"
            "| Pass =>\n"
            "  ok\n"
            "| Fail =>\n"
            "  err"
        )
        e = first(parse(src))
        assert isinstance(e, Case)
        assert len(e.branches) == 2
        assert isinstance(e.branches[0].body, Block)

    def test_multiline_try(self) -> None:
        """try with catch on new lines via catch-continuation."""
        src = (
            "try x\n"
            "catch AgentCallError => e1\n"
            "catch _ => e2"
        )
        e = first(parse(src))
        assert isinstance(e, Try)
        assert len(e.handlers) == 2


# ---------------------------------------------------------------------------
# REPL seam
# ---------------------------------------------------------------------------


class TestReplSeam:
    def test_parse_program_seeded(self) -> None:
        prog, next_id = parse_program_seeded("let x = 1\nx", start_id=100)
        assert isinstance(prog, Program)
        assert next_id > 100

    def test_node_ids_globally_unique(self) -> None:
        prog1, next_id1 = parse_program_seeded("let x = 1", start_id=0)
        prog2, next_id2 = parse_program_seeded("let y = 2", start_id=next_id1)
        ids1 = {prog1.node_id, prog1.body.node_id}
        ids2 = {prog2.node_id, prog2.body.node_id}
        assert not ids1.intersection(ids2)

    def test_is_incomplete_source_complete(self) -> None:
        assert not is_incomplete_source("let x = 1")

    def test_is_incomplete_source_dangling(self) -> None:
        # "let x =" — dangling, needs more input.
        assert is_incomplete_source("let x =")

    def test_is_incomplete_source_open_block(self) -> None:
        # record header without body
        assert is_incomplete_source("record R")

    def test_is_incomplete_source_real_error(self) -> None:
        # == is a real error, not an incomplete source.
        assert not is_incomplete_source("x == y")


# ---------------------------------------------------------------------------
# Negative cases (parse errors)
# ---------------------------------------------------------------------------


class TestNegativeCases:
    def test_juxt_does_not_chain(self) -> None:
        """f a b is a parse error — juxt does not chain."""
        with pytest.raises(AglSyntaxError):
            parse("f a b")

    def test_print_call_result_needs_parens(self) -> None:
        """print classify(x) is a parse error; use print(classify(x))."""
        with pytest.raises(AglSyntaxError):
            parse("print classify(x)")

    def test_bare_assignment_is_equality(self) -> None:
        """In v2, n = 2 is a BinaryOp(EQ) expression (not a mutation).
        Mutation uses `set n = 2`. The parser accepts n = 2 as an expression.
        The scope pass would verify mutation intent.
        """
        e = first(parse("n = 2"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.EQ

    def test_unknown_catch_var_raises(self) -> None:
        """catch lowercase (not _ or TYPE_NAME) is a parse error."""
        with pytest.raises(AglSyntaxError):
            parse("try x catch err => x")

    def test_duplicate_constructor_arg_raises(self) -> None:
        with pytest.raises(AglSyntaxError, match="duplicate"):
            parse("Issue(title: a, title: b)")

    def test_def_without_return_type_raises(self) -> None:
        """def without -> return type is a parse error."""
        with pytest.raises(AglSyntaxError):
            parse("def f(x: int) = x")

    def test_lambda_as_juxt_arg_raises(self) -> None:
        """A lambda cannot be a bare juxt argument (it starts with fn, a keyword)."""
        with pytest.raises(AglSyntaxError):
            parse("print fn(x: int) => x")


# ---------------------------------------------------------------------------
# Full program examples (integration)
# ---------------------------------------------------------------------------


class TestFullPrograms:
    def test_classify_function(self) -> None:
        src = (
            "def classify(n: int) -> text =\n"
            '  if n > 0 => "pos"\n'
            '  | n < 0  => "neg"\n'
            '  | else   => "zero"'
        )
        prog = parse(src)
        fd = first(prog)
        assert isinstance(fd, FuncDef)
        assert fd.name == "classify"
        # A suite body always produces a Block; the single item is the If.
        body = fd.body
        assert isinstance(body, Block)
        assert isinstance(body.items[0], If)

    def test_summarize_with_let_continuation(self) -> None:
        src = (
            "def summarize(doc: text, limit: int = 3) -> text =\n"
            '  let head = ask "summary"\n'
            "  let tagged = head\n"
            "  tagged"
        )
        prog = parse(src)
        fd = first(prog)
        assert isinstance(fd, FuncDef)
        body = fd.body
        assert isinstance(body, Block)
        assert len(body.items) == 3

    def test_agent_and_ask_program(self) -> None:
        src = (
            "agent reviewer\n"
            "agent planner\n"
            'let s = ask "Hello?"\n'
            'let r = ask("Review", agent: reviewer)\n'
            'print r'
        )
        prog = parse(src)
        assert len(items(prog)) == 5
        assert isinstance(items(prog)[0], AgentDecl)
        assert isinstance(items(prog)[2], LetDecl)
        # 4th item: ask with named arg
        let_r = items(prog)[3]
        assert isinstance(let_r, LetDecl)
        call = let_r.value
        assert isinstance(call, Call)
        assert len(call.named_args) == 1
        assert call.named_args[0].name == "agent"

    def test_factorial_recursion(self) -> None:
        src = (
            "def fact(n: int) -> int =\n"
            "  if n <= 1 => 1\n"
            "  | else => n"
        )
        prog = parse(src)
        fd = first(prog)
        assert isinstance(fd, FuncDef)

    def test_function_value_annotation(self) -> None:
        src = "let g: (int) -> text = classify"
        let = first(parse(src))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, FuncT)

    def test_exec_result_program(self) -> None:
        src = (
            'let res = exec "ls -la"\n'
            "print(res.stdout)\n"
            "if res.exit_code != 0 => print(x)"
        )
        prog = parse(src)
        assert len(items(prog)) == 3

    def test_agent_as_type_field(self) -> None:
        """agent as field name in a record."""
        prog = parse("record AgentRef\n  agent: agent")
        rec = first(prog)
        assert isinstance(rec, RecordDef)
        assert rec.fields[0].name == "agent"
        assert isinstance(rec.fields[0].type_expr, AgentT)


# ---------------------------------------------------------------------------
# Coverage-gap tests — Fix 3
# ---------------------------------------------------------------------------


class TestBinaryOperatorsCoverage:
    """Covers binary operators not yet tested: >=, -, /."""

    def test_bin_ge(self) -> None:
        """x >= 0 produces BinaryOp(GE)."""
        e = first(parse("x >= 0"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.GE

    def test_bin_sub(self) -> None:
        """x - 1 produces BinaryOp(SUB)."""
        e = first(parse("x - 1"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.SUB

    def test_bin_div(self) -> None:
        """x / 2 produces BinaryOp(DIV)."""
        e = first(parse("x / 2"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.DIV

    def test_is_not_qualified(self) -> None:
        """x is not Review.Pass produces a negated, qualified IsTest."""
        e = first(parse("x is not Review.Pass"))
        assert isinstance(e, IsTest)
        assert e.qualifier == "Review"
        assert e.variant == "Pass"
        assert e.negated


class TestLiteralPatternsCoverage:
    """Covers literal patterns other than int: decimal, true, false, null, string."""

    def test_literal_decimal_pattern(self) -> None:
        e = first(parse("case x of | 3.14 => a"))
        assert isinstance(e, Case)
        pat = e.branches[0].pattern
        assert isinstance(pat, LiteralPattern)
        assert isinstance(pat.literal, DecimalLit)

    def test_literal_true_pattern(self) -> None:
        e = first(parse("case x of | true => a"))
        assert isinstance(e, Case)
        assert isinstance(e.branches[0].pattern, LiteralPattern)
        assert isinstance(e.branches[0].pattern.literal, BoolLit)
        assert e.branches[0].pattern.literal.value is True

    def test_literal_false_pattern(self) -> None:
        e = first(parse("case x of | false => a"))
        assert isinstance(e, Case)
        pat = e.branches[0].pattern
        assert isinstance(pat, LiteralPattern)
        assert isinstance(pat.literal, BoolLit)
        assert pat.literal.value is False

    def test_literal_null_pattern(self) -> None:
        e = first(parse("case x of | null => a"))
        assert isinstance(e, Case)
        pat = e.branches[0].pattern
        assert isinstance(pat, LiteralPattern)
        assert isinstance(pat.literal, NullLit)

    def test_literal_string_pattern(self) -> None:
        e = first(parse('case x of | "hello" => a'))
        assert isinstance(e, Case)
        pat = e.branches[0].pattern
        assert isinstance(pat, LiteralPattern)
        assert isinstance(pat.literal, StringLit)
        assert pat.literal.value == "hello"


class TestDeclarationsCoverage:
    """Covers config pragma values not yet tested: false and decimal."""

    def test_config_pragma_false(self) -> None:
        cfg = first(parse("config log = false"))
        assert isinstance(cfg, ConfigPragma)
        assert cfg.key == "log"
        assert cfg.value is False

    def test_config_pragma_decimal(self) -> None:
        cfg = first(parse("config rate = 1.5"))
        assert isinstance(cfg, ConfigPragma)
        assert cfg.key == "rate"
        assert cfg.value == decimal.Decimal("1.5")


class TestTypeExprCoverage:
    """Covers type expression paths not yet tested."""

    def test_named_type_via_varname(self) -> None:
        """A lowercase VAR_NAME that isn't a keyword maps to NameT."""
        let = first(parse("let x: mytype = 1"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, NameT)
        assert let.type_ann.name == "mytype"

    def test_generic_type_unknown_raises(self) -> None:
        """foo[int] raises AglSyntaxError — only list[T] is valid."""
        with pytest.raises(AglSyntaxError, match="Unknown generic type"):
            parse("let x: foo[int] = 1")

    def test_dict_type_bad_key_raises(self) -> None:
        """dict[int, text] raises — dict keys must be text."""
        with pytest.raises(AglSyntaxError, match="text"):
            parse("let x: dict[int, text] = 1")

    def test_dict_type_bad_head_raises(self) -> None:
        """foo[text, int] raises — only dict[] is valid for two-param generic."""
        with pytest.raises(AglSyntaxError, match="Unknown generic type"):
            parse("let x: foo[text, int] = 1")


class TestCallsCoverage:
    """Covers call paths not yet tested."""

    def test_constructor_positional_arg_raises(self) -> None:
        """Constructor called with positional arg raises AglSyntaxError."""
        with pytest.raises(AglSyntaxError, match="named"):
            parse("Issue(1)")

    def test_paren_expr_unwrap(self) -> None:
        """(expr) parses as the inner expr (paren_expr rule unwraps)."""
        e = first(parse("(1 + 2)"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.ADD


class TestVariantPayloadCoverage:
    """Covers the empty variant_payload () path."""

    def test_empty_variant_payload(self) -> None:
        """Variant with empty payload () produces a VariantDef with no fields."""
        prog = parse("enum E\n  | Empty()")
        en = first(prog)
        assert isinstance(en, EnumDef)
        assert en.variants[0].name == "Empty"
        assert en.variants[0].fields == ()


class TestTryBodyCoverage:
    """Covers try_body with multiple semicolon-separated or_exprs."""

    def test_try_body_multi_stmt_wraps_block(self) -> None:
        """try x; y catch _ => err wraps the two exprs in a Block."""
        src = "try x; y catch _ => err"
        e = first(parse(src))
        assert isinstance(e, Try)
        assert isinstance(e.body, Block)
        assert len(e.body.items) == 2


class TestReplSeamCoverage:
    """Covers is_incomplete_source cache-hit and LexError paths."""

    def test_is_incomplete_source_cache_hit(self) -> None:
        """Calling is_incomplete_source twice with the same text uses the cache."""
        text = "let x ="
        result1 = is_incomplete_source(text)
        result2 = is_incomplete_source(text)
        assert result1 == result2
        assert result1 is True  # dangling '=' is incomplete

    def test_is_incomplete_source_lex_error(self) -> None:
        """An input that causes a LexError returns False (real error, not incomplete)."""
        assert not is_incomplete_source("@@@")


class TestParserErrorCoverage:
    """Covers parser error paths not yet tested."""

    def test_lex_error_in_parse_raises(self) -> None:
        """A character the lexer cannot tokenize raises AglSyntaxError."""
        with pytest.raises(AglSyntaxError):
            parse("@@@")


# ---------------------------------------------------------------------------
# errors.py coverage gap tests
# ---------------------------------------------------------------------------


class TestAglSyntaxErrorSourceSpan:
    """Covers AglSyntaxError.source_span (lines 71-72)."""

    def test_source_span_returns_span(self) -> None:
        """source_span returns the same SourceSpan object as .span."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("x == y")
        err = exc_info.value
        assert err.span is not None
        assert err.source_span is err.span

    def test_source_span_is_1based(self) -> None:
        """source_span on a parse error has a valid 1-based line and column."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("x == y")
        span = exc_info.value.source_span
        assert span.start_line >= 1
        assert span.start_col >= 1


class TestInlineCompoundElseBranch:
    """Covers _make_inline_compound_error else branch (line 148).

    Line 148 is the else of the ``if stmt_context / elif keyword=='case'``
    dispatch.  It fires when the unexpected inline-blocked token is NOT
    ``case`` AND the parser is in an *expression* context (stmt_context=False),
    i.e. ``if`` appearing as an operand inside an arithmetic or unary
    expression.  The message is identical to the stmt_context=True branch
    but reaches a different code path.
    """

    def test_inline_if_in_arithmetic_expression(self) -> None:
        """`1 + if x => y` triggers the else branch with stmt_context=False."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("1 + if x => y")
        msg = str(exc_info.value)
        assert "`if` is not allowed inline here" in msg
        assert "indented block" in msg

    def test_inline_if_after_unary_not(self) -> None:
        """`not if x => y` also gives stmt_context=False for the `if` token."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("not if x => y")
        msg = str(exc_info.value)
        assert "`if` is not allowed inline here" in msg


class TestSyntaxErrorFromLarkDirect:
    """Covers syntax_error_from_lark handlers not reachable via parse_program.

    The custom AglLexer pre-empts Lark's character-level lexer, so
    UnexpectedCharacters, UnexpectedEOF, and generic LarkError are never
    raised through the normal parse path.  We call the pure mapping helper
    directly with minimally-constructed Lark exception instances.
    """

    def test_unexpected_characters_message(self) -> None:
        """UnexpectedCharacters maps to 'Unexpected character.' with a 1-based span."""
        from lark.exceptions import UnexpectedCharacters

        from agm.agl.parser.errors import syntax_error_from_lark

        # seq='hello', lex_pos=2, line=3, column=5
        exc = UnexpectedCharacters("hello", 2, 3, 5)
        err = syntax_error_from_lark(exc)
        assert str(err) == "Unexpected character."
        assert err.span is not None
        assert err.span.start_line == 3
        assert err.span.start_col == 5

    def test_unexpected_characters_span_width_one(self) -> None:
        """The span produced for UnexpectedCharacters is exactly one character wide."""
        from lark.exceptions import UnexpectedCharacters

        from agm.agl.parser.errors import syntax_error_from_lark

        exc = UnexpectedCharacters("abc", 1, 1, 2)
        err = syntax_error_from_lark(exc)
        span = err.span
        assert span is not None
        assert span.end_col == span.start_col + 1
        assert span.end_offset == span.start_offset + 1

    def test_unexpected_eof_message(self) -> None:
        """UnexpectedEOF maps to 'Unexpected end of input.' with (1,1) fallback span."""
        from lark.exceptions import UnexpectedEOF

        from agm.agl.parser.errors import syntax_error_from_lark

        exc = UnexpectedEOF([])
        err = syntax_error_from_lark(exc)
        assert str(err) == "Unexpected end of input."
        span = err.span
        assert span is not None
        assert span.start_line == 1
        assert span.start_col == 1
        assert span.start_offset == 0

    def test_generic_lark_error_fallback(self) -> None:
        """A plain LarkError falls back to str(exc) as the message with (1,1) span."""
        from lark.exceptions import LarkError

        from agm.agl.parser.errors import syntax_error_from_lark

        message = "some unexpected grammar state"
        exc = LarkError(message)
        err = syntax_error_from_lark(exc)
        assert str(err) == message
        span = err.span
        assert span is not None
        assert span.start_line == 1
        assert span.start_col == 1
