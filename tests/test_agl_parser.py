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
- Binders: let/var/assignment.
- Input/agent/config declarations.
- Error cases: == produces friendly AglSyntaxError; bad syntax raises AglSyntaxError.
- REPL seam: parse_program_seeded and is_incomplete_source.
- Negative cases: f a b (juxt does not chain).

NOTE: This file must NOT modify tests/test_agl_e2e.py or tests/agl/.
      No static-analysis suppression comments in this file.
"""

from __future__ import annotations

import dataclasses
import decimal
import importlib.resources
import logging
import logging.handlers

import pytest

import agm.agl.syntax as syntax
from agm.agl.parser import (
    AglSyntaxError,
    is_incomplete_source,
    parse_program,
    parse_program_seeded,
    parse_type_expr,
)
from agm.agl.syntax import (
    AgentDecl,
    AssignStmt,
    BinaryOp,
    BinOp,
    Block,
    BoolLit,
    Call,
    Case,
    CaseBranch,
    Cast,
    CatchClause,
    ConfigDecl,
    ConstructorPattern,
    DecimalLit,
    DictEntry,
    DictLit,
    Do,
    EnumDef,
    ExceptionDef,
    FieldAccess,
    FieldDef,
    FuncDef,
    If,
    IfBranch,
    IndexAccess,
    IndexTarget,
    InterpSegment,
    IntLit,
    IsTest,
    Lambda,
    LetDecl,
    ListLit,
    LiteralPattern,
    NamedArg,
    NameTarget,
    NullLit,
    Param,
    ParamDecl,
    PatternField,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
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
    AppliedT,
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


def _collect_node_ids(obj: object, result: list[int]) -> None:
    """Recursively collect every node_id into *result*, preserving duplicates.

    Unlike ``all_node_ids`` (which deduplicates into a set), this appends every
    encountered node_id so that ``len(result) != len(set(result))`` reliably
    detects collisions.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        nid = getattr(obj, "node_id", None)
        if isinstance(nid, int):
            result.append(nid)
        for f in dataclasses.fields(obj):
            _collect_node_ids(getattr(obj, f.name), result)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _collect_node_ids(item, result)


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
        prog = parse("let x = 1\nlet y = 2\nx + y")
        id_list: list[int] = []
        _collect_node_ids(prog, id_list)
        assert len(id_list) == len(set(id_list)), "duplicate node_ids detected in parsed tree"


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

    def test_assign_stmt(self) -> None:
        s = first(parse("x := 10"))
        assert isinstance(s, AssignStmt)
        assert isinstance(s.target, NameTarget)
        assert s.target.name == "x"
        assert isinstance(s.value, IntLit)

    def test_assign_index_target(self) -> None:
        s = first(parse('xs[0] := "first"'))
        assert isinstance(s, AssignStmt)
        assert isinstance(s.target, IndexTarget)
        assert isinstance(s.target.obj, VarRef)
        assert s.target.obj.name == "xs"
        assert isinstance(s.target.index, IntLit)
        assert s.target.index.value == 0
        assert isinstance(s.value, StringLit)

    def test_legacy_set_syntax_is_not_assignment(self) -> None:
        assert not isinstance(first(parse("set x = 10")), AssignStmt)

    def test_assignment_target_must_have_variable_root(self) -> None:
        with pytest.raises(AglSyntaxError, match="assignment target"):
            parse("make()[0] := 10")

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

    def test_list_wrong_arg_count_raises(self) -> None:
        from agm.agl.parser.errors import AglSyntaxError
        with pytest.raises(AglSyntaxError, match="exactly one"):
            parse("let x: list[int, text] = []")

    def test_dict_wrong_arg_count_raises(self) -> None:
        from agm.agl.parser.errors import AglSyntaxError
        with pytest.raises(AglSyntaxError, match="exactly two"):
            parse("let d: dict[int] = {}")

    def test_dict_non_text_key_complex_type_raises(self) -> None:
        # dict key type must be text; a complex key triggers _type_expr_spelling fallback
        # (ListT → "list", covering the cls[:-1].lower() branch).
        from agm.agl.parser.errors import AglSyntaxError
        with pytest.raises(AglSyntaxError, match="text"):
            parse("let d: dict[list[int], int] = {}")

    def test_dict_named_type_key_raises(self) -> None:
        # dict key type must be text; a named type key triggers the NameT branch
        # in _type_expr_spelling, which returns its name in the error message.
        from agm.agl.parser.errors import AglSyntaxError
        with pytest.raises(AglSyntaxError, match="Review"):
            parse("let d: dict[Review, int] = {}")

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
# Declarations: record / enum / type alias / param / program / agent / config
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

    def test_record_def_with_parens(self) -> None:
        prog = parse("record Point(x: int, y: int)")
        rec = first(prog)
        assert isinstance(rec, RecordDef)
        assert rec.name == "Point"
        assert [field.name for field in rec.fields] == ["x", "y"]

    def test_record_def_empty_parens(self) -> None:
        prog = parse("record Empty()")
        rec = first(prog)
        assert isinstance(rec, RecordDef)
        assert rec.name == "Empty"
        assert rec.fields == ()

    def test_record_def_with_braces_raises(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse("record Point {x: int, y: int}")

    def test_record_def_inline_without_braces(self) -> None:
        prog = parse("record Point x: int, y: int")
        rec = first(prog)
        assert isinstance(rec, RecordDef)
        assert rec.name == "Point"
        assert [field.name for field in rec.fields] == ["x", "y"]

    def test_enum_def(self) -> None:
        prog = parse("enum Status\n  | Pass\n  | Fail")
        en = first(prog)
        assert isinstance(en, EnumDef)
        assert en.name == "Status"
        assert len(en.variants) == 2
        assert en.variants[0].name == "Pass"
        assert en.variants[1].name == "Fail"

    def test_enum_def_first_pipe_optional(self) -> None:
        prog = parse("enum Status Pass | Fail")
        en = first(prog)
        assert isinstance(en, EnumDef)
        assert [variant.name for variant in en.variants] == ["Pass", "Fail"]

    def test_enum_def_equals_after_name_optional(self) -> None:
        prog = parse("enum Status = Pass | Fail")
        en = first(prog)
        assert isinstance(en, EnumDef)
        assert [variant.name for variant in en.variants] == ["Pass", "Fail"]

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

    def test_param_decl_no_annotation(self) -> None:
        inp = first(parse("param spec"))
        assert isinstance(inp, ParamDecl)
        assert inp.name == "spec"
        assert inp.annotation is None

    def test_param_decl_annotated(self) -> None:
        inp = first(parse("param count: int"))
        assert isinstance(inp, ParamDecl)
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

    def test_config_decl_bool(self) -> None:
        cfg = first(parse("config log = true"))
        assert isinstance(cfg, ConfigDecl)
        assert cfg.name == "log"
        assert isinstance(cfg.value, BoolLit)
        assert cfg.value.value is True

    def test_config_decl_int(self) -> None:
        cfg = first(parse("config max-iters = 10"))
        assert isinstance(cfg, ConfigDecl)
        assert isinstance(cfg.value, IntLit)
        assert cfg.value.value == 10

    def test_config_decl_string(self) -> None:
        cfg = first(parse('config runner = "claude"'))
        assert isinstance(cfg, ConfigDecl)
        assert isinstance(cfg.value, StringLit)
        assert cfg.value.value == "claude"

    def test_exception_def_simple(self) -> None:
        exc = first(parse("exception MyErr(msg: text)"))
        assert isinstance(exc, ExceptionDef)
        assert exc.name == "MyErr"
        assert len(exc.fields) == 1
        assert exc.fields[0].name == "msg"
        assert exc.base is None
        assert exc.is_private is False
        assert exc.is_builtin is False

    def test_exception_def_with_base(self) -> None:
        exc = first(parse("exception DerivedErr extends BaseErr(msg: text)"))
        assert isinstance(exc, ExceptionDef)
        assert exc.name == "DerivedErr"
        assert exc.base == "BaseErr"
        assert len(exc.fields) == 1
        assert exc.fields[0].name == "msg"

    def test_exception_def_no_fields(self) -> None:
        exc = first(parse("exception SimpleErr()"))
        assert isinstance(exc, ExceptionDef)
        assert exc.name == "SimpleErr"
        assert exc.fields == ()
        assert exc.base is None

    def test_exception_def_indent_body(self) -> None:
        exc = first(parse("exception MultiErr\n  code: int\n  reason: text"))
        assert isinstance(exc, ExceptionDef)
        assert exc.name == "MultiErr"
        assert len(exc.fields) == 2
        assert [f.name for f in exc.fields] == ["code", "reason"]

    def test_program_decl(self) -> None:
        pd = first(parse("program myapp\n()"))
        assert isinstance(pd, ProgramDecl)
        assert pd.name == "myapp"

    def test_builtin_func_def(self) -> None:
        fd = first(parse("builtin def encode(x: text) -> int"))
        assert isinstance(fd, FuncDef)
        assert fd.name == "encode"
        assert fd.is_builtin is True
        assert fd.body is None
        assert len(fd.params) == 1

    def test_builtin_record_def(self) -> None:
        rec = first(parse("builtin record Token(id: int)"))
        assert isinstance(rec, RecordDef)
        assert rec.name == "Token"
        assert rec.is_builtin is True
        assert len(rec.fields) == 1

    def test_builtin_enum_def(self) -> None:
        en = first(parse("builtin enum Status = Ok | Err"))
        assert isinstance(en, EnumDef)
        assert en.name == "Status"
        assert en.is_builtin is True
        assert len(en.variants) == 2


# ---------------------------------------------------------------------------
# parse_type_expr — direct unit tests
# ---------------------------------------------------------------------------


class TestParseTypeExpr:
    """parse_type_expr(text) parses a single AgL type expression."""

    def test_int(self) -> None:
        from agm.agl.syntax.types import IntT
        result = parse_type_expr("int")
        assert isinstance(result, IntT)

    def test_text(self) -> None:
        from agm.agl.syntax.types import TextT
        result = parse_type_expr("text")
        assert isinstance(result, TextT)

    def test_bool(self) -> None:
        from agm.agl.syntax.types import BoolT
        result = parse_type_expr("bool")
        assert isinstance(result, BoolT)

    def test_decimal(self) -> None:
        from agm.agl.syntax.types import DecimalT
        result = parse_type_expr("decimal")
        assert isinstance(result, DecimalT)

    def test_list_int(self) -> None:
        from agm.agl.syntax.types import IntT, ListT
        result = parse_type_expr("list[int]")
        assert isinstance(result, ListT)
        assert isinstance(result.elem, IntT)

    def test_dict_text_int(self) -> None:
        from agm.agl.syntax.types import DictT, IntT
        result = parse_type_expr("dict[text, int]")
        assert isinstance(result, DictT)
        assert isinstance(result.value, IntT)

    def test_named_type(self) -> None:
        from agm.agl.syntax.types import NameT
        result = parse_type_expr("MyRecord")
        assert isinstance(result, NameT)
        assert result.name == "MyRecord"

    def test_applied_generic(self) -> None:
        from agm.agl.syntax.types import AppliedT, IntT
        result = parse_type_expr("Option[int]")
        assert isinstance(result, AppliedT)
        assert result.name == "Option"
        assert len(result.args) == 1
        assert isinstance(result.args[0], IntT)

    def test_func_type(self) -> None:
        from agm.agl.syntax.types import FuncT, IntT, TextT
        result = parse_type_expr("(int) -> text")
        assert isinstance(result, FuncT)
        assert len(result.params) == 1
        assert isinstance(result.params[0], IntT)
        assert isinstance(result.result, TextT)

    def test_unit_type(self) -> None:
        from agm.agl.syntax.types import UnitT
        result = parse_type_expr("unit")
        assert isinstance(result, UnitT)

    def test_agent_type(self) -> None:
        from agm.agl.syntax.types import AgentT
        result = parse_type_expr("agent")
        assert isinstance(result, AgentT)

    def test_qualified_named_type(self) -> None:
        from agm.agl.syntax.types import AppliedT
        # mod::Box[int] — qualified applied type
        result = parse_type_expr("mymod::Box[int]")
        assert isinstance(result, AppliedT)
        assert result.name == "Box"
        assert result.module_qualifier is not None
        assert result.module_qualifier.segments == ("mymod",)

    def test_invalid_raises_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_type_expr("let x = 1")

    def test_empty_raises_syntax_error(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_type_expr("")

    def test_start_id_is_honoured(self) -> None:
        """start_id offsets the first assigned node_id."""
        result = parse_type_expr("int", start_id=100)
        from agm.agl.syntax.types import IntT
        assert isinstance(result, IntT)
        # Node id is ≥ start_id (offset applied)
        assert result.node_id >= 100


# ---------------------------------------------------------------------------
# program NAME declaration — side-table assertions via scope resolve
# ---------------------------------------------------------------------------


class TestProgramDeclScopeSideTables:
    """Parse a 'program NAME' source and assert the side tables on ResolvedProgram."""

    def _parse_and_resolve(self, source: str) -> object:
        from agm.agl.scope import resolve
        return resolve(parse_program(source))

    def test_program_name_set_in_resolved_program(self) -> None:
        """Parsing 'program myapp' sets program_name on ResolvedProgram."""
        r = self._parse_and_resolve("program myapp\n()")
        from agm.agl.scope.symbols import ResolvedProgram
        assert isinstance(r, ResolvedProgram)
        assert r.program_name == "myapp"

    def test_no_program_decl_gives_none(self) -> None:
        """No 'program' declaration → program_name is None."""
        r = self._parse_and_resolve("()")
        from agm.agl.scope.symbols import ResolvedProgram
        assert isinstance(r, ResolvedProgram)
        assert r.program_name is None

    def test_builtin_calls_populated_for_print(self) -> None:
        """A 'print' call is classified in builtin_calls."""
        r = self._parse_and_resolve('print "hello"')
        from agm.agl.scope import BuiltinKind
        from agm.agl.scope.symbols import ResolvedProgram
        assert isinstance(r, ResolvedProgram)
        assert BuiltinKind.PRINT in r.builtin_calls.values()

    def test_builtin_calls_populated_for_exec(self) -> None:
        """An 'exec' call is classified in builtin_calls."""
        r = self._parse_and_resolve('let x = exec "ls"\nx')
        from agm.agl.scope import BuiltinKind
        from agm.agl.scope.symbols import ResolvedProgram
        assert isinstance(r, ResolvedProgram)
        assert BuiltinKind.EXEC in r.builtin_calls.values()

    def test_bare_variant_patterns_populated(self) -> None:
        """A bare name in a case pattern that names a constructor is in bare_variant_patterns."""
        source = (
            "enum Status\n"
            "  | Ok\n"
            "  | Fail\n"
            "let s = Ok()\n"
            "case s of\n"
            "  | Ok => 1\n"
            "  | Fail => 0\n"
        )
        r = self._parse_and_resolve(source)
        from agm.agl.scope.symbols import ResolvedProgram
        assert isinstance(r, ResolvedProgram)
        # At least one VarPattern node_id was recognised as a bare-variant constructor.
        assert len(r.bare_variant_patterns) >= 1


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

    def test_juxt_call_index_access(self) -> None:
        """print xs[0] desugars to Call(print, (IndexAccess(xs, 0),), ())."""
        call = first(parse("print xs[0]"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "print"
        assert len(call.args) == 1
        arg = call.args[0]
        assert isinstance(arg, IndexAccess)
        assert isinstance(arg.obj, VarRef)
        assert arg.obj.name == "xs"
        assert isinstance(arg.index, IntLit)
        assert arg.index.value == 0

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

    def test_juxt_call_with_call_result(self) -> None:
        """print classify(x) desugars to print(classify(x))."""
        call = first(parse("print classify(x)"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "print"
        inner = call.args[0]
        assert isinstance(inner, Call)
        assert isinstance(inner.callee, VarRef)
        assert inner.callee.name == "classify"

    def test_juxt_call_with_qualified_constructor_call_argument(self) -> None:
        """f Opt.Some(x: 1) desugars to f(Opt.Some(x: 1))."""
        call = first(parse("f Opt.Some(x: 1)"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "f"

        arg = call.args[0]
        assert isinstance(arg, Call)
        assert isinstance(arg.callee, FieldAccess)
        assert arg.callee.field == "Some"
        assert isinstance(arg.callee.obj, VarRef)
        assert arg.callee.obj.name == "Opt"
        assert arg.args == ()
        assert len(arg.named_args) == 1
        assert arg.named_args[0].name == "x"
        assert isinstance(arg.named_args[0].value, IntLit)

    def test_juxt_call_list_literal_preserved(self) -> None:
        call = first(parse("f [2]"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "f"
        assert len(call.args) == 1
        assert isinstance(call.args[0], ListLit)

    def test_print_list_literal_juxt_preserved(self) -> None:
        call = first(parse("print [1,2,3]"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "print"
        assert isinstance(call.args[0], ListLit)

    def test_spaced_lsqb_does_not_index(self) -> None:
        call = first(parse("l [2]"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "l"
        assert isinstance(call.args[0], ListLit)

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


class TestTypedCalls:
    """``callee::[Type](args)`` desugars to a Call with static type_args."""

    def test_typed_call_basic(self) -> None:
        call = first(parse('ask-request::[Review]("p")'))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "ask-request"
        assert len(call.args) == 1
        assert call.named_args == ()
        assert len(call.type_args) == 1
        assert isinstance(call.type_args[0], NameT)
        assert call.type_args[0].name == "Review"

    def test_typed_call_type_brackets_survive_index_bracket_remap(self) -> None:
        call = first(parse('ask-request::[Review]("p")'))
        assert isinstance(call, Call)
        assert isinstance(call.type_args[0], NameT)
        assert call.type_args[0].name == "Review"

        generic = first(parse('ask-request::[list[Review]]("p")'))
        assert isinstance(generic, Call)
        assert isinstance(generic.type_args[0], ListT)
        assert isinstance(generic.type_args[0].elem, NameT)
        assert generic.type_args[0].elem.name == "Review"

    def test_typed_call_no_type_arg(self) -> None:
        # ``ask-request("p")`` (no ``::[...]``) is an ordinary Call with no type_args.
        call = first(parse('ask-request("p")'))
        assert isinstance(call, Call)
        assert call.type_args == ()

    def test_typed_call_primitive_type(self) -> None:
        call = first(parse('ask-request::[text]("p")'))
        assert isinstance(call, Call)
        assert isinstance(call.type_args[0], TextT)

    def test_typed_call_agent_type(self) -> None:
        call = first(parse('ask-request::[agent]("p")'))
        assert isinstance(call, Call)
        assert isinstance(call.type_args[0], AgentT)

    def test_typed_call_generic_type(self) -> None:
        call = first(parse('ask-request::[list[Review]]("p")'))
        assert isinstance(call, Call)
        assert isinstance(call.type_args[0], ListT)
        assert isinstance(call.type_args[0].elem, NameT)
        assert call.type_args[0].elem.name == "Review"

    def test_typed_call_dict_type(self) -> None:
        call = first(parse('ask-request::[dict[text, Review]]("p")'))
        assert isinstance(call, Call)
        assert isinstance(call.type_args[0], DictT)
        assert isinstance(call.type_args[0].value, NameT)
        assert call.type_args[0].value.name == "Review"

    def test_typed_call_with_named_args(self) -> None:
        call = first(parse('ask-request::[Review]("p", agent: reviewer)'))
        assert isinstance(call, Call)
        assert len(call.named_args) == 1
        assert call.named_args[0].name == "agent"
        assert len(call.type_args) == 1

    def test_typed_call_no_args(self) -> None:
        # An empty arg list is syntactically valid (the checker rejects it).
        call = first(parse("ask-request::[Review]()"))
        assert isinstance(call, Call)
        assert call.args == ()
        assert len(call.type_args) == 1

    def test_typed_call_trailing_comma(self) -> None:
        call = first(parse('ask-request::[Review]("p",)'))
        assert isinstance(call, Call)
        assert len(call.args) == 1

    def test_typed_call_preserves_list_literal_juxt(self) -> None:
        # ``::`` introduces the typed form without disturbing list-literal
        # juxtaposition (``print [1,2,3]`` still parses).
        call = first(parse("print [1,2,3]"))
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "print"
        assert isinstance(call.args[0], ListLit)

    def test_typed_call_accepts_field_access_callee(self) -> None:
        prog = parse("f.g::[T](x)")
        call = first(prog)
        assert isinstance(call, Call)
        assert isinstance(call.callee, FieldAccess)
        assert len(call.type_args) == 1

    def test_typed_call_allowed_as_juxt_argument(self) -> None:
        prog = parse("f Opt.None::[int]()")
        call = first(prog)
        assert isinstance(call, Call)
        assert isinstance(call.callee, VarRef)
        assert call.callee.name == "f"

        arg = call.args[0]
        assert isinstance(arg, Call)
        assert isinstance(arg.callee, FieldAccess)
        assert len(arg.type_args) == 1
        assert isinstance(arg.type_args[0], IntT)
        assert arg.args == ()

    def test_typed_call_with_args_allowed_as_juxt_argument(self) -> None:
        prog = parse("f Box::[int](1)")
        call = first(prog)
        assert isinstance(call, Call)
        arg = call.args[0]
        assert isinstance(arg, Call)
        assert isinstance(arg.callee, VarRef)
        assert arg.callee.name == "Box"
        assert len(arg.args) == 1
        assert isinstance(arg.args[0], IntLit)
        assert isinstance(arg.type_args[0], IntT)

    def test_dcolon_without_type_brackets_is_not_a_call(self) -> None:
        # ``ask-request::`` with no ``[...]`` is rejected.
        with pytest.raises(AglSyntaxError):
            parse('ask-request::"p"')


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
        assert isinstance(c, VarRef)
        assert c.name == "Pass"

    def test_constructor_with_args(self) -> None:
        c = first(parse("Issue(title: x, severity: 1)"))
        assert isinstance(c, Call)
        assert isinstance(c.callee, VarRef)
        assert c.callee.name == "Issue"
        assert len(c.named_args) == 2

    def test_constructor_with_brace_args_raises(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse("Issue{title: x, severity: 1}")

    def test_qualified_constructor(self) -> None:
        c = first(parse("Review.Pass"))
        assert isinstance(c, FieldAccess)
        assert isinstance(c.obj, VarRef)
        assert c.obj.name == "Review"
        assert c.field == "Pass"


class TestIndexAccess:
    def test_list_index_access(self) -> None:
        idx = first(parse("l[2]"))
        assert isinstance(idx, IndexAccess)
        assert isinstance(idx.obj, VarRef)
        assert idx.obj.name == "l"
        assert isinstance(idx.index, IntLit)
        assert idx.index.value == 2

    def test_dict_index_access(self) -> None:
        idx = first(parse('d["a"]'))
        assert isinstance(idx, IndexAccess)
        assert isinstance(idx.obj, VarRef)
        assert idx.obj.name == "d"
        assert isinstance(idx.index, StringLit)
        assert idx.index.value == "a"

    def test_chained_indexes(self) -> None:
        idx = first(parse("matrix[0][1]"))
        assert isinstance(idx, IndexAccess)
        assert isinstance(idx.index, IntLit)
        assert idx.index.value == 1
        assert isinstance(idx.obj, IndexAccess)
        assert isinstance(idx.obj.obj, VarRef)
        assert idx.obj.obj.name == "matrix"

    def test_index_then_field(self) -> None:
        fa = first(parse("rows[0].name"))
        assert isinstance(fa, FieldAccess)
        assert fa.field == "name"
        assert isinstance(fa.obj, IndexAccess)
        assert isinstance(fa.obj.obj, VarRef)
        assert fa.obj.obj.name == "rows"

    def test_call_result_index(self) -> None:
        idx = first(parse("make()[0]"))
        assert isinstance(idx, IndexAccess)
        assert isinstance(idx.obj, Call)
        assert isinstance(idx.obj.callee, VarRef)
        assert idx.obj.callee.name == "make"

    def test_list_literal_index(self) -> None:
        idx = first(parse("[1, 2, 3][0]"))
        assert isinstance(idx, IndexAccess)
        assert isinstance(idx.obj, ListLit)


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

    def test_eq_eq_message(self) -> None:
        """== triggers the 'Use `=` for equality.' friendly diagnostic (design §2.3)."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse("let b = a == b")
        assert str(exc_info.value) == "Use `=` for equality."

    @pytest.mark.parametrize(
        "src",
        [
            "x = y = z",
            "1 < 2 < 3",
            "a <= b != c",
        ],
    )
    def test_chained_comparison_raises(self, src: str) -> None:
        """Chained comparisons are non-associative in AgL (design §4.3)."""
        with pytest.raises(AglSyntaxError, match="non-associative"):
            parse(src)

    def test_chained_comparison_full_message(self) -> None:
        """Pins the full designed wording for the chained-comparison diagnostic."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse("x = y = z")
        assert str(exc_info.value) == (
            "Comparisons are non-associative; parenthesize explicitly, "
            "e.g. `(x = y) = z`."
        )


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
        src = "case x of | Pass() => ok | Fail() => err"
        e = first(parse(src))
        assert isinstance(e, Case)
        assert isinstance(e.subject, VarRef)
        assert len(e.branches) == 2

    def test_case_first_pipe_optional(self) -> None:
        src = "case x of Pass() => ok | Fail() => err"
        e = first(parse(src))
        assert isinstance(e, Case)
        assert isinstance(e.subject, VarRef)
        assert len(e.branches) == 2

    def test_case_first_pipe_optional_multiline(self) -> None:
        src = "case x of\n  Pass() => ok\n  | Fail() => err"
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
        src = "case x of | Pass() => ok | Fail() => err"
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
        e = first(parse("do x := 1 until x > 5"))
        assert isinstance(e, Do)
        assert e.limit is None
        assert isinstance(e.condition, BinaryOp)

    def test_do_with_bound(self) -> None:
        e = first(parse("do[10] x := 1 until x > 5"))
        assert isinstance(e, Do)
        assert e.limit == 10

    def test_do_zero_bound_raises(self) -> None:
        with pytest.raises(AglSyntaxError, match="positive"):
            parse("do[0] x until true")

    def test_do_suite_body(self) -> None:
        src = "do\n  x := 1\n  y := 2\nuntil x > 5"
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
        assert isinstance(e.exc, Call)


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
        prog2, _next_id2 = parse_program_seeded("let y = 2", start_id=next_id1)
        ids1: list[int] = []
        _collect_node_ids(prog1, ids1)
        ids2: list[int] = []
        _collect_node_ids(prog2, ids2)
        assert set(ids1).isdisjoint(set(ids2)), (
            "node_ids from separate parse_program_seeded calls must not overlap"
        )

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

    def test_bare_assignment_is_equality(self) -> None:
        """In v2, n = 2 is a BinaryOp(EQ) expression (not a mutation).
        Mutation uses `n := 2`. The parser accepts n = 2 as an expression.
        The scope pass would verify mutation intent.
        """
        e = first(parse("n = 2"))
        assert isinstance(e, BinaryOp)
        assert e.op == BinOp.EQ

    def test_duplicate_constructor_arg_raises(self) -> None:
        with pytest.raises(AglSyntaxError, match="duplicate"):
            parse("Issue(title: a, title: b)")

    def test_braced_constructor_arg_raises(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse("Issue{title: a, title: b}")

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
    """Covers config decl values not yet tested: false and decimal."""

    def test_config_decl_false(self) -> None:
        cfg = first(parse("config log = false"))
        assert isinstance(cfg, ConfigDecl)
        assert cfg.name == "log"
        assert isinstance(cfg.value, BoolLit)
        assert cfg.value.value is False

    def test_config_decl_decimal(self) -> None:
        cfg = first(parse("config rate = 1.5"))
        assert isinstance(cfg, ConfigDecl)
        assert cfg.name == "rate"
        assert isinstance(cfg.value, DecimalLit)
        assert cfg.value.value == decimal.Decimal("1.5")


class TestTypeExprCoverage:
    """Covers type expression paths not yet tested."""

    def test_named_type_via_varname(self) -> None:
        """A lowercase VAR_NAME that isn't a keyword maps to NameT."""
        let = first(parse("let x: mytype = 1"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, NameT)
        assert let.type_ann.name == "mytype"

    def test_dict_type_bad_key_raises(self) -> None:
        """dict[int, text] raises — dict keys must be text; message shows source spelling."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse("let x: dict[int, text] = 1")
        msg = str(exc_info.value)
        assert "'int'" in msg
        assert "IntT" not in msg


class TestCallsCoverage:
    """Covers call paths not yet tested."""

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
        """An param that causes a LexError returns False (real error, not incomplete)."""
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
    """Covers AglSyntaxError.source_span returning a valid SourceSpan."""

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
    """Covers the else branch of the inline-compound error dispatch.

    The else branch fires when the unexpected inline-blocked token is NOT
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


# ---------------------------------------------------------------------------
# Generics / parametric polymorphism
# ---------------------------------------------------------------------------


class TestGenerics:
    """Tests for parametric polymorphism: record/enum/def/type with type_params."""

    def test_generic_record_def(self) -> None:
        """record Pair[A, B] with two type params."""
        prog = parse("record Pair[A, B]\n  first: A\n  second: B")
        rec = first(prog)
        assert isinstance(rec, RecordDef)
        assert rec.name == "Pair"
        assert rec.type_params == ("A", "B")
        assert len(rec.fields) == 2

    def test_generic_enum_def(self) -> None:
        """enum Option[T] with one type param."""
        prog = parse("enum Option[T]\n  | None\n  | Some(value: T)")
        en = first(prog)
        assert isinstance(en, EnumDef)
        assert en.name == "Option"
        assert en.type_params == ("T",)
        assert len(en.variants) == 2

    def test_generic_type_alias(self) -> None:
        """type Map[K] = dict[text, K] — type alias with type param."""
        prog = parse("type Map[K] = dict[text, K]")
        alias = first(prog)
        assert isinstance(alias, TypeAlias)
        assert alias.name == "Map"
        assert alias.type_params == ("K",)
        assert isinstance(alias.type_expr, DictT)

    def test_generic_func_def(self) -> None:
        """def identity[T](x: T) -> T = x — polymorphic function."""
        prog = parse("def identity[T](x: T) -> T = x")
        fn = first(prog)
        assert isinstance(fn, FuncDef)
        assert fn.name == "identity"
        assert fn.type_params == ("T",)
        assert len(fn.params) == 1

    def test_applied_type_in_annotation(self) -> None:
        """let x: Option[int] = y produces AppliedT in type_ann."""
        let = first(parse("let x: Option[int] = y"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, AppliedT)
        assert let.type_ann.name == "Option"
        assert len(let.type_ann.args) == 1
        assert isinstance(let.type_ann.args[0], IntT)

    def test_applied_type_multi_arg(self) -> None:
        """let x: Pair[int, text] = y produces AppliedT with 2 args."""
        let = first(parse("let x: Pair[int, text] = y"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, AppliedT)
        assert let.type_ann.name == "Pair"
        assert len(let.type_ann.args) == 2
        assert isinstance(let.type_ann.args[0], IntT)
        assert isinstance(let.type_ann.args[1], TextT)

    def test_non_generic_record_has_empty_type_params(self) -> None:
        """Plain record Issue has type_params == ()."""
        prog = parse("record Issue\n  title: text")
        rec = first(prog)
        assert isinstance(rec, RecordDef)
        assert rec.type_params == ()

    def test_typed_call_multi_type_args(self) -> None:
        """callee::[A, B](x) produces Call with two type_args."""
        call = first(parse("f::[int, text](x)"))
        assert isinstance(call, Call)
        assert len(call.type_args) == 2
        assert isinstance(call.type_args[0], IntT)
        assert isinstance(call.type_args[1], TextT)


class TestSpacedBrackets:
    """type_lsqb accepts both adjacent and spaced brackets for type params/args."""

    def test_spaced_type_params_func_def(self) -> None:
        """def identity [T](x: T) -> T = x — spaced bracket form."""
        fn = first(parse("def identity [T](x: T) -> T = x"))
        assert isinstance(fn, FuncDef)
        assert fn.type_params == ("T",)

    def test_spaced_type_params_equals_adjacent(self) -> None:
        """Spaced and adjacent forms produce identical type_params tuples."""
        adjacent = first(parse("def identity[T](x: T) -> T = x"))
        spaced = first(parse("def identity [T](x: T) -> T = x"))
        assert isinstance(adjacent, FuncDef)
        assert isinstance(spaced, FuncDef)
        assert spaced.type_params == adjacent.type_params

    def test_spaced_applied_type_annotation(self) -> None:
        """let x: Option [int] = y — spaced bracket in type annotation."""
        let = first(parse("let x: Option [int] = y"))
        assert isinstance(let, LetDecl)
        assert isinstance(let.type_ann, AppliedT)
        assert let.type_ann.name == "Option"
        assert len(let.type_ann.args) == 1
        assert isinstance(let.type_ann.args[0], IntT)

    def test_spaced_applied_type_equals_adjacent(self) -> None:
        """Spaced applied type produces the same AST as adjacent."""
        adjacent = first(parse("let x: Option[int] = y"))
        spaced = first(parse("let x: Option [int] = y"))
        assert isinstance(adjacent, LetDecl)
        assert isinstance(spaced, LetDecl)
        assert spaced.type_ann == adjacent.type_ann

    def test_spaced_type_params_record_def(self) -> None:
        """record Pair [A, B] — spaced bracket in record definition."""
        rec = first(parse("record Pair [A, B]\n  first: A\n  second: B"))
        assert isinstance(rec, RecordDef)
        assert rec.type_params == ("A", "B")


# ---------------------------------------------------------------------------
# Case-neutral names (M1 generics)
# ---------------------------------------------------------------------------


class TestCaseNeutralNamesParser:
    """Case-neutral names: uppercase and lowercase identifiers are both NAME tokens."""

    def test_uppercase_let_binding(self) -> None:
        # Names are case-neutral; uppercase is just a NAME
        prog = parse("let X = 1")
        d = first(prog)
        assert isinstance(d, LetDecl)
        assert d.name == "X"

    def test_lowercase_type_in_type_ann(self) -> None:
        prog = parse("let x: mytype = 1")
        d = first(prog)
        assert isinstance(d, LetDecl)
        assert isinstance(d.type_ann, NameT)
        assert d.type_ann.name == "mytype"

    def test_uppercase_type_in_type_ann(self) -> None:
        prog = parse("let x: MyType = 1")
        d = first(prog)
        assert isinstance(d, LetDecl)
        assert isinstance(d.type_ann, NameT)
        assert d.type_ann.name == "MyType"

    def test_constructor_as_var_ref(self) -> None:
        # Uppercase name in expression position -> VarRef
        prog = parse("Foo")
        ref = first(prog)
        assert isinstance(ref, VarRef)
        assert ref.name == "Foo"

    def test_lowercase_name_as_var_ref(self) -> None:
        prog = parse("none")
        ref = first(prog)
        assert isinstance(ref, VarRef)
        assert ref.name == "none"

    def test_constructor_call_becomes_call(self) -> None:
        # some(value: 1) -> Call(callee=VarRef("some"), named_args=(...))
        prog = parse("some(value: 1)")
        c = first(prog)
        assert isinstance(c, Call)
        assert isinstance(c.callee, VarRef)
        assert c.callee.name == "some"
        assert len(c.named_args) == 1
        assert c.named_args[0].name == "value"

    def test_uppercase_constructor_call_becomes_call(self) -> None:
        prog = parse("Some(value: 1)")
        c = first(prog)
        assert isinstance(c, Call)
        assert isinstance(c.callee, VarRef)
        assert c.callee.name == "Some"

    def test_qualified_access_becomes_field_access(self) -> None:
        # Option.some -> FieldAccess(VarRef("Option"), "some")
        prog = parse("Option.some")
        fa = first(prog)
        assert isinstance(fa, FieldAccess)
        assert isinstance(fa.obj, VarRef)
        assert fa.obj.name == "Option"
        assert fa.field == "some"

    def test_qualified_call_becomes_call_with_field_access(self) -> None:
        # Option.some(value: 1) -> Call(callee=FieldAccess(VarRef("Option"), "some"), ...)
        prog = parse("Option.some(value: 1)")
        c = first(prog)
        assert isinstance(c, Call)
        assert isinstance(c.callee, FieldAccess)
        assert c.callee.field == "some"


class TestCaseNeutralPatterns:
    """Case-neutral pattern matching: both upper- and lower-case names work the same."""

    def test_bare_lowercase_name_is_var_pattern(self) -> None:
        prog = parse("case x of | y => y")
        case = first(prog)
        assert isinstance(case, Case)
        pat = case.branches[0].pattern
        assert isinstance(pat, VarPattern)
        assert pat.name == "y"

    def test_bare_uppercase_name_is_var_pattern(self) -> None:
        # With case-neutral names, bare uppercase is also a VarPattern
        prog = parse("case x of | Y => Y")
        case = first(prog)
        assert isinstance(case, Case)
        pat = case.branches[0].pattern
        assert isinstance(pat, VarPattern)
        assert pat.name == "Y"

    def test_underscore_is_wildcard(self) -> None:
        prog = parse("case x of | _ => 1")
        case = first(prog)
        assert isinstance(case, Case)
        pat = case.branches[0].pattern
        assert isinstance(pat, WildcardPattern)

    def test_constructor_pattern_with_parens(self) -> None:
        prog = parse("case x of | none() => 1")
        case = first(prog)
        assert isinstance(case, Case)
        pat = case.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.qualifier is None
        assert pat.name == "none"
        assert pat.fields == ()

    def test_uppercase_constructor_pattern_with_parens(self) -> None:
        prog = parse("case x of | None() => 1")
        case = first(prog)
        assert isinstance(case, Case)
        pat = case.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.name == "None"

    def test_qualified_constructor_pattern(self) -> None:
        prog = parse("case x of | Option.none => 1")
        case = first(prog)
        assert isinstance(case, Case)
        pat = case.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.qualifier == "Option"
        assert pat.name == "none"

    def test_constructor_pattern_with_fields(self) -> None:
        prog = parse("case x of | some(value: v) => v")
        case = first(prog)
        assert isinstance(case, Case)
        pat = case.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.name == "some"
        assert len(pat.fields) == 1
        assert pat.fields[0].name == "value"


# ---------------------------------------------------------------------------
# Cast expression tests (M1)
# ---------------------------------------------------------------------------


class TestCastParsing:
    """Tests for `as` (cast) and `as?` (convertibility test) operator parsing.

    Precedence (D5): unary(-) > [cast] > * / > + -
    So:
      -1 as text     = (-1) as text       (unary binds tighter)
      2 * 3 as text  = 2 * (3 as text)    (cast binds tighter than *)
      1 + 2 as text  = 1 + (2 as text)    (cast binds tighter than +)
      f x as int     = (f x) as int        (juxt binds tighter than cast)
    Left-associative chaining:
      x as json as text = (x as json) as text
    as? precedence:
      a as? int and b = (a as? int) and b  (cast > and)
    """

    def _cast(self, src: str) -> Cast:
        """Parse a single top-level expression and assert it is a Cast node."""
        prog = parse(src)
        expr = first(prog)
        assert isinstance(expr, Cast), f"expected Cast, got {type(expr).__name__}: {src!r}"
        return expr

    def test_simple_as_cast(self) -> None:
        node = self._cast("x as text")
        assert isinstance(node.expr, VarRef)
        assert node.expr.name == "x"
        assert isinstance(node.target_type, TextT)
        assert node.test_only is False

    def test_simple_as_question_test(self) -> None:
        node = self._cast("x as? int")
        assert isinstance(node.expr, VarRef)
        assert node.expr.name == "x"
        assert isinstance(node.target_type, IntT)
        assert node.test_only is True

    def test_unary_neg_binds_tighter_than_cast(self) -> None:
        # -1 as text  =>  (-1) as text
        node = self._cast("-1 as text")
        assert isinstance(node.expr, UnaryNeg)
        inner = node.expr.operand
        assert isinstance(inner, IntLit)
        assert inner.value == 1
        assert isinstance(node.target_type, TextT)
        assert node.test_only is False

    def test_cast_binds_tighter_than_multiply(self) -> None:
        # 2 * 3 as text  =>  2 * (3 as text)
        prog = parse("2 * 3 as text")
        expr = first(prog)
        assert isinstance(expr, BinaryOp)
        assert expr.op == BinOp.MUL
        assert isinstance(expr.left, IntLit)
        assert expr.left.value == 2
        assert isinstance(expr.right, Cast)
        assert isinstance(expr.right.expr, IntLit)
        assert expr.right.expr.value == 3

    def test_cast_binds_tighter_than_additive(self) -> None:
        # 1 + 2 as text  =>  1 + (2 as text)
        prog = parse("1 + 2 as text")
        expr = first(prog)
        assert isinstance(expr, BinaryOp)
        assert expr.op == BinOp.ADD
        assert isinstance(expr.left, IntLit)
        assert expr.left.value == 1
        assert isinstance(expr.right, Cast)
        assert isinstance(expr.right.expr, IntLit)
        assert expr.right.expr.value == 2

    def test_juxt_binds_tighter_than_cast(self) -> None:
        # f x as int  =>  (f x) as int  (juxt is tighter)
        node = self._cast("f x as int")
        assert isinstance(node.expr, Call)
        callee = node.expr.callee
        assert isinstance(callee, VarRef)
        assert callee.name == "f"
        assert isinstance(node.target_type, IntT)

    def test_cast_left_associative_chaining(self) -> None:
        # x as json as text  =>  (x as json) as text
        outer = self._cast("x as json as text")
        assert isinstance(outer.target_type, TextT)
        inner = outer.expr
        assert isinstance(inner, Cast)
        assert isinstance(inner.target_type, JsonT)
        assert isinstance(inner.expr, VarRef)

    def test_as_question_left_associative(self) -> None:
        # a as? int and b  =>  (a as? int) and b
        prog = parse("a as? int and b")
        expr = first(prog)
        assert isinstance(expr, BinaryOp)
        assert expr.op == BinOp.AND
        assert isinstance(expr.left, Cast)
        assert expr.left.test_only is True

    def test_cast_with_named_type(self) -> None:
        node = self._cast("x as MyType")
        assert isinstance(node.target_type, NameT)
        assert node.target_type.name == "MyType"

    def test_cast_result_type_bool(self) -> None:
        node = self._cast("val as? bool")
        assert node.test_only is True
        assert isinstance(node.target_type, BoolT)

    def test_cast_with_decimal_type(self) -> None:
        node = self._cast("x as decimal")
        assert isinstance(node.target_type, DecimalT)
        assert node.test_only is False

    def test_as_missing_type_raises(self) -> None:
        """Bare `as` with no type raises AglSyntaxError (EOF or newline)."""
        with pytest.raises(AglSyntaxError):
            parse("x as")


# ---------------------------------------------------------------------------
# Import declaration tests
# ---------------------------------------------------------------------------


class TestImportDecl:
    """Tests for import declaration parsing."""

    def test_simple_import(self) -> None:
        prog = parse("import foo.bar")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.module_path == ("foo", "bar")
        assert decl.wildcard is False
        assert decl.qualified is False
        assert decl.alias is None
        assert decl.mode == syntax.ImportMode.ALL
        assert decl.items == ()

    def test_single_segment_import(self) -> None:
        prog = parse("import utils")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.module_path == ("utils",)

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("import FooBar", ("FooBar",)),
            ("import foo.Qux2", ("foo", "Qux2")),
        ],
    )
    def test_import_path_accepts_uppercase_segments(
        self, source: str, expected: tuple[str, ...]
    ) -> None:
        prog = parse(source)
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.module_path == expected

    def test_import_wildcard(self) -> None:
        prog = parse("import foo.bar.*")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.wildcard is True
        assert decl.module_path == ("foo", "bar")

    def test_import_qualified(self) -> None:
        prog = parse("import foo qualified")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.qualified is True

    def test_import_with_alias(self) -> None:
        prog = parse("import foo.bar as fb")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.alias == "fb"

    def test_import_qualified_with_alias(self) -> None:
        prog = parse("import foo qualified as f")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.qualified is True
        assert decl.alias == "f"

    def test_import_using_single_item(self) -> None:
        prog = parse("import foo using bar")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.mode == syntax.ImportMode.USING
        assert len(decl.items) == 1
        assert decl.items[0].name == "bar"
        assert decl.items[0].rename is None

    def test_import_using_multiple_items(self) -> None:
        prog = parse("import foo using bar, baz")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.mode == syntax.ImportMode.USING
        assert len(decl.items) == 2
        names = {i.name for i in decl.items}
        assert names == {"bar", "baz"}

    def test_import_using_with_rename(self) -> None:
        prog = parse("import foo using bar as b")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.items[0].name == "bar"
        assert decl.items[0].rename == "b"

    def test_import_hiding_single_name(self) -> None:
        prog = parse("import foo hiding secret")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.mode == syntax.ImportMode.HIDING
        assert len(decl.items) == 1
        assert decl.items[0].name == "secret"

    def test_import_hiding_multiple_names(self) -> None:
        prog = parse("import foo hiding a, b")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.mode == syntax.ImportMode.HIDING
        names = {i.name for i in decl.items}
        assert names == {"a", "b"}

    def test_import_using_type_name(self) -> None:
        # TYPE_NAME in using clause
        prog = parse("import foo using Bar")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.items[0].name == "Bar"

    def test_import_after_other_decl(self) -> None:
        """import can appear after other declarations."""
        prog = parse("let x = 1\nimport foo")
        it = items(prog)
        assert isinstance(it[1], syntax.ImportDecl)

    def test_import_is_in_declaration_union(self) -> None:
        """ImportDecl is part of the Declaration union (item-level)."""
        prog = parse("import foo")
        (decl,) = items(prog)
        # Declaration is a type alias union; check it's ImportDecl
        assert type(decl).__name__ == "ImportDecl"

    def test_import_qualified_using_two_items(self) -> None:
        """import foo.bar qualified using x, y — qualified flag + using list."""
        prog = parse("import foo.bar qualified using x, y")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.module_path == ("foo", "bar")
        assert decl.qualified is True
        assert decl.mode == syntax.ImportMode.USING
        assert {i.name for i in decl.items} == {"x", "y"}

    def test_import_qualified_as_alias_using_two_items(self) -> None:
        """import foo.bar qualified as A using x, y — all three clauses."""
        prog = parse("import foo.bar qualified as A using x, y")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.qualified is True
        assert decl.alias == "A"
        assert decl.mode == syntax.ImportMode.USING
        assert {i.name for i in decl.items} == {"x", "y"}

    def test_import_qualified_hiding_two_items(self) -> None:
        """import foo.bar qualified hiding x, y — qualified flag + hiding list."""
        prog = parse("import foo.bar qualified hiding x, y")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.qualified is True
        assert decl.mode == syntax.ImportMode.HIDING
        assert {i.name for i in decl.items} == {"x", "y"}

    def test_import_wildcard_with_alias(self) -> None:
        """import foo.bar.* as A — wildcard import with alias."""
        prog = parse("import foo.bar.* as A")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.module_path == ("foo", "bar")
        assert decl.wildcard is True
        assert decl.alias == "A"

    def test_import_alias_uppercase(self) -> None:
        """import foo.bar as A — uppercase TYPE_NAME alias is accepted."""
        prog = parse("import foo.bar as A")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.module_path == ("foo", "bar")
        assert decl.alias == "A"

    # --- Wildcard import with using/hiding (Finding 1) ---

    def test_import_wildcard_using(self) -> None:
        """import foo.* using x, y — wildcard + using clause."""
        prog = parse("import foo.* using x, y")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.wildcard is True
        assert decl.module_path == ("foo",)
        assert decl.mode == syntax.ImportMode.USING
        assert {i.name for i in decl.items} == {"x", "y"}
        assert decl.alias is None
        assert decl.qualified is False

    def test_import_wildcard_hiding(self) -> None:
        """import foo.* hiding x — wildcard + hiding clause."""
        prog = parse("import foo.* hiding x")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.wildcard is True
        assert decl.module_path == ("foo",)
        assert decl.mode == syntax.ImportMode.HIDING
        assert len(decl.items) == 1
        assert decl.items[0].name == "x"

    def test_import_wildcard_using_with_rename(self) -> None:
        """import foo.* using x as X — wildcard + using item rename."""
        prog = parse("import foo.* using x as X")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.wildcard is True
        assert decl.mode == syntax.ImportMode.USING
        assert len(decl.items) == 1
        assert decl.items[0].name == "x"
        assert decl.items[0].rename == "X"

    def test_import_wildcard_qualified_using(self) -> None:
        """import foo.* qualified using x — wildcard + qualified + using."""
        prog = parse("import foo.* qualified using x")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.wildcard is True
        assert decl.qualified is True
        assert decl.mode == syntax.ImportMode.USING
        assert decl.items[0].name == "x"

    def test_import_wildcard_alias_using(self) -> None:
        """import foo.bar.* as A using x — wildcard + alias + using."""
        prog = parse("import foo.bar.* as A using x")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ImportDecl)
        assert decl.wildcard is True
        assert decl.module_path == ("foo", "bar")
        assert decl.alias == "A"
        assert decl.mode == syntax.ImportMode.USING
        assert decl.items[0].name == "x"


# ---------------------------------------------------------------------------
# Qualified reference tests
# ---------------------------------------------------------------------------


class TestQualifiedRefs:
    """Tests for qualified variable and constructor references in expressions."""

    def test_qual_var_ref_simple(self) -> None:
        prog = parse("foo::bar")
        (expr,) = items(prog)
        assert isinstance(expr, syntax.VarRef)
        assert expr.name == "bar"
        assert expr.module_qualifier is not None
        assert expr.module_qualifier.segments == ("foo",)

    def test_qual_var_ref_dotted(self) -> None:
        prog = parse("foo.bar::baz")
        (expr,) = items(prog)
        assert isinstance(expr, syntax.VarRef)
        assert expr.name == "baz"
        assert expr.module_qualifier is not None
        assert expr.module_qualifier.segments == ("foo", "bar")

    def test_qual_constructor_simple(self) -> None:
        # foo::Bar → VarRef(name="Bar", module_qualifier=Qualifier(["foo"]))
        prog = parse("foo::Bar")
        (expr,) = items(prog)
        assert isinstance(expr, syntax.VarRef)
        assert expr.name == "Bar"
        assert expr.module_qualifier is not None
        assert expr.module_qualifier.segments == ("foo",)

    def test_qual_constructor_dotted(self) -> None:
        prog = parse("my.mod::Baz")
        (expr,) = items(prog)
        assert isinstance(expr, syntax.VarRef)
        assert expr.name == "Baz"
        assert expr.module_qualifier is not None
        assert expr.module_qualifier.segments == ("my", "mod")

    def test_self_ref_var(self) -> None:
        prog = parse("::myvar")
        (expr,) = items(prog)
        assert isinstance(expr, syntax.VarRef)
        assert expr.name == "myvar"
        assert expr.module_qualifier is not None
        assert expr.module_qualifier.segments == ()

    def test_self_ref_constructor(self) -> None:
        # ::MyType → VarRef(name="MyType", module_qualifier=Qualifier(segments=()))
        prog = parse("::MyType")
        (expr,) = items(prog)
        assert isinstance(expr, syntax.VarRef)
        assert expr.name == "MyType"
        assert expr.module_qualifier is not None
        assert expr.module_qualifier.segments == ()

    def test_qual_constructor_with_payload(self) -> None:
        # m::Color(r: 1) → Call(VarRef("Color", mq=...), named_args=[NamedArg("r", 1)])
        prog = parse("m::Color(r: 1)")
        (call,) = items(prog)
        assert isinstance(call, syntax.Call)
        callee = call.callee
        assert isinstance(callee, syntax.VarRef)
        assert callee.name == "Color"
        assert callee.module_qualifier is not None
        assert callee.module_qualifier.segments == ("m",)
        assert len(call.named_args) == 1
        assert call.named_args[0].name == "r"

    def test_qual_var_ref_in_call_position(self) -> None:
        prog = parse("m::myfunc(1)")
        (call,) = items(prog)
        assert isinstance(call, syntax.Call)
        callee = call.callee
        assert isinstance(callee, syntax.VarRef)
        assert callee.module_qualifier is not None

    def test_qual_enum_variant_access(self) -> None:
        # m::Color.Red → FieldAccess(VarRef("Color", mq=...), "Red")
        prog = parse("m::Color.Red")
        (expr,) = items(prog)
        assert isinstance(expr, syntax.FieldAccess)
        assert isinstance(expr.obj, syntax.VarRef)
        assert expr.obj.name == "Color"
        assert expr.obj.module_qualifier is not None
        assert expr.field == "Red"

    def test_typed_call_not_confused_with_qual(self) -> None:
        # foo::[int](...) is typed call, not a MODQUAL
        prog = parse("foo::[int](1)")
        (call,) = items(prog)
        assert isinstance(call, syntax.Call)
        # module_qualifier on callee should be None (plain VarRef)
        callee = call.callee
        assert isinstance(callee, syntax.VarRef)
        assert callee.module_qualifier is None


# ---------------------------------------------------------------------------
# Qualified type reference tests
# ---------------------------------------------------------------------------


class TestQualifiedTypeRefs:
    """Tests for qualified type references in type expressions."""

    def test_qual_named_type_in_annotation(self) -> None:
        prog = parse("let x: m::MyType = null")
        (decl,) = items(prog)
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.type_ann, NameT)
        assert decl.type_ann.name == "MyType"
        assert decl.type_ann.module_qualifier is not None
        assert decl.type_ann.module_qualifier.segments == ("m",)

    def test_qual_applied_type_in_annotation(self) -> None:
        prog = parse("let x: m::Box[int] = null")
        (decl,) = items(prog)
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.type_ann, syntax.AppliedT)
        assert decl.type_ann.name == "Box"
        assert decl.type_ann.module_qualifier is not None
        assert decl.type_ann.module_qualifier.segments == ("m",)

    def test_qualified_enum_constructor_with_type_args(self) -> None:
        prog = parse("Option.some::[int](value: 1)")
        (call,) = items(prog)
        assert isinstance(call, syntax.Call)
        assert isinstance(call.callee, syntax.FieldAccess)
        assert call.callee.field == "some"
        assert len(call.type_args) == 1

    def test_qual_prim_type_in_annotation(self) -> None:
        prog = parse("let x: m::text = null")
        (decl,) = items(prog)
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.type_ann, NameT)
        assert decl.type_ann.name == "text"
        assert decl.type_ann.module_qualifier is not None

    def test_self_ref_named_type(self) -> None:
        prog = parse("let x: ::MyT = null")
        (decl,) = items(prog)
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.type_ann, NameT)
        assert decl.type_ann.module_qualifier is not None
        assert decl.type_ann.module_qualifier.segments == ()

    def test_qual_named_type_in_func_return(self) -> None:
        prog = parse("def f() -> m::Result = null")
        (decl,) = items(prog)
        assert isinstance(decl, FuncDef)
        assert isinstance(decl.return_type, NameT)
        assert decl.return_type.module_qualifier is not None

    def test_qual_pattern_constructor(self) -> None:
        prog = parse("case x of | m::Foo => 1")
        (expr,) = items(prog)
        assert isinstance(expr, Case)
        pat = expr.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.name == "Foo"
        assert pat.module_qualifier is not None
        assert pat.module_qualifier.segments == ("m",)

    def test_self_ref_pattern_constructor(self) -> None:
        prog = parse("case x of | ::Bar => 2")
        (expr,) = items(prog)
        assert isinstance(expr, Case)
        pat = expr.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.module_qualifier is not None
        assert pat.module_qualifier.segments == ()

    def test_qual_pattern_enum_variant(self) -> None:
        # Qualified enum variant: m::Color.Red (two TYPE_NAME tokens after qual_prefix)
        prog = parse("case x of | m::Color.Red => 1")
        (expr,) = items(prog)
        assert isinstance(expr, Case)
        pat = expr.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.qualifier == "Color"
        assert pat.name == "Red"
        assert pat.module_qualifier is not None
        assert pat.module_qualifier.segments == ("m",)

    def test_qual_pattern_constructor_with_fields(self) -> None:
        # Qualified constructor pattern with payload fields
        prog = parse("case x of | m::Foo(y: z) => 1")
        (expr,) = items(prog)
        assert isinstance(expr, Case)
        pat = expr.branches[0].pattern
        assert isinstance(pat, ConstructorPattern)
        assert pat.name == "Foo"
        assert len(pat.fields) == 1
        assert pat.module_qualifier is not None


# ---------------------------------------------------------------------------
# Private declaration tests
# ---------------------------------------------------------------------------


class TestPrivateDecls:
    """Tests for private record/enum/type alias/func declarations."""

    def test_private_func_def(self) -> None:
        prog = parse('private def f() -> text = "hi"')
        (decl,) = items(prog)
        assert isinstance(decl, FuncDef)
        assert decl.is_private is True
        assert decl.name == "f"

    def test_private_func_preserves_body(self) -> None:
        prog = parse("private def add(x: int) -> int = x")
        (decl,) = items(prog)
        assert isinstance(decl, FuncDef)
        assert decl.is_private is True
        assert len(decl.params) == 1

    def test_private_record_def(self) -> None:
        prog = parse(
            "private record Foo\n"
            "    bar: text"
        )
        (decl,) = items(prog)
        assert isinstance(decl, RecordDef)
        assert decl.is_private is True
        assert decl.name == "Foo"

    def test_private_enum_def(self) -> None:
        prog = parse("private enum Color | Red | Green | Blue")
        (decl,) = items(prog)
        assert isinstance(decl, EnumDef)
        assert decl.is_private is True
        assert decl.name == "Color"

    def test_private_type_alias(self) -> None:
        prog = parse("private type Alias = text")
        (decl,) = items(prog)
        assert isinstance(decl, TypeAlias)
        assert decl.is_private is True
        assert decl.name == "Alias"

    def test_non_private_func_is_not_private(self) -> None:
        prog = parse('def g() -> text = "g"')
        (decl,) = items(prog)
        assert isinstance(decl, FuncDef)
        assert decl.is_private is False

    def test_non_private_record_is_not_private(self) -> None:
        prog = parse("record Bar\n    x: int")
        (decl,) = items(prog)
        assert isinstance(decl, RecordDef)
        assert decl.is_private is False

    def test_private_and_public_decls_together(self) -> None:
        """Public and private declarations can coexist."""
        prog = parse('def pub() -> text = "a"\nprivate def priv() -> text = "b"')
        it = items(prog)
        assert isinstance(it[0], FuncDef)
        assert it[0].is_private is False
        assert isinstance(it[1], FuncDef)
        assert it[1].is_private is True


class TestModifierDecoratorNewline:
    """`builtin` and `private` act as decorators: a newline may follow them.

    The modifier and the declaration it adorns may sit on the same line or on
    consecutive lines; the newline after the modifier is insignificant.
    """

    def test_builtin_enum_separate_line(self) -> None:
        prog = parse("builtin\nenum Option[T]\n  | Some(elem: T)\n  | None")
        (decl,) = items(prog)
        assert isinstance(decl, EnumDef)
        assert decl.is_builtin is True
        assert decl.name == "Option"

    def test_builtin_record_separate_line(self) -> None:
        prog = parse("builtin\nrecord Point\n  x: int\n  y: int")
        (decl,) = items(prog)
        assert isinstance(decl, RecordDef)
        assert decl.is_builtin is True
        assert decl.name == "Point"

    def test_builtin_exception_separate_line(self) -> None:
        prog = parse("builtin\nexception Boom\n  reason: text")
        (decl,) = items(prog)
        assert isinstance(decl, syntax.ExceptionDef)
        assert decl.is_builtin is True
        assert decl.name == "Boom"

    def test_builtin_func_separate_line(self) -> None:
        prog = parse("builtin\ndef identity[T](x: T) -> T")
        (decl,) = items(prog)
        assert isinstance(decl, FuncDef)
        assert decl.is_builtin is True
        assert decl.name == "identity"

    def test_private_enum_separate_line(self) -> None:
        prog = parse("private\nenum Color | Red | Green | Blue")
        (decl,) = items(prog)
        assert isinstance(decl, EnumDef)
        assert decl.is_private is True
        assert decl.name == "Color"

    def test_private_record_separate_line(self) -> None:
        prog = parse("private\nrecord Foo\n    bar: text")
        (decl,) = items(prog)
        assert isinstance(decl, RecordDef)
        assert decl.is_private is True
        assert decl.name == "Foo"

    def test_private_func_separate_line(self) -> None:
        prog = parse('private\ndef f() -> text = "hi"')
        (decl,) = items(prog)
        assert isinstance(decl, FuncDef)
        assert decl.is_private is True
        assert decl.name == "f"

    def test_private_type_alias_separate_line(self) -> None:
        prog = parse("private\ntype Alias = text")
        (decl,) = items(prog)
        assert isinstance(decl, TypeAlias)
        assert decl.is_private is True
        assert decl.name == "Alias"

    def test_modifier_same_line_still_parses(self) -> None:
        """The same-line form remains valid (newline is optional, not required)."""
        prog = parse("builtin enum Option[T] | Some(elem: T) | None")
        (decl,) = items(prog)
        assert isinstance(decl, EnumDef)
        assert decl.is_builtin is True

    def test_decorator_decl_among_other_items(self) -> None:
        """A decorator-style declaration coexists with surrounding items."""
        prog = parse('def pub() -> text = "a"\nprivate\nenum E | A | B')
        it = items(prog)
        assert isinstance(it[0], FuncDef)
        assert isinstance(it[1], EnumDef)
        assert it[1].is_private is True
