"""Tests for the AgL evaluator (Component 6) via WorkflowRuntime.run().

All assertions are on user-visible RunResult attributes: .ok, .diagnostics,
.error, .bindings (root scope snapshot), and process stdout (capsys).

Agent calls use registered Python stub agents — no subprocess.

NOTE: Tests are scoped to what the M1 parser and typecheck support:
- Statements: input, let, var, set, pass, print, expr_stmt
- Expressions: var refs, scalar literals, templates, agent calls
- Types: text (default), int, decimal, bool, json
- Templates with default and raw renderers
"""

from __future__ import annotations

import decimal
import itertools

import pytest

from agm.agl import WorkflowRuntime
from agm.agl.eval.scope import Scope
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.request import AgentRequest
from agm.agl.runtime.runtime import RunResult
from agm.agl.syntax import nodes as ast
from agm.agl.syntax import types as tast
from agm.agl.syntax.nodes import (
    BinOp,
    Expr,
    IfBranch,
    NamedArg,
    Pattern,
    PatternField,
    Stmt,
    TemplateSegment,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import CheckedProgram

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(source: str, *, inputs: dict[str, object] | None = None) -> RunResult:
    """Build a WorkflowRuntime with no named agents and run *source*."""
    rt = WorkflowRuntime()
    return rt.run(source, inputs=inputs or {})


def run_with_default_agent(
    source: str,
    fn: AgentFn,
    *,
    inputs: dict[str, object] | None = None,
) -> RunResult:
    """Build a WorkflowRuntime with a default agent and run *source*."""
    rt = WorkflowRuntime(default_agent=fn)
    return rt.run(source, inputs=inputs or {})


def run_with_agents(
    source: str,
    agents: dict[str, AgentFn],
    *,
    inputs: dict[str, object] | None = None,
) -> RunResult:
    """Build a WorkflowRuntime, register agents, and run *source*."""
    default = agents.get("prompt")
    others = {k: v for k, v in agents.items() if k != "prompt"}
    rt = WorkflowRuntime(default_agent=default)
    for name, fn in others.items():
        rt.register_agent(name, fn)
    return rt.run(source, inputs=inputs or {})


# ---------------------------------------------------------------------------
# AST builders + a public-contract execution driver
#
# These let tests pin M3+ constructs that the M1 parser cannot yet parse by
# hand-building a real ``syntax.Program``, running it through the REAL
# ``resolve`` + ``check`` passes, and executing it via ``Interpreter.execute``.
# Assertions are on user-visible outcomes: bound values in the (public) root
# ``Scope`` snapshot, ``print`` output via capsys, and raised ``AglRaise`` /
# ``RuntimeError`` surfaced by ``execute``.
#
# Branch-scoped statements (if / case / try / do) are observed by declaring a
# mutable ``var`` at root scope and ``set``-ing it inside the branch, then
# reading the root binding back — exactly the user-visible workflow.
# ---------------------------------------------------------------------------

_node_ids = itertools.count(1)


def _nid() -> int:
    """Return a fresh, program-unique node id (side tables key on these)."""
    return next(_node_ids)


def _sp() -> SourceSpan:
    return SourceSpan(1, 1, 1, 5, 0, 4)


# --- expression builders ---------------------------------------------------


def _int(value: int) -> ast.IntLit:
    return ast.IntLit(value=value, span=_sp(), node_id=_nid())


def _dec(value: str) -> ast.DecimalLit:
    return ast.DecimalLit(value=decimal.Decimal(value), span=_sp(), node_id=_nid())


def _bool(value: bool) -> ast.BoolLit:
    return ast.BoolLit(value=value, span=_sp(), node_id=_nid())


def _null() -> ast.NullLit:
    return ast.NullLit(span=_sp(), node_id=_nid())


def _str(value: str) -> ast.StringLit:
    return ast.StringLit(value=value, span=_sp(), node_id=_nid())


def _ref(name: str) -> ast.VarRef:
    return ast.VarRef(name=name, span=_sp(), node_id=_nid())


def _field(obj: Expr, name: str) -> ast.FieldAccess:
    return ast.FieldAccess(obj=obj, field=name, span=_sp(), node_id=_nid())


def _binop(op: BinOp, left: Expr, right: Expr) -> ast.BinaryOp:
    return ast.BinaryOp(op=op, left=left, right=right, span=_sp(), node_id=_nid())


def _unary_not(operand: Expr) -> ast.UnaryNot:
    return ast.UnaryNot(operand=operand, span=_sp(), node_id=_nid())


def _unary_neg(operand: Expr) -> ast.UnaryNeg:
    return ast.UnaryNeg(operand=operand, span=_sp(), node_id=_nid())


def _is_test(
    expr: Expr, variant: str, *, qualifier: str | None = None, negated: bool = False
) -> ast.IsTest:
    return ast.IsTest(
        expr=expr, qualifier=qualifier, variant=variant, negated=negated,
        span=_sp(), node_id=_nid(),
    )


def _ctor(
    name: str, *, qualifier: str | None = None, args: tuple[NamedArg, ...] = ()
) -> ast.Constructor:
    return ast.Constructor(
        qualifier=qualifier, name=name, args=tuple(args), span=_sp(), node_id=_nid()
    )


def _arg(name: str, value: Expr) -> ast.NamedArg:
    return ast.NamedArg(name=name, value=value, span=_sp(), node_id=_nid())


def _list(*elements: Expr) -> ast.ListLit:
    return ast.ListLit(elements=tuple(elements), span=_sp(), node_id=_nid())


def _dict(**entries: Expr) -> ast.DictLit:
    items = tuple(
        ast.DictEntry(key=_str(k), value=v, span=_sp(), node_id=_nid())
        for k, v in entries.items()
    )
    return ast.DictLit(entries=items, span=_sp(), node_id=_nid())


def _template(*segments: TemplateSegment) -> ast.Template:
    return ast.Template(segments=tuple(segments), span=_sp(), node_id=_nid())


def _text_seg(text: str) -> ast.TextSegment:
    return ast.TextSegment(text=text, span=_sp(), node_id=_nid())


def _interp_seg(expr: Expr, *, render: str | None = None) -> ast.InterpSegment:
    return ast.InterpSegment(expr=expr, render=render, span=_sp(), node_id=_nid())


def _case_expr(subject: Expr, *branches: ast.CaseExprBranch) -> ast.CaseExpr:
    return ast.CaseExpr(subject=subject, branches=tuple(branches), span=_sp(), node_id=_nid())


def _case_expr_branch(pattern: Pattern, body: Expr) -> ast.CaseExprBranch:
    return ast.CaseExprBranch(pattern=pattern, body=body, span=_sp(), node_id=_nid())


# --- pattern builders ------------------------------------------------------


def _wild() -> ast.WildcardPattern:
    return ast.WildcardPattern(span=_sp(), node_id=_nid())


def _var_pat(name: str) -> ast.VarPattern:
    return ast.VarPattern(name=name, span=_sp(), node_id=_nid())


def _lit_pat(
    literal: ast.IntLit | ast.DecimalLit | ast.BoolLit | ast.StringLit | ast.NullLit,
) -> ast.LiteralPattern:
    return ast.LiteralPattern(literal=literal, span=_sp(), node_id=_nid())


def _ctor_pat(
    name: str, *, qualifier: str | None = None, fields: tuple[PatternField, ...] = ()
) -> ast.ConstructorPattern:
    return ast.ConstructorPattern(
        qualifier=qualifier, name=name, fields=tuple(fields), span=_sp(), node_id=_nid()
    )


def _pat_field(name: str, pattern: Pattern) -> ast.PatternField:
    return ast.PatternField(name=name, pattern=pattern, span=_sp(), node_id=_nid())


# --- statement builders ----------------------------------------------------


def _let(name: str, value: Expr, *, type_ann: tast.TypeExpr | None = None) -> ast.LetDecl:
    return ast.LetDecl(name=name, type_ann=type_ann, value=value, span=_sp(), node_id=_nid())


def _var(name: str, value: Expr, *, type_ann: tast.TypeExpr | None = None) -> ast.VarDecl:
    return ast.VarDecl(name=name, type_ann=type_ann, value=value, span=_sp(), node_id=_nid())


def _set(target: str, value: Expr) -> ast.SetStmt:
    return ast.SetStmt(target=target, value=value, span=_sp(), node_id=_nid())


def _pass() -> ast.PassStmt:
    return ast.PassStmt(span=_sp(), node_id=_nid())


def _print(value: Expr) -> ast.PrintStmt:
    return ast.PrintStmt(value=value, span=_sp(), node_id=_nid())


def _expr_stmt(expr: Expr) -> ast.ExprStmt:
    return ast.ExprStmt(expr=expr, span=_sp(), node_id=_nid())


def _do_until(
    condition: Expr, body: tuple[Stmt, ...], *, limit: int | None = None
) -> ast.DoUntil:
    return ast.DoUntil(
        limit=limit, body=tuple(body), condition=condition, span=_sp(), node_id=_nid()
    )


def _if(*branches: IfBranch) -> ast.IfStmt:
    return ast.IfStmt(branches=tuple(branches), span=_sp(), node_id=_nid())


def _if_branch(cond: Expr, body: tuple[Stmt, ...]) -> ast.IfBranch:
    return ast.IfBranch(cond=cond, body=tuple(body), span=_sp(), node_id=_nid())


def _else_branch(body: tuple[Stmt, ...]) -> ast.IfBranch:
    return ast.IfBranch(cond=ast.ELSE, body=tuple(body), span=_sp(), node_id=_nid())


def _case_stmt(subject: Expr, *branches: ast.CaseStmtBranch) -> ast.CaseStmt:
    return ast.CaseStmt(subject=subject, branches=tuple(branches), span=_sp(), node_id=_nid())


def _case_stmt_branch(pattern: Pattern, body: tuple[Stmt, ...]) -> ast.CaseStmtBranch:
    return ast.CaseStmtBranch(pattern=pattern, body=tuple(body), span=_sp(), node_id=_nid())


def _try(body: tuple[Stmt, ...], *handlers: ast.CatchClause) -> ast.TryCatch:
    return ast.TryCatch(body=tuple(body), handlers=tuple(handlers), span=_sp(), node_id=_nid())


def _catch(
    body: tuple[Stmt, ...], *, exc_type: str | None = None, binding: str | None = None
) -> ast.CatchClause:
    return ast.CatchClause(
        exc_type=exc_type, binding=binding, body=tuple(body), span=_sp(), node_id=_nid()
    )


def _raise(exc: Expr) -> ast.Raise:
    return ast.Raise(exc=exc, span=_sp(), node_id=_nid())


def _input(name: str, *, annotation: tast.TypeExpr | None = None) -> ast.InputDecl:
    return ast.InputDecl(name=name, annotation=annotation, span=_sp(), node_id=_nid())


# --- type-declaration / annotation builders --------------------------------


def _field_def(name: str, type_expr: tast.TypeExpr) -> ast.FieldDef:
    return ast.FieldDef(name=name, type_expr=type_expr, span=_sp(), node_id=_nid())


def _record_def(name: str, *fields: ast.FieldDef) -> ast.RecordDef:
    return ast.RecordDef(name=name, fields=tuple(fields), span=_sp(), node_id=_nid())


def _variant_def(name: str, *fields: ast.FieldDef) -> ast.VariantDef:
    return ast.VariantDef(name=name, fields=tuple(fields), span=_sp(), node_id=_nid())


def _enum_def(name: str, *variants: ast.VariantDef) -> ast.EnumDef:
    return ast.EnumDef(name=name, variants=tuple(variants), span=_sp(), node_id=_nid())


def _type_alias(name: str, type_expr: tast.TypeExpr) -> ast.TypeAlias:
    return ast.TypeAlias(name=name, type_expr=type_expr, span=_sp(), node_id=_nid())


def _ty(kind: str) -> tast.TypeExpr:
    """Build a scalar type-annotation node by kind name."""
    if kind == "text":
        return tast.TextT(span=_sp(), node_id=_nid())
    if kind == "int":
        return tast.IntT(span=_sp(), node_id=_nid())
    if kind == "decimal":
        return tast.DecimalT(span=_sp(), node_id=_nid())
    if kind == "bool":
        return tast.BoolT(span=_sp(), node_id=_nid())
    if kind == "json":
        return tast.JsonT(span=_sp(), node_id=_nid())
    raise AssertionError(f"unknown scalar kind {kind!r}")


def _list_ty(elem: tast.TypeExpr) -> tast.ListT:
    return tast.ListT(elem=elem, span=_sp(), node_id=_nid())


# --- the public-contract execution driver ----------------------------------


def _check_program(body: tuple[Stmt, ...], *, has_fallback: bool = False) -> CheckedProgram:
    """Run *body* statements through the real resolve + check passes."""
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.scope import resolve
    from agm.agl.syntax.nodes import Program
    from agm.agl.typecheck import check

    program = Program(body=tuple(body), span=_sp(), node_id=_nid())
    resolved = resolve(program)
    caps = HostCapabilities(
        agent_names=frozenset(),
        has_fallback_agent=has_fallback,
        codec_kinds={"text": frozenset({"text"})},
        renderer_names=frozenset({"default", "raw"}),
    )
    return check(resolved, caps)


def _execute(
    body: tuple[Stmt, ...],
    *,
    default_agent: AgentFn | None = None,
    named: dict[str, AgentFn] | None = None,
    has_fallback: bool = False,
) -> Scope:
    """Build + resolve + check + execute *body*, returning the root ``Scope``.

    Mirrors ``WorkflowRuntime.run`` for constructs the M1 parser cannot parse
    yet: drives the program through the real static passes and the public
    ``Interpreter.execute`` entry point.  Raised ``AglRaise`` / ``RuntimeError``
    propagate to the caller (the user-visible failure surface).
    """
    from agm.agl.eval.interpreter import Interpreter
    from agm.agl.runtime.agents import AgentRegistry

    checked = _check_program(body, has_fallback=has_fallback or default_agent is not None)
    registry = AgentRegistry(named=named or {}, default_agent=default_agent)
    interp = Interpreter(
        checked=checked,
        registry=registry,
        contracts={},
        type_env=checked.type_env,
        loop_limit=3,
        strict_json=False,
    )
    root = Scope(parent=None)
    interp.execute(root)
    return root


def _eval_value(expr: Expr, *, prelude: tuple[Stmt, ...] = ()) -> object:
    """Bind ``let r = expr`` (after *prelude*) and return r's runtime value."""
    body = (*prelude, _let("r", expr))
    return _execute(body).snapshot()["r"]


# ---------------------------------------------------------------------------
# Basic ok semantics
# ---------------------------------------------------------------------------


class TestOkSemantics:
    def test_empty_pass_ok(self) -> None:
        result = run("pass")
        assert result.ok is True
        assert result.error is None
        assert result.diagnostics == []

    def test_let_binding_ok(self) -> None:
        result = run("let x = 1")
        assert result.ok is True

    def test_static_error_not_ok(self) -> None:
        result = run("let x = undefined_var")
        assert result.ok is False
        assert result.error is None
        assert result.diagnostics  # at least one error diagnostic

    def test_static_error_has_line(self) -> None:
        result = run("let x = undefined_var")
        assert result.diagnostics[0].line >= 1

    def test_static_error_has_message(self) -> None:
        result = run("let x = undefined_var")
        assert result.diagnostics[0].message

    def test_bindings_present_on_ok(self) -> None:
        result = run("let x = 1")
        assert hasattr(result, "bindings")
        assert isinstance(result.bindings, dict)

    def test_bindings_empty_on_failure(self) -> None:
        result = run("let x = undefined_var")
        assert result.bindings == {}


# ---------------------------------------------------------------------------
# RunResult.bindings — root scope snapshot
# ---------------------------------------------------------------------------


class TestRootBindings:
    def test_let_binding_visible_in_bindings(self) -> None:
        result = run("let x = 42")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["x"] == IntValue(42)

    def test_var_binding_visible_in_bindings(self) -> None:
        result = run('var msg = "hello"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["msg"] == TextValue("hello")

    def test_set_updates_binding(self) -> None:
        result = run('var x: text = "first"\nset x = "second"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("second")

    def test_multiple_bindings(self) -> None:
        result = run("let a = 1\nlet b = 2\nlet c = 3")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["a"] == IntValue(1)
        assert result.bindings["b"] == IntValue(2)
        assert result.bindings["c"] == IntValue(3)


# ---------------------------------------------------------------------------
# Literal evaluation
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_int_literal(self) -> None:
        result = run("let x = 7")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["x"] == IntValue(7)

    def test_decimal_literal(self) -> None:
        result = run("let x = 1.5")
        assert result.ok
        from agm.agl.eval.values import DecimalValue

        assert result.bindings["x"] == DecimalValue(decimal.Decimal("1.5"))

    def test_bool_true(self) -> None:
        result = run("let x = true")
        assert result.ok
        from agm.agl.eval.values import BoolValue

        assert result.bindings["x"] == BoolValue(True)

    def test_bool_false(self) -> None:
        result = run("let x = false")
        assert result.ok
        from agm.agl.eval.values import BoolValue

        assert result.bindings["x"] == BoolValue(False)

    def test_string_literal(self) -> None:
        result = run('let x = "hello world"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("hello world")

    def test_null_literal(self) -> None:
        result = run("let j: json = null")
        assert result.ok
        from agm.agl.eval.values import JsonValue

        assert result.bindings["j"] == JsonValue(None)

    def test_large_int(self) -> None:
        bignum = 123456789012345678901234567890
        result = run(f"let x = {bignum}")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["x"] == IntValue(bignum)


# ---------------------------------------------------------------------------
# Template evaluation
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_template_no_interp(self) -> None:
        result = run('let x = "plain text"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("plain text")

    def test_template_with_int_interp_raw(self) -> None:
        # int interpolation as raw: no boundary marker
        result = run("let n = 5\nlet msg = \"n is ${n as raw}\"")
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["msg"] == TextValue("n is 5")

    def test_template_with_text_interp_raw(self) -> None:
        result = run('let s = "hello"\nlet msg = "say ${s as raw}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["msg"] == TextValue("say hello")

    def test_template_default_text_uses_boundary_markers(self) -> None:
        result = run('let s = "abc"\nlet msg = "x: ${s}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        # Default rendering for text includes boundary markers
        assert "<dsl-value" in v.value
        assert "abc" in v.value
        assert "</dsl-value>" in v.value

    def test_template_with_bool_interp_raw(self) -> None:
        result = run('let b = true\nlet msg = "${b as raw}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        assert "true" == v.value

    def test_template_escape_newline(self) -> None:
        result = run('let x = "line1\\nline2"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("line1\nline2")

    def test_template_int_default_is_scalar(self) -> None:
        # int default rendering is scalar (no boundary markers)
        result = run('let n = 42\nlet msg = "value: ${n}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        assert "42" in v.value
        assert "<dsl-value" not in v.value

    def test_template_bool_default_is_scalar(self) -> None:
        result = run('let b = false\nlet msg = "${b}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        assert "false" in v.value
        assert "<dsl-value" not in v.value


# ---------------------------------------------------------------------------
# Print console rendering
# ---------------------------------------------------------------------------


class TestPrintRendering:
    def test_print_text_verbatim(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run('print "hello world"')
        assert result.ok
        out = capsys.readouterr().out
        assert out == "hello world\n"

    def test_print_int_scalar(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print 42")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "42\n"

    def test_print_decimal_scalar(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print 2.5")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "2.5\n"

    def test_print_bool_true(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print true")
        assert result.ok
        out = capsys.readouterr().out
        assert "true" in out  # lowercase
        assert "True" not in out  # NOT Python repr

    def test_print_bool_false(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print false")
        assert result.ok
        out = capsys.readouterr().out
        assert "false" in out
        assert "False" not in out

    def test_print_no_boundary_markers(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run('let s = "hello"\nprint s')
        assert result.ok
        out = capsys.readouterr().out
        assert "<dsl-value" not in out
        assert out == "hello\n"

    def test_print_json_null(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let j: json = null\nprint j")
        assert result.ok
        out = capsys.readouterr().out
        assert "null" in out

    def test_print_multiline_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run('print "two\\nlines"')
        assert result.ok
        out = capsys.readouterr().out
        assert out == "two\nlines\n"

    def test_print_large_int(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print 123456789012345678901234567890")
        assert result.ok
        out = capsys.readouterr().out
        assert "123456789012345678901234567890" in out
        assert "e+" not in out.lower()

    def test_print_var_ref(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let x = 99\nprint x")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "99\n"


# ---------------------------------------------------------------------------
# Agent call evaluation (text codec)
# ---------------------------------------------------------------------------


class TestAgentCalls:
    def test_prompt_call_binds_response(self) -> None:
        def agent(req: AgentRequest) -> str:
            return "response text"

        result = run_with_default_agent('let x = prompt "Hello"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("response text")

    def test_named_agent_call(self) -> None:
        def impl(req: AgentRequest) -> str:
            return "output"

        rt = WorkflowRuntime()
        rt.register_agent("impl", impl)
        result = rt.run('let x = impl "Do something"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("output")

    def test_agent_receives_rendered_prompt_raw(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let name = "world"\nlet x = prompt "Hello ${name as raw}"', agent)
        assert len(prompts) == 1
        assert prompts[0] == "Hello world"

    def test_agent_prompt_contains_boundary_markers_for_text(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let artifact = "content"\nlet x = prompt "see ${artifact}"', agent)
        assert len(prompts) == 1
        assert "<dsl-value" in prompts[0]
        assert "content" in prompts[0]
        assert "</dsl-value>" in prompts[0]

    def test_agent_receives_request_with_agent_name(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('let x = reviewer "Review this."')
        assert len(received) == 1
        assert received[0].agent == "reviewer"

    def test_empty_response_is_valid_for_text_target(self) -> None:
        def agent(req: AgentRequest) -> str:
            return ""

        result = run_with_default_agent('let x = prompt "Say nothing."', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("")

    def test_no_default_agent_without_registration_fails_statically(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let x = prompt "Hi"')
        # No default agent, no fallback → static capability error
        assert result.ok is False
        assert result.error is None

    def test_agent_response_object_accepted(self) -> None:
        from agm.agl.runtime.request import AgentResponse

        def agent(req: AgentRequest) -> AgentResponse:
            return AgentResponse(content="from object")

        result = run_with_default_agent('let x = prompt "Hi"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("from object")

    def test_expr_stmt_call_result_discarded(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        result = run_with_default_agent('prompt "Note something."', agent)
        assert result.ok
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Input validation (§11.3, §9.5)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_declared_input_fails(self) -> None:
        result = run("input name\nprint name", inputs={})
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "name" in msgs.lower()

    def test_undeclared_extra_input_fails(self) -> None:
        result = run("input name\nprint name", inputs={"name": "bob", "bogus": "x"})
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "bogus" in msgs.lower()

    def test_text_input_taken_verbatim(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input name\nprint name", inputs={"name": "alice"})
        assert result.ok
        out = capsys.readouterr().out
        assert "alice" in out

    def test_int_input_from_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input n: int\nprint n", inputs={"n": 42})
        assert result.ok
        out = capsys.readouterr().out
        assert "42" in out

    def test_bool_input_from_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input flag: bool\nprint flag", inputs={"flag": True})
        assert result.ok
        out = capsys.readouterr().out
        assert "true" in out

    def test_json_input_from_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input meta: json\nprint meta", inputs={"meta": {"key": 1}})
        assert result.ok
        out = capsys.readouterr().out
        assert "key" in out

    def test_type_invalid_int_input_fails(self) -> None:
        result = run("input n: int\nprint n", inputs={"n": "not a number"})
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "n" in msgs.lower()

    def test_input_bound_immutably(self) -> None:
        # set on an input binding is a static error (scope pass)
        result = run("input x\nset x = \"y\"", inputs={"x": "hello"})
        assert result.ok is False

    def test_no_agent_called_on_input_failure(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('input name\nlet x = prompt "Hi ${name as raw}"', inputs={})
        assert calls == []

    def test_input_used_in_template(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent(
            'input name\nlet x = prompt "Hello ${name as raw}"',
            agent,
            inputs={"name": "Alice"},
        )
        assert len(prompts) == 1
        assert "Alice" in prompts[0]

    def test_decimal_input_from_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input r: decimal\nprint r", inputs={"r": decimal.Decimal("2.5")})
        assert result.ok
        out = capsys.readouterr().out
        assert "2.5" in out


# ---------------------------------------------------------------------------
# Boundary-marked rendering (§2.12)
# ---------------------------------------------------------------------------


class TestBoundaryRendering:
    def test_text_interpolation_default_has_dsl_value_tag(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let artifact = "hello"\nlet x = prompt "see ${artifact}"', agent)
        assert '<dsl-value name="artifact" type="text">' in prompts[0]
        assert "hello" in prompts[0]
        assert "</dsl-value>" in prompts[0]

    def test_raw_renderer_bypasses_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let s = "raw content"\nlet x = prompt "${s as raw}"', agent)
        assert prompts[0] == "raw content"
        assert "<dsl-value" not in prompts[0]

    def test_int_interp_default_is_scalar_no_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let n = 5\nlet x = prompt "n=${n}"', agent)
        assert "5" in prompts[0]
        assert "<dsl-value" not in prompts[0]

    def test_bool_interp_default_is_scalar_no_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let b = true\nlet x = prompt "b=${b}"', agent)
        assert "true" in prompts[0]
        assert "<dsl-value" not in prompts[0]

    def test_text_interp_name_attribute_is_varname(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent(
            'let my_artifact = "content"\nlet x = prompt "${my_artifact}"', agent
        )
        assert 'name="my_artifact"' in prompts[0]

    def test_null_interp_boundary_marked(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let j: json = null\nlet x = prompt "data: ${j}"', agent)
        assert "<dsl-value" in prompts[0]
        assert "null" in prompts[0]


# ---------------------------------------------------------------------------
# AgentRequest fields
# ---------------------------------------------------------------------------


class TestAgentRequest:
    def test_request_has_prompt(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        run_with_default_agent('let x = prompt "Hello world"', agent)
        assert received[0].prompt == "Hello world"

    def test_request_has_agent_name_prompt(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        run_with_default_agent('let x = prompt "Hi"', agent)
        assert received[0].agent == "prompt"

    def test_request_has_agent_name_custom(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('let x = reviewer "Review this"')
        assert received[0].agent == "reviewer"


# ---------------------------------------------------------------------------
# var / set statements
# ---------------------------------------------------------------------------


class TestVarSet:
    def test_var_initial_value(self) -> None:
        result = run('var x: text = "initial"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("initial")

    def test_set_updates_var(self) -> None:
        result = run('var x: text = "a"\nset x = "b"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("b")

    def test_set_on_let_is_static_error(self) -> None:
        result = run('let x = "a"\nset x = "b"')
        assert result.ok is False

    def test_set_undeclared_is_static_error(self) -> None:
        result = run('set x = "value"')
        assert result.ok is False

    def test_var_from_agent_response(self) -> None:
        def agent(req: AgentRequest) -> str:
            return "from agent"

        result = run_with_default_agent('var x: text = prompt "Get value"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("from agent")

    def test_set_from_agent_response(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append("call")
            return "v2"

        result = run_with_default_agent(
            'var x: text = prompt "First"\nset x = prompt "Second"', agent
        )
        assert result.ok
        assert len(calls) == 2
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("v2")


# ---------------------------------------------------------------------------
# Pass statement
# ---------------------------------------------------------------------------


class TestPassStmt:
    def test_pass_is_noop(self) -> None:
        result = run("pass")
        assert result.ok
        assert result.bindings == {}

    def test_pass_with_bindings(self) -> None:
        result = run("let x = 1\npass\nlet y = 2")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["x"] == IntValue(1)
        assert result.bindings["y"] == IntValue(2)


# ---------------------------------------------------------------------------
# Multiple agent calls and response chaining
# ---------------------------------------------------------------------------


class TestMultipleAgentCalls:
    def test_two_sequential_calls(self) -> None:
        responses = ["v1", "v2"]
        idx = [0]

        def agent(req: AgentRequest) -> str:
            r = responses[idx[0]]
            idx[0] += 1
            return r

        result = run_with_default_agent(
            'let a = prompt "First"\nlet b = prompt "Second"', agent
        )
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["a"] == TextValue("v1")
        assert result.bindings["b"] == TextValue("v2")

    def test_chaining_response_into_next_prompt(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            if len(calls) == 1:
                return "first-output"
            return "second-output"

        result = run_with_default_agent(
            'let a = prompt "First"\nlet b = prompt "Use ${a as raw}"', agent
        )
        assert result.ok
        assert "first-output" in calls[1]

    def test_named_agent_and_default_agent(self) -> None:
        default_calls: list[str] = []
        impl_calls: list[str] = []

        def default_agent(req: AgentRequest) -> str:
            default_calls.append(req.prompt)
            return "prompt-response"

        def impl(req: AgentRequest) -> str:
            impl_calls.append(req.prompt)
            return "impl-response"

        result = run_with_agents(
            'let a = prompt "Hello"\nlet b = impl "Build"',
            {"prompt": default_agent, "impl": impl},
        )
        assert result.ok
        assert len(default_calls) == 1
        assert len(impl_calls) == 1
        from agm.agl.eval.values import TextValue

        assert result.bindings["a"] == TextValue("prompt-response")
        assert result.bindings["b"] == TextValue("impl-response")


# ---------------------------------------------------------------------------
# Type coercion (int → decimal)
# ---------------------------------------------------------------------------


class TestTypeCoercion:
    def test_int_to_decimal_annotation(self) -> None:
        result = run("let x: decimal = 3")
        assert result.ok
        from agm.agl.eval.values import DecimalValue

        assert result.bindings["x"] == DecimalValue(decimal.Decimal(3))

    def test_decimal_literal_stays_decimal(self) -> None:
        result = run("let x: decimal = 1.5")
        assert result.ok
        from agm.agl.eval.values import DecimalValue

        assert result.bindings["x"] == DecimalValue(decimal.Decimal("1.5"))


# ---------------------------------------------------------------------------
# Print with var refs
# ---------------------------------------------------------------------------


class TestPrintVarRef:
    def test_print_int_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let n = 42\nprint n")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "42\n"

    def test_print_text_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run('let s = "hello"\nprint s')
        assert result.ok
        out = capsys.readouterr().out
        assert out == "hello\n"

    def test_print_bool_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let b = true\nprint b")
        assert result.ok
        out = capsys.readouterr().out
        assert "true" in out

    def test_print_decimal_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let d = 3.14\nprint d")
        assert result.ok
        out = capsys.readouterr().out
        assert "3.14" in out

    def test_print_input_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input msg\nprint msg", inputs={"msg": "from input"})
        assert result.ok
        out = capsys.readouterr().out
        assert out == "from input\n"

    def test_print_input_int(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input n: int\nprint n", inputs={"n": 7})
        assert result.ok
        out = capsys.readouterr().out
        assert out == "7\n"


# ---------------------------------------------------------------------------
# Value types module exports
# ---------------------------------------------------------------------------


class TestValueTypes:
    def test_text_value_equality(self) -> None:
        from agm.agl.eval.values import TextValue

        assert TextValue("a") == TextValue("a")
        assert TextValue("a") != TextValue("b")

    def test_int_value_equality(self) -> None:
        from agm.agl.eval.values import IntValue

        assert IntValue(1) == IntValue(1)
        assert IntValue(1) != IntValue(2)

    def test_decimal_value_equality(self) -> None:
        from agm.agl.eval.values import DecimalValue

        assert DecimalValue(decimal.Decimal("1.5")) == DecimalValue(decimal.Decimal("1.5"))

    def test_bool_value_equality(self) -> None:
        from agm.agl.eval.values import BoolValue

        assert BoolValue(True) == BoolValue(True)
        assert BoolValue(True) != BoolValue(False)

    def test_json_value_equality(self) -> None:
        from agm.agl.eval.values import JsonValue

        assert JsonValue(None) == JsonValue(None)
        assert JsonValue({"a": 1}) == JsonValue({"a": 1})


# ---------------------------------------------------------------------------
# M3+ Value types: hash/eq for DictValue, RecordValue, EnumValue, ExceptionValue
# ---------------------------------------------------------------------------


class TestM3ValueTypes:
    def test_dict_value_equality(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue

        d1 = DictValue(entries={"a": IntValue(1)})
        d2 = DictValue(entries={"a": IntValue(1)})
        assert d1 == d2

    def test_dict_value_inequality(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue

        d1 = DictValue(entries={"a": IntValue(1)})
        d2 = DictValue(entries={"a": IntValue(2)})
        assert d1 != d2

    def test_dict_value_not_equal_to_other_type(self) -> None:
        from agm.agl.eval.values import DictValue

        d = DictValue(entries={})
        assert d.__eq__("not a dict") is NotImplemented

    def test_dict_value_hashable(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue

        d = DictValue(entries={"x": IntValue(1)})
        # Hashable means it can be used as a dict key.
        mapping = {d: "hello"}
        assert mapping[d] == "hello"

    def test_record_value_equality(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue

        r1 = RecordValue(type_name="Point", fields={"x": IntValue(1), "y": IntValue(2)})
        r2 = RecordValue(type_name="Point", fields={"x": IntValue(1), "y": IntValue(2)})
        assert r1 == r2

    def test_record_value_inequality_type(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue

        r1 = RecordValue(type_name="Point", fields={"x": IntValue(1)})
        r2 = RecordValue(type_name="Line", fields={"x": IntValue(1)})
        assert r1 != r2

    def test_record_value_not_equal_to_other_type(self) -> None:
        from agm.agl.eval.values import RecordValue

        r = RecordValue(type_name="Point", fields={})
        assert r.__eq__("other") is NotImplemented

    def test_record_value_hashable(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue

        r = RecordValue(type_name="Point", fields={"x": IntValue(3)})
        h = hash(r)
        assert isinstance(h, int)

    def test_enum_value_equality(self) -> None:
        from agm.agl.eval.values import EnumValue, TextValue

        e1 = EnumValue(type_name="Color", variant="Red", fields={"label": TextValue("r")})
        e2 = EnumValue(type_name="Color", variant="Red", fields={"label": TextValue("r")})
        assert e1 == e2

    def test_enum_value_inequality_variant(self) -> None:
        from agm.agl.eval.values import EnumValue

        e1 = EnumValue(type_name="Color", variant="Red", fields={})
        e2 = EnumValue(type_name="Color", variant="Blue", fields={})
        assert e1 != e2

    def test_enum_value_not_equal_to_other_type(self) -> None:
        from agm.agl.eval.values import EnumValue

        e = EnumValue(type_name="Color", variant="Red", fields={})
        assert e.__eq__(42) is NotImplemented

    def test_enum_value_hashable(self) -> None:
        from agm.agl.eval.values import EnumValue

        e = EnumValue(type_name="Color", variant="Blue", fields={})
        h = hash(e)
        assert isinstance(h, int)

    def test_exception_value_equality(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue

        e1 = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("fatal"), "trace_id": TextValue("")}
        )
        e2 = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("fatal"), "trace_id": TextValue("")}
        )
        assert e1 == e2

    def test_exception_value_inequality(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue

        e1 = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("a"), "trace_id": TextValue("")}
        )
        e2 = ExceptionValue(
            type_name="Other", fields={"message": TextValue("a"), "trace_id": TextValue("")}
        )
        assert e1 != e2

    def test_exception_value_not_equal_to_other_type(self) -> None:
        from agm.agl.eval.values import ExceptionValue

        e = ExceptionValue(type_name="Abort", fields={})
        assert e.__eq__(None) is NotImplemented

    def test_exception_value_hashable(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue

        e = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("x"), "trace_id": TextValue("")}
        )
        h = hash(e)
        assert isinstance(h, int)


# ---------------------------------------------------------------------------
# AglRaise carrier
# ---------------------------------------------------------------------------


class TestAglRaise:
    def test_agl_raise_carries_exc_value(self) -> None:
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import ExceptionValue, TextValue

        exc_val = ExceptionValue(
            type_name="TestError",
            fields={"message": TextValue("oops"), "trace_id": TextValue("")},
        )
        carrier = AglRaise(exc_val)
        assert carrier.exc is exc_val
        assert str(carrier) == "TestError"

    def test_agl_raise_is_exception(self) -> None:
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import ExceptionValue, TextValue

        exc_val = ExceptionValue(
            type_name="E", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        carrier = AglRaise(exc_val)
        assert isinstance(carrier, Exception)


# ---------------------------------------------------------------------------
# Scope unit tests (parent chain set_value)
# ---------------------------------------------------------------------------


class TestScopeUnit:
    def test_set_value_in_parent_scope(self) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        parent = Scope(parent=None)
        parent.define("x", IntValue(1), mutable=True, decl_span=span)
        child = Scope(parent=parent)

        # Set in child scope: updates parent's binding.
        result = child.set_value("x", IntValue(99))
        assert result is True
        assert parent.bindings["x"].value == IntValue(99)

    def test_set_value_not_found_returns_false(self) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue

        scope = Scope(parent=None)
        result = scope.set_value("nonexistent", IntValue(5))
        assert result is False

    def test_lookup_from_parent_scope(self) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        parent = Scope(parent=None)
        parent.define("x", IntValue(42), mutable=False, decl_span=span)
        child = Scope(parent=parent)

        # Child scope has no binding for "x", must walk to parent.
        binding = child.lookup("x")
        assert binding is not None
        assert binding.value == IntValue(42)

    def test_lookup_not_found_returns_none(self) -> None:
        from agm.agl.eval.scope import Scope

        scope = Scope(parent=None)
        assert scope.lookup("missing") is None


# ---------------------------------------------------------------------------
# Codec unit tests
# ---------------------------------------------------------------------------


class TestCodecUnit:
    def test_text_codec_supports_text_type(self) -> None:
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.typecheck.types import IntType, TextType

        c = TextCodec()
        assert c.supports_type(TextType()) is True
        assert c.supports_type(IntType()) is False

    def test_parse_result_failure(self) -> None:
        from agm.agl.runtime.codec import ParseResult

        r = ParseResult.failure("bad format")
        assert r.ok is False
        assert r.value is None
        assert r.error_msg == "bad format"


# ---------------------------------------------------------------------------
# Render unit tests
# ---------------------------------------------------------------------------


class TestRenderUnit:
    def test_render_for_console_list_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import render_for_console

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        text = render_for_console(v)
        assert "1" in text
        assert "2" in text

    def test_render_for_console_dict_value(self) -> None:
        from agm.agl.eval.values import DictValue, TextValue
        from agm.agl.runtime.render import render_for_console

        v = DictValue(entries={"key": TextValue("val")})
        text = render_for_console(v)
        assert "key" in text
        assert "val" in text

    def test_render_for_console_record_value(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.render import render_for_console

        v = RecordValue(type_name="Point", fields={"x": IntValue(3)})
        text = render_for_console(v)
        assert "3" in text

    def test_render_for_console_enum_value(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.render import render_for_console

        v = EnumValue(type_name="Status", variant="Active", fields={})
        text = render_for_console(v)
        assert "Active" in text

    def test_render_for_console_exception_value(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.render import render_for_console

        v = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("fatal"), "trace_id": TextValue("")}
        )
        text = render_for_console(v)
        assert "fatal" in text

    def test_render_for_console_json_value_null(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import render_for_console

        v = JsonValue(None)
        text = render_for_console(v)
        assert "null" in text

    def test_value_to_json_obj_decimal(self) -> None:
        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = DecimalValue(decimal.Decimal("3.14"))
        result = _value_to_json_obj(v)
        assert isinstance(result, float)
        assert abs(float(result) - 3.14) < 0.001

    def test_value_to_json_obj_bool(self) -> None:
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.render import _value_to_json_obj

        assert _value_to_json_obj(BoolValue(True)) is True
        assert _value_to_json_obj(BoolValue(False)) is False

    def test_value_to_json_obj_json(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = JsonValue({"nested": [1, 2]})
        result = _value_to_json_obj(v)
        assert result == {"nested": [1, 2]}

    def test_value_to_json_obj_list(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        result = _value_to_json_obj(v)
        assert result == [1, 2]

    def test_value_to_json_obj_dict(self) -> None:
        from agm.agl.eval.values import DictValue, TextValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = DictValue(entries={"k": TextValue("v")})
        result = _value_to_json_obj(v)
        assert result == {"k": "v"}

    def test_value_to_json_obj_record(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = RecordValue(type_name="P", fields={"x": IntValue(5)})
        result = _value_to_json_obj(v)
        assert result == {"x": 5}

    def test_value_to_json_obj_enum(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = EnumValue(type_name="C", variant="Red", fields={})
        result = _value_to_json_obj(v)
        assert result == {"$case": "Red"}

    def test_value_to_json_obj_exception(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = ExceptionValue(
            type_name="E", fields={"message": TextValue("oops"), "trace_id": TextValue("")}
        )
        result = _value_to_json_obj(v)
        assert isinstance(result, dict)
        assert result.get("message") == "oops"

    def test_render_for_prompt_json_renderer(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_for_prompt

        v = IntValue(42)
        text = render_for_prompt(v, renderer_name="json", var_name=None)
        assert "42" in text

    def test_render_for_prompt_bullets_list(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import render_for_prompt

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        text = render_for_prompt(v, renderer_name="bullets", var_name=None)
        assert "- 1" in text
        assert "- 2" in text

    def test_render_for_prompt_bullets_non_list(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_for_prompt

        v = IntValue(5)
        text = render_for_prompt(v, renderer_name="bullets", var_name=None)
        assert "5" in text

    def test_render_for_prompt_unknown_renderer_falls_back(self) -> None:
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_prompt

        v = TextValue("hello")
        text = render_for_prompt(v, renderer_name="nonexistent", var_name="x")
        # Falls back to default: boundary-marked text
        assert "<dsl-value" in text

    def test_render_default_json_value(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import render_for_prompt

        v = JsonValue({"a": 1})
        text = render_for_prompt(v, renderer_name="default", var_name="data")
        assert "<dsl-value" in text
        assert '"a"' in text

    def test_type_kind_str_for_all_types(self) -> None:
        from agm.agl.eval.values import (
            BoolValue,
            DecimalValue,
            DictValue,
            EnumValue,
            ExceptionValue,
            IntValue,
            JsonValue,
            ListValue,
            RecordValue,
            TextValue,
        )
        from agm.agl.runtime.render import _type_kind_str

        assert _type_kind_str(TextValue("x")) == "text"
        assert _type_kind_str(IntValue(1)) == "int"
        assert _type_kind_str(DecimalValue(decimal.Decimal("1.5"))) == "decimal"
        assert _type_kind_str(BoolValue(True)) == "bool"
        assert _type_kind_str(JsonValue(None)) == "json"
        assert _type_kind_str(ListValue(elements=())) == "list"
        assert _type_kind_str(DictValue(entries={})) == "dict"
        assert _type_kind_str(RecordValue(type_name="P", fields={})) == "P"
        assert _type_kind_str(EnumValue(type_name="E", variant="V", fields={})) == "E"
        assert _type_kind_str(ExceptionValue(type_name="Ex", fields={})) == "Ex"

    def test_scalar_text_json_value(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import _scalar_text

        text = _scalar_text(JsonValue({"a": 1}))
        assert "a" in text

    def test_scalar_text_list_falls_back_to_pretty_json(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import _scalar_text

        v = ListValue(elements=(IntValue(1),))
        text = _scalar_text(v)
        assert "1" in text


# ---------------------------------------------------------------------------
# Contract materialization error
# ---------------------------------------------------------------------------


class TestContractError:
    def test_unknown_codec_raises_value_error(self) -> None:
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.typecheck.env import OutputContractSpec
        from agm.agl.typecheck.types import TextType

        spec = OutputContractSpec(
            codec_name="unknown_codec",
            target_type=TextType(),
            strict_json=None,
        )
        codecs = {"text": TextCodec()}
        with pytest.raises(ValueError, match="unknown_codec"):
            materialize_contract(spec, codecs)


# ---------------------------------------------------------------------------
# WorkflowRuntime: exception handlers and edge cases
# ---------------------------------------------------------------------------


class TestRuntimeExceptionHandlers:
    def test_generic_parse_exception_returns_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AglSyntaxError from parse_program → ok=False diagnostic."""
        import agm.agl.runtime.runtime as rt_mod

        def bad_parse(source: str) -> object:
            raise RuntimeError("unexpected parser crash")

        monkeypatch.setattr(rt_mod, "parse_program", bad_parse, raising=False)
        # Need to patch the import inside run()
        import agm.agl.parser as parser_mod

        monkeypatch.setattr(parser_mod, "parse_program", bad_parse)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert "unexpected parser crash" in result.diagnostics[0].message

    def test_generic_scope_exception_returns_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AglScopeError from resolve → ok=False diagnostic."""
        import agm.agl.scope as scope_mod


        def bad_resolve(program: object) -> object:
            raise RuntimeError("resolve crash")

        monkeypatch.setattr(scope_mod, "resolve", bad_resolve)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert "resolve crash" in result.diagnostics[0].message

    def test_generic_typecheck_exception_returns_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AglTypeError from check → ok=False diagnostic."""
        import agm.agl.typecheck as tc_mod

        def bad_check(resolved: object, caps: object) -> object:
            raise RuntimeError("typecheck crash")

        monkeypatch.setattr(tc_mod, "check", bad_check)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert "typecheck crash" in result.diagnostics[0].message

    def test_internal_interpreter_error_returns_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AglRaise from interpreter.execute → ok=False diagnostic."""
        from agm.agl.eval.interpreter import Interpreter

        def bad_execute(self: Interpreter, root_scope: object) -> None:
            raise RuntimeError("internal crash")

        monkeypatch.setattr(Interpreter, "execute", bad_execute)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert "internal crash" in result.diagnostics[0].message

    def test_exception_value_to_run_error_maps_all_field_kinds(self) -> None:
        """_exception_value_to_run_error converts every Value kind to JSON shape.

        This is the pure converter used to surface an uncaught AgL exception
        (e.g. AgentParseError) as a RunError.
        """
        from agm.agl.eval.values import (
            BoolValue,
            DecimalValue,
            DictValue,
            EnumValue,
            ExceptionValue,
            IntValue,
            JsonValue,
            ListValue,
            RecordValue,
            TextValue,
        )
        from agm.agl.runtime.runtime import RunError, _exception_value_to_run_error

        exc_val = ExceptionValue(
            type_name="AgentParseError",
            fields={
                "message": TextValue("failed"),
                "trace_id": TextValue(""),
                "raw": TextValue("abc"),
                "agent": TextValue("prompt"),
                "attempts": IntValue(1),
                "target_type": TextValue("text"),
                "decimal_val": DecimalValue(decimal.Decimal("1.5")),
                "bool_val": BoolValue(True),
                "json_val": JsonValue({"k": "v"}),
                "list_val": ListValue(elements=(IntValue(1),)),
                "dict_val": DictValue(entries={"x": IntValue(2)}),
                "rec_val": RecordValue(type_name="R", fields={"f": TextValue("v")}),
                "enum_val": EnumValue(type_name="E", variant="V", fields={}),
                "exc_val": ExceptionValue(type_name="Inner", fields={}),
                "none_val": JsonValue(None),
            },
        )
        error = _exception_value_to_run_error(exc_val)
        assert isinstance(error, RunError)
        assert error.type_name == "AgentParseError"
        assert error.fields["message"] == "failed"
        assert isinstance(error.fields["decimal_val"], float)
        assert error.fields["bool_val"] is True
        assert error.fields["json_val"] == {"k": "v"}
        assert error.fields["list_val"] == [1]
        assert error.fields["dict_val"] == {"x": 2}
        assert error.fields["rec_val"] == {"f": "v"}
        assert error.fields["enum_val"] == {"$case": "V"}
        assert isinstance(error.fields["exc_val"], dict)

    def test_resolve_annotation_all_types(self) -> None:
        from agm.agl.runtime.runtime import _resolve_annotation
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import BoolT, DecimalT, IntT, JsonT, TextT
        from agm.agl.typecheck.types import BoolType, DecimalType, IntType, JsonType, TextType

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        assert isinstance(_resolve_annotation(None), TextType)
        assert isinstance(_resolve_annotation(TextT(span=span, node_id=0)), TextType)
        assert isinstance(_resolve_annotation(IntT(span=span, node_id=0)), IntType)
        assert isinstance(_resolve_annotation(DecimalT(span=span, node_id=0)), DecimalType)
        assert isinstance(_resolve_annotation(BoolT(span=span, node_id=0)), BoolType)
        assert isinstance(_resolve_annotation(JsonT(span=span, node_id=0)), JsonType)
        # Unknown type falls back to text
        assert isinstance(_resolve_annotation(object()), TextType)

    def test_convert_input_int_from_decimal_string(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import IntType

        # "1.0" parses as Decimal("1.0") which equals int(1) → IntValue
        result = _convert_input("n", "1.0", IntType())
        assert result == IntValue(1)

    def test_convert_input_invalid_json_raises(self) -> None:
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import IntType

        with pytest.raises(ValueError, match="JSON"):
            _convert_input("n", "not_json", IntType())

    def test_convert_input_decimal_from_int(self) -> None:
        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import DecimalType

        # Int passed as raw int → DecimalValue
        result = _convert_input("r", 5, DecimalType())
        assert isinstance(result, DecimalValue)

    def test_convert_input_decimal_invalid_raises(self) -> None:
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import DecimalType

        with pytest.raises(ValueError, match="decimal"):
            _convert_input("r", True, DecimalType())

    def test_convert_input_bool_invalid_raises(self) -> None:
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import BoolType

        # Pass an integer (valid JSON type, but not a bool).
        with pytest.raises(ValueError, match="bool"):
            _convert_input("b", 42, BoolType())

    def test_convert_input_json_type_accepts_any(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import JsonType

        result = _convert_input("meta", [1, 2, 3], JsonType())
        assert result == JsonValue([1, 2, 3])

    def test_convert_input_unknown_type_fallback(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.runtime import _convert_input

        # Passing an unknown type_obj uses the fallback JSON path.
        result = _convert_input("x", [1, 2], object())
        assert isinstance(result, JsonValue)

    def test_convert_input_text_non_str_raises(self) -> None:
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import TextType

        with pytest.raises(ValueError, match="text"):
            _convert_input("t", 42, TextType())

    def test_agl_raise_from_interpreter_becomes_run_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An AglRaise from the interpreter → RunResult.error (not None)."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.values import ExceptionValue, TextValue

        def bad_execute(self: Interpreter, root_scope: object) -> None:
            exc_val = ExceptionValue(
                type_name="Abort",
                fields={"message": TextValue("fatal"), "trace_id": TextValue("")},
            )
            raise AglRaise(exc_val)

        monkeypatch.setattr(Interpreter, "execute", bad_execute)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "Abort"
        assert result.error.fields.get("message") == "fatal"


# ---------------------------------------------------------------------------
# Interpreter unit tests (direct method calls for M3+ features)
# ---------------------------------------------------------------------------


class TestInterpreterUnit:
    """Unit tests for interpreter methods that are not reachable via M1 parser."""

    def test_make_exc_value_helper(self) -> None:
        from agm.agl.eval.interpreter import _make_exc_value
        from agm.agl.eval.values import ExceptionValue

        exc = _make_exc_value("TestError", "something went wrong")
        assert isinstance(exc, ExceptionValue)
        assert exc.type_name == "TestError"

    def test_coerce_non_decimal_unchanged(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import TextValue
        from agm.agl.typecheck.types import TextType

        v = TextValue("hello")
        result = _coerce(v, TextType())
        assert result is v

    def test_add_text_values(self) -> None:
        from agm.agl.eval.interpreter import _add
        from agm.agl.eval.values import TextValue

        result = _add(TextValue("hello "), TextValue("world"))
        assert result == TextValue("hello world")

    def test_add_int_and_decimal(self) -> None:
        from agm.agl.eval.interpreter import _add
        from agm.agl.eval.values import DecimalValue, IntValue

        result = _add(IntValue(1), DecimalValue(decimal.Decimal("0.5")))
        assert isinstance(result, DecimalValue)

    def test_add_type_error(self) -> None:
        from agm.agl.eval.interpreter import _add
        from agm.agl.eval.values import BoolValue, TextValue

        with pytest.raises(RuntimeError, match="Cannot add"):
            _add(TextValue("x"), BoolValue(True))

    def test_arith_subtraction_int(self) -> None:
        from agm.agl.eval.interpreter import _arith
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _arith(IntValue(5), IntValue(3), BinOp.SUB)
        assert result == IntValue(2)

    def test_arith_multiplication_decimal(self) -> None:
        from agm.agl.eval.interpreter import _arith
        from agm.agl.eval.values import DecimalValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _arith(IntValue(3), DecimalValue(decimal.Decimal("2.0")), BinOp.MUL)
        assert isinstance(result, DecimalValue)

    def test_arith_type_error(self) -> None:
        from agm.agl.eval.interpreter import _arith
        from agm.agl.eval.values import TextValue
        from agm.agl.syntax.nodes import BinOp

        with pytest.raises(RuntimeError, match="Cannot perform"):
            _arith(TextValue("a"), TextValue("b"), BinOp.SUB)

    def test_div_decimal(self) -> None:
        from agm.agl.eval.interpreter import _div
        from agm.agl.eval.values import DecimalValue, IntValue

        result = _div(IntValue(10), IntValue(4))
        assert isinstance(result, DecimalValue)

    def test_div_by_zero_raises_agl_raise(self) -> None:
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import _div
        from agm.agl.eval.values import IntValue

        with pytest.raises(AglRaise) as exc_info:
            _div(IntValue(5), IntValue(0))
        assert exc_info.value.exc.type_name == "ArithmeticError"

    def test_div_type_error(self) -> None:
        from agm.agl.eval.interpreter import _div
        from agm.agl.eval.values import TextValue

        with pytest.raises(RuntimeError, match="Cannot divide"):
            _div(TextValue("a"), TextValue("b"))

    def test_compare_eq_text(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, TextValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(TextValue("abc"), TextValue("abc"), BinOp.EQ)
        assert result == BoolValue(True)

    def test_compare_neq(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(IntValue(1), IntValue(2), BinOp.NEQ)
        assert result == BoolValue(True)

    def test_compare_int_widen_to_decimal(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(IntValue(1), DecimalValue(decimal.Decimal("1.0")), BinOp.EQ)
        assert result == BoolValue(True)

    def test_compare_decimal_widen_left(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(DecimalValue(decimal.Decimal("2.0")), IntValue(2), BinOp.EQ)
        assert result == BoolValue(True)

    def test_compare_ordering_decimal_lt(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("1.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.LT,
        )
        assert result == BoolValue(True)

    def test_compare_ordering_decimal_le(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("2.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.LE,
        )
        assert result == BoolValue(True)

    def test_compare_ordering_decimal_gt(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("3.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.GT,
        )
        assert result == BoolValue(True)

    def test_compare_ordering_decimal_ge(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("2.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.GE,
        )
        assert result == BoolValue(True)

    def test_compare_ordering_int_lt(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        assert _compare(IntValue(1), IntValue(2), BinOp.LT) == BoolValue(True)
        assert _compare(IntValue(2), IntValue(1), BinOp.LE) == BoolValue(False)
        assert _compare(IntValue(3), IntValue(2), BinOp.GT) == BoolValue(True)
        assert _compare(IntValue(2), IntValue(2), BinOp.GE) == BoolValue(True)

    def test_compare_ordering_text(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, TextValue
        from agm.agl.syntax.nodes import BinOp

        assert _compare(TextValue("a"), TextValue("b"), BinOp.LT) == BoolValue(True)
        assert _compare(TextValue("a"), TextValue("a"), BinOp.LE) == BoolValue(True)
        assert _compare(TextValue("b"), TextValue("a"), BinOp.GT) == BoolValue(True)
        assert _compare(TextValue("a"), TextValue("a"), BinOp.GE) == BoolValue(True)

    def test_compare_type_error(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, TextValue
        from agm.agl.syntax.nodes import BinOp

        with pytest.raises(RuntimeError, match="Cannot compare"):
            _compare(TextValue("x"), BoolValue(True), BinOp.LT)

    def test_in_op_list(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, IntValue, ListValue

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        assert _in_op(IntValue(1), v) == BoolValue(True)
        assert _in_op(IntValue(3), v) == BoolValue(False)

    def test_in_op_dict_key(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, DictValue, IntValue, TextValue

        v = DictValue(entries={"key": IntValue(1)})
        assert _in_op(TextValue("key"), v) == BoolValue(True)
        assert _in_op(TextValue("missing"), v) == BoolValue(False)

    def test_in_op_dict_non_text_key(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, DictValue, IntValue

        v = DictValue(entries={"key": IntValue(1)})
        assert _in_op(IntValue(1), v) == BoolValue(False)

    def test_in_op_text_substring(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, TextValue

        assert _in_op(TextValue("ell"), TextValue("hello")) == BoolValue(True)
        assert _in_op(TextValue("xyz"), TextValue("hello")) == BoolValue(False)

    def test_in_op_type_error(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import IntValue, TextValue

        with pytest.raises(RuntimeError, match="in"):
            _in_op(IntValue(1), TextValue("hello"))

    def test_case_wildcard_pattern_always_matches(self) -> None:
        """A ``_`` wildcard branch matches any subject (here an int)."""
        from agm.agl.eval.values import IntValue

        body = (
            _let("r", _case_expr(_int(42), _case_expr_branch(_wild(), _int(1)))),
        )
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_case_var_pattern_binds_subject(self) -> None:
        """A var pattern captures the subject into the branch body scope."""
        from agm.agl.eval.values import TextValue

        body = (
            _let("r", _case_expr(_str("hello"), _case_expr_branch(_var_pat("x"), _ref("x")))),
        )
        assert _execute(body).snapshot()["r"] == TextValue("hello")

    def test_case_literal_int_pattern_match_and_no_match(self) -> None:
        """An int literal pattern matches an equal subject and skips otherwise."""
        from agm.agl.eval.values import IntValue

        hit = (
            _let(
                "r",
                _case_expr(
                    _int(42),
                    _case_expr_branch(_lit_pat(_int(42)), _int(1)),
                    _case_expr_branch(_wild(), _int(0)),
                ),
            ),
        )
        assert _execute(hit).snapshot()["r"] == IntValue(1)

        miss = (
            _let(
                "r",
                _case_expr(
                    _int(42),
                    _case_expr_branch(_lit_pat(_int(99)), _int(1)),
                    _case_expr_branch(_wild(), _int(0)),
                ),
            ),
        )
        assert _execute(miss).snapshot()["r"] == IntValue(0)

    def test_case_literal_decimal_pattern_match(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _let(
                "r",
                _case_expr(
                    _dec("1.5"),
                    _case_expr_branch(_lit_pat(_dec("1.5")), _int(1)),
                    _case_expr_branch(_wild(), _int(0)),
                ),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_case_literal_bool_pattern_match(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _let(
                "r",
                _case_expr(
                    _bool(True),
                    _case_expr_branch(_lit_pat(_bool(True)), _int(1)),
                    _case_expr_branch(_wild(), _int(0)),
                ),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_case_literal_string_pattern_match(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _let(
                "r",
                _case_expr(
                    _str("hello"),
                    _case_expr_branch(_lit_pat(_str("hello")), _int(1)),
                    _case_expr_branch(_wild(), _int(0)),
                ),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_case_literal_null_pattern_match(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _let("j", _null(), type_ann=_ty("json")),
            _let(
                "r",
                _case_expr(
                    _ref("j"),
                    _case_expr_branch(_lit_pat(_null()), _int(1)),
                    _case_expr_branch(_wild(), _int(0)),
                ),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_matches_catch_bare_handler(self) -> None:
        from agm.agl.eval.interpreter import _matches_catch
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        handler = CatchClause(exc_type=None, binding=None, body=(), span=span, node_id=0)
        exc = ExceptionValue(
            type_name="Any", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        assert _matches_catch(handler, exc) is True

    def test_matches_catch_exception_base_type(self) -> None:
        from agm.agl.eval.interpreter import _matches_catch
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        handler = CatchClause(exc_type="Exception", binding=None, body=(), span=span, node_id=0)
        exc = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        assert _matches_catch(handler, exc) is True

    def test_matches_catch_exact_type(self) -> None:
        from agm.agl.eval.interpreter import _matches_catch
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        handler = CatchClause(exc_type="Abort", binding=None, body=(), span=span, node_id=0)
        exc = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        assert _matches_catch(handler, exc) is True

    def test_matches_catch_wrong_type(self) -> None:
        from agm.agl.eval.interpreter import _matches_catch
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        handler = CatchClause(exc_type="NetworkError", binding=None, body=(), span=span, node_id=0)
        exc = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        assert _matches_catch(handler, exc) is False

    def test_describe_value(self) -> None:
        from agm.agl.eval.interpreter import _describe_value
        from agm.agl.eval.values import (
            EnumValue,
            ExceptionValue,
            IntValue,
            RecordValue,
        )

        assert "Status" in _describe_value(
            EnumValue(type_name="Status", variant="Active", fields={})
        )
        assert "Point" in _describe_value(RecordValue(type_name="Point", fields={}))
        assert "Abort" in _describe_value(ExceptionValue(type_name="Abort", fields={}))
        assert "IntValue" in _describe_value(IntValue(1))

    def test_unary_not_on_non_bool_raises_at_runtime(self) -> None:
        """``not <int>`` passes typecheck but raises a runtime error.

        ``not`` accepts any operand statically, so the interpreter's runtime
        type guard is reachable through a real resolve+check-valid program.
        """
        body = (_let("a", _unary_not(_int(1))),)
        with pytest.raises(RuntimeError, match="not: expected bool"):
            _execute(body)


# ---------------------------------------------------------------------------
# Runtime: build_type_env, uncaught AglRaise via monkeypatched codec
# ---------------------------------------------------------------------------


class TestCheckedProgramTypeEnv:
    def test_checked_program_carries_type_env(self) -> None:
        """check() populates CheckedProgram.type_env with a TypeEnvironment."""
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.parser import parse_program
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check
        from agm.agl.typecheck.env import TypeEnvironment

        source = "pass"
        program = parse_program(source)
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=False,
            codec_kinds={},
            renderer_names=frozenset(),
        )
        checked = check(resolved, caps)
        assert isinstance(checked.type_env, TypeEnvironment)


class TestRuntimeContractError:
    def test_contract_error_returns_failure_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """materialize_contract raising ValueError → ok=False, diagnostics list."""
        import agm.agl.runtime.contract as contract_mod

        def bad_materialize(spec: object, codecs: object) -> object:
            raise ValueError("bad codec")

        monkeypatch.setattr(contract_mod, "materialize_contract", bad_materialize)

        def agent(req: AgentRequest) -> str:
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('let x = prompt "Hi"')
        assert result.ok is False
        assert result.error is None
        assert any("bad codec" in d.message for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Coverage: interpreter M3+ statement dispatch (_exec_stmt branches)
# ---------------------------------------------------------------------------


class TestInterpreterM3Stmts:
    """M3+ statements driven through resolve + check + execute.

    Branch bodies are observed by mutating a root-scope ``var`` and reading the
    public root binding back.
    """

    def test_do_until_runs_body_until_condition_true(self) -> None:
        """A ``do`` loop whose condition is true exits after running its body."""
        from agm.agl.eval.values import IntValue

        body = (
            _var("n", _int(0), type_ann=_ty("int")),
            _do_until(_bool(True), (_set("n", _int(1)),), limit=3),
        )
        assert _execute(body).snapshot()["n"] == IntValue(1)

    def test_do_until_exhausts_uses_runtime_loop_limit(self) -> None:
        """A ``do`` loop with no explicit limit and a false condition exhausts.

        Exhaustion raises ``MaxIterationsExceeded`` (an uncaught AgL exception).
        """
        from agm.agl.eval.exceptions import AglRaise

        body = (_do_until(_bool(False), (_pass(),)),)
        with pytest.raises(AglRaise) as exc_info:
            _execute(body)
        assert exc_info.value.exc.type_name == "MaxIterationsExceeded"

    def test_if_true_branch_executes(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _var("r", _int(0), type_ann=_ty("int")),
            _if(_if_branch(_bool(True), (_set("r", _int(1)),))),
        )
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_if_false_branch_skipped(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _var("r", _int(0), type_ann=_ty("int")),
            _if(_if_branch(_bool(False), (_set("r", _int(1)),))),
        )
        # Branch skipped → r keeps its initial value.
        assert _execute(body).snapshot()["r"] == IntValue(0)

    def test_if_else_branch_executes_when_no_prior_match(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _var("r", _int(0), type_ann=_ty("int")),
            _if(
                _if_branch(_bool(False), (_set("r", _int(1)),)),
                _else_branch((_set("r", _int(2)),)),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(2)

    def test_case_stmt_wildcard_branch_matches(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _var("r", _int(0), type_ann=_ty("int")),
            _case_stmt(_int(5), _case_stmt_branch(_wild(), (_set("r", _int(1)),))),
        )
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_case_stmt_no_match_raises_match_error(self) -> None:
        from agm.agl.eval.exceptions import AglRaise

        body = (
            _case_stmt(_int(5), _case_stmt_branch(_lit_pat(_int(99)), (_pass(),))),
        )
        with pytest.raises(AglRaise) as exc_info:
            _execute(body)
        assert exc_info.value.exc.type_name == "MatchError"

    def test_try_catch_no_exception_runs_body(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _var("r", _int(0), type_ann=_ty("int")),
            _try(
                (_set("r", _int(1)),),
                _catch((_set("r", _int(2)),)),
            ),
        )
        # Body ran without raising → handler skipped, r == 1.
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_try_catch_matching_handler_binds_exception(self) -> None:
        """A matching handler catches the AgL exception and binds its value."""
        from agm.agl.eval.values import TextValue

        # 1 / 0 raises ArithmeticError; the handler binds it as ``e`` and reads
        # its public ``message`` field.
        body = (
            _var("msg", _str("none"), type_ann=_ty("text")),
            _try(
                (_let("z", _binop(BinOp.DIV, _int(1), _int(0))),),
                _catch(
                    (_set("msg", _field(_ref("e"), "message")),),
                    exc_type="ArithmeticError",
                    binding="e",
                ),
            ),
        )
        assert _execute(body).snapshot()["msg"] == TextValue("Division by zero")

    def test_try_catch_reraises_when_no_handler_matches(self) -> None:
        from agm.agl.eval.exceptions import AglRaise

        body = (
            _try(
                (_let("z", _binop(BinOp.DIV, _int(1), _int(0))),),
                _catch((_pass(),), exc_type="Abort"),
            ),
        )
        with pytest.raises(AglRaise) as exc_info:
            _execute(body)
        assert exc_info.value.exc.type_name == "ArithmeticError"

    def test_raise_bound_exception_propagates(self) -> None:
        """``raise e`` of a caught exception value re-propagates it."""
        from agm.agl.eval.exceptions import AglRaise

        body = (
            _try(
                (_let("z", _binop(BinOp.DIV, _int(1), _int(0))),),
                _catch(
                    (_raise(_ref("e")),),
                    exc_type="ArithmeticError",
                    binding="e",
                ),
            ),
        )
        with pytest.raises(AglRaise) as exc_info:
            _execute(body)
        assert exc_info.value.exc.type_name == "ArithmeticError"

    def test_raise_non_exception_value_raises_runtime(self) -> None:
        """``raise <int>`` passes typecheck but fails at runtime."""
        body = (_raise(_int(5)),)
        with pytest.raises(RuntimeError, match="expected an ExceptionValue"):
            _execute(body)

    def test_input_and_type_declarations_are_runtime_noops(self) -> None:
        """input / record / enum / type-alias declarations bind nothing at runtime."""
        body = (
            _input("x"),
            _record_def("Point", _field_def("v", _ty("int"))),
            _enum_def("Color", _variant_def("Red")),
            _type_alias("Num", _ty("int")),
            _pass(),
        )
        root = _execute(body)
        assert root.snapshot() == {}


# ---------------------------------------------------------------------------
# Coverage: let/var type annotation coercion (_exec_let, _exec_var)
# ---------------------------------------------------------------------------


class TestLetVarCoercion:
    """let/var with explicit type annotations that trigger coercion."""

    def test_let_with_decimal_annotation_coerces_int(self) -> None:
        """``let x: decimal = 3`` widens the int to a DecimalValue."""
        from agm.agl.eval.values import DecimalValue

        body = (_let("x", _int(3), type_ann=_ty("decimal")),)
        assert _execute(body).snapshot()["x"] == DecimalValue(decimal.Decimal(3))

    def test_var_with_text_annotation_no_coerce(self) -> None:
        from agm.agl.eval.values import TextValue

        body = (_var("msg", _str("hi"), type_ann=_ty("text")),)
        assert _execute(body).snapshot()["msg"] == TextValue("hi")

    def test_let_with_bool_annotation(self) -> None:
        from agm.agl.eval.values import BoolValue

        body = (_let("b", _bool(True), type_ann=_ty("bool")),)
        assert _execute(body).snapshot()["b"] == BoolValue(True)

    def test_let_with_json_annotation(self) -> None:
        from agm.agl.eval.values import JsonValue

        body = (_let("j", _null(), type_ann=_ty("json")),)
        assert _execute(body).snapshot()["j"] == JsonValue(None)

    def test_set_into_decimal_binding_coerces_int(self) -> None:
        """``set`` of an int into a decimal-typed var widens to DecimalValue."""
        from agm.agl.eval.values import DecimalValue

        body = (
            _var("x", _dec("1.0"), type_ann=_ty("decimal")),
            _set("x", _int(7)),
        )
        assert _execute(body).snapshot()["x"] == DecimalValue(decimal.Decimal(7))


# ---------------------------------------------------------------------------
# Coverage: _eval_expr paths (operators, list, dict)
# ---------------------------------------------------------------------------


class TestEvalExprCompound:
    """Compound expressions evaluated through resolve + check + execute.

    Each test binds ``let r = <expr>`` and inspects the public root binding.
    """

    def test_binary_op_add_int(self) -> None:
        from agm.agl.eval.values import IntValue

        assert _eval_value(_binop(BinOp.ADD, _int(3), _int(4))) == IntValue(7)

    def test_binary_op_sub(self) -> None:
        from agm.agl.eval.values import IntValue

        assert _eval_value(_binop(BinOp.SUB, _int(10), _int(3))) == IntValue(7)

    def test_binary_op_mul(self) -> None:
        from agm.agl.eval.values import IntValue

        assert _eval_value(_binop(BinOp.MUL, _int(4), _int(5))) == IntValue(20)

    def test_binary_op_div_yields_decimal(self) -> None:
        from agm.agl.eval.values import DecimalValue

        assert isinstance(_eval_value(_binop(BinOp.DIV, _int(7), _int(2))), DecimalValue)

    def test_binary_op_compare_eq(self) -> None:
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_binop(BinOp.EQ, _int(1), _int(1))) == BoolValue(True)

    def test_binary_op_in_list(self) -> None:
        from agm.agl.eval.values import BoolValue

        prelude = (_let("lst", _list(_int(1), _int(2))),)
        result = _eval_value(_binop(BinOp.IN, _int(1), _ref("lst")), prelude=prelude)
        assert result == BoolValue(True)

    def test_binary_op_and_short_circuit_false(self) -> None:
        """``false and true`` short-circuits to false."""
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_binop(BinOp.AND, _bool(False), _bool(True))) == BoolValue(False)

    def test_binary_op_and_both_true(self) -> None:
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_binop(BinOp.AND, _bool(True), _bool(True))) == BoolValue(True)

    def test_binary_op_or_short_circuit_true(self) -> None:
        """``true or false`` short-circuits to true."""
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_binop(BinOp.OR, _bool(True), _bool(False))) == BoolValue(True)

    def test_binary_op_or_both_false(self) -> None:
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_binop(BinOp.OR, _bool(False), _bool(False))) == BoolValue(False)

    def test_is_test_matching_variant(self) -> None:
        from agm.agl.eval.values import BoolValue

        prelude = (
            _enum_def("Status", _variant_def("Active"), _variant_def("Inactive")),
            _let("s", _ctor("Active", qualifier="Status")),
        )
        assert _eval_value(_is_test(_ref("s"), "Active"), prelude=prelude) == BoolValue(True)

    def test_is_test_negated(self) -> None:
        from agm.agl.eval.values import BoolValue

        prelude = (
            _enum_def("Status", _variant_def("Active"), _variant_def("Inactive")),
            _let("s", _ctor("Inactive", qualifier="Status")),
        )
        result = _eval_value(_is_test(_ref("s"), "Active", negated=True), prelude=prelude)
        assert result == BoolValue(True)

    def test_case_expr_matches(self) -> None:
        """A case expression returns the matching branch's body value."""
        from agm.agl.eval.values import IntValue

        expr = _case_expr(
            _int(5),
            _case_expr_branch(_lit_pat(_int(5)), _int(99)),
            _case_expr_branch(_wild(), _int(0)),
        )
        assert _eval_value(expr) == IntValue(99)

    def test_case_expr_no_match_raises(self) -> None:
        from agm.agl.eval.exceptions import AglRaise

        expr = _case_expr(_int(1), _case_expr_branch(_lit_pat(_int(99)), _int(0)))
        with pytest.raises(AglRaise) as exc_info:
            _eval_value(expr)
        assert exc_info.value.exc.type_name == "MatchError"

    def test_list_literal(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue

        result = _eval_value(_list(_int(1), _int(2)))
        assert isinstance(result, ListValue)
        assert result.elements == (IntValue(1), IntValue(2))

    def test_dict_literal(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue

        result = _eval_value(_dict(k=_int(42)))
        assert isinstance(result, DictValue)
        assert result.entries["k"] == IntValue(42)

    def test_unary_not_true(self) -> None:
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_unary_not(_bool(True))) == BoolValue(False)

    def test_unary_neg_int(self) -> None:
        from agm.agl.eval.values import IntValue

        assert _eval_value(_unary_neg(_int(5))) == IntValue(-5)

    def test_unary_neg_decimal(self) -> None:
        from agm.agl.eval.values import DecimalValue

        assert _eval_value(_unary_neg(_dec("2.5"))) == DecimalValue(decimal.Decimal("-2.5"))


# ---------------------------------------------------------------------------
# Coverage: field access on RecordValue, EnumValue, ExceptionValue
# ---------------------------------------------------------------------------


class TestFieldAccess:
    """Field access on record and exception values via the public pipeline.

    Field access on enums, on non-record/enum/exception values, and on
    undeclared fields is rejected by the type checker, so only the
    record-field and exception-field paths are reachable at runtime.
    """

    def test_field_access_on_record_value(self) -> None:
        from agm.agl.eval.values import IntValue

        body = (
            _record_def("Point", _field_def("x", _ty("int"))),
            _let("p", _ctor("Point", args=(_arg("x", _int(3)),))),
            _let("r", _field(_ref("p"), "x")),
        )
        assert _execute(body).snapshot()["r"] == IntValue(3)

    def test_field_access_on_exception_value(self) -> None:
        """Reading a field off a caught exception value returns that field."""
        from agm.agl.eval.values import TextValue

        body = (
            _var("msg", _str("none"), type_ann=_ty("text")),
            _try(
                (_let("z", _binop(BinOp.DIV, _int(1), _int(0))),),
                _catch(
                    (_set("msg", _field(_ref("e"), "message")),),
                    exc_type="ArithmeticError",
                    binding="e",
                ),
            ),
        )
        assert _execute(body).snapshot()["msg"] == TextValue("Division by zero")


# ---------------------------------------------------------------------------
# Coverage: constructor evaluation
# ---------------------------------------------------------------------------


class TestConstructorEval:
    """Record / enum constructors via the public pipeline.

    Constructing an unknown type is a static error, so only the record and
    enum-variant constructor paths are reachable at runtime.
    """

    def test_constructor_record(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue

        body = (
            _record_def("Point", _field_def("x", _ty("int"))),
            _let("p", _ctor("Point", args=(_arg("x", _int(5)),))),
        )
        value = _execute(body).snapshot()["p"]
        assert isinstance(value, RecordValue)
        assert value.type_name == "Point"
        assert value.fields["x"] == IntValue(5)

    def test_constructor_enum_qualified(self) -> None:
        """``Color.Red`` builds an EnumValue."""
        from agm.agl.eval.values import EnumValue

        body = (
            _enum_def("Color", _variant_def("Red")),
            _let("c", _ctor("Red", qualifier="Color")),
        )
        value = _execute(body).snapshot()["c"]
        assert isinstance(value, EnumValue)
        assert value.variant == "Red"

    def test_constructor_unqualified_enum_variant(self) -> None:
        """An unqualified variant name resolves by scanning enum types."""
        from agm.agl.eval.values import EnumValue

        body = (
            _enum_def("Status", _variant_def("Active")),
            _let("s", _ctor("Active")),
        )
        value = _execute(body).snapshot()["s"]
        assert isinstance(value, EnumValue)
        assert value.variant == "Active"


# ---------------------------------------------------------------------------
# Coverage: agent call edge cases (fallback contract, strict_json, retries)
# ---------------------------------------------------------------------------


class TestAgentCallEdgeCases:
    def test_agent_call_uses_fallback_contract_when_missing(self) -> None:
        """With no contract registered for a call node, a TextCodec fallback is used.

        Driven through the interpreter's public ``execute`` entry on a real
        parsed program (``let x = prompt "hi"``) with an empty contract map, so
        the defensive fallback path materializes the contract.
        """
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import TextValue
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check

        program = parse_program('let x = prompt "hi"')
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        def my_fn(req: AgentRequest) -> str:
            return "hello"

        registry = AgentRegistry(named={}, default_agent=my_fn)
        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts={},  # No contracts → triggers the fallback contract.
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
        )
        root = Scope(parent=None)
        interp.execute(root)
        assert root.snapshot()["x"] == TextValue("hello")

    def test_agent_call_strict_json_from_contract(self) -> None:
        """A call uses ``contract.strict_json`` when it is not None.

        Driven through ``execute`` on a real parsed program with a host-supplied
        contract whose ``strict_json`` overrides the runtime default.
        """
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import TextValue
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.runtime.codec import ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import AgentCall, LetDecl
        from agm.agl.typecheck import check
        from agm.agl.typecheck.types import TextType, Type

        program = parse_program('let x = prompt "hi"')
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        let_stmt = checked.resolved.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        call_expr = let_stmt.value
        assert isinstance(call_expr, AgentCall)

        seen_strict: list[bool] = []

        class RecordingCodec(TextCodec):
            def parse(
                self, raw: str, target_type: Type, *, strict_json: bool = False
            ) -> ParseResult:
                seen_strict.append(strict_json)
                return super().parse(raw, target_type, strict_json=strict_json)

        def my_fn(req: AgentRequest) -> str:
            return "ok"

        registry = AgentRegistry(named={}, default_agent=my_fn)
        contract = OutputContract(
            target_type=TextType(),
            codec=RecordingCodec(),
            strict_json=True,  # Overrides the runtime default of False.
            format_instructions="",
            json_schema=None,
        )
        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts={call_expr.node_id: contract},
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
        )
        root = Scope(parent=None)
        interp.execute(root)
        assert root.snapshot()["x"] == TextValue("ok")
        # The contract's strict_json=True flowed through to the codec.
        assert seen_strict == [True]

    def test_agent_call_retry_policy_exhausts(self) -> None:
        """A retry-policy call whose codec always fails raises AgentParseError.

        Driven through the interpreter's public ``execute`` entry on a real
        parsed program (``let x = prompt[on_parse_error: retry[1]] "hi"``).  In
        M1 only the JSON codec (M2) can fail, so this exercises the path with a
        public, host-supplied failing codec injected via the contract map.
        """
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.runtime.codec import ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import AgentCall, LetDecl
        from agm.agl.typecheck import check
        from agm.agl.typecheck.types import TextType

        source = 'let x = prompt[on_parse_error: retry[1]] "hi"'
        program = parse_program(source)
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        # Locate the real AgentCall node so we can key its output contract.
        let_stmt = checked.resolved.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        call_expr = let_stmt.value
        assert isinstance(call_expr, AgentCall)

        class AlwaysFailCodec(TextCodec):
            def parse(
                self, raw: str, target_type: object, *, strict_json: bool = False
            ) -> ParseResult:
                return ParseResult.failure("always fails")

        def my_fn(req: AgentRequest) -> str:
            return "not-valid"

        registry = AgentRegistry(named={}, default_agent=my_fn)
        contract = OutputContract(
            target_type=TextType(),
            codec=AlwaysFailCodec(),
            strict_json=None,
            format_instructions="",
            json_schema=None,
        )
        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts={call_expr.node_id: contract},
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
        )

        with pytest.raises(AglRaise) as exc_info:
            interp.execute(Scope(parent=None))
        assert exc_info.value.exc.type_name == "AgentParseError"


# ---------------------------------------------------------------------------
# Coverage: _resolve_type_ann branches (Text, Bool, Json)
# ---------------------------------------------------------------------------


class TestResolveTypeAnn:
    def test_resolve_type_ann_text(self) -> None:
        from agm.agl.eval.interpreter import _resolve_type_ann
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import TextT
        from agm.agl.typecheck.types import TextType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        result = _resolve_type_ann(TextT(span=span, node_id=0))
        assert isinstance(result, TextType)

    def test_resolve_type_ann_bool(self) -> None:
        from agm.agl.eval.interpreter import _resolve_type_ann
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import BoolT
        from agm.agl.typecheck.types import BoolType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        result = _resolve_type_ann(BoolT(span=span, node_id=0))
        assert isinstance(result, BoolType)

    def test_resolve_type_ann_json(self) -> None:
        from agm.agl.eval.interpreter import _resolve_type_ann
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import JsonT
        from agm.agl.typecheck.types import JsonType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        result = _resolve_type_ann(JsonT(span=span, node_id=0))
        assert isinstance(result, JsonType)

    def test_resolve_type_ann_unknown_returns_none(self) -> None:
        from agm.agl.eval.interpreter import _resolve_type_ann

        result = _resolve_type_ann(object())
        assert result is None


# ---------------------------------------------------------------------------
# Coverage: _match_pattern ConstructorPattern for RecordValue and EnumValue
# ---------------------------------------------------------------------------


class TestMatchPatternConstructor:
    """Constructor (enum-variant) patterns matched via ``case`` statements.

    The type checker only allows constructor patterns against enum subjects
    (record / non-enum subjects and undeclared variant fields are static
    errors), so only the enum-variant paths are reachable at runtime.  Each
    test sets a root ``var`` inside the matching branch to observe the outcome.
    """

    @staticmethod
    def _option_prelude() -> tuple[Stmt, ...]:
        # enum Option { Some(n: int), Nothing }; let o = Option.Some(n: 1)
        return (
            _enum_def(
                "Option",
                _variant_def("Some", _field_def("n", _ty("int"))),
                _variant_def("Nothing"),
            ),
            _let("o", _ctor("Some", qualifier="Option", args=(_arg("n", _int(1)),))),
        )

    def test_enum_variant_pattern_binds_field(self) -> None:
        """``Some(n: val)`` matches and binds the variant field into ``val``."""
        from agm.agl.eval.values import IntValue

        body = (
            *self._option_prelude(),
            _var("r", _int(0), type_ann=_ty("int")),
            _case_stmt(
                _ref("o"),
                _case_stmt_branch(
                    _ctor_pat("Some", fields=(_pat_field("n", _var_pat("val")),)),
                    (_set("r", _ref("val")),),
                ),
                _case_stmt_branch(_wild(), (_pass(),)),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(1)

    def test_enum_variant_name_mismatch_skips_branch(self) -> None:
        """A different variant name does not match (falls through to wildcard)."""
        from agm.agl.eval.values import IntValue

        body = (
            *self._option_prelude(),
            _var("r", _int(0), type_ann=_ty("int")),
            _case_stmt(
                _ref("o"),
                _case_stmt_branch(_ctor_pat("Nothing"), (_set("r", _int(1)),)),
                _case_stmt_branch(_wild(), (_pass(),)),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(0)

    def test_enum_variant_sub_pattern_no_match_skips_branch(self) -> None:
        """A sub-pattern that fails (``Some(n: 99)`` vs n=1) does not match."""
        from agm.agl.eval.values import IntValue

        body = (
            *self._option_prelude(),
            _var("r", _int(0), type_ann=_ty("int")),
            _case_stmt(
                _ref("o"),
                _case_stmt_branch(
                    _ctor_pat("Some", fields=(_pat_field("n", _lit_pat(_int(99))),)),
                    (_set("r", _int(1)),),
                ),
                _case_stmt_branch(_wild(), (_pass(),)),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(0)


# ---------------------------------------------------------------------------
# CheckedProgram.type_env carries the constructor namespace to the interpreter
# ---------------------------------------------------------------------------


class TestCheckedProgramTypeEnvConstructors:
    """The constructor namespace flows from check() to the interpreter via
    ``CheckedProgram.type_env``.

    The M1 parser cannot yet parse ``record``/constructor syntax, so the program
    is hand-built from AST nodes, then driven through the *real* resolve + check
    passes (no fabricated side tables) and evaluated via the interpreter's public
    ``execute`` entry.  This is the regression test for the bug where the runtime
    reconstructed an empty type env from expression-only ``node_types``, leaving
    every constructor unresolvable at runtime.
    """

    def test_record_constructor_resolves_to_record_value(self) -> None:
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DecimalValue, RecordValue
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.scope import resolve
        from agm.agl.scope.symbols import ResolvedProgram
        from agm.agl.syntax.nodes import (
            Constructor,
            FieldDef,
            IntLit,
            LetDecl,
            NamedArg,
            Program,
            RecordDef,
        )
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import DecimalT
        from agm.agl.typecheck import check

        span = SourceSpan(1, 1, 1, 1, 0, 0)

        # record Point { x: decimal }
        record = RecordDef(
            name="Point",
            fields=(FieldDef(name="x", type_expr=DecimalT(span=span, node_id=1), span=span,
                             node_id=2),),
            span=span,
            node_id=3,
        )
        # let p = Point(x: 1)
        ctor = Constructor(
            qualifier=None,
            name="Point",
            args=(NamedArg(name="x", value=IntLit(value=1, span=span, node_id=4), span=span,
                           node_id=5),),
            span=span,
            node_id=6,
        )
        let_p = LetDecl(name="p", type_ann=None, value=ctor, span=span, node_id=7)
        program = Program(body=(record, let_p), span=span, node_id=8)

        # Real resolve + check passes — no fabricated side tables.
        resolved: ResolvedProgram = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=False,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        # The type namespace must include the user record after checking.
        assert checked.type_env.get_type("Point") is not None

        interp = Interpreter(
            checked=checked,
            registry=AgentRegistry(named={}, default_agent=None),
            contracts={},
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
        )
        root = Scope(parent=None)
        interp.execute(root)

        binding = root.lookup("p")
        assert binding is not None
        value = binding.value
        assert isinstance(value, RecordValue)
        assert value.type_name == "Point"
        # int arg coerces to the decimal field type.
        assert value.fields["x"] == DecimalValue(decimal.Decimal(1))


# ---------------------------------------------------------------------------
# Coverage: runtime.py _convert_input int-from-Decimal (line 401)
# ---------------------------------------------------------------------------


class TestConvertInputDecimalToInt:
    def test_convert_input_int_from_decimal_string_parsed(self) -> None:
        """'1.0' parses as Decimal('1.0') == int(1) → IntValue(1)."""
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import IntType

        result = _convert_input("n", "1.0", IntType())
        assert result == IntValue(1)

    def test_convert_input_int_from_decimal_non_integer_raises(self) -> None:
        """'1.5' parses as Decimal('1.5') ≠ int(1) → ValueError."""
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import IntType

        with pytest.raises(ValueError, match="integer"):
            _convert_input("n", "1.5", IntType())


# ---------------------------------------------------------------------------
# Coverage: template interpolation non-VarRef expr path (lines 408->410)
# ---------------------------------------------------------------------------


class TestTemplateInterpSegment:
    def test_template_with_text_and_interp_segments(self) -> None:
        """A template (routed through ``_eval_expr``) concatenates its segments.

        The interpolation here is a non-VarRef (int literal), so the boundary
        tag has no variable name — the rendered value is the bare scalar.
        """
        from agm.agl.eval.values import TextValue

        template = _template(_text_seg("n="), _interp_seg(_int(42)))
        value = _eval_value(template)
        assert isinstance(value, TextValue)
        assert value.value == "n=42"


# ---------------------------------------------------------------------------
# Coverage: agent call shell_exec path (line 421)
# ---------------------------------------------------------------------------


class TestAgentCallShellExec:
    def test_exec_call_raises_exec_error(self) -> None:
        """An ``exec`` call surfaces ExecError (shell exec is unsupported in M1).

        ``exec "cmd"`` is M1-parseable, so this exercises the shell_exec call
        path through the public ``run`` surface: the uncaught AgL exception
        becomes ``RunResult.error``.
        """
        result = run('exec "cmd"')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "ExecError"


# ---------------------------------------------------------------------------
# Coverage: constructor coercion (int→decimal in record/enum fields)
# ---------------------------------------------------------------------------


class TestConstructorCoercion:
    def test_constructor_record_int_to_decimal_coercion(self) -> None:
        """An int arg for a decimal record field is coerced to DecimalValue."""
        from agm.agl.eval.values import DecimalValue, RecordValue

        body = (
            _record_def("Box", _field_def("v", _ty("decimal"))),
            _let("b", _ctor("Box", args=(_arg("v", _int(3)),))),
        )
        value = _execute(body).snapshot()["b"]
        assert isinstance(value, RecordValue)
        assert value.fields["v"] == DecimalValue(decimal.Decimal(3))

    def test_constructor_enum_int_to_decimal_coercion(self) -> None:
        """An int arg for a decimal enum-variant field is coerced to DecimalValue."""
        from agm.agl.eval.values import DecimalValue, EnumValue

        body = (
            _enum_def("Measure", _variant_def("Amount", _field_def("v", _ty("decimal")))),
            _let("m", _ctor("Amount", qualifier="Measure", args=(_arg("v", _int(5)),))),
        )
        value = _execute(body).snapshot()["m"]
        assert isinstance(value, EnumValue)
        assert value.fields["v"] == DecimalValue(decimal.Decimal(5))


# ---------------------------------------------------------------------------
# Coverage: binary op and/or with non-BoolValue right (lines 609, 616)
# ---------------------------------------------------------------------------


class TestBinaryOpNonBoolRight:
    def test_and_true_left_returns_non_bool_right(self) -> None:
        """``true and <int>`` returns the right operand value verbatim."""
        from agm.agl.eval.values import IntValue

        assert _eval_value(_binop(BinOp.AND, _bool(True), _int(42))) == IntValue(42)

    def test_or_false_left_returns_non_bool_right(self) -> None:
        """``false or <int>`` returns the right operand value verbatim."""
        from agm.agl.eval.values import IntValue

        assert _eval_value(_binop(BinOp.OR, _bool(False), _int(7))) == IntValue(7)


# ---------------------------------------------------------------------------
# Coverage: _to_decimal with invalid type (line 751)
# ---------------------------------------------------------------------------


class TestToDecimalInvalidType:
    def test_to_decimal_non_numeric_raises(self) -> None:
        from agm.agl.eval.interpreter import _to_decimal
        from agm.agl.eval.values import TextValue

        with pytest.raises(RuntimeError, match="Not a numeric value"):
            _to_decimal(TextValue("hello"))


# ---------------------------------------------------------------------------
# Coverage: _compare fallback return BoolValue(False) (line 802)
# ---------------------------------------------------------------------------


class TestCompareFallback:
    def test_compare_non_ordering_op_on_int_returns_false(self) -> None:
        """Passing a non-ordering, non-EQ/NEQ op to _compare with int/int
        falls through all if-branches and returns BoolValue(False) at line 802.

        Path: op=ADD, left=IntValue, right=IntValue
        → EQ check: False
        → NEQ check: False
        → int/int block entered, LT/LE/GT/GE all False
        → falls to return BoolValue(False)
        """
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(IntValue(1), IntValue(2), BinOp.ADD)
        assert result == BoolValue(False)


# ---------------------------------------------------------------------------
# Coverage: _match_pattern unknown literal type (line 868)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coverage: _exec_case_stmt with VarPattern binding (line 247)
# ---------------------------------------------------------------------------


class TestCaseStmtVarBinding:
    def test_case_stmt_var_pattern_captures_binding(self) -> None:
        """A var pattern in a ``case`` statement captures the subject value.

        Observed by copying the captured ``n`` into a root-scope ``var``.
        """
        from agm.agl.eval.values import IntValue

        body = (
            _var("r", _int(0), type_ann=_ty("int")),
            _case_stmt(
                _int(5),
                _case_stmt_branch(_var_pat("n"), (_set("r", _ref("n")),)),
            ),
        )
        assert _execute(body).snapshot()["r"] == IntValue(5)


# ---------------------------------------------------------------------------
# Coverage: _exec_try_catch handler without binding (line 270->277)
# ---------------------------------------------------------------------------


class TestTryCatchHandlerNoBinding:
    def test_try_catch_handler_without_binding_catches(self) -> None:
        """A handler with ``binding=None`` still catches a matching exception."""
        from agm.agl.eval.values import TextValue

        body = (
            _var("caught", _str("no"), type_ann=_ty("text")),
            _try(
                (_let("z", _binop(BinOp.DIV, _int(1), _int(0))),),
                _catch((_set("caught", _str("yes")),), exc_type="ArithmeticError"),
            ),
        )
        assert _execute(body).snapshot()["caught"] == TextValue("yes")


# ---------------------------------------------------------------------------
# Coverage: _exec_let/_exec_var with annotation that resolves to None
# ---------------------------------------------------------------------------


class TestLetVarAnnotationNone:
    """let/var with a list annotation: ``_resolve_type_ann`` returns None.

    A ``list`` annotation is not one of the scalar coercion targets, so no
    coercion is applied and the value passes through unchanged.
    """

    def test_let_with_list_annotation_no_coerce(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue

        body = (
            _let("lst", _list(_int(7)), type_ann=_list_ty(_ty("int"))),
        )
        value = _execute(body).snapshot()["lst"]
        assert value == ListValue(elements=(IntValue(7),))

    def test_var_with_list_annotation_no_coerce(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue

        body = (
            _var("lst", _list(_int(8)), type_ann=_list_ty(_ty("int"))),
        )
        value = _execute(body).snapshot()["lst"]
        assert value == ListValue(elements=(IntValue(8),))


# ---------------------------------------------------------------------------
# Coverage: remaining interpreter gaps (lines 366, 404->401, 560, 570,
# 655, 688, 725, runtime.py 517->510, 521->510)
# ---------------------------------------------------------------------------


class TestRemainingCoverage:
    def test_template_two_text_segments_concatenate(self) -> None:
        """A template with two text segments concatenates them in order."""
        from agm.agl.eval.values import TextValue

        template = _template(_text_seg("hello "), _text_seg("world"))
        assert _eval_value(template) == TextValue("hello world")

    def test_case_expr_var_pattern_binds_and_uses_binding(self) -> None:
        """A case-expression var pattern binds the subject and the body reads it."""
        from agm.agl.eval.values import IntValue

        expr = _case_expr(_int(42), _case_expr_branch(_var_pat("n"), _ref("n")))
        assert _eval_value(expr) == IntValue(42)

    def test_resolve_type_ann_int(self) -> None:
        """_resolve_type_ann returns IntType for IntT annotation."""
        from agm.agl.eval.interpreter import _resolve_type_ann
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import IntT
        from agm.agl.typecheck.types import IntType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        result = _resolve_type_ann(IntT(span=span, node_id=0))
        assert isinstance(result, IntType)

    def test_arith_decimal_subtraction(self) -> None:
        """_arith with decimal-decimal subtraction returns DecimalValue."""
        from agm.agl.eval.interpreter import _arith
        from agm.agl.eval.values import DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _arith(
            DecimalValue(decimal.Decimal("3.0")),
            DecimalValue(decimal.Decimal("1.5")),
            BinOp.SUB,
        )
        assert isinstance(result, DecimalValue)
        assert result == DecimalValue(decimal.Decimal("1.5"))

    def test_compare_non_ordering_op_on_decimal_returns_false(self) -> None:
        """Passing a non-ordering op to _compare with decimal/decimal falls to line 802."""
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("1.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.ADD,
        )
        assert result == BoolValue(False)

    def test_compare_non_ordering_op_on_text_returns_false(self) -> None:
        """Passing a non-ordering op to _compare with text/text falls to line 802."""
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, TextValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(TextValue("a"), TextValue("b"), BinOp.ADD)
        assert result == BoolValue(False)

