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
from agm.agl.eval.values import DecimalValue, IntValue
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
    default = agents.get("ask")
    others = {k: v for k, v in agents.items() if k != "ask"}
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


def _if_expr(*branches: ast.IfExprBranch) -> ast.IfExpr:
    return ast.IfExpr(branches=tuple(branches), span=_sp(), node_id=_nid())


def _if_expr_branch(cond: Expr, body: Expr) -> ast.IfExprBranch:
    return ast.IfExprBranch(cond=cond, body=body, span=_sp(), node_id=_nid())


def _if_expr_else(body: Expr) -> ast.IfExprBranch:
    return ast.IfExprBranch(cond=ast.ELSE, body=body, span=_sp(), node_id=_nid())


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


def _check_program(
    body: tuple[Stmt, ...], *, has_default_agent: bool = False
) -> CheckedProgram:
    """Run *body* statements through the real resolve + check passes."""
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.scope import resolve
    from agm.agl.syntax.nodes import Program
    from agm.agl.typecheck import check

    program = Program(body=tuple(body), span=_sp(), node_id=_nid())
    resolved = resolve(program)
    caps = HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=has_default_agent,
        codec_kinds={"text": frozenset({"text"})},
        renderer_names=frozenset({"default", "raw"}),
    )
    return check(resolved, caps)


def _execute(
    body: tuple[Stmt, ...],
    *,
    default_agent: AgentFn | None = None,
    named: dict[str, AgentFn] | None = None,
    has_default_agent: bool = False,
) -> Scope:
    """Build + resolve + check + execute *body*, returning the root ``Scope``.

    Mirrors ``WorkflowRuntime.run`` for constructs the M1 parser cannot parse
    yet: drives the program through the real static passes and the public
    ``Interpreter.execute`` entry point.  Raised ``AglRaise`` / ``RuntimeError``
    propagate to the caller (the user-visible failure surface).
    """
    from agm.agl.eval.interpreter import Interpreter
    from agm.agl.runtime.agents import AgentRegistry

    checked = _check_program(
        body, has_default_agent=has_default_agent or default_agent is not None
    )
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

    def test_template_in_let_binding_no_boundary_markers(self) -> None:
        """Templates in ``let`` bindings use console rendering (no boundary tags).

        Boundary-marker rendering (``<dsl-value …>``) is for agent-call prompts
        only.  Any other template context — ``let``, ``var``, ``set``, ``print``
        — produces plain text so that bindings contain the value as-is.
        """
        result = run('let s = "abc"\nlet msg = "x: ${s}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        assert v.value == "x: abc"
        assert "<dsl-value" not in v.value

    def test_template_explicit_as_default_no_boundary_markers(self) -> None:
        """F3: an explicit ``${v as default}`` in a console context renders as
        plain text, NOT as a ``<dsl-value>`` boundary tag.

        Regression: the console renderer previously only special-cased an
        *implicit* (``None``) renderer, so ``as default`` routed through the
        prompt renderer and leaked boundary tags into ``let``/``print`` output.
        """
        result = run('let v = "hi"\nlet t = "x: ${v as default}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["t"]
        assert isinstance(v, TextValue)
        assert v.value == "x: hi"
        assert "<dsl-value" not in v.value

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

    def test_print_template_text_no_dsl_tags(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Template interpolation inside ``print`` must NOT add <dsl-value> tags."""
        result = run('let name = "world"\nprint "hello ${name}"')
        assert result.ok
        out = capsys.readouterr().out
        assert "<dsl-value" not in out
        assert out == "hello world\n"

    def test_print_template_scalar_interpolation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Int/decimal/bool values inside a print template are rendered as plain text."""
        result = run("let n = 42\nprint \"n=${n}\"")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "n=42\n"


# ---------------------------------------------------------------------------
# Agent call evaluation (text codec)
# ---------------------------------------------------------------------------


class TestAgentCalls:
    def test_ask_call_binds_response(self) -> None:
        def agent(req: AgentRequest) -> str:
            return "response text"

        result = run_with_default_agent('let x = ask "Hello"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("response text")

    def test_named_agent_call(self) -> None:
        def impl(req: AgentRequest) -> str:
            return "output"

        rt = WorkflowRuntime()
        rt.register_agent("impl", impl)
        result = rt.run('agent impl\nlet x = impl "Do something"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("output")

    def test_agent_receives_rendered_prompt_raw(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let name = "world"\nlet x = ask "Hello ${name as raw}"', agent)
        assert len(prompts) == 1
        assert prompts[0] == "Hello world"

    def test_agent_prompt_contains_boundary_markers_for_text(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let artifact = "content"\nlet x = ask "see ${artifact}"', agent)
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
        rt.run('agent reviewer\nlet x = reviewer "Review this."')
        assert len(received) == 1
        assert received[0].agent == "reviewer"

    def test_empty_response_is_valid_for_text_target(self) -> None:
        def agent(req: AgentRequest) -> str:
            return ""

        result = run_with_default_agent('let x = ask "Say nothing."', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("")

    def test_no_default_agent_without_registration_fails_statically(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let x = ask "Hi"')
        # No default agent, no fallback → static capability error
        assert result.ok is False
        assert result.error is None

    def test_agent_response_object_accepted(self) -> None:
        from agm.agl.runtime.request import AgentResponse

        def agent(req: AgentRequest) -> AgentResponse:
            return AgentResponse(content="from object")

        result = run_with_default_agent('let x = ask "Hi"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("from object")

    def test_expr_stmt_call_result_discarded(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        result = run_with_default_agent('ask "Note something."', agent)
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
        rt.run('input name\nlet x = ask "Hi ${name as raw}"', inputs={})
        assert calls == []

    def test_input_used_in_template(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent(
            'input name\nlet x = ask "Hello ${name as raw}"',
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

        run_with_default_agent('let artifact = "hello"\nlet x = ask "see ${artifact}"', agent)
        assert '<dsl-value name="artifact" type="text">' in prompts[0]
        assert "hello" in prompts[0]
        assert "</dsl-value>" in prompts[0]

    def test_raw_renderer_bypasses_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let s = "raw content"\nlet x = ask "${s as raw}"', agent)
        assert prompts[0] == "raw content"
        assert "<dsl-value" not in prompts[0]

    def test_int_interp_default_is_scalar_no_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let n = 5\nlet x = ask "n=${n}"', agent)
        assert "5" in prompts[0]
        assert "<dsl-value" not in prompts[0]

    def test_bool_interp_default_is_scalar_no_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let b = true\nlet x = ask "b=${b}"', agent)
        assert "true" in prompts[0]
        assert "<dsl-value" not in prompts[0]

    def test_text_interp_name_attribute_is_varname(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent(
            'let my_artifact = "content"\nlet x = ask "${my_artifact}"', agent
        )
        assert 'name="my_artifact"' in prompts[0]

    def test_null_interp_boundary_marked(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let j: json = null\nlet x = ask "data: ${j}"', agent)
        assert "<dsl-value" in prompts[0]
        assert "null" in prompts[0]

    def test_text_interp_non_varref_has_no_name_attribute(self) -> None:
        """Non-VarRef interpolation in an agent prompt has no ``name=`` tag.

        When the interpolated expression is not a simple ``VarRef`` (e.g. a
        field access like ``e.message``), the boundary tag omits the ``name=``
        attribute (``var_name=None`` in ``_eval_template``).
        """
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        # ``e.message`` is a FieldAccess, not a VarRef — exercises the
        # ``var_name = None`` branch in ``_eval_template``.
        source = 'let x = ask "saw ${e.message}"'
        run_with_default_agent(
            "record Err\n  message: text\n"
            'let e = Err(message: "oops")\n' + source,
            agent,
        )
        # The tag should not have a ``name=`` attribute.
        assert 'name="e"' not in prompts[0]
        # But the boundary marker is still present for text values.
        assert "<dsl-value" in prompts[0]
        assert "oops" in prompts[0]


# ---------------------------------------------------------------------------
# AgentRequest fields
# ---------------------------------------------------------------------------


class TestAgentRequest:
    def test_request_has_prompt(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        run_with_default_agent('let x = ask "Hello world"', agent)
        assert received[0].prompt == "Hello world"

    def test_request_has_agent_name_ask(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        run_with_default_agent('let x = ask "Hi"', agent)
        assert received[0].agent == "ask"

    def test_request_has_agent_name_custom(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('agent reviewer\nlet x = reviewer "Review this"')
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

        result = run_with_default_agent('var x: text = ask "Get value"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("from agent")

    def test_set_from_agent_response(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append("call")
            return "v2"

        result = run_with_default_agent(
            'var x: text = ask "First"\nset x = ask "Second"', agent
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
            'let a = ask "First"\nlet b = ask "Second"', agent
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
            'let a = ask "First"\nlet b = ask "Use ${a as raw}"', agent
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
            'agent impl\nlet a = ask "Hello"\nlet b = impl "Build"',
            {"ask": default_agent, "impl": impl},
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


class TestJsonEquality:
    """F3: json = json compares the wrapped JSON trees with numeric int/decimal
    equivalence (JSON numbers compare numerically), without conflating bool and
    numbers."""

    def test_json_scalar_int_decimal_numeric_equivalence(self) -> None:
        from agm.agl.eval.values import BoolValue

        result = run("let a: json = 1\nlet b: json = 1.0\nlet x = (a = b)")
        assert result.ok
        assert result.bindings["x"] == BoolValue(True)

    def test_json_nested_int_decimal_numeric_equivalence(self) -> None:
        from agm.agl.eval.values import BoolValue

        result = run(
            "let a: json = {n: 1}\nlet b: json = {n: 1.0}\nlet x = (a = b)"
        )
        assert result.ok
        assert result.bindings["x"] == BoolValue(True)

    def test_json_list_int_decimal_numeric_equivalence(self) -> None:
        from agm.agl.eval.values import BoolValue

        result = run("let a: json = [1, 2]\nlet b: json = [1.0, 2.0]\nlet x = (a = b)")
        assert result.ok
        assert result.bindings["x"] == BoolValue(True)

    def test_json_bool_not_equal_to_number(self) -> None:
        """JSON ``true`` must not compare equal to JSON ``1`` (no bool/number
        conflation, unlike Python's ``True == 1``)."""
        from agm.agl.eval.values import BoolValue

        result = run("let a: json = true\nlet b: json = 1\nlet x = (a = b)")
        assert result.ok
        assert result.bindings["x"] == BoolValue(False)

    def test_json_distinct_numbers_not_equal(self) -> None:
        from agm.agl.eval.values import BoolValue

        result = run("let a: json = 1\nlet b: json = 2\nlet x = (a = b)")
        assert result.ok
        assert result.bindings["x"] == BoolValue(False)

    def test_json_lists_of_different_length_not_equal(self) -> None:
        from agm.agl.eval.values import BoolValue

        result = run("let a: json = [1, 2]\nlet b: json = [1]\nlet x = (a = b)")
        assert result.ok
        assert result.bindings["x"] == BoolValue(False)

    def test_json_dicts_with_different_keys_not_equal(self) -> None:
        from agm.agl.eval.values import BoolValue

        result = run("let a: json = {x: 1}\nlet b: json = {y: 1}\nlet x = (a = b)")
        assert result.ok
        assert result.bindings["x"] == BoolValue(False)

    # Task 1: bool/number equality guard inside containers (list[json] / dict[text,json])

    def test_list_json_bool_not_equal_to_number(self) -> None:
        """[true] ≠ [1] for list[json] — bool/number guard must apply inside containers.

        Regression: before Task 1 fix, list[json] comparison fell through to
        dataclass __eq__ which used Python ``True == 1`` semantics.
        """
        from agm.agl.eval.values import BoolValue

        result = run(
            "let a: list[json] = [true]\nlet b: list[json] = [1]\nlet x = (a = b)"
        )
        assert result.ok
        assert result.bindings["x"] == BoolValue(False)

    def test_list_json_bool_equal_to_bool(self) -> None:
        """[true] = [true] for list[json] — equal json values inside containers compare true."""
        from agm.agl.eval.values import BoolValue

        result = run(
            "let a: list[json] = [true]\nlet b: list[json] = [true]\nlet x = (a = b)"
        )
        assert result.ok
        assert result.bindings["x"] == BoolValue(True)

    def test_dict_json_value_bool_not_equal_to_number(self) -> None:
        """{k: true} ≠ {k: 1} for dict[text, json] — guard applies inside dict values."""
        from agm.agl.eval.values import BoolValue

        result = run(
            'let a: dict[text, json] = {k: true}\n'
            'let b: dict[text, json] = {k: 1}\n'
            "let x = (a = b)"
        )
        assert result.ok
        assert result.bindings["x"] == BoolValue(False)

    def test_dict_json_value_equal_to_same(self) -> None:
        """{k: true} = {k: true} for dict[text, json]."""
        from agm.agl.eval.values import BoolValue

        result = run(
            'let a: dict[text, json] = {k: true}\n'
            'let b: dict[text, json] = {k: true}\n'
            "let x = (a = b)"
        )
        assert result.ok
        assert result.bindings["x"] == BoolValue(True)

    # Task 2: JsonValue __eq__/__hash__ and container hash stability

    def test_json_value_direct_equality_bool_not_number(self) -> None:
        """JsonValue([True]) != JsonValue([1]) via the new __eq__."""
        from agm.agl.eval.values import JsonValue

        assert JsonValue([True]) != JsonValue([1])
        assert JsonValue([True]) == JsonValue([True])

    def test_json_value_hash_consistent_with_eq(self) -> None:
        """JsonValue items that compare equal must have the same hash."""
        from agm.agl.eval.values import JsonValue

        a = JsonValue(1)
        b = JsonValue(1)
        assert a == b
        assert hash(a) == hash(b)

    def test_json_value_bool_and_number_different_hashes(self) -> None:
        """JsonValue(True) and JsonValue(1) compare unequal and hash differently."""
        from agm.agl.eval.values import JsonValue

        assert JsonValue(True) != JsonValue(1)
        # They should hash differently (not a hard invariant but strongly expected).
        assert hash(JsonValue(True)) != hash(JsonValue(1))

    def test_container_with_json_value_payload_hashable(self) -> None:
        """DictValue/RecordValue/EnumValue/ExceptionValue containing JsonValue
        payloads wrapping dicts/lists must be hashable (Task 2).

        Before the fix, ``__hash__`` called ``hash(tuple(sorted(...)))`` which
        invoked Python's built-in ``hash`` on the JsonValue, which in turn tried
        to hash the underlying dict/list — raising ``TypeError: unhashable type``.
        """
        from agm.agl.eval.values import (
            DictValue,
            EnumValue,
            ExceptionValue,
            JsonValue,
            RecordValue,
        )

        # JsonValue wrapping an unhashable payload.
        jv = JsonValue({"a": [1, 2]})

        d = DictValue(entries={"data": jv})
        assert isinstance(hash(d), int)

        r = RecordValue(type_name="R", fields={"f": jv})
        assert isinstance(hash(r), int)

        e = EnumValue(type_name="E", variant="V", fields={"f": jv})
        assert isinstance(hash(e), int)

        x = ExceptionValue(type_name="X", fields={"f": jv})
        assert isinstance(hash(x), int)

    def test_json_value_hash_list_and_dict(self) -> None:
        """_json_hash covers the list and dict branches (coverage for values.py)."""
        from agm.agl.eval.values import JsonValue

        # List branch.
        h_list = hash(JsonValue([1, 2, 3]))
        assert isinstance(h_list, int)
        # Dict branch.
        h_dict = hash(JsonValue({"k": "v"}))
        assert isinstance(h_dict, int)
        # Fallback (str/None).
        h_str = hash(JsonValue("hello"))
        assert isinstance(h_str, int)
        h_none = hash(JsonValue(None))
        assert isinstance(h_none, int)

    def test_json_value_eq_not_implemented_for_non_json_value(self) -> None:
        """JsonValue.__eq__ returns NotImplemented for non-JsonValue objects."""
        from agm.agl.eval.values import JsonValue

        jv = JsonValue(42)
        result = jv.__eq__("not a json value")
        assert result is NotImplemented

    def test_dict_value_order_permuted_json_payload_equal_and_equal_hashes(self) -> None:
        """DictValue with order-permuted JsonValue dict payloads: equal AND same hash.

        Regression for repr-based hashing: two DictValues whose entries map to
        identical JsonValue(dict) payloads but with different insertion order must
        hash the same (because _json_hash is order-insensitive for dicts).
        """
        from agm.agl.eval.values import DictValue, JsonValue

        # Same key/value content, different insertion order in the JsonValue dict.
        jv1 = JsonValue({"a": 1, "b": 2})
        jv2 = JsonValue({"b": 2, "a": 1})

        # The two JsonValues compare equal (order-insensitive __eq__).
        assert jv1 == jv2
        # They must also hash the same.
        assert hash(jv1) == hash(jv2)

        d1 = DictValue(entries={"data": jv1})
        d2 = DictValue(entries={"data": jv2})

        # The two DictValues compare equal.
        assert d1 == d2
        # Their hashes must also be equal (eq/hash contract).
        assert hash(d1) == hash(d2)
        # A set collapses them to one element.
        assert len({d1, d2}) == 1

    def test_record_value_int_vs_decimal_json_payload_equal_and_equal_hashes(self) -> None:
        """RecordValue with int vs Decimal("1.0") JsonValue payloads: equal AND same hash.

        Regression for repr-based hashing: repr(JsonValue(1)) != repr(JsonValue(Decimal("1.0")))
        but the two values compare equal under _json_eq, so their hashes must match.
        """
        import decimal as _decimal

        from agm.agl.eval.values import JsonValue, RecordValue

        jv_int = JsonValue(1)
        jv_dec = JsonValue(_decimal.Decimal("1.0"))

        # The two JsonValues compare equal (numeric equivalence).
        assert jv_int == jv_dec
        # They must hash the same.
        assert hash(jv_int) == hash(jv_dec)

        r1 = RecordValue(type_name="R", fields={"n": jv_int})
        r2 = RecordValue(type_name="R", fields={"n": jv_dec})

        # The two RecordValues compare equal.
        assert r1 == r2
        # Their hashes must also be equal.
        assert hash(r1) == hash(r2)
        # A set collapses them to one element.
        assert len({r1, r2}) == 1


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

    def test_wildcard_rethrow_propagates_original_exception(self) -> None:
        """F2: ``catch _ as e => raise e`` rethrows the original exception, which
        then propagates uncaught out of ``run()``."""
        result = run("try\n  let z: decimal = 1 / 0\ncatch _ as e =>\n  raise e\n")
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ArithmeticError"

    def test_named_base_rethrow_propagates_original_exception(self) -> None:
        """F2: ``catch Exception as e => raise e`` is equivalent to the wildcard
        rethrow and propagates the original exception."""
        result = run("try\n  let z: decimal = 1 / 0\ncatch Exception as e =>\n  raise e\n")
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "ArithmeticError"


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
        # F3/F9: Decimal is preserved exactly, never routed through float.
        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.serialize import value_to_json_obj

        v = DecimalValue(decimal.Decimal("3.14"))
        result = value_to_json_obj(v)
        assert isinstance(result, decimal.Decimal)
        assert result == decimal.Decimal("3.14")

    def test_value_to_json_obj_bool(self) -> None:
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.serialize import value_to_json_obj

        assert value_to_json_obj(BoolValue(True)) is True
        assert value_to_json_obj(BoolValue(False)) is False

    def test_value_to_json_obj_json(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.serialize import value_to_json_obj

        v = JsonValue({"nested": [1, 2]})
        result = value_to_json_obj(v)
        assert result == {"nested": [1, 2]}

    def test_value_to_json_obj_list(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.serialize import value_to_json_obj

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        result = value_to_json_obj(v)
        assert result == [1, 2]

    def test_value_to_json_obj_dict(self) -> None:
        from agm.agl.eval.values import DictValue, TextValue
        from agm.agl.runtime.serialize import value_to_json_obj

        v = DictValue(entries={"k": TextValue("v")})
        result = value_to_json_obj(v)
        assert result == {"k": "v"}

    def test_value_to_json_obj_record(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.serialize import value_to_json_obj

        v = RecordValue(type_name="P", fields={"x": IntValue(5)})
        result = value_to_json_obj(v)
        assert result == {"x": 5}

    def test_value_to_json_obj_enum(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.serialize import value_to_json_obj

        v = EnumValue(type_name="C", variant="Red", fields={})
        result = value_to_json_obj(v)
        assert result == {"$case": "Red"}

    def test_value_to_json_obj_exception(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.serialize import value_to_json_obj

        v = ExceptionValue(
            type_name="E", fields={"message": TextValue("oops"), "trace_id": TextValue("")}
        )
        result = value_to_json_obj(v)
        assert isinstance(result, dict)
        assert result.get("message") == "oops"

    def test_dumps_exact_decimal_unquoted_exact_text(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        # Decimal emitted as exact unquoted numeric text (no float round trip).
        assert dumps_exact(decimal.Decimal("1.5"), indent=None) == "1.5"
        # Trailing-zero significance is preserved (1.50 != 1.5 here).
        assert dumps_exact(decimal.Decimal("1.50"), indent=None) == "1.50"
        assert dumps_exact(decimal.Decimal("0.1"), indent=None) == "0.1"
        # No scientific notation for large/small magnitudes.
        assert dumps_exact(decimal.Decimal("1E+2"), indent=None) == "100"

    def test_dumps_exact_nested_structure(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        obj = {"a": decimal.Decimal("1.5"), "b": [1, True, None, "x"]}
        out = dumps_exact(obj, indent=None)
        assert out == '{"a": 1.5, "b": [1, true, null, "x"]}'

    def test_dumps_exact_pretty_indent(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        out = dumps_exact({"a": [decimal.Decimal("0.1")]}, indent=2)
        assert out == '{\n  "a": [\n    0.1\n  ]\n}'

    def test_dumps_exact_empty_containers(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact([], indent=2) == "[]"
        assert dumps_exact({}, indent=2) == "{}"
        assert dumps_exact([], indent=None) == "[]"
        assert dumps_exact({}, indent=None) == "{}"

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

    def test_render_for_prompt_unknown_renderer_raises(self) -> None:
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_prompt

        v = TextValue("hello")
        # Unknown renderer is a loud internal error, not a silent default
        # fallback (F2, M3b).
        with pytest.raises(AssertionError, match="nonexistent"):
            render_for_prompt(
                v, renderer_name="nonexistent", var_name="x", renderers={}
            )

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
        from agm.agl.runtime.trace import noop_trace

        result = _div(IntValue(10), IntValue(4), trace=noop_trace())
        assert isinstance(result, DecimalValue)

    def test_arithmetic_uses_pinned_context_not_ambient(self) -> None:
        """F7: AgL arithmetic must not depend on the host's ambient decimal
        context.  With ambient precision deliberately lowered to 5, ``1 / 3``
        must still be computed at the pinned 28-digit precision."""
        import decimal as _decimal

        from agm.agl.eval.values import DecimalValue

        ctx = _decimal.getcontext()
        saved_prec = ctx.prec
        ctx.prec = 5
        try:
            result = run("let x: decimal = 1 / 3\n")
        finally:
            ctx.prec = saved_prec
        assert result.ok
        value = result.bindings["x"]
        assert isinstance(value, DecimalValue)
        # 28-digit precision: 0.3333... with 28 significant digits.
        assert value.value == _decimal.Decimal(1) / _decimal.Decimal(3)
        digits = len(value.value.as_tuple().digits)
        assert digits == 28, f"expected 28 significant digits, got {digits}"

    def test_div_by_zero_raises_agl_raise(self) -> None:
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import _div
        from agm.agl.eval.values import IntValue, TextValue
        from agm.agl.runtime.trace import noop_trace

        with pytest.raises(AglRaise) as exc_info:
            _div(IntValue(5), IntValue(0), trace=noop_trace())
        assert exc_info.value.exc.type_name == "ArithmeticError"
        # The minted trace_id is present on the raised exception (F1).
        trace_id = exc_info.value.exc.fields.get("trace_id")
        assert isinstance(trace_id, TextValue) and trace_id.value

    def test_div_type_error(self) -> None:
        from agm.agl.eval.interpreter import _div
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.trace import noop_trace

        with pytest.raises(RuntimeError, match="Cannot divide"):
            _div(TextValue("a"), TextValue("b"), trace=noop_trace())

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

    def test_compare_ordering_int_decimal_widen(self) -> None:
        """Ordering widens int→decimal: ``1 < 1.5`` and ``2.0 > 1``."""
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        assert _compare(
            IntValue(1), DecimalValue(decimal.Decimal("1.5")), BinOp.LT
        ) == BoolValue(True)
        assert _compare(
            DecimalValue(decimal.Decimal("2.0")), IntValue(1), BinOp.GT
        ) == BoolValue(True)

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

    def test_in_op_list_int_decimal_widening(self) -> None:
        """F8: ``IntValue(1) in [DecimalValue(1.0)]`` is true (``=`` semantics)."""
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, DecimalValue, IntValue, ListValue

        v = ListValue(elements=(DecimalValue(decimal.Decimal("1.0")),))
        assert _in_op(IntValue(1), v) == BoolValue(True)

    def test_value_eq_int_decimal_widening(self) -> None:
        """F8: the shared ``_value_eq`` widens int↔decimal for equality."""
        from agm.agl.eval.interpreter import _value_eq
        from agm.agl.eval.values import DecimalValue, IntValue

        assert _value_eq(IntValue(1), DecimalValue(decimal.Decimal("1.0"))) is True
        assert _value_eq(DecimalValue(decimal.Decimal("2.0")), IntValue(2)) is True
        assert _value_eq(IntValue(1), DecimalValue(decimal.Decimal("2.0"))) is False

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

    def test_case_literal_int_decimal_widening_match(self) -> None:
        """F5: a literal pattern matches with int→decimal widening, so a decimal
        scrutinee ``1.0`` matches an int literal pattern ``1`` (consistent with
        ``1 = 1.0``)."""
        result = run(
            "let n: decimal = 1.0\n"
            "case n of\n"
            '  | 1 => print "matched"\n'
            '  | _ => print "no"\n'
        )
        assert result.ok

    def test_case_literal_int_decimal_widening_match_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run(
            "let n: decimal = 1.0\n"
            "case n of\n"
            '  | 1 => print "matched"\n'
            '  | _ => print "no"\n'
        )
        assert capsys.readouterr().out == "matched\n"

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
        """F6: nominal values keep their declared names; built-in values map to
        their AgL type names (not Python class names like ``IntValue``)."""
        import decimal as _decimal

        from agm.agl.eval.interpreter import _describe_value
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

        # Nominal types keep their declared names.
        assert "Status" == _describe_value(
            EnumValue(type_name="Status", variant="Active", fields={})
        )
        assert "Point" == _describe_value(RecordValue(type_name="Point", fields={}))
        assert "Abort" == _describe_value(ExceptionValue(type_name="Abort", fields={}))
        # Built-ins map to AgL type names.
        assert _describe_value(IntValue(1)) == "int"
        assert _describe_value(TextValue("x")) == "text"
        assert _describe_value(BoolValue(True)) == "bool"
        assert _describe_value(DecimalValue(_decimal.Decimal("1.5"))) == "decimal"
        assert _describe_value(JsonValue(None)) == "json"
        assert _describe_value(ListValue(elements=())) == "list"
        assert _describe_value(DictValue(entries={})) == "dict"

    def test_match_error_scrutinee_type_is_agl_name(self) -> None:
        """F6: an uncaught MatchError exposes the AgL type name (``int``), not the
        Python class name (``IntValue``)."""
        result = run("case 5 of\n  | 0 => pass\n")
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "MatchError"
        assert result.error.fields["scrutinee_type"] == "int"

    def test_unary_not_on_non_bool_rejected_statically(self) -> None:
        """F1: ``not <int>`` is rejected by the checker (the runtime type guard
        is now genuinely unreachable)."""
        from agm.agl.typecheck import AglTypeError

        body = (_let("a", _unary_not(_int(1))),)
        with pytest.raises(AglTypeError, match="not"):
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
        result = rt.run('let x = ask "Hi"')
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

    def test_do_until_exhaustion_populates_schema_fields(self) -> None:
        """F2: ``MaxIterationsExceeded`` carries limit + last_condition_value.

        With a constant-false condition, ``last_condition_value`` is ``False``
        and ``limit`` reflects the loop bound.  (Condition source text requires
        the source-threaded runtime path; here we assert the value fields.)
        """
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import BoolValue, IntValue

        body = (_do_until(_bool(False), (_pass(),), limit=2),)
        with pytest.raises(AglRaise) as exc_info:
            _execute(body)
        fields = exc_info.value.exc.fields
        assert fields["limit"] == IntValue(2)
        assert fields["last_condition_value"] == BoolValue(False)
        assert "metadata" in fields
        assert "condition" in fields

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

    def test_raise_non_exception_value_rejected_statically(self) -> None:
        """F1: ``raise <int>`` is rejected by the checker (the runtime type guard
        is now genuinely unreachable)."""
        from agm.agl.typecheck import AglTypeError

        body = (_raise(_int(5)),)
        with pytest.raises(AglTypeError, match="raise"):
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

    def test_exception_constructor_eval(self) -> None:
        """Exception constructors (e.g. Abort) are evaluated to ExceptionValue."""
        result = run('raise Abort(message: "stop now")')
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "Abort"
        assert result.error.fields["message"] == "stop now"


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

    def test_binary_op_in_list_int_decimal_widening(self) -> None:
        """F8: ``1 in [1.0]`` is true — membership uses ``=`` value-equality.

        The static pass already accepts ``int in list[decimal]`` (int→decimal),
        so the runtime must agree instead of using raw dataclass equality.
        """
        from agm.agl.eval.values import BoolValue

        prelude = (_let("lst", _list(_dec("1.0"), _dec("2.0"))),)
        result = _eval_value(_binop(BinOp.IN, _int(1), _ref("lst")), prelude=prelude)
        assert result == BoolValue(True)

    def test_binary_op_in_dict_key_unaffected(self) -> None:
        """F8: dict-key membership (``"a" in {"a": 1}``) is unaffected."""
        from agm.agl.eval.values import BoolValue

        prelude = (_let("d", _dict(a=_int(1))),)
        result = _eval_value(_binop(BinOp.IN, _str("a"), _ref("d")), prelude=prelude)
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

    def test_case_expr_no_match_carries_span(self) -> None:
        """Task 4a: uncaught MatchError from a case expression carries the
        expression's source span (line/col), so WorkflowRuntime can report the
        raise site.

        Before the fix, ``_eval_case_expr`` raised ``AglRaise`` without a
        span, so ``exc.span`` was always ``None``.
        """
        result = run(
            "enum C\n  | A\n  | B\n"
            "let v: C = A\n"
            "let x = case v of\n"
            "  | B => 1\n"  # no match for A → MatchError
        )
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "MatchError"
        # The raise-site line must be set (not None) because the span was threaded.
        assert result.error.line is not None
        assert result.error.line >= 1

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
# F3: alias-qualified enum constructors and patterns (design §5.4)
# ---------------------------------------------------------------------------


class TestAliasQualifiedConstructors:
    def test_alias_qualified_construction(self) -> None:
        """``Status.Pass`` builds the underlying ``Review`` enum value."""
        from agm.agl.eval.values import EnumValue

        body = (
            _enum_def("Review", _variant_def("Pass")),
            _type_alias("Status", tast.NameT(name="Review", span=_sp(), node_id=_nid())),
            _let("r", _ctor("Pass", qualifier="Status")),
        )
        value = _execute(body).snapshot()["r"]
        assert isinstance(value, EnumValue)
        # Runtime carries the resolved underlying enum, not the alias name.
        assert value.type_name == "Review"
        assert value.variant == "Pass"

    def test_alias_of_alias_construction(self) -> None:
        """A multi-hop alias chain resolves to the underlying enum."""
        from agm.agl.eval.values import EnumValue

        body = (
            _enum_def("Review", _variant_def("Pass")),
            _type_alias("A", tast.NameT(name="Review", span=_sp(), node_id=_nid())),
            _type_alias("B", tast.NameT(name="A", span=_sp(), node_id=_nid())),
            _let("r", _ctor("Pass", qualifier="B")),
        )
        value = _execute(body).snapshot()["r"]
        assert isinstance(value, EnumValue)
        assert value.type_name == "Review"
        assert value.variant == "Pass"

    def test_alias_qualified_pattern_matches(self) -> None:
        """An alias-qualified pattern (``Status.Pass``) matches at runtime."""
        from agm.agl.eval.values import TextValue

        body = (
            _enum_def("Review", _variant_def("Pass"), _variant_def("Fail")),
            _type_alias("Status", tast.NameT(name="Review", span=_sp(), node_id=_nid())),
            _var("out", _str("none"), type_ann=_ty("text")),
            _let("r", _ctor("Pass", qualifier="Status")),
            _case_stmt(
                _ref("r"),
                _case_stmt_branch(
                    _ctor_pat("Pass", qualifier="Status"),
                    (_set("out", _str("matched")),),
                ),
                _case_stmt_branch(_ctor_pat("Fail"), (_set("out", _str("fail")),)),
            ),
        )
        value = _execute(body).snapshot()["out"]
        assert value == TextValue("matched")

    def test_alias_qualified_is_test_accepted(self) -> None:
        """An alias-qualified ``is`` test type-checks and evaluates true."""
        from agm.agl.eval.values import BoolValue

        body = (
            _enum_def("Review", _variant_def("Pass"), _variant_def("Fail")),
            _type_alias("Status", tast.NameT(name="Review", span=_sp(), node_id=_nid())),
            _let("r", _ctor("Pass", qualifier="Status")),
            _let("b", _is_test(_ref("r"), "Pass", qualifier="Status")),
        )
        value = _execute(body).snapshot()["b"]
        assert value == BoolValue(True)

    def test_wrong_qualifier_is_test_rejected(self) -> None:
        """An ``is`` test whose qualifier names a different enum is rejected."""
        from agm.agl.typecheck import AglTypeError

        body = (
            _enum_def("Review", _variant_def("Pass")),
            _enum_def("Other", _variant_def("Pass")),
            _let("r", _ctor("Pass", qualifier="Review")),
            _let("b", _is_test(_ref("r"), "Pass", qualifier="Other")),
        )
        with pytest.raises(AglTypeError):
            _execute(body)

    def test_wrong_qualifier_pattern_rejected(self) -> None:
        """A constructor pattern whose qualifier names a different enum is rejected."""
        from agm.agl.typecheck import AglTypeError

        body = (
            _enum_def("Review", _variant_def("Pass")),
            _enum_def("Other", _variant_def("Pass")),
            _let("r", _ctor("Pass", qualifier="Review")),
            _case_stmt(
                _ref("r"),
                _case_stmt_branch(
                    _ctor_pat("Pass", qualifier="Other"), (_pass(),)
                ),
            ),
        )
        with pytest.raises(AglTypeError):
            _execute(body)

    def test_non_enum_alias_qualifier_in_is_test_rejected(self) -> None:
        """An ``is`` test qualified by a non-enum alias is rejected."""
        from agm.agl.typecheck import AglTypeError

        body = (
            _enum_def("Review", _variant_def("Pass")),
            _type_alias("Nums", _list_ty(_ty("int"))),
            _let("r", _ctor("Pass", qualifier="Review")),
            _let("b", _is_test(_ref("r"), "Pass", qualifier="Nums")),
        )
        with pytest.raises(AglTypeError):
            _execute(body)


# ---------------------------------------------------------------------------
# Coverage: agent call edge cases (fallback contract, strict_json, retries)
# ---------------------------------------------------------------------------


class TestAgentCallEdgeCases:
    def test_agent_call_uses_fallback_contract_when_missing(self) -> None:
        """With no contract registered for a call node, a TextCodec fallback is used.

        Driven through the interpreter's public ``execute`` entry on a real
        parsed program (``let x = ask "hi"``) with an empty contract map, so
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

        program = parse_program('let x = ask "hi"')
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
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

        program = parse_program('let x = ask "hi"')
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
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
                self,
                raw: str,
                target_type: Type,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> ParseResult:
                seen_strict.append(strict_json)
                return super().parse(
                    raw, target_type, strict_json=strict_json, schema=schema
                )

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
        parsed program (``let x = ask[on_parse_error: retry[1]] "hi"``).  In
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

        source = 'let x = ask[on_parse_error: retry[1]] "hi"'
        program = parse_program(source)
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
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
                self,
                raw: str,
                target_type: object,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
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

    def test_agent_parse_error_normalized_raw_threaded(self) -> None:
        """F5: AgentParseError.normalized_raw is the recovered text, not the raw.

        A codec failure that carries a ``normalized_raw`` (the recovered JSON
        text after fence-stripping) surfaces on the exception's
        ``normalized_raw`` field, distinct from the raw fenced response.
        """
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.exceptions import AglRaise
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
        from agm.agl.typecheck.types import TextType

        raw_response = '```json\n{"bad": 1}\n```'
        recovered = '{"bad": 1}'

        source = 'let x = ask[on_parse_error: abort] "hi"'
        checked = check(
            resolve(parse_program(source)),
            HostCapabilities(
                agent_names=frozenset(),
                has_default_agent=True,
                codec_kinds={"text": frozenset({"text"})},
                renderer_names=frozenset({"default"}),
            ),
        )
        let_stmt = checked.resolved.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        call_expr = let_stmt.value
        assert isinstance(call_expr, AgentCall)

        class RecoveringFailCodec(TextCodec):
            def parse(
                self,
                raw: str,
                target_type: object,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> ParseResult:
                return ParseResult.failure(
                    "schema invalid", normalized_raw=recovered
                )

        registry = AgentRegistry(named={}, default_agent=lambda req: raw_response)
        contract = OutputContract(
            target_type=TextType(),
            codec=RecoveringFailCodec(),
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
        exc = exc_info.value.exc
        assert exc.type_name == "AgentParseError"
        assert exc.fields["raw"] == TextValue(raw_response)
        assert exc.fields["normalized_raw"] == TextValue(recovered)
        assert exc.fields["normalized_raw"] != exc.fields["raw"]


# ---------------------------------------------------------------------------
# Coverage: _coerce — the §5.8 runtime coercion (int→decimal, json wrapping,
# element-wise container recursion)
# ---------------------------------------------------------------------------


class TestCoerceToType:
    """``_coerce`` materializes the §5.8 coercions in the runtime value."""

    def test_int_widens_to_decimal(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import DecimalValue, IntValue
        from agm.agl.typecheck.types import DecimalType

        result = _coerce(IntValue(3), DecimalType())
        assert result == DecimalValue(decimal.Decimal(3))

    def test_text_unchanged(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import TextValue
        from agm.agl.typecheck.types import TextType

        value = TextValue("hi")
        assert _coerce(value, TextType()) is value

    def test_json_wraps_scalar(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import IntValue, JsonValue
        from agm.agl.typecheck.types import JsonType

        result = _coerce(IntValue(5), JsonType())
        assert result == JsonValue(5)

    def test_json_wraps_list_to_json_obj(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import BoolValue, IntValue, JsonValue, ListValue
        from agm.agl.typecheck.types import JsonType

        value = ListValue(elements=(IntValue(1), BoolValue(True)))
        result = _coerce(value, JsonType())
        assert result == JsonValue([1, True])

    def test_existing_json_passes_through(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import JsonValue
        from agm.agl.typecheck.types import JsonType

        value = JsonValue({"a": 1})
        assert _coerce(value, JsonType()) is value

    def test_list_widens_elements(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import DecimalValue, IntValue, ListValue
        from agm.agl.typecheck.types import DecimalType, ListType

        value = ListValue(elements=(IntValue(1), IntValue(2)))
        result = _coerce(value, ListType(elem=DecimalType()))
        assert result == ListValue(
            elements=(DecimalValue(decimal.Decimal(1)), DecimalValue(decimal.Decimal(2)))
        )

    def test_dict_widens_values(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import DecimalValue, DictValue, IntValue
        from agm.agl.typecheck.types import DecimalType, DictType

        value = DictValue(entries={"k": IntValue(7)})
        result = _coerce(value, DictType(value=DecimalType()))
        assert result == DictValue(entries={"k": DecimalValue(decimal.Decimal(7))})


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
# Coverage: runtime.py convert_input int-from-Decimal (line 401)
# ---------------------------------------------------------------------------


class TestConvertInputDecimalToInt:
    def test_convert_input_int_from_decimal_string_parsed(self) -> None:
        """'1.0' parses as Decimal('1.0') == int(1) → IntValue(1)."""
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import IntType

        result = convert_input("n", "1.0", IntType())
        assert result == IntValue(1)

    def test_convert_input_int_from_decimal_non_integer_raises(self) -> None:
        """'1.5' parses as Decimal('1.5') ≠ int(1) → ValueError."""
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import IntType

        with pytest.raises(ValueError, match="integer"):
            convert_input("n", "1.5", IntType())


# ---------------------------------------------------------------------------
# Template evaluation via _eval_expr (non-agent-call context uses console rendering)
# ---------------------------------------------------------------------------


class TestTemplateInterpSegment:
    def test_template_with_text_and_interp_segments(self) -> None:
        """A template evaluated via ``_eval_expr`` (non-agent-call context) concatenates
        its segments using console rendering (no ``<dsl-value>`` boundary markers).
        """
        from agm.agl.eval.values import TextValue

        template = _template(_text_seg("n="), _interp_seg(_int(42)))
        value = _eval_value(template)
        assert isinstance(value, TextValue)
        assert value.value == "n=42"
        assert "<dsl-value" not in value.value


# ---------------------------------------------------------------------------
# M4: exec shell executor (design §4.12, §11.13)
# ---------------------------------------------------------------------------


class TestAgentCallShellExec:
    def test_exec_call_succeeds_with_text_output(self) -> None:
        """An ``exec`` call runs the shell command and returns stdout (M4).

        ``exec "echo hi"`` runs via ``sh -c``, strips trailing newlines, and
        binds the stdout text.  The run succeeds (no static error, no exception).
        """
        result = run('let x = exec "echo hi"\n')
        assert result.ok is True
        assert result.error is None
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("hi")

    def test_exec_nonzero_exit_raises_exec_error(self) -> None:
        """A nonzero exit from a shell command raises a catchable ExecError (M4).

        The ExecError carries ``exit_code`` and ``timed_out=False``; the run
        ends with an uncaught AgL exception (``result.error``).
        """
        result = run('let x = exec "false"\n')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "ExecError"
        assert result.error.fields["exit_code"] == 1
        assert result.error.fields["timed_out"] is False

    def test_exec_null_byte_in_interpolation_raises_catchable_exec_error(self) -> None:
        """A NUL byte in an interpolated exec value surfaces as a catchable
        ``ExecError`` (spawn failure), never a raw Python ``ValueError`` (F1).

        ``subprocess.Popen`` rejects an argv element containing an embedded NUL
        byte with ``ValueError('embedded null byte')``.  That must be mapped to
        the spawn-failure ``ExecError`` so it stays inside the AgL exception
        model and can be caught by a ``try``/``catch``.
        """
        result = run(
            'input bad\nlet x = exec "echo ${bad as raw}"\n',
            inputs={"bad": "a\x00b"},
        )
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "ExecError"
        assert result.error.fields["exit_code"] == -1

    def test_exec_null_byte_in_interpolation_is_catchable(self) -> None:
        """The NUL-induced ExecError is catchable like any spawn-failure (F1)."""
        from agm.agl.eval.values import TextValue

        result = run(
            "input bad\n"
            'var ok = "no"\n'
            "try\n"
            '  let x = exec "echo ${bad as raw}"\n'
            "catch ExecError as e =>\n"
            '  set ok = "caught"\n',
            inputs={"bad": "a\x00b"},
        )
        assert result.ok is True
        assert result.error is None
        assert result.bindings["ok"] == TextValue("caught")

    def test_exec_checker_admits_exec_with_supports_shell_exec_true(self) -> None:
        """With ``supports_shell_exec=True`` the checker accepts ``exec`` calls.

        M4: WorkflowRuntime sets this flag so exec passes static checking.
        """
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import CallOptions, Program
        from agm.agl.typecheck import check

        exec_call = ast.AgentCall(
            agent="exec",
            options=CallOptions(
                format=None, strict_json=None, parse_policy=None,
                span=_sp(), node_id=_nid(),
            ),
            template=_template(_text_seg("true")),
            span=_sp(),
            node_id=_nid(),
        )
        program = Program(body=(_let("x", exec_call),), span=_sp(), node_id=_nid())
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
            supports_shell_exec=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default", "raw"}),
        )
        # Should not raise — exec is admitted by the checker.
        checked = check(resolved, caps)
        assert checked is not None


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
# F7: and/or short-circuit and always return BoolValue (design §4.3)
# ---------------------------------------------------------------------------


def _div_by_zero_eq() -> Expr:
    """A bool-typed expression that RAISES ArithmeticError when evaluated.

    ``(1 / 0) = 0`` type-checks as bool (division yields decimal; decimal = int
    is allowed), but evaluating it divides by zero and raises ``AglRaise``.  Used
    to observe whether the right operand of and/or is evaluated.
    """
    return _binop(BinOp.EQ, _binop(BinOp.DIV, _int(1), _int(0)), _int(0))


class TestBoolShortCircuit:
    def test_and_false_left_does_not_evaluate_right(self) -> None:
        """``false and (1/0 = 0)`` short-circuits: the right operand never runs."""
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_binop(BinOp.AND, _bool(False), _div_by_zero_eq())) == BoolValue(
            False
        )

    def test_or_true_left_does_not_evaluate_right(self) -> None:
        """``true or (1/0 = 0)`` short-circuits: the right operand never runs."""
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_binop(BinOp.OR, _bool(True), _div_by_zero_eq())) == BoolValue(
            True
        )

    def test_and_true_left_does_evaluate_right(self) -> None:
        """``true and (1/0 = 0)`` evaluates the right operand → ArithmeticError."""
        from agm.agl.eval.exceptions import AglRaise

        with pytest.raises(AglRaise) as exc_info:
            _eval_value(_binop(BinOp.AND, _bool(True), _div_by_zero_eq()))
        assert exc_info.value.exc.type_name == "ArithmeticError"

    def test_or_false_left_does_evaluate_right(self) -> None:
        """``false or (1/0 = 0)`` evaluates the right operand → ArithmeticError."""
        from agm.agl.eval.exceptions import AglRaise

        with pytest.raises(AglRaise) as exc_info:
            _eval_value(_binop(BinOp.OR, _bool(False), _div_by_zero_eq()))
        assert exc_info.value.exc.type_name == "ArithmeticError"

    def test_and_both_bool_returns_bool(self) -> None:
        """``true and true`` returns a BoolValue (not the right operand verbatim)."""
        from agm.agl.eval.values import BoolValue

        assert _eval_value(_binop(BinOp.AND, _bool(True), _bool(True))) == BoolValue(True)


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
    """let/var with a ``list[int]`` annotation: elements need no coercion.

    ``_coerce`` recurses into the list toward ``list[int]``; the ``int``
    elements are already the target type, so the value passes through equal.
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


# ---------------------------------------------------------------------------
# M4b: §7.8 retry-feedback composition — AgentRequest fields on retries
# ---------------------------------------------------------------------------


class TestRetryFeedbackComposition:
    """§7.8: The retry path threads previous_invalid_output, validation_errors,
    and attempt number into the AgentRequest sent on each retry call.

    The original rendered prompt is re-used unchanged for retries (design §7.8
    "Exact wording is host-configurable; the prompt template is rendered once").
    Confirmed by the acceptance test pin:
      parse_policies.scenarios.json:retry_recovers_on_last_attempt,
      prompts[call=1..3] all equal "Second review." (the raw template text).
    """

    def _build_retry_interpreter(
        self, source: str, agent_fn: AgentFn
    ) -> tuple[object, object, object]:
        """Build an Interpreter+contract for a retry test (codec always fails).

        Returns (interp, call_expr_node_id, contract).  Typed as object so
        callers can narrow via isinstance after importing the concrete types.
        """
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.runtime.codec import ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import AgentCall, LetDecl
        from agm.agl.typecheck import check
        from agm.agl.typecheck.types import TextType

        program = parse_program(source)
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        let_stmt = checked.resolved.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        call_expr = let_stmt.value
        assert isinstance(call_expr, AgentCall)

        registry = AgentRegistry(named={}, default_agent=agent_fn)

        class AlwaysFailCodec(TextCodec):
            def parse(
                self,
                raw: str,
                target_type: object,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> ParseResult:
                return ParseResult.failure("always fails")

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
            loop_limit=5,
            strict_json=False,
        )
        return interp, call_expr.node_id, contract

    def test_first_attempt_has_no_previous_output(self) -> None:
        """§7.8 r4: on the first attempt previous_invalid_output is None."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope

        requests: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            requests.append(req)
            return "raw-output-1"

        source = 'let x = ask[on_parse_error: retry[1]] "Initial."'
        interp, _, _ = self._build_retry_interpreter(source, agent)

        with pytest.raises(AglRaise):
            interp.execute(Scope(parent=None))

        assert len(requests) == 2
        assert requests[0].attempt == 0
        assert requests[0].previous_invalid_output is None
        assert requests[0].validation_errors == []

    def test_retry_carries_previous_invalid_output(self) -> None:
        """§7.8 r4: on retry, previous_invalid_output is the prior raw output."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope

        requests: list[AgentRequest] = []
        responses = ["first-bad-output", "second-bad-output"]

        def agent(req: AgentRequest) -> str:
            requests.append(req)
            return responses[req.attempt]

        source = 'let x = ask[on_parse_error: retry[1]] "Retry prompt."'
        interp, _, _ = self._build_retry_interpreter(source, agent)

        with pytest.raises(AglRaise):
            interp.execute(Scope(parent=None))

        assert len(requests) == 2
        # First attempt: no previous output.
        assert requests[0].previous_invalid_output is None
        # Retry: the raw output from the first attempt.
        assert requests[1].attempt == 1
        assert requests[1].previous_invalid_output == "first-bad-output"

    def test_retry_prompt_is_original_unchanged(self) -> None:
        """§7.8: The rendered prompt template is reused unchanged for retries.

        Feedback is carried in previous_invalid_output/validation_errors, not
        injected into the prompt text itself.
        """
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope

        seen_prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            seen_prompts.append(req.prompt)
            return "bad"

        source = 'let x = ask[on_parse_error: retry[2]] "The original prompt."'
        interp, _, _ = self._build_retry_interpreter(source, agent)

        with pytest.raises(AglRaise):
            interp.execute(Scope(parent=None))

        # All three attempts (1 initial + 2 retries) receive the same prompt.
        assert len(seen_prompts) == 3
        assert all(p == "The original prompt." for p in seen_prompts)

    def test_retry_attempt_counter_increments(self) -> None:
        """§7.8 r4: attempt counter is 0-based and increments on each retry."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope

        attempt_numbers: list[int] = []

        def agent(req: AgentRequest) -> str:
            attempt_numbers.append(req.attempt)
            return "bad"

        source = 'let x = ask[on_parse_error: retry[2]] "Hi."'
        interp, _, _ = self._build_retry_interpreter(source, agent)

        with pytest.raises(AglRaise):
            interp.execute(Scope(parent=None))

        assert attempt_numbers == [0, 1, 2]

    def test_multiple_retries_accumulate_previous_outputs(self) -> None:
        """§7.8: Each retry gets the immediately preceding failed output, not all."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope

        received: list[str | None] = []
        raw_outputs = ["out-0", "out-1", "out-2"]

        def agent(req: AgentRequest) -> str:
            received.append(req.previous_invalid_output)
            return raw_outputs[req.attempt]

        source = 'let x = ask[on_parse_error: retry[2]] "Hi."'
        interp, _, _ = self._build_retry_interpreter(source, agent)

        with pytest.raises(AglRaise):
            interp.execute(Scope(parent=None))

        # First: no previous; retry1: out-0; retry2: out-1.
        assert received == [None, "out-0", "out-1"]

    def test_agent_parse_error_attempts_field_reflects_max_attempts(self) -> None:
        """§7.8 r7 / §7.9: AgentParseError.attempts = max_attempts (not retries)."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue

        source = 'let x = ask[on_parse_error: retry[2]] "Hi."'
        interp, _, _ = self._build_retry_interpreter(source, lambda req: "bad")

        with pytest.raises(AglRaise) as exc_info:
            interp.execute(Scope(parent=None))

        exc = exc_info.value.exc
        assert exc.type_name == "AgentParseError"
        # retry[2] = 1 initial + 2 retries = 3 total attempts.
        assert exc.fields["attempts"] == IntValue(3)

    def test_abort_policy_single_attempt_no_retry(self) -> None:
        """§7.9: on_parse_error: abort calls the agent exactly once."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.scope import Scope

        call_count = [0]

        def agent(req: AgentRequest) -> str:
            call_count[0] += 1
            return "bad"

        source = 'let x = ask[on_parse_error: abort] "Hi."'
        interp, _, _ = self._build_retry_interpreter(source, agent)

        with pytest.raises(AglRaise) as exc_info:
            interp.execute(Scope(parent=None))

        assert call_count[0] == 1
        assert exc_info.value.exc.type_name == "AgentParseError"


# ---------------------------------------------------------------------------
# M4b: §8.1 AgentCallError schema — eval/catch conformance
# ---------------------------------------------------------------------------


class TestAgentCallErrorSchema:
    """§8.1 / §0 resolution 11: AgentCallError has agent/cause/metadata fields.

    AgentCallError is raised by M5 (the subprocess runner), not M4. These tests
    verify that the exception schema is correctly defined so that M4c (exec) and
    M5 (AgentCallError raises) can rely on the field set.
    """

    def test_agent_call_error_can_be_caught_and_fields_accessed(self) -> None:
        """Catch AgentCallError and access agent, cause, metadata fields at eval."""
        from agm.agl.eval.values import ExceptionValue, JsonValue, TextValue

        # Build an ExceptionValue with all §8.1 AgentCallError fields populated.
        exc_val = ExceptionValue(
            type_name="AgentCallError",
            fields={
                "message": TextValue("agent failed"),
                "trace_id": TextValue(""),
                "agent": TextValue("reviewer"),
                "cause": TextValue("spawn_failure"),
                "metadata": JsonValue({"exit_code": None, "elapsed": 0.5}),
            },
        )
        # Verify the fields are accessible (mirrors what the eval does on catch).
        assert exc_val.fields["agent"] == TextValue("reviewer")
        assert exc_val.fields["cause"] == TextValue("spawn_failure")
        assert isinstance(exc_val.fields["metadata"], JsonValue)

    def test_agent_call_error_schema_all_fields_correct_types(self) -> None:
        """§8.1: all AgentCallError fields are present and have correct AgL types."""
        from agm.agl.typecheck.types import BUILTIN_EXCEPTIONS, JsonType, TextType

        exc_type = BUILTIN_EXCEPTIONS["AgentCallError"]
        assert exc_type.fields["agent"] == TextType()
        assert exc_type.fields["cause"] == TextType()
        assert exc_type.fields["metadata"] == JsonType()

    def test_agent_call_error_cause_values(self) -> None:
        """§0 resolution 11: cause is "spawn_failure"|"nonzero_exit"|"timeout".

        The type is text (not an enum) — the runtime encodes the enumerated
        string cause values; the AgL type system has no enum narrowing for text.
        """
        from agm.agl.typecheck.types import BUILTIN_EXCEPTIONS, TextType

        exc_type = BUILTIN_EXCEPTIONS["AgentCallError"]
        # cause must be text (not a custom enum — the values are runtime-defined
        # string literals per §0 resolution 11).
        assert exc_type.fields["cause"] == TextType()

    def test_agent_call_error_metadata_is_json(self) -> None:
        """§0 resolution 11: metadata carries exit_code/stderr_tail/elapsed as json."""
        from agm.agl.typecheck.types import BUILTIN_EXCEPTIONS, JsonType

        exc_type = BUILTIN_EXCEPTIONS["AgentCallError"]
        assert exc_type.fields["metadata"] == JsonType()

    def test_undefined_variable_error_raised_at_runtime(self) -> None:
        """§8.1: UndefinedVariableError can be constructed as an ExceptionValue
        with the name field populated (the interpreter raises it for runtime
        undefined-variable paths that bypass the static check).
        """
        from agm.agl.eval.values import ExceptionValue, TextValue

        exc = ExceptionValue(
            type_name="UndefinedVariableError",
            fields={
                "message": TextValue("Undefined variable: 'x'"),
                "trace_id": TextValue(""),
                "name": TextValue("x"),
            },
        )
        assert exc.fields["name"] == TextValue("x")

    def test_immutable_binding_error_raised_at_runtime(self) -> None:
        """§8.1: ImmutableBindingError can be constructed with name/operation."""
        from agm.agl.eval.values import ExceptionValue, TextValue

        exc = ExceptionValue(
            type_name="ImmutableBindingError",
            fields={
                "message": TextValue("Cannot assign to immutable binding 'x'"),
                "trace_id": TextValue(""),
                "name": TextValue("x"),
                "operation": TextValue("set"),
            },
        )
        assert exc.fields["name"] == TextValue("x")
        assert exc.fields["operation"] == TextValue("set")


# ---------------------------------------------------------------------------
# M4: exec shell executor — interpreter coverage for exec code paths
# (design §4.12, §11.13)
# ---------------------------------------------------------------------------


class TestShellExecInterpreter:
    """Unit tests for the _exec_shell_exec interpreter code paths."""

    def test_exec_spawn_error_raises_exec_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spawn failure of sh itself maps to ExecError with exit_code=-1 (§11.13)."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.runtime.agents import AgentRegistry
        from agm.core.process import ProcessCaptureResult

        def fake_run_capture_result(
            cmd: list[str],
            *,
            idle_timeout: object = None,
            **_kwargs: object,
        ) -> ProcessCaptureResult:
            return ProcessCaptureResult(
                returncode=None,
                stdout="",
                stderr="",
                elapsed=0.0,
                timed_out=False,
                spawn_error="[Errno 2] No such file or directory: 'sh'",
                spawn_errno=2,
            )

        from agm.core import process as process_mod

        monkeypatch.setattr(process_mod, "run_capture_result", fake_run_capture_result)

        from agm.agl.capabilities import HostCapabilities
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import CallOptions, Program
        from agm.agl.typecheck import check

        exec_call = ast.AgentCall(
            agent="exec",
            options=CallOptions(
                format=None, strict_json=None, parse_policy=None,
                span=_sp(), node_id=_nid(),
            ),
            template=_template(_text_seg("cmd")),
            span=_sp(),
            node_id=_nid(),
        )
        program = Program(body=(_let("x", exec_call),), span=_sp(), node_id=_nid())
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
            supports_shell_exec=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default", "raw"}),
        )
        checked = check(resolved, caps)
        interp = Interpreter(
            checked=checked,
            registry=AgentRegistry(named={}, default_agent=None),
            contracts={},
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
        )
        from agm.agl.eval.values import BoolValue, IntValue

        with pytest.raises(AglRaise) as exc_info:
            interp.execute(Scope(parent=None))
        exc = exc_info.value.exc
        assert exc.type_name == "ExecError"
        assert exc.fields["exit_code"] == IntValue(-1)
        assert exc.fields["timed_out"] == BoolValue(False)

    def test_exec_no_contract_returns_text_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no contract exists for an exec node the result is plain TextValue.

        This exercises the ``contract is None`` fallback path which mirrors the
        same defensive fallback in ``_eval_agent_call``.  We drive the interpreter
        directly with an empty ``contracts`` dict so the node has no entry.
        """
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.agents import AgentRegistry
        from agm.core import process as process_mod
        from agm.core.process import ProcessCaptureResult

        def fake_run_capture_result(
            cmd: list[str],
            *,
            idle_timeout: object = None,
            **_kwargs: object,
        ) -> ProcessCaptureResult:
            return ProcessCaptureResult(
                returncode=0,
                stdout="hello\n",
                stderr="",
                elapsed=0.01,
                timed_out=False,
                spawn_error=None,
                spawn_errno=None,
            )

        monkeypatch.setattr(process_mod, "run_capture_result", fake_run_capture_result)

        from agm.agl.capabilities import HostCapabilities
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import CallOptions, Program
        from agm.agl.typecheck import check

        exec_call = ast.AgentCall(
            agent="exec",
            options=CallOptions(
                format=None, strict_json=None, parse_policy=None,
                span=_sp(), node_id=_nid(),
            ),
            template=_template(_text_seg("echo hello")),
            span=_sp(),
            node_id=_nid(),
        )
        program = Program(body=(_let("x", exec_call),), span=_sp(), node_id=_nid())
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
            supports_shell_exec=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default", "raw"}),
        )
        checked = check(resolved, caps)
        # Pass contracts={} so the node has no contract entry.
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
        assert root.snapshot()["x"] == TextValue("hello")

    def test_exec_parse_failure_raises_agent_parse_error(self) -> None:
        """Exec with a non-text typed target raises AgentParseError on bad output.

        Exercises the parse-failure path (lines 797-804): the codec cannot parse
        the stdout as the declared type and the parse policy is abort (1 attempt).
        """
        result = run('let x: int = exec "echo not-an-int"\n')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "AgentParseError"
        assert result.error.fields.get("agent") == "exec"

    def test_exec_typed_target_retry_policy_reruns_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typed exec retry re-runs the command so later valid output can succeed."""
        from agm.agl.eval.values import IntValue
        from agm.core import process as process_mod
        from agm.core.process import ProcessCaptureResult

        outputs = iter(("bad\n", "7\n"))
        calls = 0

        def fake_run_capture_result(
            cmd: list[str],
            *,
            idle_timeout: object = None,
            **_kwargs: object,
        ) -> ProcessCaptureResult:
            nonlocal calls
            calls += 1
            return ProcessCaptureResult(
                returncode=0,
                stdout=next(outputs),
                stderr="",
                elapsed=0.1,
                timed_out=False,
                spawn_error=None,
                spawn_errno=None,
            )

        monkeypatch.setattr(process_mod, "run_capture_result", fake_run_capture_result)

        result = run('let x: int = exec[on_parse_error: retry[2]] "ignored"\n')

        assert result.ok is True
        assert result.bindings.get("x") == IntValue(7)
        assert calls == 2

    def test_exec_retry_failure_reports_last_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every retry fails, the AgentParseError reflects the LAST attempt's output."""
        from agm.core import process as process_mod
        from agm.core.process import ProcessCaptureResult

        outputs = iter(("first-bad\n", "last-bad\n"))

        def fake_run_capture_result(
            cmd: list[str],
            *,
            idle_timeout: object = None,
            **_kwargs: object,
        ) -> ProcessCaptureResult:
            return ProcessCaptureResult(
                returncode=0,
                stdout=next(outputs),
                stderr="",
                elapsed=0.1,
                timed_out=False,
                spawn_error=None,
                spawn_errno=None,
            )

        monkeypatch.setattr(process_mod, "run_capture_result", fake_run_capture_result)

        result = run('let x: int = exec[on_parse_error: retry[1]] "ignored"\n')

        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "AgentParseError"
        assert result.error.fields.get("raw") == "last-bad"
        message = result.error.fields.get("message")
        assert isinstance(message, str)
        assert "last-bad" in message
        assert "first-bad" not in message

    def test_exec_typed_target_strict_json_branch(self) -> None:
        """strict_json per-call option flows through exec typed-target parsing.

        When ``strict_json: true`` is set on the call, the codec uses strict
        mode.  A valid bare JSON value succeeds even in strict mode.
        """
        result = run('let x: int = exec[strict_json: true] "echo 7"\n')
        assert result.ok is True
        from agm.agl.eval.values import IntValue

        assert result.bindings.get("x") == IntValue(7)


class TestExecIdleTimeoutBoundsWallTime:
    """Regression: exec idle timeout must bound wall time for compound commands (F1)."""

    def test_compound_command_idle_timeout_kills_whole_tree(self) -> None:
        """A compound ``sh -c "sleep N; echo x"`` must be torn down at the idle
        timeout, not run to completion.

        Without ``isolate_process_group=True`` at the exec call site, the
        orphaned ``sleep`` grandchild keeps the stdout pipe open and the idle
        timeout never fires — wall time would be ~N seconds.  With the fix the
        whole process group is killed and wall time stays near the timeout.

        A generous-but-bounding elapsed assertion (< 3s for a 5s sleep with a
        0.3s timeout) avoids flakiness while still proving the tree was killed.
        """
        import time

        rt = WorkflowRuntime(shell_exec_timeout=0.3)
        start = time.monotonic()
        result = rt.run('let x = exec "sleep 5; echo x"\n')
        elapsed = time.monotonic() - start

        # The command timed out → catchable ExecError (not a 5s run to success).
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "ExecError"
        assert result.error.fields.get("timed_out") is True
        # Wall time bounded well below the 5s sleep: the tree was actually killed.
        assert elapsed < 3.0, f"exec did not bound wall time: {elapsed:.1f}s elapsed"

    def test_timeout_message_mentions_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task 4b: the ExecError message for a timed-out command must mention
        timeout (not the generic "exited with code -1" message).

        Before the fix, the timeout path fell through to the generic
        ``f"Shell command exited with code {exit_code}: ..."`` branch, which
        said nothing about a timeout.
        """
        from agm.core import process as process_mod
        from agm.core.process import ProcessCaptureResult

        def fake_run_capture_result(
            cmd: list[str],
            *,
            idle_timeout: object = None,
            isolate_process_group: bool = False,
        ) -> ProcessCaptureResult:
            return ProcessCaptureResult(
                returncode=None,
                stdout="",
                stderr="",
                elapsed=0.1,
                timed_out=True,
                spawn_error=None,
                spawn_errno=None,
            )

        monkeypatch.setattr(process_mod, "run_capture_result", fake_run_capture_result)

        rt = WorkflowRuntime(shell_exec_timeout=0.1)
        result = rt.run('let x = exec "sleep 5"\n')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "ExecError"
        # timed_out field must be True.
        assert result.error.fields.get("timed_out") is True
        # The message must explicitly mention "timeout" — not "exited with code".
        message = str(result.error.fields.get("message", ""))
        assert "timeout" in message.lower(), (
            f"Expected timeout in message, got: {message!r}"
        )
        assert "exited with code" not in message


class TestExecParseErrorTraceLinkage:
    """F3: typed-exec parse failure links to the exec_command record's trace_id."""

    def test_uncaught_exec_parse_failure_links_to_exec_command_record(
        self, tmp_path: "object"
    ) -> None:
        """With logging on, an uncaught exec-parse AgentParseError carries the
        SAME non-empty ``trace_id`` as the emitted ``exec_command`` record."""
        import json
        from pathlib import Path

        log_file = Path(str(tmp_path)) / "trace.log"
        rt = WorkflowRuntime()
        result = rt.run('let x: int = exec "echo not-an-int"\n', log_file=log_file)

        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "AgentParseError"
        error_trace_id = result.error.fields.get("trace_id")
        assert isinstance(error_trace_id, str)
        assert error_trace_id != ""

        # The exception's trace_id must equal the exec_command record's id.
        records = [
            json.loads(line)
            for line in log_file.read_text().splitlines()
            if line.strip()
        ]
        exec_records = [r for r in records if r["kind"] == "exec_command"]
        assert len(exec_records) == 1
        assert exec_records[0]["trace_id"] == error_trace_id


class TestRenderForShell:
    """Unit tests for the render_for_shell function and _shell_plain_text helper."""

    def test_shell_plain_text_structured_value_is_compact_json(self) -> None:
        """Structured values are rendered as compact (single-line) JSON for quoting.

        Exercises the ``from agm.agl.runtime.serialize import dumps_exact`` path
        in ``_shell_plain_text`` (lines 275-277).
        """
        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.render import render_for_shell

        val = ListValue(elements=(TextValue("a"), TextValue("b")))
        result = render_for_shell(val, renderer_name=None)
        # Quoted compact JSON — no newlines inside the shell argument.
        import json
        import shlex

        # shlex.quote wraps in single quotes or leaves safe strings bare.
        # The result is a valid shell-quoted string containing compact JSON.
        # We verify it unquotes to the expected list.
        unquoted = shlex.split(result)
        assert len(unquoted) == 1
        assert json.loads(unquoted[0]) == ["a", "b"]

    def test_render_for_shell_explicit_non_raw_renderer_quotes_result(self) -> None:
        """An explicit non-raw renderer is applied then the result is shell-quoted.

        Exercises the else-branch of render_for_shell: a named renderer other
        than ``"default"`` or ``"raw"`` is looked up, applied, and the output is
        passed through shlex.quote.  Passes no ``renderers`` arg so the None→fallback
        path (line 314→315) is taken.
        """
        import shlex

        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.render import render_for_shell

        val = ListValue(elements=(TextValue("x"), TextValue("y")))
        result = render_for_shell(val, renderer_name="json")
        # "json" renderer produces pretty-printed JSON; the result is shell-quoted.
        unquoted = shlex.split(result)
        assert len(unquoted) == 1
        import json

        assert json.loads(unquoted[0]) == ["x", "y"]

    def test_render_for_shell_explicit_renderer_with_renderers_table(self) -> None:
        """Exercises the path where ``renderers`` is not None (line 314->316).

        Passes an explicit renderers table; ``render_for_shell`` uses it directly
        without falling back to the built-in default.
        """
        import shlex

        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_shell

        custom_renderers = {"myrender": lambda v, name: "CUSTOM"}
        val = TextValue("anything")
        result = render_for_shell(val, renderer_name="myrender", renderers=custom_renderers)
        assert shlex.split(result) == ["CUSTOM"]

    def test_render_for_shell_unknown_renderer_raises_assertion_error(self) -> None:
        """An unknown renderer name raises AssertionError (internal invariant, line 318).

        After type-checking this is unreachable through WorkflowRuntime.run
        because the checker validates renderer references.
        """
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_shell

        val = TextValue("x")
        with pytest.raises(AssertionError, match="not in the renderers table"):
            render_for_shell(val, renderer_name="no_such_renderer", renderers={})

    def test_render_for_shell_scalar_default_quotes(self) -> None:
        """Default rendering of a scalar text value quotes it with shlex.quote."""
        import shlex

        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_shell

        val = TextValue("hello world")
        result = render_for_shell(val, renderer_name=None)
        assert shlex.split(result) == ["hello world"]

    def test_render_for_shell_raw_bypasses_quoting(self) -> None:
        """``as raw`` returns plain text without shell-quoting."""
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_shell

        val = TextValue("a b c")
        result = render_for_shell(val, renderer_name="raw")
        # Raw: no quoting, value is inserted verbatim.
        assert result == "a b c"


class TestShellExecTemplateInterpolation:
    """Tests for _eval_template_for_shell interpolation path in the interpreter."""

    def test_exec_with_interpolated_value(self) -> None:
        """An interpolated value in exec template is shell-quoted by default."""
        # exec "printf '%s' ${name}" where name = "hello world" should produce
        # "hello world" (the shell-quoted value prevents word-splitting).
        result = run(
            'input name\n'
            'let out = exec "printf \'%s\' ${name}"\n',
            inputs={"name": "hello world"},
        )
        assert result.ok is True
        from agm.agl.eval.values import TextValue

        assert result.bindings.get("out") == TextValue("hello world")


# ---------------------------------------------------------------------------
# Source-level null literal pattern (grammar + parser + typecheck + eval)
# ---------------------------------------------------------------------------


class TestNullLiteralPatternSource:
    """End-to-end tests for ``null`` patterns written in AgL source.

    These tests drive the full pipeline (lexer → grammar → transformer →
    scope → typecheck → eval) to verify that ``case j of | null => ...`` works
    when written in source, not just when ASTs are constructed directly.
    """

    def test_null_pattern_matches_json_null(self) -> None:
        """``null`` pattern matches when the scrutinee is JSON null."""
        from agm.agl.eval.values import IntValue

        result = run(
            "let j: json = null\n"
            "let r: int = case j of\n"
            "  | null => 1\n"
            "  | _ => 0\n"
        )
        assert result.ok is True, result.error
        assert result.bindings.get("r") == IntValue(1)

    def test_null_pattern_fallback_for_non_null_json(self) -> None:
        """``null`` pattern does NOT match JSON false, 0, empty string, {}, or []."""
        from agm.agl.eval.values import IntValue

        for src_val in ("false", "0", '""', "{}", "[]"):
            source = (
                f"let j: json = {src_val}\n"
                "let r: int = case j of\n"
                "  | null => 1\n"
                "  | _ => 0\n"
            )
            result = run(source)
            assert result.ok is True, f"failed for j={src_val}: {result.error}"
            assert result.bindings.get("r") == IntValue(0), f"null matched {src_val}"

    def test_null_pattern_in_case_stmt(self) -> None:
        """``null`` pattern works in a case statement (not just case expression)."""
        from agm.agl.eval.values import IntValue

        result = run(
            "let j: json = null\n"
            "var r: int = 0\n"
            "case j of\n"
            "  | null => set r = 1\n"
            "  | _ => set r = 2\n"
        )
        assert result.ok is True, result.error
        assert result.bindings.get("r") == IntValue(1)



# ---------------------------------------------------------------------------
# Scope redefinition (the incremental-session promote/shadow primitive)
# ---------------------------------------------------------------------------


class TestScopeRedefinition:
    """``Scope.define`` of an existing name in the same frame replaces it.

    The incremental REPL session relies on this: promoting a redefinition into
    the persistent root scope must overwrite the prior value AND its mutability,
    not merely add a duplicate.  ``define`` already overwrites
    ``self.bindings[name]``, so no extra API is needed.
    """

    def test_redefine_replaces_value_and_mutability(self) -> None:
        from agm.agl.eval.values import IntValue, TextValue

        scope = Scope(parent=None)
        scope.define("x", IntValue(1), mutable=True, decl_span=_sp())
        # Redefine with a new value and different mutability.
        scope.define("x", TextValue("two"), mutable=False, decl_span=_sp())
        b = scope.lookup("x")
        assert b is not None
        assert b.value == TextValue("two")
        assert b.mutable is False
        # No duplicate binding lingers — exactly one entry for the name.
        assert list(scope.bindings) == ["x"]


# ---------------------------------------------------------------------------
# IfExpr evaluation
# ---------------------------------------------------------------------------


class TestIfExprEval:
    """Evaluator tests for the ``if``-expression (``IfExpr`` AST node).

    The type checker has already proven that every ``if``-expression has an
    ``else`` branch and all conditions are ``bool``, so the interpreter can
    trust those invariants and just evaluate branches left-to-right.
    """

    # --- via source (WorkflowRuntime.run) ---

    def test_if_expr_true_cond_returns_first_branch(self) -> None:
        """``if true => 1 | else => 2`` returns 1."""
        result = run("let x = if true => 1 | else => 2")
        assert result.ok is True
        assert result.bindings["x"] == IntValue(1)

    def test_if_expr_false_cond_falls_through_to_else(self) -> None:
        """When the first cond is false the else branch is taken."""
        result = run("var c = false\nlet x = if c => 1 | else => 2")
        assert result.ok is True
        assert result.bindings["x"] == IntValue(2)

    def test_if_expr_first_true_cond_wins_over_later_true_conds(self) -> None:
        """With multiple conditions, the first true branch is returned."""
        result = run(
            "var a = false\n"
            "var b = true\n"
            "var c = true\n"
            "let x = if a => 10 | b => 20 | c => 30 | else => 99"
        )
        assert result.ok is True
        assert result.bindings["x"] == IntValue(20)

    def test_if_expr_else_branch_taken_when_no_cond_holds(self) -> None:
        """All false conditions → else branch value is returned."""
        result = run(
            "var a = false\n"
            "var b = false\n"
            "let x = if a => 1 | b => 2 | else => 99"
        )
        assert result.ok is True
        assert result.bindings["x"] == IntValue(99)

    def test_if_expr_body_can_reference_outer_binding(self) -> None:
        """Branch body has access to outer scope bindings."""
        result = run("let outer = 42\nlet x = if true => outer | else => 0")
        assert result.ok is True
        assert result.bindings["x"] == IntValue(42)

    def test_if_expr_in_print_position(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``if``-expression works in bare ``print`` position."""
        result = run("print if true => 7 | else => 0")
        assert result.ok is True
        out = capsys.readouterr().out
        assert "7" in out

    def test_if_expr_in_parenthesized_print_position(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``if``-expression works in parenthesized ``print (...)`` position."""
        result = run("print (if true => 7 | else => 0)")
        assert result.ok is True
        out = capsys.readouterr().out
        assert "7" in out

    def test_if_expr_widening_at_binding_boundary(self) -> None:
        """int→decimal widening materializes at the binding boundary, not inside the if-expr.

        ``let x: decimal = if c => 1 | else => 2`` stores a DecimalValue even
        though both branches are int literals, mirroring how case_expr results
        widen at the surrounding boundary.
        """
        import decimal as dec

        result = run("var c = true\nlet x: decimal = if c => 1 | else => 2")
        assert result.ok is True
        assert result.bindings["x"] == DecimalValue(dec.Decimal(1))

    def test_if_expr_in_var_rhs(self) -> None:
        """``if``-expression works in a ``var`` RHS binding."""
        result = run("var x = if false => 10 | else => 20")
        assert result.ok is True
        assert result.bindings["x"] == IntValue(20)

    # --- via AST builder (for branch-scope isolation) ---

    def test_if_expr_returns_chosen_branch_value(self) -> None:
        """The evaluated value equals the body of the branch whose condition is true."""
        # let x = if true => 1 | else => 2
        expr = _if_expr(
            _if_expr_branch(_bool(True), _int(1)),
            _if_expr_else(_int(2)),
        )
        value = _eval_value(expr)
        assert value == IntValue(1)

    def test_if_expr_else_path_via_ast(self) -> None:
        """AST-level: false cond causes else branch to be evaluated."""
        expr = _if_expr(
            _if_expr_branch(_bool(False), _int(1)),
            _if_expr_else(_int(99)),
        )
        assert _eval_value(expr) == IntValue(99)

    def test_if_expr_cond_evaluated_in_outer_scope(self) -> None:
        """Conditions are evaluated in the outer scope (can see outer vars)."""
        # var flag = true
        # let r = if flag => 10 | else => 20
        body = (
            _var("flag", _bool(True), type_ann=_ty("bool")),
            _let("r", _if_expr(_if_expr_branch(_ref("flag"), _int(10)), _if_expr_else(_int(20)))),
        )
        scope = _execute(body)
        assert scope.snapshot()["r"] == IntValue(10)
