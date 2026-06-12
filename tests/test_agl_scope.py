"""Tests for the AgL scope/name-resolution pass (Component 4).

All tests drive real AgL source through ``parse_program`` + ``resolve``,
asserting on user-visible behavior: raised ``AglScopeError`` diagnostics
(message fragment + source line) and observable side-table behavior via the
public ``ResolvedProgram`` API.

Tests deliberately do *not* pin internal implementation details.

Note on M1 parser scope
------------------------
The M1 parser supports: let/var/set/input/pass/print/agent-calls (including
prompt/exec) and string templates with interpolation.  Constructs added in
later milestones (record/enum/type-alias, if/case/do/try) are *not* yet
parseable; tests for those are deferred.  Tests that need those constructs
are marked with ``pytest.mark.skip`` until the parser is extended.
"""

from __future__ import annotations

import pytest

from agm.agl.parser import parse_program
from agm.agl.scope import (
    AglScopeError,
    CallKind,
    ResolvedProgram,
    resolve,
)
from agm.agl.syntax.nodes import (
    BoolLit,
    CallOptions,
    Expr,
    FieldDef,
    IntLit,
    LetDecl,
    PassStmt,
    PrintStmt,
    Program,
    SetStmt,
    Stmt,
    Template,
    VarDecl,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_and_resolve(source: str) -> ResolvedProgram:
    """Parse *source* and run the scope resolution pass."""
    return resolve(parse_program(source))


def reject_scope(source: str) -> AglScopeError:
    """Assert that *source* fails scope resolution and return the error."""
    with pytest.raises(AglScopeError) as exc_info:
        parse_and_resolve(source)
    return exc_info.value


def diag(err: AglScopeError) -> tuple[int, str]:
    """Return (line, message) from an AglScopeError."""
    d = err.to_diagnostic()
    return d.line, d.message


# ---------------------------------------------------------------------------
# Acceptance: simple valid programs (M1-parseable)
# ---------------------------------------------------------------------------


class TestAcceptance:
    def test_simple_let(self) -> None:
        r = parse_and_resolve('let x = "hello"')
        assert r.program is not None

    def test_let_and_print(self) -> None:
        r = parse_and_resolve('let x = "hello"\nprint x')
        assert r.program is not None

    def test_var_and_set(self) -> None:
        r = parse_and_resolve("var n: int = 0\nset n = 1")
        assert r.program is not None

    def test_input_at_root(self) -> None:
        r = parse_and_resolve("input spec")
        assert r.program is not None

    def test_input_with_type(self) -> None:
        r = parse_and_resolve("input spec: text\nprint spec")
        assert r.program is not None

    def test_let_after_input(self) -> None:
        r = parse_and_resolve("input spec\nlet x = spec")
        assert r.program is not None

    def test_agent_call_prompt(self) -> None:
        from agm.agl.syntax.nodes import LetDecl
        r = parse_and_resolve('let x = prompt "Hi"')
        let_stmt = r.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        call_id = let_stmt.value.node_id
        assert r.call_kinds[call_id] == CallKind.default_agent

    def test_agent_call_exec(self) -> None:
        from agm.agl.syntax.nodes import LetDecl
        r = parse_and_resolve('let x = exec "ls"')
        let_stmt = r.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        call_id = let_stmt.value.node_id
        assert r.call_kinds[call_id] == CallKind.shell_exec

    def test_named_agent_call(self) -> None:
        from agm.agl.syntax.nodes import LetDecl
        r = parse_and_resolve('let x = reviewer "Review this"')
        let_stmt = r.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        call_id = let_stmt.value.node_id
        assert r.call_kinds[call_id] == CallKind.agent

    def test_let_with_interpolation(self) -> None:
        r = parse_and_resolve(
            "input name\n"
            'let greeting = "Hello ${name}"\n'
        )
        assert r.program is not None

    def test_multiple_inputs(self) -> None:
        r = parse_and_resolve(
            "input spec\n"
            "input max_severity: int\n"
            "let x = spec\n"
        )
        assert r.program is not None

    def test_pass_stmt(self) -> None:
        r = parse_and_resolve("pass")
        assert r.program is not None

    def test_agent_call_in_print(self) -> None:
        r = parse_and_resolve('print prompt "Q"')
        assert r.program is not None

    def test_let_then_var_set(self) -> None:
        r = parse_and_resolve(
            "let x = \"a\"\n"
            "var y = x\n"
            "set y = x\n"
        )
        assert r.program is not None


# ---------------------------------------------------------------------------
# Rejection: redeclaration in same scope
# ---------------------------------------------------------------------------


class TestRedeclaration:
    def test_redeclare_same_scope_let_var(self) -> None:
        # matches tests/agl/rejections/scope/redeclare_same_scope.agl
        err = reject_scope("let twice = 1\nvar twice = 2")
        line, msg = diag(err)
        assert line == 2
        assert "twice" in msg

    def test_redeclare_same_scope_let_let(self) -> None:
        err = reject_scope("let x = 1\nlet x = 2")
        line, msg = diag(err)
        assert line == 2
        assert "x" in msg

    def test_redeclare_same_scope_var_var(self) -> None:
        err = reject_scope("var a = 1\nvar a = 2")
        line, msg = diag(err)
        assert line == 2
        assert "a" in msg

    def test_redeclare_input_with_let(self) -> None:
        # matches tests/agl/rejections/scope/input_redeclared.agl
        err = reject_scope('input spec\nlet spec = "again"')
        line, msg = diag(err)
        assert line == 2
        assert "spec" in msg

    def test_redeclare_input_with_input(self) -> None:
        err = reject_scope("input x\ninput x")
        line, msg = diag(err)
        assert line == 2
        assert "x" in msg


# ---------------------------------------------------------------------------
# Rejection: set errors
# ---------------------------------------------------------------------------


class TestSetErrors:
    def test_set_on_let(self) -> None:
        # matches tests/agl/rejections/scope/set_on_let.agl
        err = reject_scope("let stable = 1\nset stable = 2")
        line, msg = diag(err)
        assert line == 2
        assert "stable" in msg

    def test_set_on_undeclared(self) -> None:
        # matches tests/agl/rejections/scope/set_undeclared.agl
        err = reject_scope("set ghost = 1")
        line, msg = diag(err)
        assert line == 1
        assert "ghost" in msg

    def test_set_on_input(self) -> None:
        # input bindings are immutable like let.
        err = reject_scope("input spec\nset spec = 2")
        line, msg = diag(err)
        assert line == 2
        assert "spec" in msg


# ---------------------------------------------------------------------------
# Rejection: undefined name reads
# ---------------------------------------------------------------------------


class TestUndefinedRead:
    def test_undefined_print(self) -> None:
        # matches tests/agl/rejections/scope/undefined_read.agl
        err = reject_scope("print missing_thing")
        line, msg = diag(err)
        assert line == 1
        assert "missing_thing" in msg

    def test_undefined_in_assignment(self) -> None:
        err = reject_scope("let x = undeclared")
        line, msg = diag(err)
        assert line == 1
        assert "undeclared" in msg

    def test_undefined_in_interpolation(self) -> None:
        err = reject_scope('let x = "Hi ${ghost}"')
        line, msg = diag(err)
        assert line == 1
        assert "ghost" in msg

    def test_undefined_in_agent_template(self) -> None:
        err = reject_scope('let x = prompt "Hi ${ghost}"')
        line, msg = diag(err)
        assert line == 1
        assert "ghost" in msg


# ---------------------------------------------------------------------------
# Rejection: reserved contextual keywords
# ---------------------------------------------------------------------------


class TestReservedNames:
    def test_reserve_prompt_let(self) -> None:
        # matches tests/agl/rejections/scope/reserve_prompt.agl
        err = reject_scope('let prompt = "not allowed"')
        line, msg = diag(err)
        assert line == 1
        assert "prompt" in msg

    def test_reserve_exec_var(self) -> None:
        # matches tests/agl/rejections/scope/reserve_exec.agl
        err = reject_scope('var exec = "not allowed"')
        line, msg = diag(err)
        assert line == 1
        assert "exec" in msg

    def test_reserve_prompt_input(self) -> None:
        err = reject_scope("input prompt")
        line, msg = diag(err)
        assert line == 1
        assert "prompt" in msg

    def test_reserve_exec_input(self) -> None:
        err = reject_scope("input exec")
        line, msg = diag(err)
        assert line == 1
        assert "exec" in msg


# ---------------------------------------------------------------------------
# Rejection: input not at root (deferred: requires if/do/try parser)
# ---------------------------------------------------------------------------


class TestInputNotRoot:
    def test_input_inside_if(self) -> None:
        # matches tests/agl/rejections/scope/input_not_root.agl
        err = reject_scope(
            "if true =>\n"
            "  input late\n"
            "| else =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "input" in msg.lower()

    def test_input_inside_do(self) -> None:
        err = reject_scope("do[2]\n  input x\nuntil true\n")
        line, msg = diag(err)
        assert line == 2
        assert "input" in msg.lower()

    def test_input_inside_try(self) -> None:
        err = reject_scope("try\n  input x\ncatch _ =>\n  pass\n")
        line, msg = diag(err)
        assert line == 2
        assert "input" in msg.lower()


# ---------------------------------------------------------------------------
# Rejection: scope-escape (deferred: requires do/case/try parser)
# ---------------------------------------------------------------------------


class TestScopeEscape:
    def test_loop_binding_escapes_after_loop(self) -> None:
        # matches tests/agl/rejections/scope/loop_binding_escapes.agl
        err = reject_scope(
            "var n: int = 0\n"
            "do[2]\n"
            "  let probe = n + 1\n"
            "  set n = probe\n"
            "until probe >= 1\n"
            "print probe\n"
        )
        line, msg = diag(err)
        assert line == 6
        assert "probe" in msg

    def test_pattern_var_escapes(self) -> None:
        # matches tests/agl/rejections/scope/pattern_var_escapes.agl
        err = reject_scope(
            "enum R\n"
            "  | Pass\n"
            "  | Fail(issues: list[text])\n"
            'let r: R = Fail(issues: ["a"])\n'
            "case r of\n"
            "  | Fail(issues) => pass\n"
            "  | Pass => pass\n"
            "print issues\n"
        )
        line, msg = diag(err)
        assert line == 8
        assert "issues" in msg

    def test_catch_var_escapes(self) -> None:
        # matches tests/agl/rejections/scope/catch_var_escapes.agl
        err = reject_scope(
            "try\n"
            "  pass\n"
            "catch _ as err =>\n"
            "  pass\n"
            "print err\n"
        )
        line, msg = diag(err)
        assert line == 5
        assert "err" in msg


# ---------------------------------------------------------------------------
# CallKind side table
# ---------------------------------------------------------------------------


class TestCallKinds:
    def test_prompt_is_default_agent(self) -> None:
        r = parse_and_resolve('let x = prompt "Q"')
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        stmt = r.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        assert r.call_kinds[stmt.value.node_id] == CallKind.default_agent

    def test_exec_is_shell_exec(self) -> None:
        r = parse_and_resolve('let x = exec "ls"')
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        stmt = r.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        assert r.call_kinds[stmt.value.node_id] == CallKind.shell_exec

    def test_named_agent_is_agent(self) -> None:
        r = parse_and_resolve('let x = my_agent "Q"')
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        stmt = r.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        assert r.call_kinds[stmt.value.node_id] == CallKind.agent

    def test_exec_in_template_prompt(self) -> None:
        r = parse_and_resolve('input spec\nlet x = prompt "Here: ${spec}"')
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        stmt = r.program.body[1]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        assert r.call_kinds[stmt.value.node_id] == CallKind.default_agent


# ---------------------------------------------------------------------------
# Resolution side table: VarRef and SetStmt
# ---------------------------------------------------------------------------


class TestResolution:
    def test_varref_resolves_to_let(self) -> None:
        r = parse_and_resolve('let x = "v"\nprint x')
        from agm.agl.syntax.nodes import PrintStmt, VarRef

        print_stmt = r.program.body[1]
        assert isinstance(print_stmt, PrintStmt)
        assert isinstance(print_stmt.value, VarRef)
        ref = r.resolution[print_stmt.value.node_id]
        assert ref.name == "x"
        assert not ref.mutable

    def test_set_resolves_to_var(self) -> None:
        r = parse_and_resolve("var n = 0\nset n = 1")
        from agm.agl.syntax.nodes import SetStmt

        set_stmt = r.program.body[1]
        assert isinstance(set_stmt, SetStmt)
        ref = r.resolution[set_stmt.node_id]
        assert ref.name == "n"
        assert ref.mutable

    def test_input_binding_is_immutable(self) -> None:
        r = parse_and_resolve("input spec\nprint spec")
        from agm.agl.syntax.nodes import PrintStmt, VarRef

        print_stmt = r.program.body[1]
        assert isinstance(print_stmt, PrintStmt)
        assert isinstance(print_stmt.value, VarRef)
        ref = r.resolution[print_stmt.value.node_id]
        assert ref.name == "spec"
        assert not ref.mutable

    def test_interp_varref_resolved(self) -> None:
        r = parse_and_resolve('input name\nlet q = prompt "Hello ${name}"')
        from agm.agl.syntax.nodes import AgentCall, InterpSegment, LetDecl, Template, VarRef

        stmt = r.program.body[1]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        tmpl = stmt.value.template
        assert isinstance(tmpl, Template)
        interp = tmpl.segments[1]
        assert isinstance(interp, InterpSegment)
        assert isinstance(interp.expr, VarRef)
        ref = r.resolution[interp.expr.node_id]
        assert ref.name == "name"


# ---------------------------------------------------------------------------
# Direct AST construction tests (covers M2/M3/M4 code paths)
#
# These tests construct AST nodes programmatically to exercise scope-resolution
# code paths for constructs (do/if/case/try, operators, patterns, etc.) that
# are not yet parseable by the M1 parser.  They verify the resolver's behavior
# through the same public ResolvedProgram API.
# ---------------------------------------------------------------------------


def _sp(line: int = 1) -> "SourceSpan":
    """Make a minimal SourceSpan at the given line."""
    from agm.agl.syntax.spans import SourceSpan

    return SourceSpan(
        start_line=line, start_col=1, end_line=line, end_col=2,
        start_offset=0, end_offset=1,
    )


_NID = 0


def _nid() -> int:
    global _NID
    _NID += 1
    return _NID


def _make_options() -> "CallOptions":
    from agm.agl.syntax.nodes import CallOptions

    return CallOptions(format=None, strict_json=None, parse_policy=None, span=_sp(), node_id=_nid())


def _make_template(text: str = "Q") -> "Template":
    from agm.agl.syntax.nodes import Template, TextSegment

    return Template(
        segments=(TextSegment(text=text, span=_sp(), node_id=_nid()),),
        span=_sp(),
        node_id=_nid(),
    )


def _make_intlit(val: int = 0) -> "IntLit":
    from agm.agl.syntax.nodes import IntLit

    return IntLit(value=val, span=_sp(), node_id=_nid())


def _make_boollit(val: bool = True) -> "BoolLit":
    from agm.agl.syntax.nodes import BoolLit

    return BoolLit(value=val, span=_sp(), node_id=_nid())


def _make_varref(name: str) -> "VarRef":
    from agm.agl.syntax.nodes import VarRef

    return VarRef(name=name, span=_sp(), node_id=_nid())


def _make_program(*stmts: "Stmt") -> "Program":
    from agm.agl.syntax.nodes import Program

    return Program(body=tuple(stmts), span=_sp(), node_id=_nid())


def _make_let(name: str, value: "Expr", line: int = 1) -> "LetDecl":
    from agm.agl.syntax.nodes import LetDecl

    return LetDecl(name=name, type_ann=None, value=value, span=_sp(line), node_id=_nid())


def _make_var(name: str, value: "Expr", line: int = 1) -> "VarDecl":
    from agm.agl.syntax.nodes import VarDecl

    return VarDecl(name=name, type_ann=None, value=value, span=_sp(line), node_id=_nid())


def _make_set(target: str, value: "Expr", line: int = 1) -> "SetStmt":
    from agm.agl.syntax.nodes import SetStmt

    return SetStmt(target=target, value=value, span=_sp(line), node_id=_nid())


def _make_pass(line: int = 1) -> "PassStmt":
    from agm.agl.syntax.nodes import PassStmt

    return PassStmt(span=_sp(line), node_id=_nid())


def _make_print(value: "Expr", line: int = 1) -> "PrintStmt":
    from agm.agl.syntax.nodes import PrintStmt

    return PrintStmt(value=value, span=_sp(line), node_id=_nid())


def resolve_program(*stmts: "Stmt") -> ResolvedProgram:
    """Construct and resolve a Program from the given statements."""
    return resolve(_make_program(*stmts))


def reject_program(*stmts: Stmt) -> AglScopeError:
    """Assert that a program built from the given statements fails scope resolution."""
    with pytest.raises(AglScopeError) as exc_info:
        resolve_program(*stmts)
    return exc_info.value


class TestScopeViaSourceParseable:
    """F11: constructs that now parse from source — type declarations, expression
    statements, and ``raise`` — exercised through the real parser + scope pass
    instead of hand-built AST."""

    def test_record_def_ignored_by_scope(self) -> None:
        r = parse_and_resolve("record P\n  n: int\n")
        assert r.program is not None

    def test_enum_def_ignored_by_scope(self) -> None:
        r = parse_and_resolve("enum E\n  | A\n  | B\n")
        assert r.program is not None

    def test_type_alias_ignored_by_scope(self) -> None:
        r = parse_and_resolve("type MyText = text\n")
        assert r.program is not None

    def test_expr_stmt(self) -> None:
        r = parse_and_resolve("1\n")
        assert r.program is not None

    def test_expr_stmt_with_varref(self) -> None:
        r = parse_and_resolve("let x = 1\nx\n")
        assert r.program is not None

    def test_raise_stmt(self) -> None:
        # Scope resolves a ``raise`` operand like any expression; the operand's
        # type is the typechecker's concern (F1), not the scope pass.
        r = parse_and_resolve("raise 1\n")
        assert r.program is not None

    def test_raise_undefined_name_rejected(self) -> None:
        # A ``raise`` of an undefined name is a scope error (the operand is
        # resolved like any other expression).
        err = reject_scope("raise nope\n")
        assert "nope" in err.to_diagnostic().message


class TestScopeViaAstConstruction:
    """Scope-resolution behavior driven by direct AST construction.

    These cases build AST nodes to exercise scope-pass code paths in isolation
    (precise control over nested-scope visibility, cross-scope leaks, and the
    ``input``-not-at-root rule).  The constructs themselves now all parse from
    source — the previously-stale "not reachable through the M1 parser"
    justification has been removed.  The genuinely-parseable standalone cases
    (type declarations, expression statements, and ``raise``) are exercised as
    source tests in :class:`TestScopeViaSourceParseable` below; what remains here
    keeps the AST form only because hand-built nodes are the clearest way to pin
    a specific resolution path.
    """

    # --- DoUntil scope ---

    def test_do_until_body_and_condition_resolved(self) -> None:
        from agm.agl.syntax.nodes import DoUntil

        var_stmt = _make_var("n", _make_intlit(0))
        let_in_body = _make_let("probe", _make_varref("n"))
        do_stmt = DoUntil(
            limit=5,
            body=(let_in_body,),
            condition=_make_varref("probe"),  # probe visible in condition
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(var_stmt, do_stmt)
        assert r.program is not None

    def test_do_until_body_not_visible_after(self) -> None:
        from agm.agl.syntax.nodes import DoUntil

        let_in_body = _make_let("inner", _make_intlit(1))
        do_stmt = DoUntil(
            limit=5,
            body=(let_in_body,),
            condition=_make_boollit(True),
            span=_sp(),
            node_id=_nid(),
        )
        # Using `inner` after the loop should fail.
        print_after = _make_print(_make_varref("inner"), line=3)
        err = reject_program(do_stmt, print_after)
        assert "inner" in err.to_diagnostic().message

    def test_do_until_input_not_root_error(self) -> None:
        from agm.agl.syntax.nodes import DoUntil, InputDecl

        input_in_body = InputDecl(name="x", annotation=None, span=_sp(2), node_id=_nid())
        do_stmt = DoUntil(
            limit=5,
            body=(input_in_body,),
            condition=_make_boollit(True),
            span=_sp(),
            node_id=_nid(),
        )
        err = reject_program(do_stmt)
        assert "input" in err.to_diagnostic().message.lower()

    # --- IfStmt scope ---

    def test_if_stmt_branch_scope(self) -> None:
        from agm.agl.syntax.nodes import IfBranch, IfStmt

        let_in_branch = _make_let("inner", _make_intlit(1))
        branch = IfBranch(
            cond=_make_boollit(True),
            body=(let_in_branch,),
            span=_sp(),
            node_id=_nid(),
        )
        if_stmt = IfStmt(branches=(branch,), span=_sp(), node_id=_nid())
        r = resolve_program(if_stmt)
        assert r.program is not None

    def test_if_stmt_else_branch(self) -> None:
        from agm.agl.syntax.nodes import ELSE, IfBranch, IfStmt

        branch_if = IfBranch(
            cond=_make_boollit(True),
            body=(_make_pass(),),
            span=_sp(),
            node_id=_nid(),
        )
        branch_else = IfBranch(
            cond=ELSE,
            body=(_make_pass(),),
            span=_sp(),
            node_id=_nid(),
        )
        if_stmt = IfStmt(branches=(branch_if, branch_else), span=_sp(), node_id=_nid())
        r = resolve_program(if_stmt)
        assert r.program is not None

    def test_if_stmt_inner_var_not_visible_outside(self) -> None:
        from agm.agl.syntax.nodes import IfBranch, IfStmt

        let_in = _make_let("inner", _make_intlit(1))
        branch = IfBranch(cond=_make_boollit(True), body=(let_in,), span=_sp(), node_id=_nid())
        if_stmt = IfStmt(branches=(branch,), span=_sp(), node_id=_nid())
        print_stmt = _make_print(_make_varref("inner"), line=3)
        err = reject_program(if_stmt, print_stmt)
        assert "inner" in err.to_diagnostic().message

    def test_if_stmt_input_not_root_error(self) -> None:
        from agm.agl.syntax.nodes import IfBranch, IfStmt, InputDecl

        input_decl = InputDecl(name="x", annotation=None, span=_sp(2), node_id=_nid())
        branch = IfBranch(
            cond=_make_boollit(True),
            body=(input_decl,),
            span=_sp(),
            node_id=_nid(),
        )
        if_stmt = IfStmt(branches=(branch,), span=_sp(), node_id=_nid())
        err = reject_program(if_stmt)
        assert "input" in err.to_diagnostic().message.lower()

    # --- CaseStmt scope ---

    def test_case_stmt_wildcard_pattern(self) -> None:
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, WildcardPattern

        let_x = _make_let("x", _make_intlit(1))
        branch = CaseStmtBranch(
            pattern=WildcardPattern(span=_sp(), node_id=_nid()),
            body=(_make_pass(),),
            span=_sp(),
            node_id=_nid(),
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_x, case_stmt)
        assert r.program is not None

    def test_case_stmt_var_pattern_binds(self) -> None:
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, VarPattern

        let_x = _make_let("x", _make_intlit(1))
        pattern_var = VarPattern(name="matched", span=_sp(), node_id=_nid())
        branch = CaseStmtBranch(
            pattern=pattern_var,
            body=(_make_print(_make_varref("matched")),),
            span=_sp(),
            node_id=_nid(),
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_x, case_stmt)
        assert r.program is not None

    def test_case_stmt_literal_pattern(self) -> None:
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, LiteralPattern

        let_x = _make_let("x", _make_intlit(1))
        pattern = LiteralPattern(literal=_make_intlit(1), span=_sp(), node_id=_nid())
        branch = CaseStmtBranch(
            pattern=pattern,
            body=(_make_pass(),),
            span=_sp(),
            node_id=_nid(),
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_x, case_stmt)
        assert r.program is not None

    def test_case_stmt_constructor_pattern_with_field(self) -> None:
        from agm.agl.syntax.nodes import (
            CaseStmt,
            CaseStmtBranch,
            ConstructorPattern,
            PatternField,
            VarPattern,
        )

        let_x = _make_let("x", _make_intlit(1))
        sub_pattern = VarPattern(name="issues", span=_sp(), node_id=_nid())
        pf = PatternField(name="issues", pattern=sub_pattern, span=_sp(), node_id=_nid())
        ctor_pattern = ConstructorPattern(
            qualifier=None, name="Fail", fields=(pf,), span=_sp(), node_id=_nid()
        )
        branch = CaseStmtBranch(
            pattern=ctor_pattern,
            body=(_make_print(_make_varref("issues")),),
            span=_sp(),
            node_id=_nid(),
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_x, case_stmt)
        assert r.program is not None

    def test_case_stmt_pattern_var_escapes(self) -> None:
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, VarPattern

        let_x = _make_let("x", _make_intlit(1))
        pv = VarPattern(name="inner", span=_sp(), node_id=_nid())
        branch = CaseStmtBranch(
            pattern=pv, body=(_make_pass(),), span=_sp(), node_id=_nid()
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        print_after = _make_print(_make_varref("inner"), line=4)
        err = reject_program(let_x, case_stmt, print_after)
        assert "inner" in err.to_diagnostic().message

    def test_case_stmt_input_not_root_error(self) -> None:
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, InputDecl, WildcardPattern

        let_x = _make_let("x", _make_intlit(1))
        input_decl = InputDecl(name="y", annotation=None, span=_sp(3), node_id=_nid())
        branch = CaseStmtBranch(
            pattern=WildcardPattern(span=_sp(), node_id=_nid()),
            body=(input_decl,),
            span=_sp(),
            node_id=_nid(),
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        err = reject_program(let_x, case_stmt)
        assert "input" in err.to_diagnostic().message.lower()

    # --- TryCatch scope ---

    def test_try_catch_wildcard(self) -> None:
        from agm.agl.syntax.nodes import CatchClause, TryCatch

        clause = CatchClause(
            exc_type=None, binding=None, body=(_make_pass(),), span=_sp(), node_id=_nid()
        )
        try_stmt = TryCatch(body=(_make_pass(),), handlers=(clause,), span=_sp(), node_id=_nid())
        r = resolve_program(try_stmt)
        assert r.program is not None

    def test_try_catch_with_binder(self) -> None:
        from agm.agl.syntax.nodes import CatchClause, TryCatch

        clause = CatchClause(
            exc_type="AgentCallError",
            binding="err",
            body=(_make_print(_make_varref("err")),),
            span=_sp(),
            node_id=_nid(),
        )
        try_stmt = TryCatch(body=(_make_pass(),), handlers=(clause,), span=_sp(), node_id=_nid())
        r = resolve_program(try_stmt)
        assert r.program is not None

    def test_try_catch_binder_not_visible_outside(self) -> None:
        from agm.agl.syntax.nodes import CatchClause, TryCatch

        clause = CatchClause(
            exc_type=None,
            binding="err",
            body=(_make_pass(),),
            span=_sp(),
            node_id=_nid(),
        )
        try_stmt = TryCatch(body=(_make_pass(),), handlers=(clause,), span=_sp(), node_id=_nid())
        print_after = _make_print(_make_varref("err"), line=3)
        err = reject_program(try_stmt, print_after)
        assert "err" in err.to_diagnostic().message

    def test_try_catch_input_not_root_error(self) -> None:
        from agm.agl.syntax.nodes import CatchClause, InputDecl, TryCatch

        input_decl = InputDecl(name="x", annotation=None, span=_sp(2), node_id=_nid())
        clause = CatchClause(
            exc_type=None, binding=None, body=(_make_pass(),), span=_sp(), node_id=_nid()
        )
        try_stmt = TryCatch(
            body=(input_decl,), handlers=(clause,), span=_sp(), node_id=_nid()
        )
        err = reject_program(try_stmt)
        assert "input" in err.to_diagnostic().message.lower()

    # --- Expression resolution: operators, constructors, lists, dicts ---

    def test_field_access_on_varref(self) -> None:
        from agm.agl.syntax.nodes import FieldAccess

        let_x = _make_let("x", _make_intlit(1))
        field_expr = FieldAccess(obj=_make_varref("x"), field="f", span=_sp(), node_id=_nid())
        print_stmt = _make_print(field_expr)
        r = resolve_program(let_x, print_stmt)
        assert r.program is not None

    def test_binary_op_resolved(self) -> None:
        from agm.agl.syntax.nodes import BinaryOp, BinOp

        let_x = _make_let("x", _make_intlit(1))
        let_y = _make_let("y", _make_intlit(2))
        binop = BinaryOp(
            op=BinOp.ADD,
            left=_make_varref("x"),
            right=_make_varref("y"),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, let_y, _make_print(binop))
        assert r.program is not None

    def test_unary_not_resolved(self) -> None:
        from agm.agl.syntax.nodes import UnaryNot

        let_b = _make_let("b", _make_boollit(True))
        expr = UnaryNot(operand=_make_varref("b"), span=_sp(), node_id=_nid())
        r = resolve_program(let_b, _make_print(expr))
        assert r.program is not None

    def test_unary_neg_resolved(self) -> None:
        from agm.agl.syntax.nodes import UnaryNeg

        let_n = _make_let("n", _make_intlit(1))
        expr = UnaryNeg(operand=_make_varref("n"), span=_sp(), node_id=_nid())
        r = resolve_program(let_n, _make_print(expr))
        assert r.program is not None

    def test_is_test_resolved(self) -> None:
        from agm.agl.syntax.nodes import IsTest

        let_x = _make_let("x", _make_intlit(1))
        expr = IsTest(
            expr=_make_varref("x"),
            qualifier=None,
            variant="Pass",
            negated=False,
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, _make_print(expr))
        assert r.program is not None

    def test_constructor_args_resolved(self) -> None:
        from agm.agl.syntax.nodes import Constructor, NamedArg

        let_n = _make_let("n", _make_intlit(5))
        arg = NamedArg(name="n", value=_make_varref("n"), span=_sp(), node_id=_nid())
        ctor = Constructor(
            qualifier=None, name="Point", args=(arg,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_n, _make_print(ctor))
        assert r.program is not None

    def test_list_lit_resolved(self) -> None:
        from agm.agl.syntax.nodes import ListLit

        let_x = _make_let("x", _make_intlit(1))
        lst = ListLit(elements=(_make_varref("x"),), span=_sp(), node_id=_nid())
        r = resolve_program(let_x, _make_print(lst))
        assert r.program is not None

    def test_dict_lit_resolved(self) -> None:
        from agm.agl.syntax.nodes import DictEntry, DictLit, StringLit

        let_x = _make_let("x", _make_intlit(1))
        key = StringLit(value="a", span=_sp(), node_id=_nid())
        entry = DictEntry(key=key, value=_make_varref("x"), span=_sp(), node_id=_nid())
        dlit = DictLit(entries=(entry,), span=_sp(), node_id=_nid())
        r = resolve_program(let_x, _make_print(dlit))
        assert r.program is not None

    def test_case_expr_in_let(self) -> None:
        from agm.agl.syntax.nodes import CaseExpr, CaseExprBranch, VarPattern

        let_x = _make_let("x", _make_intlit(1))
        var_p = VarPattern(name="v", span=_sp(), node_id=_nid())
        branch = CaseExprBranch(
            pattern=var_p,
            body=_make_varref("v"),
            span=_sp(),
            node_id=_nid(),
        )
        case_expr = CaseExpr(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(),
            node_id=_nid(),
        )
        let_y = _make_let("y", case_expr)
        r = resolve_program(let_x, let_y)
        assert r.program is not None

    def test_case_expr_wildcard_branch(self) -> None:
        from agm.agl.syntax.nodes import CaseExpr, CaseExprBranch, WildcardPattern

        let_x = _make_let("x", _make_intlit(1))
        branch = CaseExprBranch(
            pattern=WildcardPattern(span=_sp(), node_id=_nid()),
            body=_make_intlit(0),
            span=_sp(),
            node_id=_nid(),
        )
        case_expr = CaseExpr(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(),
            node_id=_nid(),
        )
        let_y = _make_let("y", case_expr)
        r = resolve_program(let_x, let_y)
        assert r.program is not None

    def test_case_expr_literal_pattern_branch(self) -> None:
        from agm.agl.syntax.nodes import CaseExpr, CaseExprBranch, LiteralPattern

        let_x = _make_let("x", _make_intlit(1))
        branch = CaseExprBranch(
            pattern=LiteralPattern(literal=_make_intlit(1), span=_sp(), node_id=_nid()),
            body=_make_intlit(99),
            span=_sp(),
            node_id=_nid(),
        )
        case_expr = CaseExpr(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, _make_let("y", case_expr))
        assert r.program is not None

    def test_case_expr_ctor_pattern_with_field_var(self) -> None:
        from agm.agl.syntax.nodes import (
            CaseExpr,
            CaseExprBranch,
            ConstructorPattern,
            PatternField,
            VarPattern,
        )

        let_x = _make_let("x", _make_intlit(1))
        sub_pv = VarPattern(name="issues", span=_sp(), node_id=_nid())
        pf = PatternField(name="issues", pattern=sub_pv, span=_sp(), node_id=_nid())
        ctor_p = ConstructorPattern(
            qualifier=None, name="Fail", fields=(pf,), span=_sp(), node_id=_nid()
        )
        branch = CaseExprBranch(
            pattern=ctor_p,
            body=_make_varref("issues"),
            span=_sp(),
            node_id=_nid(),
        )
        case_expr = CaseExpr(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_x, _make_let("y", case_expr))
        assert r.program is not None


def _make_field(name: str, type_expr: "TypeExpr") -> FieldDef:
    return FieldDef(name=name, type_expr=type_expr, span=_sp(), node_id=_nid())
