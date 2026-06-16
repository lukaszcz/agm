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
        r = parse_and_resolve('agent reviewer\nlet x = reviewer "Review this"')
        let_stmt = r.program.body[1]
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

    def test_set_on_let_names_let(self) -> None:
        # A genuine let binding still reports the 'let' phrasing (F8).
        err = reject_scope("let stable = 1\nset stable = 2")
        _, msg = diag(err)
        assert "let" in msg
        assert "immutable" in msg

    def test_set_on_input_names_input_not_let(self) -> None:
        # An input binding must NOT be mislabelled as declared with 'let' (F8).
        err = reject_scope("input spec\nset spec = 2")
        _, msg = diag(err)
        assert "input" in msg
        assert "declared with 'let'" not in msg

    def test_set_on_catch_binder_names_catch_not_let(self) -> None:
        # Mutating a catch binder must name the catch binder, not 'let' (F8).
        err = reject_scope(
            "try\n"
            "  pass\n"
            "catch _ as err =>\n"
            "  set err = 1\n"
        )
        line, msg = diag(err)
        assert line == 4
        assert "catch binder" in msg
        assert "declared with 'let'" not in msg

    def test_set_on_pattern_binding_names_pattern_not_let(self) -> None:
        # Mutating a pattern variable must name the pattern binding, not 'let' (F8).
        err = reject_scope(
            "let v = 1\n"
            "case v of\n"
            "  | n =>\n"
            "    set n = 2\n"
        )
        line, msg = diag(err)
        assert line == 4
        assert "pattern binding" in msg
        assert "declared with 'let'" not in msg


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
        r = parse_and_resolve('agent my_agent\nlet x = my_agent "Q"')
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        stmt = r.program.body[1]
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


# ---------------------------------------------------------------------------
# Task 1: type/record/enum declarations rejected outside the program root
# ---------------------------------------------------------------------------


class TestTypeDeclarationNotAtRoot:
    """Task 1: nested record/enum/type declarations must be rejected at resolver time."""

    # --- record ---

    def test_record_inside_try_body_rejected(self) -> None:
        err = reject_scope(
            "try\n"
            "  record R\n"
            "    n: int\n"
            "catch _ =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_record_inside_do_body_rejected(self) -> None:
        err = reject_scope(
            "do[2]\n"
            "  record R\n"
            "    n: int\n"
            "until true\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_record_inside_if_branch_rejected(self) -> None:
        err = reject_scope(
            "if true =>\n"
            "  record R\n"
            "    n: int\n"
            "| else =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    # --- enum ---

    def test_enum_inside_try_body_rejected(self) -> None:
        err = reject_scope(
            "try\n"
            "  enum E\n"
            "    | A\n"
            "    | B\n"
            "catch _ =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_enum_inside_do_body_rejected(self) -> None:
        err = reject_scope(
            "do[2]\n"
            "  enum E\n"
            "    | A\n"
            "until true\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_enum_inside_if_branch_rejected(self) -> None:
        err = reject_scope(
            "if true =>\n"
            "  enum E\n"
            "    | A\n"
            "| else =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    # --- type alias ---

    def test_type_alias_inside_try_body_rejected(self) -> None:
        err = reject_scope(
            "try\n"
            "  type T = text\n"
            "catch _ =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_type_alias_inside_do_body_rejected(self) -> None:
        err = reject_scope(
            "do[2]\n"
            "  type T = int\n"
            "until true\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_type_alias_inside_if_branch_rejected(self) -> None:
        err = reject_scope(
            "if true =>\n"
            "  type T = bool\n"
            "| else =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    # --- inline (semicolon) form of do body ---

    def test_type_alias_inline_do_body_rejected(self) -> None:
        """do[2] type T = int until true — inline form must also be rejected."""
        err = reject_scope("do[2] type T = int until true\n")
        line, msg = diag(err)
        assert line == 1
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_enum_inline_do_body_rejected(self) -> None:
        """do[2] enum E | A until true — inline form must also be rejected."""
        err = reject_scope("do[2] enum E | A until true\n")
        line, msg = diag(err)
        assert line == 1
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_input_inline_do_body_rejected(self) -> None:
        """do[2] input x until true — InputDecl in inline body must be rejected."""
        err = reject_scope("do[2] input x until true\n")
        line, msg = diag(err)
        assert line == 1
        assert "input" in msg.lower()

    # --- top-level still accepted ---

    def test_record_at_root_accepted(self) -> None:
        r = parse_and_resolve("record P\n  n: int\n")
        assert r.program is not None

    def test_enum_at_root_accepted(self) -> None:
        r = parse_and_resolve("enum E\n  | A\n  | B\n")
        assert r.program is not None

    def test_type_alias_at_root_accepted(self) -> None:
        r = parse_and_resolve("type MyText = text\n")
        assert r.program is not None


# ---------------------------------------------------------------------------
# Task 2: duplicate variable bindings within one pattern rejected
# ---------------------------------------------------------------------------


class TestDuplicatePatternBindings:
    """Task 2: a name bound twice in one pattern is an error (§9 rule 1)."""

    def test_duplicate_var_pattern_name_in_constructor(self) -> None:
        # case f of | Fail(reason: x, hint: x) => print x
        err = reject_scope(
            "enum Failure\n"
            "  | Fail(reason: text, hint: text)\n"
            "let f: Failure = Fail(reason: \"r\", hint: \"h\")\n"
            "case f of\n"
            "  | Fail(reason: x, hint: x) => print x\n"
        )
        _, msg = diag(err)
        assert "x" in msg

    def test_duplicate_var_pattern_name_in_constructor_ast(self) -> None:
        """Direct AST construction: same name bound twice in one ConstructorPattern."""
        from agm.agl.syntax.nodes import (
            CaseStmt,
            CaseStmtBranch,
            ConstructorPattern,
            PatternField,
            VarPattern,
        )

        let_x = _make_let("x", _make_intlit(1))
        sub1 = VarPattern(name="dup", span=_sp(5), node_id=_nid())
        sub2 = VarPattern(name="dup", span=_sp(5), node_id=_nid())
        pf1 = PatternField(name="a", pattern=sub1, span=_sp(5), node_id=_nid())
        pf2 = PatternField(name="b", pattern=sub2, span=_sp(5), node_id=_nid())
        ctor_pat = ConstructorPattern(
            qualifier=None, name="Pair", fields=(pf1, pf2), span=_sp(5), node_id=_nid()
        )
        branch = CaseStmtBranch(
            pattern=ctor_pat, body=(_make_pass(),), span=_sp(5), node_id=_nid()
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(5), node_id=_nid()
        )
        err = reject_program(let_x, case_stmt)
        assert "dup" in err.to_diagnostic().message

    def test_unique_var_pattern_names_in_constructor_accepted(self) -> None:
        """Two different names in a constructor pattern — should pass."""
        from agm.agl.syntax.nodes import (
            CaseStmt,
            CaseStmtBranch,
            ConstructorPattern,
            PatternField,
            VarPattern,
        )

        let_x = _make_let("x", _make_intlit(1))
        sub1 = VarPattern(name="a", span=_sp(), node_id=_nid())
        sub2 = VarPattern(name="b", span=_sp(), node_id=_nid())
        pf1 = PatternField(name="f1", pattern=sub1, span=_sp(), node_id=_nid())
        pf2 = PatternField(name="f2", pattern=sub2, span=_sp(), node_id=_nid())
        ctor_pat = ConstructorPattern(
            qualifier=None, name="Pair", fields=(pf1, pf2), span=_sp(), node_id=_nid()
        )
        branch = CaseStmtBranch(
            pattern=ctor_pat,
            body=(_make_print(_make_varref("a")),),
            span=_sp(),
            node_id=_nid(),
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_x, case_stmt)
        assert r.program is not None

    def test_pattern_var_shadows_outer_scope_accepted(self) -> None:
        """A pattern variable shadowing an outer name is OK (shadowing rule)."""
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, VarPattern

        let_x = _make_let("x", _make_intlit(1))
        let_outer = _make_let("v", _make_intlit(99))
        # pattern `v` shadows outer `v`
        pv = VarPattern(name="v", span=_sp(), node_id=_nid())
        branch = CaseStmtBranch(
            pattern=pv, body=(_make_print(_make_varref("v")),), span=_sp(), node_id=_nid()
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_outer, let_x, case_stmt)
        assert r.program is not None


# ---------------------------------------------------------------------------
# Task 3: prompt/exec reserved as pattern-variable and catch-binder names
# ---------------------------------------------------------------------------


class TestReservedNamesInPatternAndCatch:
    """Task 3: prompt/exec are reserved; extend check to pattern binders and catch binders."""

    # --- catch binder ---

    def test_exec_as_catch_binder_rejected(self) -> None:
        err = reject_scope(
            "try\n"
            "  pass\n"
            "catch _ as exec =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 3
        assert "exec" in msg
        assert "reserved" in msg.lower() or "contextual" in msg.lower()

    def test_prompt_as_catch_binder_rejected(self) -> None:
        err = reject_scope(
            "try\n"
            "  pass\n"
            "catch _ as prompt =>\n"
            "  pass\n"
        )
        line, msg = diag(err)
        assert line == 3
        assert "prompt" in msg
        assert "reserved" in msg.lower() or "contextual" in msg.lower()

    def test_normal_catch_binder_accepted(self) -> None:
        r = parse_and_resolve(
            "try\n"
            "  pass\n"
            "catch _ as err =>\n"
            "  pass\n"
        )
        assert r.program is not None

    # --- pattern variable ---

    def test_prompt_as_var_pattern_rejected(self) -> None:
        # case x of | prompt => pass
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, VarPattern

        let_x = _make_let("x", _make_intlit(1))
        pv = VarPattern(name="prompt", span=_sp(3), node_id=_nid())
        branch = CaseStmtBranch(
            pattern=pv, body=(_make_pass(),), span=_sp(3), node_id=_nid()
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(3), node_id=_nid()
        )
        err = reject_program(let_x, case_stmt)
        msg = err.to_diagnostic().message
        assert "prompt" in msg
        assert "reserved" in msg.lower() or "contextual" in msg.lower()

    def test_exec_as_var_pattern_rejected(self) -> None:
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, VarPattern

        let_x = _make_let("x", _make_intlit(1))
        pv = VarPattern(name="exec", span=_sp(3), node_id=_nid())
        branch = CaseStmtBranch(
            pattern=pv, body=(_make_pass(),), span=_sp(3), node_id=_nid()
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(3), node_id=_nid()
        )
        err = reject_program(let_x, case_stmt)
        msg = err.to_diagnostic().message
        assert "exec" in msg
        assert "reserved" in msg.lower() or "contextual" in msg.lower()

    # --- pattern field binder ---

    def test_exec_as_pattern_field_binder_rejected(self) -> None:
        from agm.agl.syntax.nodes import (
            CaseStmt,
            CaseStmtBranch,
            ConstructorPattern,
            PatternField,
            VarPattern,
        )

        let_x = _make_let("x", _make_intlit(1))
        pv = VarPattern(name="exec", span=_sp(3), node_id=_nid())
        pf = PatternField(name="field", pattern=pv, span=_sp(3), node_id=_nid())
        ctor_pat = ConstructorPattern(
            qualifier=None, name="Ctor", fields=(pf,), span=_sp(3), node_id=_nid()
        )
        branch = CaseStmtBranch(
            pattern=ctor_pat, body=(_make_pass(),), span=_sp(3), node_id=_nid()
        )
        case_stmt = CaseStmt(
            subject=_make_varref("x"), branches=(branch,), span=_sp(3), node_id=_nid()
        )
        err = reject_program(let_x, case_stmt)
        msg = err.to_diagnostic().message
        assert "exec" in msg
        assert "reserved" in msg.lower() or "contextual" in msg.lower()


# ---------------------------------------------------------------------------
# parent_scope seam (incremental REPL sessions)
# ---------------------------------------------------------------------------


class TestParentScopeSeam:
    """``resolve(program, parent_scope=...)`` resolves into a session scope."""

    def test_default_none_is_standalone(self) -> None:
        """Without a parent, an undefined name is still an error."""
        err = reject_scope("print x")
        assert "not defined" in diag(err)[1]

    def test_reference_resolves_into_parent(self) -> None:
        """A VarRef to a parent-scope binding resolves through the parent."""
        session = parse_and_resolve("let x = 1")
        entry = resolve(parse_program("print x"), parent_scope=session.root_scope)
        # The print's VarRef resolved to the session's let binding.
        print_stmt = entry.program.body[0]
        assert isinstance(print_stmt, PrintStmt)
        ref = entry.resolution[print_stmt.value.node_id]
        assert ref.name == "x"

    def test_redeclaring_parent_name_shadows_without_error(self) -> None:
        """Redeclaring a parent-visible name in the entry shadows (no error)."""
        session = parse_and_resolve("let x = 1")
        # Must not raise a duplicate-declaration error.
        entry = resolve(parse_program("let x = 2"), parent_scope=session.root_scope)
        let_stmt = entry.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        # The new binding lives in the entry's own root scope.
        assert "x" in entry.root_scope.bindings
        assert entry.root_scope.bindings["x"].decl_node_id == let_stmt.node_id

    def test_set_on_parent_mutable_resolves(self) -> None:
        """``set`` of a parent mutable (var) binding resolves through the parent."""
        session = parse_and_resolve("var n: int = 0")
        entry = resolve(parse_program("set n = 1"), parent_scope=session.root_scope)
        set_stmt = entry.program.body[0]
        assert isinstance(set_stmt, SetStmt)
        ref = entry.resolution[set_stmt.node_id]
        assert ref.name == "n"
        assert ref.mutable is True

    def test_set_on_parent_immutable_still_errors(self) -> None:
        """``set`` of a parent immutable (let) binding is still rejected."""
        session = parse_and_resolve("let k = 1")
        with pytest.raises(AglScopeError) as exc_info:
            resolve(parse_program("set k = 2"), parent_scope=session.root_scope)
        assert "Cannot assign" in str(exc_info.value)

    def test_set_on_parent_input_still_errors(self) -> None:
        """``set`` of a parent ``input`` binding is still rejected."""
        session = parse_and_resolve("input spec")
        with pytest.raises(AglScopeError) as exc_info:
            resolve(parse_program("set spec = 2"), parent_scope=session.root_scope)
        assert "Cannot assign" in str(exc_info.value)

    def test_set_across_entries_resolves_and_typechecks(self) -> None:
        """A ``set`` in entry 2 resolves and type-checks against entry 1's ``var``.

        Drives the realistic combined session path: entry 1 declares ``var v = 0``
        and is both resolved and checked; entry 2's ``set v = 5`` is resolved with
        entry 1's root scope as the parent and type-checked with entry 1's
        ``type_env`` as the seed.  Both passes must succeed (the ``set`` binds to
        the seeded mutable binding and its declared ``int`` type).
        """
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.typecheck import check
        from agm.agl.typecheck.types import IntType

        caps = HostCapabilities(
            agent_names=frozenset(),
            has_default_agent=True,
            codec_kinds={"text": frozenset({"text"})},
        )

        # Entry 1: declare and check a mutable binding.
        p1 = parse_program("var v = 0")
        r1 = resolve(p1)
        c1 = check(r1, caps)
        var_v = r1.program.body[0]
        assert isinstance(var_v, VarDecl)
        assert c1.type_env.get_binding_type(var_v.node_id) == IntType()

        # Entry 2: ``set v = 5`` resolves into entry 1's scope and checks against
        # the seeded binding type.
        p2 = parse_program("set v = 5")
        r2 = resolve(p2, parent_scope=r1.root_scope)
        set_stmt = r2.program.body[0]
        assert isinstance(set_stmt, SetStmt)
        ref = r2.resolution[set_stmt.node_id]
        assert ref.name == "v"
        assert ref.mutable is True
        assert ref.decl_node_id == var_v.node_id
        # Type-checking must succeed with the seeded env (no mismatch).
        c2 = check(r2, caps, seed_env=c1.type_env)
        assert c2 is not None


# ---------------------------------------------------------------------------
# Agent declarations and the binding rule
# ---------------------------------------------------------------------------


class TestAgentDeclarations:
    """``agent NAME`` declarations and undeclared-call binding errors."""

    def test_undeclared_agent_call_rejected(self) -> None:
        # matches tests/agl/rejections/scope/undeclared_agent.agl
        err = reject_scope('let x = reviewer "Review this"')
        line, msg = diag(err)
        assert line == 1
        assert "reviewer" in msg
        assert "unknown agent" in msg.lower()

    def test_declared_and_called_resolves(self) -> None:
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        r = parse_and_resolve('agent reviewer\nlet x = reviewer "Review this"')
        stmt = r.program.body[1]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        assert r.call_kinds[stmt.value.node_id] == CallKind.agent
        assert "reviewer" in r.declared_agents
        # A declared-and-called agent produces no unused warning.
        assert r.warnings == ()

    def test_declared_with_runner_resolves(self) -> None:
        r = parse_and_resolve(
            'agent impl = "claude -p %{PROMPT_FILE}"\nlet x = impl "Do it"'
        )
        assert "impl" in r.declared_agents
        assert r.declared_agents["impl"].runner == "claude -p %{PROMPT_FILE}"

    def test_duplicate_declaration_rejected(self) -> None:
        # matches tests/agl/rejections/scope/agent_redeclared.agl
        err = reject_scope("agent dup\nagent dup")
        _, msg = diag(err)
        assert "dup" in msg
        assert "already declared" in msg.lower()

    def test_declare_prompt_rejected(self) -> None:
        # matches tests/agl/rejections/scope/agent_reserved_name.agl
        err = reject_scope("agent prompt")
        _, msg = diag(err)
        assert "prompt" in msg
        assert "built-in" in msg.lower()

    def test_declare_exec_rejected(self) -> None:
        err = reject_scope("agent exec")
        _, msg = diag(err)
        assert "exec" in msg
        assert "built-in" in msg.lower()

    def test_declaration_inside_if_rejected(self) -> None:
        # matches tests/agl/rejections/scope/agent_not_root.agl
        err = reject_scope("if true =>\n  agent late\n| else =>\n  pass")
        line, msg = diag(err)
        assert line == 2
        assert "agent" in msg.lower()
        assert "root" in msg.lower()

    def test_declaration_inside_do_rejected(self) -> None:
        err = reject_scope("do[2]\n  agent late\nuntil true\n")
        _, msg = diag(err)
        assert "agent" in msg.lower()
        assert "root" in msg.lower()

    def test_declared_but_unused_warns(self) -> None:
        r = parse_and_resolve("agent unused")
        assert "unused" in r.declared_agents
        assert len(r.warnings) == 1
        warning = r.warnings[0]
        assert warning.severity == "warning"
        assert "unused" in warning.message
        assert warning.line == 1

    def test_prompt_needs_no_declaration(self) -> None:
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        r = parse_and_resolve('let x = prompt "Q"')
        stmt = r.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        assert r.call_kinds[stmt.value.node_id] == CallKind.default_agent
        assert r.declared_agents == {}
        assert r.warnings == ()

    def test_exec_needs_no_declaration(self) -> None:
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        r = parse_and_resolve('let x = exec "ls"')
        stmt = r.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        assert r.call_kinds[stmt.value.node_id] == CallKind.shell_exec

    def test_ambient_agent_call_resolves(self) -> None:
        from agm.agl.syntax.nodes import AgentCall, LetDecl

        r = resolve(
            parse_program('let x = session_agent "Q"'),
            ambient_agents=frozenset({"session_agent"}),
        )
        stmt = r.program.body[0]
        assert isinstance(stmt, LetDecl)
        assert isinstance(stmt.value, AgentCall)
        assert r.call_kinds[stmt.value.node_id] == CallKind.agent
        # Ambient agents are not in-program declarations and never warn.
        assert r.declared_agents == {}
        assert r.warnings == ()

    def test_agent_namespace_separate_from_variables(self) -> None:
        # An `agent impl` declaration must not collide with `let impl`.
        r = parse_and_resolve('agent impl\nlet impl = "x"\nlet y = impl "Q"')
        assert "impl" in r.declared_agents
        assert "impl" in r.root_scope.bindings


class TestAmbientAgentsHelper:
    """The ``ambient_agents_for`` test helper collects named-agent call targets."""

    def test_collects_named_agents_only(self) -> None:
        from tests._agl_helpers import ambient_agents_for

        prog = parse_program(
            'let a = reviewer "R"\n'
            'let b = impl "I"\n'
            'let c = prompt "P"\n'
            'let d = exec "ls"\n'
        )
        assert ambient_agents_for(prog) == frozenset({"reviewer", "impl"})

    def test_empty_when_no_named_calls(self) -> None:
        from tests._agl_helpers import ambient_agents_for

        prog = parse_program('let a = prompt "P"')
        assert ambient_agents_for(prog) == frozenset()

    def test_helper_output_resolves_without_declarations(self) -> None:
        from tests._agl_helpers import ambient_agents_for

        prog = parse_program('let a = reviewer "R"')
        # Passing the helper output as ambient_agents lets a program that calls
        # a named agent resolve without any in-source `agent` declaration.
        r = resolve(prog, ambient_agents=ambient_agents_for(prog))
        assert r.warnings == ()
        assert r.declared_agents == {}
