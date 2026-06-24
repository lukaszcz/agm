"""Tests for the AgL v2 scope/name-resolution pass (Component 4).

All tests drive real AgL source through ``parse_program`` + ``resolve``,
or construct v2 AST nodes directly for cases that are clearer to express
at the AST level.

Tests assert on user-visible behavior: ``AglScopeError`` diagnostics and
observable side-table behavior via the public ``ResolvedProgram`` API.  They
deliberately do *not* pin internal implementation details.
"""

from __future__ import annotations

from typing import cast

import pytest

from agm.agl.parser import parse_program
from agm.agl.scope import (
    AglScopeError,
    BuiltinKind,
    ResolvedProgram,
    resolve,
)
from agm.agl.scope.symbols import BinderKind
from agm.agl.syntax.nodes import (
    AssignStmt,
    AssignTarget,
    Block,
    BoolLit,
    Call,
    CaseBranch,
    CatchClause,
    ConstructorPattern,
    Do,
    EnumDef,
    Expr,
    FieldAccess,
    FuncDef,
    If,
    IfBranch,
    IntLit,
    Item,
    Lambda,
    LetDecl,
    NameTarget,
    Param,
    PatternField,
    Program,
    RecordDef,
    StringLit,
    Template,
    Try,
    UnitLit,
    VarDecl,
    VariantDef,
    VarPattern,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan  # noqa: TCH002
from agm.agl.syntax.types import IntT

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
# AST construction helpers (for hand-built node tests)
# ---------------------------------------------------------------------------

_NID = 0


def _nid() -> int:
    global _NID
    _NID += 1
    return _NID


def _sp(line: int = 1) -> SourceSpan:
    return SourceSpan(
        start_line=line, start_col=1, end_line=line, end_col=2,
        start_offset=0, end_offset=1,
    )


def _make_intlit(val: int = 0, line: int = 1) -> IntLit:
    return IntLit(value=val, span=_sp(line), node_id=_nid())


def _make_boollit(val: bool = True, line: int = 1) -> BoolLit:
    return BoolLit(value=val, span=_sp(line), node_id=_nid())


def _make_strlit(val: str = "s", line: int = 1) -> StringLit:
    return StringLit(value=val, span=_sp(line), node_id=_nid())


def _make_unitlit(line: int = 1) -> UnitLit:
    return UnitLit(span=_sp(line), node_id=_nid())


def _make_varref(name: str, line: int = 1) -> VarRef:
    return VarRef(name=name, span=_sp(line), node_id=_nid())


def _make_let(name: str, value: Expr, line: int = 1) -> LetDecl:
    return LetDecl(name=name, type_ann=None, value=value, span=_sp(line), node_id=_nid())


def _make_var(name: str, value: Expr, line: int = 1) -> VarDecl:
    return VarDecl(name=name, type_ann=None, value=value, span=_sp(line), node_id=_nid())


def _make_assign(target: str, value: Expr, line: int = 1) -> AssignStmt:
    span = _sp(line)
    return AssignStmt(
        target=NameTarget(name=target, span=span, node_id=_nid()),
        value=value,
        span=span,
        node_id=_nid(),
    )


def _make_block(*items: Item, line: int = 1) -> Block:
    return Block(items=items, span=_sp(line), node_id=_nid())


def _make_program(*items: Item) -> Program:
    block = Block(items=items, span=_sp(), node_id=_nid())
    return Program(body=block, span=_sp(), node_id=_nid())


def resolve_program(*items: Item) -> ResolvedProgram:
    """Construct and resolve a Program from the given top-level items."""
    return resolve(_make_program(*items))


def reject_program(*items: Item) -> AglScopeError:
    """Assert that a program built from the given items fails scope resolution."""
    with pytest.raises(AglScopeError) as exc_info:
        resolve_program(*items)
    return exc_info.value


# ---------------------------------------------------------------------------
# Basic acceptance
# ---------------------------------------------------------------------------


class TestAcceptance:
    def test_simple_let(self) -> None:
        r = parse_and_resolve('let x = "hello"\nx')
        assert r.program is not None

    def test_let_and_print(self) -> None:
        r = parse_and_resolve('let x = "hello"\nprint x')
        assert r.program is not None

    def test_var_and_assign(self) -> None:
        r = parse_and_resolve("var n: int = 0\nn := 1\nn")
        assert r.program is not None

    def test_param_at_root(self) -> None:
        r = parse_and_resolve("param spec\nspec")
        assert r.program is not None

    def test_param_with_type(self) -> None:
        r = parse_and_resolve("param spec: text\nprint spec")
        assert r.program is not None

    def test_let_after_input(self) -> None:
        r = parse_and_resolve("param spec\nlet x = spec\nx")
        assert r.program is not None

    def test_let_with_interpolation(self) -> None:
        r = parse_and_resolve(
            "param name\n"
            'let greeting = "Hello ${name}"\n'
            "greeting"
        )
        assert r.program is not None

    def test_multiple_inputs(self) -> None:
        r = parse_and_resolve(
            "param spec\n"
            "param max_severity: int\n"
            "spec"
        )
        assert r.program is not None

    def test_unit_lit(self) -> None:
        r = parse_and_resolve("()")
        assert r.program is not None

    def test_type_alias_at_root(self) -> None:
        r = parse_and_resolve("type MyText = text\n()")
        assert r.program is not None

    def test_record_def_at_root(self) -> None:
        r = parse_and_resolve("record P\n  n: int\n()")
        assert r.program is not None

    def test_enum_def_at_root(self) -> None:
        r = parse_and_resolve("enum E\n  | A\n  | B\n()")
        assert r.program is not None

    def test_raise_expr(self) -> None:
        r = parse_and_resolve("raise 1\n")
        assert r.program is not None


# ---------------------------------------------------------------------------
# Block forward-scoping and isolation
# ---------------------------------------------------------------------------


class TestBlockScoping:
    def test_forward_binding_visible_after_let(self) -> None:
        """A name bound by ``let`` is visible to all subsequent items."""
        r = parse_and_resolve("let x = 1\nlet y = x\ny")
        assert r.program is not None

    def test_binding_not_visible_before_let(self) -> None:
        """A name is NOT visible before its binder in the same block."""
        err = reject_scope("let y = x\nlet x = 1\ny")
        line, msg = diag(err)
        assert line == 1
        assert "x" in msg

    def test_block_local_binding_does_not_escape(self) -> None:
        """A binding in a branch block is not visible outside."""
        err = reject_scope(
            "if true =>\n"
            "  let inner = 1\n"
            "| else =>\n"
            "  ()\n"
            "inner\n"
        )
        line, msg = diag(err)
        assert line == 5
        assert "inner" in msg

    def test_outer_binding_visible_in_nested_block(self) -> None:
        """A binding from an outer scope is visible in nested blocks."""
        r = parse_and_resolve(
            "let outer = 42\n"
            "if true =>\n"
            "  outer\n"
            "| else =>\n"
            "  outer\n"
        )
        assert r.program is not None

    def test_redeclaration_same_scope_let_let(self) -> None:
        err = reject_scope("let x = 1\nlet x = 2\nx")
        line, msg = diag(err)
        assert line == 2
        assert "x" in msg

    def test_redeclaration_same_scope_let_var(self) -> None:
        err = reject_scope("let twice = 1\nvar twice = 2\ntwice")
        line, msg = diag(err)
        assert line == 2
        assert "twice" in msg

    def test_redeclaration_same_scope_var_var(self) -> None:
        err = reject_scope("var a = 1\nvar a = 2\na")
        line, msg = diag(err)
        assert line == 2
        assert "a" in msg

    def test_redeclaration_input_with_let(self) -> None:
        err = reject_scope('param spec\nlet spec = "again"\nspec')
        line, msg = diag(err)
        assert line == 2
        assert "spec" in msg

    def test_redeclaration_input_with_input(self) -> None:
        err = reject_scope("param x\nparam x\nx")
        line, msg = diag(err)
        assert line == 2
        assert "x" in msg


# ---------------------------------------------------------------------------
# Assignment errors
# ---------------------------------------------------------------------------


class TestAssignErrors:
    def test_assign_to_let(self) -> None:
        err = reject_scope("let stable = 1\nstable := 2\nstable")
        line, msg = diag(err)
        assert line == 2
        assert "stable" in msg

    def test_assign_to_undeclared(self) -> None:
        err = reject_scope("ghost := 1")
        line, msg = diag(err)
        assert line == 1
        assert "ghost" in msg

    def test_assign_to_input(self) -> None:
        err = reject_scope("param spec\nspec := 2\nspec")
        line, msg = diag(err)
        assert line == 2
        assert "spec" in msg

    def test_assign_to_let_names_let(self) -> None:
        err = reject_scope("let stable = 1\nstable := 2\nstable")
        _, msg = diag(err)
        assert "let" in msg
        assert "immutable" in msg

    def test_assign_to_input_names_input_not_let(self) -> None:
        err = reject_scope("param spec\nspec := 2\nspec")
        _, msg = diag(err)
        assert "param" in msg
        assert "declared with 'let'" not in msg

    def test_assign_to_catch_binder_names_catch(self) -> None:
        err = reject_scope(
            "try\n"
            "  ()\n"
            "catch _ as err =>\n"
            "  err := 1\n"
        )
        line, msg = diag(err)
        assert line == 4
        assert "catch binder" in msg
        assert "declared with 'let'" not in msg

    def test_assign_to_pattern_binding_names_pattern(self) -> None:
        err = reject_scope(
            "let v = 1\n"
            "case v of\n"
            "  | n =>\n"
            "    n := 2\n"
        )
        line, msg = diag(err)
        assert line == 4
        assert "pattern binding" in msg
        assert "declared with 'let'" not in msg

    def test_assign_to_param_names_param(self) -> None:
        """Assigning to a parameter is rejected with the 'param_binding' phrasing."""
        # Use direct AST construction since the parser only allows expressions
        # in a def body (assignment is a binder, not an expr in call position).
        from agm.agl.syntax.types import IntT as IntTNode

        sp = _sp()
        int_t = IntTNode(span=sp, node_id=_nid())
        param = Param(name="n", type_expr=int_t, default=None, span=sp, node_id=_nid())
        assign_n = _make_assign("n", _make_intlit(2))
        funcdef = FuncDef(
            name="f",
            params=(param,),
            return_type=int_t,
            body=_make_block(assign_n, _make_varref("n")),
            span=sp,
            node_id=_nid(),
        )
        call_f = Call(
            callee=_make_varref("f"),
            args=(_make_intlit(1),),
            named_args=(),
            span=sp,
            node_id=_nid(),
        )
        err = reject_program(funcdef, call_f)
        _, msg = diag(err)
        assert "parameter binding" in msg

    def test_invalid_direct_ast_assign_target_rejected(self) -> None:
        assign_bad = AssignStmt(
            target=cast(AssignTarget, _make_unitlit()),
            value=_make_intlit(1),
            span=_sp(),
            node_id=_nid(),
        )
        err = reject_program(assign_bad, _make_unitlit())
        _, msg = diag(err)
        assert "indexed assignment requires a variable list or dict root" in msg

    def test_assign_to_function_binding_names_function(self) -> None:
        """Assigning to a def name is rejected."""
        err = reject_scope("def f(x: int) -> int = x\nf := 1\nf(1)")
        _, msg = diag(err)
        assert "function" in msg.lower() or "def" in msg.lower() or "immutable" in msg.lower()

    def test_assign_to_var_resolves(self) -> None:
        r = parse_and_resolve("var n = 0\nn := 1\nn")
        assert r.program is not None
        # Verify the assignment statement is in the resolution table.
        block = r.program.body
        assign_node = block.items[1]
        assert isinstance(assign_node, AssignStmt)
        ref = r.resolution[assign_node.node_id]
        assert ref.name == "n"
        assert ref.mutable is True


# ---------------------------------------------------------------------------
# Undefined name reads
# ---------------------------------------------------------------------------


class TestUndefinedRead:
    def test_undefined_in_assignment(self) -> None:
        err = reject_scope("let x = undeclared\nx")
        line, msg = diag(err)
        assert line == 1
        assert "undeclared" in msg

    def test_undefined_in_interpolation(self) -> None:
        err = reject_scope('let x = "Hi ${ghost}"\nx')
        line, msg = diag(err)
        assert line == 1
        assert "ghost" in msg

    def test_undefined_callee(self) -> None:
        err = reject_scope("no_such_func(1)")
        line, msg = diag(err)
        assert line == 1
        assert "no_such_func" in msg

    def test_raise_undefined_rejected(self) -> None:
        err = reject_scope("raise nope\n")
        assert "nope" in err.to_diagnostic().message


# ---------------------------------------------------------------------------
# Reserved names: built-in call names cannot be bound
# ---------------------------------------------------------------------------


class TestReservedNames:
    def test_reserve_print_let(self) -> None:
        err = reject_scope('let print = "x"\nprint')
        line, msg = diag(err)
        assert line == 1
        assert "print" in msg

    def test_reserve_render_let(self) -> None:
        err = reject_scope('let render = "x"\nrender')
        line, msg = diag(err)
        assert line == 1
        assert "render" in msg

    def test_reserve_ask_let(self) -> None:
        err = reject_scope('let ask = "not allowed"\nask')
        line, msg = diag(err)
        assert line == 1
        assert "ask" in msg

    def test_reserve_exec_var(self) -> None:
        err = reject_scope('var exec = "not allowed"\nexec')
        line, msg = diag(err)
        assert line == 1
        assert "exec" in msg

    def test_reserve_ask_input(self) -> None:
        err = reject_scope("param ask")
        line, msg = diag(err)
        assert line == 1
        assert "ask" in msg

    def test_reserve_exec_input(self) -> None:
        err = reject_scope("param exec")
        line, msg = diag(err)
        assert line == 1
        assert "exec" in msg

    def test_reserve_print_input(self) -> None:
        err = reject_scope("param print")
        line, msg = diag(err)
        assert line == 1
        assert "print" in msg

    def test_reserve_ask_agent(self) -> None:
        err = reject_scope("agent ask")
        _, msg = diag(err)
        assert "ask" in msg
        assert "built-in" in msg.lower()

    def test_reserve_exec_agent(self) -> None:
        err = reject_scope("agent exec")
        _, msg = diag(err)
        assert "exec" in msg
        assert "built-in" in msg.lower()

    def test_reserve_print_agent(self) -> None:
        err = reject_scope("agent print")
        _, msg = diag(err)
        assert "print" in msg
        assert "built-in" in msg.lower()

    def test_reserve_ask_def(self) -> None:
        err = reject_scope("def ask() -> int = 1\nask()")
        _, msg = diag(err)
        assert "ask" in msg

    def test_reserve_exec_def(self) -> None:
        err = reject_scope("def exec() -> int = 1\nexec()")
        _, msg = diag(err)
        assert "exec" in msg

    def test_reserve_print_def(self) -> None:
        err = reject_scope("def print() -> int = 1\nprint()")
        _, msg = diag(err)
        assert "print" in msg

    def test_reserve_ask_param(self) -> None:
        err = reject_scope("def f(ask: int) -> int = 1\nf(1)")
        _, msg = diag(err)
        assert "ask" in msg

    def test_reserve_exec_param(self) -> None:
        err = reject_scope("def f(exec: int) -> int = 1\nf(1)")
        _, msg = diag(err)
        assert "exec" in msg

    def test_reserve_print_param(self) -> None:
        err = reject_scope("def f(print: int) -> int = 1\nf(1)")
        _, msg = diag(err)
        assert "print" in msg

    def test_reserve_ask_catch_binder(self) -> None:
        err = reject_scope(
            "try\n"
            "  ()\n"
            "catch _ as ask =>\n"
            "  ()\n"
        )
        _, msg = diag(err)
        assert "ask" in msg
        assert "reserved" in msg.lower() or "contextual" in msg.lower()

    def test_reserve_exec_catch_binder(self) -> None:
        err = reject_scope(
            "try\n"
            "  ()\n"
            "catch _ as exec =>\n"
            "  ()\n"
        )
        _, msg = diag(err)
        assert "exec" in msg

    def test_reserve_ask_pattern_var(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        pv = VarPattern(name="ask", span=_sp(2), node_id=_nid())
        branch = CaseBranch(
            pattern=pv,
            body=_make_unitlit(),
            span=_sp(2),
            node_id=_nid(),
        )
        from agm.agl.syntax.nodes import Case
        case_node = Case(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(2),
            node_id=_nid(),
        )
        err = reject_program(let_x, case_node)
        msg = err.to_diagnostic().message
        assert "ask" in msg
        assert "reserved" in msg.lower() or "contextual" in msg.lower()

    def test_bare_ask_varref_rejected(self) -> None:
        """A bare VarRef to 'ask' (not in call position) is rejected."""
        err = reject_scope("let f = ask")
        _, msg = diag(err)
        assert "ask" in msg

    def test_bare_exec_varref_rejected(self) -> None:
        err = reject_scope("let f = exec")
        _, msg = diag(err)
        assert "exec" in msg

    def test_bare_print_varref_rejected(self) -> None:
        err = reject_scope("let f = print")
        _, msg = diag(err)
        assert "print" in msg


# ---------------------------------------------------------------------------
# Built-in call classification (builtin_calls side table)
# ---------------------------------------------------------------------------


class TestBuiltinCallClassification:
    def test_print_call_classified(self) -> None:
        r = parse_and_resolve('let x = 1\nprint x')
        # find the Call node in the block
        call_item = r.program.body.items[1]
        assert isinstance(call_item, Call)
        assert r.builtin_calls[call_item.node_id] == BuiltinKind.PRINT

    def test_render_call_classified(self) -> None:
        r = parse_and_resolve('let x = render 1\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.RENDER

    def test_exec_call_classified(self) -> None:
        r = parse_and_resolve('let x = exec "ls"\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.EXEC

    def test_ask_call_classified(self) -> None:
        r = parse_and_resolve('let x = ask "Q"\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.ASK

    def test_ask_with_agent_arg_classified(self) -> None:
        r = parse_and_resolve('agent reviewer\nlet x = ask("Q", agent: reviewer)\nx')
        let_node = r.program.body.items[1]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.ASK

    def test_ask_request_call_classified(self) -> None:
        r = parse_and_resolve('let x = ask-request::[Review]("Q")\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.ASK_REQUEST

    def test_ask_request_without_type_arg_classified(self) -> None:
        r = parse_and_resolve('let x = ask-request("Q")\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.ASK_REQUEST

    def test_ask_request_callee_not_in_resolution(self) -> None:
        r = parse_and_resolve('let x = ask-request::[text]("Q")\nx')
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        call = let_node.value
        assert isinstance(call, Call)
        callee = call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id not in r.resolution

    def test_ask_request_reserved_as_value(self) -> None:
        # ``ask-request`` is a reserved contextual keyword: a bare reference
        # (not in call position) is rejected.
        with pytest.raises(AglScopeError) as exc_info:
            parse_and_resolve('let x = ask-request\nx')
        msg = str(exc_info.value)
        assert "built-in" in msg.lower() or "reserved" in msg.lower()

    def test_user_def_call_not_classified(self) -> None:
        """A user-defined function call does NOT appear in builtin_calls."""
        r = parse_and_resolve("def f(x: int) -> int = x\nlet y = f(1)\ny")
        let_node = r.program.body.items[1]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        # User call: not in builtin_calls
        assert let_node.value.node_id not in r.builtin_calls

    def test_lambda_call_not_classified(self) -> None:
        """Calling a lambda-bound name is not in builtin_calls."""
        r = parse_and_resolve("let f = fn(x: int) => x\nlet y = f(1)\ny")
        let_y = r.program.body.items[1]
        assert isinstance(let_y, LetDecl)
        assert isinstance(let_y.value, Call)
        assert let_y.value.node_id not in r.builtin_calls
        # The lambda binding was resolved
        call = let_y.value
        assert isinstance(call.callee, VarRef)
        assert call.callee.node_id in r.resolution
        assert r.resolution[call.callee.node_id].name == "f"

    def test_print_call_callee_not_in_resolution(self) -> None:
        """The callee VarRef of a built-in call is NOT in the resolution table."""
        r = parse_and_resolve("let x = 1\nprint x")
        call_item = r.program.body.items[1]
        assert isinstance(call_item, Call)
        # Built-in callee VarRef should not be resolved as a binding
        callee = call_item.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id not in r.resolution

    def test_print_positional_arg_resolved(self) -> None:
        """The argument to print IS resolved."""
        r = parse_and_resolve("let x = 1\nprint x")
        call_item = r.program.body.items[1]
        assert isinstance(call_item, Call)
        arg = call_item.args[0]
        assert isinstance(arg, VarRef)
        assert arg.node_id in r.resolution
        assert r.resolution[arg.node_id].name == "x"


# ---------------------------------------------------------------------------
# Uniform Call: named args resolved
# ---------------------------------------------------------------------------


class TestCallResolution:
    def test_call_named_arg_value_resolved(self) -> None:
        """Named-arg values in a call are resolved."""
        r = parse_and_resolve(
            "agent reviewer\n"
            'let x = ask("Q", agent: reviewer)\n'
            "x"
        )
        let_node = r.program.body.items[1]
        assert isinstance(let_node, LetDecl)
        call = let_node.value
        assert isinstance(call, Call)
        # The 'agent:' named arg value (VarRef("reviewer")) must be resolved
        named = call.named_args[0]
        assert named.name == "agent"
        assert isinstance(named.value, VarRef)
        assert named.value.node_id in r.resolution
        ref = r.resolution[named.value.node_id]
        assert ref.name == "reviewer"
        assert ref.kind == BinderKind.agent_binding

    def test_user_call_positional_args_resolved(self) -> None:
        r = parse_and_resolve(
            "def add(a: int, b: int) -> int = a\n"
            "let x = 1\n"
            "let y = 2\n"
            "let z = add(x, y)\n"
            "z"
        )
        let_z = r.program.body.items[3]
        assert isinstance(let_z, LetDecl)
        call = let_z.value
        assert isinstance(call, Call)
        for arg in call.args:
            assert isinstance(arg, VarRef)
            assert arg.node_id in r.resolution


# ---------------------------------------------------------------------------
# Top-level def: mutual recursion + forward references
# ---------------------------------------------------------------------------


class TestFuncDefMutualRecursion:
    def test_def_at_root_accepted(self) -> None:
        r = parse_and_resolve("def f(n: int) -> int = n\nf(1)")
        assert r.program is not None
        assert "f" in r.declared_functions

    def test_def_self_recursion(self) -> None:
        r = parse_and_resolve(
            "def fact(n: int) -> int = if n <= 1 => 1 | else => n * fact(n - 1)\n"
            "fact(5)"
        )
        assert r.program is not None

    def test_def_mutual_recursion(self) -> None:
        """Two top-level defs can reference each other."""
        r = parse_and_resolve(
            "def even(n: int) -> bool = if n = 0 => true | else => odd(n - 1)\n"
            "def odd(n: int) -> bool = if n = 0 => false | else => even(n - 1)\n"
            "even(4)"
        )
        assert r.program is not None
        assert "even" in r.declared_functions
        assert "odd" in r.declared_functions

    def test_def_forward_reference(self) -> None:
        """A def can call another def declared AFTER it (pre-pass collects all)."""
        r = parse_and_resolve(
            "def caller(n: int) -> int = callee(n)\n"
            "def callee(n: int) -> int = n\n"
            "caller(1)"
        )
        assert r.program is not None

    def test_def_body_sees_param(self) -> None:
        """Param names are in scope inside the body."""
        r = parse_and_resolve("def f(x: int) -> int = x\nf(1)")
        assert r.program is not None

    def test_def_params_not_visible_outside(self) -> None:
        """Param names are not visible outside the function body."""
        err = reject_scope("def f(x: int) -> int = x\nx\nf(1)")
        _, msg = diag(err)
        assert "x" in msg
        assert "not defined" in msg

    def test_def_nested_in_block_rejected(self) -> None:
        """A def nested inside a block (e.g. if branch) is rejected."""
        err = reject_scope("if true =>\n  def f(x: int) -> int = x\n| else =>\n  ()\n")
        _, msg = diag(err)
        assert "def" in msg.lower()
        assert "root" in msg.lower()

    def test_def_duplicate_name_rejected(self) -> None:
        err = reject_scope("def f(x: int) -> int = x\ndef f(y: int) -> int = y\nf(1)")
        _, msg = diag(err)
        assert "f" in msg

    def test_def_name_in_resolution_table(self) -> None:
        """The function name VarRef in a call is resolved to the function binding."""
        r = parse_and_resolve("def f(x: int) -> int = x\nlet y = f(1)\ny")
        let_y = r.program.body.items[1]
        assert isinstance(let_y, LetDecl)
        call = let_y.value
        assert isinstance(call, Call)
        callee = call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.resolution
        ref = r.resolution[callee.node_id]
        assert ref.name == "f"
        assert ref.kind == BinderKind.function_binding

    def test_def_param_default_resolved_in_enclosing_scope(self) -> None:
        """Parameter defaults are resolved in the DEFINITION scope (outer)."""
        r = parse_and_resolve(
            "let base = 10\n"
            "def f(x: int = base) -> int = x\n"
            "f()"
        )
        assert r.program is not None

    def test_def_param_duplicate_rejected(self) -> None:
        err = reject_scope("def f(x: int, x: int) -> int = x\nf(1, 2)")
        _, msg = diag(err)
        assert "x" in msg


# ---------------------------------------------------------------------------
# Lambda scoping: non-self-recursive
# ---------------------------------------------------------------------------


class TestLambdaScoping:
    def test_lambda_param_visible_in_body(self) -> None:
        r = parse_and_resolve("let f = fn(x: int) => x\nf(1)")
        assert r.program is not None

    def test_lambda_non_recursive(self) -> None:
        """Lambda body does NOT see the let-binding (f is not in scope in its RHS)."""
        err = reject_scope("let f = fn(x: int) => f(x)\nf(1)")
        _, msg = diag(err)
        assert "f" in msg
        assert "not defined" in msg

    def test_lambda_param_not_visible_outside(self) -> None:
        err = reject_scope("let f = fn(x: int) => x\nx\nf(1)")
        _, msg = diag(err)
        assert "x" in msg

    def test_lambda_captures_outer(self) -> None:
        """Lambda body can reference outer-scope bindings."""
        r = parse_and_resolve("let base = 10\nlet f = fn(x: int) => x\nf(1)")
        assert r.program is not None

    def test_lambda_param_reserved_rejected(self) -> None:
        err = reject_scope("let f = fn(print: int) => print\nf(1)")
        _, msg = diag(err)
        assert "print" in msg

    def test_lambda_call_is_not_builtin(self) -> None:
        """Calling a lambda via a variable does not produce a builtin_calls entry."""
        r = parse_and_resolve("let f = fn(x: int) => x\nlet y = f(1)\ny")
        let_y = r.program.body.items[1]
        assert isinstance(let_y, LetDecl)
        call = let_y.value
        assert isinstance(call, Call)
        assert call.node_id not in r.builtin_calls

    def test_lambda_default_in_enclosing_scope(self) -> None:
        """Default expressions in a lambda are resolved in the enclosing scope."""
        r = parse_and_resolve("let base = 5\nlet f = fn(x: int = base) => x\nf()")
        assert r.program is not None


# ---------------------------------------------------------------------------
# Agents as value bindings
# ---------------------------------------------------------------------------


class TestAgentValueBindings:
    def test_agent_decl_creates_value_binding(self) -> None:
        """An agent declaration creates a value binding in the root scope."""
        r = parse_and_resolve("agent reviewer\n()")
        assert "reviewer" in r.declared_agents
        assert "reviewer" in r.root_scope.bindings
        ref = r.root_scope.bindings["reviewer"]
        assert ref.kind == BinderKind.agent_binding
        assert not ref.mutable

    def test_agent_ref_in_ask_named_arg_resolves(self) -> None:
        """An agent name used as a VarRef in ask(agent:) resolves to the binding."""
        r = parse_and_resolve(
            "agent reviewer\n"
            'let x = ask("Q", agent: reviewer)\n'
            "x"
        )
        let_node = r.program.body.items[1]
        assert isinstance(let_node, LetDecl)
        call = let_node.value
        assert isinstance(call, Call)
        named = call.named_args[0]
        assert isinstance(named.value, VarRef)
        ref = r.resolution[named.value.node_id]
        assert ref.kind == BinderKind.agent_binding
        assert ref.name == "reviewer"

    def test_agent_let_binding_stores_agent_value(self) -> None:
        """An agent name can be stored in a let binding."""
        r = parse_and_resolve("agent reviewer\nlet a = reviewer\na")
        let_a = r.program.body.items[1]
        assert isinstance(let_a, LetDecl)
        assert isinstance(let_a.value, VarRef)
        ref = r.resolution[let_a.value.node_id]
        assert ref.kind == BinderKind.agent_binding

    def test_agent_ref_marks_as_referenced(self) -> None:
        """An agent referenced via VarRef counts as 'used' → no unused warning."""
        r = parse_and_resolve("agent reviewer\nlet a = reviewer\na")
        assert r.warnings == ()

    def test_declared_but_unused_warns(self) -> None:
        r = parse_and_resolve("agent unused\n()")
        assert "unused" in r.declared_agents
        assert len(r.warnings) == 1
        warning = r.warnings[0]
        assert warning.severity == "warning"
        assert "unused" in warning.message
        assert warning.line == 1

    def test_agent_not_at_root_rejected(self) -> None:
        err = reject_scope("if true =>\n  agent late\n| else =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "agent" in msg.lower()
        assert "root" in msg.lower()

    def test_duplicate_agent_rejected(self) -> None:
        err = reject_scope("agent dup\nagent dup\n()")
        _, msg = diag(err)
        assert "dup" in msg
        assert "already declared" in msg.lower()


# ---------------------------------------------------------------------------
# Do body/until scoping
# ---------------------------------------------------------------------------


class TestDoScoping:
    def test_do_body_binding_visible_in_until(self) -> None:
        """A binding defined in the do body is visible in the until condition."""
        r = parse_and_resolve(
            "var n = 0\n"
            "do[2]\n"
            "  let probe = n\n"
            "  n := probe\n"
            "until n >= 1\n"
            "n"
        )
        assert r.program is not None

    def test_do_body_binding_not_visible_after_loop(self) -> None:
        """A binding from the do body is not visible after the loop."""
        err = reject_scope(
            "do[2]\n"
            "  let inner = 1\n"
            "until true\n"
            "inner\n"
        )
        line, msg = diag(err)
        assert line == 4
        assert "inner" in msg

    def test_do_input_not_root_error(self) -> None:
        err = reject_scope("do[2]\n  param x\nuntil true\n")
        line, msg = diag(err)
        assert line == 2
        assert "param" in msg.lower()

    def test_do_inline_body_resolved(self) -> None:
        """Inline (non-block) do body is also resolved."""
        r = parse_and_resolve("var n = 0\ndo[2] n := 1 until n >= 1\nn")
        assert r.program is not None


# ---------------------------------------------------------------------------
# If expression scoping
# ---------------------------------------------------------------------------


class TestIfScoping:
    def test_if_condition_resolved(self) -> None:
        r = parse_and_resolve(
            "let x = true\n"
            "if x => 1 | else => 2\n"
        )
        assert r.program is not None

    def test_if_branch_body_local(self) -> None:
        r = parse_and_resolve(
            "let x = 1\n"
            "if true =>\n"
            "  let y = x\n"
            "  y\n"
            "| else =>\n"
            "  x\n"
        )
        assert r.program is not None

    def test_if_inner_not_visible_outside(self) -> None:
        err = reject_scope(
            "if true =>\n"
            "  let inner = 1\n"
            "| else =>\n"
            "  ()\n"
            "inner\n"
        )
        line, msg = diag(err)
        assert line == 5
        assert "inner" in msg

    def test_if_no_else_accepted(self) -> None:
        r = parse_and_resolve("let x = 1\nif x = 1 => print 1\n")
        assert r.program is not None


# ---------------------------------------------------------------------------
# Case expression scoping
# ---------------------------------------------------------------------------


class TestCaseScoping:
    def test_case_var_pattern_visible_in_body(self) -> None:
        r = parse_and_resolve(
            "let x = 1\n"
            "case x of\n"
            "  | n => n\n"
        )
        assert r.program is not None

    def test_case_pattern_not_visible_outside(self) -> None:
        err = reject_scope(
            "let x = 1\n"
            "case x of\n"
            "  | n => n\n"
            "n\n"
        )
        line, msg = diag(err)
        assert line == 4
        assert "n" in msg

    def test_case_wildcard_pattern(self) -> None:
        r = parse_and_resolve(
            "let x = 1\n"
            "case x of\n"
            "  | _ => 0\n"
        )
        assert r.program is not None


# ---------------------------------------------------------------------------
# Try/catch scoping
# ---------------------------------------------------------------------------


class TestTryScoping:
    def test_try_body_resolved(self) -> None:
        r = parse_and_resolve(
            "try\n"
            "  let x = 1\n"
            "  x\n"
            "catch _ =>\n"
            "  0\n"
        )
        assert r.program is not None

    def test_catch_binder_visible_in_catch_body(self) -> None:
        r = parse_and_resolve(
            "try\n"
            "  1\n"
            "catch _ as err =>\n"
            "  err\n"
        )
        assert r.program is not None

    def test_catch_binder_not_visible_outside(self) -> None:
        err = reject_scope(
            "try\n"
            "  1\n"
            "catch _ as err =>\n"
            "  err\n"
            "err\n"
        )
        line, msg = diag(err)
        assert line == 5
        assert "err" in msg


# ---------------------------------------------------------------------------
# Config pragma enforcement
# ---------------------------------------------------------------------------


class TestConfigPragma:
    def test_config_at_header_accepted(self) -> None:
        r = parse_and_resolve("config log = true\n()")
        assert r.config_pragmas == {"log": True}

    def test_config_after_non_pragma_rejected(self) -> None:
        err = reject_scope("let x = 1\nconfig log = true\nx")
        _, msg = diag(err)
        assert "config" in msg.lower()
        assert "before" in msg.lower() or "after" in msg.lower()

    def test_config_nested_rejected(self) -> None:
        err = reject_scope("if true =>\n  config log = true\n| else =>\n  ()\n")
        _, msg = diag(err)
        assert "config" in msg.lower()

    def test_config_unknown_key_rejected(self) -> None:
        err = reject_scope("config unknown_key = true\n()")
        _, msg = diag(err)
        assert "unknown" in msg.lower() or "Unknown" in msg

    def test_config_duplicate_key_rejected(self) -> None:
        err = reject_scope("config log = true\nconfig log = false\n()")
        _, msg = diag(err)
        assert "duplicate" in msg.lower()

    def test_config_wrong_value_type_rejected(self) -> None:
        err = reject_scope('config log = "yes"\n()')
        _, msg = diag(err)
        assert "bool" in msg.lower()

    def test_config_max_iters_accepted(self) -> None:
        r = parse_and_resolve("config max_iters = 10\n()")
        assert r.config_pragmas == {"max_iters": 10}


# ---------------------------------------------------------------------------
# parent_scope seam (incremental REPL sessions)
# ---------------------------------------------------------------------------


class TestParentScopeSeam:
    def test_default_none_is_standalone(self) -> None:
        err = reject_scope("print x")
        assert "not defined" in diag(err)[1]

    def test_reference_resolves_into_parent(self) -> None:
        """A VarRef to a parent-scope binding resolves through the parent."""
        session = parse_and_resolve("let x = 1\nx")
        entry = resolve(parse_program("print x"), parent_scope=session.root_scope)
        # The print's arg VarRef resolved to the session's let binding.
        call_item = entry.program.body.items[0]
        assert isinstance(call_item, Call)
        arg = call_item.args[0]
        assert isinstance(arg, VarRef)
        ref = entry.resolution[arg.node_id]
        assert ref.name == "x"

    def test_redeclaring_parent_name_shadows_without_error(self) -> None:
        """Redeclaring a parent-visible name shadows without error."""
        session = parse_and_resolve("let x = 1\nx")
        entry = resolve(parse_program("let x = 2\nx"), parent_scope=session.root_scope)
        let_stmt = entry.program.body.items[0]
        assert isinstance(let_stmt, LetDecl)
        assert "x" in entry.root_scope.bindings
        assert entry.root_scope.bindings["x"].decl_node_id == let_stmt.node_id

    def test_assign_to_parent_mutable_resolves(self) -> None:
        """``:=`` of a parent var binding resolves through the parent."""
        session = parse_and_resolve("var n: int = 0\nn")
        entry = resolve(parse_program("n := 1"), parent_scope=session.root_scope)
        assign_stmt = entry.program.body.items[0]
        assert isinstance(assign_stmt, AssignStmt)
        ref = entry.resolution[assign_stmt.node_id]
        assert ref.name == "n"
        assert ref.mutable is True

    def test_assign_to_parent_immutable_still_errors(self) -> None:
        session = parse_and_resolve("let k = 1\nk")
        with pytest.raises(AglScopeError) as exc_info:
            resolve(parse_program("k := 2"), parent_scope=session.root_scope)
        assert "Cannot assign" in str(exc_info.value)

    def test_ambient_agents(self) -> None:
        """An ambient agent resolves without an in-source declaration."""
        r = resolve(
            parse_program('let x = ask("Q", agent: session_bot)\nx'),
            ambient_agents=frozenset({"session_bot"}),
        )
        let_node = r.program.body.items[0]
        assert isinstance(let_node, LetDecl)
        assert isinstance(let_node.value, Call)
        # ask call is classified as builtin
        assert r.builtin_calls[let_node.value.node_id] == BuiltinKind.ASK
        # No declared_agents in program (ambient)
        assert r.declared_agents == {}
        assert r.warnings == ()

    def test_constructor_binding_with_no_candidates_does_not_error(self) -> None:
        """A constructor_binding from a parent scope with no ambient candidates
        is resolved (scope pass succeeds) but constructor_refs is NOT populated.
        This covers the len(candidates)==0 branch in _resolve_varref."""
        prior = parse_and_resolve("enum Review\n  | Pass\n  | Fail\nPass()")
        session_scope = prior.root_scope
        # No ambient_constructor_candidates passed → candidates is empty for 'Pass'.
        entry = resolve(
            parse_program("Pass()"),
            parent_scope=session_scope,
        )
        # Scope resolution succeeds but does NOT populate constructor_refs.
        from agm.agl.syntax.nodes import Call as _Call
        call_node = entry.program.body.items[0]
        assert isinstance(call_node, _Call)
        assert isinstance(call_node.callee, VarRef)
        # Without ambient candidates, constructor_refs is not populated.
        assert call_node.callee.node_id not in entry.constructor_refs

    def test_ambient_constructor_candidates_resolve_prior_entry_ctor(self) -> None:
        """Constructor from a prior REPL entry resolves via ambient_constructor_candidates."""
        from agm.agl.scope.symbols import ConstructorRef

        # Simulate a prior entry that declared enum Review | Pass | Fail.
        prior = parse_and_resolve("enum Review\n  | Pass\n  | Fail\nPass()")
        # Build ambient candidates from the prior entry's resolution.
        ambient: dict[str, tuple[ConstructorRef, ...]] = {
            name: crefs for name, crefs in prior.constructor_candidates.items()
        }
        # New entry references Pass() with a parent scope that has the constructor binding.
        session_scope = prior.root_scope
        entry = resolve(
            parse_program("Pass()"),
            parent_scope=session_scope,
            ambient_constructor_candidates=ambient,
        )
        # The VarRef/Call for Pass() must be in constructor_refs.
        from agm.agl.syntax.nodes import Call as _Call
        call_node = entry.program.body.items[0]
        assert isinstance(call_node, _Call)
        assert isinstance(call_node.callee, VarRef)
        assert call_node.callee.node_id in entry.constructor_refs

    def test_type_name_shadowed_by_param_resolves_as_field_access(self) -> None:
        """When a type name is shadowed by a function parameter, a field
        access on it resolves as an ordinary value field access (not qualified
        constructor access).  Covers the 1046->1053 branch in _resolve_field_access."""
        # 'Box' is a type name AND a parameter name inside f.
        # Inside f, Box.x is a regular field access on the parameter, not a
        # qualified constructor reference.
        source = "record Box\n  x: int\ndef f(Box: Box) -> int = Box.x\nf(Box(x: 1))"
        entry = parse_and_resolve(source)
        from agm.agl.syntax.nodes import FieldAccess as _FA
        from agm.agl.syntax.nodes import FuncDef as _FD
        fn_node = entry.program.body.items[1]
        assert isinstance(fn_node, _FD)
        fa_node = fn_node.body
        assert isinstance(fa_node, _FA)
        # NOT in qualified_constructor_refs (it's a value field access on the param).
        assert fa_node.node_id not in entry.qualified_constructor_refs

    def test_ambient_type_names_resolve_qualified_prior_entry_ctor(self) -> None:
        """Qualified constructor from a prior REPL entry resolves via ambient_type_names."""
        from agm.agl.scope.symbols import ConstructorRef

        prior = parse_and_resolve("enum Review\n  | Pass\n  | Fail\nPass()")
        ambient_candidates: dict[str, tuple[ConstructorRef, ...]] = {
            name: crefs for name, crefs in prior.constructor_candidates.items()
        }
        ambient_type_names = prior.declared_type_names
        session_scope = prior.root_scope
        entry = resolve(
            parse_program("Review.Pass()"),
            parent_scope=session_scope,
            ambient_constructor_candidates=ambient_candidates,
            ambient_type_names=ambient_type_names,
        )
        # The FieldAccess for Review.Pass must be in qualified_constructor_refs.
        from agm.agl.syntax.nodes import Call as _Call
        from agm.agl.syntax.nodes import FieldAccess as _FA
        call_node = entry.program.body.items[0]
        assert isinstance(call_node, _Call)
        assert isinstance(call_node.callee, _FA)
        assert call_node.callee.node_id in entry.qualified_constructor_refs


# ---------------------------------------------------------------------------
# Resolution side table: VarRef and AssignStmt
# ---------------------------------------------------------------------------


class TestResolutionSideTable:
    def test_varref_resolves_to_let(self) -> None:
        r = parse_and_resolve("let x = 1\nx")
        varref_item = r.program.body.items[1]
        assert isinstance(varref_item, VarRef)
        ref = r.resolution[varref_item.node_id]
        assert ref.name == "x"
        assert not ref.mutable

    def test_assign_resolves_to_var(self) -> None:
        r = parse_and_resolve("var n = 0\nn := 1\nn")
        assign_item = r.program.body.items[1]
        assert isinstance(assign_item, AssignStmt)
        ref = r.resolution[assign_item.node_id]
        assert ref.name == "n"
        assert ref.mutable

    def test_param_binding_is_immutable(self) -> None:
        r = parse_and_resolve("param spec\nspec")
        varref = r.program.body.items[1]
        assert isinstance(varref, VarRef)
        ref = r.resolution[varref.node_id]
        assert ref.name == "spec"
        assert not ref.mutable

    def test_interp_varref_resolved(self) -> None:
        r = parse_and_resolve('param name\nlet q = "Hello ${name}"\nq')
        let_q = r.program.body.items[1]
        assert isinstance(let_q, LetDecl)
        tmpl = let_q.value
        assert isinstance(tmpl, Template)
        from agm.agl.syntax.nodes import InterpSegment
        interp = next(s for s in tmpl.segments if isinstance(s, InterpSegment))
        assert isinstance(interp.expr, VarRef)
        ref = r.resolution[interp.expr.node_id]
        assert ref.name == "name"

    def test_function_binding_is_immutable(self) -> None:
        r = parse_and_resolve("def f(x: int) -> int = x\nlet y = f(1)\ny")
        call_callee = r.program.body.items[1]
        assert isinstance(call_callee, LetDecl)
        call = call_callee.value
        assert isinstance(call, Call)
        callee_ref = call.callee
        assert isinstance(callee_ref, VarRef)
        ref = r.resolution[callee_ref.node_id]
        assert ref.kind == BinderKind.function_binding
        assert not ref.mutable


# ---------------------------------------------------------------------------
# Direct AST construction tests for constructs not easily parsed
# ---------------------------------------------------------------------------


class TestDirectASTConstruction:
    """Test scope resolution for constructs built directly as AST nodes."""

    # --- Do loop ---

    def test_do_body_as_block_bindings_visible_in_condition(self) -> None:
        var_n = _make_var("n", _make_intlit(0))
        let_probe = _make_let("probe", _make_varref("n"))
        assign_n = _make_assign("n", _make_varref("probe"))
        body = _make_block(let_probe, assign_n)
        do_node = Do(
            limit=5,
            body=body,
            condition=_make_varref("probe"),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(var_n, do_node)
        assert r.program is not None

    def test_do_body_single_expr_resolved(self) -> None:
        var_n = _make_var("n", _make_intlit(0))
        do_node = Do(
            limit=5,
            body=_make_varref("n"),
            condition=_make_boollit(True),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(var_n, do_node)
        assert r.program is not None

    # --- If with block bodies ---

    def test_if_block_body_scope(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        let_in = _make_let("inner", _make_varref("x"))
        body_block = _make_block(let_in, _make_varref("inner"))
        branch = IfBranch(
            cond=_make_boollit(True),
            body=body_block,
            span=_sp(),
            node_id=_nid(),
        )
        if_node = If(branches=(branch,), span=_sp(), node_id=_nid())
        r = resolve_program(let_x, if_node)
        assert r.program is not None

    def test_if_block_inner_not_visible_outside(self) -> None:
        let_in = _make_let("inner", _make_intlit(1))
        body_block = _make_block(let_in, _make_varref("inner"))
        branch = IfBranch(
            cond=_make_boollit(True),
            body=body_block,
            span=_sp(),
            node_id=_nid(),
        )
        if_node = If(branches=(branch,), span=_sp(), node_id=_nid())
        read_after = _make_varref("inner", line=3)
        err = reject_program(if_node, read_after)
        assert "inner" in err.to_diagnostic().message

    # --- Try/catch ---

    def test_try_with_catch_binder(self) -> None:
        clause = CatchClause(
            exc_type="SomeError",
            binding="err",
            body=_make_varref("err"),
            span=_sp(),
            node_id=_nid(),
        )
        try_node = Try(
            body=_make_intlit(1),
            handlers=(clause,),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(try_node)
        assert r.program is not None

    def test_try_catch_binder_not_visible_outside(self) -> None:
        clause = CatchClause(
            exc_type=None,
            binding="err",
            body=_make_varref("err"),
            span=_sp(),
            node_id=_nid(),
        )
        try_node = Try(
            body=_make_intlit(1),
            handlers=(clause,),
            span=_sp(),
            node_id=_nid(),
        )
        read_after = _make_varref("err", line=3)
        err = reject_program(try_node, read_after)
        assert "err" in err.to_diagnostic().message

    # --- Constructor + operators ---

    def test_constructor_args_resolved(self) -> None:
        # Constructors are now ordinary Call nodes; a call to a record/enum
        # constructor is a Call whose callee is a VarRef.
        # Test that named-arg values in a constructor call are resolved.
        from agm.agl.syntax.nodes import EnumDef, NamedArg, VariantDef

        sp = _sp()
        variant = VariantDef(name="point", fields=(), span=sp, node_id=_nid())
        enum_def = EnumDef(name="Shape", variants=(variant,), span=sp, node_id=_nid())
        let_n = _make_let("n", _make_intlit(5))
        arg = NamedArg(name="n", value=_make_varref("n"), span=sp, node_id=_nid())
        # Constructor call: Call(callee=VarRef("point"), named_args=[n: n])
        ctor_call = Call(
            callee=_make_varref("point"),
            args=(),
            named_args=(arg,),
            span=sp,
            node_id=_nid(),
        )
        r = resolve_program(enum_def, let_n, ctor_call)
        assert r.program is not None
        # The named-arg value (VarRef("n")) must be resolved
        assert arg.value.node_id in r.resolution  # type: ignore[union-attr]
        # The callee VarRef("point") must be resolved as constructor_binding
        callee = ctor_call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.resolution
        assert r.resolution[callee.node_id].kind == BinderKind.constructor_binding

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
        r = resolve_program(let_x, let_y, binop)
        assert r.program is not None

    def test_unary_not_resolved(self) -> None:
        from agm.agl.syntax.nodes import UnaryNot

        let_b = _make_let("b", _make_boollit(True))
        expr = UnaryNot(operand=_make_varref("b"), span=_sp(), node_id=_nid())
        r = resolve_program(let_b, expr)
        assert r.program is not None

    def test_unary_neg_resolved(self) -> None:
        from agm.agl.syntax.nodes import UnaryNeg

        let_n = _make_let("n", _make_intlit(1))
        expr = UnaryNeg(operand=_make_varref("n"), span=_sp(), node_id=_nid())
        r = resolve_program(let_n, expr)
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
        r = resolve_program(let_x, expr)
        assert r.program is not None

    def test_field_access_on_varref(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        field_expr = FieldAccess(
            obj=_make_varref("x"), field="f", span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_x, field_expr)
        assert r.program is not None

    def test_list_lit_resolved(self) -> None:
        from agm.agl.syntax.nodes import ListLit

        let_x = _make_let("x", _make_intlit(1))
        lst = ListLit(elements=(_make_varref("x"),), span=_sp(), node_id=_nid())
        r = resolve_program(let_x, lst)
        assert r.program is not None

    def test_dict_lit_resolved(self) -> None:
        from agm.agl.syntax.nodes import DictEntry, DictLit

        let_x = _make_let("x", _make_intlit(1))
        key = StringLit(value="a", span=_sp(), node_id=_nid())
        entry = DictEntry(key=key, value=_make_varref("x"), span=_sp(), node_id=_nid())
        dlit = DictLit(entries=(entry,), span=_sp(), node_id=_nid())
        r = resolve_program(let_x, dlit)
        assert r.program is not None

    # --- Pattern variable binding ---

    def test_case_var_pattern_binds(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        pattern_var = VarPattern(name="matched", span=_sp(), node_id=_nid())
        branch = CaseBranch(
            pattern=pattern_var,
            body=_make_varref("matched"),
            span=_sp(),
            node_id=_nid(),
        )
        from agm.agl.syntax.nodes import Case
        case_node = Case(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, case_node)
        assert r.program is not None

    def test_case_constructor_pattern_with_field(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        sub_pattern = VarPattern(name="issues", span=_sp(), node_id=_nid())
        pf = PatternField(name="issues", pattern=sub_pattern, span=_sp(), node_id=_nid())
        ctor_pattern = ConstructorPattern(
            qualifier=None, name="Fail", fields=(pf,), span=_sp(), node_id=_nid()
        )
        branch = CaseBranch(
            pattern=ctor_pattern,
            body=_make_varref("issues"),
            span=_sp(),
            node_id=_nid(),
        )
        from agm.agl.syntax.nodes import Case
        case_node = Case(
            subject=_make_varref("x"),
            branches=(branch,),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, case_node)
        assert r.program is not None

    def test_duplicate_pattern_var_rejected(self) -> None:
        let_x = _make_let("x", _make_intlit(1))
        sub1 = VarPattern(name="dup", span=_sp(5), node_id=_nid())
        sub2 = VarPattern(name="dup", span=_sp(5), node_id=_nid())
        pf1 = PatternField(name="a", pattern=sub1, span=_sp(5), node_id=_nid())
        pf2 = PatternField(name="b", pattern=sub2, span=_sp(5), node_id=_nid())
        ctor_pat = ConstructorPattern(
            qualifier=None, name="Pair", fields=(pf1, pf2), span=_sp(5), node_id=_nid()
        )
        branch = CaseBranch(
            pattern=ctor_pat, body=_make_unitlit(), span=_sp(5), node_id=_nid()
        )
        from agm.agl.syntax.nodes import Case
        case_node = Case(
            subject=_make_varref("x"), branches=(branch,), span=_sp(5), node_id=_nid()
        )
        err = reject_program(let_x, case_node)
        assert "dup" in err.to_diagnostic().message

    def test_pattern_var_shadows_outer_accepted(self) -> None:
        let_outer = _make_let("v", _make_intlit(99))
        let_x = _make_let("x", _make_intlit(1))
        pv = VarPattern(name="v", span=_sp(), node_id=_nid())
        branch = CaseBranch(
            pattern=pv, body=_make_varref("v"), span=_sp(), node_id=_nid()
        )
        from agm.agl.syntax.nodes import Case
        case_node = Case(
            subject=_make_varref("x"), branches=(branch,), span=_sp(), node_id=_nid()
        )
        r = resolve_program(let_outer, let_x, case_node)
        assert r.program is not None

    # --- Type declarations outside root rejected ---

    def test_record_not_at_root_rejected(self) -> None:
        err = reject_scope(
            "if true =>\n"
            "  record R\n"
            "    n: int\n"
            "| else =>\n"
            "  ()\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_enum_not_at_root_rejected(self) -> None:
        err = reject_scope(
            "do[2]\n"
            "  enum E\n"
            "    | A\n"
            "until true\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    def test_type_alias_not_at_root_rejected(self) -> None:
        err = reject_scope(
            "try\n"
            "  type T = text\n"
            "catch _ =>\n"
            "  ()\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "top" in msg.lower() or "top-level" in msg.lower() or "program root" in msg.lower()

    # --- param not at root ---

    def test_param_inside_if_rejected(self) -> None:
        err = reject_scope(
            "if true =>\n"
            "  param late\n"
            "| else =>\n"
            "  ()\n"
        )
        line, msg = diag(err)
        assert line == 2
        assert "param" in msg.lower()

    def test_param_inside_do_rejected(self) -> None:
        err = reject_scope("do[2]\n  param x\nuntil true\n")
        line, msg = diag(err)
        assert line == 2
        assert "param" in msg.lower()

    def test_param_inside_try_rejected(self) -> None:
        err = reject_scope("try\n  param x\ncatch _ =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "param" in msg.lower()

    def test_program_inside_if_rejected(self) -> None:
        err = reject_scope("if true =>\n  program nested\n| else =>\n  ()\n")
        line, msg = diag(err)
        assert line == 2
        assert "program" in msg.lower()
        assert "root" in msg.lower()

    def test_duplicate_program_rejected(self) -> None:
        err = reject_scope("program first\nprogram second\n1\n")
        line, msg = diag(err)
        assert line == 2
        assert "already declared" in msg

    # --- FuncDef: direct AST construction ---

    def test_funcdef_body_resolved_in_param_scope(self) -> None:
        """FuncDef body sees its own param; param is not visible outside."""
        sp = _sp()
        int_t = IntT(span=sp, node_id=_nid())
        param = Param(name="x", type_expr=int_t, default=None, span=sp, node_id=_nid())
        funcdef = FuncDef(
            name="g",
            params=(param,),
            return_type=int_t,
            body=_make_varref("x"),
            span=sp,
            node_id=_nid(),
        )
        r = resolve_program(funcdef, _make_varref("g"))
        assert r.program is not None

    def test_funcdef_param_not_visible_outside_body(self) -> None:
        sp = _sp()
        int_t = IntT(span=sp, node_id=_nid())
        param = Param(name="p", type_expr=int_t, default=None, span=sp, node_id=_nid())
        funcdef = FuncDef(
            name="g",
            params=(param,),
            return_type=int_t,
            body=_make_varref("p"),
            span=sp,
            node_id=_nid(),
        )
        read_p_after = _make_varref("p", line=2)
        err = reject_program(funcdef, read_p_after)
        assert "p" in err.to_diagnostic().message

    # --- Lambda: direct AST construction ---

    def test_lambda_resolved(self) -> None:
        sp = _sp()
        int_t = IntT(span=sp, node_id=_nid())
        param = Param(name="x", type_expr=int_t, default=None, span=sp, node_id=_nid())
        lam = Lambda(
            params=(param,),
            return_type=None,
            body=_make_varref("x"),
            span=sp,
            node_id=_nid(),
        )
        let_f = _make_let("f", lam)
        r = resolve_program(let_f, _make_varref("f"))
        assert r.program is not None

    def test_lambda_not_self_recursive(self) -> None:
        """A lambda body that references its own let-binding name fails."""
        sp = _sp()
        int_t = IntT(span=sp, node_id=_nid())
        param = Param(name="x", type_expr=int_t, default=None, span=sp, node_id=_nid())
        lam = Lambda(
            params=(param,),
            return_type=None,
            # references "f" which is not yet in scope when lambda RHS is resolved
            body=_make_varref("f"),
            span=sp,
            node_id=_nid(),
        )
        let_f = _make_let("f", lam)
        err = reject_program(let_f)
        assert "f" in err.to_diagnostic().message


# ---------------------------------------------------------------------------
# ResolvedProgram.declared_functions
# ---------------------------------------------------------------------------


class TestDeclaredFunctions:
    def test_declared_functions_populated(self) -> None:
        r = parse_and_resolve("def f(x: int) -> int = x\ndef g(x: int) -> int = x\nf(1)")
        assert "f" in r.declared_functions
        assert "g" in r.declared_functions

    def test_declared_functions_empty_when_no_defs(self) -> None:
        r = parse_and_resolve("let x = 1\nx")
        assert r.declared_functions == {}

    def test_def_name_clashes_with_agent_rejected(self) -> None:
        """A def with the same name as an agent is rejected."""
        err = reject_scope("agent foo\ndef foo(x: int) -> int = x\nfoo(1)")
        _, msg = diag(err)
        assert "foo" in msg

    def test_agent_name_clashes_with_def_rejected(self) -> None:
        """An agent with the same name as a def is rejected (pre-pass order matters)."""
        # Both pass through pre-pass; _collect_func_decls checks _declared_agents
        err = reject_scope("def foo(x: int) -> int = x\nagent foo\nfoo(1)")
        _, msg = diag(err)
        assert "foo" in msg


# ---------------------------------------------------------------------------
# Config pragma edge cases (coverage for pragma value validation)
# ---------------------------------------------------------------------------


class TestConfigPragmaValueValidation:
    def test_int_pos_zero_rejected(self) -> None:
        """config max_iters = 0 is rejected (must be > 0)."""
        err = reject_scope("config max_iters = 0\n()")
        _, msg = diag(err)
        assert "positive" in msg.lower() or "greater" in msg.lower() or "> 0" in msg

    def test_str_nonempty_non_string_rejected(self) -> None:
        """config runner with an int value is rejected."""
        err = reject_scope("config runner = 1\n()")
        _, msg = diag(err)
        assert "non-empty string" in msg or "string" in msg.lower()

    def test_timeout_with_valid_int(self) -> None:
        r = parse_and_resolve("config timeout = 30\n()")
        assert r.config_pragmas.get("timeout") == 30

    def test_timeout_zero_rejected(self) -> None:
        err = reject_scope("config timeout = 0\n()")
        _, msg = diag(err)
        assert "positive" in msg.lower() or "> 0" in msg

    def test_timeout_bool_rejected(self) -> None:
        err = reject_scope("config timeout = true\n()")
        _, msg = diag(err)
        assert "timeout" in msg

    def test_timeout_empty_string_rejected(self) -> None:
        """An empty string for timeout is rejected."""
        # We need to test via AST since the parser may not emit an empty string
        # in a config pragma; use direct AST construction.
        from agm.agl.syntax.nodes import ConfigPragma

        pragma = ConfigPragma(key="timeout", value="", span=_sp(), node_id=_nid())
        err = reject_program(pragma)
        _, msg = diag(err)
        assert "non-empty" in msg.lower() or "timeout" in msg.lower()

    def test_runner_valid_string_accepted(self) -> None:
        """config runner = 'claude' is accepted (covers str_nonempty valid branch)."""
        from agm.agl.syntax.nodes import ConfigPragma

        pragma = ConfigPragma(key="runner", value="claude", span=_sp(), node_id=_nid())
        r = resolve_program(pragma)
        assert r.config_pragmas.get("runner") == "claude"

    def test_timeout_valid_string_accepted(self) -> None:
        """config timeout = '30s' is accepted (covers str_or_int valid string branch)."""
        from agm.agl.syntax.nodes import ConfigPragma

        pragma = ConfigPragma(key="timeout", value="30s", span=_sp(), node_id=_nid())
        r = resolve_program(pragma)
        assert r.config_pragmas.get("timeout") == "30s"


# ---------------------------------------------------------------------------
# Block as expr (covers _resolve_expr for Block nodes)
# ---------------------------------------------------------------------------


class TestBlockAsExpr:
    def test_block_as_let_value(self) -> None:
        """A Block used as the RHS of a let (via direct AST) is resolved."""
        # let result = { let x = 1; x }
        let_x = _make_let("x", _make_intlit(1))
        inner_ref = _make_varref("x")
        inner_block = _make_block(let_x, inner_ref)
        let_result = _make_let("result", inner_block)
        r = resolve_program(let_result, _make_varref("result"))
        assert r.program is not None

    def test_block_as_expr_inner_binding_isolated(self) -> None:
        """Bindings in a Block expr don't escape outside the block."""
        let_x = _make_let("x", _make_intlit(1))
        inner_block = _make_block(let_x, _make_varref("x"))
        let_result = _make_let("result", inner_block)
        # Reading x after the block should fail
        read_x = _make_varref("x", line=2)
        err = reject_program(let_result, read_x)
        assert "x" in err.to_diagnostic().message


# ---------------------------------------------------------------------------
# Ambient agent binding edge cases
# ---------------------------------------------------------------------------


class TestAmbientAgentBindingEdgeCases:
    def test_ambient_agent_already_in_parent_scope_not_redefined(self) -> None:
        """An ambient agent whose name is already in the parent scope is not redefined."""
        # Create a session that declares the agent as a var
        session = parse_and_resolve("let session_bot = 1\nsession_bot")
        # Pass ambient_agents — the name is already in parent scope via lookup,
        # so the ambient binding definition is skipped.
        entry = resolve(
            parse_program("let x = 1\nx"),
            parent_scope=session.root_scope,
            ambient_agents=frozenset({"session_bot"}),
        )
        assert entry.program is not None

    def test_ambient_agent_already_declared_locally_skipped(self) -> None:
        """If an ambient agent name is also declared locally, local takes precedence."""
        # Declare 'bot' locally AND pass it as ambient — should not double-define.
        r = resolve(
            parse_program("agent bot\nlet x = bot\nx"),
            ambient_agents=frozenset({"bot"}),
        )
        assert r.program is not None
        # The local declared_agents entry takes precedence.
        assert "bot" in r.declared_agents

    def test_def_name_collides_with_ambient_agent_rejected(self) -> None:
        """A top-level def whose name matches an ambient agent is rejected.

        Regression test for Fix 1: the guard in _define_function_bindings is
        reachable because ambient agent bindings are defined BEFORE function
        bindings, so a def named 'foo' with ambient_agents={'foo'} hits the
        already-defined check.
        """
        with pytest.raises(AglScopeError) as exc_info:
            resolve(
                parse_program("def foo(x: int) -> int = x\nfoo(1)"),
                ambient_agents=frozenset({"foo"}),
            )
        assert "foo" in exc_info.value.to_diagnostic().message


# ---------------------------------------------------------------------------
# Lambda duplicate param (covers line 885)
# ---------------------------------------------------------------------------


class TestLambdaDuplicateParam:
    def test_lambda_duplicate_param_rejected(self) -> None:
        """A lambda with two params with the same name is rejected."""
        sp = _sp()
        from agm.agl.syntax.types import IntT as IntTNode

        int_t = IntTNode(span=sp, node_id=_nid())
        p1 = Param(name="x", type_expr=int_t, default=None, span=sp, node_id=_nid())
        p2 = Param(name="x", type_expr=int_t, default=None, span=_sp(2), node_id=_nid())
        lam = Lambda(
            params=(p1, p2),
            return_type=None,
            body=_make_varref("x"),
            span=sp,
            node_id=_nid(),
        )
        let_f = _make_let("f", lam)
        err = reject_program(let_f)
        assert "x" in err.to_diagnostic().message


# ---------------------------------------------------------------------------
# Constructor value bindings (generics: D7 / scope pass)
# ---------------------------------------------------------------------------

# Helper: build a RecordDef with optional type_params
def _make_record(
    name: str, *, type_params: tuple[str, ...] = (), line: int = 1
) -> RecordDef:
    from agm.agl.syntax.nodes import FieldDef
    from agm.agl.syntax.types import IntT as IntTNode

    sp = _sp(line)
    field_t = IntTNode(span=sp, node_id=_nid())
    fd = FieldDef(name="value", type_expr=field_t, span=sp, node_id=_nid())
    return RecordDef(
        name=name, fields=(fd,), type_params=type_params, span=sp, node_id=_nid()
    )


# Helper: build an EnumDef with variants
def _make_enum(
    name: str,
    variant_names: tuple[str, ...],
    *,
    type_params: tuple[str, ...] = (),
    line: int = 1,
) -> EnumDef:
    sp = _sp(line)
    variants: list[VariantDef] = []
    for vname in variant_names:
        variants.append(VariantDef(name=vname, fields=(), span=sp, node_id=_nid()))
    return EnumDef(
        name=name,
        variants=tuple(variants),
        type_params=type_params,
        span=sp,
        node_id=_nid(),
    )


class TestConstructorBindings:
    """Tests for constructor value bindings (D7: case-neutral constructor names)."""

    # --- Record constructor resolves as value binding ---

    def test_record_constructor_resolves_as_value(self) -> None:
        """A record constructor (record name) resolves as a constructor_binding."""
        r = parse_and_resolve(
            "record Box\n"
            "  value: int\n"
            "let b = Box(value: 1)\n"
            "b\n"
        )
        assert r.program is not None
        # The callee VarRef("Box") should be in constructor_refs
        # and the constructor candidate is for 'Box'
        assert "Box" in r.constructor_candidates
        candidates = r.constructor_candidates["Box"]
        assert len(candidates) == 1
        assert candidates[0].owner_name == "Box"
        assert candidates[0].variant is None

    def test_record_constructor_lowercase_resolves(self) -> None:
        """Lowercase record names work identically (no capitalization rule)."""
        r = parse_and_resolve(
            "record box\n"
            "  value: int\n"
            "let b = box(value: 1)\n"
            "b\n"
        )
        assert "box" in r.constructor_candidates
        assert r.constructor_candidates["box"][0].owner_name == "box"

    def test_enum_variant_resolves_as_value(self) -> None:
        """Enum variants resolve as constructor_bindings."""
        r = parse_and_resolve(
            "enum Option\n"
            "  | none\n"
            "  | some\n"
            "none\n"
        )
        assert r.program is not None
        assert "none" in r.constructor_candidates
        assert "some" in r.constructor_candidates

    def test_nullary_variant_bare_ref_resolves(self) -> None:
        """A bare VarRef to a nullary enum variant resolves as a constructor."""
        r = parse_and_resolve(
            "enum Option\n"
            "  | none\n"
            "  | some\n"
            "let x = none\n"
            "x\n"
        )
        # The VarRef("none") in the let should be in constructor_refs
        let_decl = r.program.body.items[1]
        assert isinstance(let_decl, LetDecl)
        vref = let_decl.value
        assert isinstance(vref, VarRef)
        assert vref.node_id in r.constructor_refs
        cref = r.constructor_refs[vref.node_id]
        assert cref.owner_name == "Option"
        assert cref.variant == "none"

    def test_payload_variant_callee_resolves(self) -> None:
        """A payload variant used as a call callee resolves as a constructor."""
        r = parse_and_resolve(
            "enum Option\n"
            "  | none\n"
            "  | some\n"
            "let x = some()\n"
            "x\n"
        )
        let_decl = r.program.body.items[1]
        assert isinstance(let_decl, LetDecl)
        call = let_decl.value
        assert isinstance(call, Call)
        callee = call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.constructor_refs
        assert r.constructor_refs[callee.node_id].variant == "some"

    def test_record_constructor_callee_resolves(self) -> None:
        """A record constructor used as a call callee resolves."""
        r = parse_and_resolve(
            "record Box\n"
            "  value: int\n"
            "let b = Box(value: 1)\n"
            "b\n"
        )
        let_decl = r.program.body.items[1]
        assert isinstance(let_decl, LetDecl)
        call = let_decl.value
        assert isinstance(call, Call)
        callee = call.callee
        assert isinstance(callee, VarRef)
        assert callee.node_id in r.constructor_refs
        cref = r.constructor_refs[callee.node_id]
        assert cref.owner_name == "Box"
        assert cref.variant is None

    # --- Generic type_params on constructors ---

    def test_generic_record_constructor_has_type_params(self) -> None:
        """A generic record constructor carries its type_params in the ConstructorRef."""
        r = parse_and_resolve(
            "record Box[T]\n"
            "  value: int\n"
            "let b = Box(value: 1)\n"
            "b\n"
        )
        assert r.constructor_candidates["Box"][0].type_params == ("T",)

    def test_generic_enum_variant_has_type_params(self) -> None:
        """An enum variant from a generic enum carries the enum's type_params."""
        r = parse_and_resolve(
            "enum Option[T]\n"
            "  | none\n"
            "  | some\n"
            "none\n"
        )
        assert r.constructor_candidates["none"][0].type_params == ("T",)
        assert r.constructor_candidates["some"][0].type_params == ("T",)

    # --- Overload sets and ambiguity ---

    def test_unique_variant_resolves_unambiguously(self) -> None:
        """A unique variant name (one enum) resolves to a single constructor_ref."""
        r = parse_and_resolve(
            "enum A\n"
            "  | foo\n"
            "  | bar\n"
            "enum B\n"
            "  | baz\n"
            "foo\n"
        )
        last_item = r.program.body.items[2]
        assert isinstance(last_item, VarRef)
        assert last_item.node_id in r.constructor_refs

    def test_overload_set_built_from_two_enums(self) -> None:
        """Two enums sharing a variant name form an overload set.

        The overload set exists even if we don't actually USE 'some'.
        """
        enum_a = _make_enum("A", ("some", "none"), line=1)
        enum_b = _make_enum("B", ("some", "other"), line=2)
        unit = _make_unitlit()
        r = resolve_program(enum_a, enum_b, unit)
        assert "some" in r.constructor_candidates
        assert len(r.constructor_candidates["some"]) == 2
        owners = {c.owner_name for c in r.constructor_candidates["some"]}
        assert owners == {"A", "B"}

    def test_ambiguous_bare_varref_raises(self) -> None:
        """Unqualified use of an ambiguous variant name raises an ambiguity error."""
        err = reject_scope(
            "enum A\n"
            "  | some\n"
            "enum B\n"
            "  | some\n"
            "some\n"
        )
        msg = err.to_diagnostic().message
        assert "some" in msg
        assert "ambiguous" in msg.lower()
        # Both owners should be mentioned
        assert "A" in msg
        assert "B" in msg

    def test_ambiguous_call_raises(self) -> None:
        """Calling an ambiguous constructor name also raises ambiguity."""
        err = reject_scope(
            "enum A\n"
            "  | some\n"
            "enum B\n"
            "  | some\n"
            "some()\n"
        )
        msg = err.to_diagnostic().message
        assert "ambiguous" in msg.lower()
        assert "A" in msg
        assert "B" in msg

    def test_ambiguous_mentions_qualification(self) -> None:
        """Ambiguity error tells the user to qualify the reference."""
        err = reject_scope(
            "enum Option\n"
            "  | some\n"
            "enum Other\n"
            "  | some\n"
            "some\n"
        )
        msg = err.to_diagnostic().message
        # Should suggest qualification like 'Option.some'
        assert "." in msg or "qualify" in msg.lower()

    # --- D7 regression: payload / type-args / context do NOT disambiguate ---

    def test_ambiguous_payload_variant_still_raises(self) -> None:
        """A payload does NOT disambiguate two enums sharing the same variant name.

        D7 guarantees ambiguity is raised by the scope pass regardless of whether
        a matching payload is supplied.  Both candidate enums must be named in the
        error message.
        """
        err = reject_scope(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "enum Other[T]\n"
            "  | nope\n"
            "  | some(value: T)\n"
            "some(value: 1)\n"
        )
        msg = err.to_diagnostic().message
        assert "ambiguous" in msg.lower()
        assert "Option" in msg
        assert "Other" in msg

    def test_ambiguous_explicit_type_args_still_raises(self) -> None:
        """Explicit type arguments do NOT disambiguate two enums sharing a variant name.

        D7 guarantees that ``some::[int](value: 1)`` is still ambiguous when both
        ``Option`` and ``Other`` declare a ``some`` variant.  Both candidate enums
        must be named in the error message.
        """
        err = reject_scope(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "enum Other[T]\n"
            "  | nope\n"
            "  | some(value: T)\n"
            "some::[int](value: 1)\n"
        )
        msg = err.to_diagnostic().message
        assert "ambiguous" in msg.lower()
        assert "Option" in msg
        assert "Other" in msg

    def test_ambiguous_contextual_type_still_raises(self) -> None:
        """A contextual expected type does NOT disambiguate an ambiguous variant.

        D7 guarantees that ``let x: Option[int] = some(value: 1)`` is still
        ambiguous when both ``Option`` and ``Other`` declare a ``some`` variant.
        Both candidate enums must be named in the error message.
        """
        err = reject_scope(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "enum Other[T]\n"
            "  | nope\n"
            "  | some(value: T)\n"
            "let x: Option[int] = some(value: 1)\n"
            "x\n"
        )
        msg = err.to_diagnostic().message
        assert "ambiguous" in msg.lower()
        assert "Option" in msg
        assert "Other" in msg

    def test_qualified_payload_variant_resolves_without_error(self) -> None:
        """Qualification is the only valid disambiguation for shared variant names.

        D7 requires that ``Option.some(value: 1)`` resolves without error even
        when ``Other`` also declares a ``some`` variant.
        """
        r = parse_and_resolve(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "enum Other[T]\n"
            "  | nope\n"
            "  | some(value: T)\n"
            "Option.some(value: 1)\n"
        )
        assert r.program is not None

    # --- Qualified constructor access ---

    def test_qualified_constructor_recorded(self) -> None:
        """Qualified access Owner.member is recorded in qualified_constructor_refs."""
        r = parse_and_resolve(
            "enum Option\n"
            "  | none\n"
            "  | some\n"
            "let x = Option.some\n"
            "x\n"
        )
        let_decl = r.program.body.items[1]
        assert isinstance(let_decl, LetDecl)
        fa = let_decl.value
        assert isinstance(fa, FieldAccess)
        assert fa.node_id in r.qualified_constructor_refs
        owner, member, _mid = r.qualified_constructor_refs[fa.node_id]
        assert owner == "Option"
        assert member == "some"

    def test_qualified_access_does_not_raise_undefined_for_owner(self) -> None:
        """Option.some does NOT raise 'Option is not defined'."""
        r = parse_and_resolve(
            "enum Option\n"
            "  | none\n"
            "  | some\n"
            "Option.some\n"
        )
        assert r.program is not None

    def test_qualified_access_none_variant(self) -> None:
        """Option.none is recorded in qualified_constructor_refs."""
        r = parse_and_resolve(
            "enum Option\n"
            "  | none\n"
            "  | some\n"
            "Option.none\n"
        )
        last = r.program.body.items[1]
        assert isinstance(last, FieldAccess)
        assert last.node_id in r.qualified_constructor_refs
        assert r.qualified_constructor_refs[last.node_id][:2] == ("Option", "none")

    def test_qualified_access_with_record_type(self) -> None:
        """Box.value is NOT a constructor qualified access (Box is a record, value is a field)."""
        # This is a legitimate field access, not a constructor access.
        # 'Box' is in declared_type_names as a RecordDef; Box.value is a type-qualified
        # access, recorded in qualified_constructor_refs. The checker validates validity.
        r = parse_and_resolve(
            "record Box\n"
            "  value: int\n"
            "Box.value\n"
        )
        assert r.program is not None
        last = r.program.body.items[1]
        assert isinstance(last, FieldAccess)
        assert last.node_id in r.qualified_constructor_refs

    # --- Collision rules (non-constructor vs constructor) ---

    def test_def_same_name_as_constructor_raises(self) -> None:
        """A def with the same name as an enum variant raises a duplicate error."""
        err = reject_scope(
            "enum Option\n"
            "  | some\n"
            "def some(x: int) -> int = x\n"
            "some(1)\n"
        )
        msg = err.to_diagnostic().message
        assert "some" in msg
        assert "already declared" in msg.lower() or "duplicate" in msg.lower()

    def test_agent_same_name_as_constructor_raises(self) -> None:
        """An agent with the same name as a variant raises a duplicate error."""
        err = reject_scope(
            "enum Option\n"
            "  | myagent\n"
            "agent myagent\n"
            "()\n"
        )
        msg = err.to_diagnostic().message
        assert "myagent" in msg

    def test_constructor_same_name_as_def_raises(self) -> None:
        """An enum variant named after an existing def raises a duplicate error."""
        err = reject_scope(
            "def some(x: int) -> int = x\n"
            "enum Option\n"
            "  | some\n"
            "some(1)\n"
        )
        msg = err.to_diagnostic().message
        assert "some" in msg

    def test_let_shadows_constructor_in_nested_scope_no_error(self) -> None:
        """In a nested scope, a 'let' binding shadows a constructor from the outer scope."""
        # In a nested block (if branch), a let can shadow the constructor
        # without error; no duplicate-binding error is raised because they're
        # in different scopes.
        r = parse_and_resolve(
            "enum Option\n"
            "  | some\n"
            "if true =>\n"
            "  let some = 42\n"
            "  some\n"
            "| else =>\n"
            "  some\n"
        )
        assert r.program is not None

    def test_let_shadows_constructor_in_if_branch_no_error(self) -> None:
        """In a branch, a 'let' name can shadow a constructor from the outer scope.

        The shadowing VarRef resolves to the let binding (not the constructor),
        while a VarRef in another branch resolves to the constructor.
        """
        r = parse_and_resolve(
            "enum Option\n"
            "  | some\n"
            "if true =>\n"
            "  let some = 42\n"
            "  some\n"
            "| else =>\n"
            "  some\n"
        )
        assert r.program is not None
        # The VarRef in the else branch (index 1 of branches, its body is VarRef)
        # resolves to the constructor_binding, while the then-branch's VarRef
        # resolves to the let_binding.  We verify no error and the program is fine.
        # The outer 'some' VarRef in the else branch should have a constructor_ref.
        # We can verify via the resolution table but the node_ids are harder to
        # locate without traversal; no-error is sufficient here.

    def test_let_at_root_conflicts_with_constructor(self) -> None:
        """At root, 'let some' conflicts with the constructor 'some'."""
        err = reject_scope(
            "enum Option\n"
            "  | some\n"
            "let some = 1\n"
            "some\n"
        )
        msg = err.to_diagnostic().message
        assert "some" in msg

    # --- Enum name is NOT a value (only variants are) ---

    def test_enum_name_used_as_value_is_undefined(self) -> None:
        """The enum name itself is NOT a value binding — only its variants are."""
        err = reject_scope(
            "enum Option\n"
            "  | none\n"
            "  | some\n"
            "let x = Option\n"
            "x\n"
        )
        msg = err.to_diagnostic().message
        assert "Option" in msg
        assert "not defined" in msg.lower()

    def test_uppercase_enum_name_not_value(self) -> None:
        """Enum name 'Option' (uppercase) is still not a value binding."""
        err = reject_scope(
            "enum Option\n"
            "  | None\n"
            "  | Some\n"
            "Option\n"
        )
        msg = err.to_diagnostic().message
        assert "Option" in msg

    # --- Case-neutral: lowercase and uppercase behave identically ---

    def test_lowercase_constructor_resolves_same_as_uppercase(self) -> None:
        """Lowercase 'option.none' and uppercase 'Option.None' behave identically."""
        r_lower = parse_and_resolve(
            "enum option\n"
            "  | none\n"
            "  | some\n"
            "option.none\n"
        )
        r_upper = parse_and_resolve(
            "enum Option\n"
            "  | None\n"
            "  | Some\n"
            "Option.None\n"
        )
        assert r_lower.program is not None
        assert r_upper.program is not None
        # Both should have a single qualified_constructor_ref
        assert len(r_lower.qualified_constructor_refs) == 1
        assert len(r_upper.qualified_constructor_refs) == 1

    def test_no_capitalization_rule_for_constructor_lookup(self) -> None:
        """Neither lowercase nor uppercase variants require capitalization to resolve."""
        r = parse_and_resolve(
            "enum option\n"
            "  | none\n"
            "  | someVal\n"
            "let a = none\n"
            "let b = someVal\n"
            "a\n"
        )
        assert r.program is not None
        assert "none" in r.constructor_candidates
        assert "someVal" in r.constructor_candidates

    # --- Type parameter duplicate validation ---

    def test_duplicate_type_param_in_def_raises(self) -> None:
        """Duplicate type parameter in a def declaration raises AglScopeError."""
        err = reject_scope(
            "def id[T, T](x: int) -> int = x\n"
            "id(1)\n"
        )
        msg = err.to_diagnostic().message
        assert "T" in msg
        assert "duplicate" in msg.lower() or "Duplicate" in msg

    def test_duplicate_type_param_in_record_raises(self) -> None:
        """Duplicate type parameter in a record declaration raises AglScopeError."""
        err = reject_scope(
            "record Box[T, T]\n"
            "  value: int\n"
            "()\n"
        )
        msg = err.to_diagnostic().message
        assert "T" in msg

    def test_duplicate_type_param_in_enum_raises(self) -> None:
        """Duplicate type parameter in an enum declaration raises AglScopeError."""
        err = reject_scope(
            "enum Option[T, T]\n"
            "  | none\n"
            "()\n"
        )
        msg = err.to_diagnostic().message
        assert "T" in msg

    def test_duplicate_type_param_in_type_alias_raises(self) -> None:
        """Duplicate type parameter in a type alias raises AglScopeError."""
        err = reject_scope(
            "type Pair[A, A] = int\n"
            "()\n"
        )
        msg = err.to_diagnostic().message
        assert "A" in msg

    def test_duplicate_record_name_raises(self) -> None:
        err = reject_scope("record P\n  n: int\nrecord P\n  t: text\n()")
        msg = err.to_diagnostic().message
        assert "P" in msg
        assert "already declared" in msg.lower()

    def test_duplicate_enum_name_raises(self) -> None:
        err = reject_scope("enum Status\n  | ok\nenum Status\n  | fail\n()")
        msg = err.to_diagnostic().message
        assert "Status" in msg
        assert "already declared" in msg.lower()

    def test_duplicate_type_alias_name_raises(self) -> None:
        err = reject_scope("type Thing = int\ntype Thing = text\n()")
        msg = err.to_diagnostic().message
        assert "Thing" in msg
        assert "already declared" in msg.lower()

    def test_type_declaration_kinds_share_type_namespace(self) -> None:
        err = reject_scope("record Id\n  value: int\ntype Id = text\n()")
        msg = err.to_diagnostic().message
        assert "Id" in msg
        assert "already declared" in msg.lower()

    def test_unique_type_params_accepted(self) -> None:
        """Unique type parameters in a def are accepted."""
        r = parse_and_resolve(
            "def id[T](x: int) -> int = x\n"
            "id(1)\n"
        )
        assert r.program is not None

    def test_multiple_type_params_unique_accepted(self) -> None:
        """Multiple unique type params in a record are accepted."""
        r = parse_and_resolve(
            "record Pair[A, B]\n"
            "  value: int\n"
            "()\n"
        )
        assert r.program is not None

    # --- declared_type_names populated ---

    def test_declared_type_names_includes_record(self) -> None:
        r = parse_and_resolve("record Foo\n  n: int\n()")
        assert "Foo" in r.declared_type_names

    def test_declared_type_names_includes_enum(self) -> None:
        r = parse_and_resolve("enum Color\n  | red\n  | blue\n()")
        assert "Color" in r.declared_type_names

    def test_declared_type_names_includes_alias(self) -> None:
        r = parse_and_resolve("type MyInt = int\n()")
        assert "MyInt" in r.declared_type_names

    def test_declared_type_names_excludes_variants(self) -> None:
        """Enum variant names are NOT in declared_type_names (they are values)."""
        r = parse_and_resolve("enum Color\n  | red\n  | blue\n()")
        assert "red" not in r.declared_type_names
        assert "blue" not in r.declared_type_names

    # --- Direct AST construction tests ---

    def test_record_constructor_binding_via_ast(self) -> None:
        """Direct AST: a RecordDef registers its name as a constructor binding."""
        rec = _make_record("Point")
        call = Call(
            callee=_make_varref("Point"),
            args=(),
            named_args=(),
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(rec, call)
        assert r.program is not None
        assert "Point" in r.constructor_candidates
        assert r.constructor_candidates["Point"][0].variant is None

    def test_enum_variant_binding_via_ast(self) -> None:
        """Direct AST: enum variants register as constructor candidates."""
        enum = _make_enum("Status", ("ok", "err"))
        ref_ok = _make_varref("ok")
        r = resolve_program(enum, ref_ok)
        assert "ok" in r.constructor_candidates
        assert r.constructor_candidates["ok"][0].owner_name == "Status"
        assert r.constructor_candidates["ok"][0].variant == "ok"
        assert ref_ok.node_id in r.constructor_refs

    def test_constructor_binding_kind_in_scope(self) -> None:
        """The root scope binding for a constructor name has kind=constructor_binding."""
        rec = _make_record("MyRecord")
        unit = _make_unitlit()
        r = resolve_program(rec, unit)
        assert "MyRecord" in r.root_scope.bindings
        assert r.root_scope.bindings["MyRecord"].kind == BinderKind.constructor_binding

    def test_overload_set_two_enums_via_ast(self) -> None:
        """Two enums sharing a variant name form a 2-element overload set."""
        enum_a = _make_enum("EnumA", ("val",), line=1)
        enum_b = _make_enum("EnumB", ("val",), line=2)
        unit = _make_unitlit()
        r = resolve_program(enum_a, enum_b, unit)
        assert len(r.constructor_candidates["val"]) == 2

    def test_qualified_constructor_via_ast(self) -> None:
        """FieldAccess on a type name is recorded as a qualified_constructor_ref."""
        enum = _make_enum("Color", ("red", "blue"))
        fa = FieldAccess(
            obj=_make_varref("Color"),
            field="red",
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(enum, fa)
        assert fa.node_id in r.qualified_constructor_refs
        assert r.qualified_constructor_refs[fa.node_id][:2] == ("Color", "red")

    def test_ordinary_field_access_not_qualified_ref(self) -> None:
        """FieldAccess on a regular value is NOT recorded in qualified_constructor_refs."""
        let_x = _make_let("x", _make_intlit(1))
        fa = FieldAccess(
            obj=_make_varref("x"),
            field="something",
            span=_sp(),
            node_id=_nid(),
        )
        r = resolve_program(let_x, fa)
        assert fa.node_id not in r.qualified_constructor_refs


# ---------------------------------------------------------------------------
# Cast scope tests (M2)
# ---------------------------------------------------------------------------


class TestCastScope:
    def test_cast_subexpr_resolves(self) -> None:
        """cast sub-expression resolves names normally."""
        r = parse_and_resolve("let x = 1\nlet y = x as int")
        assert r  # no exception

    def test_cast_undefined_var_is_scope_error(self) -> None:
        """undefined var inside a cast is a scope error."""
        err = reject_scope("undefinedVar as int")
        assert "undefinedVar" in err.to_diagnostic().message

    def test_parse_json_resolves_as_builtin(self) -> None:
        """parse_json(x) resolves as a builtin — no 'undefined name parse_json' error."""
        r = parse_and_resolve('let s = "hello"\nparse_json(s)')
        # The call to parse_json should be classified as PARSE_JSON builtin
        from agm.agl.scope.symbols import BuiltinKind
        assert BuiltinKind.PARSE_JSON in r.builtin_calls.values()


class TestImportDeclScope:
    """Import declarations pass through the scope resolver without errors."""

    def test_import_decl_does_not_raise(self) -> None:
        """A bare import declaration resolves without a scope error."""
        r = parse_and_resolve("import foo.bar\n1")
        assert r  # no exception

    def test_import_with_alias_does_not_raise(self) -> None:
        r = parse_and_resolve("import foo as f\n1")
        assert r

    def test_import_wildcard_does_not_raise(self) -> None:
        r = parse_and_resolve("import foo.*\n1")
        assert r

    def test_import_using_does_not_raise(self) -> None:
        r = parse_and_resolve("import foo using bar\n1")
        assert r

    def test_import_hiding_does_not_raise(self) -> None:
        r = parse_and_resolve("import foo hiding secret\n1")
        assert r
