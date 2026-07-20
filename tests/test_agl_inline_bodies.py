"""Tests for inline branch, catch, loop, try, and parenthesized bodies.

An inline body is the form written on the same line after ``=>`` (or after
``do``), as opposed to the suite form written as an indented block.

A ``;`` sequence is admissible exactly where a token marks the body's end:
``)`` for a parenthesized block, ``until``/``done`` for a loop body, ``catch``
for a try body, a newline for a ``def`` body.  A body after ``=>`` has no such
marker — a following ``|``, ``else``, or ``catch`` could belong to either the
body or the enclosing form — so it holds exactly one item.

Covers:
- ``:=``, ``raise``, and ``return`` as whole inline ``=>`` bodies.
- Binders and ``;`` sequences rejected after ``=>``, with parentheses and the
  suite form as the two escape hatches.
- Open forms (``case``/``if``/``try``/``do``) admitted inline in loop bodies,
  where the terminator resolves the ambiguity.
- Parenthesized blocks, and try bodies taking the same sequence.

NOTE: the grammar's zero-conflict invariant is guarded by
      ``tests/test_agl_parser.py::test_zero_conflicts``.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
from lark.exceptions import UnexpectedToken
from lark.lexer import Token

from agm.agl import PipelineDriver
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.parser.errors import syntax_error_from_lark


def _run(source: str) -> tuple[bool, str, list[str]]:
    """Run *source*, returning its success flag, stdout, and diagnostics."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = PipelineDriver().run(source, param_values={})
    return result.ok, buffer.getvalue(), [d.message for d in result.diagnostics]


class TestAssignmentInlineInBranchBody:
    """``:=`` is admitted as a whole inline branch or catch body."""

    @pytest.mark.parametrize(
        ("scrutinee", "expected"),
        [("1", "one"), ("2", "other")],
    )
    def test_assignment_inline_in_case_arm(self, scrutinee: str, expected: str) -> None:
        ok, out, diags = _run(
            f"""\
var label: text = "unset"
case {scrutinee} of
  | 1 => label := "one"
  | _ => label := "other"
print label
"""
        )
        assert ok, diags
        assert out.strip() == expected

    @pytest.mark.parametrize(
        ("cond", "expected"),
        [("true", "yes"), ("false", "no")],
    )
    def test_assignment_inline_in_if_branch(self, cond: str, expected: str) -> None:
        ok, out, diags = _run(
            f"""\
var label: text = "unset"
if {cond} => label := "yes" else => label := "no"
print label
"""
        )
        assert ok, diags
        assert out.strip() == expected

    def test_assignment_inline_in_catch_body(self) -> None:
        ok, out, diags = _run(
            """\
var label: text = "unset"
try
  print (1 / 0)
catch ArithmeticError => label := "caught"
print label
"""
        )
        assert ok, diags
        assert out.strip() == "caught"

    def test_indexed_assignment_inline_in_case_arm(self) -> None:
        ok, out, diags = _run(
            """\
var xs: list[int] = [1, 2]
case 0 of
  | 0 => xs[0] := 99
  | _ => ()
print xs[0]
"""
        )
        assert ok, diags
        assert out.strip() == "99"


class TestRaiseAndReturnInlineInBranchBody:
    """``raise`` and ``return`` are admitted inline in branch bodies."""

    def test_raise_inline_in_case_arm(self) -> None:
        """The program compiles and fails at runtime, not statically."""
        ok, _out, diags = _run(
            """\
exception Halt extends Exception
  code: int

case 1 of
  | 1 => raise Halt(code = 1, message = "stop")
  | _ => ()
"""
        )
        assert diags == []
        assert not ok

    def test_raise_inline_in_if_branch_is_bottom_typed(self) -> None:
        """A diverging inline branch unifies with the other branch's type."""
        ok, out, diags = _run(
            """\
exception Halt extends Exception
  code: int

let x: int = if false => raise Halt(code = 1, message = "!") else => 7
print x
"""
        )
        assert ok, diags
        assert out.strip() == "7"

    @pytest.mark.parametrize(("n", "expected"), [("5", "big"), ("0", "small")])
    def test_return_inline_in_branch_body(self, n: str, expected: str) -> None:
        ok, out, diags = _run(
            f"""\
def classify(n: int) -> text =
  if n > 1 => return "big" else => return "small"

print classify({n})
"""
        )
        assert ok, diags
        assert out.strip() == expected


class TestBinderRejectedInlineInBranchBody:
    """An inline ``=>`` body is a single item: no binders, no ``;`` sequence."""

    def test_binder_inline_in_case_arm_is_rejected(self) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(
                """\
case 1 of
  | 1 => let orphan = 2
  | _ => ()
"""
            )
        error = exc_info.value
        assert "cannot be an inline `=>` body" in str(error)
        assert error.source_span.start_line == 2, "diagnostic anchors on the binder"

    def test_binder_with_continuation_inline_is_still_rejected(self) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(
                """\
case 1 of
  | 1 => let a = 1; a
  | _ => 0
"""
            )
        assert "cannot be an inline `=>` body" in str(exc_info.value)

    def test_binder_inline_in_catch_body_is_rejected(self) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(
                """\
try
  print (1 / 0)
catch ArithmeticError as e => var m = e.message; print m
"""
            )
        assert "cannot be an inline `=>` body" in str(exc_info.value)

    def test_parenthesized_block_is_the_inline_escape_hatch(self) -> None:
        ok, out, diags = _run(
            """\
case 1 of
  | 1 => (let a = 2; let b = 3; print(a * b))
  | _ => ()
"""
        )
        assert ok, diags
        assert out.strip() == "6"

    def test_suite_body_is_the_other_escape_hatch(self) -> None:
        ok, out, diags = _run(
            """\
case 1 of
  | 1 =>
    let doubled = 21 * 2
    print doubled
  | _ => ()
"""
        )
        assert ok, diags
        assert out.strip() == "42"

    def test_unrelated_error_keeps_its_own_diagnostic(self) -> None:
        """A branch after `else` is not reported as a binder problem."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let k = 1\nif k == 1 => 1 else => 2 | k == 2 => 3")
        assert "cannot be an inline `=>` body" not in str(exc_info.value)

    def test_binder_outside_an_inline_body_keeps_its_own_diagnostic(self) -> None:
        """A `let` not opening an inline `=>` body takes the generic path."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let a = let b = 1\n")
        assert "cannot be an inline `=>` body" not in str(exc_info.value)

    def test_binder_partway_through_an_inline_body_keeps_its_own_diagnostic(self) -> None:
        """A `let` after an `=>` but not opening the body is a different error."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let k = if true => 1 + let a = 2 else => 3\n")
        assert "cannot be an inline `=>` body" not in str(exc_info.value)

    def test_binder_diagnosis_needs_source_text(self) -> None:
        """Without source text the binder shape cannot be confirmed.

        The custom AglLexer means this state is not reachable through
        ``parse_program``; call the mapping helper directly.
        """
        token = Token("LET", "let", start_pos=0, line=1, column=1)
        err = syntax_error_from_lark(UnexpectedToken(token, expected={"NAME"}))
        assert "cannot be an inline `=>` body" not in str(err)


class TestParenthesizedBlock:
    """``)`` marks the end of a body, so parens take a full block."""

    def test_binder_sequence_in_parens(self) -> None:
        ok, out, diags = _run("let v = (let x = 4; x * 2)\nprint v\n")
        assert ok, diags
        assert out.strip() == "8"

    def test_assignment_in_parens(self) -> None:
        ok, out, diags = _run("var x: int = 0\nlet a = (x := 1)\nprint x\n")
        assert ok, diags
        assert out.strip() == "1"

    def test_expression_sequence_in_parens(self) -> None:
        ok, out, diags = _run("var x: int = 0\nlet v = (x := 5; x + 1)\nprint v\n")
        assert ok, diags
        assert out.strip() == "6"

    def test_open_form_in_parenthesized_sequence(self) -> None:
        ok, out, diags = _run(
            "var x: int = 0\nlet v = (if true => x := 2 else => x := 3; x + 1)\nprint v\n"
        )
        assert ok, diags
        assert out.strip() == "3"

    def test_parenthesized_block_inside_a_call_needs_its_own_parens(self) -> None:
        """Parens right after a callee are the argument list, as in any call."""
        ok, out, diags = _run("print((let x = 4; x * 2))\n")
        assert ok, diags
        assert out.strip() == "8"

    def test_single_expression_in_parens_is_unchanged(self) -> None:
        ok, out, diags = _run("let v = (1 + 2) * 3\nprint v\n")
        assert ok, diags
        assert out.strip() == "9"

    def test_block_in_parens_must_end_in_a_value(self) -> None:
        ok, _out, diags = _run("let v = (let x = 4; let y = 5)\nprint v\n")
        assert not ok
        assert diags


class TestTryBodyMatchesTheOtherMarkedBodies:
    """`catch` marks the try body's end, so it takes the same sequence."""

    def test_assignment_and_binder_in_inline_try_body(self) -> None:
        ok, out, diags = _run(
            """\
var i: int = 0
let v = try i := 1; let z = i + 1; z catch _ => 0
print v
"""
        )
        assert ok, diags
        assert out.strip() == "2"

    @pytest.mark.parametrize(
        "tail",
        ["if true => 1 else => 2", "case 1 of | _ => 5", "7"],
    )
    def test_open_forms_allowed_as_the_final_item(self, tail: str) -> None:
        """Only a nested `try` is barred from the final position."""
        ok, out, diags = _run(f"let v = try {tail} catch _ => 0\nprint v\n")
        assert ok, diags
        assert out.strip() != ""

    def test_nested_try_as_the_final_item_is_rejected(self) -> None:
        """`catch` is repeatable, so an inner `try` there would take them all."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let v = try try 1 catch _ => 2 catch _ => 3\nprint v\n")
        assert "nested `try`" in str(exc_info.value)

    def test_nested_try_diagnosis_needs_the_parser_state(self) -> None:
        """Without a parse stack the completed rule is unknown; fall back.

        Reachable only by synthesising the exception — a real parse always
        carries the state.
        """
        token = Token("_NEWLINE", "0", start_pos=0, line=1, column=1)
        err = syntax_error_from_lark(UnexpectedToken(token, expected={"SEMICOLON"}))
        assert "nested `try`" not in str(err)

    def test_nested_try_as_the_final_item_works_parenthesized(self) -> None:
        ok, out, diags = _run("let v = try (try 1 catch _ => 2) catch _ => 3\nprint v\n")
        assert ok, diags
        assert out.strip() == "1"

    def test_try_without_a_catch_clause_is_diagnosed(self) -> None:
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program("let v = try 1\nprint v\n")
        assert "at least one `catch`" in str(exc_info.value)

    def test_raise_in_inline_try_body_is_caught(self) -> None:
        ok, out, diags = _run(
            """\
let v = try raise Abort(message = "boom") catch Abort => 7
print v
"""
        )
        assert ok, diags
        assert out.strip() == "7"


class TestOpenFormsInlineInLoopBody:
    """``case``/``if``/``try``/``do`` are admitted inline in loop bodies."""

    def test_case_inline_in_loop_body(self) -> None:
        ok, out, diags = _run(
            """\
var total: int = 0
for i in 1 to 3 do case i of | 2 => total := total + 10 | _ => total := total + 1 done
print total
"""
        )
        assert ok, diags
        assert out.strip() == "12"

    def test_if_inline_in_loop_body(self) -> None:
        ok, out, diags = _run(
            """\
var total: int = 0
for i in 1 to 4 do if i > 2 => total := total + i else => () done
print total
"""
        )
        assert ok, diags
        assert out.strip() == "7"

    def test_try_inline_in_loop_body(self) -> None:
        ok, out, diags = _run(
            """\
var caught: int = 0
for i in 1 to 2 do try print (1 / 0) catch ArithmeticError => caught := caught + 1 done
print caught
"""
        )
        assert ok, diags
        assert out.strip() == "2"

    def test_case_inline_in_while_loop_body(self) -> None:
        """`for`/`while` are prefix clauses on the same loop rule as bare `do`."""
        ok, out, diags = _run(
            """\
var i: int = 0
var total: int = 0
while i < 3 do i := i + 1; case i of | 2 => total := total + 10 | _ => total := total + 1 done
print total
"""
        )
        assert ok, diags
        assert out.strip() == "12"

    def test_open_form_inline_in_for_while_loop_body(self) -> None:
        ok, out, diags = _run(
            """\
var total: int = 0
for i in 1 to 5 while total < 6 do if i > 1 => total := total + i else => () done
print total
"""
        )
        assert ok, diags
        assert out.strip() == "9"

    def test_nested_loop_inline_binds_innermost_terminator(self) -> None:
        """In ``do do ... until p until q`` the inner ``until`` binds inward."""
        ok, out, diags = _run(
            """\
var outer: int = 0
var inner: int = 0
do
  inner := 0
  do inner := inner + 1 until inner >= 2
  outer := outer + 1
until outer >= 3
print "${outer}/${inner}"
"""
        )
        assert ok, diags
        assert out.strip() == "3/2"

    def test_case_inline_in_loop_body_with_semicolon_sequence(self) -> None:
        ok, out, diags = _run(
            """\
var total: int = 0
for i in 1 to 2 do case i of | _ => total := total + i; print total done
"""
        )
        assert ok, diags
        assert out.split() == ["1", "3"]


class TestInlineRestrictionsRetained:
    """Forms that remain blocked inline in branch bodies."""

    @pytest.mark.parametrize("keyword", ["case", "if", "try"])
    def test_open_form_inline_in_case_arm_is_rejected(self, keyword: str) -> None:
        bodies = {
            "case": "case 1 of | _ => 2",
            "if": "if true => 1 else => 2",
            "try": "try 1 catch _ => 2",
        }
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program(
                f"""\
case 1 of
  | 1 => {bodies[keyword]}
  | _ => ()
"""
            )
        assert "not allowed inline here" in str(exc_info.value)

    def test_lambda_inline_in_case_arm_is_rejected(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program(
                """\
case 1 of
  | 1 => fn(x: int) => x
  | _ => ()
"""
            )

    def test_open_form_as_inline_assignment_rhs_is_rejected(self) -> None:
        """An inline assignment's RHS is restricted to a closed expression."""
        with pytest.raises(AglSyntaxError):
            parse_program(
                """\
var x: int = 0
case 1 of
  | 1 => x := if true => 1 else => 2
  | _ => ()
"""
            )

    def test_open_form_as_assignment_rhs_still_allowed_in_suite(self) -> None:
        """The block form keeps the unrestricted right-hand side."""
        ok, out, diags = _run(
            """\
var x: int = 0
case 1 of
  | 1 =>
    x := if true => 11 else => 22
  | _ => ()
print x
"""
        )
        assert ok, diags
        assert out.strip() == "11"
