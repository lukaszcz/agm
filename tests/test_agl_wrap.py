"""Tests for the pure inline-source ``program def main`` AST transform."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program_seeded, wrap_inline_program
from agm.agl.pipeline import PipelineDriver
from agm.agl.syntax import (
    AssignStmt,
    Block,
    BuiltinVarDecl,
    Call,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    FuncDef,
    ImportDecl,
    InfixDecl,
    LetDecl,
    RecordDef,
    ScopeRegion,
    TypeAlias,
    UnitT,
    UseDecl,
    VarDecl,
)
from tests._agl_helpers import run_inline_command


def test_wrap_inline_program_partitions_root_items_and_preserves_statement_order() -> None:
    program, next_node_id = parse_program_seeded(
        """\
import helpers
export helpers
use Shared::*
builtin var setting: int
def helper() -> unit = ()
record Item()
enum State = Ready
exception Problem()
type Count = int
infixl |> at 12

scope Shared
  def member() -> unit = ()
end Shared

let value = 1
var total = 0
total := value
print value
""",
        start_id=0,
    )
    original_items = program.body.items

    wrapped, transformed_next_node_id = wrap_inline_program(program, next_node_id=next_node_id)

    root_items = wrapped.body.items
    assert root_items[:-1] == original_items[:11]
    assert all(actual is expected for actual, expected in zip(root_items[:-1], original_items[:11]))
    assert all(
        isinstance(
            item,
            (
                ImportDecl,
                ExportDecl,
                FuncDef,
                RecordDef,
                EnumDef,
                ExceptionDef,
                TypeAlias,
                InfixDecl,
                ScopeRegion,
                BuiltinVarDecl,
                UseDecl,
            ),
        )
        for item in root_items[:-1]
    )

    main = root_items[-1]
    assert isinstance(main, FuncDef)
    assert main.name == "main"
    assert main.is_program is True
    assert main.is_synthetic is True
    assert main.params == ()
    assert main.type_param_slots == ()
    assert isinstance(main.return_type, UnitT)
    assert isinstance(main.body, Block)
    assert main.body.items == original_items[11:]
    assert all(actual is expected for actual, expected in zip(main.body.items, original_items[11:]))
    assert all(isinstance(item, (LetDecl, VarDecl, AssignStmt)) for item in main.body.items[:3])
    assert isinstance(main.body.items[3], Call)
    assert transformed_next_node_id == next_node_id + 3


def test_wrap_inline_program_keeps_scoped_bindings_at_root_and_wraps_local_bindings() -> None:
    program, next_node_id = parse_program_seeded(
        "let Config::answer = 42\nvar Config::count = 0\nlet local = 1\nvar scratch = 0\n",
        start_id=0,
    )
    scoped_let, scoped_var, local_let, local_var = program.body.items

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    wrapped_scoped_let, wrapped_scoped_var, main = wrapped.body.items
    assert wrapped_scoped_let is scoped_let
    assert wrapped_scoped_var is scoped_var
    assert isinstance(wrapped_scoped_let, LetDecl)
    assert isinstance(wrapped_scoped_var, VarDecl)
    assert wrapped_scoped_let.scope_path
    assert wrapped_scoped_var.scope_path
    assert isinstance(main, FuncDef)
    assert main.body.items == (local_let, local_var)
    assert all(
        isinstance(item, (LetDecl, VarDecl)) and not item.scope_path for item in main.body.items
    )


def test_wrap_inline_program_keeps_late_exports_with_the_executable_body() -> None:
    program, next_node_id = parse_program_seeded("let value = 1\nexport helpers\n", start_id=0)

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    assert len(wrapped.body.items) == 1
    main = wrapped.body.items[0]
    assert isinstance(main, FuncDef)
    assert main.body.items == program.body.items
    assert isinstance(main.body.items[1], ExportDecl)


def test_wrap_inline_program_wraps_an_empty_declaration_only_source() -> None:
    program, next_node_id = parse_program_seeded("def helper() -> unit = ()", start_id=0)

    wrapped, transformed_next_node_id = wrap_inline_program(program, next_node_id=next_node_id)

    (helper, main) = wrapped.body.items
    assert isinstance(helper, FuncDef)
    assert isinstance(main, FuncDef)
    assert main.is_synthetic
    assert isinstance(main.body, Block)
    assert main.body.items == ()
    assert transformed_next_node_id == next_node_id + 3


def test_wrap_inline_program_is_an_identity_when_a_program_def_is_present() -> None:
    program, next_node_id = parse_program_seeded(
        """\
scope Jobs

  program def run() -> unit = ()
end Jobs

let value = 1
""",
        start_id=0,
    )

    wrapped, transformed_next_node_id = wrap_inline_program(program, next_node_id=next_node_id)

    assert wrapped is program
    assert transformed_next_node_id == next_node_id


@pytest.mark.parametrize(
    "source",
    (
        "Config::answer\nlet Config::answer = 42",
        "Config::answer\n\nscope Config\n  let answer = 42\nend Config",
        "Config::answer\n\nscope Config\n  var answer: int = 42\nend Config",
        "answer\nvar answer: int = 42",
    ),
    ids=("shorthand-let", "region-let", "scoped-var", "root-var"),
)
def test_inline_expression_cannot_reference_a_later_textual_binding(source: str) -> None:
    parsed = PipelineDriver.parse_entry(source)
    assert parsed.program is not None
    wrapped, next_node_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)

    prepared = PipelineDriver.prepare_parsed_entry(
        replace(parsed, program=wrapped, next_id=next_node_id),
        roots=RootSet(roots=frozenset()),
        default_stdlib=False,
    )

    assert prepared.resolved is None
    assert len(prepared.diagnostics) == 1
    assert prepared.diagnostics[0].line == 1


def test_wrapped_root_declaration_can_reference_an_earlier_constant_binding() -> None:
    parsed = PipelineDriver.parse_entry("let value = 1\ndef read() -> int = value\nread()")
    assert parsed.program is not None
    wrapped, next_node_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)

    assert isinstance(wrapped.body.items[0], LetDecl)
    prepared = PipelineDriver.prepare_parsed_entry(
        replace(parsed, program=wrapped, next_id=next_node_id),
        roots=RootSet(roots=frozenset()),
        default_stdlib=False,
    )

    assert prepared.resolved is not None
    assert prepared.diagnostics == ()


def test_wrapped_root_declaration_reference_is_reported_by_the_scope_pipeline() -> None:
    parsed = PipelineDriver.parse_entry("def read() -> int = value\nlet value = 1")
    assert parsed.program is not None
    wrapped, next_node_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)

    prepared = PipelineDriver.prepare_parsed_entry(
        replace(parsed, program=wrapped, next_id=next_node_id),
        roots=RootSet(roots=frozenset()),
        default_stdlib=False,
    )

    assert prepared.resolved is None
    assert len(prepared.diagnostics) == 1
    assert "value" in prepared.diagnostics[0].message


def test_wrap_inline_program_keeps_a_binding_a_root_declaration_references() -> None:
    program, next_node_id = parse_program_seeded(
        """\
record Point(x: int, y: int)
let used = Point(1, 2)
let unused = Point(3, 4)
def read() -> int = used.x
print(read())
""",
        start_id=0,
    )
    point, used, unused, read, call = program.body.items

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    root_point, root_used, root_read, main = wrapped.body.items
    assert root_point is point
    assert root_used is used
    assert root_read is read
    assert isinstance(main, FuncDef)
    assert main.body.items == (unused, call)
    assert main.body.items[0] is unused


def test_wrap_inline_program_retains_bindings_reachable_through_other_retained_bindings() -> None:
    program, next_node_id = parse_program_seeded(
        """\
record Point(x: int, y: int)
record Boxed(inner: Point)
let corner = Point(1, 2)
let boxed = Boxed(corner)
def read() -> int = boxed.inner.y
print(read())
""",
        start_id=0,
    )
    point, boxed_def, corner, boxed, read, call = program.body.items

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    assert wrapped.body.items[:-1] == (point, boxed_def, corner, boxed, read)
    main = wrapped.body.items[-1]
    assert isinstance(main, FuncDef)
    assert main.body.items == (call,)


def test_wrap_inline_program_retains_a_binding_a_root_declaration_assigns() -> None:
    program, next_node_id = parse_program_seeded(
        """\
var count = 0
def reset() -> unit =
  count := 0
reset()
""",
        start_id=0,
    )
    count, reset, call = program.body.items

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    assert wrapped.body.items[:-1] == (count, reset)
    main = wrapped.body.items[-1]
    assert isinstance(main, FuncDef)
    assert main.body.items == (call,)


def test_wrap_inline_program_demotes_a_binding_shadowed_inside_the_declaration() -> None:
    program, next_node_id = parse_program_seeded(
        """\
let x = 1 + 1
def f() -> int =
  let x = 5
  x
print(f())
print(x)
""",
        start_id=0,
    )
    binding, func, print_call, print_x = program.body.items

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    assert wrapped.body.items[:-1] == (func,)
    main = wrapped.body.items[-1]
    assert isinstance(main, FuncDef)
    assert main.body.items == (binding, print_call, print_x)


@pytest.mark.parametrize(
    "declaration",
    (
        "def f() -> int =\n  fn(hidden: int) -> int => hidden\n  5",
        "def f(hidden: int) -> int = hidden",
        "def f() -> int =\n  for hidden in [1] do print hidden done\n  5",
        "def f() -> int =\n  try raise Boom() catch Boom as hidden => print hidden\n  5",
        "def f() -> int =\n  case Point(1, 2) of | Point(hidden, _) => hidden",
        "def f() -> int =\n  case Point(1, 2) of | Point(_, _) as hidden => hidden.x",
    ),
    ids=("lambda", "param", "loop", "catch", "pattern", "as-pattern"),
)
def test_wrap_inline_program_treats_any_binder_in_a_declaration_as_shadowing(
    declaration: str,
) -> None:
    program, next_node_id = parse_program_seeded(
        f"record Point(x: int, y: int)\nexception Boom()\nlet hidden = 1\n{declaration}\n",
        start_id=0,
    )
    binding = program.body.items[2]

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    main = wrapped.body.items[-1]
    assert isinstance(main, FuncDef)
    assert main.body.items == (binding,)


def test_wrap_inline_program_keeps_an_unreferenced_scoped_binding_at_the_root() -> None:
    program, next_node_id = parse_program_seeded(
        "let Config::answer = 42\nlet plain = 42\nprint 1\n",
        start_id=0,
    )
    scoped, plain, call = program.body.items

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    assert wrapped.body.items[:-1] == (scoped,)
    main = wrapped.body.items[-1]
    assert isinstance(main, FuncDef)
    assert main.body.items == (plain, call)


def test_wrap_inline_program_retains_a_binding_a_scoped_binding_references() -> None:
    program, next_node_id = parse_program_seeded(
        """\
record Point(x: int, y: int)
let corner = Point(1, 2)
let Config::origin = corner
print 1
""",
        start_id=0,
    )
    point, corner, scoped, call = program.body.items

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    assert wrapped.body.items[:-1] == (point, corner, scoped)
    main = wrapped.body.items[-1]
    assert isinstance(main, FuncDef)
    assert main.body.items == (call,)


def test_wrapped_computed_binding_is_readable_from_a_root_declaration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A retained binding an inline declaration reads may be computed."""
    result = run_inline_command(
        PipelineDriver(),
        'let a = 2\nlet b = a + 1\ndef show() -> unit = print("%{b}")\nshow()',
    )

    assert result.ok, f"expected success but got: {result.diagnostics!r}"
    assert capsys.readouterr().out == "3\n"


def test_wrapped_computed_binding_chain_initializes_in_source_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retained computed bindings run in source order before the entry body."""
    result = run_inline_command(
        PipelineDriver(),
        'let base = 2\nlet doubled = base * 2\nlet label = "v%{doubled}"\n'
        "var total = doubled\n"
        'def report() -> text = "%{label}:%{total}"\n'
        "total := total + 1\n"
        "print(report())",
    )

    assert result.ok, f"expected success but got: {result.diagnostics!r}"
    assert capsys.readouterr().out == "v4:5\n"


def test_wrapped_entry_var_is_assignable_from_a_closure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A lambda in the synthetic entry may assign the entry's own ``var``."""
    result = run_inline_command(
        PipelineDriver(),
        "var count = 0\nlet bump = fn() -> unit => do count := count + 1 until true\n"
        'bump()\nbump()\nprint("%{count}")',
    )

    assert result.ok, f"expected success but got: {result.diagnostics!r}"
    assert capsys.readouterr().out == "2\n"


def test_wrapped_entry_closure_sees_later_writes_to_a_captured_var(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The capture shares the entry's cell rather than snapshotting its value."""
    result = run_inline_command(
        PipelineDriver(),
        "var factor = 2\nlet scale = fn(n: int) -> int => n * factor\n"
        "print(scale(3))\nfactor := 10\nprint(scale(3))",
    )

    assert result.ok, f"expected success but got: {result.diagnostics!r}"
    assert capsys.readouterr().out == "6\n30\n"


def test_late_import_in_inline_source_reports_the_header_rule() -> None:
    """An import after an executable item names the ordering rule it broke."""
    result = run_inline_command(PipelineDriver(), 'print("a")\nimport std/text\nprint("b")')

    assert not result.ok
    assert [diagnostic.line for diagnostic in result.diagnostics] == [2]
    # The ordering rule, not the nesting rule the generated wrapper would suggest.
    assert "nested block" not in result.diagnostics[0].message


def test_import_inside_an_inline_nested_block_still_names_the_block() -> None:
    """A block the source itself wrote keeps the placement diagnostic."""
    result = run_inline_command(
        PipelineDriver(), 'print("a")\nfor i in 1 to 2 do\n  import std/text\n  print(i)\ndone'
    )

    assert not result.ok
    assert [diagnostic.line for diagnostic in result.diagnostics] == [3]
    assert "nested block" in result.diagnostics[0].message


def test_static_root_rejection_without_a_program_def_omits_the_inline_explanation() -> None:
    """Source declaring no entry is not told to move items into a program body."""
    statement = PipelineDriver().run("print 1")
    binding = PipelineDriver().run("let x = 1 + 1")

    for result in (statement, binding):
        assert not result.ok
        assert "program def" not in result.diagnostics[0].message


def test_static_root_rejection_explains_a_declared_inline_program_def() -> None:
    """Inline source with its own entry is told where its items belong."""
    statement = PipelineDriver().run("program def main() -> unit = ()\nprint 1")
    binding = PipelineDriver().run("program def main() -> unit = ()\nlet x = 1 + 1")

    for result in (statement, binding):
        assert not result.ok
        assert "program def" in result.diagnostics[0].message
