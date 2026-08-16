"""IR evaluation tests for multi-module linking."""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.ir.ids import FunctionId
from agm.agl.ir.nodes import IrBlock, IrConstUnit, IrDirectCall, IrPrint
from agm.agl.ir.program import IrFunctionBody
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.lower.program import lower_program
from agm.agl.modules.ids import STD_CONFIG_ID, ModuleId
from agm.agl.semantics.values import BoolValue, EnumValue, IntValue, RecordValue, TextValue
from agm.agl.typecheck import AglTypeError
from tests.agl.ir_harness import (
    _checked,
    _compiled_checked,
    evaluate_ir,
    evaluate_ir_graph,
    evaluate_ir_graph_raises,
    lower_inline_ir,
    lower_ir,
    nominal_id_for,
)


@pytest.mark.parametrize(
    "scoped_bindings",
    (
        "scope Static\nlet constant = 1\nvar offset = 1\nend Static\n",
        "let Static::constant = 1\nvar Static::offset = 1\n",
    ),
    ids=("region", "shorthand"),
)
def test_wrap_mode_captures_static_bindings_and_synthetic_main_locals(
    scoped_bindings: str,
) -> None:
    """Statement-style harness sources execute through the synthetic main."""
    result = evaluate_ir(
        scoped_bindings + "let first = Static::constant + Static::offset\n"
        "var second = first + 1\n"
        "second := second + 1\n"
    )

    assert result == {
        "Static::constant": IntValue(1),
        "Static::offset": IntValue(1),
        "first": IntValue(2),
        "second": IntValue(4),
    }


_COUNTING_INITIALIZER = (
    "def bump() -> int =\n  Static::calls := Static::calls + 1\n  41 + Static::calls\n"
)


@pytest.mark.parametrize(
    "scoped_bindings",
    (
        "scope Static\nvar calls = 0\nlet value = bump()\nend Static\n",
        "var Static::calls = 0\nlet Static::value = bump()\n",
    ),
    ids=("region", "shorthand"),
)
def test_inline_scoped_bindings_may_compute_their_initializers(scoped_bindings: str) -> None:
    """Inline source is a command, never an imported module.

    Its root bindings are the ones the entry transform kept there, so they
    compute like any other statement: the initializer runs exactly once, and
    the value it produced is readable through the binding's scoped path.
    """
    result = evaluate_ir(scoped_bindings + _COUNTING_INITIALIZER + "let seen = Static::value\n")

    assert result == {
        "Static::calls": IntValue(1),
        "Static::value": IntValue(42),
        "seen": IntValue(42),
    }


@pytest.mark.parametrize(
    "root_bindings",
    (
        "scope Static\nlet value = 1 + 1\nend Static\n",
        "let Static::value = 1 + 1\n",
        "let value = 1 + 1\n",
    ),
    ids=("region", "shorthand", "unscoped"),
)
def test_file_module_rejects_a_non_constant_root_initializer(
    root_bindings: str, tmp_path: Path
) -> None:
    """A module another program can import still executes nothing at its root."""
    with pytest.raises(AglTypeError):
        lower_ir(
            root_bindings + "program def main() -> unit = ()\n",
            origin_path=tmp_path / "library.agl",
        )


def test_synthetic_main_runs_only_when_explicitly_selected() -> None:
    executable = lower_inline_ir("print 1")
    synthetic_main = executable.synthetic_main_symbol
    assert synthetic_main is not None

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        IrInterpreter(executable).run()
    assert output.getvalue() == ""

    with contextlib.redirect_stdout(output):
        IrInterpreter(executable).run(program_symbol=synthetic_main)
    assert output.getvalue() == "1\n"


def test_empty_synthetic_main_lowers_and_runs_as_unit() -> None:
    executable = lower_inline_ir("def helper() -> unit = ()")
    synthetic_main = executable.synthetic_main_symbol
    assert synthetic_main is not None

    assert IrInterpreter(executable).run(program_symbol=synthetic_main) == {}


def test_inline_user_main_coexists_with_host_synthetic_entry() -> None:
    executable = lower_inline_ir("def main() -> int = 41\nlet answer = main() + 1")
    synthetic_main = executable.synthetic_main_symbol
    assert synthetic_main is not None

    result = IrInterpreter(executable).run(program_symbol=synthetic_main)

    assert result["answer"] == IntValue(42)
    synthetic_symbol = executable.symbols[synthetic_main]
    assert synthetic_symbol.public_name is None
    assert synthetic_symbol.synthetic


def test_inline_open_imported_main_resolves_to_helper_without_recursion(tmp_path: Path) -> None:
    result = evaluate_ir_graph(
        "import helper::*\nlet answer = main()",
        {"helper": "def main() -> int = 42"},
        tmp_path,
    )

    assert result["answer"] == IntValue(42)


def test_synthetic_main_discards_a_non_unit_body_result() -> None:
    executable = lower_inline_ir("1")
    synthetic_main = executable.synthetic_main_symbol
    assert synthetic_main is not None
    main = executable.functions[executable.program_functions[synthetic_main]]
    assert isinstance(main.impl, IrFunctionBody)
    assert isinstance(main.impl.body, IrBlock)
    assert isinstance(main.impl.body.items[-1], IrConstUnit)


def test_validation_rejects_an_invalid_synthetic_main_symbol() -> None:
    wrapped = lower_inline_ir("let value = 1")
    missing_symbol = replace(wrapped, synthetic_main_symbol=None)
    file_program = lower_inline_ir("program def main() -> unit = ()")
    (main_symbol,) = file_program.program_functions
    unmarked_symbol = replace(file_program, synthetic_main_symbol=main_symbol)

    for invalid in (missing_symbol, unmarked_symbol):
        with pytest.raises(InvalidIrError, match="synthetic main"):
            validate_ir(invalid, deep=True)


def test_file_mode_excludes_function_bindings() -> None:
    source = """\
let constant = 1
def helper() -> int = constant + 1
program def main() -> unit =
  let local = helper()
  print local
"""
    executable = lower_inline_ir(source)
    (main_symbol,) = executable.program_functions

    result = IrInterpreter(executable).run(program_symbol=main_symbol)

    assert result == {"constant": IntValue(1)}


def test_library_const_captured_by_library_function(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A library's static binding is initialized before its entry function is invoked."""
    checked = _checked(
        "import lib\nprogram def main() -> unit = print lib::read()\n",
        {
            "lib": (
                "let constant = 4\nvar adjustment = 1\ndef read() -> int = constant + adjustment\n"
            )
        },
        tmp_path,
    )
    executable = lower_program(_compiled_checked(checked))
    (main_symbol,) = executable.program_functions

    IrInterpreter(executable).run(program_symbol=main_symbol)

    assert capsys.readouterr().out == "5\n"


def test_library_binding_initializers_follow_import_dependency_order(tmp_path: Path) -> None:
    """Library binding modules link after their dependencies and before entry."""
    checked = _checked(
        "import library\nprogram def main() -> unit = print library::read()\n",
        {
            "dependency": "let seed = 1\n",
            "library": "import dependency\nlet value = 2\ndef read() -> int = value\n",
        },
        tmp_path,
    )

    executable = lower_program(_compiled_checked(checked))

    assert list(executable.modules) == [
        STD_CONFIG_ID,
        ModuleId.from_path("dependency"),
        ModuleId.from_path("library"),
        executable.entry_module,
    ]


def test_imported_function_and_local_let(tmp_path: Path) -> None:
    """Entry imports a function from a library module and uses it in a let binding."""
    lib_source = """
def add(a: int, b: int) -> int =
    a + b
"""
    entry_source = """
import lib
let result = lib::add(3, 4)
let x = 10
()
"""
    r = evaluate_ir_graph(entry_source, {"lib": lib_source}, tmp_path)
    assert r["result"] == IntValue(7)
    assert r["x"] == IntValue(10)


def test_local_alias_constructor_lowers() -> None:
    source = """
record Box[T]
  value: T
type Alias[T] = Box[T]
let box = Alias(value = 1)
box
"""
    result = evaluate_ir(source)
    program = lower_inline_ir(source)

    assert result["box"] == RecordValue(
        nominal_id_for(program, "Box"), "Box", {"value": IntValue(1)}
    )


def test_imported_alias_constructor_value_lowers(tmp_path: Path) -> None:
    entry_source = """
import lib
let factory: (int) -> lib::Box[int] = lib::Alias
let box = factory(1)
box
"""
    modules = {"lib": "record Box[T]\n  value: T\ntype Alias[T] = Box[T]"}
    result = evaluate_ir_graph(entry_source, modules, tmp_path)
    checked = _checked(entry_source, modules, tmp_path)
    program = lower_program(_compiled_checked(checked))

    assert result["box"] == RecordValue(
        nominal_id_for(program, "Box"), "Box", {"value": IntValue(1)}
    )


def test_scoped_linked_calls_use_function_handles_and_validate(tmp_path: Path) -> None:
    """Cross-module scoped calls lower to linked handles accepted by IR validation."""
    workflow = (
        Path(__file__).parent / "agl" / "multi_file" / "scoped_execution" / "workflow.agl"
    ).read_text()
    entry = (
        "import scoped_execution/workflow\n"
        "import scoped_execution/workflow::{Workflow::Task}\n"
        "import scoped_execution/workflow::*\n"
        "use scoped_execution/workflow::Workflow::*\n"
        "use scoped_execution/workflow::Workflow::Status::*\n"
        'let task = Task(name = "root", children = [Task(name = "ready", children = [])])\n'
        "let status: Status = completed\n"
        "print(score(task))\n"
        "print(label(status))\n"
    )
    checked = _checked(entry, {"scoped_execution/workflow": workflow}, tmp_path)
    executable = lower_program(_compiled_checked(checked))

    validate_ir(executable, deep=True)
    assert executable.synthetic_main_symbol is not None
    main = executable.functions[executable.program_functions[executable.synthetic_main_symbol]]
    assert isinstance(main.impl, IrFunctionBody)
    assert isinstance(main.impl.body, IrBlock)
    calls = [
        initializer.value
        for initializer in main.impl.body.items
        if isinstance(initializer, IrPrint) and isinstance(initializer.value, IrDirectCall)
    ]
    assert len(calls) == 2
    assert all(isinstance(call.function_id, FunctionId) for call in calls)
    assert all(call.function_id in executable.functions for call in calls)
    assert all(not hasattr(call, "function_name") for call in calls)


def test_cross_module_mutual_recursion(tmp_path: Path) -> None:
    """Even/odd mutual recursion across two cyclic-import modules."""
    even_source = """
import odd
def is_even(n: int) -> bool =
    if n == 0 => true
    | else => odd::is_odd(n - 1)
"""
    odd_source = """
import even
def is_odd(n: int) -> bool =
    if n == 0 => false
    | else => even::is_even(n - 1)
"""
    entry_source = """
import even
let r1 = even::is_even(4)
let r2 = even::is_even(3)
()
"""
    r = evaluate_ir_graph(entry_source, {"even": even_source, "odd": odd_source}, tmp_path)
    assert r["r1"] == BoolValue(True)
    assert r["r2"] == BoolValue(False)


def test_cross_module_abstract_exception_constructor_rejected(tmp_path: Path) -> None:
    lib_source = """
exception Root
  code: int
"""
    entry_source = """
import lib
let boom = lib::Root(code = 1)
()
"""

    with pytest.raises(AglTypeError):
        evaluate_ir_graph(entry_source, {"lib": lib_source}, tmp_path)


def test_imported_record_and_enum(tmp_path: Path) -> None:
    """Entry uses records and enums from a library module."""
    shapes_source = """
record Point
  x: int
  y: int
enum Color
  | Red
  | Blue
  | Green
"""
    entry_source = """
import shapes::*
let p = shapes::Point(x = 1, y = 2)
let c = shapes::Color::Red
let px = p.x
let is_red = case c of
    | Red => true
    | _ => false
()
"""
    r = evaluate_ir_graph(entry_source, {"shapes": shapes_source}, tmp_path)
    assert r["px"] == IntValue(1)
    assert r["is_red"] == BoolValue(True)
    p = r["p"]
    assert isinstance(p, RecordValue)
    assert p.fields["x"] == IntValue(1)
    assert p.fields["y"] == IntValue(2)
    c = r["c"]
    assert isinstance(c, EnumValue)
    assert c.variant == "Red"


def test_same_named_types_in_two_modules(tmp_path: Path) -> None:
    """Two modules with same-named types/functions don't shadow each other."""
    mod_a_source = """
record Pair
  a: int
  b: int
def get_first(p: Pair) -> int =
    p.a
"""
    mod_b_source = """
record Pair
  x: text
  y: text
def get_first(p: Pair) -> text =
    p.x
"""
    entry_source = """
import mod_a
import mod_b
let p1 = mod_a::Pair(a = 1, b = 2)
let p2 = mod_b::Pair(x = "hello", y = "world")
let first_value = mod_a::get_first(p1)
let second_value = mod_b::get_first(p2)
()
"""
    r = evaluate_ir_graph(entry_source, {"mod_a": mod_a_source, "mod_b": mod_b_source}, tmp_path)
    assert r["first_value"] == IntValue(1)
    assert r["second_value"] == TextValue("hello")


def test_runtime_failure_inside_library_function(tmp_path: Path) -> None:
    """ArithmeticError raised inside a library function propagates to entry."""
    mathlib_source = """
def safe_div(a: int, b: int) -> decimal =
    a / b
"""
    entry_source = """
import mathlib
let result = mathlib::safe_div(10, 0)
()
"""
    exc = evaluate_ir_graph_raises(entry_source, {"mathlib": mathlib_source}, tmp_path)
    assert exc.display_name == "ArithmeticError"


def test_open_imported_nullary_enum_as_value(tmp_path: Path) -> None:
    """Import-tail-exposed nullary enum variant used as a value.

    Covers the cref non-FunctionType path.
    """
    status_source = """
enum Status
  | Running
  | Done
"""
    entry_source = """
import status::*
let s: Status = Running
let is_running = case s of
    | Running => true
    | Done => false
()
"""
    r = evaluate_ir_graph(entry_source, {"status": status_source}, tmp_path)
    assert r["is_running"] == BoolValue(True)
    s = r["s"]
    assert isinstance(s, EnumValue)
    assert s.variant == "Running"
