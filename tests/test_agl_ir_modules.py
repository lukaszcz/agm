"""IR evaluation tests for multi-module linking.
"""
from __future__ import annotations

from pathlib import Path

from agm.agl.semantics.values import BoolValue, EnumValue, IntValue, RecordValue, TextValue
from tests.agl.ir_harness import evaluate_ir_graph, evaluate_ir_graph_raises


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
    r = evaluate_ir_graph(
        entry_source, {"even": even_source, "odd": odd_source}, tmp_path
    )
    assert r["r1"] == BoolValue(True)
    assert r["r2"] == BoolValue(False)


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
import shapes
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
    r = evaluate_ir_graph(
        entry_source, {"mod_a": mod_a_source, "mod_b": mod_b_source}, tmp_path
    )
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
    """Open-imported nullary enum variant used as a value (covers cref non-FunctionType path)."""
    status_source = """
enum Status
  | Running
  | Done
"""
    entry_source = """
import status.*
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
