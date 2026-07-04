"""IR execution tests for partial application closures."""

from __future__ import annotations

import decimal
from pathlib import Path

from agm.agl.semantics.values import DecimalValue, IntValue, RecordValue, TextValue
from tests.agl.ir_harness import evaluate_ir, evaluate_ir_graph


def test_partial_call_captures_non_hole_argument_by_value() -> None:
    source = """
var current = 1

def add(a: int, b: int) -> int = a + b

let add_current = add(?, current)
current := 100
let result = add_current(2)
()
"""
    result = evaluate_ir(source)
    assert result["result"] == IntValue(3)
    assert result["current"] == IntValue(100)


def test_partial_call_evaluates_callee_then_non_holes_in_written_order_at_creation() -> None:
    source = """
var log = ""

def mark(label: text, value: int) -> int =
  log := log + label
  value

def digits(a: int, b: int, c: int) -> int = a * 100 + b * 10 + c

let digits_value = fn(a: int, b: int, c: int) -> int => digits(a, b, c)
def make_callee() -> (int, int, int) -> int =
  log := log + "callee"
  digits_value

let h = make_callee()(mark("a", 1), ?, mark("c", 3))
let after_create = log
let first = h(2)
let second = h(4)
()
"""
    result = evaluate_ir(source)
    assert result["after_create"] == TextValue("calleeac")
    assert result["log"] == TextValue("calleeac")
    assert result["first"] == IntValue(123)
    assert result["second"] == IntValue(143)


def test_partial_creation_time_exception_propagates_before_closure_invocation() -> None:
    source = """
def fail_arg() -> int = raise Abort(message = "create")
def add(a: int, b: int) -> int = a + b

let message = try
  let h = add(?, fail_arg())
  let value = h(1)
  "no error"
catch Abort as e =>
  e.message
message
"""
    result = evaluate_ir(source)
    assert result["message"] == TextValue("create")


def test_partial_invocation_time_exception_propagates_from_underlying_call() -> None:
    source = """
def fail_call(x: int) -> int = raise Abort(message = "invoke")
let h = fail_call(?)
let message = try
  let value = h(1)
  "no error"
catch Abort as e =>
  e.message
message
"""
    result = evaluate_ir(source)
    assert result["message"] == TextValue("invoke")


def test_partial_call_coerces_captured_arguments_when_invoked() -> None:
    source = """
def add_dec(a: decimal, b: decimal) -> decimal = a + b
let h = add_dec(?, 1)
let result = h(2.5)
()
"""
    result = evaluate_ir(source)
    assert result["result"] == DecimalValue(decimal.Decimal("3.5"))


def test_cross_module_constructor_partial_uses_shared_lowering(tmp_path: Path) -> None:
    result = evaluate_ir_graph(
        "import mylib\n"
        "let make: (int) -> mylib::Point = mylib::Point(x = ?)\n"
        "let point = make(7)\n"
        "()",
        {"mylib": "record Point\n  x: int"},
        tmp_path,
    )
    point = result["point"]
    assert isinstance(point, RecordValue)
    assert point.fields["x"] == IntValue(7)
