"""IR evaluation tests for coercion round-trips.

Each test evaluates a source program through the IR pipeline
(lower_module → IrInterpreter) and asserts the final binding snapshots.

Where noted, a test also structurally asserts that the lowered IR contains
the expected ``IrCoerce``/``Coercion`` node (complementing the golden
tests in ``test_agl_lower.py``).

Coercion families covered
--------------------------
1. Identity (no coercion) — scalar, array, dict.
2. ``IntToDecimal`` — scalar ``let d: decimal = 1``.
3. ``IntToDecimal`` — nested in an array literal (element-level).
4. ``IntToDecimal`` — not reachable as whole-value container coercion
   (``array[int] → array[decimal]`` is rejected by the type checker) — deferred.
5. ``IntToDecimal`` — dict values whole-value coercion — not reachable
   (``dict[text, int] → dict[text, decimal]`` rejected by checker) — deferred.
6. ``ToJson`` — scalar (``let j: json = 42``).
7. ``ToJson`` — array literal (``let j: json = [1, 2]``).
8. ``ToJson`` — dict literal (``let m: dict[text, json] = {"a": 1}``).
9. ``ToJson`` — via record-like dict (``let r: json = {"x": 1}``).
10. ``MapArray(ToJson)`` — whole-value via var ref (array[int] → json).
11. ``MapDictValues(ToJson)`` — whole-value via var ref (dict[text, int] → json).
12. ``MapDictValues(IntToDecimal)`` — not reachable — deferred (see #5).
13. Deeper nesting — ``array[dict[text, decimal]]``.
14. Mutable assignment with coercion — ``var x: decimal = 0; x := 5``.
15. Multiple bindings — several coercions in one program.
16. "Effectful once" note — coercion does not duplicate evaluation (trivial for pure programs).
"""

from __future__ import annotations

import decimal

from agm.agl.ir.nodes import IrCoerce, IrMakeArray, IrMakeDict
from agm.agl.ir.operations import IntToDecimal, ToJson
from agm.agl.ir.program import ExecutableProgram
from agm.agl.lower import lower_module
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.semantics.values import (
    ArrayValue,
    DecimalValue,
    DictValue,
    IntValue,
    JsonValue,
    TextValue,
    Value,
)
from agm.agl.typecheck import check_module
from tests._agl_helpers import let_root_capture
from tests.agl.ir_harness import _compiled_checked, base_caps, evaluate_ir

# ---------------------------------------------------------------------------
# Shared pipeline helper for structural IR assertions
# ---------------------------------------------------------------------------


def _lower(source: str) -> ExecutableProgram:
    checked = check_module(resolve_module(parse_program(source)), base_caps())
    return lower_module(_compiled_checked(checked), source_text=source, source_label="<test>")


# ---------------------------------------------------------------------------
# 1. Identity cases (no coercion) — control group
# ---------------------------------------------------------------------------


def test_identity_int_let() -> None:
    """let x: int = 5  — no coercion, IR pipeline returns IntValue(5)."""
    source = "let x: int = 5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(5)

    # Structural: the IR bind value is a plain IrConstInt (no IrCoerce).
    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind = let_root_capture(inits[0])
    assert not isinstance(bind.value, IrCoerce)


def test_identity_text_let() -> None:
    """let s: text = \"hello\"  — no coercion."""
    source = 'let s: text = "hello"\n()'
    ir = evaluate_ir(source)
    assert ir["s"] == TextValue("hello")


def test_identity_array_int() -> None:
    """let xs: array[int] = [1, 2, 3]  — no coercion at element or array level."""
    source = "let xs: array[int] = [1, 2, 3]\n()"
    ir = evaluate_ir(source)
    assert ir["xs"] == ArrayValue((IntValue(1), IntValue(2), IntValue(3)))

    # Structural: no IrCoerce around the IrMakeArray.
    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind = let_root_capture(inits[0])
    assert isinstance(bind.value, IrMakeArray)
    for item in bind.value.items:
        assert not isinstance(item, IrCoerce)


def test_identity_dict_text_int() -> None:
    """let d: dict[text, int] = {\"a\": 1}  — no coercion."""
    source = 'let d: dict[text, int] = {"a": 1}\n()'
    ir = evaluate_ir(source)
    assert ir["d"] == DictValue({"a": IntValue(1)})


# ---------------------------------------------------------------------------
# 2. IntToDecimal — scalar widening
# ---------------------------------------------------------------------------


def test_int_to_decimal_scalar() -> None:
    """let d: decimal = 1  — scalar IntToDecimal coercion at the binding level."""
    source = "let d: decimal = 1\n()"
    ir = evaluate_ir(source)
    assert ir["d"] == DecimalValue(decimal.Decimal(1))

    # Structural: the bind value is IrCoerce(IrConstInt, IntToDecimal).
    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind = let_root_capture(inits[0])
    coerce = bind.value
    assert isinstance(coerce, IrCoerce)
    assert coerce.operation == IntToDecimal()


def test_int_to_decimal_zero() -> None:
    """let z: decimal = 0  — widening of zero."""
    source = "let z: decimal = 0\n()"
    ir = evaluate_ir(source)
    assert ir["z"] == DecimalValue(decimal.Decimal(0))


# ---------------------------------------------------------------------------
# 3. Element-level coercion in an array literal
# ---------------------------------------------------------------------------


def test_array_decimal_element_coercion() -> None:
    """let xs: array[decimal] = [1, 2, 3]  — each element gets IntToDecimal."""
    source = "let xs: array[decimal] = [1, 2, 3]\n()"
    ir = evaluate_ir(source)
    expected: Value = ArrayValue(
        (
            DecimalValue(decimal.Decimal(1)),
            DecimalValue(decimal.Decimal(2)),
            DecimalValue(decimal.Decimal(3)),
        )
    )
    assert ir["xs"] == expected

    # Structural: elements inside IrMakeArray are wrapped in IrCoerce(IntToDecimal).
    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind = let_root_capture(inits[0])
    make_list = bind.value
    assert isinstance(make_list, IrMakeArray)
    for item in make_list.items:
        assert isinstance(item, IrCoerce)
        assert item.operation == IntToDecimal()


# ---------------------------------------------------------------------------
# 4. Whole-value container coercion via literal (MapArray, MapDictValues)
#
# Note: the current type checker's ``is_assignable`` does NOT allow
# ``array[int] → array[decimal]`` or ``dict[text,int] → dict[text,decimal]``
# at binding boundaries — those require exact structural match (or int→decimal
# scalar widening, which is not extended to containers by ``is_assignable``).
# The MapArray/MapDictValues coercions from ``compile_coercion`` are wired into
# the IR machinery and exercised by the lowerer unit tests; they do not arise
# from valid programs that the current type checker accepts.
#
# What we CAN prove is that these coercions are correctly applied at
# literal-element boundaries (already covered in test 3 above) and via ToJson
# (covered in tests 8–10 below).  The whole-value container widening path will
# be exercised once the type checker is extended in a future change.
# ---------------------------------------------------------------------------


def test_array_ref_identity_no_coercion() -> None:
    """let a: array[int] = [1, 2]; let b: array[int] = a  — exact type, no coercion."""
    source = "let a: array[int] = [1, 2]\nlet b: array[int] = a\n()"
    ir = evaluate_ir(source)
    expected: Value = ArrayValue((IntValue(1), IntValue(2)))
    assert ir["a"] == expected
    assert ir["b"] == expected

    # Structural: IrBind for b is a plain IrLoad (no IrCoerce).
    from agm.agl.ir.nodes import IrLoad

    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind_b = let_root_capture(inits[1])
    assert isinstance(bind_b.value, IrLoad)


def test_dict_ref_identity_no_coercion() -> None:
    """let a: dict[text, int] = {\"x\": 1}; let b: dict[text, int] = a  — no coercion."""
    source = 'let a: dict[text, int] = {"x": 1}\nlet b: dict[text, int] = a\n()'
    ir = evaluate_ir(source)
    expected: Value = DictValue({"x": IntValue(1)})
    assert ir["b"] == expected


# ---------------------------------------------------------------------------
# 5. Array-to-JSON coercion via var ref (whole-value MapArray not applicable,
#    but ToJson of an array[int] variable IS allowed because json accepts any
#    JSON-shaped type — proven here)
# ---------------------------------------------------------------------------


def test_array_int_ref_to_json() -> None:
    """let a: array[int] = [1, 2]; let j: json = a  — whole array coerced via ToJson."""
    source = "let a: array[int] = [1, 2]\nlet j: json = a\n()"
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue([1, 2])

    # Structural: IrBind for j wraps IrLoad(a) in IrCoerce(ToJson).
    from agm.agl.ir.nodes import IrLoad

    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind_j = let_root_capture(inits[1])
    coerce = bind_j.value
    assert isinstance(coerce, IrCoerce)
    assert coerce.operation == ToJson()
    assert isinstance(coerce.value, IrLoad)


# ---------------------------------------------------------------------------
# 6. ToJson — scalar
# ---------------------------------------------------------------------------


def test_to_json_scalar_int() -> None:
    """let j: json = 42  — scalar int wrapped in JsonValue."""
    source = "let j: json = 42\n()"
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue(42)


def test_to_json_scalar_text() -> None:
    """let j: json = \"hello\"  — scalar text wrapped in JsonValue."""
    source = 'let j: json = "hello"\n()'
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue("hello")


def test_to_json_null() -> None:
    """let j: json = null  — null is already JSON; no coercion needed."""
    source = "let j: json = null\n()"
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue(None)


# ---------------------------------------------------------------------------
# 7. ToJson — array literal
# ---------------------------------------------------------------------------


def test_to_json_array_literal() -> None:
    """let j: json = [1, 2]  — array[int] literal coerced to JsonValue."""
    source = "let j: json = [1, 2]\n()"
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue([1, 2])

    # Structural: the whole IrMakeArray is wrapped in IrCoerce(ToJson).
    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind = let_root_capture(inits[0])
    coerce = bind.value
    assert isinstance(coerce, IrCoerce)
    assert coerce.operation == ToJson()
    assert isinstance(coerce.value, IrMakeArray)


# ---------------------------------------------------------------------------
# 8. ToJson — dict literal value coercion
# ---------------------------------------------------------------------------


def test_dict_text_json_from_int_values() -> None:
    """let m: dict[text, json] = {\"a\": 1}  — dict values coerced to JSON."""
    source = 'let m: dict[text, json] = {"a": 1}\n()'
    ir = evaluate_ir(source)
    assert ir["m"] == DictValue({"a": JsonValue(1)})

    # Structural: the IrMakeDict entry VALUE nodes are wrapped in IrCoerce(ToJson).
    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind = let_root_capture(inits[0])
    make_dict = bind.value
    assert isinstance(make_dict, IrMakeDict)
    for _key_node, value_node in make_dict.entries:
        assert isinstance(value_node, IrCoerce)
        assert value_node.operation == ToJson()


# ---------------------------------------------------------------------------
# 9. ToJson — dict literal as json
# ---------------------------------------------------------------------------


def test_to_json_dict_literal_as_json() -> None:
    """let r: json = {\"x\": 1}  — dict literal coerced to JsonValue."""
    source = 'let r: json = {"x": 1}\n()'
    ir = evaluate_ir(source)
    assert ir["r"] == JsonValue({"x": 1})


# ---------------------------------------------------------------------------
# 10. Dict[text, json] from dict[text, int] via ref — via ToJson (whole-value)
#
# The current type checker DOES allow dict[text, int] → json because a dict of
# ints is JSON-shaped.  However, dict[text, int] → dict[text, json] is not
# allowed (requires exact structural match).  We prove the json-target path.
# ---------------------------------------------------------------------------


def test_dict_int_ref_to_json() -> None:
    """let a: dict[text, int] = {\"k\": 5}; let j: json = a  — whole dict coerced via ToJson."""
    source = 'let a: dict[text, int] = {"k": 5}\nlet j: json = a\n()'
    ir = evaluate_ir(source)
    assert ir["j"] == JsonValue({"k": 5})

    # Structural: IrCoerce(ToJson) around the IrLoad(a).
    from agm.agl.ir.nodes import IrLoad

    prog = _lower(source)
    inits = prog.modules[prog.entry_module].initializers
    bind_j = let_root_capture(inits[1])
    coerce = bind_j.value
    assert isinstance(coerce, IrCoerce)
    assert coerce.operation == ToJson()
    assert isinstance(coerce.value, IrLoad)


# ---------------------------------------------------------------------------
# 11. Deeper nesting — array[dict[text, decimal]]
# ---------------------------------------------------------------------------


def test_nested_array_dict_decimal() -> None:
    """let n: array[dict[text, decimal]] = [{\"a\": 1}].

    Element-level coercion (IntToDecimal on dict value) applied inside
    each dict element of the outer array.
    """
    source = 'let n: array[dict[text, decimal]] = [{"a": 1}]\n()'
    ir = evaluate_ir(source)
    expected: Value = ArrayValue((DictValue({"a": DecimalValue(decimal.Decimal(1))}),))
    assert ir["n"] == expected


def test_nested_array_dict_json() -> None:
    """let n: array[dict[text, json]] = [{\"a\": 1, \"b\": true}]."""
    source = 'let n: array[dict[text, json]] = [{"a": 1, "b": true}]\n()'
    ir = evaluate_ir(source)
    expected: Value = ArrayValue((DictValue({"a": JsonValue(1), "b": JsonValue(True)}),))
    assert ir["n"] == expected


# ---------------------------------------------------------------------------
# 12. Mutable assignment coercion (var + :=)
# ---------------------------------------------------------------------------


def test_var_assign_int_to_decimal() -> None:
    """var x: decimal = 0; x := 5  — coercion on both initial binding and assignment."""
    source = "var x: decimal = 0\nx := 5\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal(5))


def test_var_assign_preserves_type() -> None:
    """var x: decimal = 1.5; x := 2  — init is decimal literal, assign coerces int."""
    source = "var x: decimal = 1.5\nx := 2\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal(2))


def test_var_assign_multiple_coercions() -> None:
    """var x: decimal = 0; x := 1; x := 2  — repeated coerced assignments."""
    source = "var x: decimal = 0\nx := 1\nx := 2\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == DecimalValue(decimal.Decimal(2))


# ---------------------------------------------------------------------------
# 13. Multiple bindings in one program
# ---------------------------------------------------------------------------


def test_multiple_coercions_in_one_program() -> None:
    """Program with several binding coercions; each is applied correctly."""
    source = (
        "let a: decimal = 1\n"
        "let b: array[decimal] = [2, 3]\n"
        "let c: json = 4\n"
        'let d: dict[text, json] = {"x": 5}\n'
        "()"
    )
    ir = evaluate_ir(source)
    assert ir["a"] == DecimalValue(decimal.Decimal(1))
    assert ir["b"] == ArrayValue(
        (DecimalValue(decimal.Decimal(2)), DecimalValue(decimal.Decimal(3)))
    )
    assert ir["c"] == JsonValue(4)
    assert ir["d"] == DictValue({"x": JsonValue(5)})


# ---------------------------------------------------------------------------
# 14. Var binding + load (no assignment)
# ---------------------------------------------------------------------------


def test_var_binding_no_assign() -> None:
    """var x: int = 7  — mutable binding, no reassignment, both return 7."""
    source = "var x: int = 7\n()"
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(7)


# ---------------------------------------------------------------------------
# 15. Empty containers
# ---------------------------------------------------------------------------


def test_empty_array() -> None:
    """let xs: array[int] = []  — empty array; no coercions."""
    source = "let xs: array[int] = []\n()"
    ir = evaluate_ir(source)
    assert ir["xs"] == ArrayValue(())


def test_empty_dict() -> None:
    """let d: dict[text, int] = {}  — empty dict; no coercions."""
    source = "let d: dict[text, int] = {}\n()"
    ir = evaluate_ir(source)
    assert ir["d"] == DictValue({})


# ---------------------------------------------------------------------------
# 16. "Effectful once" note (trivial for pure programs, documents extension point)
# ---------------------------------------------------------------------------


def test_coercion_evaluates_operand_once() -> None:
    """Coercion does not duplicate evaluation of its operand.

    All current AgL expressions are pure (no side effects), so the strong version
    of this property (observable side effects fired exactly once) cannot be
    violated.  The IR pipeline produces the correct coercion result.  The stronger
    test (with counters/exec side-effects) is deferred to a future coverage.

    We exercise this with a multi-step program where each binding's RHS is a
    constant; the IR pipeline must produce the correct value for each binding.
    """
    source = "let a: decimal = 10\nlet b: decimal = 20\nlet c: array[decimal] = [a, b]\n()"
    ir = evaluate_ir(source)
    assert ir["a"] == DecimalValue(decimal.Decimal(10))
    assert ir["b"] == DecimalValue(decimal.Decimal(20))
    assert ir["c"] == ArrayValue(
        (DecimalValue(decimal.Decimal(10)), DecimalValue(decimal.Decimal(20)))
    )


# ---------------------------------------------------------------------------
# 17. Additional coercion families — array[json] from array[text]
# ---------------------------------------------------------------------------


def test_array_text_to_array_json_element_coercion() -> None:
    """let xs: array[json] = [\"a\", \"b\"]  — element-level ToJson."""
    source = 'let xs: array[json] = ["a", "b"]\n()'
    ir = evaluate_ir(source)
    assert ir["xs"] == ArrayValue((JsonValue("a"), JsonValue("b")))
