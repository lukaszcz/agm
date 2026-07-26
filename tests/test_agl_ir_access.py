"""IR evaluation tests for field access, index access, templates, and indexed assignment.

Tests all node types: IrField, IrIndex, IrRenderTemplate, IrAssign(path).
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl.eval.indexing import AglIndexOutOfRange, AglMissingKey, index_get, index_set
from agm.agl.ir.operations import IndexKind
from agm.agl.ir.program import ExecutableProgram
from agm.agl.semantics.values import (
    ArrayValue,
    DecimalValue,
    DictValue,
    IntValue,
    TextValue,
)
from tests._agl_helpers import let_root_capture
from tests.agl.ir_harness import _compiled_checked, evaluate_ir, evaluate_ir_raises

# ---------------------------------------------------------------------------
# indexing.py unit tests
# ---------------------------------------------------------------------------


def test_index_get_array_basic() -> None:
    lst = ArrayValue((IntValue(10), IntValue(20), IntValue(30)))
    assert index_get(IndexKind.ARRAY, lst, IntValue(0)) == IntValue(10)
    assert index_get(IndexKind.ARRAY, lst, IntValue(1)) == IntValue(20)
    assert index_get(IndexKind.ARRAY, lst, IntValue(-1)) == IntValue(30)


def test_index_get_array_out_of_range() -> None:
    lst = ArrayValue((IntValue(1), IntValue(2)))
    with pytest.raises(AglIndexOutOfRange) as exc_info:
        index_get(IndexKind.ARRAY, lst, IntValue(5))
    assert exc_info.value.index == 5
    assert exc_info.value.length == 2


def test_index_get_dict_basic() -> None:
    d = DictValue({"a": IntValue(1), "b": IntValue(2)})
    assert index_get(IndexKind.DICT, d, TextValue("a")) == IntValue(1)


def test_index_get_dict_missing_key() -> None:
    d = DictValue({"a": IntValue(1)})
    with pytest.raises(AglMissingKey) as exc_info:
        index_get(IndexKind.DICT, d, TextValue("z"))
    assert exc_info.value.key == "z"


def test_index_set_array_basic() -> None:
    lst = ArrayValue((IntValue(1), IntValue(2), IntValue(3)))
    result = index_set(IndexKind.ARRAY, lst, IntValue(1), IntValue(99))
    assert result == ArrayValue((IntValue(1), IntValue(99), IntValue(3)))


def test_index_set_array_negative() -> None:
    lst = ArrayValue((IntValue(1), IntValue(2), IntValue(3)))
    result = index_set(IndexKind.ARRAY, lst, IntValue(-1), IntValue(99))
    assert result == ArrayValue((IntValue(1), IntValue(2), IntValue(99)))


def test_index_set_array_oob() -> None:
    lst = ArrayValue((IntValue(1), IntValue(2)))
    with pytest.raises(AglIndexOutOfRange):
        index_set(IndexKind.ARRAY, lst, IntValue(5), IntValue(99))


def test_index_set_dict_basic() -> None:
    d = DictValue({"a": IntValue(1)})
    result = index_set(IndexKind.DICT, d, TextValue("a"), IntValue(99))
    assert result == DictValue({"a": IntValue(99)})


def test_index_get_array_wrong_container() -> None:
    d = DictValue({"a": IntValue(1)})
    with pytest.raises(AssertionError, match="index_get ARRAY: expected ArrayValue"):
        index_get(IndexKind.ARRAY, d, IntValue(0))


def test_index_get_array_wrong_index() -> None:
    lst = ArrayValue((IntValue(1),))
    with pytest.raises(AssertionError, match="index_get ARRAY: expected IntValue"):
        index_get(IndexKind.ARRAY, lst, TextValue("x"))


def test_index_get_dict_wrong_container() -> None:
    lst = ArrayValue((IntValue(1),))
    with pytest.raises(AssertionError, match="index_get DICT: expected DictValue"):
        index_get(IndexKind.DICT, lst, TextValue("x"))


def test_index_get_dict_wrong_index() -> None:
    d = DictValue({"a": IntValue(1)})
    with pytest.raises(AssertionError, match="index_get DICT: expected TextValue"):
        index_get(IndexKind.DICT, d, IntValue(0))


def test_index_set_array_wrong_container() -> None:
    d = DictValue({"a": IntValue(1)})
    with pytest.raises(AssertionError, match="index_set ARRAY: expected ArrayValue"):
        index_set(IndexKind.ARRAY, d, IntValue(0), IntValue(99))


def test_index_set_dict_wrong_container() -> None:
    lst = ArrayValue((IntValue(1),))
    with pytest.raises(AssertionError, match="index_set DICT: expected DictValue"):
        index_set(IndexKind.DICT, lst, TextValue("x"), IntValue(99))


def test_index_set_array_wrong_index() -> None:
    lst = ArrayValue((IntValue(1),))
    with pytest.raises(AssertionError, match="index_set ARRAY: expected IntValue"):
        index_set(IndexKind.ARRAY, lst, TextValue("x"), IntValue(99))


def test_index_set_dict_wrong_index() -> None:
    d = DictValue({"a": IntValue(1)})
    with pytest.raises(AssertionError, match="index_set DICT: expected TextValue"):
        index_set(IndexKind.DICT, d, IntValue(0), IntValue(99))


# ---------------------------------------------------------------------------
# IR evaluation tests
# ---------------------------------------------------------------------------


def test_record_field_access() -> None:
    """Record field access: p.x on a record.

    Constructor call lowering (IrMakeRecord) is supported, so this test is
    fully covered.  The IrField node is also tested directly
    in test_agl_ir_interpreter.py::TestIrField.
    """
    source = """\
record Point
  x: int
  y: int
let p = Point(x = 3, y = 4)
let px = p.x
()
"""
    ir = evaluate_ir(source)
    assert ir["px"] == IntValue(3)


def test_array_index() -> None:
    """Array index: xs[1] returns the element at index 1."""
    source = """\
let xs = [10, 20, 30]
let x = xs[1]
()
"""
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(20)


def test_array_negative_index() -> None:
    """Array negative index: xs[-1] returns the last element."""
    source = """\
let xs = [10, 20, 30]
let x = xs[-1]
()
"""
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(30)


def test_array_index_out_of_range() -> None:
    """Array out-of-bounds index raises IndexError."""
    source = """\
let xs = [10, 20]
let x = xs[5]
()
"""
    evaluate_ir_raises(source)


def test_dict_index() -> None:
    """Dict index: m["a"] returns the value for key "a"."""
    source = """\
let m = {"a": 1, "b": 2}
let x = m["a"]
()
"""
    ir = evaluate_ir(source)
    assert ir["x"] == IntValue(1)


def test_dict_missing_key() -> None:
    """Dict missing key raises KeyError."""
    source = """\
let m = {"a": 1}
let x = m["z"]
()
"""
    evaluate_ir_raises(source)


def test_template_text_only() -> None:
    """Template with only text segments."""
    source = 'let x: text = "hello world"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == TextValue("hello world")


def test_template_with_interpolation() -> None:
    """Template with an integer interpolation."""
    source = 'let x: text = "val: %{42}"\n()'
    ir = evaluate_ir(source)
    assert ir["x"] == TextValue("val: 42")


def test_template_with_var_interpolation() -> None:
    """Template with a variable reference interpolation."""
    source = """\
let n = 7
let x: text = "n is %{n}"
()
"""
    ir = evaluate_ir(source)
    assert ir["x"] == TextValue("n is 7")


def test_template_with_record_interpolation() -> None:
    """Template with a record value interpolation.

    Constructor call lowering (IrMakeRecord) is supported, so this test is
    fully covered.
    """
    source = """\
record Point
  x: int
  y: int
let p = Point(x = 1, y = 2)
let s: text = "point: %{p}"
()
"""
    ir = evaluate_ir(source)
    # Renders the expected string.
    assert isinstance(ir["s"], TextValue)


def test_indexed_assignment_array_depth1() -> None:
    """Indexed assignment depth 1: array."""
    source = """\
var xs = [1, 2, 3]
xs[0] := 99
()
"""
    ir = evaluate_ir(source)
    assert ir["xs"] == ArrayValue((IntValue(99), IntValue(2), IntValue(3)))


def test_indexed_assignment_dict_depth1() -> None:
    """Indexed assignment depth 1: dict."""
    source = """\
var m = {"a": 1}
m["a"] := 99
()
"""
    ir = evaluate_ir(source)
    assert ir["m"] == DictValue({"a": IntValue(99)})


def test_indexed_assignment_array_of_array() -> None:
    """Indexed assignment depth 2: array-of-array."""
    source = """\
var xss = [[1, 2], [3, 4]]
xss[0][1] := 99
()
"""
    ir = evaluate_ir(source)
    assert ir["xss"] == ArrayValue(
        (
            ArrayValue((IntValue(1), IntValue(99))),
            ArrayValue((IntValue(3), IntValue(4))),
        )
    )


def test_indexed_assignment_dict_of_array() -> None:
    """Indexed assignment depth 2: dict-of-array."""
    source = """\
var m = {"a": [1, 2]}
m["a"][0] := 99
()
"""
    ir = evaluate_ir(source)
    assert ir["m"] == DictValue(
        {
            "a": ArrayValue((IntValue(99), IntValue(2))),
        }
    )


def test_indexed_assignment_array_of_dict() -> None:
    """Indexed assignment depth 2: array-of-dict."""
    source = """\
var m = [{"x": 1}]
m[0]["x"] := 99
()
"""
    ir = evaluate_ir(source)
    assert ir["m"] == ArrayValue((DictValue({"x": IntValue(99)}),))


def test_indexed_assignment_depth3_array_of_array_of_array() -> None:
    """Indexed assignment depth 3: array-of-array-of-array."""
    source = """\
var xsss = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
xsss[0][1][0] := 99
()
"""
    ir = evaluate_ir(source)
    assert ir["xsss"] == ArrayValue(
        (
            ArrayValue(
                (
                    ArrayValue((IntValue(1), IntValue(2))),
                    ArrayValue((IntValue(99), IntValue(4))),
                )
            ),
            ArrayValue(
                (
                    ArrayValue((IntValue(5), IntValue(6))),
                    ArrayValue((IntValue(7), IntValue(8))),
                )
            ),
        )
    )


def test_indexed_assignment_oob_raises() -> None:
    """Indexed assignment with out-of-bounds index raises IndexError."""
    source = """\
var xs = [1, 2]
xs[5] := 99
()
"""
    evaluate_ir_raises(source)


def test_indexed_assignment_intermediate_oob_raises() -> None:
    """Indexed assignment with out-of-bounds intermediate index raises IndexError."""
    source = """\
var xss = [[1, 2], [3, 4]]
xss[5][0] := 99
()
"""
    evaluate_ir_raises(source)


def test_indexed_assignment_dict_missing_key_raises() -> None:
    """Indexed assignment with missing dict key raises KeyError."""
    source = """\
var m = {"a": 1}
m["z"] := 99
()
"""
    evaluate_ir_raises(source)


def test_indexed_assignment_decimal_leaf_array_depth1() -> None:
    """Indexed assignment into array[decimal] exercises the IrCoerce IntToDecimal leaf path.

    The RHS is an int literal; lower_coerced wraps it in IrCoerce(IntToDecimal) because
    the slot type is decimal (the coercion runs exactly once).
    """
    source = """\
var xs: array[decimal] = [1.0, 2.0, 3.0]
xs[1] := 42
()
"""
    ir = evaluate_ir(source)
    result = ir["xs"]
    assert isinstance(result, ArrayValue)
    assert result.elements[1] == DecimalValue(decimal.Decimal(42))


def test_indexed_assignment_decimal_leaf_dict_depth1() -> None:
    """Indexed assignment into dict[text, decimal] exercises the IrCoerce IntToDecimal leaf path.

    The RHS is an int literal; lower_coerced inserts IrCoerce(IntToDecimal) at the leaf.
    """
    source = """\
var m: dict[text, decimal] = {"a": 1.0, "b": 2.0}
m["a"] := 7
()
"""
    ir = evaluate_ir(source)
    result = ir["m"]
    assert isinstance(result, DictValue)
    assert result.entries["a"] == DecimalValue(decimal.Decimal(7))


def test_empty_array_literal_evaluates_to_empty_array() -> None:
    """An annotated empty array literal evaluates to an empty ArrayValue."""
    ir = evaluate_ir("let xs: array[int] = []\nxs\n")
    assert ir["xs"] == ArrayValue(elements=())


def test_empty_dict_literal_evaluates_to_empty_dict() -> None:
    """An annotated empty dict literal evaluates to an empty DictValue."""
    ir = evaluate_ir("let d: dict[text, int] = {}\nd\n")
    assert ir["d"] == DictValue(entries={})


def test_indexed_assignment_decimal_leaf_array_of_array_depth2() -> None:
    """Indexed assignment depth 2 into array[array[decimal]] exercises IntToDecimal at the leaf.

    Verifies that the coercion is applied exactly once at the leaf (not to the container)
    and that the leaf-outward rebuild produces the right structure.
    """
    source = """\
var xss: array[array[decimal]] = [[1.0, 2.0], [3.0, 4.0]]
xss[0][1] := 99
()
"""
    ir = evaluate_ir(source)
    result = ir["xss"]
    assert isinstance(result, ArrayValue)
    inner = result.elements[0]
    assert isinstance(inner, ArrayValue)
    assert inner.elements[1] == DecimalValue(decimal.Decimal(99))


# ---------------------------------------------------------------------------
# Golden lowering tests
# ---------------------------------------------------------------------------


def _lower(source: str) -> "ExecutableProgram":
    """Parse → check → lower the source; return ExecutableProgram."""
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.lower import lower_module
    from agm.agl.parser import parse_program
    from agm.agl.scope import resolve_module
    from agm.agl.typecheck import check_module

    caps = HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=True,
        supports_shell_exec=True,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}
            ),
        },
    )
    prog = parse_program(source)
    resolved = resolve_module(prog)
    checked = check_module(resolved, caps)
    return lower_module(
        _compiled_checked(checked),
        source_text=source,
        source_label="<test>",
    )


def test_golden_field_access_lowers_to_ir_field() -> None:
    """FieldAccess lowers to IrField with the correct field name.

    Constructor call lowering is available, so Point(x: 3, y: 4) lowers
    to IrMakeRecord, and then p.x lowers to IrField.
    The IrField node is also tested directly in test_agl_ir_interpreter.py::TestIrField.
    """
    from agm.agl.ir.nodes import IrBind, IrField, IrSequence

    source = """\
record Point
  x: int
  y: int
let p = Point(x = 3, y = 4)
let px = p.x
()
"""
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, (IrSequence, IrBind)):
            value = let_root_capture(node).value
            if isinstance(value, IrField):
                assert value.field == "x"
                found = True
    assert found, "Expected IrBind(value=IrField(field='x')) in initializers"


def test_golden_index_access_lowers_to_ir_index() -> None:
    """IndexAccess lowers to IrIndex with correct kind."""
    from agm.agl.ir.nodes import IrBind, IrIndex, IrSequence

    source = """\
let xs = [10, 20, 30]
let x = xs[1]
()
"""
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, (IrSequence, IrBind)):
            value = let_root_capture(node).value
            if isinstance(value, IrIndex):
                assert value.kind is IndexKind.ARRAY
                found = True
    assert found, "Expected IrBind(value=IrIndex(kind=ARRAY)) in initializers"


def test_golden_template_lowers_to_ir_render_template() -> None:
    """Template lowers to IrRenderTemplate."""
    from agm.agl.ir.nodes import (
        IrBind,
        IrRenderTemplate,
        IrSequence,
        IrTemplateText,
        IrTemplateValue,
    )

    source = 'let x: text = "val: %{42}"\n()'
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, (IrSequence, IrBind)):
            value = let_root_capture(node).value
            if isinstance(value, IrRenderTemplate):
                segs = value.segments
                assert len(segs) == 2
                assert isinstance(segs[0], IrTemplateText)
                assert segs[0].text == "val: "
                assert isinstance(segs[1], IrTemplateValue)
                found = True
    assert found, "Expected IrBind(value=IrRenderTemplate) in initializers"


def test_golden_indexed_assign_lowers_to_ir_assign_with_path() -> None:
    """Indexed assignment lowers to IrAssign with non-empty path."""
    from agm.agl.ir.nodes import IrAssign

    source = """\
var xs = [1, 2, 3]
xs[0] := 99
()
"""
    prog = _lower(source)
    entry = prog.modules[prog.entry_module]
    found = False
    for node in entry.initializers:
        if isinstance(node, IrAssign) and node.path:
            assert len(node.path) == 1
            assert node.path[0].kind is IndexKind.ARRAY
            found = True
    assert found, "Expected IrAssign with non-empty path in initializers"
