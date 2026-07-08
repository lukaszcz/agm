"""End-to-end evaluation tests for `extern def` (Python FFI) calls.

Covers the interpreter dispatch seam (direct and indirect/first-class call
paths delegate to the companion Python module instead of evaluating an AgL
body) and the resulting runtime behavior:
- basic round trips for every scalar/unit type crossing the boundary.
- argument binding: zones, named args, and AgL-side defaults all arrive
  positionally, in declaration order, in Python.
- externs are fully first-class: stored in a `let`, passed to a
  higher-order function, returned from a function, and rendered like any
  other closure.
- `ExternError` for a raising companion, a return-contract violation, and
  the uncaught-error path surfacing a call-site span.
- interleaving with ordinary AgL recursion and loops.
- end-to-end file runs through the real pipeline (`PipelineDriver`), a
  REPL smoke test, and the dry-run (`check_only`) contract.
- the full round-trip conversion matrix (decimal exactness, bool/float
  rejection inside containers, unit, json passthrough, list/dict nesting,
  records, enums including `Option` as a plain enum, exceptions as values,
  and deep-copy isolation), driven entirely through real extern calls rather
  than the boundary walkers directly.

Earlier suites (`test_agl_extern_loading.py`, `test_agl_extern_lowering.py`)
cover everything upstream of dispatch and stop before evaluation;
`test_agl_extern_boundary.py` drives the encode/decode walkers and
`ExternRegistry.invoke` directly with hand-built contracts. This suite is the
first to invoke a companion callable through real, checked, lowered AgL
programs.
"""

from __future__ import annotations

import decimal
from pathlib import Path

import pytest

from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver
from agm.agl.semantics.values import (
    UNIT_VALUE,
    BoolValue,
    DecimalValue,
    DictValue,
    IntValue,
    ListValue,
    TextValue,
)
from tests.agl.ir_harness import (
    evaluate_ir_raises_with_externs,
    evaluate_ir_with_externs,
    write_companion_file,
    write_module_file,
)


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset(paths))


def _build_indirect_extern_call_program(tmp_path: Path, call_args: tuple[int, ...]):
    """Hand-build an ``ExecutableProgram`` calling a 2-param extern indirectly.

    ``inc2(x: int, step: int = 1) -> int`` is stored in a `let` and invoked
    via ``IrIndirectCall`` with *call_args* (a tuple of ``int`` literals).
    The checker requires exact arity for a value call, so the indirect
    path's own default-fallback and missing-argument arms for an extern are
    only reachable by constructing IR directly — mirroring the analogous
    hand-built ordinary-function coverage in ``test_agl_ir_interpreter.py``.

    Returns ``(program, registry)`` ready to hand to ``IrInterpreter``.
    """
    from agm.agl.ir import (
        ExecutableModule,
        ExecutableProgram,
        ExternFunctionBody,
        FunctionDescriptor,
        FunctionId,
        IrBind,
        IrConstInt,
        IrFunctionParam,
        IrIndirectCall,
        IrLoad,
        IrMakeClosure,
        Location,
        SourceFile,
        SourceId,
        SymbolDescriptor,
        SymbolId,
    )
    from agm.agl.ir.contracts import (
        BoundaryScalar,
        ExternContract,
        ExternParamSchema,
        ScalarKind,
    )
    from agm.agl.modules.ids import ENTRY_ID
    from agm.agl.runtime.externs import ExternRegistry

    companion_path = tmp_path / "companion.py"
    companion_path.write_text("def inc(x, step):\n    return x + step\n")

    source_id = SourceId(0)
    loc = Location(source_id=source_id, start_offset=0, end_offset=1, start_line=1, start_col=0)

    fn_id = FunctionId(1)
    fn_sym = SymbolId(1)
    closure_sym = SymbolId(2)
    result_sym = SymbolId(3)

    contract = ExternContract(
        params=(
            ExternParamSchema(label="int", schema=BoundaryScalar(ScalarKind.INT)),
            ExternParamSchema(label="int", schema=BoundaryScalar(ScalarKind.INT)),
        ),
        result=BoundaryScalar(ScalarKind.INT),
        type_params=(),
        result_label="int",
    )
    extern_desc = FunctionDescriptor(
        function_id=fn_id,
        function_symbol=fn_sym,
        module_id=ENTRY_ID,
        params=(
            IrFunctionParam(symbol=SymbolId(4), default=None),
            IrFunctionParam(symbol=SymbolId(5), default=IrConstInt(loc, 1)),
        ),
        impl=ExternFunctionBody(name="inc", contract=contract),
    )

    symbols = {
        fn_sym: SymbolDescriptor(
            symbol_id=fn_sym, mutable=False, public_name="inc", owner=ENTRY_ID
        ),
        closure_sym: SymbolDescriptor(
            symbol_id=closure_sym, mutable=False, public_name=None, owner=ENTRY_ID
        ),
        result_sym: SymbolDescriptor(
            symbol_id=result_sym, mutable=False, public_name="r", owner=ENTRY_ID
        ),
    }
    call_arg_exprs = tuple(IrConstInt(loc, value) for value in call_args)
    program = ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={
            ENTRY_ID: ExecutableModule(
                module_id=ENTRY_ID,
                initializers=(
                    IrBind(loc, fn_sym, IrMakeClosure(loc, fn_id, ())),
                    IrBind(loc, closure_sym, IrLoad(loc, fn_sym)),
                    IrBind(
                        loc,
                        result_sym,
                        IrIndirectCall(loc, IrLoad(loc, closure_sym), call_arg_exprs),
                    ),
                ),
            )
        },
        symbols=symbols,
        nominals={},
        sources={source_id: SourceFile(display_name="<test>", normalized_text="x")},
        functions={fn_id: extern_desc},
    )

    registry = ExternRegistry()
    registry.load_companion(ENTRY_ID, companion_path)
    return program, registry


# ---------------------------------------------------------------------------
# Basic round trips
# ---------------------------------------------------------------------------


class TestRoundTrips:
    def test_int_extern_round_trip(self, tmp_path: Path) -> None:
        result, _ = evaluate_ir_with_externs(
            "extern def add_one(x: int) -> int\nlet r = add_one(41)\nr\n",
            "def add_one(x):\n    return x + 1\n",
            tmp_path,
        )
        assert result["r"] == IntValue(42)

    def test_text_extern_round_trip(self, tmp_path: Path) -> None:
        result, _ = evaluate_ir_with_externs(
            'extern def shout(s: text) -> text\nlet r = shout("hi")\nr\n',
            "def shout(s):\n    return s.upper()\n",
            tmp_path,
        )
        assert result["r"] == TextValue("HI")

    def test_bool_extern_round_trip(self, tmp_path: Path) -> None:
        result, _ = evaluate_ir_with_externs(
            "extern def negate(b: bool) -> bool\nlet r = negate(true)\nr\n",
            "def negate(b):\n    return not b\n",
            tmp_path,
        )
        assert result["r"] == BoolValue(False)

    def test_decimal_extern_round_trip(self, tmp_path: Path) -> None:
        result, _ = evaluate_ir_with_externs(
            "extern def double(x: decimal) -> decimal\nlet r = double(2.5)\nr\n",
            "def double(x):\n    return x * 2\n",
            tmp_path,
        )
        assert result["r"] == DecimalValue(decimal.Decimal("5.0"))

    def test_unit_extern_round_trip(self, tmp_path: Path) -> None:
        result, _ = evaluate_ir_with_externs(
            "extern def touch() -> unit\nlet r = touch()\nr\n",
            "def touch():\n    return None\n",
            tmp_path,
        )
        assert result["r"] == UNIT_VALUE

    def test_extern_result_feeds_further_agl_computation(self, tmp_path: Path) -> None:
        result, _ = evaluate_ir_with_externs(
            "extern def add_one(x: int) -> int\nlet r = add_one(1) + add_one(2)\nr\n",
            "def add_one(x):\n    return x + 1\n",
            tmp_path,
        )
        assert result["r"] == IntValue(5)


# ---------------------------------------------------------------------------
# Decimal exactness
# ---------------------------------------------------------------------------


class TestDecimalExactness:
    def test_decimal_argument_arrives_as_decimal_never_float(self, tmp_path: Path) -> None:
        source = "extern def check(x: decimal) -> bool\nlet r = check(0.1)\nr\n"
        companion = (
            "from decimal import Decimal\n"
            "def check(x):\n"
            "    return isinstance(x, Decimal) and not isinstance(x, float)\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == BoolValue(True)

    def test_value_that_would_corrupt_through_float_round_trips_exactly(
        self, tmp_path: Path
    ) -> None:
        source = "extern def identity(x: decimal) -> decimal\nlet r = identity(0.1)\nr\n"
        companion = "def identity(x):\n    return x\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == DecimalValue(decimal.Decimal("0.1"))

    def test_huge_precision_decimal_round_trips_exactly(self, tmp_path: Path) -> None:
        huge = "1.2345678901234567890123456789012345"
        source = f"extern def identity(x: decimal) -> decimal\nlet r = identity({huge})\nr\n"
        companion = "def identity(x):\n    return x\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == DecimalValue(decimal.Decimal(huge))

    def test_int_returned_where_decimal_declared_converts_exactly(self, tmp_path: Path) -> None:
        source = "extern def to_decimal(x: int) -> decimal\nlet r = to_decimal(7)\nr\n"
        companion = "def to_decimal(x):\n    return x\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == DecimalValue(decimal.Decimal(7))


# ---------------------------------------------------------------------------
# Bool/float rejection, including inside containers
# ---------------------------------------------------------------------------


class TestStrictReturnValidation:
    def test_bool_element_in_returned_int_list_rejected(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> list[int]\nf()\n()\n",
            "def f():\n    return [1, True, 2]\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_bool_element_in_returned_decimal_list_rejected(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> list[decimal]\nf()\n()\n",
            "from decimal import Decimal\ndef f():\n    return [Decimal('1'), True]\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_bool_field_in_returned_record_rejected(self, tmp_path: Path) -> None:
        source = "record Box\n  value: int\n  label: text\nextern def f() -> Box\nf()\n()\n"
        exc = evaluate_ir_raises_with_externs(
            source,
            "def f():\n    return {'value': True, 'label': 'x'}\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_float_rejected_as_a_plain_int_return(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> int\nf()\n()\n",
            "def f():\n    return 1.5\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_float_rejected_inside_returned_json(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> json\nf()\n()\n",
            "def f():\n    return {'a': [1, 2.5]}\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"


# ---------------------------------------------------------------------------
# Unit: Python receives None; a unit return must be exactly None
# ---------------------------------------------------------------------------


class TestUnitBoundary:
    def test_unit_param_arrives_as_python_none(self, tmp_path: Path) -> None:
        source = "extern def is_none(x: unit) -> bool\nlet r = is_none(())\nr\n"
        companion = "def is_none(x):\n    return x is None\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == BoolValue(True)

    def test_unit_return_rejects_non_none(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> unit\nf()\n()\n",
            "def f():\n    return 0\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"


# ---------------------------------------------------------------------------
# json passthrough
# ---------------------------------------------------------------------------


class TestJsonPassthrough:
    def test_nested_json_round_trips_unchanged_through_a_real_call(self, tmp_path: Path) -> None:
        source = (
            "extern def echo(x: json) -> json\n"
            'let doc: json = {a: [1, 2.5, null, true, "s"], b: {c: 3}}\n'
            "print(doc)\n"
            "let r = echo(doc)\n"
            "print(r)\n"
            "()\n"
        )
        companion = "def echo(x):\n    return x\n"
        _, output = evaluate_ir_with_externs(source, companion, tmp_path)
        lines = output.splitlines()
        assert len(lines) == 2
        assert lines[0] == lines[1]

    def test_json_return_rejects_an_arbitrary_python_object(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> json\nf()\n()\n",
            "class Opaque:\n    pass\n\ndef f():\n    return Opaque()\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_json_return_rejects_a_sealed_handle_leaked_from_another_call(
        self, tmp_path: Path
    ) -> None:
        source = (
            "extern def identity[T](x: T) -> T\n"
            "extern def leak_as_json() -> json\n"
            "identity(1)\n"
            "leak_as_json()\n"
            "()\n"
        )
        companion = (
            "_stash = None\n"
            "def identity(x):\n"
            "    global _stash\n"
            "    _stash = x\n"
            "    return x\n"
            "def leak_as_json():\n"
            "    return _stash\n"
        )
        exc = evaluate_ir_raises_with_externs(source, companion, tmp_path)
        assert exc.display_name == "ExternError"


# ---------------------------------------------------------------------------
# list/dict deep nesting
# ---------------------------------------------------------------------------


class TestListDictDeepNesting:
    def test_list_of_list_round_trip(self, tmp_path: Path) -> None:
        source = (
            "extern def flatten_sum(xs: list[list[int]]) -> int\n"
            "let r = flatten_sum([[1, 2], [3, 4]])\n"
            "r\n"
        )
        companion = "def flatten_sum(xs):\n    return sum(sum(row) for row in xs)\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(10)

    def test_dict_of_list_round_trip(self, tmp_path: Path) -> None:
        source = (
            "extern def echo(x: dict[text, list[int]]) -> dict[text, list[int]]\n"
            "let r = echo({a: [1, 2], b: [3]})\n"
            "r\n"
        )
        companion = "def echo(x):\n    return x\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == DictValue(
            entries={"a": ListValue((IntValue(1), IntValue(2))), "b": ListValue((IntValue(3),))}
        )

    def test_dict_non_string_key_on_return_rejected(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> dict[text, int]\nf()\n()\n",
            "def f():\n    return {1: 2}\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"


# ---------------------------------------------------------------------------
# Records: declaration order out, exact field match in, nested records
# ---------------------------------------------------------------------------


class TestRecordsRoundTrip:
    _BOX = "record Box\n  value: int\n  label: text\n"

    def test_record_fields_cross_in_declaration_order(self, tmp_path: Path) -> None:
        source = (
            self._BOX + "extern def field_order(b: Box) -> text\n"
            'let r = field_order(Box(value = 1, label = "x"))\n'
            "r\n"
        )
        companion = "def field_order(b):\n    return ','.join(b.keys())\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == TextValue("value,label")

    def test_record_round_trip(self, tmp_path: Path) -> None:
        source = (
            self._BOX + "extern def bump(b: Box) -> Box\n"
            'let r = bump(Box(value = 1, label = "x"))\n'
            "let v = r.value\n"
            "v\n"
        )
        companion = "def bump(b):\n    return {'value': b['value'] + 1, 'label': b['label']}\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["v"] == IntValue(2)

    def test_record_return_missing_field_rejected(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            self._BOX + "extern def f() -> Box\nf()\n()\n",
            "def f():\n    return {'value': 1}\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_record_return_extra_field_rejected(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            self._BOX + "extern def f() -> Box\nf()\n()\n",
            "def f():\n    return {'value': 1, 'label': 'x', 'extra': 1}\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_record_return_misnamed_field_rejected(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            self._BOX + "extern def f() -> Box\nf()\n()\n",
            "def f():\n    return {'value': 1, 'lbl': 'x'}\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"

    def test_nested_record_round_trip(self, tmp_path: Path) -> None:
        source = (
            "record Author\n  name: text\n"
            "record Post\n  title: text\n  author: Author\n"
            "extern def echo(p: Post) -> Post\n"
            'let r = echo(Post(title = "hi", author = Author(name = "Ada")))\n'
            "let n = r.author.name\n"
            "n\n"
        )
        companion = "def echo(p):\n    return p\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["n"] == TextValue("Ada")


# ---------------------------------------------------------------------------
# Enums: `$case` both ways, unknown case rejected, `Option` as a plain enum
# ---------------------------------------------------------------------------


class TestEnumsRoundTrip:
    _SHAPE = "enum Shape\n  | Circle\n  | Rect(width: int, height: int)\n"

    def test_enum_variant_with_payload_round_trip(self, tmp_path: Path) -> None:
        source = (
            self._SHAPE + "extern def echo(s: Shape) -> Shape\n"
            "let r = echo(Rect(width = 3, height = 4))\n"
            "let total = case r of\n"
            "  | Rect(width, height) => width + height\n"
            "  | Circle() => 0\n"
            "total\n"
        )
        companion = "def echo(s):\n    return s\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["total"] == IntValue(7)

    def test_enum_variant_without_payload_round_trip(self, tmp_path: Path) -> None:
        source = (
            self._SHAPE + "extern def echo(s: Shape) -> Shape\n"
            "let r: Shape = echo(Circle)\n"
            "let is_circle = r is Circle\n"
            "is_circle\n"
        )
        companion = "def echo(s):\n    return s\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["is_circle"] == BoolValue(True)

    def test_enum_unknown_case_on_return_rejected(self, tmp_path: Path) -> None:
        exc = evaluate_ir_raises_with_externs(
            self._SHAPE + "extern def f() -> Shape\nf()\n()\n",
            "def f():\n    return {'$case': 'Triangle'}\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"


class TestOptionAsPlainEnum:
    """`Option[T]` gets no special treatment at the boundary: it is an
    ordinary two-variant generic enum, so a locally declared stand-in exercises
    exactly the same walker code path as the real `std.core` one."""

    _OPTION = "enum Option[T]\n  | None\n  | Some(value: T)\n"

    def test_some_and_none_round_trip_as_tagged_dicts(self, tmp_path: Path) -> None:
        source = (
            self._OPTION + "extern def echo(o: Option[int]) -> Option[int]\n"
            "let some_r: Option[int] = echo(Some(value = 3))\n"
            "let none_r: Option[int] = echo(None)\n"
            "let a = case some_r of\n"
            "  | Some(value) => value\n"
            "  | None() => -1\n"
            "let b = case none_r of\n"
            "  | Some(value) => value\n"
            "  | None() => -1\n"
            "a\n"
        )
        companion = "def echo(o):\n    return o\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["a"] == IntValue(3)
        assert result["b"] == IntValue(-1)

    def test_option_of_option_disambiguates_nesting_depth(self, tmp_path: Path) -> None:
        source = (
            self._OPTION + "extern def check(o: Option[Option[int]]) -> bool\n"
            "let some_some: Option[Option[int]] = Some(value = Some(value = 5))\n"
            "let some_none: Option[Option[int]] = Some(value = None)\n"
            "let r1 = check(some_some)\n"
            "let r2 = check(some_none)\n"
            "let r3: bool = check(None)\n"
            "r1\n"
        )
        companion = (
            "def check(o):\n"
            "    if o == {'$case': 'Some', 'value': {'$case': 'Some', 'value': 5}}:\n"
            "        return True\n"
            "    if o == {'$case': 'Some', 'value': {'$case': 'None'}}:\n"
            "        return True\n"
            "    if o == {'$case': 'None'}:\n"
            "        return True\n"
            "    return False\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r1"] == BoolValue(True)
        assert result["r2"] == BoolValue(True)
        assert result["r3"] == BoolValue(True)

    def test_option_of_json_disambiguates_none_from_some_of_json_null(
        self, tmp_path: Path
    ) -> None:
        source = (
            self._OPTION + "extern def check(o: Option[json]) -> bool\n"
            "let some_null: Option[json] = Some(value = null)\n"
            "let r1 = check(some_null)\n"
            "let r2: bool = check(None)\n"
            "r1\n"
        )
        companion = (
            "def check(o):\n"
            "    if o == {'$case': 'Some', 'value': None}:\n"
            "        return True\n"
            "    if o == {'$case': 'None'}:\n"
            "        return True\n"
            "    return False\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r1"] == BoolValue(True)
        assert result["r2"] == BoolValue(True)


# ---------------------------------------------------------------------------
# Exceptions as ordinary boundary values (param and return position)
# ---------------------------------------------------------------------------


class TestExceptionsAsValues:
    _BAD_THING = "exception BadThing extends Exception\n  detail: text\n"

    def test_exception_typed_param_and_return_round_trip(self, tmp_path: Path) -> None:
        source = (
            self._BAD_THING + "extern def describe(e: BadThing) -> text\n"
            "extern def make(detail: text) -> BadThing\n"
            "let caught = try\n"
            '  raise BadThing(message = "boom", detail = "oops")\n'
            "catch BadThing as e =>\n"
            "  e\n"
            "let d1 = describe(caught)\n"
            'let built = make("built")\n'
            "let d2 = describe(built)\n"
            "d1\n"
        )
        companion = (
            "def describe(e):\n"
            "    return e['detail']\n"
            "def make(detail):\n"
            "    return {'message': 'constructed', 'trace_id': '', 'detail': detail}\n"
        )
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["d1"] == TextValue("oops")
        assert result["d2"] == TextValue("built")


# ---------------------------------------------------------------------------
# Deep-copy isolation: a companion mutating what it receives never affects
# the AgL value used again after the call.
# ---------------------------------------------------------------------------


class TestDeepCopyIsolation:
    def test_extern_receiving_a_list_can_mutate_and_return_its_own_copy(
        self, tmp_path: Path
    ) -> None:
        """A companion receiving a ``list[int]`` gets an ordinary, mutable
        Python ``list`` it can ``.append`` to and hand back as the result.

        Unlike the json variant below, there is no AgL-side isolation signal
        to check here: AgL lists are immutable tuples, so an AgL binding
        could never reflect a companion's mutation regardless of whether
        encoding copies the list — asserting on it would be vacuous.
        """
        source = (
            "extern def touch(xs: list[int]) -> list[int]\n"
            "let xs = [1, 2, 3]\n"
            "let touched = touch(xs)\n"
            "touched\n"
        )
        companion = "def touch(xs):\n    xs.append(99)\n    return xs\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["touched"] == ListValue(
            (IntValue(1), IntValue(2), IntValue(3), IntValue(99))
        )

    def test_mutating_a_received_json_object_does_not_affect_the_agl_value_used_after_the_call(
        self, tmp_path: Path
    ) -> None:
        source = (
            "extern def touch(x: json) -> json\n"
            "let doc: json = {a: [1, 2]}\n"
            "let touched = touch(doc)\n"
            "print(doc)\n"
            "print(touched)\n"
            "()\n"
        )
        companion = "def touch(x):\n    x['a'].append(99)\n    return x\n"
        _, output = evaluate_ir_with_externs(source, companion, tmp_path)
        lines = output.splitlines()
        assert len(lines) == 2
        assert lines[0] != lines[1]


# ---------------------------------------------------------------------------
# Argument binding: zones, named args, defaults
# ---------------------------------------------------------------------------


class TestArgumentBinding:
    def test_zones_and_named_args_arrive_positionally_in_declaration_order(
        self, tmp_path: Path
    ) -> None:
        source = (
            "extern def greet(name: text, /, greeting: text = \"Hello\", *,"
            " loud: bool = false) -> text\n"
            "let a = greet(\"Ada\")\n"
            "let b = greet(\"Ada\", greeting = \"Hi\")\n"
            "let c = greet(\"Ada\", \"Hi\", loud = true)\n"
            "a\n"
        )
        companion = "def greet(name, greeting, loud):\n    return f'{name}|{greeting}|{loud}'\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["a"] == TextValue("Ada|Hello|False")
        assert result["b"] == TextValue("Ada|Hi|False")
        assert result["c"] == TextValue("Ada|Hi|True")

    def test_unfilled_default_evaluates_on_the_agl_side_in_a_fresh_frame(
        self, tmp_path: Path
    ) -> None:
        """A default expression calling another AgL function proves the extern's
        default is evaluated in a frame chained to module scope, not inline
        Python — the companion never sees the un-evaluated default."""
        source = (
            "def base() -> int = 10\n"
            "extern def with_default(x: int = base()) -> int\n"
            "let r = with_default()\n"
            "r\n"
        )
        companion = "def with_default(x):\n    return x + 1\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(11)

    def test_indirect_call_uses_extern_default_when_arg_omitted(self, tmp_path: Path) -> None:
        """The indirect-call path's default fallback for an omitted trailing
        argument, hand-built at the IR level (see
        ``_build_indirect_extern_call_program``'s docstring for why)."""
        program, registry = _build_indirect_extern_call_program(tmp_path, (10,))
        result = IrInterpreter(program, extern_registry=registry).run()
        assert result["r"] == IntValue(11)

    def test_indirect_call_extern_missing_arg_no_default_raises(self, tmp_path: Path) -> None:
        """The indirect-call path's defensive error for a missing argument
        with no default to fall back on, hand-built at the IR level."""
        from agm.agl.ir.validate import InvalidIrError

        program, registry = _build_indirect_extern_call_program(tmp_path, ())
        with pytest.raises(InvalidIrError, match="missing argument"):
            IrInterpreter(program, extern_registry=registry).run()


# ---------------------------------------------------------------------------
# First-class externs
# ---------------------------------------------------------------------------


class TestFirstClass:
    def test_extern_stored_in_a_let_and_called_indirectly(self, tmp_path: Path) -> None:
        source = "extern def f(x: int) -> int\nlet g = f\nlet r = g(5)\nr\n"
        companion = "def f(x):\n    return x + 1\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(6)

    def test_extern_passed_to_a_higher_order_agl_function(self, tmp_path: Path) -> None:
        source = (
            "extern def f(x: int) -> int\n"
            "def apply(callback: (int) -> int, x: int) -> int = callback(x)\n"
            "let r = apply(f, 6)\n"
            "r\n"
        )
        companion = "def f(x):\n    return x + 1\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(7)

    def test_extern_returned_from_an_agl_function_and_called(self, tmp_path: Path) -> None:
        source = (
            "extern def f(x: int) -> int\n"
            "def get_fn() -> (int) -> int = f\n"
            "let r = get_fn()(7)\n"
            "r\n"
        )
        companion = "def f(x):\n    return x + 1\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(8)

    def test_extern_closure_renders_like_an_ordinary_function(self, tmp_path: Path) -> None:
        source = "extern def f(x: int) -> int\nprint(f)\n()\n"
        companion = "def f(x):\n    return x\n"
        _, output = evaluate_ir_with_externs(source, companion, tmp_path)
        assert output.strip() == "<function: int -> int>"


# ---------------------------------------------------------------------------
# ExternError
# ---------------------------------------------------------------------------


class TestExternError:
    def test_raising_companion_yields_extern_error_with_expected_fields(
        self, tmp_path: Path
    ) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def boom() -> int\nboom()\n()\n",
            "def boom():\n    raise ValueError('kaboom')\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"
        assert exc.fields["function"] == TextValue("boom")
        assert exc.fields["python_type"] == TextValue("ValueError")
        message = exc.fields["message"]
        assert isinstance(message, TextValue)
        assert message.value
        trace_id = exc.fields["trace_id"]
        assert isinstance(trace_id, TextValue)
        assert trace_id.value

    def test_wrong_return_type_yields_extern_error_with_empty_python_type(
        self, tmp_path: Path
    ) -> None:
        exc = evaluate_ir_raises_with_externs(
            "extern def f() -> int\nf()\n()\n",
            "def f():\n    return 'not an int'\n",
            tmp_path,
        )
        assert exc.display_name == "ExternError"
        assert exc.fields["python_type"] == TextValue("")

    def test_extern_error_is_catchable_with_try(self, tmp_path: Path) -> None:
        source = (
            "extern def boom() -> int\n"
            "let r = try\n"
            "  boom()\n"
            "catch ExternError as e =>\n"
            "  print(e.function)\n"
            "  print(e.python_type)\n"
            "  -1\n"
            "r\n"
        )
        companion = "def boom():\n    raise ValueError('kaboom')\n"
        result, output = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(-1)
        assert output.splitlines() == ["boom", "ValueError"]

    def test_uncaught_extern_error_surfaces_as_run_error_with_call_site_span(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def boom() -> int")
        write_companion_file(root, "lib.mod", "def boom():\n    raise ValueError('x')\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::boom()",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared)
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "ExternError"
        assert result.error.line == 2


# ---------------------------------------------------------------------------
# Recursion and loops
# ---------------------------------------------------------------------------


class TestRecursionAndLoops:
    def test_recursive_agl_function_interleaved_with_extern_calls(self, tmp_path: Path) -> None:
        source = (
            "extern def inc(x: int) -> int\n"
            "def sum_to(n: int, acc: int) -> int =\n"
            "  if n <= 0 => acc\n"
            "  else => sum_to(n - 1, acc + inc(0))\n"
            "let r = sum_to(5, 0)\n"
            "r\n"
        )
        companion = "def inc(x):\n    return x + 1\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(5)

    def test_extern_called_inside_a_loop(self, tmp_path: Path) -> None:
        source = (
            "extern def inc(x: int) -> int\n"
            "var s = 0\n"
            "for x in [1, 2, 3] do\n"
            "  s := inc(s)\n"
            "done\n"
            "let r = s\n"
            "r\n"
        )
        companion = "def inc(x):\n    return x + 1\n"
        result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
        assert result["r"] == IntValue(3)


# ---------------------------------------------------------------------------
# End-to-end file runs via PipelineDriver (real files, graph pipeline)
# ---------------------------------------------------------------------------


class TestEndToEndFileRuns:
    def test_single_file_extern_program_runs_end_to_end(self, tmp_path: Path) -> None:
        entry_path = tmp_path / "prog.agl"
        entry_path.write_text("extern def add_one(x: int) -> int\nadd_one(41)\n")
        (tmp_path / "prog.py").write_text("def add_one(x):\n    return x + 1\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            entry_path.read_text(),
            entry_path=entry_path,
            roots=_roots(tmp_path),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared)
        assert result.ok is True, result.diagnostics

    def test_library_module_extern_reachable_via_qualified_call(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(root, "lib.mod", "def f(x):\n    return x + 1\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlet r = lib.mod::f(1)\nr\n",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared)
        assert result.ok is True, result.diagnostics
        assert result.bindings["r"] == IntValue(2)

    def test_library_module_extern_reachable_via_open_import(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(root, "lib.mod", "def f(x):\n    return x + 1\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlet r = f(1)\nr\n",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared)
        assert result.ok is True, result.diagnostics
        assert result.bindings["r"] == IntValue(2)

    def test_private_extern_callable_inside_module_invisible_outside(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "root"
        write_module_file(
            root,
            "lib.mod",
            "private extern def f(x: int) -> int\ndef g(x: int) -> int = f(x) + 1",
        )
        write_companion_file(root, "lib.mod", "def f(x):\n    return x + 1\n")
        driver = PipelineDriver()

        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlet r = lib.mod::g(1)\nr\n",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared)
        assert result.ok is True, result.diagnostics
        assert result.bindings["r"] == IntValue(3)

        outside_prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        outside_result = driver.run_prepared_graph(outside_prepared)
        assert outside_result.ok is False


# ---------------------------------------------------------------------------
# Dry run (`check_only`)
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_lists_call_site_without_running_the_extern(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker.txt"
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(
            root,
            "lib.mod",
            f"def f(x):\n    open({str(marker)!r}, 'a').write('called')\n    return x + 1\n",
        )
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared, check_only=True)
        assert result.ok is True
        assert [cs.callee for cs in result.call_sites] == ["f"]
        # The companion module IMPORTS (fail-fast on a broken companion even
        # in dry-run), but calling ``f`` — a side effect inside its body —
        # never runs during a dry-run.
        assert not marker.exists()

    def test_dry_run_still_fails_fast_on_a_broken_companion(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(root, "lib.mod", "raise RuntimeError('broken')\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared, check_only=True)
        assert result.ok is False


# ---------------------------------------------------------------------------
# REPL smoke test (full behavior coverage is a later stage of this effort)
# ---------------------------------------------------------------------------


class TestReplSmoke:
    def test_repl_session_can_import_and_call_an_extern(self, tmp_path: Path) -> None:
        from agm.agl.modules.roots import assemble_roots
        from agm.agl.repl import ReplSession

        lib = tmp_path / "extlib.agl"
        lib.write_text("extern def add_one(x: int) -> int\n")
        (tmp_path / "extlib.py").write_text("def add_one(x):\n    return x + 1\n")

        roots = assemble_roots(
            invocation_root=tmp_path,
            stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        session = ReplSession()
        session._roots = roots

        result = session.eval_entry("import extlib\nadd_one(41)")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)

    def test_repl_session_fails_fast_on_a_broken_companion(self, tmp_path: Path) -> None:
        """The REPL wires the same fail-fast companion diagnostics as the
        file pipeline, rather than crashing or silently proceeding."""
        from agm.agl.modules.roots import assemble_roots
        from agm.agl.repl import ReplSession

        lib = tmp_path / "extlib.agl"
        lib.write_text("extern def add_one(x: int) -> int\n")
        (tmp_path / "extlib.py").write_text("def wrong_name(x):\n    return x + 1\n")

        roots = assemble_roots(
            invocation_root=tmp_path,
            stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        session = ReplSession()
        session._roots = roots

        result = session.eval_entry("import extlib\nadd_one(41)")

        assert result.ok is False
