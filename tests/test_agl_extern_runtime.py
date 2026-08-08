"""End-to-end runtime behavior of the value-directed extern boundary."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver
from agm.agl.semantics.values import (
    UNIT_VALUE,
    BoolValue,
    DecimalValue,
    ExceptionValue,
    IntValue,
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


def test_native_scalars_round_trip_without_a_declared_schema(tmp_path: Path) -> None:
    source = (
        "extern def echo_text(value: text) -> text\n"
        "extern def increment(value: int) -> int\n"
        "extern def double(value: decimal) -> decimal\n"
        "extern def negate(value: bool) -> bool\n"
        "extern def touch() -> unit\n"
        'let a = echo_text("hi")\nlet b = increment(2)\nlet c = double(1.5)\n'
        "let d = negate(true)\nlet e = touch()\n()\n"
    )
    companion = (
        "def echo_text(value): return value + '!'\n"
        "def increment(value): return value + 1\n"
        "def double(value): return value * 2\n"
        "def negate(value): return not value\n"
        "def touch(): return None\n"
    )
    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)

    assert result["a"] == TextValue("hi!")
    assert result["b"] == IntValue(3)
    assert result["c"] == DecimalValue(Decimal("3.0"))
    assert result["d"] == BoolValue(False)
    assert result["e"] == UNIT_VALUE


def test_enum_nested_variant_classes_support_matching(tmp_path: Path) -> None:
    source = (
        "enum Shape\n  | Circle(radius: int)\n  | Square(side: int)\n"
        "extern def area(shape: Shape) -> int\n"
        "let result = area(Circle(radius = 3))\nresult\n"
    )
    companion = (
        "from agl import Shape\n"
        "def area(shape):\n"
        "    match shape:\n"
        "        case Shape.Circle(radius=r): return r * r\n"
        "        case Shape.Square(side=s): return s * s\n"
    )
    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
    assert result["result"] == IntValue(9)


def test_exception_values_cross_as_plain_nominal_objects(tmp_path: Path) -> None:
    source = (
        "exception Problem extends Exception\n  detail: text\n"
        "extern def add_detail(problem: Problem) -> Problem\n"
        'let result = add_detail(Problem(message = "m", detail = "x"))\n'
        "result\n"
    )
    companion = (
        "from agl import Problem\n"
        "def add_detail(problem):\n"
        "    return Problem(\n"
        "        message=problem.message,\n"
        "        detail=problem.detail + '!'\n"
        "    )\n"
    )
    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
    assert isinstance(result["result"], ExceptionValue)
    assert result["result"].fields["detail"] == TextValue("x!")


def test_companion_exceptions_still_become_extern_errors(tmp_path: Path) -> None:
    source = "extern def fail() -> int\nlet _ = fail()\n()\n"
    companion = "def fail(): raise RuntimeError('boom')\n"
    exc = evaluate_ir_raises_with_externs(source, companion, tmp_path)
    assert exc.fields["python_type"] == TextValue("RuntimeError")


def test_extern_defaults_work_for_direct_calls(tmp_path: Path) -> None:
    source = "extern def increment(value: int = 2) -> int\nlet direct = increment()\n()\n"
    result, _ = evaluate_ir_with_externs(
        source, "def increment(value): return value + 1\n", tmp_path
    )

    assert result["direct"] == IntValue(3)


def test_indirect_extern_default_and_missing_argument_guards(tmp_path: Path) -> None:
    from agm.agl.eval.ir_interpreter import IrInterpreter
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
    from agm.agl.ir.validate import InvalidIrError
    from agm.agl.modules.ids import ENTRY_ID
    from agm.agl.runtime.externs import ExternRegistry

    companion = tmp_path / "companion.py"
    companion.write_text("def increment(value, step): return value + step\n")
    source_id = SourceId(0)
    location = Location(source_id, 0, 1, 1, 0)
    function_id = FunctionId(1)
    function_symbol = SymbolId(1)
    closure_symbol = SymbolId(2)
    result_symbol = SymbolId(3)
    descriptor = FunctionDescriptor(
        function_id=function_id,
        function_symbol=function_symbol,
        module_id=ENTRY_ID,
        params=(
            IrFunctionParam(SymbolId(4), default=None),
            IrFunctionParam(SymbolId(5), default=IrConstInt(location, 1)),
        ),
        impl=ExternFunctionBody(name="increment"),
    )

    def program(arguments: tuple[int, ...]) -> ExecutableProgram:
        return ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={
                ENTRY_ID: ExecutableModule(
                    module_id=ENTRY_ID,
                    initializers=(
                        IrBind(location, function_symbol, IrMakeClosure(location, function_id, ())),
                        IrBind(location, closure_symbol, IrLoad(location, function_symbol)),
                        IrBind(
                            location,
                            result_symbol,
                            IrIndirectCall(
                                location,
                                IrLoad(location, closure_symbol),
                                tuple(IrConstInt(location, value) for value in arguments),
                            ),
                        ),
                    ),
                )
            },
            symbols={
                function_symbol: SymbolDescriptor(function_symbol, False, "increment", ENTRY_ID),
                closure_symbol: SymbolDescriptor(closure_symbol, False, None, ENTRY_ID),
                result_symbol: SymbolDescriptor(result_symbol, False, "result", ENTRY_ID),
            },
            nominals={},
            sources={source_id: SourceFile("<test>", "x")},
            functions={function_id: descriptor},
        )

    registry = ExternRegistry()
    registry.load_companion(ENTRY_ID, companion)
    assert IrInterpreter(program((10,)), extern_registry=registry).run()["result"] == IntValue(11)
    with pytest.raises(InvalidIrError, match="missing argument"):
        IrInterpreter(program(()), extern_registry=registry).run()


def test_dry_run_lists_call_site_without_running_the_extern(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    root = tmp_path / "root"
    write_module_file(root, "lib/mod", "extern def f(x: int) -> int")
    write_companion_file(
        root,
        "lib/mod",
        f"open({str(marker)!r}, 'a').write('imported')\n"
        "def f(x):\n"
        f"    open({str(marker)!r}, 'a').write('called')\n"
        "    return x + 1\n",
    )
    driver = PipelineDriver()
    prepared = PipelineDriver.prepare_program(
        "import lib/mod\nlib/mod::f(1)",
        entry_path=None,
        roots=_roots(root),
        default_stdlib=False,
    )
    result = driver.run_prepared(prepared, check_only=True)
    assert result.ok is True
    assert [cs.callee for cs in result.call_sites] == ["f"]
    assert not marker.exists()


def test_dry_run_does_not_import_a_broken_companion(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write_module_file(root, "lib/mod", "extern def f(x: int) -> int")
    write_companion_file(root, "lib/mod", "raise RuntimeError('broken')\n")
    driver = PipelineDriver()
    prepared = PipelineDriver.prepare_program(
        "import lib/mod\nlib/mod::f(1)",
        entry_path=None,
        roots=_roots(root),
        default_stdlib=False,
    )
    result = driver.run_prepared(prepared, check_only=True)
    assert result.ok is True
    assert [cs.callee for cs in result.call_sites] == ["f"]
