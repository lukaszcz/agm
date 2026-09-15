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
from tests._agl_helpers import prepare_inline_command
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


def test_enum_referencing_a_scoped_record_keeps_the_record_companion_visible(
    tmp_path: Path,
) -> None:
    """Listing an existing record as a qualified enum member must not hide it.

    Only an inline member (declared bare inside its enum body) is a companion
    implementation detail; a qualified member keeps its own independent name
    path -- both its bare name and its scoped ``nominals`` path -- regardless
    of which enums also list it.
    """
    source = (
        "scope M\n"
        "  record Go(amount: int)\n"
        "end M\n"
        "\n"
        "enum Step\n"
        "  | Continue(amount: int)\n"
        "  | M::Go\n"
        "\n"
        "extern def make(n: int) -> M::Go\n"
        "extern def probe() -> bool\n"
        "let made = make(3)\n"
        "let seen = probe()\n"
        "made\n"
    )
    companion = (
        "from agl import Go, nominals\n"
        "def make(n): return Go(amount=n)\n"
        "def probe(): return nominals.entry.M.Go is Go\n"
    )
    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
    assert result["made"].fields == {"amount": IntValue(3)}
    assert result["seen"] == BoolValue(True)


def test_enum_member_classes_never_subclass_their_enum_class(tmp_path: Path) -> None:
    """An inline and a referenced member behave uniformly: neither subclasses
    its enum. The enum class is a pure namespace over each member's own
    record class.
    """
    source = (
        "scope M\n"
        "  record Go(amount: int)\n"
        "end M\n"
        "\n"
        "enum Step\n"
        "  | Continue(amount: int)\n"
        "  | M::Go\n"
        "\n"
        "extern def probe() -> bool\n"
        "let ok = probe()\n"
        "ok\n"
    )
    companion = (
        "from agl import Step\n"
        "def probe():\n"
        "    inline_ok = not issubclass(Step.Continue, Step)\n"
        "    referenced_ok = not issubclass(Step.Go, Step)\n"
        "    return inline_ok and referenced_ok\n"
    )
    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)
    assert result["ok"] == BoolValue(True)


def test_generic_inline_enum_member_is_not_a_direct_companion_import(tmp_path: Path) -> None:
    """A generic inline member remains private to its enum's namespace."""
    source = (
        "enum Box[T]\n"
        "  | Item(value: T)\n"
        "\n"
        "extern def probe() -> bool\n"
        "let visible = probe()\n"
        "visible\n"
    )
    companion = "import agl\ndef probe(): return hasattr(agl, 'Box') and not hasattr(agl, 'Item')\n"

    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)

    assert result["visible"] == BoolValue(True)


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
    assert exc.fields["python-type"] == "RuntimeError"
    assert exc.fields["function"] == "fail"


def test_an_extern_error_names_the_extern_as_declared(tmp_path: Path) -> None:
    source = '@extern-name("py_fail")\nextern def fail-it!() -> int\nlet _ = fail-it!()\n()\n'
    companion = "def py_fail(): raise RuntimeError('boom')\n"
    exc = evaluate_ir_raises_with_externs(source, companion, tmp_path)
    assert exc.fields["python-type"] == "RuntimeError"
    assert exc.fields["function"] == "fail-it!"


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
        impl=ExternFunctionBody(name="increment", companion_name="increment"),
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
    prepared = prepare_inline_command(
        "import lib/mod\nlib/mod::f(1)",
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
    prepared = prepare_inline_command(
        "import lib/mod\nlib/mod::f(1)",
        roots=_roots(root),
        default_stdlib=False,
    )
    result = driver.run_prepared(prepared, check_only=True)
    assert result.ok is True
    assert [cs.callee for cs in result.call_sites] == ["f"]


def test_an_attributed_extern_calls_the_companion_it_names(tmp_path: Path) -> None:
    source = (
        '@extern-name("first_option")\n'
        "extern def first?(xs: array[int]) -> Option[int]\n"
        '@extern-name("py_shout")\n'
        "extern def shout-it(value: text) -> text\n"
        "extern def plain(value: int) -> int\n"
        "let empty: array[int] = []\n"
        "let a = first?([7, 8]).unwrap-or(0)\n"
        "let b = first?(empty).unwrap-or(0)\n"
        'let c = shout-it("hi")\n'
        "let d = plain(2)\n"
        "()\n"
    )
    companion = (
        "from agl import nominals\n"
        "\n"
        "Option = nominals.std.option.Option\n"
        "\n"
        "def first_option(xs):\n"
        "    return Option.Some(value=xs[0]) if len(xs) else getattr(Option, 'None')()\n"
        "\n"
        "def py_shout(value):\n"
        "    return value.upper()\n"
        "\n"
        "def plain(value):\n"
        "    return value + 1\n"
    )
    result, _ = evaluate_ir_with_externs(source, companion, tmp_path)

    assert result["a"] == IntValue(7)
    assert result["b"] == IntValue(0)
    assert result["c"] == TextValue("HI")
    assert result["d"] == IntValue(3)
