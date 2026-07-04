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

Earlier suites (`test_agl_extern_loading.py`, `test_agl_extern_lowering.py`)
cover everything upstream of dispatch and stop before evaluation; this suite
is the first to actually invoke a companion callable.
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
    from agm.agl.ir.program import ExternFunctionDescriptor
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
    extern_desc = ExternFunctionDescriptor(
        function_id=fn_id,
        function_symbol=fn_sym,
        module_id=ENTRY_ID,
        name="inc",
        params=(
            IrFunctionParam(symbol=SymbolId(4), default=None),
            IrFunctionParam(symbol=SymbolId(5), default=IrConstInt(loc, 1)),
        ),
        contract=contract,
        companion_path=companion_path,
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
        functions={},
        externs={fn_id: extern_desc},
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
        assert output.strip() == "<function: (int) -> int>"


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
