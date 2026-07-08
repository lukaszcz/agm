"""Tests for IR lowering of `extern def`.

Covers everything from a checked program/graph to the linked
``ExecutableProgram`` for `extern def`:
- an extern lowers to an ``ExternFunctionDescriptor`` in ``program.externs``
  (never ``program.functions``), with a closure-initialization binding
  identical in shape to an ordinary function's.
- default expressions lower to ``IrExpr``s on the descriptor's params.
- direct and first-class (indirect) calls to an extern lower through the
  same machinery as calls to an ordinary function.
- whole-graph lowering keeps ids consistent between an extern-bearing
  library module and the entry module that calls it.
- the dry-run inventory carries a row per extern call site.
- ``validate_ir`` accepts the shared function-id space between
  ``program.functions``/``program.externs`` and rejects every way it can be
  broken (aliasing, dangling references, an inconsistent boundary contract).

NO interpreter dispatch is exercised here — evaluating a program that calls
an extern is out of scope until dispatch lands (a later stage of this
effort); the pipeline-level test below stops at ``check_only`` (static
passes, lowering, and dry-run inventory only, no evaluation).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir import (
    ExecutableModule,
    ExecutableProgram,
    FunctionDescriptor,
    FunctionId,
    IrBind,
    IrConstInt,
    IrDirectCall,
    IrFunctionParam,
    IrIndirectCall,
    IrMakeClosure,
    Location,
    NominalDescriptor,
    NominalId,
    NominalKind,
    SourceFile,
    SourceId,
    SymbolDescriptor,
    SymbolId,
    UseDefault,
)
from agm.agl.ir.contracts import (
    BoundaryRef,
    BoundaryScalar,
    BoundarySchema,
    BoundarySealVar,
    ExternContract,
    ExternParamSchema,
    ScalarKind,
)
from agm.agl.ir.program import ExternFunctionDescriptor
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.lower import lower_program
from agm.agl.lower.graph import lower_graph
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.pipeline import PipelineDriver
from agm.agl.scope import resolve
from agm.agl.scope.graph import resolve_graph
from agm.agl.typecheck import check
from agm.agl.typecheck.graph import check_graph
from tests.agl.ir_harness import make_graph_from_files, write_companion_file, write_module_file

# ---------------------------------------------------------------------------
# Full-pipeline helpers (parse -> resolve -> check -> lower)
# ---------------------------------------------------------------------------

_PATH = Path("/virtual/extern_lowering.agl")

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    supports_extern=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset(
            {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
        ),
    },
)


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset(paths))


def _lower_source(
    source: str,
    *,
    validate: bool = True,
) -> ExecutableProgram:
    """Parse + resolve (file-backed, so `extern def` is allowed) + check + lower."""
    resolved = resolve(parse_program(source), origin_path=_PATH)
    checked = check(resolved, _CAPS)
    return lower_program(
        checked,
        source_text=source,
        source_label="<extern-lowering-test>",
        validate=validate,
    )


def _only_extern(executable: ExecutableProgram) -> ExternFunctionDescriptor:
    assert len(executable.externs) == 1
    return next(iter(executable.externs.values()))


# ---------------------------------------------------------------------------
# Single-module descriptor shape
# ---------------------------------------------------------------------------


class TestExternDescriptor:
    def test_extern_produces_a_descriptor_not_a_function(self) -> None:
        executable = _lower_source("extern def f(x: int) -> int\nf(1)")
        assert executable.functions == {}
        desc = _only_extern(executable)
        assert desc.name == "f"
        assert desc.result_label == "int"
        assert desc.param_labels == ("int",)
        assert len(desc.params) == 1

    def test_contract_matches_the_declared_signature(self) -> None:
        executable = _lower_source("extern def f(x: int) -> int\nf(1)")
        desc = _only_extern(executable)
        assert desc.contract.params == (
            ExternParamSchema(label="int", schema=BoundaryScalar(ScalarKind.INT)),
        )
        assert desc.contract.result == BoundaryScalar(ScalarKind.INT)
        assert desc.contract.result_label == "int"

    def test_private_extern_is_unexported_but_lowers_the_same(self) -> None:
        executable = _lower_source("private extern def f(x: int) -> int\nf(1)")
        desc = _only_extern(executable)
        assert executable.symbols[desc.function_symbol].public_name is None

    def test_default_lowers_to_an_ir_expr_on_the_descriptors_param(self) -> None:
        executable = _lower_source("extern def f(a: int, b: int = 3) -> int\nf(1)")
        desc = _only_extern(executable)
        assert len(desc.params) == 2
        assert desc.params[0].default is None
        assert isinstance(desc.params[1].default, IrConstInt)
        assert desc.params[1].default.value == 3


# ---------------------------------------------------------------------------
# Calls: direct and first-class
# ---------------------------------------------------------------------------


class TestExternCalls:
    def test_direct_call_lowers_to_ir_direct_call_with_the_externs_function_id(self) -> None:
        executable = _lower_source("extern def f(x: int) -> int\nf(1)")
        desc = _only_extern(executable)
        inits = executable.modules[executable.entry_module].initializers
        call = inits[-1]
        assert isinstance(call, IrDirectCall)
        assert call.function_id == desc.function_id

    def test_first_class_reference_lowers_through_load_and_indirect_call(self) -> None:
        executable = _lower_source(
            "extern def f(x: int) -> int\nlet fn_ref = f\nlet result = fn_ref(5)\n()"
        )
        inits = executable.modules[executable.entry_module].initializers
        assert isinstance(inits[0], IrBind)
        assert isinstance(inits[0].value, IrMakeClosure)
        result_bind = next(
            item
            for item in inits
            if isinstance(item, IrBind) and isinstance(item.value, IrIndirectCall)
        )
        assert isinstance(result_bind.value, IrIndirectCall)


# ---------------------------------------------------------------------------
# Whole-graph lowering
# ---------------------------------------------------------------------------


class TestGraphLowering:
    def test_extern_bearing_library_and_importing_entry_have_consistent_ids(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "root"
        write_companion_file(root, "lib.mod", "def f(x):\n    return x + 1\n")
        graph = make_graph_from_files(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib.mod": "extern def f(x: int) -> int",
            },
        )
        checked = check_graph(resolve_graph(graph), _CAPS)
        executable = lower_graph(checked, validate=True)

        lib_mid = ModuleId.from_dotted("lib.mod")
        desc = _only_extern(executable)
        assert desc.module_id == lib_mid

        entry_inits = executable.modules[executable.entry_module].initializers
        call = next(node for node in entry_inits if isinstance(node, IrDirectCall))
        assert call.function_id == desc.function_id


# ---------------------------------------------------------------------------
# Dry-run inventory
# ---------------------------------------------------------------------------


class TestDryRunInventory:
    def test_lowered_program_dry_run_inventory_has_an_extern_row(self) -> None:
        executable = _lower_source("extern def f(x: int) -> int\nf(1)")
        assert len(executable.dry_run_inventory) == 1
        entry = executable.dry_run_inventory[0]
        assert entry.callee == "f"
        assert entry.codec_name == "extern"
        assert entry.target_type_label == "int"

    def test_pipeline_check_only_lists_the_extern_call_site(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(root, "lib.mod", "def f(x):\n    return x + 1\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared, check_only=True)
        assert result.ok is True
        assert len(result.call_sites) == 1
        assert result.call_sites[0].callee == "f"
        assert result.call_sites[0].codec_name == "extern"


# ---------------------------------------------------------------------------
# Validator: hand-built minimal programs (following test_agl_ir_validate.py style)
# ---------------------------------------------------------------------------

MOD_A = ModuleId.from_dotted("mod_a")
SID0 = SourceId(value=0)
FN_EXT = FunctionId(value=0)
SYM_EXT_FN = SymbolId(value=0)
SYM_EXT_PARAM = SymbolId(value=1)
NOM0 = NominalId(module_id=MOD_A, declared_name="Foo")

_SOURCE_TEXT = "let x = 1"  # 9 characters

LOC = Location(source_id=SID0, start_offset=0, end_offset=5, start_line=1, start_col=0)


def _extern_contract(
    *,
    type_params: tuple[str, ...] = (),
    n_params: int = 1,
    result: "BoundarySchema | None" = None,
    defs: "tuple[tuple[str, BoundarySchema], ...]" = (),
) -> ExternContract:
    return ExternContract(
        params=tuple(
            ExternParamSchema(label="int", schema=BoundaryScalar(ScalarKind.INT))
            for _ in range(n_params)
        ),
        result=result if result is not None else BoundaryScalar(ScalarKind.INT),
        type_params=type_params,
        result_label="int",
        defs=defs,
    )


def _extern_desc(
    *,
    function_id: FunctionId = FN_EXT,
    function_symbol: SymbolId = SYM_EXT_FN,
    module_id: ModuleId = MOD_A,
    params: "tuple[IrFunctionParam, ...] | None" = None,
    contract: ExternContract | None = None,
) -> ExternFunctionDescriptor:
    return ExternFunctionDescriptor(
        function_id=function_id,
        function_symbol=function_symbol,
        module_id=module_id,
        name="f",
        params=(
            params
            if params is not None
            else (IrFunctionParam(symbol=SYM_EXT_PARAM, default=None),)
        ),
        contract=contract if contract is not None else _extern_contract(),
    )


def _default_symbols() -> dict[SymbolId, SymbolDescriptor]:
    return {
        SYM_EXT_FN: SymbolDescriptor(
            symbol_id=SYM_EXT_FN, mutable=False, public_name="f", owner=MOD_A
        ),
        SYM_EXT_PARAM: SymbolDescriptor(
            symbol_id=SYM_EXT_PARAM, mutable=False, public_name=None, owner=FN_EXT
        ),
    }


def _make_program(
    *,
    externs: "dict[FunctionId, ExternFunctionDescriptor] | None" = None,
    functions: "dict[FunctionId, FunctionDescriptor] | None" = None,
    symbols: "dict[SymbolId, SymbolDescriptor] | None" = None,
    initializers: tuple = (),
) -> ExecutableProgram:
    """Build a valid base program with one extern; callers override individual tables."""
    nom_desc = NominalDescriptor(nominal=NOM0, display_name="Foo", kind=NominalKind.RECORD)
    sf = SourceFile(display_name="main.agl", normalized_text=_SOURCE_TEXT)
    em = ExecutableModule(module_id=MOD_A, initializers=initializers)
    return ExecutableProgram(
        entry_module=MOD_A,
        modules={MOD_A: em},
        symbols=symbols if symbols is not None else _default_symbols(),
        nominals={NOM0: nom_desc},
        sources={SID0: sf},
        functions=functions or {},
        externs=externs if externs is not None else {FN_EXT: _extern_desc()},
    )


class TestValidatorAcceptsAWellFormedExternProgram:
    def test_baseline_program_is_valid(self) -> None:
        validate_ir(_make_program())

    def test_valid_direct_call_to_the_extern_is_accepted(self) -> None:
        call = IrDirectCall(
            location=LOC, function_id=FN_EXT, arguments=(IrConstInt(location=LOC, value=1),)
        )
        validate_ir(_make_program(initializers=(call,)))

    def test_valid_closure_reference_to_the_extern_is_accepted(self) -> None:
        bind = IrBind(
            location=LOC,
            symbol=SYM_EXT_FN,
            value=IrMakeClosure(location=LOC, function_id=FN_EXT, captures=()),
        )
        validate_ir(_make_program(initializers=(bind,)))

    def test_use_default_for_a_defaulted_extern_param_is_accepted(self) -> None:
        """A direct call using UseDefault at a param with a default passes."""
        defaulted_param = IrFunctionParam(
            symbol=SYM_EXT_PARAM, default=IrConstInt(location=LOC, value=42)
        )
        extern = _extern_desc(params=(defaulted_param,))
        call = IrDirectCall(
            location=LOC, function_id=FN_EXT, arguments=(UseDefault(param_index=0),)
        )
        validate_ir(_make_program(externs={FN_EXT: extern}, initializers=(call,)))

    def test_every_boundary_schema_shape_is_walked_without_error(self) -> None:
        """One extern whose signature exercises every BoundarySchema variant.

        Covers list/dict/record/enum/exception/unit/seal-var recursion in a
        single lowering, through the real checker-types-to-contract compiler.
        """
        source = (
            "record Box\n"
            "  value: int\n"
            "enum Shape\n"
            "  | circle(radius: decimal)\n"
            "exception BadThing extends Exception\n"
            "  detail: text\n"
            "extern def f[T](a: Box, b: list[dict[text, int]], c: unit, d: T, e: Shape)"
            " -> BadThing\n"
            "0"
        )
        executable = _lower_source(source)
        assert len(executable.externs) == 1


class TestValidatorNegatives:
    def test_function_id_in_both_tables_is_rejected(self) -> None:
        fn_desc = FunctionDescriptor(
            function_id=FN_EXT,
            function_symbol=SYM_EXT_FN,
            module_id=MOD_A,
            params=(),
            body=IrConstInt(location=LOC, value=1),
        )
        program = _make_program(functions={FN_EXT: fn_desc})
        with pytest.raises(InvalidIrError, match="both"):
            validate_ir(program)

    def test_extern_function_id_key_mismatch_is_rejected(self) -> None:
        bad = _extern_desc(function_id=FunctionId(value=1))
        program = _make_program(externs={FN_EXT: bad})
        with pytest.raises(InvalidIrError, match="mismatch"):
            validate_ir(program)

    def test_extern_function_symbol_must_be_registered(self) -> None:
        bad = _extern_desc(function_symbol=SymbolId(value=999))
        with pytest.raises(InvalidIrError, match="function_symbol"):
            validate_ir(_make_program(externs={FN_EXT: bad}))

    def test_extern_module_id_must_be_registered(self) -> None:
        bad = _extern_desc(module_id=ModuleId.from_dotted("nope"))
        with pytest.raises(InvalidIrError, match="module_id"):
            validate_ir(_make_program(externs={FN_EXT: bad}))

    def test_extern_param_symbol_must_be_registered(self) -> None:
        bad_param = IrFunctionParam(symbol=SymbolId(value=999), default=None)
        bad = _extern_desc(params=(bad_param,))
        with pytest.raises(InvalidIrError, match="param symbol"):
            validate_ir(_make_program(externs={FN_EXT: bad}))

    def test_extern_contract_param_count_must_match_ir_params(self) -> None:
        bad = _extern_desc(contract=_extern_contract(n_params=2))
        with pytest.raises(InvalidIrError, match="boundary params"):
            validate_ir(_make_program(externs={FN_EXT: bad}))

    def test_seal_var_not_declared_in_type_params_is_rejected(self) -> None:
        bad = _extern_desc(contract=_extern_contract(result=BoundarySealVar("T")))
        with pytest.raises(InvalidIrError, match="T"):
            validate_ir(_make_program(externs={FN_EXT: bad}))

    def test_boundaryref_to_unknown_defs_key_is_rejected(self) -> None:
        bad = _extern_desc(contract=_extern_contract(result=BoundaryRef("missing")))
        with pytest.raises(InvalidIrError, match="unknown defs key"):
            validate_ir(_make_program(externs={FN_EXT: bad}))

    def test_boundaryref_cycle_is_rejected(self) -> None:
        # A defs key that only refs itself never reaches a body.
        contract = _extern_contract(
            result=BoundaryRef("a"), defs=(("a", BoundaryRef("a")),)
        )
        with pytest.raises(InvalidIrError, match="cycle"):
            validate_ir(_make_program(externs={FN_EXT: _extern_desc(contract=contract)}))

    def test_duplicate_defs_key_is_rejected(self) -> None:
        contract = _extern_contract(
            defs=(
                ("a", BoundaryScalar(ScalarKind.INT)),
                ("a", BoundaryScalar(ScalarKind.INT)),
            )
        )
        with pytest.raises(InvalidIrError, match="duplicate"):
            validate_ir(_make_program(externs={FN_EXT: _extern_desc(contract=contract)}))

    def test_direct_call_arg_count_mismatch_against_extern_is_rejected(self) -> None:
        call = IrDirectCall(location=LOC, function_id=FN_EXT, arguments=())
        with pytest.raises(InvalidIrError, match="arguments"):
            validate_ir(_make_program(initializers=(call,)))

    def test_use_default_at_a_no_default_extern_param_is_rejected(self) -> None:
        # The default extern descriptor's only param has no default.
        call = IrDirectCall(
            location=LOC, function_id=FN_EXT, arguments=(UseDefault(param_index=0),)
        )
        with pytest.raises(InvalidIrError, match="no default"):
            validate_ir(_make_program(initializers=(call,)))

    def test_direct_call_to_unknown_function_id_is_rejected(self) -> None:
        call = IrDirectCall(location=LOC, function_id=FunctionId(value=999), arguments=())
        with pytest.raises(InvalidIrError, match="not in program.functions or program.externs"):
            validate_ir(_make_program(initializers=(call,)))

    def test_closure_reference_to_unknown_function_id_is_rejected(self) -> None:
        bind = IrBind(
            location=LOC,
            symbol=SYM_EXT_FN,
            value=IrMakeClosure(location=LOC, function_id=FunctionId(value=999), captures=()),
        )
        with pytest.raises(InvalidIrError, match="not in program.functions or program.externs"):
            validate_ir(_make_program(initializers=(bind,)))

    def test_symbol_owned_by_unknown_function_id_is_rejected(self) -> None:
        symbols = _default_symbols()
        symbols[SymbolId(value=2)] = SymbolDescriptor(
            symbol_id=SymbolId(value=2),
            mutable=False,
            public_name=None,
            owner=FunctionId(value=999),
        )
        with pytest.raises(InvalidIrError, match="not in program.functions or program.externs"):
            validate_ir(_make_program(symbols=symbols))
