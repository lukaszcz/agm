"""Selection-table coverage for methods on built-in receiver types."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.ir.nodes import IrCopyValue, IrDirectCall, IrPrint
from agm.agl.ir.program import IrFunctionBody
from agm.agl.lower.program import lower_program
from agm.agl.modules.ids import ENTRY_ID, STD_BUILTIN_METHODS_ID, ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.semantics.type_table import MethodDef, TypeTable
from agm.agl.semantics.types import (
    ArrayType,
    BoolType,
    DecimalType,
    DictType,
    FunctionType,
    IntType,
    JsonType,
    TextType,
    Type,
    UnitType,
)


@pytest.mark.parametrize(
    ("constructor", "receiver"),
    (
        ("array", ArrayType(IntType())),
        ("dict", DictType(TextType())),
        ("text", TextType()),
        ("json", JsonType()),
        ("int", IntType()),
        ("decimal", DecimalType()),
        ("bool", BoolType()),
    ),
)
def test_builtin_method_table_selects_by_receiver_constructor(
    constructor: str, receiver: Type
) -> None:
    table = TypeTable()
    method = MethodDef(
        module_id=ModuleId(("std", "test")),
        scope_path=(constructor,),
        name="selected",
        decl_node_id=1,
        signature=FunctionType(params=(receiver,), result=IntType()),
        receiver_type_param_arity=1 if constructor in {"array", "dict"} else 0,
    )

    table.register_builtin_method(constructor, method)

    assert table.lookup_builtin_method(receiver, "selected") == method
    assert table.lookup_builtin_method(UnitType(), "selected") is None

    merged = TypeTable()
    merged.merge_from(table)
    assert merged.lookup_builtin_method(receiver, "selected") == method


def test_builtin_direct_method_calls_lower_as_receiver_first_direct_calls() -> None:
    stdlib_root = Path(__file__).parent / "agl" / "program_modules" / "builtin_method_stdlib"
    prepared = PipelineDriver.prepare_program(
        "program def main() -> unit = print([1].size())\n",
        roots=RootSet(roots=frozenset({stdlib_root})),
    )
    discovery = PipelineDriver().discover_params(prepared)

    assert discovery.compiled is not None, discovery.diagnostics
    executable = lower_program(discovery.compiled)
    ((_, entry_function_id),) = executable.program_functions.items()
    entry = executable.functions[entry_function_id]
    assert isinstance(entry.impl, IrFunctionBody)
    assert isinstance(entry.impl.body, IrPrint)
    assert isinstance(entry.impl.body.value, IrDirectCall)
    assert len(entry.impl.body.value.arguments) == 1


def test_builtin_receiver_host_method_reuses_its_core_lowering_route() -> None:
    stdlib_root = Path(__file__).parent / "agl" / "program_modules" / "builtin_method_stdlib"
    prepared = PipelineDriver.prepare_program(
        "program def main() -> unit = print([1].copy())\n",
        roots=RootSet(roots=frozenset({stdlib_root})),
    )
    discovery = PipelineDriver().discover_params(prepared)

    assert discovery.compiled is not None, discovery.diagnostics
    executable = lower_program(discovery.compiled)
    ((_, entry_function_id),) = executable.program_functions.items()
    entry = executable.functions[entry_function_id]
    assert isinstance(entry.impl, IrFunctionBody)
    assert isinstance(entry.impl.body, IrPrint)
    assert isinstance(entry.impl.body.value, IrCopyValue)


def test_ambient_builtin_methods_are_inferred_before_consumers_without_source_imports() -> None:
    """Ambient method modules are inference dependencies, not user imports."""
    stdlib_root = Path(__file__).parent / "agl" / "program_modules" / "builtin_method_stdlib"
    prepared = PipelineDriver.prepare_program(
        "def generic_first[E](values: array[E]) = values.first()\n"
        "program def main() = print(generic_first([9]))\n",
        roots=RootSet(roots=frozenset({stdlib_root})),
    )

    assert prepared.resolved is not None, prepared.diagnostics
    graph = prepared.resolved.graph
    assert STD_BUILTIN_METHODS_ID not in graph.adjacency[ENTRY_ID]
    inference_component_index = {
        mid: index for index, component in enumerate(graph.inference_sccs) for mid in component
    }
    assert all(
        inference_component_index[mid] < inference_component_index[ENTRY_ID]
        for mid in graph.ambient_modules
    )

    selected = PipelineDriver().discover_params(prepared)

    assert selected.checked is not None, selected.diagnostics


def test_dry_run_attributes_ambient_method_externs_to_the_calling_module() -> None:
    """A builtin method backed by an extern is inventoried where the call is written.

    The ambient module declaring the method contributes no call sites of its own,
    so the inventory describes the program's own source rather than the standard
    library's internals.
    """
    prepared = PipelineDriver.prepare_program("program def main() -> unit = print([1].size())\n")
    discovery = PipelineDriver().discover_params(prepared)

    assert discovery.compiled is not None, discovery.diagnostics
    inventory = lower_program(discovery.compiled).dry_run_inventory
    assert [(entry.module, entry.callee) for entry in inventory] == [(ENTRY_ID, "size")]


def test_dry_run_keeps_source_reachable_ambient_registry_modules() -> None:
    """An explicit import remains runtime-reachable even when the registry also loads it."""
    stdlib_root = Path(__file__).resolve().parents[1] / "stdlib"
    prepared = PipelineDriver.prepare_program(
        'import std/array::join\nprogram def main() -> unit = print(join(["a"], ","))\n',
        roots=RootSet(roots=frozenset({stdlib_root})),
    )
    discovery = PipelineDriver().discover_params(prepared)

    assert discovery.compiled is not None, discovery.diagnostics
    inventory = lower_program(discovery.compiled).dry_run_inventory
    assert any(entry.module == ModuleId.from_path("std/array") for entry in inventory)


def test_dry_run_keeps_registry_method_modules_reached_through_source_imports(
    tmp_path: Path,
) -> None:
    """Source provenance reaches a registry method module through an intermediary module."""
    stdlib = tmp_path / "stdlib"
    std = stdlib / "std"
    std.mkdir(parents=True)
    (std / "prelude.agl").write_text("builtin def print[T](value: T) -> unit\n")
    (std / "builtin-methods.agl").write_text("import std/math\n")
    (std / "math.agl").write_text(
        "param limit: int = 3\nextern def helper() -> int\n"
        "def int::external(self) -> int = helper()\n"
    )
    (std / "math.py").write_text("def helper():\n    return 3\n")
    (tmp_path / "bridge.agl").write_text("import std/math\n")

    prepared = PipelineDriver.prepare_program(
        "import bridge\nprogram def main() -> unit = ()\n",
        roots=RootSet(roots=frozenset({tmp_path, stdlib})),
    )
    runtime = PipelineDriver()
    discovery = runtime.discover_params(prepared)

    assert prepared.resolved is not None, prepared.diagnostics
    assert ModuleId.from_path("std/math") in prepared.resolved.graph.ambient_modules
    assert [param.name for param in discovery.params] == ["limit"]
    assert discovery.compiled is not None, discovery.diagnostics
    preflight = runtime.preflight_params(prepared, compiled=discovery.compiled)
    assert preflight.result.ok, preflight.result.diagnostics
    assert preflight.executable is not None
    assert [param.public_name for param in preflight.executable.params] == ["limit"]
    assert [site.callee for site in preflight.result.call_sites] == ["helper"]


def test_builtin_methods_are_ambient_but_owning_module_free_functions_are_not() -> None:
    stdlib_root = Path(__file__).parent / "agl" / "program_modules" / "builtin_method_stdlib"
    prepared = PipelineDriver.prepare_program(
        'program def main() -> unit = print("value".surround("[", "]"))\n',
        roots=RootSet(roots=frozenset({stdlib_root})),
    )

    selected = PipelineDriver().discover_params(prepared)

    assert selected.checked is not None, selected.diagnostics

    unavailable = PipelineDriver.prepare_program(
        "program def main() -> unit = unavailable()\n",
        roots=RootSet(roots=frozenset({stdlib_root})),
    )
    rejected = PipelineDriver().discover_params(unavailable)

    assert rejected.checked is None
    assert rejected.diagnostics
