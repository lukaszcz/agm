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


def test_dry_run_omits_extern_calls_in_ambient_method_modules() -> None:
    prepared = PipelineDriver.prepare_program("program def main() -> unit = print([1].size())\n")
    discovery = PipelineDriver().discover_params(prepared)

    assert discovery.compiled is not None, discovery.diagnostics
    assert lower_program(discovery.compiled).dry_run_inventory == ()


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
