"""Selection-table coverage for methods on built-in receiver types."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.ir.nodes import IrCopyValue, IrDirectCall, IrPrint
from agm.agl.ir.program import IrFunctionBody
from agm.agl.lower.program import lower_program
from agm.agl.modules.ids import ENTRY_ID, STD_PRELUDE_ID, ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.runtime.arguments import ProgramArguments
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
from agm.packages.layout import MODULE_TREE_DIRNAME
from tests._agl_helpers import agl_std_package_roots


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

    assert table.method_candidates(receiver, "selected") == ((method,),)
    assert table.method_candidates(UnitType(), "selected") == ()

    merged = TypeTable()
    merged.merge_from(table)
    assert merged.method_candidates(receiver, "selected") == ((method,),)


def test_prelude_reexports_builtin_receiver_scopes_for_bare_routes() -> None:
    prepared = PipelineDriver.prepare_program(
        "program def main() -> unit =\n"
        '  print(text::trim(" value "))\n'
        "  print(array::size([1]))\n"
        "  print(int::abs(-1))\n"
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics


def test_default_prelude_exposes_receiver_methods() -> None:
    prepared = PipelineDriver.prepare_program(
        "program def main() -> unit =\n"
        "  print([1].size())\n"
        '  print(" value ".trim())\n'
        "  print((-1).abs())\n",
        roots=agl_std_package_roots(),
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics


def test_receiver_methods_require_an_import_without_the_prelude() -> None:
    unavailable = PipelineDriver.prepare_program(
        "def get-size() -> int = [1].size()\n", default_stdlib=False
    )

    rejected = PipelineDriver().discover_programs(unavailable)

    assert rejected.checked is None
    assert rejected.diagnostics

    available = PipelineDriver.prepare_program(
        "import std/array\ndef get-size() -> int = [1].size()\n",
        default_stdlib=False,
    )
    selected = PipelineDriver().discover_programs(available)

    assert selected.checked is not None, selected.diagnostics


def test_local_receiver_scope_coexists_with_the_prelude_reexport() -> None:
    prepared = PipelineDriver.prepare_program(
        "scope text\n"
        "  def shout(value: text) -> text = value.upper()\n"
        "end text\n"
        "\n"
        'program def main() -> unit = print(text::shout("value"))\n'
    )

    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics


@pytest.mark.parametrize("hidden", ("text", "text::trim"))
def test_prelude_hiding_receiver_scope_routes_resolves(hidden: str) -> None:
    prepared = PipelineDriver.prepare_program(
        f"import std/prelude::* hiding {hidden}\ndef value() -> unit = ()\n"
    )

    assert prepared.resolved is not None, prepared.diagnostics


def test_prelude_hiding_a_method_keeps_other_receiver_methods_visible() -> None:
    visible = PipelineDriver.prepare_program(
        'import std/prelude::* hiding text::trim\ndef value() -> int = "value".size()\n',
        roots=agl_std_package_roots(),
    )
    selected = PipelineDriver().discover_programs(visible)

    assert selected.checked is not None, selected.diagnostics

    hidden = PipelineDriver.prepare_program(
        'import std/prelude::* hiding text::trim\ndef value() -> text = " value ".trim()\n',
        roots=agl_std_package_roots(),
    )
    rejected = PipelineDriver().discover_programs(hidden)

    assert rejected.checked is None
    assert rejected.diagnostics


def test_builtin_direct_method_calls_lower_as_receiver_first_direct_calls() -> None:
    stdlib_root = Path(__file__).parent / "agl" / "program_modules" / "builtin_method_stdlib"
    prepared = PipelineDriver.prepare_program(
        "program def main() -> unit = print([1].size())\n",
        roots=RootSet(roots=frozenset(), stdlib_roots=frozenset({stdlib_root})),
    )
    discovery = PipelineDriver().discover_programs(prepared)

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
        roots=RootSet(roots=frozenset(), stdlib_roots=frozenset({stdlib_root})),
    )
    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.compiled is not None, discovery.diagnostics
    executable = lower_program(discovery.compiled)
    ((_, entry_function_id),) = executable.program_functions.items()
    entry = executable.functions[entry_function_id]
    assert isinstance(entry.impl, IrFunctionBody)
    assert isinstance(entry.impl.body, IrPrint)
    assert isinstance(entry.impl.body.value, IrCopyValue)


@pytest.mark.parametrize(
    ("receiver", "method", "argument", "receiver_call"),
    (
        ("array", "first", "[7]", "[8].first()"),
        ("dict", "value", '{"value": 7}', '{"value": 8}.value()'),
    ),
)
def test_applied_receiver_scope_is_reexportable_and_selectable_by_route(
    tmp_path: Path, receiver: str, method: str, argument: str, receiver_call: str
) -> None:
    stdlib_root = Path(__file__).parent / "agl" / "program_modules" / "builtin_method_stdlib"
    (tmp_path / "facade.agl").write_text(
        f"export std/{receiver}::{{{receiver}}}\n", encoding="utf-8"
    )
    prepared = PipelineDriver.prepare_program(
        f"import facade::{{{receiver}}}\n"
        "program def main() -> unit =\n"
        f"  print({receiver}::{method}({argument}))\n"
        f"  print({receiver_call})\n",
        roots=RootSet(roots=frozenset({tmp_path}), stdlib_roots=frozenset({stdlib_root})),
    )

    assert prepared.resolved is not None, prepared.diagnostics
    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.checked is not None, discovery.diagnostics


def test_prelude_method_modules_are_inferred_before_consumers() -> None:
    """Prelude scope exports make method modules ordinary dependencies."""
    stdlib_root = Path(__file__).parent / "agl" / "program_modules" / "builtin_method_stdlib"
    prepared = PipelineDriver.prepare_program(
        "def generic-first[E](values: array[E]) = values.first()\n"
        "program def main() = print(generic-first([9]))\n",
        roots=RootSet(roots=frozenset(), stdlib_roots=frozenset({stdlib_root})),
    )

    assert prepared.resolved is not None, prepared.diagnostics
    graph = prepared.resolved.graph
    component_index = {
        mid: index for index, component in enumerate(graph.sccs) for mid in component
    }
    assert ModuleId.from_path("std/array") in graph.adjacency[STD_PRELUDE_ID]
    assert component_index[ModuleId.from_path("std/array")] < component_index[ENTRY_ID]

    selected = PipelineDriver().discover_programs(prepared)

    assert selected.checked is not None, selected.diagnostics


def test_dry_run_attributes_ambient_method_externs_to_the_calling_module() -> None:
    """A builtin method backed by an extern is inventoried where the call is written.

    The module declaring the method contributes no call sites of its own,
    so the inventory describes the program's own source rather than the standard
    library's internals.
    """
    prepared = PipelineDriver.prepare_program("program def main() -> unit = print([1].size())\n")
    discovery = PipelineDriver().discover_programs(prepared)

    assert discovery.compiled is not None, discovery.diagnostics
    inventory = lower_program(discovery.compiled).dry_run_inventory
    assert [(entry.module, entry.callee) for entry in inventory] == [(ENTRY_ID, "size")]


def test_dry_run_keeps_method_modules_reached_through_source_imports(
    tmp_path: Path,
) -> None:
    """Source provenance reaches a method module through an intermediary module."""
    stdlib = tmp_path / "stdlib"
    std = stdlib / MODULE_TREE_DIRNAME
    std.mkdir(parents=True)
    (std / "prelude.agl").write_text(
        "export std/math::{int}\n\nbuiltin def print[T](value: T) -> unit\n"
    )
    (std / "math.agl").write_text(
        "extern def helper() -> int\ndef int::external(self) -> int = helper()\n"
    )
    (std / "math.py").write_text("def helper():\n    return 3\n")
    (tmp_path / "bridge.agl").write_text("import std/math\n")

    prepared = PipelineDriver.prepare_program(
        "import bridge\nprogram def main() -> unit = ()\n",
        roots=RootSet(roots=frozenset({tmp_path}), stdlib_roots=frozenset({stdlib})),
    )
    runtime = PipelineDriver()
    discovery = runtime.discover_programs(prepared)

    assert prepared.resolved is not None, prepared.diagnostics
    assert ModuleId.from_path("std/math") in prepared.resolved.graph.adjacency[STD_PRELUDE_ID]
    assert discovery.compiled is not None, discovery.diagnostics
    (program,) = discovery.programs
    preflight = runtime.preflight_arguments(
        prepared, program, ProgramArguments(positional=(), named={}), compiled=discovery.compiled
    )
    assert preflight.result.ok, preflight.result.diagnostics
    assert [site.callee for site in preflight.result.call_sites] == ["helper"]


def test_receiver_methods_are_available_but_owning_module_free_functions_are_not() -> None:
    stdlib_root = Path(__file__).parent / "agl" / "program_modules" / "builtin_method_stdlib"
    prepared = PipelineDriver.prepare_program(
        'program def main() -> unit = print("value".surround("[", "]"))\n',
        roots=RootSet(roots=frozenset(), stdlib_roots=frozenset({stdlib_root})),
    )

    selected = PipelineDriver().discover_programs(prepared)

    assert selected.checked is not None, selected.diagnostics

    unavailable = PipelineDriver.prepare_program(
        "program def main() -> unit = unavailable()\n",
        roots=RootSet(roots=frozenset(), stdlib_roots=frozenset({stdlib_root})),
    )
    rejected = PipelineDriver().discover_programs(unavailable)

    assert rejected.checked is None
    assert rejected.diagnostics
