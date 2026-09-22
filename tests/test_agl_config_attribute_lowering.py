"""Lowering and preflight evaluation of the ``@config`` program attribute.

Builds on the static semantics covered by ``test_agl_config_attribute_static.py``:
once a ``program def``'s ``@config`` entries are checked, the checker
publishes their resolved targets, lowering carries them into the executable
keyed by the owning program's own symbol, and preflight evaluates the
selected program's own entries into runtime values for a host to merge into
its own precedence rules.
"""

from __future__ import annotations

import decimal
from pathlib import Path

import pytest

from agm.agl.ir.nodes import IrCoerce, IrConstBool, IrConstInt, IrExpr
from agm.agl.ir.static_keys import StaticBindingKey
from agm.agl.lower.program import lower_program
from agm.agl.matchcompile import MatchCompiledProgram, compile_program_matches
from agm.agl.modules.ids import STD_CONFIG_ID
from agm.agl.pipeline import PipelineDriver
from agm.agl.runtime.arguments import ProgramArguments
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.values import BoolValue, DecimalValue, IntValue, JsonValue
from agm.agl.syntax.nodes import FuncDef, static_items
from agm.agl.typecheck.env import CheckedModule
from agm.agl.typecheck.program import check_program
from tests._agl_helpers import agl_roots
from tests.agl.ir_harness import base_caps, make_file_graph_from_files
from tests.agl.module_graph import resolve_and_check_entry


def _config_targets(checked: CheckedModule, program_node_id: int) -> tuple[StaticBindingKey, ...]:
    """Return *program_node_id*'s ``@config`` entries' resolved targets, in source order."""
    return tuple(
        checked.program_config_targets[entry.key.node_id]
        for entry in checked.resolved.attributes.program_configs.get(program_node_id, ())
    )


def _program_node_id(checked: CheckedModule, name: str) -> int:
    """Return the node id of the ``program def`` named *name* in *checked*'s own module."""
    return next(
        item.node_id
        for item in static_items(checked.resolved.program.body.items)
        if isinstance(item, FuncDef) and item.is_program and item.name == name
    )


class TestProgramConfigTargetsPublishedByTheChecker:
    """The checker resolves each ``@config`` key to a ``StaticBindingKey`` target,
    published on the checked module for lowering to read."""

    def test_own_module_param_target(self) -> None:
        checked = resolve_and_check_entry(
            "@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = ()\n",
            base_caps(),
        )
        node_id = _program_node_id(checked, "main")
        (target,) = _config_targets(checked, node_id)
        assert target == (checked.module_id, (), "count")

    def test_scoped_param_target(self) -> None:
        checked = resolve_and_check_entry(
            "scope Logging\n"
            "  @param let level: int = 1\n"
            "end Logging\n\n"
            "@config(Logging::level = 2)\n"
            "program def main() -> unit = ()\n",
            base_caps(),
        )
        node_id = _program_node_id(checked, "main")
        (target,) = _config_targets(checked, node_id)
        assert target == (checked.module_id, ("Logging",), "level")

    def test_engine_setting_target(self) -> None:
        checked = resolve_and_check_entry(
            "import std/config\n\n@config(config::trace = true)\nprogram def main() -> unit = ()\n",
            base_caps(),
        )
        node_id = _program_node_id(checked, "main")
        (target,) = _config_targets(checked, node_id)
        assert target == (STD_CONFIG_ID, (), "trace")

    def test_cross_module_param_target(self, tmp_path: Path) -> None:
        graph = make_file_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import helper\n\n@config(helper::value = 2)\nprogram def main() -> unit = ()\n"
                ),
                "helper": "@param let value: int = 1\n",
            },
        )
        resolved_program = resolve_program(graph)
        checked = check_program(resolved_program, base_caps())
        entry_checked = checked.modules[graph.entry_id]
        helper_id = next(mid for mid in graph.modules if mid != graph.entry_id)
        node_id = _program_node_id(entry_checked, "main")
        (target,) = _config_targets(entry_checked, node_id)
        assert target == (helper_id, (), "value")


class TestLoweringIntoTheExecutable:
    """Lowering carries each ``@config`` entry into ``ExecutableProgram.program_configs``,
    keyed by the owning program's own symbol."""

    def test_program_configs_carries_target_and_lowered_value(self, tmp_path: Path) -> None:
        graph = make_file_graph_from_files(
            tmp_path,
            {
                "entry": (
                    "import std/config\n\n"
                    "@param let count: int = 1\n\n"
                    "@config(config::trace = true, count = 2)\n"
                    "program def main() -> unit = ()\n"
                ),
            },
        )
        resolved_program = resolve_program(graph)
        checked = check_program(resolved_program, base_caps())
        entry_checked = checked.modules[graph.entry_id]
        compiled = compile_program_matches(checked)
        assert isinstance(compiled.compiled, MatchCompiledProgram)
        executable = lower_program(compiled.compiled)

        node_id = _program_node_id(entry_checked, "main")
        symbol = executable.program_symbols[node_id]
        entries = dict(executable.program_configs[symbol])

        trace_target = (STD_CONFIG_ID, (), "trace")
        count_target = (graph.entry_id, (), "count")
        trace_value = entries[trace_target]
        count_value = entries[count_target]
        assert isinstance(trace_value, IrConstBool)
        assert trace_value.value is True
        assert isinstance(count_value, IrConstInt)
        assert count_value.value == 2

    def test_a_program_def_without_config_has_no_entry(self, tmp_path: Path) -> None:
        graph = make_file_graph_from_files(tmp_path, {"entry": "program def main() -> unit = ()\n"})
        resolved_program = resolve_program(graph)
        checked = check_program(resolved_program, base_caps())
        entry_checked = checked.modules[graph.entry_id]
        compiled = compile_program_matches(checked)
        assert isinstance(compiled.compiled, MatchCompiledProgram)
        executable = lower_program(compiled.compiled)
        node_id = _program_node_id(entry_checked, "main")
        symbol = executable.program_symbols[node_id]
        assert symbol not in executable.program_configs


class TestPreflightProgramConfig:
    """Preflight evaluates the selected program's own ``@config`` entries to values,
    coercing each to its target's declared type."""

    def test_decodes_an_engine_setting_value(self, tmp_path: Path) -> None:
        source = (
            "import std/config\n\n@config(config::trace = true)\nprogram def main() -> unit = ()\n"
        )
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        assert discovery.diagnostics == ()
        (program,) = discovery.programs

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        target = (STD_CONFIG_ID, (), "trace")
        assert preflight.program_config == {target: BoolValue(True)}

    def test_decodes_a_param_target_value(self, tmp_path: Path) -> None:
        source = (
            "@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = ()\n"
        )
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        assert discovery.diagnostics == ()
        (program,) = discovery.programs

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        target = (program.module, (), "count")
        assert preflight.program_config == {target: IntValue(2)}

    def test_decodes_an_int_literal_coerced_to_a_decimal_target(self, tmp_path: Path) -> None:
        source = (
            "@param let ratio: decimal = 1.0\n\n"
            "@config(ratio = 2)\n"
            "program def main() -> unit = ()\n"
        )
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        assert discovery.diagnostics == ()
        (program,) = discovery.programs

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        target = (program.module, (), "ratio")
        assert preflight.program_config == {target: DecimalValue(decimal.Decimal(2))}

    def test_decodes_an_int_literal_coerced_to_a_json_target(self, tmp_path: Path) -> None:
        source = "@param let blob: json = 0\n\n@config(blob = 3)\nprogram def main() -> unit = ()\n"
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        assert discovery.diagnostics == ()
        (program,) = discovery.programs

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        target = (program.module, (), "blob")
        assert preflight.program_config == {target: JsonValue(3)}

    def test_a_scoped_param_target_key_is_present_in_param_bindings(self, tmp_path: Path) -> None:
        """The ``@config`` target's ``StaticBindingKey`` matches the key a host
        finds in ``executable.param_bindings``, exactly as a host merge relies on."""
        source = (
            "scope Logging\n"
            "  @param let level: int = 1\n"
            "end Logging\n\n"
            "@config(Logging::level = 2)\n"
            "program def main() -> unit = ()\n"
        )
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        assert discovery.diagnostics == ()
        (program,) = discovery.programs

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        target = (program.module, ("Logging",), "level")
        assert preflight.program_config == {target: IntValue(2)}
        assert preflight.executable is not None
        assert target in preflight.executable.param_bindings

    def test_a_cross_module_param_target_key_is_present_in_param_bindings(
        self, tmp_path: Path
    ) -> None:
        """Same as the scoped case, for a target declared in an imported module."""
        (tmp_path / "helper.agl").write_text("@param let value: int = 1\n")
        source = "import helper\n\n@config(helper::value = 2)\nprogram def main() -> unit = ()\n"
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        assert discovery.diagnostics == ()
        (program,) = discovery.programs

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        assert preflight.executable is not None
        helper_id = next(mid for mid in preflight.executable.modules if mid.segments == ("helper",))
        target = (helper_id, (), "value")
        assert preflight.program_config == {target: IntValue(2)}
        assert target in preflight.executable.param_bindings

    def test_empty_when_the_program_carries_no_config(self, tmp_path: Path) -> None:
        source = "program def main() -> unit = ()\n"
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        (program,) = discovery.programs

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        assert preflight.program_config == {}

    def test_a_second_program_defs_config_does_not_leak(self, tmp_path: Path) -> None:
        source = (
            "@param let count: int = 1\n\n"
            "@config(count = 2)\n"
            "program def build() -> unit = ()\n\n"
            "@config(count = 3)\n"
            "program def alt() -> unit = ()\n"
        )
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        assert discovery.diagnostics == ()
        by_name = {program.name: program for program in discovery.programs}
        target = (by_name["build"].module, (), "count")

        build_preflight = runtime.preflight_arguments(
            prepared,
            by_name["build"],
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert build_preflight.result.ok
        assert build_preflight.program_config == {target: IntValue(2)}

        alt_preflight = runtime.preflight_arguments(
            prepared,
            by_name["alt"],
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert alt_preflight.result.ok
        assert alt_preflight.program_config == {target: IntValue(3)}


class TestConfigAttributeCacheInvalidation:
    """The lowered value for a ``@config`` target follows its type, not a stale
    lowered-module cache entry keyed only by module identity."""

    def test_a_cross_module_target_type_change_changes_the_lowered_value(
        self, tmp_path: Path
    ) -> None:
        """A cross-module target's declared type change (int -> decimal) changes
        the lowered ``@config`` value's own shape (a plain constant becomes a
        coercion), not a cached constant from the target's earlier type."""
        entry_source = (
            "import helper\n\n@config(helper::value = 2)\nprogram def main() -> unit = ()\n"
        )

        def _lower(helper_source: str) -> IrExpr:
            graph = make_file_graph_from_files(
                tmp_path, {"entry": entry_source, "helper": helper_source}
            )
            resolved_program = resolve_program(graph)
            checked = check_program(resolved_program, base_caps())
            entry_checked = checked.modules[graph.entry_id]
            compiled = compile_program_matches(checked)
            assert isinstance(compiled.compiled, MatchCompiledProgram)
            executable = lower_program(compiled.compiled)
            node_id = _program_node_id(entry_checked, "main")
            symbol = executable.program_symbols[node_id]
            helper_id = next(mid for mid in graph.modules if mid != graph.entry_id)
            target = (helper_id, (), "value")
            (value,) = (v for k, v in executable.program_configs[symbol] if k == target)
            return value

        int_value = _lower("@param let value: int = 1\n")
        assert isinstance(int_value, IrConstInt)

        decimal_value = _lower("@param let value: decimal = 1.0\n")
        assert isinstance(decimal_value, IrCoerce)

        int_value_again = _lower("@param let value: int = 1\n")
        assert isinstance(int_value_again, IrConstInt)


class TestProgramDefCalledAsAnOrdinaryFunction:
    """A ``program def``'s ``@config`` entries apply only to its own preflight and
    program-symbol selection, never to an ordinary call reaching the same function."""

    def test_config_is_ignored_when_called_as_a_function(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = (
            "@param let count: int = 1\n\n"
            "@config(count = 99)\n"
            "program def build() -> unit = print(count)\n\n"
            "def call_build() -> unit = build()\n\n"
            "program def main() -> unit = call_build()\n"
        )
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(source, roots=agl_roots(tmp_path))
        discovery = runtime.discover_programs(prepared)
        assert discovery.diagnostics == ()
        main = next(program for program in discovery.programs if program.name == "main")

        preflight = runtime.preflight_arguments(
            prepared,
            main,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        assert preflight.program_config == {}
        assert preflight.executable is not None

        result = runtime.run_prepared(
            prepared,
            compiled=discovery.compiled,
            executable=preflight.executable,
            program_symbol=preflight.executable.program_symbols[main.node_id],
            arguments=preflight.arguments,
        )
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == "1\n"
