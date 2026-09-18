"""Static module-parameter inventory exposed by program discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.ir.static_keys import StaticBindingKey
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import ArgumentPreflight, PipelineDriver, PreparedProgram, ProgramDiscovery
from agm.agl.runtime.arguments import ProgramArguments
from tests._agl_helpers import agl_roots


def _discover(source: str, tmp_path: Path) -> ProgramDiscovery:
    """Discover source with modules rooted at *tmp_path*."""
    prepared = PipelineDriver.prepare_program(
        source,
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )
    discovery = PipelineDriver().discover_programs(prepared)
    assert discovery.diagnostics == ()
    return discovery


def _prepared(source: str, tmp_path: Path) -> PreparedProgram:
    """Prepare source with modules rooted at *tmp_path*."""
    return PipelineDriver.prepare_program(
        source,
        roots=RootSet(roots=frozenset({tmp_path})),
        default_stdlib=False,
    )


def _preflight(
    runtime: PipelineDriver,
    prepared: PreparedProgram,
    discovery: ProgramDiscovery,
    *,
    param_values: dict[StaticBindingKey, object] | None = None,
    param_values_lower: dict[StaticBindingKey, object] | None = None,
) -> ArgumentPreflight:
    """Preflight the first discovered program with no value arguments."""
    return runtime.preflight_arguments(
        prepared,
        discovery.programs[0],
        ProgramArguments(positional=(), named={}),
        compiled=discovery.compiled,
        param_values=param_values,
        param_values_lower=param_values_lower,
    )


def test_discovery_inventories_module_params_in_declaration_order(tmp_path: Path) -> None:
    discovery = _discover(
        """
@param @opt-name("loud") @doc("Root verbosity") let verbose: bool = false
@param var retries: int = 3

scope Logging
  @param @doc("Trace logging") let trace: bool = false
  @param var level: int = 1
end Logging

program def main() -> unit = ()
""",
        tmp_path,
    )

    params = discovery.module_params[discovery.programs[0].module]

    assert [(param.name, param.cli.name, param.mutable, param.scope_path) for param in params] == [
        ("verbose", "loud", False, ()),
        ("retries", "retries", True, ()),
        ("trace", "trace", False, ("Logging",)),
        ("level", "level", True, ("Logging",)),
    ]
    assert params[0].doc == "Root verbosity"
    assert params[2].doc == "Trace logging"
    assert params[0].declaration_path == "<entry>::verbose"
    assert params[2].declaration_path == "<entry>::Logging::trace"
    assert params[0].key == (discovery.programs[0].module, (), "verbose")


def test_discovery_closure_and_params_follow_source_reachability_order(tmp_path: Path) -> None:
    (tmp_path / "first.agl").write_text("@param let first: int = 1\n", encoding="utf-8")
    (tmp_path / "second.agl").write_text(
        "import third\n@param let second: int = 2\n", encoding="utf-8"
    )
    (tmp_path / "third.agl").write_text("@param let third: int = 3\n", encoding="utf-8")

    discovery = _discover(
        """
import first
import second
@param let root: int = 0
program def main() -> unit = ()
""",
        tmp_path,
    )
    program = discovery.programs[0]

    assert program.closure == (
        program.module,
        ModuleId(("first",)),
        ModuleId(("second",)),
        ModuleId(("third",)),
    )
    assert [param.name for param in discovery.params_for(program)] == [
        "root",
        "first",
        "second",
        "third",
    ]


def test_imported_program_uses_its_own_source_reachable_closure(tmp_path: Path) -> None:
    (tmp_path / "library.agl").write_text(
        "import dependency\n@param let library: int = 1\nprogram def run() -> unit = ()\n",
        encoding="utf-8",
    )
    (tmp_path / "dependency.agl").write_text("@param let dependency: int = 2\n", encoding="utf-8")

    discovery = _discover(
        "import library\n@param let entry: int = 0\nprogram def main() -> unit = ()\n",
        tmp_path,
    )
    imported_program = next(program for program in discovery.programs if program.name == "run")

    assert imported_program.closure == (ModuleId(("library",)), ModuleId(("dependency",)))
    assert [param.name for param in discovery.params_for(imported_program)] == [
        "library",
        "dependency",
    ]
    assert discovery.module_params[ModuleId(("library",))][0].declaration_path == "library::library"


def test_discovery_has_empty_module_param_inventory_without_param_bindings(tmp_path: Path) -> None:
    (tmp_path / "library.agl").write_text("let value: int = 1\n", encoding="utf-8")

    discovery = _discover(
        "import library\nprogram def main() -> unit = ()\n",
        tmp_path,
    )

    assert all(not params for params in discovery.module_params.values())
    assert tuple(discovery.params_for(discovery.programs[0])) == ()


class TestParamSeeds:
    def test_executable_exposes_a_binding_and_decoder_for_every_inventory_key(
        self, tmp_path: Path
    ) -> None:
        runtime = PipelineDriver()
        prepared = _prepared(
            """
@param let root: int = 1
scope Tuning
  @param var retries: int = 2
end Tuning

program def main() -> unit = ()
""",
            tmp_path,
        )
        discovery = runtime.discover_programs(prepared)

        preflight = _preflight(runtime, prepared, discovery)

        assert preflight.result.ok
        assert preflight.executable is not None
        expected_keys = {param.key for param in discovery.params_for(discovery.programs[0])}
        assert set(preflight.executable.param_bindings) == expected_keys
        assert set(preflight.executable.param_decoders) == expected_keys
        assert set(preflight.executable.param_bindings.values()) <= set(
            preflight.executable.symbols
        )

    def test_preflight_decodes_all_param_values_and_reports_all_failures(
        self, tmp_path: Path
    ) -> None:
        runtime = PipelineDriver()
        prepared = _prepared(
            """
@param let number: int = 1
@param let enabled: bool = false
program def main() -> unit = ()
""",
            tmp_path,
        )
        discovery = runtime.discover_programs(prepared)
        number, enabled = discovery.params_for(discovery.programs[0])
        unknown = (ModuleId(("missing",)), (), "unknown")

        preflight = _preflight(
            runtime,
            prepared,
            discovery,
            param_values={number.key: "not-an-int", enabled.key: "not-a-bool", unknown: 1},
        )

        assert not preflight.result.ok
        assert preflight.executable is not None
        assert preflight.arguments == ()
        assert preflight.param_seeds == {}
        assert len(preflight.result.diagnostics) == 3

    def test_imported_param_decode_failure_points_to_its_declaration(self, tmp_path: Path) -> None:
        library = tmp_path / "library.agl"
        library.write_text("@param let retries: int = 1\n", encoding="utf-8")
        runtime = PipelineDriver()
        prepared = _prepared(
            "import library\nprogram def main() -> unit = ()\n",
            tmp_path,
        )
        discovery = runtime.discover_programs(prepared)
        retries = discovery.params_for(discovery.programs[0])[0]

        preflight = _preflight(
            runtime, prepared, discovery, param_values={retries.key: "not-an-int"}
        )

        assert not preflight.result.ok
        assert preflight.result.diagnostics[0].source_label == str(library)

    def test_seeds_override_defaults_and_keep_vars_writable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "library.agl").write_text(
            """
@param let imported: int = 1
""",
            encoding="utf-8",
        )
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            """
import library
@param let root: int = 2
@param var count: int = 3

scope Tuning
  @param let scoped: int = 4
end Tuning

program def main() -> unit =
  count := count + 1
  print library::imported
  print root
  print count
  print Tuning::scoped
""",
            roots=agl_roots(tmp_path),
        )
        discovery = runtime.discover_programs(prepared)
        params = {param.name: param for param in discovery.params_for(discovery.programs[0])}

        preflight = _preflight(
            runtime,
            prepared,
            discovery,
            param_values={
                params["imported"].key: 10,
                params["root"].key: 20,
                params["count"].key: 30,
                params["scoped"].key: 40,
            },
        )

        assert preflight.result.ok
        assert preflight.executable is not None
        assert set(preflight.param_seeds) == {param.key for param in params.values()}
        result = runtime.run_prepared(
            prepared,
            compiled=discovery.compiled,
            executable=preflight.executable,
            program_symbol=preflight.executable.program_symbols[discovery.programs[0].node_id],
            arguments=preflight.arguments,
            param_seeds=preflight.param_seeds,
        )

        assert result.ok
        assert capsys.readouterr().out == "10\n20\n31\n40\n"

    def test_unseeded_param_uses_its_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            """
@param let value: int = 7
program def main() -> unit = print value
""",
            roots=agl_roots(tmp_path),
        )
        discovery = runtime.discover_programs(prepared)

        result = runtime.run_prepared(
            prepared, compiled=discovery.compiled, select_default_program=True
        )

        assert result.ok
        assert capsys.readouterr().out == "7\n"

    def test_check_only_never_binds_preflighted_seeds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runtime = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            """
@param let value: int = 7
program def main() -> unit = print value
""",
            roots=agl_roots(tmp_path),
        )
        discovery = runtime.discover_programs(prepared)
        value = discovery.params_for(discovery.programs[0])[0]

        preflight = _preflight(
            runtime,
            prepared,
            discovery,
            param_values={value.key: 9},
        )
        assert preflight.result.ok
        assert preflight.executable is not None

        run_result = runtime.run_prepared(
            prepared,
            check_only=True,
            compiled=discovery.compiled,
            executable=preflight.executable,
            param_seeds=preflight.param_seeds,
        )
        check_result = runtime.check_prepared(prepared, compiled=discovery.compiled)

        assert run_result.ok
        assert run_result.bindings == {}
        assert check_result.ok
        assert check_result.bindings == {}
        assert capsys.readouterr().out == ""


class TestParamValueTierMerge:
    """``preflight_arguments`` merges two raw tiers (``param_values`` = supplied/
    program-route, ``param_values_lower`` = module-route) beneath the selected
    program's own ``@config`` param values: upper > @config > lower."""

    def test_lower_is_used_when_upper_and_config_are_silent(self, tmp_path: Path) -> None:
        from agm.agl.semantics.values import IntValue

        runtime = PipelineDriver()
        prepared = _prepared(
            "@param let count: int = 1\nprogram def main() -> unit = ()\n", tmp_path
        )
        discovery = runtime.discover_programs(prepared)
        (count,) = discovery.params_for(discovery.programs[0])

        preflight = _preflight(runtime, prepared, discovery, param_values_lower={count.key: 5})

        assert preflight.result.ok
        assert preflight.param_seeds == {count.key: IntValue(5)}

    def test_upper_overrides_lower(self, tmp_path: Path) -> None:
        from agm.agl.semantics.values import IntValue

        runtime = PipelineDriver()
        prepared = _prepared(
            "@param let count: int = 1\nprogram def main() -> unit = ()\n", tmp_path
        )
        discovery = runtime.discover_programs(prepared)
        (count,) = discovery.params_for(discovery.programs[0])

        preflight = _preflight(
            runtime,
            prepared,
            discovery,
            param_values={count.key: 7},
            param_values_lower={count.key: 5},
        )

        assert preflight.result.ok
        assert preflight.param_seeds == {count.key: IntValue(7)}

    def test_config_overrides_lower(self, tmp_path: Path) -> None:
        from agm.agl.semantics.values import IntValue

        runtime = PipelineDriver()
        prepared = _prepared(
            "@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = ()\n",
            tmp_path,
        )
        discovery = runtime.discover_programs(prepared)
        (count,) = discovery.params_for(discovery.programs[0])

        preflight = _preflight(runtime, prepared, discovery, param_values_lower={count.key: 5})

        assert preflight.result.ok
        assert preflight.param_seeds == {count.key: IntValue(2)}

    def test_upper_overrides_config(self, tmp_path: Path) -> None:
        from agm.agl.semantics.values import IntValue

        runtime = PipelineDriver()
        prepared = _prepared(
            "@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = ()\n",
            tmp_path,
        )
        discovery = runtime.discover_programs(prepared)
        (count,) = discovery.params_for(discovery.programs[0])

        preflight = _preflight(runtime, prepared, discovery, param_values={count.key: 7})

        assert preflight.result.ok
        assert preflight.param_seeds == {count.key: IntValue(7)}

    def test_a_lower_value_overridden_by_config_is_never_decoded(self, tmp_path: Path) -> None:
        """An invalid module-route raw value that ``@config`` overrides must not
        surface a decode diagnostic — exactly as a program-route override today."""
        from agm.agl.semantics.values import IntValue

        runtime = PipelineDriver()
        prepared = _prepared(
            "@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = ()\n",
            tmp_path,
        )
        discovery = runtime.discover_programs(prepared)
        (count,) = discovery.params_for(discovery.programs[0])

        preflight = _preflight(
            runtime, prepared, discovery, param_values_lower={count.key: "not-an-int"}
        )

        assert preflight.result.ok
        assert preflight.param_seeds == {count.key: IntValue(2)}

    def test_a_lower_value_overridden_by_upper_is_never_decoded(self, tmp_path: Path) -> None:
        from agm.agl.semantics.values import IntValue

        runtime = PipelineDriver()
        prepared = _prepared(
            "@param let count: int = 1\nprogram def main() -> unit = ()\n", tmp_path
        )
        discovery = runtime.discover_programs(prepared)
        (count,) = discovery.params_for(discovery.programs[0])

        preflight = _preflight(
            runtime,
            prepared,
            discovery,
            param_values={count.key: 7},
            param_values_lower={count.key: "not-an-int"},
        )

        assert preflight.result.ok
        assert preflight.param_seeds == {count.key: IntValue(7)}
