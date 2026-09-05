"""Tests for a ``program def``'s own value-parameter host surface.

Covers ``PipelineDriver.discover_programs`` (typed parameter signatures, one
per program), ``PipelineDriver.preflight_arguments`` (binding and decoding
host-supplied arguments against a program's signature without executing it),
and ``PipelineDriver.run_prepared(arguments=...)`` (executing with the bound
arguments) — the only host surface that can supply a value into a running
program.
"""

from __future__ import annotations

import pytest

from agm.agl.pipeline import PipelineDriver, PreparedProgram
from agm.agl.runtime.arguments import ProgramArguments
from agm.agl.semantics.types import BoolType, IntType, TextType
from agm.agl.zones import ParamZone


def _prepared(source: str) -> PreparedProgram:
    return PipelineDriver.prepare_program(source)


class TestDiscoverPrograms:
    def test_reports_typed_parameters_for_every_program(self) -> None:
        source = (
            'program def main(@arg-pos count: int, @arg-std label: text = "x") -> unit = ()\n\n'
            "program def alt(verbose: bool = false) -> unit = ()\n"
        )
        runtime = PipelineDriver()
        discovery = runtime.discover_programs(_prepared(source))

        assert discovery.checked is not None
        assert discovery.compiled is not None
        assert not discovery.diagnostics

        by_name = {program.name: program for program in discovery.programs}
        main_params = by_name["main"].parameters
        assert [(p.name, p.kind, p.has_default) for p in main_params] == [
            ("count", ParamZone.POSITIONAL_ONLY, False),
            ("label", ParamZone.STANDARD, True),
        ]
        assert isinstance(main_params[0].type, IntType)
        assert isinstance(main_params[1].type, TextType)

        alt_params = by_name["alt"].parameters
        assert [(p.name, p.kind, p.has_default) for p in alt_params] == [
            ("verbose", ParamZone.NAMED_ONLY, True),
        ]
        assert isinstance(alt_params[0].type, BoolType)

    def test_reports_diagnostics_and_no_programs_on_a_load_failure(self) -> None:
        runtime = PipelineDriver()
        discovery = runtime.discover_programs(_prepared("this is not @@@ valid AgL"))

        assert discovery.checked is None
        assert discovery.compiled is None
        assert discovery.programs == ()
        assert discovery.diagnostics


class TestPreflightArgumentsAndRun:
    def test_binds_positional_and_named_arguments_and_executes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = (
            "program def main(@arg-pos count: int, @arg-std label: text,\n"
            "  verbose: bool = false) -> unit =\n"
            "  print count\n"
            "  print label\n"
            "  print verbose\n"
        )
        runtime = PipelineDriver()
        prepared = _prepared(source)
        discovery = runtime.discover_programs(prepared)
        assert discovery.compiled is not None
        program = discovery.programs[0]

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(3,), named={"label": "hi", "verbose": True}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        assert preflight.executable is not None

        result = runtime.run_prepared(
            prepared,
            compiled=discovery.compiled,
            executable=preflight.executable,
            program_symbol=preflight.executable.program_symbols[program.node_id],
            arguments=preflight.arguments,
        )
        assert result.ok
        assert capsys.readouterr().out == "3\nhi\ntrue\n"

    def test_omitted_argument_resolves_to_its_own_default_expression(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = "program def main(verbose: bool = false) -> unit =\n  print verbose\n"
        runtime = PipelineDriver()
        prepared = _prepared(source)
        discovery = runtime.discover_programs(prepared)
        program = discovery.programs[0]

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        assert preflight.executable is not None

        result = runtime.run_prepared(
            prepared,
            compiled=discovery.compiled,
            executable=preflight.executable,
            program_symbol=preflight.executable.program_symbols[program.node_id],
            arguments=preflight.arguments,
        )
        assert result.ok
        assert capsys.readouterr().out == "false\n"


class TestPreflightArgumentsFailures:
    def test_missing_required_argument_reports_a_diagnostic_without_executing(self) -> None:
        source = "program def main(value: int) -> unit = print value\n"
        runtime = PipelineDriver()
        prepared = _prepared(source)
        discovery = runtime.discover_programs(prepared)
        program = discovery.programs[0]

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert not preflight.result.ok
        assert preflight.result.error is None
        messages = " | ".join(d.message for d in preflight.result.diagnostics)
        assert "value" in messages

    def test_decode_failure_reports_a_diagnostic_without_executing(self) -> None:
        source = "program def main(value: int) -> unit = print value\n"
        runtime = PipelineDriver()
        prepared = _prepared(source)
        discovery = runtime.discover_programs(prepared)
        program = discovery.programs[0]

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={"value": "not-an-int"}),
            compiled=discovery.compiled,
        )
        assert not preflight.result.ok
        assert preflight.result.error is None
        messages = " | ".join(d.message for d in preflight.result.diagnostics)
        assert "value" in messages

    def test_a_static_pipeline_failure_reports_diagnostics_without_an_executable(self) -> None:
        good_source = "program def main(value: int) -> unit = print value\n"
        runtime = PipelineDriver()
        program = runtime.discover_programs(_prepared(good_source)).programs[0]

        bad_prepared = _prepared('program def main(value: int) -> unit = value + "x"\n')
        preflight = runtime.preflight_arguments(
            bad_prepared, program, ProgramArguments(positional=(), named={"value": 1})
        )
        assert not preflight.result.ok
        assert preflight.executable is None


class TestRunFacadeDerivesArgumentsFromTheProgramSignature:
    """``PipelineDriver.run``/``run_prepared`` never binds a program's own
    value parameters (no ``preflight_arguments`` call): existing callers
    that pass no ``arguments`` must keep working exactly as they did before
    a program could declare its own parameters — a defaulted parameter
    resolves to its default, and a required one is a clean diagnostic, never
    an interpreter arity crash.
    """

    def test_a_defaulted_parameter_program_runs_with_no_arguments_supplied(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = PipelineDriver().run("program def main(value: int = 1) -> unit = print value\n")
        assert result.ok
        assert capsys.readouterr().out == "1\n"

    def test_a_required_parameter_program_reports_a_clean_diagnostic(self) -> None:
        result = PipelineDriver().run("program def main(value: int) -> unit = print value\n")
        assert not result.ok
        assert result.error is None
        messages = " | ".join(d.message for d in result.diagnostics)
        assert "value" in messages

    def test_check_only_required_parameter_program_reports_a_clean_diagnostic(self) -> None:
        result = PipelineDriver().run(
            "program def main(value: int) -> unit = print value\n", check_only=True
        )
        assert not result.ok
        assert result.error is None
        messages = " | ".join(d.message for d in result.diagnostics)
        assert "value" in messages


class TestRaisingDefault:
    def test_a_raising_default_yields_the_same_run_error_shape_as_a_body_raise(self) -> None:
        source = (
            "exception Boom extends Exception\n"
            "  code: int\n\n"
            'def blow-up() -> int = raise Boom(message = "boom!", code = 7)\n\n'
            "program def main(value: int = blow-up()) -> unit = print value\n"
        )
        runtime = PipelineDriver()
        prepared = _prepared(source)
        discovery = runtime.discover_programs(prepared)
        program = discovery.programs[0]

        preflight = runtime.preflight_arguments(
            prepared,
            program,
            ProgramArguments(positional=(), named={}),
            compiled=discovery.compiled,
        )
        assert preflight.result.ok
        assert preflight.executable is not None

        result = runtime.run_prepared(
            prepared,
            compiled=discovery.compiled,
            executable=preflight.executable,
            program_symbol=preflight.executable.program_symbols[program.node_id],
            arguments=preflight.arguments,
        )
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "Boom"
        assert result.error.fields["code"] == 7
