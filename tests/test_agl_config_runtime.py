"""Tests for AgL config declarations as runtime-resolved readable bindings.

``config KEY = expr`` now creates an executable, runtime-resolved readable
binding (like ``param``).  The resolution precedence per binding is:

    CLI --KEY  >  source value (if present)  >  config_base[KEY]

For ``Option[T]`` engine keys (``timeout``, ``log-file``) a bare source value of
type ``T`` is projected into ``some(value)``; an absent value binds ``config_base``.

NOTE: No static-analysis suppression comments in this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.ids import STD_CORE_ID
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, PreparedGraph, static_config_values
from agm.agl.runtime.codec import ParseResult
from agm.agl.runtime.contract import OutputContract
from agm.agl.runtime.params import convert_config_value
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import OPTION_TEXT_TYPE, TextType, Type
from agm.agl.semantics.values import BoolValue, EnumValue, IntValue, TextValue, Value
from agm.core.process import ProcessCaptureResult


def _run(
    source: str,
    *,
    config_cli: dict[str, Value] | None = None,
    config_base: dict[str, Value] | None = None,
) -> object:
    """Run *source* through prepare + run_prepared with optional config maps."""
    rt = PipelineDriver()
    prepared = rt.prepare(source)
    return rt.run_prepared(prepared, config_cli=config_cli, config_base=config_base)


def _prepare_graph(source: str) -> PreparedGraph:
    from pathlib import Path

    stdlib = Path(__file__).resolve().parent.parent / "stdlib"
    roots = RootSet(roots=frozenset({stdlib}))
    return PipelineDriver.prepare_program(source, entry_path=None, roots=roots)


# ---------------------------------------------------------------------------
# Runtime binding: source value readable
# ---------------------------------------------------------------------------


class TestSourceValueBinding:
    def test_config_source_value_readable(self) -> None:
        """``config max-iters = 5`` is readable as a binding at runtime."""
        result = _run("config max-iters = 5\nlet n = max-iters\nprint n")
        assert result.ok
        assert result.bindings["n"] == IntValue(5)

    def test_config_value_in_expression(self) -> None:
        """A config binding can be used inside an arithmetic expression."""
        result = _run("config max-iters = 5\nlet n = max-iters + 1\nprint n")
        assert result.ok
        assert result.bindings["n"] == IntValue(6)


# ---------------------------------------------------------------------------
# Bare config X binds from config_base
# ---------------------------------------------------------------------------


class TestBareConfigBase:
    def test_bare_config_binds_from_base(self) -> None:
        """Bare ``config max-iters`` binds the configured default from config_base."""
        result = _run(
            "config max-iters\nlet n = max-iters\nprint n",
            config_base={"max-iters": IntValue(7)},
        )
        assert result.ok
        assert result.bindings["n"] == IntValue(7)


# ---------------------------------------------------------------------------
# CLI override beats source value
# ---------------------------------------------------------------------------


class TestCliOverride:
    def test_cli_override_beats_source(self) -> None:
        """A config_cli entry overrides the source value."""
        result = _run(
            "config max-iters = 5\nlet n = max-iters\nprint n",
            config_cli={"max-iters": IntValue(10)},
            config_base={"max-iters": IntValue(99)},
        )
        assert result.ok
        assert result.bindings["n"] == IntValue(10)


# ---------------------------------------------------------------------------
# Option[text] projection
# ---------------------------------------------------------------------------


class TestOptionProjection:
    def test_source_text_projects_to_some(self) -> None:
        """``config timeout = "30s"`` binds ``some("30s")``."""
        result = _run('config timeout = "30s"\nlet t = timeout\nprint t')
        assert result.ok
        bound = result.bindings["t"]
        assert isinstance(bound, EnumValue)
        assert bound.variant == "Some"
        assert bound.fields["value"] == TextValue("30s")

    def test_bare_option_binds_none_from_base(self) -> None:
        """Bare ``config timeout`` with a ``none`` config_base binds ``none``."""
        none_val = convert_config_value("timeout", None, OPTION_TEXT_TYPE)
        result = _run(
            "config timeout\nlet t = timeout\nprint t",
            config_base={"timeout": none_val},
        )
        assert result.ok
        bound = result.bindings["t"]
        assert isinstance(bound, EnumValue)
        assert bound.variant == "None"

    def test_option_value_given_directly(self) -> None:
        """An Option[text] value given directly is bound without re-wrapping."""
        from pathlib import Path

        from agm.agl.modules.roots import RootSet

        stdlib = Path(__file__).resolve().parent.parent / "stdlib"
        roots = RootSet(roots=frozenset({stdlib}))
        rt = PipelineDriver()
        prep = rt.prepare_program(
            'import std.core\nconfig timeout = Some(value = "45s")\nlet t = timeout\nprint t\n',
            entry_path=None,
            roots=roots,
        )
        result = rt.run_prepared_graph(prep, config_base={})
        assert result.ok
        bound = result.bindings["t"]
        assert isinstance(bound, EnumValue)
        assert bound.variant == "Some"
        assert bound.fields["value"] == TextValue("45s")

    def test_bare_config_without_base_raises(self) -> None:
        """A bare config with no CLI override and no config_base entry is a host bug."""
        from agm.agl.ir.validate import InvalidIrError

        rt = PipelineDriver()
        prep = rt.prepare("config timeout\nprint 1")
        with pytest.raises(InvalidIrError):
            rt.run_prepared(prep, config_base={})


# ---------------------------------------------------------------------------
# convert_config_value / static_config_values helpers
# ---------------------------------------------------------------------------


class TestConfigHelpers:
    def test_convert_option_some(self) -> None:
        val = convert_config_value("timeout", "45s", OPTION_TEXT_TYPE)
        assert isinstance(val, EnumValue)
        assert val.nominal.module_id == STD_CORE_ID
        assert val.variant == "Some"
        assert val.fields["value"] == TextValue("45s")

    def test_convert_option_none(self) -> None:
        val = convert_config_value("timeout", None, OPTION_TEXT_TYPE)
        assert isinstance(val, EnumValue)
        assert val.variant == "None"

    def test_convert_non_option_falls_back(self) -> None:
        """A non-Option key type decodes via the plain param path."""
        from agm.agl.semantics.types import BoolType
        from agm.agl.semantics.values import BoolValue

        val = convert_config_value("strict-json", True, BoolType())
        assert val == BoolValue(True)

    def test_static_config_values_collects_literals(self) -> None:
        from agm.agl.parser import parse_program

        program = parse_program(
            "config max-iters = 5\nconfig strict-json = true\nconfig runner = \"claude\""
        )
        statics = static_config_values(program)
        assert statics == {"max-iters": 5, "strict-json": True, "runner": "claude"}

    def test_static_config_values_skips_non_literals(self) -> None:
        """Bare and non-literal (interpolated) config values are not static constants."""
        from agm.agl.parser import parse_program

        program = parse_program(
            'config max-iters = 5\nconfig runner = "a${x}"\nconfig timeout\n'
        )
        statics = static_config_values(program)
        assert statics == {"max-iters": 5}


class TestStartupConfigCollection:
    def _interpreter_for(self, source: str) -> object:
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.lower import lower_graph

        rt = PipelineDriver()
        prepared = _prepare_graph(source)
        discovery = rt.discover_params_graph(prepared)
        assert discovery.compiled_graph is not None
        executable = lower_graph(discovery.compiled_graph, validate=True)
        return IrInterpreter(executable)

    def test_unresolved_prepared_graph_returns_diagnostic(self) -> None:
        prepared = _prepare_graph("config log = ")
        result = PipelineDriver().collect_startup_config_graph(prepared, names={"log"})
        assert not result.ok
        assert result.diagnostics

    def test_missing_entry_module_returns_diagnostic(self) -> None:
        from unittest.mock import MagicMock

        prepared = MagicMock()
        prepared.warnings = ()
        prepared.resolved_graph.modules = {}

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"log"})

        assert not result.ok
        assert result.diagnostics[0].message == "Entry module not found in graph"

    def test_no_startup_config_returns_without_typecheck(self) -> None:
        prepared = _prepare_graph("let x = 1\n")
        result = PipelineDriver().collect_startup_config_graph(prepared, names={"log"})
        assert result.ok
        assert result.values == {}

    def test_startup_config_type_error_returns_diagnostic(self) -> None:
        prepared = _prepare_graph("config log = 1\n")
        result = PipelineDriver().collect_startup_config_graph(prepared, names={"log"})
        assert not result.ok
        assert result.diagnostics

    def test_startup_config_missing_required_param_returns_diagnostic(self) -> None:
        prepared = _prepare_graph("param enabled: bool\nconfig log = enabled\n")

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"log"})

        assert not result.ok
        assert result.diagnostics

    def test_startup_config_uses_provided_compiled_graph(self) -> None:
        rt = PipelineDriver()
        prepared = _prepare_graph("config log = true\n")
        discovery = rt.discover_params_graph(prepared)
        assert discovery.compiled_graph is not None

        result = rt.collect_startup_config_graph(
            prepared,
            names={"log"},
            compiled_graph=discovery.compiled_graph,
        )

        assert result.ok
        assert result.values["log"] == BoolValue(True)

    def test_startup_config_graph_reports_custom_codec_contract_error(self) -> None:
        class BadContractCodec:
            @property
            def name(self) -> str:
                return "bad-contract"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, TextType)

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                raise ValueError("boom")

            def parse(self, raw: str) -> ParseResult:
                return ParseResult.success(TextValue(raw))

        rt = PipelineDriver()
        rt.register_codec(BadContractCodec())
        prepared = _prepare_graph(
            'config runner = exec("printf startup", format = "bad-contract")\n'
        )

        result = rt.collect_startup_config_graph(prepared, names={"runner"})

        assert not result.ok
        assert result.diagnostics
        assert result.checked_graph is not None

    def test_startup_config_graph_reports_ir_contract_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.agl.diagnostics import Diagnostic

        def fail_ir_contracts(_executable: object, _codecs: object) -> object:
            return {}, [Diagnostic(message="contract failed", line=1)]

        monkeypatch.setattr("agm.agl.pipeline._materialize_ir_contracts", fail_ir_contracts)
        prepared = _prepare_graph('config runner = "ok"\n')

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})

        assert not result.ok
        assert result.diagnostics
        assert result.checked_graph is not None

    def test_startup_config_graph_materializes_custom_codec_contract(self) -> None:
        class SchemaTextCodec:
            @property
            def name(self) -> str:
                return "schema-text"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, TextType)

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=None,
                    format_instructions="SCHEMA-TEXT",
                    json_schema={"expected": "schema"},
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> ParseResult:
                if schema != {"expected": "schema"}:
                    return ParseResult.failure("missing custom schema")
                return ParseResult.success(TextValue(raw))

        rt = PipelineDriver()
        rt.register_codec(SchemaTextCodec())
        prepared = _prepare_graph(
            'config runner = exec("printf startup", format = "schema-text")\n'
        )

        result = rt.collect_startup_config_graph(prepared, names={"runner"})

        assert result.ok, result.error
        assert result.values["runner"] == TextValue("startup")

    def test_startup_config_runtime_error_returns_error(self) -> None:
        prepared = _prepare_graph('config runner = raise Abort(message = "boom")\n')
        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "Abort"

    def test_startup_config_value_from_extern_wires_companion(self, tmp_path: Path) -> None:
        from agm.agl.modules.roots import RootSet

        entry_path = tmp_path / "prog.agl"
        entry_path.write_text(
            "extern def pick_runner() -> text\nconfig runner = pick_runner()\n"
        )
        (tmp_path / "prog.py").write_text('def pick_runner():\n    return "claude"\n')
        prepared = PipelineDriver.prepare_program(
            entry_path.read_text(),
            entry_path=entry_path,
            roots=RootSet(roots=frozenset({tmp_path})),
            default_stdlib=False,
        )

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})

        assert result.ok, result.error or result.diagnostics
        assert result.values["runner"] == TextValue("claude")

    def test_startup_config_can_call_imported_function(self, tmp_path: Path) -> None:
        entry_path = tmp_path / "prog.agl"
        entry_path.write_text("import lib\nconfig runner = lib::choose()\n")
        (tmp_path / "lib.agl").write_text('def choose() -> text = "claude"\n')
        prepared = PipelineDriver.prepare_program(
            entry_path.read_text(),
            entry_path=entry_path,
            roots=RootSet(roots=frozenset({tmp_path})),
            default_stdlib=False,
        )

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})

        assert result.ok, result.error or result.diagnostics
        assert result.values["runner"] == TextValue("claude")

    def test_startup_config_function_can_read_earlier_binding(self) -> None:
        prepared = _prepare_graph(
            'let selected = "claude"\n'
            "def choose() -> text = selected\n"
            "config runner = choose()\n"
        )

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})

        assert result.ok, result.error or result.diagnostics
        assert result.values["runner"] == TextValue("claude")

    def test_startup_config_lambda_can_capture_earlier_binding(self) -> None:
        prepared = _prepare_graph(
            "def make(prefix: text) -> () -> text = "
            'fn() -> text => prefix + "aude"\n'
            'let choose = make("cl")\n'
            "config runner = choose()\n"
        )

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})

        assert result.ok, result.error or result.diagnostics
        assert result.values["runner"] == TextValue("claude")

    def test_startup_config_function_default_can_read_earlier_binding(self) -> None:
        prepared = _prepare_graph(
            'let suffix = "aude"\n'
            "def choose(prefix: text, ending: text = suffix) -> text = prefix + ending\n"
            'config runner = choose("cl")\n'
        )

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})

        assert result.ok, result.error or result.diagnostics
        assert result.values["runner"] == TextValue("claude")

    def test_startup_config_calls_through_function_value_alias(self) -> None:
        prepared = _prepare_graph(
            'def choose() -> text = "claude"\n'
            "let invoke: () -> text = choose\n"
            "config runner = invoke()\n"
        )

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})

        assert result.ok, result.error or result.diagnostics
        assert result.values["runner"] == TextValue("claude")

    def test_startup_config_preserves_earlier_strict_json_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fenced_json(
            _cmd: list[str],
            **_kwargs: object,
        ) -> ProcessCaptureResult:
            return ProcessCaptureResult(
                returncode=0,
                stdout="```json\ntrue\n```\n",
                stderr="",
                elapsed=0.01,
                timed_out=False,
                spawn_error=None,
                spawn_errno=None,
            )

        monkeypatch.setattr("agm.core.process.run_capture_result", fenced_json)
        prepared = _prepare_graph(
            "config strict-json = true\n"
            'config log = exec("choose", format = "json")\n'
        )

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"log"})

        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "AgentParseError"

    def test_startup_config_extern_wiring_failure_returns_diagnostic(
        self, tmp_path: Path
    ) -> None:
        from agm.agl.modules.roots import RootSet

        entry_path = tmp_path / "prog.agl"
        entry_path.write_text(
            "extern def pick_runner() -> text\nconfig runner = pick_runner()\n"
        )
        # Companion is present but does not define the declared extern, so
        # wiring fails during the startup-config prepass.
        (tmp_path / "prog.py").write_text("def other():\n    return 1\n")
        prepared = PipelineDriver.prepare_program(
            entry_path.read_text(),
            entry_path=entry_path,
            roots=RootSet(roots=frozenset({tmp_path})),
            default_stdlib=False,
        )

        result = PipelineDriver().collect_startup_config_graph(prepared, names={"runner"})

        assert not result.ok
        assert result.diagnostics
        assert result.checked_graph is not None

    def test_interpreter_collect_returns_empty_when_no_targets(self) -> None:
        interp = self._interpreter_for("let x = 1\nx\n")
        assert interp.collect_entry_config_values({"log"}) == {}

    def test_interpreter_initializer_recording_is_idempotent(self) -> None:
        interp = self._interpreter_for("config log = true\n")
        entry_module = interp._program.modules[interp._program.entry_module]
        node = entry_module.initializers[0]

        interp._eval_and_record_initializer(entry_module.module_id, 0, node)
        interp._eval_and_record_initializer(entry_module.module_id, 0, node)

        assert len(interp.initializer_values) == 1

    def test_interpreter_resume_does_not_reinstall_param_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.agl.ir.nodes import IrExpr

        interp = self._interpreter_for("param value: int = 1\nconfig log = true\nvalue\n")
        default = interp._program.params[0].default
        assert default is not None
        original_eval = interp._eval
        calls = 0

        def count_default(node: IrExpr) -> Value:
            nonlocal calls
            if node is default:
                calls += 1
            return original_eval(node)

        monkeypatch.setattr(interp, "_eval", count_default)
        interp.collect_entry_config_values({"log"})
        interp.run()

        assert calls == 1

    def test_interpreter_collect_skips_non_entry_symbols(self) -> None:
        from agm.agl.ir.ids import SymbolId
        from agm.agl.ir.program import SymbolDescriptor
        from agm.agl.modules.ids import ModuleId

        interp = self._interpreter_for("import std.core\nconfig log = true\n")
        interp._program.symbols[SymbolId(999_001)] = SymbolDescriptor(
            symbol_id=SymbolId(999_001),
            mutable=False,
            public_name="log",
            owner=ModuleId(("not_entry",)),
        )
        assert interp.collect_entry_config_values({"log"}) == {"log": BoolValue(True)}

    def test_interpreter_collect_returns_found_when_modules_disappear(self) -> None:
        interp = self._interpreter_for("config log = true\n")
        entry_module = interp._program.modules[interp._program.entry_module]

        class EmptyValuesModules(dict):
            def values(self) -> object:
                return ()

        object.__setattr__(
            interp._program,
            "modules",
            EmptyValuesModules({interp._program.entry_module: entry_module}),
        )

        assert interp.collect_entry_config_values({"log"}) == {}

    def test_interpreter_collect_handles_non_entry_modules_before_entry(self) -> None:
        from agm.agl.ir.ids import Location, SourceId
        from agm.agl.ir.nodes import IrConstUnit
        from agm.agl.ir.program import ExecutableModule
        from agm.agl.modules.ids import ModuleId

        interp = self._interpreter_for("config log = true\n")
        entry_module = interp._program.modules[interp._program.entry_module]
        empty_id = ModuleId(("empty",))
        init_id = ModuleId(("init",))
        location = Location(
            source_id=SourceId(0),
            start_offset=0,
            end_offset=0,
            start_line=1,
            start_col=0,
        )
        modules = {
            empty_id: ExecutableModule(module_id=empty_id, initializers=()),
            init_id: ExecutableModule(
                module_id=init_id,
                initializers=(IrConstUnit(location=location),),
            ),
            interp._program.entry_module: entry_module,
        }
        object.__setattr__(interp._program, "modules", modules)

        assert interp.collect_entry_config_values({"log"}) == {"log": BoolValue(True)}

    def test_interpreter_collect_sets_span_on_raised_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.agl.semantics.exceptions import AglRaise, make_builtin_exception

        interp = self._interpreter_for("config log = true\n")

        def fail(_node: object) -> Value:
            raise AglRaise(make_builtin_exception("Abort", "boom"))

        monkeypatch.setattr(interp, "_eval_initializer", fail)

        with pytest.raises(AglRaise) as exc_info:
            interp.collect_entry_config_values({"log"})
        assert exc_info.value.span is not None

    def test_resume_startup_config_without_buffered_trace_starts_trace(self) -> None:
        runtime = PipelineDriver()
        startup = runtime.collect_startup_config_graph(
            _prepare_graph("config log = true\n"),
            names={"log"},
        )
        assert startup.ok
        startup.trace = None

        result = runtime.resume_startup_config(startup, log_file=None)

        assert result.ok


def test_generic_constructor_alias_value_lowers_to_target_nominal() -> None:
    result = _run(
        "record Box[T]\n"
        "  value: T\n"
        "type Alias[T] = Box[T]\n"
        "let factory: (int) -> Box[int] = Alias\n"
        "factory(1)\n"
    )
    assert result.ok


# ---------------------------------------------------------------------------
# Discovery returns ConfigDeclInfo
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discovery_returns_config_decl_info(self) -> None:
        from agm.agl.runtime.types import ConfigDeclInfo

        rt = PipelineDriver()
        prepared = rt.prepare("config max-iters = 5\nconfig timeout\nprint 1")
        discovery = rt.discover_params(prepared)
        assert not discovery.diagnostics
        names = {c.name for c in discovery.configs}
        assert names == {"max-iters", "timeout"}
        by_name = {c.name: c for c in discovery.configs}
        assert isinstance(by_name["max-iters"], ConfigDeclInfo)
        assert by_name["max-iters"].has_value is True
        assert by_name["timeout"].has_value is False


# ---------------------------------------------------------------------------
# Engine behavior unchanged
# ---------------------------------------------------------------------------


class TestEngineBehaviorUnchanged:
    def test_config_strict_json_program_runs(self) -> None:
        """A program declaring config strict-json + max-iters runs cleanly."""
        result = _run(
            "config strict-json = true\nconfig max-iters = 3\nprint 1"
        )
        assert result.ok
        assert not result.diagnostics

    def test_config_only_program_runs(self) -> None:
        result = _run("config log = false\nconfig strict-json = true")
        assert result.ok
