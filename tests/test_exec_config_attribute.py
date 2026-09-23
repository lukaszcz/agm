"""Host merge of the ``@config`` program attribute.

``PipelineDriver.preflight_arguments`` evaluates a selected program's own
``@config`` entries into ``ArgumentPreflight.program_config`` (see
``agm.agl.pipeline``); ``agm exec`` (``agm.commands.exec_program.run``) is the
host that folds those values into its existing precedence chains:

* module parameter: CLI flag > ``@opt-env`` > program route > ``@config`` >
  module route > declared initializer.
* engine setting: source write > CLI flag > program table > ``@config`` >
  ``[exec]`` > declared default.

Agents are always mocked (or avoided entirely via ``exec`` shell calls) — no
real agent runs in these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import agm.commands.exec as exec_command
from agm.cli_support.args import CheckArgs, ExecArgs
from agm.commands import check as check_command
from agm.commands import exec_program as exec_engine
from agm.config.context import ConfigContext
from agm.packages.layout import MODULE_TREE_DIRNAME
from tests._agl_helpers import write_file_program
from tests._package_helpers import install_directory
from tests.test_cli_registered_commands import invoke
from tests.test_exec_command import _config_home, _exec_args_no_trace, _spy_runtime


class TestConfigParamPrecedence:
    """Each precedence pair immediately adjacent to ``@config`` for a module parameter."""

    def test_cli_beats_config(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = "
            "print(count)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file, argument_tokens=["--count", "5"]))
        assert capsys.readouterr().out == "5\n"

    def test_opt_env_beats_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            '@param @opt-env("COUNT_ENV") let count: int = 1\n\n'
            "@config(count = 2)\n"
            "program def main() -> unit = print(count)\n",
        )
        monkeypatch.setenv("COUNT_ENV", "7")
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == "7\n"

    def test_program_route_beats_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = "
            "print(count)\n",
        )
        _config_home(tmp_path, monkeypatch, "[prog.main]\ncount = 9\n")
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == "9\n"

    def test_config_beats_module_route(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "helper.agl").write_text("@param let count: int = 1\n")
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import helper\n\n@config(helper::count = 2)\n"
            "program def main() -> unit = print(helper::count)\n",
        )
        _config_home(tmp_path, monkeypatch, "[helper]\ncount = 9\n")
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == "2\n"

    def test_config_beats_declared_initializer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "@param let count: int = 1\n\n@config(count = 42)\nprogram def main() -> unit = "
            "print(count)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == "42\n"


class TestConfigParamTargetShapes:
    """``@config`` reaches a ``@param`` binding regardless of where it is declared."""

    def test_own_module_target(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "@param let count: int = 1\n\n@config(count = 2)\nprogram def main() -> unit = "
            "print(count)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == "2\n"

    def test_imported_module_target(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "helper.agl").write_text("@param let count: int = 1\n")
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import helper\n\n@config(helper::count = 2)\n"
            "program def main() -> unit = print(helper::count)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == "2\n"

    def test_scoped_target(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "scope Logging\n"
            "  @param let level: int = 1\n"
            "end Logging\n\n"
            "@config(Logging::level = 2)\n"
            "program def main() -> unit = print(Logging::level)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == "2\n"


class TestConfigEnginePrecedence:
    """Each precedence pair immediately adjacent to ``@config`` for an engine setting.

    Precedence is proven end to end through the program's own ``std/config``
    read rather than an internal call capture: ``print(config::timeout)``
    renders the winning tier's raw ``Option[text]`` spelling.
    """

    def test_cli_beats_config(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\n\n@config(config::timeout = Some("2s"))\n'
            "program def main() -> unit = print(config::timeout)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file, timeout="9s"))
        assert capsys.readouterr().out == 'Option::Some(value = "9s")\n'

    def test_program_table_beats_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\n\n@config(config::timeout = Some("2s"))\n'
            "program def main() -> unit = print(config::timeout)\n",
        )
        _config_home(tmp_path, monkeypatch, '[prog.main]\ntimeout = "9s"\n')
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == 'Option::Some(value = "9s")\n'

    def test_config_beats_exec_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\n\n@config(config::timeout = Some("2s"))\n'
            "program def main() -> unit = print(config::timeout)\n",
        )
        _config_home(tmp_path, monkeypatch, '[exec]\ntimeout = "9s"\n')
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == 'Option::Some(value = "2s")\n'

    def test_config_beats_declared_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\n\n@config(config::timeout = Some("2s"))\n'
            "program def main() -> unit = print(config::timeout)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == 'Option::Some(value = "2s")\n'

    def test_source_write_still_wins_over_config(self, tmp_path: Path) -> None:
        """A ``std/config::strict-json := VALUE`` write overrides even a
        ``@config``-seeded initial value, exactly as it overrides every other tier."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n\n@config(config::strict-json = true)\n"
            "program def main() -> unit =\n"
            "  std/config::strict-json := false\n"
            "  let r: int = exec \"printf '```json\\n5\\n```'\"\n"
            "  print r\n",
        )
        result = exec_command.run(_exec_args_no_trace(agl_file))
        assert result is None  # lenient recovery succeeds; source write overrode @config


class TestConfigTimeoutReachesEveryTimeoutConsumer:
    def test_config_timeout_reaches_agent_idle_timeout_and_shell_exec_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        real_factory = exec_engine.value_driven_agent_factory
        real_session_host = exec_engine.create_agl_session_host

        def factory_spy(*, idle_timeout: float | None, context: ConfigContext) -> object:
            captured["agent_idle_timeout"] = idle_timeout
            return real_factory(idle_timeout=idle_timeout, context=context)

        def session_host_spy(*, idle_timeout: float | None) -> object:
            captured["session_idle_timeout"] = idle_timeout
            return real_session_host(idle_timeout=idle_timeout)

        monkeypatch.setattr(exec_engine, "value_driven_agent_factory", factory_spy)
        monkeypatch.setattr(exec_engine, "create_agl_session_host", session_host_spy)
        shell_captured = _spy_runtime(monkeypatch)

        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\n\n@config(config::timeout = Some("3s"))\n'
            "program def main() -> unit = ()\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))

        assert captured["agent_idle_timeout"] == pytest.approx(3.0)
        assert captured["session_idle_timeout"] == pytest.approx(3.0)
        assert shell_captured["shell_exec_timeout"] == pytest.approx(3.0)


class TestConfigLogFileDerivesLog:
    def test_config_trace_file_enables_trace_via_derived_rule(self, tmp_path: Path) -> None:
        """A ``@config`` ``trace-file`` alone (no explicit ``trace = true``) enables
        tracing through the same derived rule ``build_host_engine_seeds`` applies
        to the config-file tiers."""
        trace_path = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            f'import std/config\n\n@config(config::trace-file = Some("{trace_path}"))\n'
            'program def main() -> unit = print "hi"\n',
        )
        exec_command.run(
            ExecArgs(
                file=str(agl_file),
                argument_tokens=[],
                strict_json=None,
                no_trace=False,
                trace_file=None,
            )
        )
        assert trace_path.exists()

    def test_config_trace_file_seeds_the_trace_register_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The program-visible ``std/config::trace`` register (not just the file
        the host actually writes) reflects a ``@config``-only ``trace-file``:
        both must agree with the derived rule, or the register lies about the
        real tracing state."""
        trace_path = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            f'import std/config\n\n@config(config::trace-file = Some("{trace_path}"))\n'
            "program def main() -> unit = print(config::trace)\n",
        )
        exec_command.run(
            ExecArgs(
                file=str(agl_file),
                argument_tokens=[],
                strict_json=None,
                no_trace=False,
                trace_file=None,
            )
        )
        assert capsys.readouterr().out == "true\n"
        assert trace_path.exists()


class TestConfigLogFlagInteractions:
    """Documented contract: ``--no-trace-file`` clears only the CLI seed and never
    hides a config-table-established trace; ``--no-trace`` always disables."""

    def test_no_trace_file_does_not_suppress_a_config_trace_file(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            f'import std/config\n\n@config(config::trace-file = Some("{trace_path}"))\n'
            'program def main() -> unit = print "hi"\n',
        )
        exec_command.run(
            ExecArgs(
                file=str(agl_file),
                argument_tokens=[],
                strict_json=None,
                no_trace=False,
                trace_file=None,
                no_trace_file=True,
            )
        )
        assert trace_path.exists()

    def test_no_trace_suppresses_a_config_trace_file(self, tmp_path: Path) -> None:
        trace_path = tmp_path / "trace.jsonl"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            f'import std/config\n\n@config(config::trace-file = Some("{trace_path}"))\n'
            'program def main() -> unit = print "hi"\n',
        )
        exec_command.run(
            ExecArgs(
                file=str(agl_file),
                argument_tokens=[],
                strict_json=None,
                no_trace=True,
                trace_file=None,
            )
        )
        assert not trace_path.exists()


class TestConfigTimeoutValidation:
    def test_malformed_config_timeout_exits_1_on_the_exec_side(self, tmp_path: Path) -> None:
        """The ``agm check`` side of this is covered separately (a program
        that is never selected/preflighted never resolves the value); once a
        program IS selected and run, a malformed ``@config`` timeout is a
        pre-execution failure like any other invalid engine value."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\n\n@config(config::timeout = Some("not-a-duration"))\n'
            "program def main() -> unit = ()\n",
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_trace(agl_file))
        assert exc_info.value.code == 1


class TestConfigDryRun:
    def test_dry_run_applies_config_without_executing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A valid ``@config`` engine value never breaks ``--dry-run``, and the
        static-only pass still never executes the program body."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\n\n@config(config::timeout = Some("5s"))\n'
            'program def main() -> unit = print "ran"\n',
        )
        assert exec_command.run(_exec_args_no_trace(agl_file)) is None
        assert capsys.readouterr().out == ""


class TestConfigDefaultAgent:
    def test_config_default_agent_is_seeded(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``default-agent`` is one of the enum-backed engine keys (``AGENT``
        kind, alongside ``default-sandbox``'s ``AGENT_SANDBOX``): a ``@config``
        value for it goes through the same executable -> standard identity
        restamp as ``timeout``/``trace-file`` (``OPTION_TEXT``)."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n\n"
            '@config(config::default-agent = AgentCommand("config-agent"))\n'
            "program def main() -> unit = print(config::default-agent)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))
        assert "config-agent" in capsys.readouterr().out


class TestConfigDefaultSandbox:
    def test_config_default_sandbox_is_seeded(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``default-sandbox`` is the ``AGENT_SANDBOX`` engine key: a ``@config`` value
        for it goes through the same executable -> standard identity restamp as
        ``default-agent``, but for the ``AgentSandbox`` reserved enum."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n\n"
            "@config(config::default-sandbox = AgentSandbox::Native)\n"
            "program def main() -> unit = "
            "print(config::default-sandbox is AgentSandbox::Native)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == "true\n"


class TestConfigStrictJson:
    def test_config_strict_json_enables_strict_parsing(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n\n@config(config::strict-json = true)\n"
            "program def main() -> unit =\n"
            "  let r: int = exec \"printf '```json\\n5\\n```'\"\n"
            "  print r\n",
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_trace(agl_file))
        assert exc_info.value.code == 2  # strict JSON rejects the fenced payload


class TestConfigCombinesParamAndEngineTargets:
    """A single ``@config`` may target a ``@param`` binding and an engine
    setting together; both partitions merge independently in the same run."""

    def test_module_param_and_engine_setting_both_apply(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n\n"
            "@param let count: int = 1\n\n"
            '@config(count = 7, config::timeout = Some("4s"))\n'
            "program def main() -> unit =\n"
            "  print count\n"
            "  print(config::timeout)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file))
        assert capsys.readouterr().out == '7\nOption::Some(value = "4s")\n'


class TestProgramDefCalledAsAnOrdinaryFunctionIgnoresConfig:
    def test_config_is_ignored_when_called_as_a_function(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "@param let count: int = 1\n\n"
            "@config(count = 99)\n"
            "program def build() -> unit = print(count)\n\n"
            "def call_build() -> unit = build()\n\n"
            "program def main() -> unit = call_build()\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file, program="main"))
        # build's own @config entry never applies: it is not the selected program.
        assert capsys.readouterr().out == "1\n"

    def test_selecting_the_sibling_program_applies_its_own_config(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The flip side: selecting ``build`` directly applies its own
        ``@config`` (distinct from ``main``'s), proving the two declarations'
        entries stay independent rather than one leaking into the other."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "@param let count: int = 1\n\n"
            "@config(count = 99)\n"
            "program def build() -> unit = print(count)\n\n"
            "@config(count = 2)\n"
            "program def main() -> unit = print(count)\n",
        )
        exec_command.run(_exec_args_no_trace(agl_file, program="build"))
        assert capsys.readouterr().out == "99\n"


class TestRegisteredCommandHonoursConfig:
    def test_registered_command_applies_its_own_config_attribute(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A registered command reaches ``run`` through the same path ``agm exec``
        uses, so its own ``@config`` entries merge in identically."""
        source = tmp_path / "source"
        (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
        (source / "package.toml").write_text(
            '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
        )
        (source / MODULE_TREE_DIRNAME / "review.agl").write_text(
            "@param let count: int = 1\n\n"
            '@command("tools review")\n'
            "@config(count = 42)\n"
            "program def main() -> unit = print(count)\n",
            encoding="utf-8",
        )
        home = tmp_path / "home"

        install_directory(source, home=home, env={})
        monkeypatch.setenv("HOME", str(home))

        result = invoke(CliRunner(), ["tools", "review"])

        assert result.exit_code == 0, result.output
        assert result.stdout == "42\n"


class TestCheckAndReplApplyNoConfig:
    def test_check_does_not_apply_config_engine_values(self, tmp_path: Path) -> None:
        """``agm check`` never selects or preflights a program, so a ``@config``
        engine value that would fail host-side resolution (an unparsable
        timeout) never reaches that resolution at all."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\n\n@config(config::timeout = Some("not-a-duration"))\n'
            "program def main() -> unit = ()\n",
        )
        assert check_command.run(CheckArgs(files=[str(agl_file)])) is None

    def test_repl_does_not_apply_config_engine_values(self) -> None:
        """The REPL has no program-selection/preflight path, so a declared
        program's ``@config`` never reaches the REPL's own engine registers."""
        from tests.test_agl_repl_builtin_settings import _ok, _read_result, _session, _variant

        session = _session()
        _ok(session, "import std/config")
        _ok(
            session,
            '@config(config::timeout = Some("45s"))\nprogram def unrelated() -> unit = ()',
        )
        result = _read_result(session, "timeout")
        assert _variant(result.value, result.descriptors) == "None"
